"""
Python translation of the I2C2 R package (https://github.com/muschellij2/I2C2).
Only includes selected functions: I2C2(), check_id_visit(), and 
demean_matrix() functions.

Reference
---------
Shou, H, Eloyan, A, Lee, S, Zipunnikov, V, Crainiceanu, AN, Nebel, MB, Caffo, B,
Lindquist, MA, Crainiceanu, CM (2013). Quantifying the reliability of image
replication studies: the image intraclass correlation coefficient (I2C2).
Cogn Affect Behav Neurosci, 13, 4:714-24.
"""

from __future__ import annotations
from typing import Any, Sequence
import numpy as np

__all__ = ["check_id_visit", "demean_matrix", "I2C2", "i2c2"]


def _group_sum(mat: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sum rows of `mat` within each group, returned ascending by group value.

    Equivalent to R's rowsum(mat, group, na.rm = FALSE): NaNs propagate
    (a group/column sum is NaN if any contributing row is NaN there).
    """
    order = np.argsort(group, kind="stable")
    group_sorted = group[order]
    mat_sorted = mat[order]
    uniq_groups, start_idx = np.unique(group_sorted, return_index=True)
    sums = np.add.reduceat(mat_sorted, start_idx, axis=0)
    return sums, uniq_groups


def check_id_visit(y: np.ndarray, id: Sequence, visit: Sequence) -> dict:
    """
    Validate and reorder I2C2 inputs.

    Parameters
    ----------
    y : (n, p) array-like
        n vectorized images (rows) by p voxels (columns).
    id : sequence of length n
        Subject id for each row of y.
    visit : sequence of length n
        Visit label for each row of y.

    Returns
    -------
    dict with keys "y", "id", "visit", "n", "p", "I".
        `y`, `id`, `visit` are reordered by (id, visit).
        `id`/`visit` are recoded to 0-based integer codes (R uses 1-based).
        `I` is the number of unique ids.
    """
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError("y is not a matrix!")

    n, p = y.shape
    if n == 1:
        raise ValueError("only one observation!")

    id = np.asarray(id)
    visit = np.asarray(visit)

    if id.shape[0] != n:
        raise ValueError("Number of ids not equal to number of rows of y")
    if visit.shape[0] != n:
        raise ValueError("Number of visits not equal to number of rows of y")

    if not (np.issubdtype(y.dtype, np.number) or np.issubdtype(y.dtype, np.bool_)):
        raise ValueError("y is not a numeric/integer/logical type!")

    I = np.unique(id).shape[0]

    y = y.astype(float, copy=True)

    # Recode id/visit to 0-based integer codes ordered by sorted unique value
    # (equivalent to R's as.numeric(factor(id)), but 0-based instead of 1-based).
    _, id_codes = np.unique(id, return_inverse=True)
    _, visit_codes = np.unique(visit, return_inverse=True)
    id_codes = np.asarray(id_codes).reshape(-1)
    visit_codes = np.asarray(visit_codes).reshape(-1)

    # Order rows by (id, visit); stable, matching R's order().
    ord_idx = np.lexsort((visit_codes, id_codes))

    if not np.array_equal(ord_idx, np.arange(n)):
        visit_codes = visit_codes[ord_idx]
        id_codes = id_codes[ord_idx]
        y = y[ord_idx, :]

    return {
        "y": y,
        "id": id_codes,
        "visit": visit_codes,
        "n": n,
        "p": p,
        "I": I,
    }


def demean_matrix(
    y: np.ndarray,
    visit: np.ndarray,
    twoway: bool = True,
    tol: float = 0,
) -> np.ndarray:
    """
    De-mean a matrix using the overall column means, and optionally the
    visit-specific means as well (twoway).

    Centering mirrors R's scale(x, center = TRUE, scale = FALSE), whose
    internal center is colMeans(x, na.rm = TRUE) -- so NaNs are ignored when
    computing the means being subtracted (np.nanmean below).
    """
    y = np.asarray(y, dtype=float)
    visit = np.asarray(visit)

    # Overall centering (colMeans(x, na.rm = TRUE) inside R's scale()).
    y = y - np.nanmean(y, axis=0, keepdims=True)

    if not twoway:
        return y

    resd = y.copy()
    uvisit = np.sort(np.unique(visit))
    for j in uvisit:
        if tol == 0:
            ind = visit == j
        else:
            ind = (visit - j) <= tol

        mat = resd[ind, :]
        mat = mat - np.nanmean(mat, axis=0, keepdims=True)
        resd[ind, :] = mat

    return resd


def I2C2(
    y: np.ndarray,
    id: Sequence,
    visit: Sequence,
    symmetric: bool = False,
    truncate: bool = False,
    twoway: bool = True,
    demean: bool = True,
    return_demean: bool = True,
    **kwargs: Any,
) -> dict:
    """
    Image Intraclass Correlation Coefficient (I2C2) via the trace method.

    Parameters
    ----------
    y : (n, p) array-like
        n vectorized image observations (rows) by p voxels (columns). Row
        order does not matter -- it is fixed internally to match (id, visit).
    id : sequence of length n
        Subject id for each row of y.
    visit : sequence of length n
        Visit label for each row of y.
    symmetric : bool
        If False, use the method-of-moments estimator formula; if True, use
        the pairwise symmetric sum formula. Default False.
    truncate : bool
        If True, negative I2C2 estimates are truncated to zero.
    twoway : bool
        If True, remove visit-specific means in addition to the overall mean
        (guards against scanner/batch effects). If False, only the overall
        mean is removed.
    demean : bool
        If True, de-mean the data before computing variance components.
    return_demean : bool
        If True (and demean is True), include the de-meaned matrix in the
        returned dict under "demean_y".

    Returns
    -------
    dict with keys:
        "lambda"   : estimated I2C2 (float)
        "Kx"       : trace of the between-cluster variance operator
        "Ku"       : trace of the within-cluster variance operator
        "demean_y" : de-meaned data matrix (only if demean and return_demean)

    Notes
    -----
    NaNs in `y` are not simply skipped voxel-by-voxel: `lambda`/`Kx`/`Ku` are
    each a single scalar summed over the *entire* matrix, so even one NaN
    anywhere in `y` will make the final `lambda` NaN. If your data can contain
    missing/invalid voxels (e.g. medial-wall parcels), drop those columns
    from `y` before calling I2C2 rather than relying on per-voxel NaN
    handling. This matches the behavior of the original R implementation.
    """
    L = check_id_visit(y=y, id=id, visit=visit)
    n = L["n"]
    y = L["y"]
    id = L["id"]
    I = L["I"]
    visit = L["visit"]
    del L

    # Visits per id cluster (0-based id codes 0..I-1, ascending).
    _, n_I0 = np.unique(id, return_counts=True)
    n_I0 = n_I0.astype(float)
    k2 = np.sum(n_I0 ** 2)  # sum_i J_i^2

    resd = None

    if demean:
        resd = demean_matrix(y=y, visit=visit, twoway=twoway, tol=0)
        W = resd
    else:
        W = y

    # Population average for the (de-meaned) dataset W.
    # NOTE: matches R's colMeans(W) -- default na.rm = FALSE, so NaNs propagate.
    Wdd = np.mean(W, axis=0)

    # Subject-specific counts/sums for the (de-meaned) dataset W.
    not_na = (~np.isnan(W)).astype(float)
    Ni, _ = _group_sum(not_na, id)   # I x p: # non-NaN obs per id, per column
    Si, _ = _group_sum(W, id)        # I x p: sum of W per id, per column
    Si = Si / Ni                     # I x p: per-id column means

    Wi = Si[id, :]                   # n x p: each row's id-specific mean

    if not symmetric:
        trKu = np.sum((W - Wi) ** 2) / (n - I)
        trKw = np.sum((W - Wdd) ** 2) / (n - 1)
        trKx = trKw - trKu
    else:
        Si_sum = Si * Ni  # back to per-id column sums
        # NOTE: n_I0[id] must be reshaped to a column vector so it scales
        # W ** 2 row-wise (each row scaled by its own cluster size), matching
        # R's `W^2 * n_I0[id]` matrix-by-vector recycling.
        trKu = (np.sum(W ** 2 * n_I0[id][:, None]) - np.sum(Si_sum ** 2)) / (k2 - n)
        trKw = (
            np.sum(W ** 2) * n
            - np.sum((n * Wdd) ** 2)
            - trKu * (k2 - n)
        ) / (n ** 2 - k2)
        trKx = trKw - trKu

    lam = trKx / (trKx + trKu)
    if truncate and lam <= 0:
        lam = 0.0

    result: dict[str, Any] = {"lambda": lam, "Kx": trKx, "Ku": trKu}
    if demean and return_demean:
        result["demean_y"] = resd

    return result


def i2c2(*args: Any, **kwargs: Any) -> dict:
    """Alias for I2C2()."""
    return I2C2(*args, **kwargs)
