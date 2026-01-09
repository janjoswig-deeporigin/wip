import numpy as np
from sklearn.metrics.pairwise import (
    pairwise_distances,
    check_paired_arrays,
    check_pairwise_arrays,
)


def rmsd_d(
    a: np.ndarray, b: np.ndarray, d: int = 3, out: np.ndarray = None
) -> np.ndarray:
    """Calculate RMSD(s) between sets of data points
    with known dimensionality.

    Consider using :func:`rmsd_n` if a calculation based on the
    known number of points is preferred.

    The RMSD is calculated here between two sets of points
    `a` and `b` as `sqrt(d * mean((a - b)^2))`,
    where `d` is the dimensionality of the points.
    To comply with the standard definition of metrics,
    `a` and `b` are generally expected to be 1D arrays
    of shape `(n * d,)`. In other words, each point set is treated as
    a point itself in a higher-dimensional space.
    Sets can be batched by adding a leading axis.

    The input arrays need to have compatible shapes within
    broadcasting rules.
    The following input shapes in `a` and `b` are supported:

      * `(n * d,)` vs `(n * d,)`: Single pair of point sets;
            returns a single RMSD value.
      * `(m, n * d)` vs `(n * d,)`: `m` point sets against
            a single reference set;
            returns `m` RMSD values.
      * `(m, n * d)` vs `(m, n * d,)`: Two batches of `m` point sets against
            each other; assumes element-wise pairing of batches and
            returns `m` RMSD values.

    Args:
        a: Input point sets, generally a 1D array of shape `(n * d,)`.
        b: Other input point sets, generally a 1D array of shape `(n * d,)`.
        d: Dimensionality `d` of points in `a` and `b`.
        out: Optional output array to store results.

    Returns:
        The RMSD between point sets in `a` and `b`.
    """

    diff = a - b
    np.square(diff, out=diff)
    return np.sqrt(d * np.mean(diff, axis=-1, out=out), out=out)


def rmsd_n(a: np.ndarray, b: np.ndarray, n: int, out: np.ndarray = None) -> np.ndarray:
    """Calculate RMSD(s) between sets of data points
    with known number of points.

    Consider using :func:`rmsd_d` if a calculation based on the
    known dimensionality of points is preferred.

    The RMSD is calculated here between two sets of points
    `a` and `b` as `sqrt(sum((a - b)^2) / n)`,
    where `n` is the number of points.
    To comply with the standard definition of metrics,
    `a` and `b` are generally expected to be 1D arrays
    of shape `(n * d,)`. In other words, each point set is treated as
    a point itself in a higher-dimensional space.
    Sets can be batched by adding a leading axis.

    The input arrays need to have compatible shapes within
    broadcasting rules.
    The following input shapes in `a` and `b` are supported:

      * `(n * d,)` vs `(n * d,)`: Single pair of point sets;
            returns a single RMSD value.
      * `(m, n * d)` vs `(n * d,)`: `m` point sets against
            a single reference set;
            returns `m` RMSD values.
      * `(m, n * d)` vs `(m, n * d,)`: Two batches of `m` point sets against
            each other; assumes element-wise pairing of batches and
            returns `m` RMSD values.

    Args:
        a: Input point sets, generally a 1D array of shape `(n * d,)`.
        b: Other input point sets, generally a 1D array of shape `(n * d,)`.
        n: Number `n` of points in `a` and `b`.
        out: Optional output array to store results.

    Returns:
        The RMSD between point sets in `a` and `b`.
    """

    diff = a - b
    np.square(diff, out=diff)
    return np.sqrt(np.sum(diff, axis=-1, out=out) / n, out=out)


def pairwise_rmsd(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    d: int = 3,
    n: int | None = None,
):
    """Calculate pairwise RMSDs.

    Analoguous to :func:`sklearn.metrics.pairwise.pairwise_distances`.
    This function assumes data to be passed as 2D arrays of
    shape `(m, n * d)`, i.e. `m` sets of `n` points in `d` dimensions.
    Please refer to :func:`~awsem.analysis.metrics.rmsd.rmsd` for a
    higher-level alternative when dealing with molecular structures,
    i.e. input of shape `(m, n, d)`.

    RMSD calculation depends on knowing either the number
    `n` or dimensionality `d` of points in `X` and `Y`.

    Args:
        X: Input point array of shape `(k, n * d)`.
        Y: Optional second input point array of shape
            `(l, n * d)`. If `None`, `X` is used as `Y`.
        d: Dimensionality `d` of points in `X` and `Y`.
        n: Number `n` of points in `X` and `Y`. If provided, this value
            takes precedence over `d`.

    Returns:
        A matrix of RMSD distances between point set pairs in `X` and `Y`
        of shape `(k, l)`.
    """

    X, Y = check_pairwise_arrays(X, Y)

    if n is None:
        n = X.shape[1] // d

    return np.sqrt(pairwise_distances(X, Y, squared=True) / n)


def paired_rmsd(
    X: np.ndarray,
    Y: np.ndarray,
    d: int = 3,
    n: int | None = None,
    out: np.ndarray = None,
):
    """Calculate paired RMSDs.

    Analoguous to :func:`sklearn.metrics.pairwise.paired_distances`.
    This function assumes data to be passed as 2D arrays of
    shape `(m, n * d)`, i.e. `m` sets of `n` points in `d` dimensions.
    Please refer to :func:`~awsem.analysis.metrics.rmsd.rmsd` for a
    higher-level alternative when dealing with molecular structures,
    i.e. input of shape `(m, n, d)`.

    RMSD calculation depends on knowing either the number
    `n` or dimensionality `d` of points in `X` and `Y`.

    Args:
        X: Input point array of shape `(k, n * d)`.
        Y: Optional second input point array of shape
            `(l, n * d)`. If `None`, `X` is used as `Y`.
        d: Dimensionality `d` of points in `X` and `Y`.
        n: Number `n` of points in `X` and `Y`. If provided, this value
            takes precedence over `d`.
        out: Optional output array to store results.

    Returns:
        A matrix of RMSD distances between point set pairs in `X` and `Y`
        of shape `(k, l)`.
    """

    X, Y = check_paired_arrays(X, Y)

    if n is not None:
        return rmsd_n(X, Y, n, out=out)
    return rmsd_d(X, Y, d, out=out)

