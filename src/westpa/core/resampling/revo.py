import numpy as np

from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import pairwise_distances
from westpa.core.resampling.abc import WEResampler
from typing import Callable, Optional, Tuple


@dataclass
class REVOResampler(WEResampler):
    """Resampling of Ensembles by Variation Optimization (REVO).[1]_

    Parameters
    ----------
    metric : str, default "euclidean"
        Metric to use for calculating the distance between walkers. It must be
        supported by the :func:`sklearn.metrics.pairwise_distances` function.
    distance_exponent : float, default 4.0
        Exponent of the distance term in the variation function.
    min_weight : float, default 1.0e-12
        Minimum walker weight.
    max_weight : float, default 0.1
        Maximum walker weight.
    merge_distance : float, optional
        Merge distance threshold. If provided, pairs of walkers separated
        by more than `merge_distance` will not be merged.
    novelty_func : Callable[[float, ArrayLike], float], optional
        A function that takes the weight and coordinates of a walker and
        returns a nonnegative number.
    max_iter : int, optional
        Maximum number of iterations to run.

    References
    ----------
    .. [1] N. Donyapour, N. M. Roussey, A. Dickson, "REVO: Resampling of
       ensembles by variation optimization", J. Chem. Phys. 150, 244112 (2019)
       https://doi.org/10.1063/1.5100521.

    """
    metric: str = "euclidean"
    distance_exponent: float = 4.0
    min_weight: float = 1.0e-12
    max_weight: float = 0.1
    merge_distance: Optional[float] = None
    novelty_func: Optional[Callable[[float, float], float]] = None
    max_iter: Optional[int] = None

    def resample(
            self,
            coords: ArrayLike,
            weights: ArrayLike,
    ) -> Tuple[NDArray[int], NDArray[float]]:
        """Perform resampling.

        Parameters
        ----------
        coords : ArrayLike, shape=(n, d)
            2D array of walker coordinates.
        weights : ArrayLike, shape=(n,)
            Vector of walker weights.

        Returns
        -------
        NDArray[int], shape=(n,)
            The parent indices of the resampled walkers.
        NDArray[float], shape=(n,)
            The weights of the resampled walkers.

        """
        return resample(
            coords,
            weights,
            metric=self.metric,
            distance_exponent=self.distance_exponent,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            merge_distance=self.merge_distance,
            novelty_func=self.novelty_func,
            max_iter=self.max_iter,
        )


def calc_variation(
        novelties: ArrayLike,
        distances: ArrayLike,
        distance_exponent: float,
) -> Tuple[float, NDArray[float]]:
    """Calculate the trajectory variation.

    Parameters
    ----------
    novelties : ArrayLike, shape=(n,)
        Vector of walker novelties.
    distances : ArrayLike, shape=(n, n)
        Matrix of inter-walker distances.
    distance_exponent : float
        Exponent of the distance term in the variation function.

    Returns
    -------
    float
        The total trajectory variation.
    NDArray[float], shape=(n,)
        The contribution of each walker to the variation.

    """
    v = (np.power(distances, distance_exponent)
         * np.einsum("i,j", novelties, novelties)).sum(axis=1)
    return v.sum(), v


def resample(
        coords: ArrayLike,
        weights: ArrayLike,
        metric: str = "euclidean",
        distance_exponent: float = 4.0,
        min_weight: float = 1.0e-12,
        max_weight: float = 0.1,
        merge_distance: Optional[float] = None,
        novelty_func: Optional[Callable[[float, float], float]] = None,
        max_iter: Optional[int] = None,
) -> Tuple[NDArray[int], NDArray[float]]:
    """Perform REVO resampling of a weighted ensemble.

    Parameters
    ----------
    coords : ArrayLike, shape=(n, d)
        2D array of walker coordinates.
    weights : ArrayLike, shape=(n,)
        Vector of walker weights.
    metric : str, default "euclidean"
        Metric to use for calculating the distance between walkers. It must be
        supported by the :func:`sklearn.metrics.pairwise_distances` function.
    distance_exponent : float, default 4.0
        Exponent of the distance term in the variation function.
    min_weight : float, default 1.0e-12
        Minimum walker weight.
    max_weight : float, default 0.1
        Maximum walker weight.
    merge_distance : float, optional
        Merge distance threshold. If provided, pairs of walkers separated
        by more than `merge_distance` will not be merged.
    novelty_func : Callable[[float, ArrayLike], float], optional
        A function that takes the weight and coordinates of a walker and
        returns a nonnegative number.
    max_iter : int, optional
        Maximum number of iterations to run.

    Returns
    -------
    NDArray[int], shape=(n,)
        The parent indices of the resampled walkers.
    NDArray[float], shape=(n,)
        The weights of the resampled walkers.

    """
    coords = np.array(coords)
    if coords.ndim != 2:
        raise ValueError("'coords' must be a 2D array")

    weights = np.array(weights)
    if weights.ndim != 1:
        raise ValueError("'weights' must be a 1D array")
    if weights.size != coords.shape[0]:
        raise ValueError('number of weights must match number of walkers')

    if novelty_func is not None:
        def get_novelties(ws, xs):
            return np.array([novelty_func(w, x) for w, x in zip(ws, xs)])
    else:
        a = np.log(min_weight / 100)
        def get_novelties(ws, xs):  # noqa
            return np.log(ws) - a  # see Donyapour et al. [1]_

    distances = pairwise_distances(coords, metric=metric, n_jobs=-1)

    # Calculate the trajectory variation.
    novelties = get_novelties(weights, coords)
    v_old, v = calc_variation(novelties, distances, distance_exponent)

    parent_indices = np.arange(coords.shape[0])

    max_iter = max_iter if max_iter is not None else np.inf
    n_iter = 1
    while n_iter <= max_iter:
        v = v.view(np.ma.MaskedArray)

        # Find the walker to be cloned.
        v.mask = weights < 2 * min_weight
        if v.mask.all():
            break
        i = v.argmax()

        # Find the walker to be pruned.
        v.mask[:] = False
        v.mask[i] = True
        if v.mask.all():
            break
        j = v.argmin()

        # Find the walker into which to merge the pruned weight.
        d = distances[j].view(np.ma.MaskedArray)
        d.mask = weights + weights[j] > max_weight
        if merge_distance is not None:
            d.mask |= d > merge_distance
        d.mask[[i, j]] = True
        if d.mask.all():
            break
        k = d.argmin()

        # Do cloning and merging.
        pruned_walker_weight = weights[j]
        pruned_walker_coord = coords[j]
        pruned_walker_parent_index = parent_indices[j]
        weights[i] = weights[j] = weights[i] / 2
        weights[k] += pruned_walker_weight
        coords[j] = coords[i]
        parent_indices[j] = parent_indices[i]
        distances[i, j] = distances[j, i] = 0
        distances[j] = distances[i]

        # Recalculate the trajectory variation.
        novelties = get_novelties(weights, coords)
        v_new, v = calc_variation(novelties, distances, distance_exponent)

        if v_new < v_old:
            # Undo the last cloning and merging step.
            weights[i] *= 2
            weights[j] = pruned_walker_weight
            weights[k] -= pruned_walker_weight
            coords[j] = pruned_walker_coord
            parent_indices[j] = pruned_walker_parent_index
            break

        v_old = v_new
        n_iter += 1

    return parent_indices, weights
