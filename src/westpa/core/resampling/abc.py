import numpy as np

from abc import ABC, abstractmethod
from numpy.typing import ArrayLike, NDArray
from typing import Tuple


class WEResampler(ABC):

    @abstractmethod
    def resample(
            self,
            coords: ArrayLike,
            weights: ArrayLike,
    ) -> Tuple[NDArray[int], NDArray[float]]:
        """Perform resampling.

        Parameters
        ----------
        coords : ArrayLike
            Walker coordinates.
        weights : ArrayLike
            Walker weights.

        Returns
        -------
        NDArray[int]
            The parent indices of the resampled walkers.
        NDArray[float]
            The weights of the resampled walkers.

        """
        # example: no splitting or merging
        parent_indices = np.arange(len(coords))
        new_weights = np.array(weights)
        return parent_indices, new_weights
