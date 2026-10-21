from abc import ABC, abstractmethod
from functools import partial
from operator import attrgetter

import numpy as np


class BinMapperBase(ABC):
    """Base class for bin mappers.
    Subclasses must implement the :meth:`assign` method and the :attr:`labels` property.

    Parameters
    ----------
    coord_getter : callable, optional
        Function that returns the coordinate time series to use for binning
        a segment. Must accept a segment and return a 2-D array. Defaults to
        ``attrgetter('pcoord')``.

    Attributes
    ----------
    labels : iterable of str
    nbins : int
        Number of bins mapped to.
    coord_getter : callable
        Function that returns the coordinate time series to use for binning
        a segment.

    Examples
    --------

    >>> import westpa.api as we
    >>> class TwoBinMapper(we.BinMapperBase):
    ...     @property
    ...     def labels(self):
    ...         return ['-', '+']
    ...     def assign(self, coords, weights, output):
    ...         output[:] = coords[:, 0] > 0
    ...         return output
    ...

    """

    def __init__(self, coord_getter=None):
        self.coord_getter = coord_getter or attrgetter('pcoord')

    @property
    @abstractmethod
    def labels(self):
        """Bin labels."""
        ...

    @property
    def nbins(self):
        return len(list(self.labels))

    @property
    def coord_getter(self):
        return self._coord_getter

    @coord_getter.setter
    def coord_getter(self, value):
        if not callable(value):
            raise TypeError("'coord_getter' must be callable")
        self._coord_getter = value

    @abstractmethod
    def assign(self, coords, weights, output):
        """Assign walkers to bins.

        Parameters
        ----------
        coords : numpy.ndarray of shape (n, d)
            Coordinates of each walker.
        weights : numpy.ndarray of shape (n,)
            Weight of each walker.
        output : numpy.ndarray of shape (n,)
            Array for storing output, initialized to -1.

        Returns
        -------
        output : numpy.ndarray of shape (n,)
            Bin assignment for each walker. Array elements must be in
            ``range(self.nbins)``.

        """
        ...

    def _get_coord(self, segment, coord_index):
        return self.coord_getter(segment)[coord_index]

    def __call__(self, segments, coord_index=-1):
        get_coord = partial(self._get_coord, coord_index=coord_index)

        coords = np.array(list(map(get_coord, segments)))
        weights = np.array(list(map(attrgetter('weight'), segments)))
        output = np.full(len(segments), -1)

        return self.assign(coords, weights, output)
