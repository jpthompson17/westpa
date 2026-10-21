import mdtraj


class MDStoragePlugin:
    """Repacks MD trajectory data into per-iteration HDF5 files.

    Parameters
    ----------
    topology_file : path-like
        Path to an MDTraj-compatible topology file.
    trajectory_filename : str
        Name of the trajectory file output for each segment.
    log_filename : str, optional
        Name of the log file output for each segment.
    iter_h5file_template : str, optional
        Path template for per-iteration HDF5 files. Must contain an
        ``{n_iter}`` replacement field. If a relative path is given, it is
        assumed to be relative to the current working directory. Defaults
        to ``'iter_{n_iter:06d}.h5'``.
    cleanup : bool, default True
        If True (the default), delete the output files after their contents
        are copied to the HDF5 file; also delete any folders left empty by
        this operation. If False, leave the original output files intact.

    Notes
    -----
    The trajectory and log files are assumed to be located in the same
    directory as the restart file (i.e., the ``final_state.file``) for a
    given segment.

    The per-iteration HDF5 files written by this plugin extend the
    `MDTraj HDF5 format <https://mdtraj.org/1.9.4/hdf5_format.html>`_.
    The ``coordinates`` array contains the chained trajectory frames for
    each segment in the iteration. Restart and log files are stored as bytes
    in the ``restart`` and ``log`` groups, respectively.

    """

    def __init__(
        self,
        topology_file,
        trajectory_filename,
        log_filename=None,
        iter_h5file_template=None,
        cleanup=True,
    ):
        self.topology = mdtraj.load(topology_file).topology
        self.trajectory_filename = trajectory_filename
        self.log_filename = log_filename
        self.iter_h5file_template = iter_h5file_template
        self.cleanup = cleanup

    def finalize_iteration(self, sim):
        pass
