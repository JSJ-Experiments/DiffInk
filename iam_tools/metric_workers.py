"""CPU-only spatial metrics in spawned processes; never fork a CUDA context."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def geometry_job(xy, target, states):
    from .trajectory_geometry import geometry_metrics
    return geometry_metrics(xy, target, states)


def metric_pool(workers=3):
    if not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError('1–4 CPU workers required')
    return ProcessPoolExecutor(max_workers=workers,
                               mp_context=multiprocessing.get_context('spawn'))
