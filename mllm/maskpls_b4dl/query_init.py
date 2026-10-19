"""Input-only, deterministic FPS for the preregistered E2-Q query experiment."""
import numpy as np


def farthest_point_indices(xyz, count):
    """Original row ids, deduped by XYZ, deterministic FP64 CPU distance/ties.

    Start farthest from the sensor origin; ties choose the first original row.
    If unique points < count, cycle the completed FPS sequence. No RNG or GT.
    """
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError('expected nonempty finite XYZ')
    if type(count) is not int or count < 1:
        raise ValueError('positive integer count required')
    _, first = np.unique(xyz, axis=0, return_index=True)
    first.sort()  # np.unique sorts coordinates; restore input-order tie breaking.
    points = xyz[first].astype(np.float64)
    distance = np.full(len(points), np.inf)
    selected = []
    current = int(np.argmax(np.sum(points * points, axis=1)))
    for _ in range(min(count, len(points))):
        selected.append(current)
        delta = points - points[current]
        distance = np.minimum(distance, np.sum(delta * delta, axis=1))
        distance[selected] = -1
        current = int(np.argmax(distance))
    return np.resize(first[np.asarray(selected)], count).astype(np.int64)
