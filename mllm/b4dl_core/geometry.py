"""Geometry conventions: ego x forward, y left, z up; units metres/seconds."""
import math
import numpy as np

CLASSES = ("car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian",
           "motorcycle", "bicycle", "traffic_cone", "barrier")
SECTORS = ("front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right")
MOTIONS = ("closer", "farther", "left", "right", "similar")
CATEGORY_MAP = {
    "vehicle.car": "car", "vehicle.truck": "truck", "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus", "vehicle.trailer": "trailer", "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle", "vehicle.bicycle": "bicycle",
    "movable_object.trafficcone": "traffic_cone", "movable_object.barrier": "barrier",
    **{f"human.pedestrian.{x}": "pedestrian" for x in ("adult", "child", "construction_worker", "police_officer")},
}


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-3:
        raise ValueError("invalid unit quaternion (w,x,y,z)")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def transform(pose):
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation(pose["rotation"])
    translation = np.asarray(pose["translation"], dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("invalid translation")
    value[:3, 3] = translation
    return value


def global_to_ego(center, ego_pose):
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("invalid box centre")
    pose = transform(ego_pose)
    return pose[:3, :3].T @ (center - pose[:3, 3])


def sector(x, y):
    if not math.isfinite(x) or not math.isfinite(y) or math.hypot(x, y) < 1e-8:
        raise ValueError("undefined direction at ego origin")
    angle = math.degrees(math.atan2(y, x))
    return SECTORS[int(math.floor((angle + 22.5) / 45)) % 8]


def sector_margin(x, y):
    return min((math.degrees(math.atan2(y, x)) + 22.5) % 45,
               45 - (math.degrees(math.atan2(y, x)) + 22.5) % 45)


def motion_values(start, end):
    d0, d1 = math.hypot(*start[:2]), math.hypot(*end[:2])
    return {"d0": d0, "d1": d1, "delta_d": d1-d0, "delta_y": float(end[1]-start[1])}


def motion_truth(values):
    dd, dy = values["delta_d"], values["delta_y"]
    if not math.isfinite(dd) or not math.isfinite(dy):
        raise ValueError("nonfinite motion")
    return {"closer": dd <= -2, "farther": dd >= 2, "left": dy >= 1,
            "right": dy <= -1, "similar": abs(dd) < 1 and abs(dy) < .5}


def motion_margin(values, predicate):
    dd, dy = values["delta_d"], values["delta_y"]
    return {"closer": abs(dd+2), "farther": abs(dd-2), "left": abs(dy-1),
            "right": abs(dy+1), "similar": min(abs(abs(dd)-1), abs(abs(dy)-.5))}[predicate]
