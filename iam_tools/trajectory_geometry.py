"""Spatial angle metrics and error diagnostics on nonuniform IAM/RDP points."""
import numpy as np
from .curve_audit import curve_metrics


def quantiles(values):
    a = np.asarray(values)
    return dict(count=int(a.size), median=float(np.median(a)) if a.size else None,
                p90=float(np.quantile(a, .9)) if a.size else None,
                p99=float(np.quantile(a, .99)) if a.size else None)


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def point_to_polyline(points, polyline):
    """Exact nearest distance from query vertices to piecewise straight segments."""
    if len(polyline) == 1:
        return np.linalg.norm(points-polyline[0], axis=1)
    a, vector = polyline[:-1], np.diff(polyline, axis=0)
    offset = points[:, None]-a[None]
    denominator = np.sum(vector**2, axis=1)
    fraction = np.sum(offset*vector[None], axis=-1)/np.maximum(denominator[None], 1e-30)
    projected = a[None]+np.clip(fraction, 0, 1)[..., None]*vector[None]
    return np.linalg.norm(points[:, None]-projected, axis=-1).min(1)


def geometry_metrics(prediction, target, states):
    """Tangent=segment direction; turn=signed angle between adjacent segments.

    These angles ARE geometric, not time derivatives. Zero-length segments are
    undefined and excluded; no normalization by point index/time is implied.
    Short segments naturally amplify angular sensitivity to position error.
    """
    p, t = np.asarray(prediction, float), np.asarray(target, float)
    s = np.asarray(states)
    result = curve_metrics(p, t, s)
    dp, dt = np.diff(p, axis=0), np.diff(t, axis=0)
    lp, lt = np.linalg.norm(dp, axis=1), np.linalg.norm(dt, axis=1)
    connected = s[:-1] == 0
    valid = connected & (lp > 1e-8) & (lt > 1e-8)
    ap, at = np.arctan2(dp[:, 1], dp[:, 0]), np.arctan2(dt[:, 1], dt[:, 0])
    tangent = np.abs(wrap_angle(ap-at))*180/np.pi
    turn_valid = valid[:-1] & valid[1:]
    turn_p, turn_t = wrap_angle(np.diff(ap)), wrap_angle(np.diff(at))
    turn_error = np.abs(wrap_angle(turn_p-turn_t))*180/np.pi
    result['tangent_angle_error_degrees'] = quantiles(tangent[valid])
    result['turn_angle_error_degrees'] = quantiles(turn_error[turn_valid])
    # Retain actual corners, not only shallow/straight regions.
    corners = turn_valid & (np.abs(turn_t)*180/np.pi >= 45)
    shallow = turn_valid & (np.abs(turn_t)*180/np.pi < 20)
    result['target_corner_turn_error_degrees'] = quantiles(turn_error[corners])
    result['target_shallow_turn_error_degrees'] = quantiles(turn_error[shallow])
    result['segment_lengths'] = quantiles(lt[connected])
    edges = []
    if valid.any():
        thresholds = np.quantile(lt[valid], [0, .25, .5, .75, 1])
        for j in range(4):
            selected = valid & (lt >= thresholds[j]) & ((lt <= thresholds[j+1]) if j == 3 else (lt < thresholds[j+1]))
            edges.append(dict(min_length=float(thresholds[j]), max_length=float(thresholds[j+1]),
                              tangent_error_degrees=quantiles(tangent[selected]),
                              delta_vector_error=quantiles(np.linalg.norm(dp-dt, axis=1)[selected])))
    result['length_quartiles'] = edges
    residual = p-t
    result['point_index_mod8'] = [dict(phase=j, count=int((np.arange(len(t))%8 == j).sum()),
                                     mean_residual=residual[np.arange(len(t))%8 == j].mean(0).tolist(),
                                     vector_rmse=float(np.sqrt(np.mean(np.sum(residual[np.arange(len(t))%8 == j]**2, axis=1)))))
                                 for j in range(min(8, len(t)))]
    distances = []
    start = 0
    for end in range(len(t)):
        if s[end] != 0 or end == len(t)-1:
            a, b = p[start:end+1], t[start:end+1]
            distances.extend([point_to_polyline(a, b), point_to_polyline(b, a)])
            start = end+1
    result['symmetric_vertex_to_polyline_distance'] = quantiles(np.concatenate(distances))
    result['symmetric_vertex_to_polyline_distance']['max'] = float(np.concatenate(distances).max())
    result['symmetric_vertex_to_polyline_distance']['definition'] = 'query both sets of vertices against corresponding true-stroke polyline; not exact continuous Hausdorff'
    return result
