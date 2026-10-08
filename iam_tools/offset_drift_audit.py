"""Separate cumulative displacement bias from remaining trajectory distortion.

The endpoint-debiased path is an ORACLE descriptive diagnostic. It uses the real
endpoint and may not be used for generation, checkpoint selection or promotion.
Index-parametric linear drift is not speed, physical time or curve smoothing.
"""
import numpy as np


def cumulative_drift(target_xy,predicted_xy):
    target=np.asarray(target_xy,dtype=float);pred=np.asarray(predicted_xy,dtype=float)
    if target.ndim!=2 or target.shape[1]!=2 or pred.shape!=target.shape or not len(target) or not np.isfinite(target).all() or not np.isfinite(pred).all():
        raise ValueError('matching finite nonempty N,2 paths required')
    # Origin-to-first point included; telescoping yields endpoint residual / N.
    e=pred-target;de=np.diff(np.r_[np.zeros((1,2)),e],axis=0);mean=de.mean(0)
    drift=np.arange(1,len(e)+1)[:,None]*mean
    debiased=e-drift
    def rmse(x):return np.sqrt((x*x).mean(0)).tolist()
    return dict(points=len(e),mean_index_displacement_error=mean.tolist(),endpoint_xy_error=e[-1].tolist(),
                raw_axis_rmse=rmse(e),oracle_endpoint_debiased_axis_rmse=rmse(debiased),
                residual_displacement_axis_rmse=rmse(de),mean_index_bias_removed=True,
                diagnostic_only=True,not_generation=True,
                definition='Subtract point-index/N times TRUE endpoint residual; oracle endpoint uses target; isolates affine-in-index drift without claiming remaining error is independent or curvature')
