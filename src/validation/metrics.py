"""Segmentation and boundary metrics for binary glacier masks."""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def binary_metrics(pred, ref, valid=None, res=10.0, tol_px=2):
    pred, ref = pred.astype(bool), ref.astype(bool)
    if valid is not None:
        pred, ref = pred & valid, ref & valid
    tp = float((pred & ref).sum())
    fp = float((pred & ~ref).sum())
    fn = float((~pred & ref).sum())
    iou = tp / max(tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    a_pred, a_ref = pred.sum() * res ** 2 / 1e6, ref.sum() * res ** 2 / 1e6
    out = dict(iou=iou, f1=f1, precision=prec, recall=rec, area_pred_km2=a_pred, area_ref_km2=a_ref,
               area_error_km2=a_pred - a_ref, rel_area_error=(a_pred - a_ref) / max(a_ref, 1e-9))
    out.update(boundary_metrics(pred, ref, res, tol_px))
    return out


def _edges(m):
    return m & ~ndi.binary_erosion(m, structure=np.ones((3, 3)), border_value=0)


def boundary_metrics(pred, ref, res=10.0, tol_px=2):
    """Boundary F1 at tolerance tol_px, mean/95th-percentile/max (Hausdorff) boundary distances."""
    ep, er = _edges(pred), _edges(ref)
    if not ep.any() or not er.any():
        return dict(boundary_f1=np.nan, mean_boundary_dist_m=np.nan, p95_boundary_dist_m=np.nan, hausdorff_m=np.nan)
    d_to_ref = ndi.distance_transform_edt(~er) * res
    d_to_pred = ndi.distance_transform_edt(~ep) * res
    dp, dr = d_to_ref[ep], d_to_pred[er]
    tol = tol_px * res
    bp, br = (dp <= tol).mean(), (dr <= tol).mean()
    both = np.concatenate([dp, dr])
    return dict(boundary_f1=float(2 * bp * br / max(bp + br, 1e-9)), mean_boundary_dist_m=float(both.mean()),
                p95_boundary_dist_m=float(np.percentile(both, 95)), hausdorff_m=float(both.max()))
