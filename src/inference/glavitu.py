"""GlaViTU inference on ROI rasters, faithful to the official deploy pipeline.

Reproduces third_party/glavitu/compile_features.py + deploy.py:
  * min-max normalisation with utils/dataset_stats.pickle (global benchmark statistics)
  * NaN -> 0, centred zero padding to a multiple of the patch size (+1 patch)
  * 384 px patches, stride 192, Gaussian-weighted blending of overlapping softmax outputs
  * confidence = 1 - Shannon entropy (base 2) of the two-class softmax
The official deploy.py cannot be used directly because it opens the training HDF5 at import time.
Adapted from GlaViTU (Maslov et al., 2025, GPL-2.0).
"""
from __future__ import annotations

import pickle
import sys
from functools import lru_cache

import numpy as np

from src.common import ROOT

GLAVITU = ROOT / "third_party" / "glavitu"


def configure_device():
    """GEOAI_DEVICE=cpu (default) | gpu. tensorflow-metal 1.1.0 hits a SIGBUS inside
    model.predict on real Chhota Shigri inputs (see docs/methodology.md), so CPU is the default."""
    import os
    import tensorflow as tf
    if os.environ.get("GEOAI_DEVICE", "cpu").lower() == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass  # devices already initialised


def _import_glavitu():
    configure_device()
    if str(GLAVITU) not in sys.path:
        sys.path.insert(0, str(GLAVITU))
    import models.mapping  # noqa: E402  (GlaViTU package)
    return models.mapping


@lru_cache(maxsize=1)
def dataset_stats():
    with open(GLAVITU / "utils" / "dataset_stats.pickle", "rb") as f:
        return pickle.load(f)


def normalise(optical, elevation, slope):
    """optical: (H, W, 6) TOA reflectance; elevation (m), slope (deg): (H, W)."""
    mins, maxs = dataset_stats()
    optical = np.where(np.isnan(optical), 0, optical)
    optical = (optical - mins["optical"]) / (maxs["optical"] - mins["optical"])
    elev = (elevation - mins["elevation"]) / (maxs["elevation"] - mins["elevation"])
    slp = (slope - mins["slope"]) / (maxs["slope"] - mins["slope"])
    return {"optical": optical.astype(np.float32), "dem": np.stack([elev, slp], -1).astype(np.float32)}


def normalise_sar(sigma0_linear, key="co_pol_sar"):
    """compile_features.py: log10(x + 1e-6), then min-max with the official dataset stats."""
    mins, maxs = dataset_stats()
    arr = np.log10(np.moveaxis(sigma0_linear, 0, -1) + 1e-6)
    return ((arr - mins[key]) / (maxs[key] - mins[key])).astype(np.float32)


def build_model(weights, patch_size=384, features=("optical", "dem"), mode="testing", dropout=0.1,
                inference_dropout=False):
    mapping = _import_glavitu()
    shapes = {"optical": (patch_size, patch_size, 6), "dem": (patch_size, patch_size, 2),
              "co_pol_sar": (patch_size, patch_size, 2), "cross_pol_sar": (patch_size, patch_size, 2)}
    train_m, test_m = mapping.GlaViTU(input_shapes={f: shapes[f] for f in features}, n_outputs=2,
                                      last_activation="softmax", dropout=dropout,
                                      inference_dropout=inference_dropout, name="GlaViTU",
                                      use_deepsupervision=True)
    if weights:
        train_m.load_weights(str(weights))
    return train_m if mode == "training" else test_m


def build_model_with_transfer(weights, features, patch_size=384, base_features=("optical", "dem")):
    """Model with a different set of input groups initialised from an optical+DEM checkpoint.

    GlaViTU's FusionBlock has one conv + squeeze-excitation branch per input group (summed).
    Branches are matched BY INPUT NAME: groups present in `base_features` get the checkpoint's
    branch, new groups (e.g. SAR) keep random init, dropped groups (e.g. DEM for an optical-only
    ablation) are simply absent. All downstream layers are copied. Returns the training model.
    """
    import tensorflow as tf
    old = build_model(weights, patch_size, features=base_features, mode="training")
    new = build_model(None, patch_size, features=tuple(features), mode="training")
    ol = [l for l in old.layers if not isinstance(l, tf.keras.layers.InputLayer)]
    nl = [l for l in new.layers if not isinstance(l, tf.keras.layers.InputLayer)]
    assert len(ol) == len(nl)
    copied = 0
    for lo, ln in zip(ol, nl):
        if type(lo).__name__ == "FusionBlock":
            for j, f in enumerate(features):
                if f in base_features:
                    i = base_features.index(f)
                    for attr in ("conv_branches", "excitation_denses1", "excitation_denses2"):
                        getattr(ln, attr)[j].set_weights(getattr(lo, attr)[i].get_weights())
                    copied += 1
        elif lo.get_weights():
            ln.set_weights(lo.get_weights()); copied += 1
    return new, copied


def gaussian_kernel(size, mu=0, sigma=1):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x ** 2 + y ** 2)
    return np.exp(-(d - mu) ** 2 / 2 / sigma ** 2) / np.sqrt(2 / np.pi) / sigma


def pad(features, patch_size):
    h, w, _ = features["optical"].shape
    th, tw = patch_size * int(h // patch_size + 1), patch_size * int(w // patch_size + 1)
    ph, pw = (th - h) // 2, (tw - w) // 2
    out = {}
    for k, a in features.items():
        p = np.zeros((th, tw, a.shape[-1]), np.float32)
        p[ph:ph + h, pw:pw + w] = a
        out[k] = p
    return out, (h, w, ph, pw)


def predict(model, features, patch_size=384, batch_size=4):
    """Sliding-window prediction. Returns (H, W, 2) softmax probabilities."""
    names = {i.name.split(":")[0] for i in model.inputs}
    features = {k: v for k, v in features.items() if k in names}
    feats, (h, w, ph, pw) = pad(features, patch_size)
    H, W, _ = feats["optical"].shape
    acc = np.zeros((H, W, 2), np.float64)
    cnt = np.zeros((H, W, 1), np.float64)
    kern = gaussian_kernel(patch_size)[..., None]
    step = patch_size // 2
    coords = [(r, c) for r in range(0, H - patch_size + 1, step) for c in range(0, W - patch_size + 1, step)]
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = {k: np.stack([a[r:r + patch_size, c:c + patch_size] for r, c in chunk]) for k, a in feats.items()}
        probs = model.predict(batch, verbose=0)
        for (r, c), p in zip(chunk, probs):
            acc[r:r + patch_size, c:c + patch_size] += kern * p
            cnt[r:r + patch_size, c:c + patch_size] += kern
    prob = acc / cnt
    return prob[ph:ph + h, pw:pw + w].astype(np.float32)


def shannon_confidence(prob, eps=1e-6):
    p = np.clip(prob, eps, 1 - eps)
    return (1 + np.sum(p * np.log2(p), axis=-1)).astype(np.float32)
