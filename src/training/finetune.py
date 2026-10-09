"""Fine-tune GlaViTU with the official data pipeline, loss, augmentations and LR schedule.

Mirrors third_party/glavitu/finetune.py (global weights -> regional fine-tuning):
  optimizer Adam, LR cosine decay from lr_start (official 5e-5), focal loss (gamma=2) on both
  deep-supervision branches (weights 0.5 / 1.0), augmentations: flips, rotation, crop&scale,
  gamma contrast, group-wise feature occlusion, Gaussian shift/noise, label smoothing 0.1.
  Checkpoint = best val IoU of the glacier class on the *validation* split.

Deviations for local hardware (Apple M4, 16 GB) are explicit CLI arguments and are recorded
in <out>/run_config.json: batch size, steps per epoch, number of epochs, tile subset.

Pseudo-label mode (--ignore_unlabelled): target pixels with label [0, 0] contribute zero loss
(focal loss is sum_c y_c * ...), and label smoothing is applied only to labelled pixels.

  python -m src.training.finetune --name M1_hma_local --init models/pretrained/glavitu_global_weights.h5 \
      --train data/benchmark/hma_train.hdf5 --val data/benchmark/hma_val.hdf5 --epochs 8 --steps 250
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

import h5py
import numpy as np

from src.common import ROOT, get_logger
from src.inference import glavitu

log = get_logger("finetune")


class MaskedLabelSmoothing:
    """Label smoothing that leaves unlabelled ([0, 0]) pixels untouched."""

    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing

    def build(self):
        import dataloaders

        smoothing = self.smoothing

        class _P(dataloaders.plugins.Plugin):
            def on_finalising(self, batch_x, batch_y):
                labelled = batch_y.sum(axis=-1, keepdims=True) > 0
                n = batch_y.shape[-1]
                sm = batch_y * (1 - smoothing) + smoothing / n
                return batch_x, np.where(labelled, sm, batch_y)

        return _P()


def build_loaders(train_path, val_path, features, batch_size, patch_size, label_smoothing, ignore_unlabelled):
    glavitu._import_glavitu()
    import dataloaders
    import dataloaders.transformations as T

    tr = h5py.File(train_path, "r")
    va = h5py.File(val_path, "r")
    train_plugins = [
        dataloaders.Augmentation([T.random_vertical_flip(), T.random_horizontal_flip(), T.random_rotation(),
                                  T.crop_and_scale(patch_size=patch_size)]),
        dataloaders.Augmentation([T.add_gamma_correction(features, 0.8, 1.2)]),
        dataloaders.Augmentation([T.groupwise_feature_occlusion(features, 192)]),
        dataloaders.Augmentation([T.add_gaussian_shift(features, [0.025] * len(features)),
                                  T.add_gaussian_noise(features, [0.005] * len(features))]),
    ]
    if label_smoothing:
        train_plugins.append(MaskedLabelSmoothing(label_smoothing).build() if ignore_unlabelled
                             else dataloaders.AddLabelSmoothing(smoothing=label_smoothing))
    train = dataloaders.DataLoader(dataloaders.RandomSampler(tr, patch_size, features, "outlines"),
                                   train_plugins, batch_size, "outlines", len_factor=4)
    val = dataloaders.DataLoader(dataloaders.ConsecutiveSampler(va, patch_size, features, "outlines"),
                                 [], batch_size, "outlines")
    return train, val


def git_rev(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--init", required=True, help="starting weights (.h5)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--features", nargs="+", default=["optical", "dem"])
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=250, help="steps per epoch (official: n_patches/bs*4)")
    ap.add_argument("--val_steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--ignore_unlabelled", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true", help="continue from <name>_last_weights.h5 after a crash")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed)
    import tensorflow as tf
    tf.random.set_seed(a.seed)
    out = ROOT / (a.out or f"models/{'hma_finetuned' if 'M1' in a.name else 'chhota_shigri'}")
    out.mkdir(parents=True, exist_ok=True)
    exp = ROOT / "experiments" / a.name.split("_")[0] / a.name
    exp.mkdir(parents=True, exist_ok=True)

    patch = 384
    train, val = build_loaders(ROOT / a.train, ROOT / a.val, a.features, a.batch, patch, a.label_smoothing,
                               a.ignore_unlabelled)
    import configs.finetuning as ft_cfg  # official focal loss + metrics
    import utils.deeplearning

    ckpt = out / f"{a.name}_weights.h5"
    last = out / f"{a.name}_last_weights.h5"
    log_csv = exp / "training_log.csv"
    initial_epoch = 0
    if a.resume and last.exists() and log_csv.exists():
        import pandas as pd
        initial_epoch = len(pd.read_csv(log_csv))
        log.info("resuming from %s at epoch %d", last.name, initial_epoch)
    base_feats = ("optical", "dem")
    if tuple(a.features) != base_feats and not (a.resume and last.exists()):
        # new input groups (e.g. SAR): transfer optical/DEM branches + trunk, init new branches
        model, n = glavitu.build_model_with_transfer(ROOT / a.init, tuple(a.features), patch)
        log.info("weight transfer: %d layers copied from %s", n, a.init)
    else:
        model = glavitu.build_model(last if (a.resume and last.exists()) else ROOT / a.init, patch,
                                    features=tuple(a.features), mode="training")
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(a.lr), loss=ft_cfg.get_loss(2),
                  metrics=ft_cfg.get_metrics(2, 2))

    cfg = dict(vars(a), weights_out=str(ckpt), glavitu_commit=git_rev(ROOT / "third_party/glavitu"),
               n_train_tiles=len(train.sampler.tiles), n_val_tiles=len(val.sampler.tiles),
               official_steps_per_epoch=len(train), val_batches=len(val), optimizer="Adam (legacy Keras)",
               loss="FocalLoss(gamma=2) x2 deep-supervision branches, weights 0.5/1.0",
               lr_schedule="cosine decay to 0 over epochs*steps", tensorflow=tf.__version__,
               device=os.environ.get("GEOAI_DEVICE", "cpu"), resumed_at_epoch=initial_epoch, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (exp / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    log.info("train tiles %d, val tiles %d, official steps/epoch %d, used %d", cfg["n_train_tiles"],
             cfg["n_val_tiles"], len(train), a.steps)

    best_so_far = None
    if initial_epoch:
        best_so_far = float(pd.read_csv(log_csv)["val_output_iou_1"].max())
    callbacks = [
        utils.deeplearning.LRCosineDecay(start=a.lr, decay_steps=a.epochs * a.steps, idle_steps=0, verbose=0),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_output_iou_1", mode="max",
                                           save_best_only=True, save_weights_only=True,
                                           initial_value_threshold=best_so_far),
        tf.keras.callbacks.ModelCheckpoint(str(last), save_weights_only=True),
        tf.keras.callbacks.CSVLogger(str(log_csv), append=initial_epoch > 0),
    ]
    callbacks[0].steps = initial_epoch * a.steps  # keep the cosine schedule continuous on resume
    model.fit(train, epochs=a.epochs, initial_epoch=initial_epoch, steps_per_epoch=a.steps, validation_data=val,
              validation_steps=a.val_steps, callbacks=callbacks, verbose=2)
    cfg["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (exp / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
