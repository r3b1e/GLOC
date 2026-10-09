"""Shared helpers: config loading, project paths, and the fixed analysis grid."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str = "chhota_shigri") -> dict:
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return logging.getLogger(name)


@dataclass
class Grid:
    """A north-up raster grid. All multi-temporal products share one Grid."""
    crs: str
    xmin: float
    ymax: float
    width: int
    height: int
    res: float

    @property
    def transform(self):
        return from_origin(self.xmin, self.ymax, self.res, self.res)

    @property
    def bounds(self):
        return (self.xmin, self.ymax - self.height * self.res, self.xmin + self.width * self.res, self.ymax)

    def profile(self, dtype="float32", count=1, nodata=None):
        p = dict(driver="GTiff", crs=self.crs, transform=self.transform, width=self.width,
                 height=self.height, count=count, dtype=dtype, compress="deflate", tiled=True)
        if nodata is not None:
            p["nodata"] = nodata
        return p

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Grid":
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def from_bounds(cls, crs, bounds, res, snap=60.0):
        # snap outward to the 60 m lattice: S2 tile origins are multiples of 60 m, so 10/20/60 m
        # bands all fall on whole pixels of this grid (no sub-pixel resampling shifts)
        xmin = np.floor(bounds[0] / snap) * snap
        ymin = np.floor(bounds[1] / snap) * snap
        xmax = np.ceil(bounds[2] / snap) * snap
        ymax = np.ceil(bounds[3] / snap) * snap
        return cls(crs, float(xmin), float(ymax), int((xmax - xmin) / res), int((ymax - ymin) / res), float(res))


def write_raster(path, arr, grid: Grid, dtype=None, nodata=None):
    import rasterio
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    dtype = dtype or arr.dtype.name
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **grid.profile(dtype=dtype, count=arr.shape[0], nodata=nodata)) as dst:
        dst.write(arr.astype(dtype))


def read_raster(path):
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read()
    return arr[0] if arr.shape[0] == 1 else arr
