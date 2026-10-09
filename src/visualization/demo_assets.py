"""Build the interactive demo assets in demo/ (real Sentinel-2 crops + automated boundaries).

  python -m src.visualization.demo_assets      # then open demo/index.html (double-click works)

All images are the clearest core-window scene of each year, same colour stretch for every year;
outlines/terminus/area come from experiments/<model>/<year>/ and outputs/ (no manual input).
"""
import json, pickle, numpy as np, geopandas as gpd, pandas as pd
from PIL import Image
from shapely.geometry import mapping
from src.common import ROOT, Grid, read_raster
from src.visualization.maps import clearest_scene
OUT = ROOT / "demo"
OUT.mkdir(exist_ok=True)
g = Grid.load(ROOT/"data/inventories/grid.json")
YEARS = [2017, 2019, 2021, 2023, 2025]
MODELS = {"M0_global": "Global model (M0)", "M1_hma_official": "Himalaya fine-tuned (M1)", "M2_cs_selftrain": "Chhota Shigri adapted (M2)", "M3_hma_sar": "+ Radar (M3)"}
rgi = gpd.read_file(ROOT/"data/inventories/chhota_shigri.gpkg", layer="rgi7")
b = rgi.total_bounds; pad = 700
x0, x1, y0, y1 = b[0]-pad, b[2]+pad, b[1]-pad, b[3]+pad
c0, c1 = int((x0-g.xmin)/g.res), int((x1-g.xmin)/g.res); r0, r1 = int((g.ymax-y1)/g.res), int((g.ymax-y0)/g.res)
S = 2  # upscale factor for display
W, H = (c1-c0)*S, (r1-r0)*S
ref = read_raster(ROOT/f"data/sentinel2/scenes/{clearest_scene(2023)}/optical_toa.tif")[[2,1,0], r0:r1, c0:c1]
lo, hi = np.nanpercentile(ref, 1), np.nanpercentile(ref, 99.5)
def rgb_img(year, win=None, scale=S):
    rr0, rr1, cc0, cc1 = win or (r0, r1, c0, c1)
    o = read_raster(ROOT/f"data/sentinel2/scenes/{clearest_scene(year)}/optical_toa.tif")[[2,1,0], rr0:rr1, cc0:cc1]
    a = np.clip((np.nan_to_num(o) - lo)/(hi - lo), 0, 1) ** 0.85
    im = Image.fromarray((np.moveaxis(a, 0, -1)*255).astype(np.uint8))
    return im.resize((im.width*scale, im.height*scale), Image.LANCZOS)
def to_px(geom, cx0, cy1, scale):
    def ring(coords):
        return "M" + "L".join(f"{(x-cx0)/g.res*scale:.1f},{(cy1-y)/g.res*scale:.1f}" for x, y, *_ in coords) + "Z"
    parts = getattr(geom, "geoms", [geom]); d = []
    for p in parts:
        if p.geom_type != "Polygon": continue
        d.append(ring(p.exterior.coords)); d += [ring(i.coords) for i in p.interiors]
    return "".join(d)
X0, Y1 = g.xmin + c0*g.res, g.ymax - r0*g.res
data = dict(width=W, height=H, pixel_m=g.res/S, years=YEARS, models=MODELS, scenes={}, outlines={}, areas={}, terminus={})
for y in YEARS:
    rgb_img(y).save(OUT/f"s2_{y}.jpg", quality=80)
    data["scenes"][y] = clearest_scene(y)
for m in MODELS:
    data["outlines"][m] = {}; data["areas"][m] = {}; data["terminus"][m] = {}
    tp = gpd.read_file(ROOT/f"outputs/polygons/{m}_terminus_points.gpkg")
    for y in YEARS:
        o = gpd.read_file(ROOT/f"experiments/{m}/{y}/outline.gpkg").geometry.iloc[0].simplify(6)
        data["outlines"][m][y] = to_px(o, X0, Y1, S)
        data["areas"][m][y] = round(json.load(open(ROOT/f"experiments/{m}/{y}/summary.json"))["area_km2"], 2)
        p = tp[tp.year == y].geometry.iloc[0]
        data["terminus"][m][y] = [round((p.x-X0)/g.res*S, 1), round((Y1-p.y)/g.res*S, 1)]
data["rgi"] = to_px(rgi.geometry.iloc[0].simplify(6), X0, Y1, S)
cl = gpd.read_file(ROOT/"data/inventories/chhota_shigri.gpkg", layer="centerline").geometry.iloc[0]
data["centerline"] = "M" + "L".join(f"{(x-X0)/g.res*S:.1f},{(Y1-yy)/g.res*S:.1f}" for x, yy, *_ in cl.coords[::5])
# AI confidence overlays (calibrated probability, M2) as transparent PNG
iso = pickle.load(open(ROOT/"models/calibration/M2_cs_selftrain_isotonic.pickle", "rb"))
for y in YEARS:
    p = read_raster(ROOT/f"experiments/M2_cs_selftrain/{y}/prob_glacier.tif")[r0:r1, c0:c1]
    pc = iso.predict(p.ravel()).reshape(p.shape)
    rgba = np.zeros(p.shape + (4,), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 40, 170, 255
    rgba[..., 3] = (np.clip(pc, 0, 1) * 190).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(OUT/f"conf_{y}.png", optimize=True)
# year-of-loss overlay
yol = read_raster(ROOT/"outputs/change_maps/M2_cs_selftrain_year_of_loss.tif")[r0:r1, c0:c1].astype(int)
cols = {1: (200, 230, 255, 120), 2: (150, 150, 150, 170), 2019: (255, 214, 10, 255), 2021: (255, 159, 28, 255), 2023: (240, 90, 30, 255), 2025: (200, 20, 40, 255)}
rgba = np.zeros(yol.shape + (4,), np.uint8)
for k, c in cols.items(): rgba[yol == k] = c
rgba[yol < 0] = (60, 120, 255, 200)
Image.fromarray(rgba, "RGBA").resize((W, H), Image.NEAREST).save(OUT/"year_of_loss.png", optimize=True)
data["year_of_loss_km2"] = json.load(open(ROOT/"outputs/statistics/M2_cs_selftrain_year_of_loss_km2.json"))
# hotspots
z = read_raster(ROOT/"outputs/change_maps/M2_cs_selftrain_hotspot_gistar.tif")[r0:r1, c0:c1]
rgba = np.zeros(z.shape + (4,), np.uint8); hot = np.nan_to_num(z) > 1.96; cold = np.nan_to_num(z) < -1.96
rgba[hot] = (220, 30, 60, 200); rgba[cold] = (60, 130, 230, 110)
Image.fromarray(rgba, "RGBA").resize((W, H), Image.NEAREST).save(OUT/"hotspots.png", optimize=True)
# terminus close-ups (M2), 1.5 km window, 5x
tp = gpd.read_file(ROOT/"outputs/polygons/M2_cs_selftrain_terminus_points.gpkg")
cx, cy = tp.geometry.x.mean(), tp.geometry.y.mean() + 150
half = 750; tc0, tc1 = int((cx-half-g.xmin)/g.res), int((cx+half-g.xmin)/g.res); tr0, tr1 = int((g.ymax-(cy+half))/g.res), int((g.ymax-(cy-half))/g.res)
TS = 5; TX0, TY1 = g.xmin + tc0*g.res, g.ymax - tr0*g.res
data["zoom"] = dict(width=(tc1-tc0)*TS, height=(tr1-tr0)*TS, pixel_m=g.res/TS, outlines={}, terminus={})
from shapely.geometry import box
clipbox = box(TX0, TY1-(tr1-tr0)*g.res, TX0+(tc1-tc0)*g.res, TY1)
for y in YEARS:
    rgb_img(y, (tr0, tr1, tc0, tc1), TS).save(OUT/f"zoom_{y}.jpg", quality=88)
    o = gpd.read_file(ROOT/f"experiments/M2_cs_selftrain/{y}/outline.gpkg").geometry.iloc[0].intersection(clipbox)
    data["zoom"]["outlines"][y] = to_px(o, TX0, TY1, TS)
    p = tp[tp.year == y].geometry.iloc[0]; data["zoom"]["terminus"][y] = [round((p.x-TX0)/g.res*TS, 1), round((TY1-p.y)/g.res*TS, 1)]
data["zoom"]["centerline"] = "M" + "L".join(f"{(x-TX0)/g.res*TS:.1f},{(TY1-yy)/g.res*TS:.1f}" for x, yy, *_ in cl.coords[::3])
# stats
data["benchmark"] = {k: round(v["pooled_iou"], 3) for k, v in json.load(open(ROOT/"experiments/benchmark_eval_test_summary.json")).items() if k.startswith("M")}
data["ablation"] = {k: round(v["pooled_iou"], 3) for k, v in json.load(open(ROOT/"experiments/benchmark_eval_test_sar_summary.json")).items() if k.startswith("M")}
t = pd.read_csv(ROOT/"outputs/statistics/M2_cs_selftrain_terminus.csv"); a = pd.read_csv(ROOT/"outputs/statistics/M2_cs_selftrain_area.csv")
data["m2_series"] = [dict(year=int(r.year), date=r.date, area=round(r.area_km2, 2), sigma=round(r.sigma_area_km2, 2), n=int(r.n_scenes)) for r in a.itertuples()]
data["m2_terminus"] = [dict(year=int(r.year), cl=float(r.retreat_cum_centerline_m), box=round(float(r.retreat_cum_box_m), 1)) for r in t.itertuples()]
data["periods"] = pd.read_csv(ROOT/"outputs/statistics/summary_terminus_periods.csv")[["period","median_box_m","min_box_m","max_box_m","median_centerline_m","min_centerline_m","max_centerline_m"]].round(0).to_dict("records")
json.dump(data, open(OUT/"demo_data.json", "w"))
(OUT/"demo_data.js").write_text("window.DEMO_DATA = " + json.dumps(data) + ";\n")  # lets index.html work from file://
print("crop", W, H, "zoom", data["zoom"]["width"], data["zoom"]["height"])
