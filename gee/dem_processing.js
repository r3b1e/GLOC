// NOT RUN — reference Earth Engine port of src/data/dem.py (Copernicus GLO-30 elevation/slope/aspect).
var rgi7 = ee.FeatureCollection('projects/<you>/assets/chhota_shigri_rgi7');
var roi = rgi7.geometry().buffer(2500).bounds();
var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').filterBounds(roi).mosaic()
    .setDefaultProjection('EPSG:32643', null, 30);
var terrain = ee.Terrain.products(dem);            // slope/aspect (deg) at 30 m
var out = dem.rename('elevation').addBands(terrain.select(['slope', 'aspect', 'hillshade']));
Export.image.toDrive({image: out.resample('bilinear').toFloat(), description: 'CS_copdem30_terrain',
  region: roi, scale: 10, crs: 'EPSG:32643'});
