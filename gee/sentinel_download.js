// NOT RUN — reference Earth Engine port of the multi-date ensemble scene selection (Option A).
// Mirrors configs/chhota_shigri.yaml: window_core 15 Aug-15 Sep, max domain cloud 0.5, TOA reflectance.
var cm = require('users/<you>/glacier:cloud_mask');           // adjust path
var rgi7 = ee.FeatureCollection('projects/<you>/assets/chhota_shigri_rgi7');   // export of data/inventories layer rgi7
var domain = rgi7.geometry().buffer(1000);
var roi = rgi7.geometry().buffer(2500).bounds();
var BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];
function yearScenes(year) {
  return ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
      .filterBounds(roi).filterDate(year + '-08-15', year + '-09-16')
      .filter(ee.Filter.eq('MGRS_TILE', '43SGR'))
      .map(cm.maskClouds).map(function (i) { return cm.domainCloudFraction(i, domain); })
      .filter(ee.Filter.lte('domain_cloud_frac', 0.5))
      .map(function (i) { return i.select(BANDS).divide(10000).toFloat()   // harmonised TOA reflectance
          .copyProperties(i, ['system:time_start', 'system:index', 'domain_cloud_frac']); });
}
[2017, 2019, 2021, 2023, 2025].forEach(function (y) {
  var col = yearScenes(y);
  print(y, col.aggregate_array('system:index'), col.aggregate_array('domain_cloud_frac'));
  col.toList(20).evaluate(function (list) {
    list.forEach(function (o) {
      var img = ee.Image(o.id);
      Export.image.toDrive({image: ee.Image(o.id).select(BANDS).divide(10000).toFloat(),
        description: 'CS_' + o.properties['system:index'], region: roi, scale: 10, crs: 'EPSG:32643'});
    });
  });
});
