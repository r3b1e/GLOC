// NOT RUN — reference Earth Engine port of src/data/sentinel2.py (this project used Planetary Computer
// + Google Cloud public L1C bucket + s2cloudless instead; no GEE account was available).
// Pixel-level cloud masking with Google Cloud Score+ (cs_cdf), evaluated over the glacier domain.
var CS_THRESHOLD = 0.60;   // Cloud Score+ recommended clear threshold
exports.maskClouds = function (img) {
  var cs = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
      .filter(ee.Filter.eq('system:index', img.get('system:index'))).first().select('cs_cdf');
  return img.updateMask(cs.gte(CS_THRESHOLD)).set('cs_joined', 1);
};
exports.domainCloudFraction = function (img, domain) {
  var clear = img.select('B8').mask().rename('clear');
  var frac = ee.Number(1).subtract(clear.reduceRegion({reducer: ee.Reducer.mean(), geometry: domain,
      scale: 20, maxPixels: 1e9}).get('clear'));
  return img.set('domain_cloud_frac', frac);
};
