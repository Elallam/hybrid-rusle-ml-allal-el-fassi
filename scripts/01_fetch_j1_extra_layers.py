"""
01_fetch_j1_extra_layers.py -- Fetch the 3 J1 layers NOT already available
from C1 (allal-erosion) or C2 (Comparative Evaluation of Machine Learning):
  MSAVI, Albedo   (Sentinel-2 -- C2 only fetched NDWI/NDBI/EVI/BSI/SAVI)
  Bulk_Density    (SoilGrids bdod -- C1's soilgrids.tif has sand/silt/clay/soc
                   bands but not bulk density)

Everything else J1 needs beyond C2's 20-feature schema (Curvature_Plan/
Profile, TPI, Valley_Depth, Rainfall_Seasonality, Max_Monthly_Rain,
Sand_Content, SOC) is derivable purely from rasters C1/C2 already
downloaded (DEM, 12-band monthly CHIRPS, 8-band SoilGrids) -- see
real_data.py's module docstring for exactly which band.

Run with the `erosion` conda env (has earthengine-api + geemap already
authenticated for EE project bircool2-c6a44):
    conda run -n erosion python scripts/01_fetch_j1_extra_layers.py

Output: data/raw/j1_extra_layers_2025.tif (3 bands: MSAVI, Albedo, Bulk_Density)
"""
import os
import ee
import geemap
import geopandas as gpd

YEAR = 2025
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
OUT_PATH = os.path.join(OUT_DIR, f"j1_extra_layers_{YEAR}.tif")
# article_j1/ lives one level deeper than C2's project (under Hybrid framework/,
# itself a sibling of allal-erosion under phd/) -- hardcoded, not derived via
# dirname chains, since that's what actually matches this project's layout.
BOUNDARY_PATH = r"C:\Users\soufi\OneDrive\Bureau\phd\allal-erosion\data\boundary\allal_watershed.geojson"


def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000).copyProperties(image, ['system:time_start'])


def compute_msavi_albedo(image):
    b2, b3, b4, b8, b11, b12 = (image.select('B2'), image.select('B3'), image.select('B4'),
                                 image.select('B8'), image.select('B11'), image.select('B12'))
    # MSAVI (Qi et al. 1994): 2*NIR+1 - sqrt((2*NIR+1)^2 - 8*(NIR-RED)) all /2
    term = b8.multiply(2).add(1)
    msavi = term.subtract(term.pow(2).subtract(b8.subtract(b4).multiply(8)).sqrt()).divide(2).rename('MSAVI')
    # Broadband surface albedo, Liang (2001)-style linear combination of S2
    # bands (blue/red/nir/swir1/swir2) -- a standard simplified approximation
    # used widely in RS erosion/LULC studies when full narrow-to-broadband
    # conversion tables aren't needed.
    albedo = (b2.multiply(0.356).add(b4.multiply(0.130)).add(b8.multiply(0.373))
              .add(b11.multiply(0.085)).add(b12.multiply(0.072)).subtract(0.0018)).rename('Albedo')
    return ee.Image.cat([msavi, albedo])


def main():
    ee.Initialize(project='bircool2-c6a44')

    gdf = gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:4326")
    geom = ee.Geometry(gdf.geometry.iloc[0].__geo_interface__)

    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterDate(f'{YEAR}-01-01', f'{YEAR}-12-31')
           .filterBounds(geom)
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
           .map(mask_s2_clouds))
    n = col.size().getInfo()
    print(f"  Sentinel-2 SR images found for {YEAR}: {n}")
    if n == 0:
        raise RuntimeError("No Sentinel-2 images found -- widen the cloud filter or date range")

    composite = col.median()
    s2_layers = compute_msavi_albedo(composite)

    # SoilGrids v2 bulk density (bdod, 0-5cm and 5-15cm), same product family
    # as C1's existing soilgrids.tif (sand/silt/clay/soc), units cg/cm3 (SoilGrids
    # native encoding -- kept as-is, config expects "kg/dm3"-ish scale so we
    # divide by 100 to match the g/kg->%% style scaling used for the other
    # soilgrids bands in this project, i.e. cg/cm3 -> kg/dm3 == g/cm3).
    bdod = ee.Image('projects/soilgrids-isric/bdod_mean')
    bdod_layers = bdod.select(['bdod_0-5cm_mean', 'bdod_5-15cm_mean']).rename(
        ['Bulk_Density_0_5', 'Bulk_Density_5_15']).divide(100.0)

    combined = ee.Image.cat([s2_layers, bdod_layers]).clip(geom)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"  Downloading MSAVI+Albedo+Bulk_Density to {OUT_PATH} (crs=EPSG:32629, scale=30m)...")
    geemap.download_ee_image(combined, OUT_PATH, region=geom, crs='EPSG:32629', scale=30,
                              max_tile_dim=800)
    print(f"  Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
