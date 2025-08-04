"""
Example usage and testing script for satfarm package.

This script demonstrates various capabilities of the satfarm package
including image processing, index calculation, and visualization.
"""

from pathlib import Path
from pprint import pprint

import geopandas as gpd
import numpy as np
import rioxarray as rxr
from numpy import sqrt

from satfarm import SatImage

# read image
image_dir = Path("./test_data")
image = rxr.open_rasterio(
    image_dir / "20250105_035834_59_24e1_3B_AnalyticMS_SR_8b_clip_bandmath.tif"
)

# process
simage = (
    SatImage()
    .read_tif(image)
    .change_pixel_dtype("float32")
    .change_nodata(new_nodata=np.nan, old_nodata=0)
    .reproject("EPSG:4326")
    .shrink(distance=30)
    .reset_band_alias()
    .set_band_alias(
        [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B9",
            "B10",
            "B11",
            "B12",
            "B13",
        ]
    )
    .extract_band(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"])
    .set_band_alias(
        [
            "Coastal Blue",
            "Blue",
            "Green I",
            "Green II",
            "Yellow",
            "Red",
            "Red Edge",
            "NIR",
        ]
    )
    .apply_scale_factor(
        {
            "Coastal Blue": 1e-4,
            "Blue": 1e-4,
            "Green I": 1e-4,
            "Green II": 1e-4,
            "Yellow": 1e-4,
            "Red": 1e-4,
            "Red Edge": 1e-4,
            "NIR": 1e-4,
        }
    )
    .to_tif(image_dir / "test_preprocess.tif")
)
print("=" * 100)
print("=" * 100)
print(simage)

# index calculation test
print("=" * 100)
print("=" * 100)
equations = {
    "NDRI": "(B[8] - B[7]) / (B[8] + B[7])",
    "MSAVI": "0.5*(B[8]+1-sqrt( (2*B[8]+1)*(2*B[8]+1)-8*(B[8]-B[6]) ))",
    "OSAVI1": "(B[8] - B[6]) / (B[8] + B[6] + 0.16)",
    "OSAVI2": "(B[8] - B[6]) / (B[8] + B[6] + 0.18)",
    "CHL2": "(B[8] / B[4]) - 1",
    "WATER": "(B[8] - B[2]) / (B[8] + B[2])",
    "NDVI": "(B[8] - B[6]) / (B[8] + B[6])",
}
index_images = [img for img in simage.calculate_index(equations)]
for iimage in index_images:
    print(iimage)

# calculate band stats
print("=" * 100)
print("=" * 100)
stats = simage.calculate_band_stats()
pprint(stats)

# calculate index stats
print("=" * 100)
print("=" * 100)
index_stats = iimage.calculate_band_stats()
pprint(index_stats)

# render index
print("=" * 100)
print("=" * 100)
rgba_image = (
    iimage.render_index(vmin=0, vmax=1, cmap="viridis")
    .rescale(0.2)
    .to_png(image_dir / "test_render.png")
    .to_tif(image_dir / "test_render.tif")
)

# copy
print("=" * 100)
simage_copy = simage.copy()
print("Original:", simage)
print("Copy:", simage_copy)
print(f"Is same object: {simage is simage_copy}")
print(
    f"Is data equal: {np.array_equal(simage.get_image().data, simage_copy.get_image().data, equal_nan=True)}"
)
pprint(simage_copy.log[-2:])

# merge
print("=" * 100)
print("=" * 100)
merged = (
    SatImage().merge(index_images).to_tif(image_dir / "test_merge.tif")
)
print(merged)

# boundary
print("=" * 100)
print("=" * 100)
boundary = simage.get_boundary()
gpd.GeoDataFrame(geometry=[boundary]).to_file(
    image_dir / "test_boundary.geojson", driver="GeoJSON"
)

# # log
# print("="*100)
# print("="*100)
# pprint(merged.log)

