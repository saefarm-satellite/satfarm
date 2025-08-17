import sys; sys.path.append("./")
from pathlib import Path
import satfarm
import numpy as np
from datetime import datetime

test_data_dir = Path("test_data")

anl0 = (
    satfarm.SatImage()
    .read_tif("test_data/20250105_035834_59_24e1_3B_AnalyticMS_SR_8b_clip_bandmath.tif")
    .set_band_alias(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13"])
    .extract_band(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"])
    .change_pixel_dtype("float32")
    .change_nodata(new_nodata=np.nan, old_nodata=0)
    .apply_scale_factor({"B1": 1e-4, "B2": 1e-4, "B3": 1e-4, "B4": 1e-4, "B5": 1e-4, "B6": 1e-4, "B7": 1e-4, "B8": 1e-4})
    .set_time(datetime(2025, 1, 5, 3, 8, 34))
)
anl1 = (
    satfarm.SatImage()
    .read_tif("test_data/20250208_040147_93_24ed_3B_AnalyticMS_SR_8b_clip_bandmath.tif")
    .set_band_alias(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13"])
    .extract_band(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"])
    .change_pixel_dtype("float32")
    .change_nodata(new_nodata=np.nan, old_nodata=0)
    .apply_scale_factor({"B1": 1e-4, "B2": 1e-4, "B3": 1e-4, "B4": 1e-4, "B5": 1e-4, "B6": 1e-4, "B7": 1e-4, "B8": 1e-4})
    .set_time(datetime(2025, 2, 8, 4, 1, 47))
)
vis0 = (
    satfarm.SatImage()
    .read_tif("test_data/20250105_035834_59_24e1_3B_Visual_clip.tif")
    .set_band_alias(["R", "G", "B", "A"])
    .change_nodata(new_nodata=0, old_nodata=0)
    .set_time(datetime(2025, 1, 5, 3, 8, 34))
)
vis1 = (
    satfarm.SatImage()
    .read_tif("test_data/20250208_040147_93_24ed_3B_Visual_clip.tif")
    .set_band_alias(["R", "G", "B", "A"])
    .change_nodata(new_nodata=0, old_nodata=0)
    .set_time(datetime(2025, 2, 8, 4, 1, 47))
)


dst_dates = [
    datetime(2025, 1, 10, 4, 0, 0),
    datetime(2025, 1, 15, 4, 0, 0),
    datetime(2025, 1, 20, 4, 0, 0),
    datetime(2025, 1, 25, 4, 0, 0),
]

# intp_simages = satfarm.interp_image([anl0, anl1], dst_dates)
# for si, simage in enumerate(intp_simages):
#     dst_date = dst_dates[si].strftime("%Y%m%d_%H%M%S")
#     save_path = f"test_data/{dst_date}_ii_intp_3B_AnalyticMS_SR_8b_clip_bandmath.tif"
#     simage.to_tif(save_path)

intp_simages = satfarm.interp_image([vis0, vis1], dst_dates)
for si, simage in enumerate(intp_simages):
    dst_date = simage.time.strftime("%Y%m%d_%H%M%S")
    save_path = f"test_data/{dst_date}_ii_intp_3B_Visual_clip.tif"
    simage.to_tif(save_path)





