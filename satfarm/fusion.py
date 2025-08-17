import rioxarray as rxr
import pandas as pd
import numpy as np
from satfarm import SatImage
from datetime import datetime

def interp_image(simages: list[SatImage], 
               time: datetime | list[datetime]
               ) -> list[SatImage]:
    # sort by time
    simages = sorted(simages, key=lambda simage: simage.time)
    # input check
    band_alias_list = [simage.image.band.values.tolist() for simage in simages]
    dtype_list = [simage.image.dtype for simage in simages]
    stime_list = [simage.time for simage in simages]
    if len(set([str(ba) for ba in band_alias_list])) != 1:
        raise ValueError("All images must have the same band alias")
    if len(set(dtype_list)) != 1:
        raise ValueError("All images must have the same data type")
    if None in stime_list:
        raise ValueError("All images must have time property")
    # input conversion
    if isinstance(time, datetime):
        time = [time]
    band_alias_list = band_alias_list[0]
    pixel_dtype = dtype_list[0]
    # calculate stats
    rows = []
    for simage in simages:
        row = dict()
        row["time"] = simage.time
        row["timestamp"] = simage.time.timestamp()
        stats = simage.calculate_band_stats()
        for band_alias in band_alias_list:
            rvalue = stats[band_alias]["mean"]
            row[band_alias] = rvalue
        rows.append(row)
    ref_df = pd.DataFrame(rows)
    # interpolate
    rows = []
    for t in time:
        row = dict()
        row["time"] = t
        row["timestamp"] = t.timestamp()
        for band_alias in band_alias_list:
            pred_rvalue = np.interp(x=row["timestamp"], xp=ref_df["timestamp"], fp=ref_df[band_alias])
            row[band_alias] = pred_rvalue
        rows.append(row)
    dst_df = pd.DataFrame(rows)
    # create new image
    base_image = simages[-1] # reference image by latest time
    base_row = ref_df.iloc[-1]
    intp_images = []
    for _, dst_row in dst_df.iterrows():
        intp_image = base_image.copy()
        intp_image.time = dst_row["time"]
        for band_alias in band_alias_list:
            value_arr = intp_image.image.sel(band=band_alias).data
            aoi = (value_arr != intp_image.image.rio.nodata)
            value_arr[aoi] = (
                value_arr[aoi]
                + dst_row[band_alias] 
                - base_row[band_alias]
            )
            value_arr = value_arr.astype(pixel_dtype)
            intp_image.image.sel(band=band_alias).data = value_arr
        
        intp_images.append(intp_image)
    # return
    return intp_images
