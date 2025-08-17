import rioxarray as rxr
from satfarm import SatImage
from datetime import datetime

def fill_image(simages: list[SatImage], 
               time: datetime | list[datetime]
               ) -> list[SatImage]:
    # extract bands
    
    for simage in simages:
        

    bands = simages[0].get_band_alias()
    # calculate rvalue of each bands
    rvalues = []
    for simage in simages:
        rvalues.append(simage.get_rvalue())
    
    
    
    # interpolate each bands
    # return new SatImage
    
    