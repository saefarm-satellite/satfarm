"""
Basic tests for the SatImage class.

This module contains unit tests for the core functionality of the SatImage class.
"""

import pytest
import numpy as np
from pathlib import Path
from satfarm import SatImage
from datetime import datetime

vis_image_path = "test_data/20250105_035834_59_24e1_3B_Visual_clip.tif"
anl_image_path = "test_data/20250105_035834_59_24e1_3B_AnalyticMS_SR_8b_clip_bandmath.tif"


class TestSatImage:
    """Test cases for SatImage class."""
    
    def test_init(self):
        """Test SatImage initialization."""
        simage = SatImage()
        assert simage.is_empty()
    
    def test_copy(self):
        """Test copying an empty SatImage."""
        simage = SatImage().read_tif(vis_image_path)
        simage_copy = simage.copy()
        simage_copy.log.pop()
        
        assert simage_copy is not simage
        assert simage_copy.log == simage.log
    
    def test_version_accessible(self):
        """Test that version is accessible."""
        from satfarm import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0
    
    def test_band_alias_operations(self):
        """Test band alias setting and resetting."""
        simage = SatImage().read_tif(vis_image_path)
        
        # Test setting band aliases on empty image (should not raise error)
        simage.set_band_alias(['red', 'green', 'blue', 'alpha'])
        
        # Test resetting band aliases
        simage.reset_band_alias()


class TestSatImageOps:
    """Test cases for SatImage operations."""

    def test_ops(self):

        sf = {
            "B1": 1e-4, 
        }
        anl = (
            SatImage()
            .read_tif(anl_image_path)
            .reset_band_alias()
            .extract_band(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"])
            .change_pixel_dtype("float32")
            .change_nodata(new_nodata=np.nan, old_nodata=0)
            .apply_scale_factor({"B1": 1e-4, "B2": 1e-4, "B3": 1e-4, "B4": 1e-4, "B5": 1e-4, "B6": 1e-4, "B7": 1e-4, "B8": 1e-4})
            .set_alias("analytic_preprocessed")
            .set_time(datetime(2025, 1, 5, 3, 8, 34))
        )
        vis = (
            SatImage()
            .read_tif(vis_image_path)
            .reset_band_alias()
            .set_band_alias(["R", "G", "B", "A"])
            .change_pixel_dtype("float32")
            .change_nodata(new_nodata=np.nan, old_nodata=0)
            .set_alias("visual_preprocessed")
            .set_time(datetime(2025, 1, 5, 3, 8, 34))
        )
        anl_stat = anl.calculate_band_stats()
        vis_stat = vis.calculate_band_stats()
        anl_index = (
            anl
            .calculate_index({"TEST": "(B[8] - B[6]) / (B[8] + B[6])"})
            .__next__()
            .render_index(vmin=0.2, vmax=0.5, cmap="viridis")
        )
        anl_index.to_png("test_data/test_render.png")

class TestSatImageIntegration:
    """Integration tests requiring test data."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Fixture providing test data directory path."""
        return Path("test_data")
    
    @pytest.mark.skipif(
        not Path("test_data").exists(),
        reason="Test data directory not available"
    )
    def test_basic_workflow(self):
        """Test basic image processing workflow if test data is available."""
        # Basic workflow test
        simage = (
            SatImage()
            .read_tif(anl_image_path)
            .change_pixel_dtype("float32")
        )
        
        assert not simage.is_empty()
        assert len(simage.log) > 0
        
        # Test copy with data
        simage_copy = simage.copy()
        assert not simage_copy.is_empty()
        assert simage_copy is not simage


if __name__ == "__main__":
    pytest.main([__file__])