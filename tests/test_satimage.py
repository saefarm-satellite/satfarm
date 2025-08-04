"""
Basic tests for the SatImage class.

This module contains unit tests for the core functionality of the SatImage class.
"""

import pytest
import numpy as np
from pathlib import Path
from satfarm import SatImage


class TestSatImage:
    """Test cases for SatImage class."""
    
    def test_init(self):
        """Test SatImage initialization."""
        simage = SatImage()
        assert simage.is_empty()
        assert simage.log == []
    
    def test_copy_empty(self):
        """Test copying an empty SatImage."""
        simage = SatImage()
        simage_copy = simage.copy()
        
        assert simage_copy.is_empty()
        assert simage_copy is not simage
        assert simage_copy.log == simage.log
    
    def test_version_accessible(self):
        """Test that version is accessible."""
        from satfarm import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0
    
    def test_band_alias_operations(self):
        """Test band alias setting and resetting."""
        simage = SatImage()
        
        # Test setting band aliases on empty image (should not raise error)
        simage.set_band_alias(['red', 'green', 'blue'])
        
        # Test resetting band aliases
        simage.reset_band_alias()


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
    def test_basic_workflow(self, test_data_dir):
        """Test basic image processing workflow if test data is available."""
        test_file = test_data_dir / "20250105_035834_59_24e1_3B_AnalyticMS_SR_8b_clip_bandmath.tif"
        
        if not test_file.exists():
            pytest.skip("Test image file not available")
        
        # Basic workflow test
        simage = (SatImage()
                 .read_tif(str(test_file))
                 .change_pixel_dtype("float32"))
        
        assert not simage.is_empty()
        assert len(simage.log) > 0
        
        # Test copy with data
        simage_copy = simage.copy()
        assert not simage_copy.is_empty()
        assert simage_copy is not simage


if __name__ == "__main__":
    pytest.main([__file__])