import sys; sys.path.append("./")
from pathlib import Path
from satfarm import SatImage, ops
import numpy as np
from datetime import datetime

import pytest


class TestFusionIntegration:
    """Integration tests for ops.interp_image using real test data if available."""

    @pytest.fixture
    def test_data_dir(self):
        return Path("test_data")

    @pytest.fixture
    def analytic_images(self, test_data_dir):
        """Prepare two analytic SatImage instances or skip if files are missing."""
        f0 = test_data_dir / "20250105_035834_59_24e1_3B_AnalyticMS_SR_8b_clip_bandmath.tif"
        f1 = test_data_dir / "20250208_040147_93_24ed_3B_AnalyticMS_SR_8b_clip_bandmath.tif"
        if not test_data_dir.exists() or not (f0.exists() and f1.exists()):
            pytest.skip("Analytic test images not available")

        bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
        scale = {k: 1e-4 for k in bands}

        anl0 = (
            SatImage()
            .read_tif(str(f0))
            .set_band_alias(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13"])
            .extract_band(bands)
            .change_pixel_dtype("float32")
            .change_nodata(new_nodata=np.nan, old_nodata=0)
            .apply_scale_factor(scale)
            .set_alias("analytic0")
            .set_time(datetime(2025, 1, 5, 3, 8, 34))
        )
        anl1 = (
            SatImage()
            .read_tif(str(f1))
            .set_alias("analytic1")
            .set_band_alias(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13"])
            .extract_band(bands)
            .change_pixel_dtype("float32")
            .change_nodata(new_nodata=np.nan, old_nodata=0)
            .apply_scale_factor(scale)
            .set_time(datetime(2025, 2, 8, 4, 1, 47))
        )
        return anl0, anl1

    @pytest.fixture
    def visual_images(self, test_data_dir):
        """Prepare two visual SatImage instances or skip if files are missing."""
        f0 = test_data_dir / "20250105_035834_59_24e1_3B_Visual_clip.tif"
        f1 = test_data_dir / "20250208_040147_93_24ed_3B_Visual_clip.tif"
        if not test_data_dir.exists() or not (f0.exists() and f1.exists()):
            pytest.skip("Visual test images not available")

        vis0 = (
            SatImage()
            .read_tif(str(f0))
            .set_alias("visual0")
            .set_band_alias(["R", "G", "B", "A"])
            .change_nodata(new_nodata=0, old_nodata=0)
            .set_time(datetime(2025, 1, 5, 3, 8, 34))
        )
        vis1 = (
            SatImage()
            .read_tif(str(f1))
            .set_alias("visual1")
            .set_band_alias(["R", "G", "B", "A"])
            .change_nodata(new_nodata=0, old_nodata=0)
            .set_time(datetime(2025, 2, 8, 4, 1, 47))
        )
        return vis0, vis1

    @pytest.mark.skipif(
        not Path("test_data").exists(),
        reason="Test data directory not available",
    )
    def test_interp_image_analytic_basic(self, analytic_images):
        anl0, anl1 = analytic_images

        dst_dates = [
            datetime(2025, 1, 10, 4, 0, 0),
            datetime(2025, 1, 15, 4, 0, 0),
            datetime(2025, 1, 20, 4, 0, 0),
            datetime(2025, 1, 25, 4, 0, 0),
        ]

        outputs = ops.interp_image([anl0, anl1], dst_dates)
        assert len(outputs) == len(dst_dates)

        for i, simage in enumerate(outputs):
            assert isinstance(simage, SatImage)
            assert simage.time == dst_dates[i]
            assert simage.image.shape == anl1.image.shape
            assert simage.image.dtype == anl1.image.dtype
            assert np.array_equal(simage.image.band.values, anl1.image.band.values)

    @pytest.mark.skipif(
        not Path("test_data").exists(),
        reason="Test data directory not available",
    )
    def test_interp_image_visual_basic(self, visual_images):
        vis0, vis1 = visual_images

        dst_dates = [
            datetime(2025, 1, 10, 4, 0, 0),
            datetime(2025, 1, 15, 4, 0, 0),
        ]

        outputs = ops.interp_image([vis0, vis1], dst_dates)
        assert len(outputs) == len(dst_dates)

        for i, simage in enumerate(outputs):
            assert isinstance(simage, SatImage)
            assert simage.time == dst_dates[i]
            assert simage.image.shape == vis1.image.shape
            assert simage.image.dtype == vis1.image.dtype
            assert np.array_equal(simage.image.band.values, vis1.image.band.values)

if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__])
