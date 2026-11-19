import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acquire import HERE, grid, save_raster
from analyze import antecedent_rain, geojson_blocks, indices, leave_year_out


def test_spectral_indices_respect_masks_and_denominators():
    raw = np.array([[[0.2, 0.2, 0, 0.2]], [[0.6, 0.6, 0, 0.6]],
                    [[0.3, 0.3, 0, 0.3]], [[4, 9, 4, 4]]], dtype=np.float32)
    mapped = indices(raw, np.array([[True, True, True, False]]))
    assert mapped[:, 0, 0] == pytest.approx([0.5, 1 / 3, 0.6 / 1.3])
    assert np.isnan(mapped[:, 0, [1, 3]]).all()
    assert np.isnan(mapped[:, 0, 2]).all()


def test_reference_excludes_target_year_and_requires_three_peers():
    values = np.array([[1, np.nan], [2, np.nan], [3, 7], [4, 8], [1000, 9]], dtype=np.float32)
    months = [f"{year}-03" for year in range(2021, 2026)]
    monthly, _ = leave_year_out(values, months)
    assert monthly[-1, 0] == 2.5
    assert np.isnan(monthly[-1, 1])
    assert monthly[0, 1] == 8


def test_rain_window_stops_before_overpass():
    records = {f"202401{day:02d}": 1 for day in range(1, 31)}
    records["20240131"] = 100
    assert antecedent_rain(records, date(2024, 1, 31)) == 30


def test_raster_and_geojson_round_trip(tmp_path):
    geometry = grid()
    shape = (6, geometry["height"], geometry["width"])
    data = np.full(shape, np.nan, dtype=np.float32)
    data[:3, 0, 0] = 0.25
    raster = tmp_path / "indices.tif"
    save_raster(raster, data, geometry, ("NDVI", "NDMI", "SAVI", "a", "b", "c"))
    with rio.open(raster) as view:
        assert view.count == 6
        assert view.crs.to_epsg() == 32629
        np.testing.assert_equal(view.read(), data)
    exported = json.loads(json.dumps(geojson_blocks(data, geometry, "2025-02"), allow_nan=False))
    assert len(exported["features"]) == 1
    feature = exported["features"][0]
    assert feature["properties"]["ndmi_valid_pixels"] == 1
    assert feature["properties"]["ndmi_anomaly"] is None
    lon, lat = feature["geometry"]["coordinates"][0][0]
    assert -7 < lon < -6 and 32 < lat < 33


def test_dashboard_controls_and_two_map_views():
    if not (HERE / "runs" / "tadla" / "observations.csv").exists():
        pytest.skip("Requires the acquired and analyzed dataset.")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(HERE / "dashboard.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 2
    app.selectbox[0].select("2025-03")
    app.selectbox[1].select("SAVI")
    app.selectbox[2].select("2024-03")
    app.radio[0].set_value("Same-month anomaly").run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 3
    assert len(app.get("download_button")) == 2
