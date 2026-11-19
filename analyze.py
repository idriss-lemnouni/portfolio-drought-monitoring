"""Calculate cropland indices and assess a historical monthly reference."""

import json
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.coords import BoundingBox
from rasterio.transform import array_bounds
from rasterio.warp import transform_geom
from rasterio.windows import Window, bounds

from acquire import CRS, HERE, PIN, grid, save_raster, valid_reflectance


INDEX_NAMES = ("NDVI", "NDMI", "SAVI")
OUT = HERE / "runs" / "tadla"


def indices(layers, cropland):
    red, nir, swir, _ = layers
    valid = valid_reflectance(layers, cropland)
    result = []
    for numerator, denominator in (
        (nir - red, nir + red), (nir - swir, nir + swir),
        (1.5 * (nir - red), nir + red + 0.5),
    ):
        data = np.full(red.shape, np.nan, dtype=np.float32)
        np.divide(numerator, denominator, out=data, where=valid & (denominator > 0))
        result.append(data)
    return np.stack(result)


def historical_median(history, minimum):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        median = np.nanmedian(history, axis=0)
    return np.where(np.isfinite(history).sum(axis=0) >= minimum, median, np.nan)


def leave_year_out(values, months):
    years = np.array([int(month[:4]) for month in months])
    season = np.array([month[5:] for month in months])
    monthly, pooled = [], []
    for year, month in zip(years, season, strict=True):
        train = years != year
        monthly.append(historical_median(values[train & (season == month)], minimum=3))
        pooled.append(historical_median(values[train], minimum=9))
    return np.stack(monthly), np.stack(pooled)


def antecedent_rain(precipitation, through):
    days = [(through - timedelta(days=lag)).strftime("%Y%m%d") for lag in range(1, 31)]
    values = [precipitation[day] for day in days]
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError("NASA POWER contains invalid precipitation.")
    return float(sum(values))


def geojson_blocks(layers, geometry, month):
    height, width = layers.shape[-2:]
    features = []
    for row in range(0, height, 20):
        for col in range(0, width, 20):
            right, bottom = min(col + 20, width), min(row + 20, height)
            section = layers[:, row:bottom, col:right]
            support = np.isfinite(section[1])
            if not support.any():
                continue
            box = BoundingBox(*bounds(Window(col, row, right - col, bottom - row), geometry["transform"]))
            west, south, east, north = box
            polygon = {
                "type": "Polygon",
                "coordinates": [[(west, south), (east, south), (east, north), (west, north), (west, south)]],
            }
            properties = {"month": month, "ndmi_valid_pixels": int(support.sum())}
            for name, data in zip((*INDEX_NAMES, *(f"{name}_anomaly" for name in INDEX_NAMES)), section, strict=True):
                valid = np.isfinite(data)
                properties[name.lower()] = round(float(data[valid].mean()), 6) if valid.any() else None
            features.append({
                "type": "Feature", "properties": properties,
                "geometry": transform_geom(CRS, "EPSG:4326", polygon, precision=7),
            })
    return {"type": "FeatureCollection", "features": features}


def forecast_errors(values, monthly, pooled, months):
    report = []
    heldout_year = max(months)[:4]
    holdout = np.array([month.startswith(heldout_year) for month in months])
    for i, name in enumerate(INDEX_NAMES):
        observed, seasonal, constant = [array[holdout, i] for array in (values, monthly, pooled)]
        common = np.isfinite(observed) & np.isfinite(seasonal) & np.isfinite(constant)
        report.append({
            "index": name, "pixel_month_pairs": int(common.sum()),
            "monthly_median_mae": float(np.abs(observed[common] - seasonal[common]).mean()),
            "pooled_median_mae": float(np.abs(observed[common] - constant[common]).mean()),
        })
    return report


def run():
    manifest = json.loads(PIN.read_text(encoding="utf-8"))
    raw = HERE / "data" / "raw"
    precipitation = json.loads((raw / "precipitation.json").read_text(encoding="utf-8"))
    with rio.open(raw / "worldcover.tif") as source:
        cropland = source.read(1) == 40
    months = [record["month"] for record in manifest["scenes"]]
    reflectance = []
    geometry = grid()
    for month in months:
        with rio.open(raw / f"{month}.tif") as source:
            if (source.crs != rio.crs.CRS.from_string(CRS) or source.transform != geometry["transform"]
                    or source.shape != cropland.shape):
                raise ValueError(f"Grid mismatch in {month}.")
            reflectance.append(source.read())
    values = np.stack([indices(scene, cropland) for scene in reflectance])
    monthly, pooled = leave_year_out(values, months)
    anomalies = values - monthly
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    names = (*INDEX_NAMES, *(f"{name} anomaly" for name in INDEX_NAMES))
    for i, record in enumerate(manifest["scenes"]):
        month = record["month"]
        observation = date.fromisoformat(record["datetime"][:10])
        output = np.concatenate((values[i], anomalies[i]))
        save_raster(OUT / f"{month}.tif", output, geometry, names)
        (OUT / f"{month}.geojson").write_text(
            json.dumps(geojson_blocks(output, geometry, month), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        row = {
            "month": month, "observation": observation.isoformat(),
            "valid_cropland_fraction": float(np.isfinite(values[i, 1]).sum() / cropland.sum()),
            "antecedent_rain_mm": antecedent_rain(precipitation, observation),
            "anomaly_fraction": float(np.isfinite(anomalies[i, 1]).sum() / cropland.sum()),
        }
        for name, data in zip(names, output, strict=True):
            key = name.lower().replace(" ", "_")
            row[key] = float(np.nanmedian(data))
            if "anomaly" not in name:
                row[f"{key}_q10"], row[f"{key}_q90"] = np.nanquantile(data, (0.1, 0.9))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "observations.csv", index=False, float_format="%.7f", lineterminator="\n")
    seasons = table.month.str[-2:]
    rain_anomaly = [
        value - table.loc[(seasons == seasons[i]) & (table.index != i), "antecedent_rain_mm"].median()
        for i, value in table.antecedent_rain_mm.items()
    ]
    darkest = table.loc[table.ndmi_anomaly.idxmin()]
    report = {
        "scene_count": len(months), "cropland_pixels": int(cropland.sum()),
        "projected_bounds_m": array_bounds(geometry["height"], geometry["width"], geometry["transform"]),
        "min_valid_cropland_fraction": float(table.valid_cropland_fraction.min()),
        "min_anomaly_fraction": float(table.anomaly_fraction.min()),
        "holdout_year": int(max(months)[:4]),
        "heldout_reference_errors": forecast_errors(values, monthly, pooled, months),
        "lowest_ndmi_anomaly": {k: darkest[k] for k in ("month", "ndmi_anomaly", "antecedent_rain_mm")},
        "pooled_seasonal_rain_ndmi_spearman": float(pd.Series(rain_anomaly).corr(table.ndmi_anomaly, method="spearman")),
    }
    report_path = HERE / "reports" / "metrics.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from plot_results import render

    render(table, values, anomalies, reflectance, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run()
