"""Acquire small, co-registered Sentinel-2 windows and daily precipitation."""

import calendar
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio as rio
import requests
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


HERE = Path(__file__).resolve().parent
BOUNDS = (-6.705, 32.893, -6.684, 32.914)
YEARS = range(2021, 2026)
MONTHS = range(2, 6)
CRS = "EPSG:32629"
RESOLUTION = 20
BANDS = ("red", "nir", "swir16", "scl")
CLEAR = (4, 5)
MIN_COVERAGE = 0.85
FALLBACK_COVERAGE = 0.80
SEARCH = "https://earth-search.aws.element84.com/v1/search"
WORLDCOVER = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_N30W009_Map.tif"
)
POWER = "https://power.larc.nasa.gov/api/temporal/daily/point"
PIN = HERE / "data" / "source-manifest.json"


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def grid():
    west, south, east, north = transform_bounds("EPSG:4326", CRS, *BOUNDS)
    x0, y0 = np.floor(np.array([west, south]) / RESOLUTION).astype(int)
    x1, y1 = np.ceil(np.array([east, north]) / RESOLUTION).astype(int)
    return dict(
        crs=CRS,
        transform=from_origin(x0 * RESOLUTION, y1 * RESOLUTION, RESOLUTION, RESOLUTION),
        width=int(x1 - x0), height=int(y1 - y0),
    )


def read_layer(url, geometry, resampling, nodata):
    addresses = [url]
    if "sentinel-cogs.s3.us-west-2.amazonaws.com/" in url:
        addresses.append(url.replace(
            "sentinel-cogs.s3.us-west-2.amazonaws.com/",
            "s3.us-west-2.amazonaws.com/sentinel-cogs/",
        ))
    with rio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_CONNECTTIMEOUT="10", GDAL_HTTP_TIMEOUT="25", GDAL_HTTP_MAX_RETRY="2",
        GDAL_HTTP_RETRY_DELAY="2", GDAL_HTTP_MULTIRANGE="SERIAL", GDAL_CACHEMAX=64,
    ):
        for address in addresses:
            try:
                with rio.open(address) as source, WarpedVRT(
                    source, **geometry, resampling=resampling, dtype="float32",
                    src_nodata=0, nodata=nodata,
                ) as view:
                    return view.read(1)
            except RasterioIOError:
                if address == addresses[-1]:
                    raise


def get_json(session, url, params):
    response = session.get(url, params=params, timeout=(15, 75))
    response.raise_for_status()
    return response.json(), response.url


def candidates(session, year, month):
    days = calendar.monthrange(year, month)[1]
    start = f"{year}-{month:02d}-01T00:00:00Z"
    end = f"{year}-{month:02d}-{days}T23:59:59Z"
    found, _ = get_json(session, SEARCH, {
        "collections": "sentinel-2-l2a", "bbox": ",".join(map(str, BOUNDS)),
        "datetime": f"{start}/{end}", "limit": 100,
    })
    total = found.get("context", {}).get("matched", len(found["features"]))
    if total > len(found["features"]):
        raise ValueError("STAC results exceeded the monthly limit.")
    eligible = [
        item for item in found["features"]
        if all(name in item["assets"] for name in BANDS)
    ]
    return sorted(eligible, key=lambda item: (
        abs(int(item["properties"]["datetime"][8:10]) - 15),
        -float(item["properties"].get("s2:processing_baseline", 0)),
        item["properties"].get("eo:cloud_cover", 100), item["id"],
    ))


def scene_assets(item):
    result = {}
    for name in BANDS:
        asset = item["assets"][name]
        band = asset.get("raster:bands", [{}])[0]
        result[name] = {
            "url": asset["href"],
            "scale": band["scale"] if name != "scl" else 1,
            "offset": band.get("offset", 0) if name != "scl" else 0,
        }
    return result


def reflectance(assets, geometry, quality):
    layers = []
    for name in BANDS[:-1]:
        info = assets[name]
        values = read_layer(info["url"], geometry, Resampling.average, np.nan)
        layers.append(values * info["scale"] + info["offset"])
    return np.stack([*layers, quality])


def valid_reflectance(layers, cropland):
    red, nir, swir, scl = layers
    return (cropland & np.isin(scl, CLEAR) & np.isfinite(layers[:3]).all(axis=0)
            & (layers[:3] >= 0).all(axis=0) & (nir + red > 0) & (nir + swir > 0))


def save_raster(path, layers, geometry, names):
    staging = path.with_suffix(".tmp.tif")
    with rio.open(
        staging, "w", driver="GTiff", **geometry, count=len(layers), dtype="float32",
        nodata=np.nan, compress="deflate",
    ) as target:
        target.write(layers)
        target.descriptions = names
    staging.replace(path)


def crop_mask(raw, geometry):
    cover = raw / "worldcover.tif"
    if not cover.exists():
        values = read_layer(WORLDCOVER, geometry, Resampling.mode, np.nan)
        if np.isnan(values).any():
            raise ValueError("WorldCover does not completely cover the analysis grid.")
        save_raster(cover, values[None], geometry, ("WorldCover class",))
    with rio.open(cover) as source:
        values = source.read(1)
    cropland = values == 40
    if cropland.sum() < 100:
        raise ValueError("Insufficient mapped cropland for the fixed study area.")
    return cropland, sha256(cover)


def choose_scene(session, year, month, geometry, cropland):
    attempts = []
    best = None
    for item in candidates(session, year, month):
        assets = scene_assets(item)
        try:
            quality = read_layer(assets["scl"]["url"], geometry, Resampling.nearest, np.nan)
        except RasterioIOError:
            attempts.append({"id": item["id"], "rejected": "unreadable SCL window"})
            continue
        coverage = float(np.isin(quality[cropland], CLEAR).mean())
        attempts.append({"id": item["id"], "clear_cropland_fraction": round(coverage, 6)})
        if coverage < FALLBACK_COVERAGE:
            continue
        try:
            layers = reflectance(assets, geometry, quality)
        except RasterioIOError:
            attempts[-1]["rejected"] = "unreadable reflectance window"
            continue
        valid_fraction = float(valid_reflectance(layers, cropland).sum() / cropland.sum())
        attempts[-1]["usable_reflectance_fraction"] = round(valid_fraction, 6)
        record = {
            "id": item["id"], "datetime": item["properties"]["datetime"], "assets": assets,
            "selection": attempts,
            "usable_reflectance_fraction": valid_fraction,
        }
        if valid_fraction >= MIN_COVERAGE:
            return record, layers
        if valid_fraction >= FALLBACK_COVERAGE and (best is None or valid_fraction > best[0]["usable_reflectance_fraction"]):
            best = record, layers
    if best is not None:
        best[0]["coverage_fallback"] = True
        return best
    raise ValueError(f"No scene reaches the 80% fallback threshold for {year}-{month:02d}: {attempts}")


def acquire():
    raw = HERE / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    geometry = grid()
    manifest = json.loads(PIN.read_text(encoding="utf-8")) if PIN.exists() else None
    cropland, worldcover_hash = crop_mask(raw, geometry)
    mask_hash = hashlib.sha256(cropland.tobytes()).hexdigest()
    if manifest and manifest["worldcover"]["mask_sha256"] != mask_hash:
        raise ValueError("WorldCover crop mask differs from the pinned acquisition.")
    pinned = {row["month"]: row for row in manifest["scenes"]} if manifest else {}
    scenes = []
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=2, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
        )))
        for year in YEARS:
            for month in MONTHS:
                key = f"{year}-{month:02d}"
                path = raw / f"{key}.tif"
                record = pinned.get(key)
                checkpoint = raw / f"{key}.json"
                if record is None and path.exists() and checkpoint.exists():
                    record = json.loads(checkpoint.read_text(encoding="utf-8"))
                    with rio.open(path) as source:
                        usable = valid_reflectance(source.read(), cropland).sum() / cropland.sum()
                    threshold = FALLBACK_COVERAGE if record.get("coverage_fallback") else MIN_COVERAGE
                    if usable < threshold:
                        record = None
                if record is None:
                    record, layers = choose_scene(session, year, month, geometry, cropland)
                    save_raster(path, layers, geometry, BANDS)
                elif not path.exists():
                    quality = read_layer(
                        record["assets"]["scl"]["url"], geometry, Resampling.nearest, np.nan,
                    )
                    save_raster(path, reflectance(record["assets"], geometry, quality), geometry, BANDS)
                raster_hash = sha256(path)
                with rio.open(path) as source:
                    array_hash = hashlib.sha256(source.read().astype("<f4").tobytes()).hexdigest()
                if "array_sha256" in record and record["array_sha256"] != array_hash:
                    raise ValueError(f"Sentinel-2 window changed for {key}.")
                record = dict(record, month=key, sha256=raster_hash, array_sha256=array_hash)
                scenes.append(record)
                if not manifest:
                    checkpoint.write_text(json.dumps(record), encoding="utf-8")
                print(key, record["id"], flush=True)
        rain_path = raw / "precipitation.json"
        start, end = date(min(YEARS), min(MONTHS), 1) - timedelta(days=30), date(max(YEARS), max(MONTHS), 31)
        parameters = {
            "parameters": "PRECTOTCORR", "community": "AG", "format": "JSON",
            "longitude": round((BOUNDS[0] + BOUNDS[2]) / 2, 4),
            "latitude": round((BOUNDS[1] + BOUNDS[3]) / 2, 4),
            "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
        }
        if not rain_path.exists():
            result, _ = get_json(session, POWER, parameters)
            series = result["properties"]["parameter"]["PRECTOTCORR"]
            expected = (end - start).days + 1
            if len(series) != expected or any(value < 0 for value in series.values()):
                raise ValueError("Incomplete NASA POWER daily precipitation series.")
            rain_path.write_text(json.dumps(series, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        rain_hash = sha256(rain_path)
        if manifest and rain_hash != manifest["precipitation"]["sha256"]:
            raise ValueError("NASA POWER series changed from the pinned snapshot.")
        if not manifest:
            PIN.write_text(json.dumps({
                "bbox_wgs84": BOUNDS, "resolution_m": RESOLUTION, "crs": CRS,
                "preferred_coverage": MIN_COVERAGE, "fallback_coverage": FALLBACK_COVERAGE,
                "worldcover": {"url": WORLDCOVER, "sha256": worldcover_hash, "mask_sha256": mask_hash,
                               "crop_pixels": int(cropland.sum())},
                "precipitation": {"url": POWER, "parameters": parameters, "sha256": rain_hash},
                "scenes": scenes,
            }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    acquire()
