"""Interactive inspection of the observed indices and seasonal anomalies."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rasterio as rio
import streamlit as st

from analyze import INDEX_NAMES, OUT


st.set_page_config(page_title="Tadla cropland monitor", layout="wide")
st.title("Tadla cropland monitor")
st.caption("Morocco · Sentinel-2 L2A · 20 m grid · February–May, 2021–2025")
if not (OUT / "observations.csv").exists():
    st.error("Run acquire.py followed by analyze.py to create the study outputs.")
    st.stop()

table = pd.read_csv(OUT / "observations.csv")
controls = st.columns([1.2, 1, 1.3, 1.2])
month = controls[0].selectbox("Observation month", list(table.month), index=len(table) - 1, key="month")
index = controls[1].selectbox("Vegetation index", INDEX_NAMES, index=1, key="index")
view = controls[2].radio("Map value", ["Observed", "Same-month anomaly"], horizontal=True, key="view")
comparison = controls[3].selectbox("Compare with", ["None", *table.month], key="comparison")
if comparison == month:
    st.info("Choose a different comparison month to see two dates side by side.")
    comparison = "None"
band = INDEX_NAMES.index(index) + 1 + (3 if view == "Same-month anomaly" else 0)
selected = table.set_index("month").loc[month]
summary = st.columns(4)
summary[0].metric(f"Median {index}", f"{selected[index.lower()]:.3f}")
summary[1].metric(f"Median Δ {index}", f"{selected[index.lower() + '_anomaly']:+.3f}")
summary[2].metric("Valid cropland", f"{selected.valid_cropland_fraction:.1%}")
summary[3].metric("Prior 30-day rainfall", f"{selected.antecedent_rain_mm:.1f} mm")


def map_figure(key):
    with rio.open(OUT / f"{key}.tif") as source:
        pixels = source.read(band)
        x = source.bounds.left + source.res[0] * (np.arange(source.width) + 0.5)
        y = source.bounds.top - source.res[1] * (np.arange(source.height) + 0.5)
    anomaly = band > 3
    plot = go.Figure(go.Heatmap(
        x=x / 1000, y=y / 1000, z=pixels,
        colorscale="RdBu" if anomaly else "Viridis", zmin=-0.35 if anomaly else -0.25,
        zmax=0.35 if anomaly else 0.9, hoverongaps=False,
        colorbar=dict(title=f"Δ {index}" if anomaly else index),
        hovertemplate="E %{x:.3f} km<br>N %{y:.3f} km<br>Value %{z:.3f}<extra></extra>",
    ))
    plot.update_layout(
        title=key, xaxis_title="UTM easting (km)", yaxis_title="UTM northing (km)",
        yaxis=dict(scaleanchor="x", scaleratio=1), height=505, margin=dict(l=20, r=10, t=45, b=15),
    )
    return plot


panels = st.columns(2) if comparison != "None" else [st.container()]
panels[0].plotly_chart(map_figure(month), key="selected")
if comparison != "None":
    panels[1].plotly_chart(map_figure(comparison), key="comparison-map")
st.caption(
    "Missing pixels are outside mapped cropland, fail the quality mask, or have too few historical observations. "
    "The colour scale stays fixed between dates."
)
series = go.Figure()
for year, panel in table.groupby(table.month.str[:4]):
    series.add_scatter(
        x=panel.observation, y=panel[index.lower()], mode="lines+markers", name=year,
    )
series.update_layout(xaxis_title="Observation date", yaxis_title=f"Median {index} (unitless)", height=340)
st.plotly_chart(series, key="series")
dl1, dl2 = st.columns(2)
for container, extension, label, mime in (
    (dl1, "tif", "Download six-band GeoTIFF", "image/tiff"),
    (dl2, "geojson", "Download block GeoJSON", "application/geo+json"),
):
    path = OUT / f"{month}.{extension}"
    container.download_button(label, path.read_bytes(), path.name, mime)
st.caption(
    "Spectral anomalies can reflect phenology, harvest timing, irrigation, or other stresses. "
    "NASA POWER precipitation is a coarse regional indicator. Field measurements are needed for water-stress diagnosis."
)
