"""Render figures directly from the verified study windows."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from acquire import HERE


FIGURES = HERE / "reports" / "figures"
COLORS = ("#017c67", "#2457a7", "#b2502d")
INK = "#162831"
SOURCE = "Source: Copernicus Sentinel-2 L2A, ESA WorldCover 2021"
CROP_NOTE = "Grey areas have no eligible cropland observation."
INDEX_NAMES = ("NDVI", "NDMI", "SAVI")


def finish(fig, slug, caption, source=SOURCE):
    fig.text(0.025, 0.018, source + "  |  " + caption, color="#455b64", fontsize=8)
    fig.tight_layout(rect=(0, 0.065, 1, 0.94))
    fig.savefig(FIGURES / f"{slug}.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def map_panel(ax, image, title, bounds, cmap, vmin, vmax):
    west, south, east, north = bounds
    ax.set_facecolor("#e4e9ec")
    plot = ax.imshow(
        image, extent=(west / 1000, east / 1000, south / 1000, north / 1000),
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none",
    )
    ax.set(title=title, xlabel="UTM easting (km)", ylabel="UTM northing (km)")
    ax.ticklabel_format(useOffset=False)
    return plot


def render(table, observed, anomalies, reflectance, metrics):
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK,
        "axes.titleweight": "bold", "figure.facecolor": "#f8faf9", "axes.spines.top": False,
        "axes.spines.right": False, "grid.alpha": 0.24, "axes.axisbelow": True,
    })
    dates = table.observation.to_numpy(dtype="datetime64[D]")
    bounds = metrics["projected_bounds_m"]
    selected = int(table.ndmi_anomaly.idxmin())
    month = table.month[selected]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    fig.suptitle(f"Vegetation indices in the Tadla pilot area · {month}", fontsize=17, weight="bold")
    for i, ax in enumerate(axes):
        image = map_panel(ax, observed[selected, i], INDEX_NAMES[i], bounds, "viridis", -0.25, 0.9)
        fig.colorbar(image, ax=ax, shrink=0.68, label="Index (unitless)")
    finish(fig, "tadla-index-maps", CROP_NOTE)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.7), sharex=True, height_ratios=[2, 1])
    fig.suptitle("Observed cropland NDMI and antecedent rainfall", fontsize=17, weight="bold")
    for _, season in table.groupby(table.month.str[:4]):
        idx = season.index
        axes[0].fill_between(dates[idx], season.ndmi_q10, season.ndmi_q90, color=COLORS[1], alpha=0.18)
        axes[0].plot(dates[idx], season.ndmi, "o-", color=COLORS[1], linewidth=2)
    axes[0].set(ylabel="NDMI (unitless)")
    axes[0].plot([], [], "o-", color=COLORS[1], label="Cropland median")
    axes[0].fill_between([], [], [], color=COLORS[1], alpha=0.18, label="Spatial 10th–90th percentiles")
    axes[0].legend(loc="lower right")
    axes[1].bar(dates, table.antecedent_rain_mm, width=12, color=COLORS[0])
    axes[1].set(ylabel="Prior 30 days (mm)", xlabel="Satellite observation date")
    for ax in axes:
        ax.grid(axis="y")
    finish(fig, "ndmi-precipitation-series", "Lines connect observations within each season.", SOURCE + ", NASA POWER")

    same_month = table.index[table.month.str[-2:] == month[-2:]].tolist()
    fig, axes = plt.subplots(1, len(same_month), figsize=(15, 4.5), sharey=True)
    fig.suptitle("Same-month NDMI anomalies across years", fontsize=17, weight="bold")
    for i, ax in zip(same_month, axes, strict=True):
        image = map_panel(ax, anomalies[i, 1], table.month[i], bounds, "RdBu", -0.35, 0.35)
        fig.colorbar(image, ax=ax, orientation="horizontal", shrink=0.78, label="Δ NDMI")
    finish(fig, "ndmi-anomaly-maps", "Reference excludes the mapped year and requires three other years.")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    fig.suptitle("Spectral reflectance on clear cropland pixels", fontsize=17, weight="bold")
    normal = min(same_month, key=lambda j: abs(table.ndmi_anomaly[j]))
    bins = np.linspace(0, 0.8, 41)
    for band, (ax, name) in enumerate(zip(axes, ("Red (B04)", "NIR (B08)", "SWIR (B11)"), strict=True)):
        for j, color in ((normal, COLORS[0]), (selected, COLORS[2])):
            valid = np.isfinite(observed[j, 1])
            ax.hist(
                reflectance[j][band, valid], bins=bins, density=True, histtype="step",
                linewidth=2, label=table.month[j], color=color,
            )
        ax.set(title=name, xlabel="Surface reflectance (fraction)")
        ax.legend()
    axes[0].set_ylabel("Probability density")
    finish(fig, "spectral-distributions", "Two observations from the same calendar month.")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.suptitle("Seasonal profile of median cropland NDMI", fontsize=17, weight="bold")
    for year, season in table.groupby(table.month.str[:4]):
        if year == str(metrics["holdout_year"]):
            ax.plot(range(4), season.ndmi, "o-", color=COLORS[1], linewidth=3, label=f"{year} holdout")
        else:
            ax.plot(range(4), season.ndmi, "o-", linewidth=1.5, label=year, alpha=0.75)
    ax.set(xticks=range(4), xticklabels=["February", "March", "April", "May"], ylabel="NDMI (unitless)")
    ax.legend(ncol=3)
    ax.grid(axis="y")
    finish(fig, "seasonal-comparison", "Each point is one date, with month-dependent crop phenology.")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.suptitle(f"Reference error on the {metrics['holdout_year']} holdout", fontsize=17, weight="bold")
    records = metrics["heldout_reference_errors"]
    x = np.arange(len(records))
    ax.bar(x - 0.19, [row["pooled_median_mae"] for row in records], 0.38, label="Pooled median", color="#798e9c")
    ax.bar(x + 0.19, [row["monthly_median_mae"] for row in records], 0.38, label="Same-month median", color=COLORS[0])
    ax.set(xticks=x, xticklabels=INDEX_NAMES, ylabel="Mean absolute error (index units)")
    ax.grid(axis="y")
    ax.legend()
    finish(fig, "heldout-reference-benchmark", "Both references are scored on the same crop pixels and dates.")
