"""Create exploratory drought and dam-impact charts from existing Ayakchi data."""

from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parent
GEE = ROOT / "gee_environmental_data"
HYDRO = ROOT / "hydromorphology_outputs"
OUT = ROOT / "drought_dam_outputs"

CHIRPS = GEE / "timeseries" / "chirps_daily_basin_mean_2021_2025.csv"
ERA5 = GEE / "timeseries" / "era5_land_daily_basin_mean_2021_2025.csv"
CLIM = GEE / "rasters" / "06_chirps_monthly_climatology_1991_2020.tif"
METRICS = HYDRO / "basin_metrics.csv"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
RESERVOIR_CAPACITY_MCM = 18.0  # From supplied layer name: reservoir_option_5_18MCM


def basin_area():
    rows = dict(csv.reader(METRICS.open(encoding="utf-8")))
    return float(rows["area_km2"])


def monthly_normal():
    with rasterio.open(CLIM) as src:
        return np.array([float(src.read(i, masked=True).mean()) for i in range(1, 13)])


def wilson_interval(successes, total, z=1.96):
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    margin = z*np.sqrt((p*(1-p) + z*z/(4*total))/total) / denominator
    return max(0, center-margin), min(1, center+margin)


def save_annual_precipitation(monthly, normal, area):
    annual = monthly.groupby("year")["precipitation"].sum()
    normal_annual = normal.sum()
    colors = np.where(annual >= normal_annual, "#2c7fb8", "#d95f0e")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(annual.index.astype(str), annual.values, color=colors)
    ax.axhline(normal_annual, color="black", ls="--", lw=1.5,
               label=f"1991–2020 normal: {normal_annual:.0f} mm")
    for i, value in enumerate(annual):
        ax.text(i, value+8, f"{value:.0f}", ha="center", fontsize=8)
    ax.set(ylabel="Annual precipitation (mm)", title="Ayakchi Basin — Recent Rainfall vs Climate Normal")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "chart_01_annual_precipitation_vs_normal.png", dpi=300, facecolor="white")
    plt.close(fig)
    return annual, normal_annual


def save_drought_probability(monthly, normal):
    ratios = monthly.pivot(index="year", columns="month", values="precipitation").divide(normal, axis=1)
    moderate = (ratios < .8).sum(axis=0)
    severe = (ratios < .6).sum(axis=0)
    n = ratios.shape[0]
    probability = moderate / n
    intervals = np.array([wilson_interval(int(value), n) for value in moderate])
    lower = probability.values - intervals[:, 0]
    upper = intervals[:, 1] - probability.values

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(12)
    ax.bar(x, probability*100, color="#fdae61", label="Below 80% of normal")
    ax.bar(x, severe/n*100, color="#d73027", label="Below 60% of normal")
    ax.errorbar(x, probability*100, yerr=np.vstack([lower, upper])*100, fmt="none",
                color="black", capsize=3, lw=1, label="95% interval (n=5)")
    ax.set_xticks(x, MONTHS); ax.set_ylim(0, 110)
    ax.set(ylabel="Observed recent-year frequency (%)",
           title="Ayakchi Basin — Monthly Meteorological-Drought Frequency, 2021–2025")
    ax.legend(ncol=3, fontsize=8); ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "chart_02_monthly_drought_probability.png", dpi=300, facecolor="white")
    plt.close(fig)
    return ratios, moderate/n, severe/n, intervals


def save_anomaly_heatmap(ratios):
    anomaly = (ratios - 1) * 100
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    image = ax.imshow(anomaly.values, aspect="auto", cmap="BrBG", vmin=-100, vmax=100)
    ax.set_xticks(np.arange(12), MONTHS); ax.set_yticks(np.arange(len(anomaly)), anomaly.index)
    ax.set_title("Ayakchi Basin — Monthly Rainfall Anomaly from 1991–2020 Normal")
    cb = fig.colorbar(image, ax=ax, shrink=.8); cb.set_label("Anomaly (%)")
    for r in range(anomaly.shape[0]):
        for c in range(12):
            value = anomaly.iloc[r, c]
            ax.text(c, r, f"{value:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(value) > 55 else "black")
    fig.savefig(OUT / "chart_03_monthly_rainfall_anomaly_heatmap.png", dpi=300, facecolor="white")
    plt.close(fig)


def annual_extremes(chirps):
    records = []
    for year, group in chirps.groupby(chirps.index.year):
        p = group["precipitation"]
        records.append({"year": year, "annual_precipitation_mm": p.sum(),
                        "max_1day_mm": p.max(), "max_3day_mm": p.rolling(3).sum().max(),
                        "max_5day_mm": p.rolling(5).sum().max(),
                        "longest_dry_spell_days": longest_run((p < 1).to_numpy())})
    return pd.DataFrame(records).set_index("year")


def longest_run(values):
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def save_extremes(extremes, area):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    x = np.arange(len(extremes)); width = .25
    for offset, field, label, color in [(-width, "max_1day_mm", "1-day", "#91bfdb"),
                                         (0, "max_3day_mm", "3-day", "#4575b4"),
                                         (width, "max_5day_mm", "5-day", "#313695")]:
        axes[0].bar(x+offset, extremes[field], width, label=label, color=color)
    axes[0].set_xticks(x, extremes.index); axes[0].set_ylabel("Maximum accumulated rainfall (mm)")
    axes[0].set_title("Short-Duration Rainfall Load"); axes[0].legend(); axes[0].grid(axis="y", alpha=.2)
    volume = extremes["max_3day_mm"] * area * .001
    axes[1].bar(extremes.index.astype(str), volume, color="#2b8cbe")
    axes[1].set_ylabel("Basin rainfall volume (million m³)")
    axes[1].set_title("Maximum 3-Day Catchment Water Input")
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("Ayakchi Dam — Rainfall-Extreme Screening, 2021–2025", fontweight="bold")
    fig.savefig(OUT / "chart_04_dam_extreme_rainfall_load.png", dpi=300, facecolor="white")
    plt.close(fig)


def save_runoff_volume(era, annual_precip, area):
    annual = era.resample("YE").agg({"surface_runoff_sum": "sum", "snow_cover": "mean",
                                     "volumetric_soil_water_layer_1": "mean"})
    annual.index = annual.index.year
    annual["runoff_volume_mcm"] = annual["surface_runoff_sum"] * area * .001
    annual["rainfall_volume_mcm"] = annual_precip * area * .001
    annual["runoff_ratio"] = annual["surface_runoff_sum"] / annual_precip

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = np.arange(len(annual)); width = .36
    ax.bar(x-width/2, annual["rainfall_volume_mcm"], width, color="#74add1", label="CHIRPS rainfall volume")
    ax.bar(x+width/2, annual["runoff_volume_mcm"], width, color="#0571b0", label="ERA5 surface-runoff proxy")
    ax.axhline(RESERVOIR_CAPACITY_MCM, color="#7a0177", ls="--", lw=1.5,
               label="Option-5 nominal capacity context (18 MCM)")
    ax.set_xticks(x, annual.index); ax.set_ylabel("Annual water volume (million m³)")
    ax.set_title("Ayakchi Dam — Catchment Water-Input and Surface-Runoff Proxies")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "chart_05_dam_annual_water_volume.png", dpi=300, facecolor="white")
    plt.close(fig)
    return annual


def save_seasonal_dam_chart(chirps, era):
    p = chirps.groupby(chirps.index.month)["precipitation"].mean()
    runoff = era.groupby(era.index.month)["surface_runoff_sum"].mean()
    snow = era.groupby(era.index.month)["snow_cover"].mean()
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(12)
    ax.bar(x, p, color="#74add1", label="Daily precipitation")
    ax.plot(x, runoff, color="#045a8d", marker="o", lw=2, label="Daily surface runoff")
    ax.set_xticks(x, MONTHS); ax.set_ylabel("Mean daily water depth (mm/day)")
    ax2 = ax.twinx(); ax2.plot(x, snow, color="#756bb1", marker="s", lw=2, label="Snow cover")
    ax2.set_ylabel("Mean snow cover (%)")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper right")
    ax.set_title("Ayakchi Dam — Seasonal Rainfall, Runoff and Snow-Cover Signals")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "chart_06_dam_seasonal_hydroclimate.png", dpi=300, facecolor="white")
    plt.close(fig)


def save_soil_moisture(era):
    sm = era["volumetric_soil_water_layer_1"]
    smooth = sm.rolling(30, center=True, min_periods=15).mean()
    monthly_threshold = sm.groupby(sm.index.month).quantile(.2)
    threshold = pd.Series(sm.index.month, index=sm.index).map(monthly_threshold)
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.plot(smooth.index, smooth, color="#238b45", lw=1.5, label="30-day mean soil moisture")
    ax.plot(threshold.index, threshold, color="#d95f0e", lw=1, alpha=.8,
            label="Monthly 20th-percentile threshold")
    ax.fill_between(smooth.index, smooth, threshold, where=smooth < threshold,
                    color="#d95f0e", alpha=.3, label="Low soil-moisture episodes")
    ax.set(ylabel="Volumetric soil water, 0–7 cm (m³/m³)",
           title="Ayakchi Basin — Surface Soil-Moisture Drought Proxy")
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(OUT / "chart_07_soil_moisture_drought_timeline.png", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    area = basin_area(); normal = monthly_normal()
    chirps = pd.read_csv(CHIRPS, parse_dates=["date"]).set_index("date")
    era = pd.read_csv(ERA5, parse_dates=["date"]).set_index("date")
    monthly = chirps.resample("MS")["precipitation"].sum().to_frame()
    monthly["year"], monthly["month"] = monthly.index.year, monthly.index.month

    annual_precip, normal_annual = save_annual_precipitation(monthly, normal, area)
    ratios, probability, severe_probability, intervals = save_drought_probability(monthly, normal)
    save_anomaly_heatmap(ratios)
    extremes = annual_extremes(chirps); save_extremes(extremes, area)
    annual_dam = save_runoff_volume(era, annual_precip, area)
    save_seasonal_dam_chart(chirps, era)
    save_soil_moisture(era)

    extremes.to_csv(OUT / "annual_rainfall_extremes.csv")
    annual_dam.to_csv(OUT / "annual_dam_water_proxies.csv", index_label="year")
    probability_table = pd.DataFrame({
        "month": np.arange(1, 13), "month_name": MONTHS,
        "normal_precipitation_mm": normal, "probability_below_80pct_normal": probability.values,
        "probability_below_60pct_normal": severe_probability.values,
        "moderate_probability_ci95_low": intervals[:, 0],
        "moderate_probability_ci95_high": intervals[:, 1],
    })
    probability_table.to_csv(OUT / "monthly_drought_probability.csv", index=False)

    driest_year = int((annual_precip / normal_annual).idxmin())
    wettest_year = int((annual_precip / normal_annual).idxmax())
    max_event_year = int(extremes["max_3day_mm"].idxmax())
    summary = {
        "basin_area_km2": area, "climate_normal_annual_precipitation_mm": normal_annual,
        "recent_mean_annual_precipitation_mm": float(annual_precip.mean()),
        "driest_recent_year": driest_year,
        "driest_year_percent_of_normal": float(annual_precip[driest_year]/normal_annual*100),
        "wettest_recent_year": wettest_year,
        "wettest_year_percent_of_normal": float(annual_precip[wettest_year]/normal_annual*100),
        "largest_3day_rainfall_year": max_event_year,
        "largest_3day_rainfall_mm": float(extremes.loc[max_event_year, "max_3day_mm"]),
        "largest_3day_basin_water_input_mcm": float(extremes.loc[max_event_year, "max_3day_mm"]*area*.001),
        "mean_annual_era5_surface_runoff_volume_mcm": float(annual_dam["runoff_volume_mcm"].mean()),
        "option_5_nominal_capacity_mcm_from_layer_name": RESERVOIR_CAPACITY_MCM,
        "largest_3day_gross_rainfall_as_percent_of_nominal_capacity":
            float(extremes.loc[max_event_year, "max_3day_mm"]*area*.001/RESERVOIR_CAPACITY_MCM*100),
        "lowest_recent_runoff_proxy_year": int(annual_dam["runoff_volume_mcm"].idxmin()),
        "lowest_recent_runoff_proxy_mcm": float(annual_dam["runoff_volume_mcm"].min()),
        "limitations": ["Only five recent years are available for empirical probability estimates.",
                        "Rainfall volume is not reservoir inflow; losses, routing and storage are not modeled.",
                        "ERA5-Land is coarse reanalysis and surface runoff is a screening proxy."]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Ayakchi drought and dam-impact charts\n\n"
        "Exploratory analysis based on CHIRPS and ERA5-Land daily records for 2021–2025, the CHIRPS "
        "1991–2020 monthly normal, and the 89.09 km² basin area. Monthly drought probability is the "
        "observed fraction of five recent years below 80% of normal (severe: below 60%); Wilson 95% "
        "intervals show the substantial small-sample uncertainty.\n\n"
        "Catchment rainfall volume and ERA5 surface runoff are dam-inflow pressure proxies, not routed "
        "reservoir inflow or design-flood estimates. They exclude infiltration, evapotranspiration, groundwater, "
        "abstraction, channel routing and reservoir operations.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
