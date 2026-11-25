"""
UNEMPLOYMENT IN INDIA | COVID-19 IMPACT ANALYSIS
A reproducible, human-centred data science workflow.

Run:
    pip install pandas numpy matplotlib seaborn jupyter
    python unemployment_analysis.py

Expected input:
    ./Unemployment in India.csv
    ./Unemployment_Rate_upto_11_2020.csv
Outputs:
    ./outputs/*.png and ./outputs/cleaned_*.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
NAVY, TEAL, CORAL, GOLD = "#14213D", "#2A9D8F", "#E76F51", "#E9C46A"
plt.rcParams.update({
    "figure.facecolor": "#F7F9FC", "axes.facecolor": "#F7F9FC",
    "axes.edgecolor": "#D9E1EA", "axes.labelcolor": NAVY,
    "text.color": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
    "axes.titleweight": "bold", "font.family": "DejaVu Sans"
})

RATE = "Estimated Unemployment Rate (%)"
EMP = "Estimated Employed"
PART = "Estimated Labour Participation Rate (%)"

def load_and_clean(path, source_name):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in ["Region", "Frequency", "Area", "Region.1"]:
        if col in df:
            df[col] = df[col].astype("string").str.strip()
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(),
                                dayfirst=True, errors="coerce")
    for col in [RATE, EMP, PART]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Region", "Date", RATE, EMP, PART]).copy()
    df = df.drop_duplicates().copy()
    df["Source"] = source_name
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp("M")
    df["Estimated Unemployed (derived)"] = (
        df[EMP] * df[RATE] / (100 - df[RATE]).replace(0, np.nan)
    )
    df["Estimated Labour Force (derived)"] = (
        df[EMP] + df["Estimated Unemployed (derived)"]
    )
    df["Unemployment Rate Check"] = df[RATE].between(0, 100)
    print(f"{source_name}: {before:,} raw rows -> {len(df):,} valid rows "
          f"({before-len(df):,} removed for missing/duplicate records)")
    return df

def weighted_monthly_rate(df):
    """Aggregate rates using implied unemployed and employed counts.

    This is an estimate: unemployed = employed * rate / (100-rate).
    It is more defensible than taking an unweighted mean of state/area rates,
    but still inherits the source data's estimation and coverage limitations.
    """
    monthly = df.groupby("Month").agg(
        employed=(EMP, "sum"),
        unemployed=("Estimated Unemployed (derived)", "sum"),
        labour_force=("Estimated Labour Force (derived)", "sum")
    )
    monthly["unemployment_rate"] = 100 * monthly["unemployed"] / monthly["labour_force"]
    return monthly.reset_index()

def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# 1. Load: keep the two files separate because their geographic grain differs.
area = load_and_clean(BASE / "Unemployment in India.csv", "Area-level dataset")
state = load_and_clean(BASE / "Unemployment_Rate_upto_11_2020.csv", "State-level dataset")

# Basic quality checks
for label, df in [("Area-level", area), ("State-level", state)]:
    print(f"\n{label} | period: {df.Date.min():%b %Y} to {df.Date.max():%b %Y}")
    print("Regions:", df.Region.nunique(), "| missing values after cleaning:", int(df.isna().sum().sum()))
    print("Rate outside 0–100:", int((~df["Unemployment Rate Check"]).sum()))

# Export cleaned data
area.to_csv(OUT / "cleaned_area_level.csv", index=False)
state.to_csv(OUT / "cleaned_state_level.csv", index=False)

# 2. National monthly estimates from each source (never concatenate the two).
area_monthly = weighted_monthly_rate(area)
state_monthly = weighted_monthly_rate(state)

# 3. COVID-era comparison: pre-COVID through Feb 2020; COVID period from Mar 2020.
area["Period"] = np.where(area.Date < "2020-03-01", "Pre-COVID (to Feb 2020)",
                          "COVID period (Mar 2020 onward)")
period_summary = area.groupby("Period").agg(
    mean_rate=(RATE, "mean"),
    median_rate=(RATE, "median"),
    observations=(RATE, "size")
)
print("\nArea-level period summary (unweighted row-level rates):\n", period_summary.round(2))
area_national = area_monthly.set_index("Month")
pre_mean = area_national.loc[: "2020-02-29", "unemployment_rate"].mean()
covid_mean = area_national.loc["2020-03-31": "2020-06-30", "unemployment_rate"].mean()
print(f"\nWeighted monthly estimate: pre-COVID mean={pre_mean:.2f}%; "
      f"Mar-Jun 2020 mean={covid_mean:.2f}%; difference={covid_mean-pre_mean:+.2f} pp")

# 4. Chart: first source, with clear intervention marker.
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(area_monthly.Month, area_monthly.unemployment_rate, color=TEAL,
        marker="o", linewidth=3, markersize=7, label="Area-level weighted estimate")
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"),
           color=CORAL, alpha=.10, label="COVID-19 period (Mar 2020 onward)")
ax.axvline(pd.Timestamp("2020-03-01"), color=CORAL, linestyle="--", linewidth=1.8)
peak = area_monthly.loc[area_monthly.unemployment_rate.idxmax()]
ax.annotate(f"Peak: {peak.unemployment_rate:.1f}%\n{peak.Month:%b %Y}",
            (peak.Month, peak.unemployment_rate), xytext=(-75, -5),
            textcoords="offset points", fontweight="bold", color=CORAL,
            arrowprops=dict(arrowstyle="->", color=CORAL))
ax.set(title="The COVID shock: unemployment rose sharply in April 2020",
       ylabel="Estimated unemployment rate (%)", xlabel="")
ax.legend(frameon=False, loc="upper left")
save_fig(fig, "01_covid_shock_area_level.png")

# 5. Independent state-level series validates the shock and shows recovery.
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(state_monthly.Month, state_monthly.unemployment_rate, color=NAVY,
        marker="o", linewidth=3, markersize=7, label="State-level weighted estimate")
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-10-31"),
           color=CORAL, alpha=.08, label="COVID period")
ax.annotate("April 2020 peak", xy=(pd.Timestamp("2020-04-30"),
            state_monthly.loc[state_monthly.Month.eq(pd.Timestamp("2020-04-30")),
                              "unemployment_rate"].iloc[0]),
            xytext=(30, 20), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=CORAL), color=CORAL, fontweight="bold")
ax.set(title="A second dataset shows the shock—and a gradual return",
       ylabel="Estimated unemployment rate (%)", xlabel="")
ax.legend(frameon=False)
save_fig(fig, "02_state_level_recovery.png")

# 6. Rural vs urban: compare average observed rates across the available window.
area_mix = area.groupby("Area")[RATE].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 4.8))
bars = ax.barh(area_mix.index, area_mix.values,
               color=[TEAL if x == "Rural" else CORAL for x in area_mix.index])
for bar, value in zip(bars, area_mix.values):
    ax.text(value + .15, bar.get_y()+bar.get_height()/2, f"{value:.1f}%",
            va="center", fontweight="bold")
ax.set(title="Urban observations carry a higher average unemployment rate",
       xlabel="Mean estimated unemployment rate (%)", ylabel="")
ax.set_xlim(0, max(area_mix.values)*1.18)
save_fig(fig, "03_rural_urban_comparison.png")

# 7. State-level April 2020 hotspots.
april = state[state.Month.eq(pd.Timestamp("2020-04-30"))].sort_values(RATE, ascending=False).head(12)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=april, x=RATE, y="Region", color=CORAL, ax=ax)
ax.set(title="April 2020: unemployment pressure varied sharply by state",
       xlabel="Estimated unemployment rate (%)", ylabel="")
save_fig(fig, "04_april_2020_state_hotspots.png")

# 8. Regional change: compare each state's mean before and from March 2020 onward.
state_period = state.assign(
    Period=np.where(state.Date < "2020-03-01", "Pre-COVID", "COVID period")
).groupby(["Region", "Period"])[RATE].mean().unstack()
if {"Pre-COVID", "COVID period"}.issubset(state_period.columns):
    state_period["change_pp"] = state_period["COVID period"] - state_period["Pre-COVID"]
    movers = state_period.sort_values("change_pp", ascending=False).head(12).sort_values("change_pp")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(movers.index, movers.change_pp, color=GOLD)
    ax.set(title="Where did unemployment rise most? Change in state averages",
           xlabel="Change in estimated unemployment rate (percentage points)", ylabel="")
    save_fig(fig, "05_state_level_change.png")

# 9. Labour participation: a second lens on labour-market stress.
participation = area.groupby("Month")[PART].mean().reset_index()
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.plot(participation.Month, participation[PART], color=GOLD,
        marker="o", linewidth=2.8, markersize=6)
ax.axvline(pd.Timestamp("2020-03-01"), color=CORAL, linestyle="--")
ax.set(title="Labour-force participation also weakened around the shock",
       ylabel="Mean labour-force participation rate (%)", xlabel="")
save_fig(fig, "06_labour_participation.png")

# 10. Compact findings for reproducibility.
print("\nKEY FINDINGS")
print(f"Area-level weighted estimate: pre-COVID average {pre_mean:.2f}% vs "
      f"Mar-Jun average {covid_mean:.2f}% (+{covid_mean-pre_mean:.2f} percentage points).")
print("Area-level peak:", peak.Month.strftime("%B %Y"), f"({peak.unemployment_rate:.2f}%).")
print("State-level April peak:", float(state_monthly.loc[
    state_monthly.Month.eq(pd.Timestamp("2020-04-30")), "unemployment_rate"].iloc[0]), "%")
print("Average area-level rate by area:\n", area_mix.round(2).to_string())
print("\nInterpretation note: the two files have different geographic granularity and coverage. "
      "They are analyzed separately, not stacked or averaged together.")
print("Seasonality note: the area-level file spans only 14 months, so recurring annual "
      "seasonality cannot be established reliably.")
