"""
generate_report.py
Generates a multi-page PDF assessment report using matplotlib.
Run AFTER solution.py file has produced all outputs..
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
EDA_DIR  = BASE / "eda_plots"
OUT_PDF  = BASE / "report.pdf"

# Reload metrics
metrics_df = pd.read_csv(BASE / "model_metrics.csv")

# -----------------------------------------------------------------------
def page_title_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#064A56")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.5, 0.68, "Freight Rate Prediction", ha="center", va="center",
            fontsize=36, fontweight="bold", color="white",
            transform=ax.transAxes)
    ax.text(0.5, 0.58, "ML Engineering Assessment — SpotterLabs", ha="center",
            va="center", fontsize=18, color="#A8D5DC", transform=ax.transAxes)
    ax.text(0.5, 0.42, "Muhammad Moazzam", ha="center", va="center",
            fontsize=14, color="white", transform=ax.transAxes)
    ax.text(0.5, 0.36,
            "Model: LightGBM + XGBoost Ensemble  |  Hold-out RMSE: $639  |  MAPE: 6.1%",
            ha="center", va="center", fontsize=12, color="#A8D5DC",
            transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_data_overview(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("1. Data Overview", fontsize=18, fontweight="bold", x=0.05, ha="left", y=0.97)

    # Rate distribution
    train_df = pd.read_csv(BASE / "train-test.csv")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35,
                           left=0.07, right=0.97, top=0.88, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(train_df["posted_rate"], bins=80, color="#064A56", edgecolor="white", linewidth=0.3)
    ax1.set_title("Rate distribution (raw)", fontsize=11)
    ax1.set_xlabel("Rate ($)")
    ax1.set_ylabel("Count")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(np.log1p(train_df["posted_rate"]), bins=80, color="#1A7D8E",
             edgecolor="white", linewidth=0.3)
    ax2.set_title("log1p(Rate) distribution", fontsize=11)
    ax2.set_xlabel("log1p(Rate)")
    ax2.spines[["top", "right"]].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 0])
    med_by_equip = train_df.groupby("equipment")["posted_rate"].median().sort_values()
    bars = ax3.barh(med_by_equip.index, med_by_equip.values, color=["#064A56", "#1A7D8E", "#2DA8BC"])
    ax3.set_title("Median rate by equipment", fontsize=11)
    ax3.set_xlabel("Median Rate ($)")
    ax3.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, med_by_equip.values):
        ax3.text(val + 20, bar.get_y() + bar.get_height() / 2,
                 f"${val:,.0f}", va="center", fontsize=9)

    ax4 = fig.add_subplot(gs[1, 1])
    train_df["month"] = pd.to_datetime(train_df["date"]).dt.month
    monthly = train_df.groupby("month")["posted_rate"].median()
    ax4.plot(monthly.index, monthly.values, marker="o", color="#064A56", linewidth=2)
    ax4.set_title("Median rate by month (train)", fontsize=11)
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Rate ($)")
    ax4.spines[["top", "right"]].set_visible(False)

    # Data summary text
    fig.text(0.07, 0.93,
             "Train: 48,000 rows | Jan-Oct 2025 | 14 features | "
             "Rates $57-$25,533 | Missing: weight(0.6%), market_index(0.8%)",
             fontsize=9.5, color="#455A60")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_split_and_features(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("2. Data Split Strategy & Feature Engineering",
                 fontsize=18, fontweight="bold", x=0.05, ha="left", y=0.97)

    # Feature importance
    feat_imp_path = EDA_DIR / "feature_importance.png"
    fi_img = plt.imread(str(feat_imp_path))

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4,
                           left=0.05, right=0.97, top=0.87, bottom=0.05)

    ax_fi = fig.add_subplot(gs[0, 0])
    ax_fi.imshow(fi_img)
    ax_fi.set_axis_off()
    ax_fi.set_title("LightGBM Feature Importance (gain)", fontsize=11)

    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.set_axis_off()

    split_info = (
        "Time-Based Split Rationale\n"
        "──────────────────────────\n\n"
        "• Training set  : Jan–Sep 2025 (43,147 rows)\n"
        "• Hold-out set  : Oct 2025     ( 4,853 rows)\n"
        "• Final test    : Nov–Dec 2025 (12,000 rows)\n\n"
        "Using a temporal split (not random) prevents\n"
        "data leakage and simulates real deployment:\n"
        "the model only ever sees past data when\n"
        "predicting future loads.\n\n"
        "Features Engineered\n"
        "───────────────────\n\n"
        "Date features\n"
        "  day_of_week, month, week_of_year,\n"
        "  day_of_month, is_weekend, quarter\n\n"
        "Load features\n"
        "  weight_per_mile = weight / distance\n"
        "  haversine_dist (from lat/lon)\n"
        "  dist_diff = stated - haversine\n\n"
        "City encoding\n"
        "  Smoothed target-mean for pickup / delivery\n"
        "  Lane-level mean rate (pickup+delivery pair)\n\n"
        "Pass-through\n"
        "  market_index, quote_signal  <- top signals\n"
        "  distance, weight, equipment_enc"
    )
    ax_text.text(0.02, 0.97, split_info, transform=ax_text.transAxes,
                 fontsize=9.5, va="top", fontfamily="monospace", color="#1A3A40",
                 linespacing=1.55)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_model_results(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("3. Model Training & Validation Results",
                 fontsize=18, fontweight="bold", x=0.05, ha="left", y=0.97)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.88, bottom=0.07)

    # Metrics bar chart
    ax_bar = fig.add_subplot(gs[0, 0])
    models = metrics_df["Model"]
    rmse   = metrics_df["RMSE ($)"]
    colors = ["#064A56", "#1A7D8E", "#2DA8BC"]
    bars = ax_bar.bar(models, rmse, color=colors, width=0.5)
    ax_bar.set_title("Hold-out RMSE by model", fontsize=11)
    ax_bar.set_ylabel("RMSE ($)")
    for bar, val in zip(bars, rmse):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 5,
                    f"${val:,.0f}", ha="center", va="bottom", fontsize=10)
    ax_bar.spines[["top", "right"]].set_visible(False)

    # MAPE bar chart
    ax_mape = fig.add_subplot(gs[0, 1])
    mape = metrics_df["MAPE (%)"]
    bars2 = ax_mape.bar(models, mape, color=colors, width=0.5)
    ax_mape.set_title("Hold-out MAPE by model", fontsize=11)
    ax_mape.set_ylabel("MAPE (%)")
    for bar, val in zip(bars2, mape):
        ax_mape.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                     f"{val:.2f}%", ha="center", va="bottom", fontsize=10)
    ax_mape.spines[["top", "right"]].set_visible(False)

    # Actual vs predicted
    avp_path = EDA_DIR / "actual_vs_predicted.png"
    avp_img = plt.imread(str(avp_path))
    ax_avp = fig.add_subplot(gs[1, :])
    ax_avp.imshow(avp_img)
    ax_avp.set_axis_off()
    ax_avp.set_title("Hold-out: Actual vs Predicted (Ensemble)", fontsize=11)

    # Model description
    fig.text(0.07, 0.93,
             "Primary: LightGBM (93 trees, lr=0.05, leaves=127)  |  "
             "Secondary: XGBoost (78 trees, lr=0.05, depth=7)  |  "
             "Ensemble: inverse-RMSE weighted average  |  "
             "Target: log1p(rate)",
             fontsize=9.5, color="#455A60")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_december_chart(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("4. December 2025 Predicted Load Rate Chart",
                 fontsize=18, fontweight="bold", x=0.05, ha="left", y=0.97)

    dec_img = plt.imread(str(BASE / "scorer_results" / "candidate_december.png"))
    ax = fig.add_axes([0.04, 0.12, 0.92, 0.75])
    ax.imshow(dec_img)
    ax.set_axis_off()

    fig.text(0.07, 0.10,
             "Fixed lane: Lexington -> Fort Wayne | 360 mi | Dry Van | 32,000 lb | "
             "market_index and quote_signal set to training medians for Dec 2025 inference.",
             fontsize=9.5, color="#455A60")
    fig.text(0.07, 0.06,
             "Observation: Rates are stable ~$840 through Dec 25, then dip to ~$805 "
             "Dec 28-31 — consistent with reduced freight demand during the holiday week.",
             fontsize=9.5, color="#455A60", style="italic")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with PdfPages(str(OUT_PDF)) as pdf:
        page_title_page(pdf)
        page_data_overview(pdf)
        page_split_and_features(pdf)
        page_model_results(pdf)
        page_december_chart(pdf)
    print(f"Report saved -> {OUT_PDF}")


if __name__ == "__main__":
    main()
