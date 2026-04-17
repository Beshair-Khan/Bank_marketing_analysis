import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os
import warnings
 
warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})
 
PALETTE = ["#1a3a5c", "#2e86ab", "#a23b72", "#f18f01", "#c73e1d",
           "#3b1f2b", "#44bba4", "#e94f37", "#393e41", "#f5c6a0"]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# 1. DATA LOADING & SYNTHETIC FALLBACK
 
def load_or_generate_data(path: str = "data/bank-full.csv") -> pd.DataFrame:
    """
    Try to load the real Kaggle CSV.
    If it is not found, generate a realistic synthetic dataset so the
    script always runs even before the CSV is downloaded.
    """
    data_path = os.path.join(os.path.dirname(__file__), path)
 
    if os.path.exists(data_path):
        print(f"[✓] Loading real dataset from: {data_path}")
        df = pd.read_csv(data_path, sep=";")
        return df
 
    print("[!] Dataset not found — generating realistic synthetic data (n=45,211).")
    print("    Download the real file from Kaggle to use actual data.\n")
 
    np.random.seed(42)
    n = 45_211
 
    jobs = ["admin.", "technician", "services", "management", "retired",
            "blue-collar", "unemployed", "entrepreneur", "housemaid",
            "self-employed", "unknown", "student"]
    job_weights = [0.11, 0.17, 0.09, 0.13, 0.07, 0.20, 0.04,
                   0.05, 0.03, 0.04, 0.02, 0.05]
 
    education = ["primary", "secondary", "tertiary", "unknown"]
    edu_weights = [0.15, 0.51, 0.31, 0.03]
 
    marital = ["married", "single", "divorced"]
    mar_weights = [0.60, 0.28, 0.12]
 
    months = ["jan","feb","mar","apr","may","jun",
              "jul","aug","sep","oct","nov","dec"]
    month_weights = [0.01,0.03,0.01,0.03,0.30,0.12,
                     0.17,0.17,0.04,0.07,0.03,0.02]
 
    contact = ["cellular", "telephone", "unknown"]
    contact_weights = [0.65, 0.15, 0.20]
 
    df = pd.DataFrame({
        "age":       np.random.normal(40, 10, n).clip(18, 95).astype(int),
        "job":       np.random.choice(jobs,      n, p=job_weights),
        "marital":   np.random.choice(marital,   n, p=mar_weights),
        "education": np.random.choice(education, n, p=edu_weights),
        "default":   np.random.choice(["yes","no"], n, p=[0.018, 0.982]),
        "balance":   np.random.exponential(1200, n).astype(int),
        "housing":   np.random.choice(["yes","no"], n, p=[0.56, 0.44]),
        "loan":      np.random.choice(["yes","no"], n, p=[0.16, 0.84]),
        "contact":   np.random.choice(contact,   n, p=contact_weights),
        "month":     np.random.choice(months,    n, p=month_weights),
        "duration":  np.random.exponential(260, n).clip(1).astype(int),
        "campaign":  np.random.poisson(2.5, n).clip(1, 50).astype(int),
        "pdays":     np.where(np.random.random(n) < 0.82, -1,
                              np.random.randint(1, 400, n)),
        "previous":  np.random.poisson(0.58, n).clip(0).astype(int),
        "poutcome":  np.random.choice(
                         ["unknown","failure","other","success"],
                         n, p=[0.82, 0.10, 0.04, 0.04]),
        "y":         np.random.choice(["yes","no"], n, p=[0.117, 0.883]),
    })
    return df
 
 
# 2.  DATA CLEANING & FEATURE ENGINEERING
 
def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("BANK MARKETING CAMPAIGN — DATA ANALYSIS")
    print("=" * 60)
    print(f"\n[1] Dataset shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"    Missing values : {df.isnull().sum().sum()}")
    print(f"    Duplicates     : {df.duplicated().sum()}")
 
    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
 
    # Binary encode target
    df["subscribed"] = (df["y"] == "yes").astype(int)
 
    # Age buckets
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 25, 35, 45, 55, 65, 100],
        labels=["18–25", "26–35", "36–45", "46–55", "56–65", "65+"]
    )
 
    # Balance buckets
    df["balance_tier"] = pd.cut(
        df["balance"],
        bins=[-999_999, 0, 500, 2000, 10_000, 999_999],
        labels=["Negative", "Low\n(0–500)", "Mid\n(500–2K)",
                "High\n(2K–10K)", "Premium\n(10K+)"]
    )
 
    # Month ordering
    month_order = ["jan","feb","mar","apr","may","jun",
                   "jul","aug","sep","oct","nov","dec"]
    df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)
 
    print(f"\n[2] Subscription rate : "
          f"{df['subscribed'].mean()*100:.1f}%  "
          f"({df['subscribed'].sum():,} / {len(df):,})")
    print(f"    Avg balance      : PKR-equivalent {df['balance'].mean():,.0f}")
    print(f"    Avg call duration: {df['duration'].mean():.0f}s "
          f"({df['duration'].mean()/60:.1f} min)")
    return df
 
 
# 3.  PLOT HELPERS

 
def save(fig: plt.Figure, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved → outputs/plots/{name}")
 

# 4.  VISUALISATION 1 — OVERVIEW DASHBOARD (2×2)

 
def plot_overview_dashboard(df: pd.DataFrame):
    print("\n[3] Generating Overview Dashboard …")
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#f8f9fa")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
 
    #  4a. Subscription pie 
    ax0 = fig.add_subplot(gs[0, 0])
    counts = df["y"].value_counts()
    wedge_colors = [PALETTE[0], PALETTE[3]]
    wedges, texts, autotexts = ax0.pie(
        counts,
        labels=["Not Subscribed", "Subscribed"],
        autopct="%1.1f%%",
        startangle=90,
        colors=wedge_colors,
        explode=(0, 0.07),
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(12); at.set_fontweight("bold"); at.set_color("white")
    ax0.set_title("Term Deposit Subscription Split", pad=12)
 
    #  4b. Age distribution by subscription 
    ax1 = fig.add_subplot(gs[0, 1])
    for label, color, lw in [("no", PALETTE[0], 1.5), ("yes", PALETTE[3], 2.2)]:
        subset = df[df["y"] == label]["age"]
        ax1.hist(subset, bins=30, alpha=0.6, color=color,
                 label=f"{'Subscribed' if label=='yes' else 'Not Subscribed'}",
                 edgecolor="white", linewidth=0.4)
    ax1.axvline(df["age"].mean(), color="#c73e1d", linestyle="--",
                linewidth=1.5, label=f"Mean age {df['age'].mean():.0f}")
    ax1.set_xlabel("Age"); ax1.set_ylabel("Count")
    ax1.set_title("Age Distribution by Subscription")
    ax1.legend(fontsize=9)
 
    # 4c. Campaign calls vs. subscription rate scatter 
    ax2 = fig.add_subplot(gs[1, 0])
    camp_rate = (df.groupby("campaign")["subscribed"]
                   .agg(["mean", "count"])
                   .reset_index()
                   .query("count >= 50 and campaign <= 20"))
    sc = ax2.scatter(
        camp_rate["campaign"], camp_rate["mean"] * 100,
        s=camp_rate["count"] / camp_rate["count"].max() * 500,
        c=camp_rate["mean"], cmap="RdYlGn", alpha=0.8, edgecolors="#333", linewidths=0.5
    )
    plt.colorbar(sc, ax=ax2, label="Subscription Rate")
    ax2.set_xlabel("Number of Campaign Calls")
    ax2.set_ylabel("Subscription Rate (%)")
    ax2.set_title("Campaign Calls vs. Subscription Rate\n(bubble size = volume)")
 
    #  4d. Balance tier subscription rate 
    ax3 = fig.add_subplot(gs[1, 1])
    bal_rate = df.groupby("balance_tier", observed=True)["subscribed"].mean() * 100
    bars = ax3.bar(bal_rate.index, bal_rate.values,
                   color=PALETTE[:len(bal_rate)], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, bal_rate.values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax3.set_xlabel("Account Balance Tier")
    ax3.set_ylabel("Subscription Rate (%)")
    ax3.set_title("Subscription Rate by Account Balance")
    ax3.set_ylim(0, bal_rate.max() * 1.25)
 
    fig.suptitle("Bank Marketing Campaign — Overview Dashboard",
                 fontsize=16, fontweight="bold", y=1.01, color="#1a3a5c")
    save(fig, "01_overview_dashboard.png")
 
 
# 5.  VISUALISATION 2 — JOB & EDUCATION ANALYSIS

 
def plot_job_education(df: pd.DataFrame):
    print("\n[4] Generating Job & Education Analysis …")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#f8f9fa")
 
    #  5a. Subscription rate by job 
    job_stats = (df.groupby("job")["subscribed"]
                   .agg(["mean", "count"])
                   .reset_index()
                   .rename(columns={"mean": "rate", "count": "n"}))
    job_stats["rate_pct"] = job_stats["rate"] * 100
    job_stats = job_stats.sort_values("rate_pct", ascending=True)
 
    colors = [PALETTE[3] if r > job_stats["rate_pct"].median()
              else PALETTE[0] for r in job_stats["rate_pct"]]
    bars = axes[0].barh(job_stats["job"], job_stats["rate_pct"],
                        color=colors, edgecolor="white", height=0.7)
    for bar, val, n in zip(bars, job_stats["rate_pct"], job_stats["n"]):
        axes[0].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%  (n={n:,})", va="center", fontsize=9)
    axes[0].set_xlabel("Subscription Rate (%)")
    axes[0].set_title("Subscription Rate by Job Category")
    axes[0].set_xlim(0, job_stats["rate_pct"].max() * 1.45)
    above = mpatches.Patch(color=PALETTE[3], label="Above median")
    below = mpatches.Patch(color=PALETTE[0], label="Below median")
    axes[0].legend(handles=[above, below], loc="lower right", fontsize=9)
 
    # 5b. Education × subscription stacked bar 
    edu_cross = (df.groupby(["education", "y"])
                   .size()
                   .unstack(fill_value=0)
                   .assign(total=lambda x: x.sum(axis=1))
                   .assign(pct_yes=lambda x: x["yes"] / x["total"] * 100))
    edu_cross = edu_cross.sort_values("pct_yes")
 
    x = np.arange(len(edu_cross))
    w = 0.5
    axes[1].bar(x, edu_cross["no"],  width=w, color=PALETTE[0], label="Not Subscribed", edgecolor="white")
    axes[1].bar(x, edu_cross["yes"], width=w, bottom=edu_cross["no"],
                color=PALETTE[3], label="Subscribed", edgecolor="white")
    for i, (_, row) in enumerate(edu_cross.iterrows()):
        axes[1].text(i, row["no"] + row["yes"] + 80,
                     f"{row['pct_yes']:.1f}%", ha="center",
                     fontsize=11, fontweight="bold", color=PALETTE[3])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(edu_cross.index, fontsize=11)
    axes[1].set_ylabel("Number of Clients")
    axes[1].set_title("Subscription by Education Level\n(% = subscription rate)")
    axes[1].legend()
 
    fig.suptitle("Job Category & Education Level Analysis",
                 fontsize=15, fontweight="bold", y=1.01, color="#1a3a5c")
    plt.tight_layout()
    save(fig, "02_job_education_analysis.png")
 
 
# 6.  VISUALISATION 3 — MONTHLY CAMPAIGN TRENDS
 
def plot_monthly_trends(df: pd.DataFrame):
    print("\n[5] Generating Monthly Campaign Trends …")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor("#f8f9fa")
 
    monthly = (df.groupby("month", observed=True)
                 .agg(
                     contacts=("y", "count"),
                     subscriptions=("subscribed", "sum"),
                     sub_rate=("subscribed", "mean"),
                     avg_duration=("duration", "mean"),
                 )
                 .reset_index())
    monthly["sub_rate_pct"] = monthly["sub_rate"] * 100
    months_str = monthly["month"].astype(str).str.capitalize()
 
    #  Top: Contact volume vs. subscriptions 
    x = np.arange(len(monthly))
    w = 0.38
    axes[0].bar(x - w/2, monthly["contacts"],     width=w,
                color=PALETTE[0], label="Total Contacts",    alpha=0.85, edgecolor="white")
    axes[0].bar(x + w/2, monthly["subscriptions"], width=w,
                color=PALETTE[3], label="Subscriptions",     alpha=0.85, edgecolor="white")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Monthly Contact Volume vs. Subscriptions")
    axes[0].legend()
    for i, (c, s) in enumerate(zip(monthly["contacts"], monthly["subscriptions"])):
        axes[0].text(i - w/2, c + 50, f"{c:,}", ha="center", fontsize=7, rotation=45)
        axes[0].text(i + w/2, s + 50, f"{s:,}", ha="center", fontsize=7, rotation=45)
 
    #  Bottom: Subscription rate line + avg duration 
    ax2 = axes[1]
    color1, color2 = PALETTE[2], PALETTE[4]
    ln1 = ax2.plot(x, monthly["sub_rate_pct"], "o-",
                   color=color1, linewidth=2.2, markersize=7, label="Sub Rate (%)")
    ax2.fill_between(x, monthly["sub_rate_pct"], alpha=0.12, color=color1)
    for xi, yi in zip(x, monthly["sub_rate_pct"]):
        ax2.annotate(f"{yi:.1f}%", (xi, yi),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color=color1, fontweight="bold")
    ax2.set_ylabel("Subscription Rate (%)", color=color1)
    ax2.tick_params(axis="y", labelcolor=color1)
 
    ax2b = ax2.twinx()
    ax2b.spines["right"].set_visible(True)
    ln2 = ax2b.plot(x, monthly["avg_duration"], "s--",
                    color=color2, linewidth=1.8, markersize=6, label="Avg Duration (s)")
    ax2b.set_ylabel("Avg Call Duration (seconds)", color=color2)
    ax2b.tick_params(axis="y", labelcolor=color2)
 
    ax2.set_xticks(x)
    ax2.set_xticklabels(months_str, fontsize=10)
    ax2.set_xlabel("Month")
    ax2.set_title("Monthly Subscription Rate & Average Call Duration")
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left")
 
    fig.suptitle("Monthly Campaign Performance Trends",
                 fontsize=15, fontweight="bold", y=1.01, color="#1a3a5c")
    plt.tight_layout()
    save(fig, "03_monthly_trends.png")
 
 
# 7.  VISUALISATION 4 — CUSTOMER SEGMENTATION HEATMAP + BALANCE BOX
 
def plot_segmentation(df: pd.DataFrame):
    print("\n[6] Generating Customer Segmentation Analysis …")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#f8f9fa")
 
    # 7a. Age group × job subscription rate heatmap
    top_jobs = df["job"].value_counts().head(8).index
    heat_df = (df[df["job"].isin(top_jobs)]
               .groupby(["age_group", "job"], observed=True)["subscribed"]
               .mean()
               .unstack(fill_value=0) * 100)
 
    import matplotlib.colors as mcolors
    cmap = matplotlib.cm.get_cmap("YlOrRd")
    im = axes[0].imshow(heat_df.values, cmap=cmap, aspect="auto", vmin=0, vmax=35)
    axes[0].set_xticks(range(len(heat_df.columns)))
    axes[0].set_xticklabels(heat_df.columns, rotation=35, ha="right", fontsize=9)
    axes[0].set_yticks(range(len(heat_df.index)))
    axes[0].set_yticklabels(heat_df.index, fontsize=10)
    plt.colorbar(im, ax=axes[0], label="Subscription Rate (%)")
    for i in range(len(heat_df.index)):
        for j in range(len(heat_df.columns)):
            val = heat_df.values[i, j]
            axes[0].text(j, i, f"{val:.0f}%",
                         ha="center", va="center",
                         fontsize=8,
                         color="white" if val > 18 else "black",
                         fontweight="bold")
    axes[0].set_title("Subscription Rate Heatmap\nAge Group × Job Category")
    axes[0].set_xlabel("Job Category")
    axes[0].set_ylabel("Age Group")
 
    # 7b. Balance distribution box plot by subscription 
    yes_data = df[df["y"] == "yes"]["balance"].clip(upper=10_000)
    no_data  = df[df["y"] == "no"]["balance"].clip(upper=10_000)
 
    bp = axes[1].boxplot(
        [no_data, yes_data],
        labels=["Not Subscribed", "Subscribed"],
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2.5},
        boxprops={"linewidth": 1.5},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.3},
        widths=0.45,
    )
    bp["boxes"][0].set_facecolor(PALETTE[0])
    bp["boxes"][1].set_facecolor(PALETTE[3])
 
    # Overlay mean markers
    for i, data in enumerate([no_data, yes_data], start=1):
        axes[1].scatter(i, data.mean(), color=PALETTE[4],
                        zorder=5, s=80, marker="D", label="Mean" if i == 1 else "")
    axes[1].set_ylabel("Account Balance (capped at 10,000)")
    axes[1].set_title("Account Balance Distribution\nby Subscription Status")
    axes[1].legend(["Mean"], loc="upper left")
 
    # Annotate medians
    for i, data in enumerate([no_data, yes_data], start=1):
        axes[1].text(i, data.median() + 50,
                     f"Median: {data.median():,.0f}",
                     ha="center", fontsize=9, fontweight="bold", color="white"
                     if i == 2 else "white")
 
    fig.suptitle("Customer Segmentation Analysis",
                 fontsize=15, fontweight="bold", y=1.01, color="#1a3a5c")
    plt.tight_layout()
    save(fig, "04_customer_segmentation.png")
 

# 8.  VISUALISATION 5 — CALL DURATION ANALYSIS + KPI SUMMARY CARD
 
def plot_duration_kpi(df: pd.DataFrame):
    print("\n[7] Generating Call Duration & KPI Summary …")
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#f8f9fa")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)
 
    # ── 8a. Duration decile subscription analysis ─────────────────────────────
    ax0 = fig.add_subplot(gs[0, :2])
    df["dur_decile"] = pd.qcut(df["duration"], q=10, labels=[f"D{i}" for i in range(1, 11)])
    dur_stats = (df.groupby("dur_decile", observed=True)["subscribed"]
                   .agg(["mean", "count"])
                   .reset_index())
    dur_stats["rate_pct"] = dur_stats["mean"] * 100
    bar_colors = [PALETTE[0] if r < 20 else PALETTE[3] if r > 40 else PALETTE[1]
                  for r in dur_stats["rate_pct"]]
    bars = ax0.bar(dur_stats["dur_decile"].astype(str), dur_stats["rate_pct"],
                   color=bar_colors, edgecolor="white", linewidth=1)
    for bar, val in zip(bars, dur_stats["rate_pct"]):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax0.set_xlabel("Call Duration Decile (D1 = shortest, D10 = longest)")
    ax0.set_ylabel("Subscription Rate (%)")
    ax0.set_title("Subscription Rate by Call Duration Decile\n"
                  "(Insight: longer calls correlate strongly with higher conversion)")
    ax0.axhline(df["subscribed"].mean() * 100, color=PALETTE[4],
                linestyle="--", linewidth=1.8, label=f"Overall avg {df['subscribed'].mean()*100:.1f}%")
    ax0.legend()
 
    # ── 8b. Contact type subscription rate
    ax1 = fig.add_subplot(gs[0, 2])
    contact_rate = (df.groupby("contact")["subscribed"].mean() * 100).sort_values()
    bars = ax1.bar(contact_rate.index, contact_rate.values,
                   color=PALETTE[:3], edgecolor="white")
    for bar, val in zip(bars, contact_rate.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", fontweight="bold", fontsize=10)
    ax1.set_ylabel("Subscription Rate (%)")
    ax1.set_title("Subscription Rate\nby Contact Type")
    ax1.set_ylim(0, contact_rate.max() * 1.3)
 
    # 8c. KPI Summary Cards
    kpi_data = [
        ("Total Clients",       f"{len(df):,}",                       PALETTE[0]),
        ("Subscriptions",       f"{df['subscribed'].sum():,}",         PALETTE[3]),
        ("Conversion Rate",     f"{df['subscribed'].mean()*100:.1f}%", PALETTE[2]),
        ("Avg Balance",         f"{df['balance'].mean():,.0f}",        PALETTE[4]),
        ("Avg Call Duration",   f"{df['duration'].mean():.0f}s",       PALETTE[1]),
        ("Avg Campaign Calls",  f"{df['campaign'].mean():.1f}",        PALETTE[5]),
    ]
    # Draw KPI cards as inset axes inside the bottom row
    kpi_fig, kpi_axes = plt.subplots(1, 6, figsize=(22, 2.2))
    kpi_fig.patch.set_facecolor("#f8f9fa")
    for idx, (title, value, color) in enumerate(kpi_data):
        ax = kpi_axes[idx]
        ax.set_facecolor(color)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.5, 0.65, value, transform=ax.transAxes,
                ha="center", va="center",
                fontsize=22, fontweight="bold", color="white")
        ax.text(0.5, 0.25, title, transform=ax.transAxes,
                ha="center", va="center",
                fontsize=10, color="white", alpha=0.9)
 
    kpi_fig.suptitle("Campaign KPI Summary Cards",
                     fontsize=13, fontweight="bold", y=1.08, color="#1a3a5c")
    kpi_fig.tight_layout()
    save(kpi_fig, "05b_kpi_cards.png")
 
    fig.suptitle("Call Duration Analysis & Contact Type",
                 fontsize=15, fontweight="bold", y=1.01, color="#1a3a5c")
    save(fig, "05_duration_analysis.png")
 
 
# =============================================================================
# 9.  NUMPY STATISTICAL SUMMARY
# =============================================================================
 
def print_numpy_stats(df: pd.DataFrame):
    print("\n[8] NumPy Statistical Summary")
    print("=" * 60)
    numeric_cols = ["age", "balance", "duration", "campaign"]
    for col in numeric_cols:
        arr = df[col].values
        print(f"\n  {col.upper()}")
        print(f"    Mean   : {np.mean(arr):>10.2f}")
        print(f"    Median : {np.median(arr):>10.2f}")
        print(f"    Std    : {np.std(arr):>10.2f}")
        print(f"    Min    : {np.min(arr):>10.2f}")
        print(f"    Max    : {np.max(arr):>10.2f}")
        print(f"    P25    : {np.percentile(arr, 25):>10.2f}")
        print(f"    P75    : {np.percentile(arr, 75):>10.2f}")
        print(f"    Skew   : {(np.mean(arr)-np.median(arr))/np.std(arr):>10.4f}")
 
    # Correlation matrix using NumPy
    sub = df[numeric_cols + ["subscribed"]].dropna()
    corr = np.corrcoef(sub.T)
    print("\n  PEARSON CORRELATION with SUBSCRIBED:")
    for i, col in enumerate(numeric_cols):
        print(f"    {col:<12}: {corr[i, -1]:+.4f}")
 
 

# 10.  MAIN

 
def main():
    df = load_or_generate_data()
    df = clean_and_engineer(df)
 
    plot_overview_dashboard(df)
    plot_job_education(df)
    plot_monthly_trends(df)
    plot_segmentation(df)
    plot_duration_kpi(df)
    print_numpy_stats(df)
 
    print("\n" + "=" * 60)
    print("✅  ANALYSIS COMPLETE!")
    print(f"    All plots saved to → outputs/plots/")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()