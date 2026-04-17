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
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

PALETTE = ["#1a3a5c", "#2e86ab", "#a23b72", "#f18f01", "#c73e1d",
           "#3b1f2b", "#44bba4", "#e94f37", "#393e41", "#f5c6a0"]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. LOAD DATA


def load_data(path="bank-additional-full.csv"):
    data_path = os.path.join(os.path.dirname(__file__), path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"[✓] Loading dataset from: {data_path}")
    return pd.read_csv(data_path, sep=";")


# 2. CLEAN + FEATURE ENGINEERING


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("BANK MARKETING CAMPAIGN — DATA ANALYSIS")
    print("=" * 60)

    print(f"\n[1] Shape : {df.shape}")
    print(f"Missing values : {df.isnull().sum().sum()}")
    print(f"Duplicates     : {df.duplicated().sum()}")

    df = df.drop_duplicates().reset_index(drop=True)

    # Target
    df["subscribed"] = (df["y"] == "yes").astype(int)

    # Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 25, 35, 45, 55, 65, 100],
        labels=["18–25", "26–35", "36–45", "46–55", "56–65", "65+"]
    )

    # Campaign intensity
    df["campaign_level"] = pd.cut(
        df["campaign"],
        bins=[0, 1, 3, 5, 10, 100],
        labels=["1 call", "2–3", "4–5", "6–10", "10+"]
    )

    # Economic indicator (VERY IMPORTANT for this dataset)
    df["euribor_level"] = pd.cut(
        df["euribor3m"],
        bins=[0, 1, 2, 3, 5, 10],
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )

    # Month ordering
    month_order = ["mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)

    print(f"\nSubscription rate: {df['subscribed'].mean()*100:.1f}%")
    print(f"Avg duration: {df['duration'].mean():.0f}s")
    print(f"Avg euribor: {df['euribor3m'].mean():.2f}")

    return df

# 3. SAVE FUNCTION


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {name}")

# 4. OVERVIEW DASHBOARD


def plot_overview_dashboard(df):
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    df["y"].value_counts().plot.pie(
        ax=ax1,
        autopct="%1.1f%%",
        colors=[PALETTE[0], PALETTE[3]],
        startangle=90
    )
    ax1.set_title("Subscription Split")

    # Age distribution
    ax2 = fig.add_subplot(gs[0, 1])
    for label, color in [("no", PALETTE[0]), ("yes", PALETTE[3])]:
        ax2.hist(df[df["y"] == label]["age"], bins=30, alpha=0.5, color=color)

    ax2.set_title("Age Distribution")

    # Campaign effect
    ax3 = fig.add_subplot(gs[1, 0])
    camp = df.groupby("campaign")["subscribed"].mean() * 100
    ax3.scatter(camp.index, camp.values, color=PALETTE[2])
    ax3.set_title("Campaign vs Conversion")

    # Euribor effect (REPLACED BALANCE)
    ax4 = fig.add_subplot(gs[1, 1])
    euribor = df.groupby("euribor_level", observed=True)["subscribed"].mean() * 100
    ax4.bar(euribor.index, euribor.values, color=PALETTE[:len(euribor)])
    ax4.set_title("Euribor vs Subscription")

    save(fig, "01_overview.png")


# 5. JOB + EDUCATION


def plot_job_education(df):
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    job = df.groupby("job")["subscribed"].mean().sort_values() * 100
    ax[0].barh(job.index, job.values, color=PALETTE[3])
    ax[0].set_title("Job vs Subscription")

    edu = df.groupby(["education", "y"]).size().unstack()
    edu.plot(kind="bar", stacked=True, ax=ax[1],
             color=[PALETTE[0], PALETTE[3]])
    ax[1].set_title("Education vs Subscription")

    save(fig, "02_job_education.png")



# 6. MONTHLY TRENDS


def plot_monthly_trends(df):
    fig, ax = plt.subplots(figsize=(12, 5))

    monthly = df.groupby("month", observed=True)["subscribed"].mean() * 100
    ax.plot(monthly.index.astype(str), monthly.values, marker="o")

    ax.set_title("Monthly Conversion Trend")

    save(fig, "03_monthly.png")


# 7. SEGMENTATION


def plot_segmentation(df):
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    heat = df.pivot_table(index="age_group", columns="job",
                          values="subscribed", aggfunc="mean") * 100

    ax[0].imshow(heat.fillna(0))
    ax[0].set_title("Heatmap")

    ax[1].boxplot([
        df[df["y"] == "no"]["duration"],
        df[df["y"] == "yes"]["duration"]
    ])
    ax[1].set_title("Duration vs Subscription")

    save(fig, "04_segmentation.png")


# 8. KPI


def plot_kpi(df):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(["Yes", "No"],
           [df[df["y"] == "yes"]["duration"].mean(),
            df[df["y"] == "no"]["duration"].mean()],
           color=[PALETTE[3], PALETTE[0]])

    ax.set_title("Avg Call Duration")

    save(fig, "05_kpi.png")



# 9. NUMPY STATS


def print_stats(df):
    print("\nNUMPY STATS")
    for col in ["age", "duration", "campaign"]:
        arr = df[col].values
        print(f"{col}: mean={np.mean(arr):.2f}, median={np.median(arr):.2f}")



# MAIN


def main():
    df = load_data()
    df = clean_and_engineer(df)

    plot_overview_dashboard(df)
    plot_job_education(df)
    plot_monthly_trends(df)
    plot_segmentation(df)
    plot_kpi(df)

    print_stats(df)

    print("\nDONE ✔")


if __name__ == "__main__":
    main()
