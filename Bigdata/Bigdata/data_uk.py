import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# === CONFIG ===
INPUT_FILE = "uk.csv"
OUTPUT_DIR = "uk25"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
CLEANED_FILE = os.path.join(OUTPUT_DIR, "cleaned_data.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# === 1. Load Dataset ===
def load_dataset(filepath):
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        print("⚠️ UTF-8 decode error — trying ISO-8859-1...")
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
    print(f"✅ Dataset loaded with shape: {df.shape}")
    return df

# === 2. Clean Data ===
def clean_dataset(df, cols):
    df_clean = df[cols].copy()
    df_clean.dropna(inplace=True)
    numeric_cols = df_clean.select_dtypes(include='number').columns
    df_clean = df_clean[(np.abs(stats.zscore(df_clean[numeric_cols])) < 3).all(axis=1)]
    print(f"📉 Cleaned dataset shape: {df_clean.shape}")
    df_clean.to_csv(CLEANED_FILE, index=False)
    return df_clean

# === 3. Summary Statistics ===
def summary(df):
    print("\n📊 Summary statistics:")
    print(df.describe(include='all'))

# === 4. Visualization Functions ===
def generate_visuals(df):
    print("\n📈 Generating and saving robust visualizations...")

    # Helper function to save and show plot
    def save_and_show(name):
        filepath = os.path.join(PLOTS_DIR, f"{name}.png")
        plt.savefig(filepath, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"🖼️ Saved plot: {filepath}")

    # 1. Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df["parl25-deprivation-score"], kde=True, bins=30)
    plt.title("Deprivation Score Distribution")
    save_and_show("deprivation_distribution")

    # 2. Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="parl25-imd-pop-quintile", y="parl25-deprivation-score")
    plt.title("Violin Plot: Score by IMD Quintile")
    save_and_show("violin_quintile_score")

    # 3. Boxen plot
    plt.figure(figsize=(10, 6))
    sns.boxenplot(data=df, x="parl25-imd-pop-decile", y="parl25-deprivation-score")
    plt.title("Boxen Plot: Score by IMD Decile")
    save_and_show("boxen_decile_score")

    # 4. Swarm plot
    plt.figure(figsize=(10, 6))
    sns.swarmplot(data=df, x="label", y="parl25-deprivation-score", size=4)
    plt.title("Swarm Plot: Score by Label")
    plt.xticks(rotation=45, ha="right")
    save_and_show("swarm_label_score")

    # 5. Pairplot
    sns.pairplot(df.select_dtypes(include='number'))
    plt.suptitle("Pairplot of Numerical Features", y=1.02)
    plt.savefig(os.path.join(PLOTS_DIR, "pairplot.png"), bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"🖼️ Saved plot: {os.path.join(PLOTS_DIR, 'pairplot.png')}")

    # 6. Correlation Heatmap
    corr = df.corr(numeric_only=True)["parl25-deprivation-score"].sort_values(ascending=False)
    print("\n📈 Correlation with Deprivation Score:")
    print(corr)

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    save_and_show("correlation_heatmap")

    # 7. Strip plot
    plt.figure(figsize=(10, 6))
    sns.stripplot(data=df, x="parl25-imd-pop-quintile", y="high-deprivation", jitter=True)
    plt.title("High Deprivation (%) by IMD Quintile")
    save_and_show("strip_high_deprivation")

    # 8. Stacked bar plot
    df["decile"] = df["parl25-imd-pop-decile"].astype(int)
    group = df.groupby("decile")[["low-deprivation", "medium-deprivation", "high-deprivation"]].mean()

    group.plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.title("Average Deprivation Breakdown by IMD Decile")
    plt.ylabel("Proportion")
    plt.xlabel("IMD Decile")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_and_show("stacked_bar_deprivation")

# === 5. Main ===
if __name__ == "__main__":
    sns.set(style="whitegrid", palette="muted", font_scale=1.1)

    columns = [
        "constituency-name",
        "parl25-deprivation-score",
        "label",
        "low-deprivation",
        "medium-deprivation",
        "high-deprivation",
        "parl25-imd-pop-quintile",
        "parl25-imd-pop-decile"
    ]

    df = load_dataset(INPUT_FILE)
    df_clean = clean_dataset(df, columns)
    summary(df_clean)
    generate_visuals(df_clean)

    print("\n✅ Deprivation data analysis and image saving completed.")
