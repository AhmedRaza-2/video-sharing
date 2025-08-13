import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# === CONFIG ===
INPUT_FILE = "liv.csv"
CLEANED_FILE = "liv/cleaned_data.csv"
os.makedirs("liv", exist_ok=True)

# === 1. Load Dataset ===
def load_dataset(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset loaded with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        exit()

# === 2. Clean Data ===
def clean_dataset(df, cols):
    df_clean = df[cols].copy()
    df_clean.dropna(inplace=True)
    
    # Optional: Remove outliers using z-score
    df_clean = df_clean[(np.abs(stats.zscore(df_clean.select_dtypes(include='number'))) < 3).all(axis=1)]
    print(f"📉 Cleaned dataset shape: {df_clean.shape}")
    
    df_clean.to_csv(CLEANED_FILE, index=False)
    return df_clean

# === 3. Summary Statistics ===
def summary(df):
    print("\n📊 Summary statistics:")
    print(df.describe())

# === 4. Visualization Functions ===
def plot_heatmap(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("liv/correlation_heatmap.png")
    plt.close()

def plot_gdp_vs_education(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='ED_educ_years_mean', y='gdp_pc', hue='country_name', legend=False)
    plt.title("GDP per Capita vs. Education Years")
    plt.xlabel("Education Years")
    plt.ylabel("GDP per Capita")
    plt.tight_layout()
    plt.savefig("liv/gdp_vs_education.png")
    plt.close()

def plot_infant_mortality(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='year', y='HL_IMR', hue='country_name', legend=False)
    plt.title("Infant Mortality Rate Over Time")
    plt.xlabel("Year")
    plt.ylabel("Infant Mortality Rate (HL_IMR)")
    plt.tight_layout()
    plt.savefig("liv/infant_mortality_trend.png")
    plt.close()

def plot_gdp_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['gdp_pc'], kde=True, bins=30)
    plt.title("GDP per Capita Distribution")
    plt.xlabel("GDP per Capita")
    plt.tight_layout()
    plt.savefig("liv/gdp_distribution.png")
    plt.close()

def plot_wealth_vs_education(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='WL_wealth_mean', y='ED_educ_years_mean', hue='year', data=df, palette="viridis")
    plt.title("Wealth vs. Education Over Years")
    plt.xlabel("Wealth Index Mean")
    plt.ylabel("Education Years Mean")
    plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("liv/wealth_vs_education.png")
    plt.close()
def plot_u5mr_trend(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='year', y='HL_U5MR', hue='country_name', legend=False)
    plt.title("Under-5 Mortality Rate (U5MR) Over Time")
    plt.xlabel("Year")
    plt.ylabel("U5MR")
    plt.tight_layout()
    plt.savefig("liv/u5mr_trend.png")
    plt.close()

def plot_gdp_by_region(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='region_name_harmonized', y='gdp_pc')
    plt.title("GDP per Capita by Region")
    plt.xlabel("Region")
    plt.ylabel("GDP per Capita")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("liv/gdp_by_region.png")
    plt.close()

def plot_education_by_country(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='country_name', y='ED_educ_years_mean')
    plt.title("Education Years by Country")
    plt.xlabel("Country")
    plt.ylabel("Education Years")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("liv/education_by_country.png")
    plt.close()

def plot_pairwise_relationships(df):
    sample_df = df.sample(n=min(1000, len(df)))  # Limit size for performance
    pairplot_fig = sns.pairplot(sample_df.drop(columns=["country_name", "region_name_harmonized"]), diag_kind="kde")
    pairplot_fig.fig.suptitle("Pairwise Plot of Numerical Features", y=1.02)
    pairplot_fig.savefig("liv/pairplot_numerical.png")
    plt.close()

def plot_facet_gdp_edu(df):
    g = sns.FacetGrid(df, col="country_name", col_wrap=4, height=3.5)
    g.map_dataframe(sns.scatterplot, x="ED_educ_years_mean", y="gdp_pc", alpha=0.7)
    g.fig.suptitle("GDP vs Education Years by Country", y=1.03)
    g.savefig("liv/faceted_gdp_education.png")
    plt.close()

# === 5. Main Execution ===
if __name__ == "__main__":
    sns.set(style="whitegrid", palette="pastel", font_scale=1.1)
    
    columns_of_interest = [
        'country_name', 'year', 'region_name_harmonized',
        'HD_size_dejure_mean', 'HL_IMR', 'HL_U5MR',
        'ED_educ_years_mean', 'WL_wealth_mean', 'gdp_pc'
    ]

    df_raw = load_dataset(INPUT_FILE)
    df_clean = clean_dataset(df_raw, columns_of_interest)
    summary(df_clean)

    print("\n📈 Generating visualizations...")
    plot_heatmap(df_clean)
    plot_gdp_vs_education(df_clean)
    plot_infant_mortality(df_clean)
    plot_gdp_distribution(df_clean)
    plot_wealth_vs_education(df_clean)
    plot_u5mr_trend(df_clean)
    plot_gdp_by_region(df_clean)
    plot_education_by_country(df_clean)
    plot_pairwise_relationships(df_clean)
    plot_facet_gdp_edu(df_clean)

    print("\n✅ Robust analysis completed. Visuals saved in [liv] folder.")
