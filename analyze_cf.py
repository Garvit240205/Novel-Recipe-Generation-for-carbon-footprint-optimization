import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
CSV_FILE_PATH = './btp/btp/results/generated_recipes_optimized.csv'

# --- Load Data ---
print(f"Attempting to load data from: {CSV_FILE_PATH}")
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print("CSV file loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {CSV_FILE_PATH}")
    print("Please ensure the file path is correct and the script is run from the appropriate directory.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
    exit()

# --- Data Cleaning and Preparation ---
print("\n--- Data Cleaning and Preparation ---")

# Check if required columns exist
required_cols = ['Original CF (kg CO2e)', 'Optimized CF (kg CO2e)']
if not all(col in df.columns for col in required_cols):
    print(f"Error: Missing one or more required columns: {required_cols}")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

# Convert CF columns to numeric, coercing errors to NaN
df['Original CF (kg CO2e)'] = pd.to_numeric(df['Original CF (kg CO2e)'], errors='coerce')
df['Optimized CF (kg CO2e)'] = pd.to_numeric(df['Optimized CF (kg CO2e)'], errors='coerce')

# Drop rows where CF conversion failed or values are missing
initial_rows = len(df)
df.dropna(subset=['Original CF (kg CO2e)', 'Optimized CF (kg CO2e)'], inplace=True)
cleaned_rows = len(df)
if initial_rows > cleaned_rows:
    print(f"Dropped {initial_rows - cleaned_rows} rows due to missing/invalid CF values.")

if df.empty:
    print("Error: No valid data remaining after cleaning. Cannot perform analysis.")
    exit()

# Calculate Absolute Reduction
df['Absolute Reduction (kg CO2e)'] = df['Original CF (kg CO2e)'] - df['Optimized CF (kg CO2e)']

# Calculate Percentage Reduction, handling division by zero
# Replace 0 in original CF with NaN temporarily for division, then fill resulting NaN% with 0
original_cf_for_pct = df['Original CF (kg CO2e)'].replace(0, np.nan)
df['Percentage Reduction (%)'] = (df['Absolute Reduction (kg CO2e)'] / original_cf_for_pct) * 100
df['Percentage Reduction (%)'].fillna(0, inplace=True) # If original was 0, reduction is 0%

# Optional: Clip extreme percentages if needed (e.g., if original CF was very small)
# df['Percentage Reduction (%)'] = df['Percentage Reduction (%)'].clip(lower=-100, upper=150) # Example clipping

print("Calculated Absolute and Percentage Reductions.")
print(df[['Original CF (kg CO2e)', 'Optimized CF (kg CO2e)', 'Absolute Reduction (kg CO2e)', 'Percentage Reduction (%)']].head())

# --- Calculate Overall Statistics ---
print("\n--- Overall Statistics ---")

avg_original_cf = df['Original CF (kg CO2e)'].mean()
avg_optimized_cf = df['Optimized CF (kg CO2e)'].mean()
avg_abs_reduction = df['Absolute Reduction (kg CO2e)'].mean()
# Calculate average percentage reduction only on recipes where original CF > 0
avg_pct_reduction = df[df['Original CF (kg CO2e)'] > 0]['Percentage Reduction (%)'].mean()
total_original_cf = df['Original CF (kg CO2e)'].sum()
total_optimized_cf = df['Optimized CF (kg CO2e)'].sum()
total_abs_reduction = df['Absolute Reduction (kg CO2e)'].sum()
median_pct_reduction = df[df['Original CF (kg CO2e)'] > 0]['Percentage Reduction (%)'].median()
num_recipes_analyzed = len(df)
num_recipes_reduced = len(df[df['Absolute Reduction (kg CO2e)'] > 0])
pct_recipes_reduced = (num_recipes_reduced / num_recipes_analyzed) * 100 if num_recipes_analyzed > 0 else 0

print(f"Number of Recipes Analyzed: {num_recipes_analyzed}")
print(f"Number of Recipes with CF Reduction: {num_recipes_reduced} ({pct_recipes_reduced:.2f}%)")
print("-" * 20)
print(f"Average Original CF per Recipe: {avg_original_cf:.4f} kg CO2e")
print(f"Average Optimized CF per Recipe: {avg_optimized_cf:.4f} kg CO2e")
print(f"Average Absolute Reduction per Recipe: {avg_abs_reduction:.4f} kg CO2e")
print(f"Average Percentage Reduction per Recipe (where Original CF > 0): {avg_pct_reduction:.2f}%")
print(f"Median Percentage Reduction per Recipe (where Original CF > 0): {median_pct_reduction:.2f}%")
print("-" * 20)
print(f"Total Original CF (Sum): {total_original_cf:.2f} kg CO2e")
print(f"Total Optimized CF (Sum): {total_optimized_cf:.2f} kg CO2e")
print(f"Total Absolute CF Reduction (Sum): {total_abs_reduction:.2f} kg CO2e")

# --- Generate Plots (Save to Files) ---
print("\n--- Generating and Saving Plots ---")
sns.set_theme(style="whitegrid")

# Create a directory for plots if it doesn't exist
plot_dir = './analysis_plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory for plots: {plot_dir}")

# Plot 1: Distribution of Original vs Optimized CF
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Original CF (kg CO2e)'], label='Original CF', fill=True, alpha=0.5)
sns.kdeplot(df['Optimized CF (kg CO2e)'], label='Optimized CF', fill=True, alpha=0.5)
plt.title('Distribution of Original vs Optimized Carbon Footprints')
plt.xlabel('Carbon Footprint (kg CO2e)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
# You might want to limit the x-axis if there are extreme outliers
# plt.xlim(0, df['Original CF (kg CO2e)'].quantile(0.95)) # Example: limit to 95th percentile
plot_filename_1 = os.path.join(plot_dir, '1_cf_distribution_comparison.png')
plt.savefig(plot_filename_1)
print(f"Saved Plot 1 to: {plot_filename_1}")
plt.close() # Close the plot figure

# Plot 2: Histogram of Absolute Reduction
plt.figure(figsize=(10, 6))
sns.histplot(df['Absolute Reduction (kg CO2e)'], bins=30, kde=True)
plt.title('Distribution of Absolute CF Reduction per Recipe')
plt.xlabel('Absolute Reduction (kg CO2e)')
plt.ylabel('Number of Recipes')
plt.axvline(0, color='red', linestyle='--', label='No Reduction') # Line at zero reduction
plt.legend()
plt.tight_layout()
plot_filename_2 = os.path.join(plot_dir, '2_absolute_reduction_distribution.png')
plt.savefig(plot_filename_2)
print(f"Saved Plot 2 to: {plot_filename_2}")
plt.close() # Close the plot figure

# Plot 3: Histogram of Percentage Reduction
# Filter out potential extreme values for better visualization if necessary
pct_reduction_filtered = df['Percentage Reduction (%)']#.clip(-50, 100)
plt.figure(figsize=(10, 6))
sns.histplot(pct_reduction_filtered, bins=40, kde=False)
plt.title('Distribution of Percentage CF Reduction per Recipe')
plt.xlabel('Percentage Reduction (%)')
plt.ylabel('Number of Recipes')
plt.axvline(0, color='red', linestyle='--', label='No Reduction') # Line at zero reduction
plt.legend()
# plt.xlim(-50, 100) # Optional: Set x-axis limits for clarity
plt.tight_layout()
plot_filename_3 = os.path.join(plot_dir, '3_percentage_reduction_distribution.png')
plt.savefig(plot_filename_3)
print(f"Saved Plot 3 to: {plot_filename_3}")
plt.close() # Close the plot figure

# Plot 4: Scatter Plot of Optimized vs Original CF
plt.figure(figsize=(8, 8))
sns.scatterplot(data=df, x='Original CF (kg CO2e)', y='Optimized CF (kg CO2e)', alpha=0.6)
# Add a y=x line (no change)
max_cf = max(df['Original CF (kg CO2e)'].max(), df['Optimized CF (kg CO2e)'].max()) if not df.empty else 1
min_cf = min(df['Original CF (kg CO2e)'].min(), df['Optimized CF (kg CO2e)'].min()) if not df.empty else 0
# Ensure min_cf is not greater than max_cf if data caused issues
min_cf = min(min_cf, max_cf)
plt.plot([min_cf, max_cf], [min_cf, max_cf], color='red', linestyle='--', label='y=x (No Change)')
plt.title('Optimized CF vs Original CF')
plt.xlabel('Original CF (kg CO2e)')
plt.ylabel('Optimized CF (kg CO2e)')
plt.legend()
plt.grid(True)
plt.axis('equal') # Ensure axes have the same scale for easy comparison to y=x
plt.tight_layout()
plot_filename_4 = os.path.join(plot_dir, '4_optimized_vs_original_scatter.png')
plt.savefig(plot_filename_4)
print(f"Saved Plot 4 to: {plot_filename_4}")
plt.close() # Close the plot figure

# Plot 5: Bar Chart of Average CFs
plt.figure(figsize=(6, 5))
avg_data = pd.DataFrame({
    'CF Type': ['Average Original', 'Average Optimized'],
    'Average CF (kg CO2e)': [avg_original_cf, avg_optimized_cf]
})
barplot = sns.barplot(data=avg_data, x='CF Type', y='Average CF (kg CO2e)')
plt.title('Average Carbon Footprint Before and After Optimization')
plt.ylabel('Average CF per Recipe (kg CO2e)')
plt.xlabel('')
# Add values on top of bars
for container in barplot.containers:
    barplot.bar_label(container, fmt='%.3f')
plt.tight_layout()
plot_filename_5 = os.path.join(plot_dir, '5_average_cf_comparison_bar.png')
plt.savefig(plot_filename_5)
print(f"Saved Plot 5 to: {plot_filename_5}")
plt.close() # Close the plot figure

print("\n--- Plot Saving Complete ---")