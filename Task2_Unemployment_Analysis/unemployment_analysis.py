# 📊 Unemployment Analysis in India during COVID-19
# Author: Vommi Uma Mahesh

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tabulate import tabulate

# Step 1: Load Dataset
data = pd.read_csv("Unemployment in India.csv")
data.columns = data.columns.str.strip()  # remove unwanted spaces
print("✅ Dataset Loaded Successfully!\n")

# Step 2: Preview Data
print("🔍 Data Preview:\n")
print(tabulate(data.head(), headers='keys', tablefmt='fancy_grid'))
print("\n📏 Dataset Shape:", data.shape)
print("📂 Columns:", list(data.columns), "\n")

# Step 3: Clean Data
data.dropna(inplace=True)
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)

data.rename(columns={
    "Region": "State",
    "Estimated Unemployment Rate (%)": "Unemployment_Rate",
    "Estimated Employed": "Employed_Population",
    "Estimated Labour Participation Rate (%)": "Labour_Participation"
}, inplace=True)

# Step 4: Visualization 1 - Unemployment Trend
print("📈 Showing Unemployment Trend Over Time...")
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x="Date", y="Unemployment_Rate", hue="State", legend=False)
plt.title("Unemployment Rate Over Time (All States)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()
plt.close()

# Step 5: Visualization 2 - Average by State
print("📊 Showing Average Unemployment Rate by State...")
state_avg = data.groupby("State")["Unemployment_Rate"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 7))
sns.barplot(x=state_avg.values, y=state_avg.index, palette="coolwarm")
plt.title("Average Unemployment Rate by State")
plt.xlabel("Average Rate (%)")
plt.ylabel("State")
plt.tight_layout()
plt.show()
plt.close()

# Step 6: Visualization 3 - Correlation Heatmap
print("🌡️ Showing Correlation Heatmap...")
plt.figure(figsize=(6, 4))
sns.heatmap(
    data[["Unemployment_Rate", "Employed_Population", "Labour_Participation"]].corr(),
    annot=True, cmap="YlGnBu", fmt=".2f"
)
plt.title("Correlation Between Employment Factors")
plt.tight_layout()
plt.show()
plt.close()

# Step 7: Interactive Choropleth Map
print("🌍 Generating Interactive Map (saved as HTML)...")
fig = px.choropleth(
    data,
    locations="State",
    locationmode="country names",
    color="Unemployment_Rate",
    hover_name="State",
    animation_frame=data["Date"].dt.strftime("%b %Y"),
    title="Unemployment Rate in India (COVID-19 Period)",
    color_continuous_scale="Reds"
)
fig.write_html("unemployment_map.html")

# Step 8: Insights
highest = data.loc[data["Unemployment_Rate"].idxmax()]
lowest = data.loc[data["Unemployment_Rate"].idxmin()]

print("\n💡 Insights:")
print(f"Highest Unemployment: {highest['Unemployment_Rate']:.2f}% in {highest['State']} ({highest['Date'].strftime('%b %Y')})")
print(f"Lowest Unemployment: {lowest['Unemployment_Rate']:.2f}% in {lowest['State']} ({lowest['Date'].strftime('%b %Y')})")

print("\n✅ All visualizations displayed successfully.")
