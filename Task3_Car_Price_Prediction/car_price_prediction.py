# 🚗 Car Price Prediction using Machine Learning
# Author: Vommi Uma Mahesh

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from tabulate import tabulate

# Step 1: Load Dataset
car_data = pd.read_csv("car data.csv")
print("✅ Dataset Loaded Successfully!\n")

# Step 2: Preview Dataset in Table Format
print("🔍 Data Preview:\n")
print(tabulate(car_data.head(), headers='keys', tablefmt='fancy_grid'))
print("\n📏 Dataset Shape:", car_data.shape)
print("📂 Columns:", list(car_data.columns), "\n")

# Step 3: Clean and Encode Data
car_data.dropna(inplace=True)
cat_cols = car_data.select_dtypes(include=['object']).columns
for col in cat_cols:
    car_data[col] = car_data[col].astype('category').cat.codes

# Step 4: Split Data into Features and Target
features = car_data.drop("Selling_Price", axis=1)
target = car_data["Selling_Price"]

train_feat, test_feat, train_label, test_label = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Step 5: Train Model
print("🧠 Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_feat, train_label)

# Step 6: Model Evaluation
print("\n📊 Model Evaluation Results:")
predictions = model.predict(test_feat)
r2 = r2_score(test_label, predictions)
mae = mean_absolute_error(test_label, predictions)
print(f"R² Score: {r2:.3f}")
print(f"Mean Absolute Error: {mae:.3f}\n")

# Step 7: Visualization 1 - Actual vs Predicted Prices
print("📈 Displaying Actual vs Predicted Car Prices...")
plt.figure(figsize=(7, 5))
sns.scatterplot(x=test_label, y=predictions)
plt.xlabel("Actual Selling Price (₹ Lakh)")
plt.ylabel("Predicted Selling Price (₹ Lakh)")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()
plt.show()
plt.close()

# Step 8: Visualization 2 - Feature Importance
print("🚀 Displaying Top Features Influencing Car Price...")
importance = pd.Series(model.feature_importances_, index=features.columns)
importance.nlargest(8).plot(kind='barh', color='teal')
plt.title("Top 8 Features Affecting Car Price")
plt.tight_layout()
plt.show()
plt.close()

# Step 9: Insights
print("💡 Key Insights:")
top_feature = importance.nlargest(1).index[0]
print(f"➡️ The most influential feature in predicting car price is: {top_feature}")
print("\n✅ All visualizations displayed successfully.")
