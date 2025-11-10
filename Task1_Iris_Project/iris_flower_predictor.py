# 🌸 Iris Flower Prediction with Visualization
# Author: Vommi Uma Mahesh

# Step 1: Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate   # for table display

# Step 2: Load dataset safely (works from any location)
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Iris.csv")
iris_data = pd.read_csv(file_path)
print("✅ Dataset Loaded Successfully!\n")

# Step 3: Data overview
if "Id" in iris_data.columns:
    iris_data.drop(columns="Id", inplace=True)

print("🔍 Data Preview:\n")
print(tabulate(iris_data.head(), headers='keys', tablefmt='fancy_grid'))
print("\n")

# Step 4: Visualization 1 - Species count
print("📊 Displaying Visualization 1: Species Count...")
plt.figure(figsize=(6, 4))
sns.countplot(data=iris_data, x="Species", hue="Species", legend=False, palette="cool")
plt.title("Count of Each Iris Species 🌸")
plt.xlabel("Flower Species")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
plt.close()

# Step 5: Visualization 2 - Pairplot
print("\n🌿 Displaying Visualization 2: Feature Relationships (Pairplot)...")
sns.pairplot(iris_data, hue="Species", palette="husl")
plt.suptitle("Feature Relationships Between Iris Species", y=1.02)
plt.show()
plt.close()

# Step 6: Visualization 3 - Correlation heatmap
print("\n🌼 Displaying Visualization 3: Feature Correlation Heatmap...")
plt.figure(figsize=(6, 4))
sns.heatmap(iris_data.drop(columns="Species").corr(), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Feature Correlation Heatmap 🌼")
plt.tight_layout()
plt.show()
plt.close()

# Step 7: Split data into features and labels
flower_features = iris_data.drop(columns="Species")
flower_labels = iris_data["Species"]

train_features, test_features, train_labels, test_labels = train_test_split(
    flower_features, flower_labels, test_size=0.2, random_state=12
)

# Step 8: Train the model
iris_classifier = RandomForestClassifier(n_estimators=120, random_state=12)
iris_classifier.fit(train_features, train_labels)

# Step 9: Evaluate model
predicted_labels = iris_classifier.predict(test_features)
model_accuracy = accuracy_score(test_labels, predicted_labels) * 100

print("\n🎯 Model Performance:")
print(f"Accuracy: {model_accuracy:.2f}%")
print("\n📊 Classification Report:\n", classification_report(test_labels, predicted_labels))

# Step 10: Test with a new sample flower
sample_flower = pd.DataFrame([[5.8, 2.8, 5.1, 2.4]], columns=flower_features.columns)
predicted_species = iris_classifier.predict(sample_flower)
print("\n🌺 Predicted Species for the sample flower:", predicted_species[0])

print("\n✅ All visualizations displayed successfully.")
