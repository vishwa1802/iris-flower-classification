# ================================
# Step 1: Import Libraries
# ================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ================================
# Step 2: Load Dataset
# ================================
iris = load_iris()

X = iris.data
y = iris.target

print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)

# ================================
# Step 3: Convert to DataFrame (EDA)
# ================================
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

print("\nFirst 5 rows:")
print(df.head())

# ================================
# Step 4: Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ================================
# Step 5: Train Model (KNN)
# ================================
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ================================
# Step 6: Prediction
# ================================
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ================================
# Step 7: Confusion Matrix
# ================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ================================
# Step 8: Test with Custom Input
# ================================
# Example flower: [sepal_length, sepal_width, petal_length, petal_width]
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = knn.predict(sample)
print("\nPredicted species:", iris.target_names[prediction][0])