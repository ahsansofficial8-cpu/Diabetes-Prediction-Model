import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from google.colab import files

# Upload dataset
print("Uploading cleaned dataset file...")
uploaded = files.upload()




print("Loading cleaned dataset...")
df = pd.read_csv("diabetes_fully_cleaned.csv")


print("Missing vakues:", df.isna().sum())

# Features and target
X = df.drop(columns=['diabetes'])
y = df['diabetes']

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5,random_state=42)
dt_model.fit(X_train,y_train)
y_pred_dt = dt_model.predict(X_test)
acc = accuracy_score(y_test,y_pred_dt)
print(f"Decision Tree Accuracy: {acc:.4%}")

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train,y_train)
y_pred_lr = lr_model.predict(X_test)
mse = mean_squared_error(y_test,y_pred_lr)
r2 = r2_score(y_test,y_pred_lr)
print(f"Linear Regression MSE: {mse:.4f}")
print(f"Linear Regression R2 Score: {r2:.4f}")

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("Features in X_test:", list(X_test.columns))
print("Shape of X_test   :", X_test.shape)

# 1. Calculate SHAP values
explainer = shap.TreeExplainer(dt_model)
shap_values = explainer.shap_values(X_test)

# 2. Handle whatever structure SHAP returns (list or 3D array)
if isinstance(shap_values, list):
    # Old SHAP: list of two arrays [class_0, class_1]
    shap_array = shap_values[1]
else:
    # New SHAP: single array of shape (samples, features, classes)
    print("Raw SHAP shape:", shap_values.shape)
    if shap_values.ndim == 3:
        shap_array = shap_values[:, :, 1]   # <-- take class 1 (diabetes positive)
    else:
        shap_array = shap_values

print("Final SHAP array shape:", shap_array.shape)  # should be (19230, 8)

# 3. Mean absolute SHAP value per feature
mean_shap = np.abs(shap_array).mean(axis=0)

# 4. Build dataframe and sort
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'mean_shap': mean_shap
}).sort_values('mean_shap', ascending=True)  # ascending = biggest bar on top

# 5. Plot horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['mean_shap'], color='steelblue')
plt.xlabel('Mean |SHAP Value|  →  Average Impact on Model Output', fontsize=12)
plt.title('Feature Importance (Average SHAP Values)', fontsize=14)
plt.tight_layout()
plt.show()

# 6. Print ranked table for your report
print("\nRanked Feature Importance (highest → lowest):")
print(importance_df.sort_values('mean_shap', ascending=False).to_string(index=False))