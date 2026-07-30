import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression



# STEP 1: CONFUSION MATRIX (BOTH MODELS)

y_pred_dt = dt_model.predict(X_test)
y_pred_lr_raw = lr_model.predict(X_test)
y_pred_lr_binary = (y_pred_lr_raw >= 0.5).astype(int)

cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_lr = confusion_matrix(y_test, y_pred_lr_binary)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Decision Tree Confusion Matrix
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Decision Tree Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Linear Regression Confusion Matrix
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Linear Regression Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()



# STEP 2: OVERFITTING & UNDERFITTING GRAPHS (BOTH MODELS)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# --- Model 1: Decision Tree (Train vs Validation Accuracy across Depths) ---
train_acc_dt, test_acc_dt = [], []
depths = range(1, 11)

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_acc_dt.append(dt.score(X_train, y_train))
    test_acc_dt.append(dt.score(X_test, y_test))

axes[0].plot(depths, train_acc_dt, label='Training Accuracy', marker='o', color='blue')
axes[0].plot(depths, test_acc_dt, label='Validation Accuracy', marker='s', color='orange')
axes[0].set_title('Decision Tree: Overfitting / Underfitting')
axes[0].set_xlabel('Tree Depth (Model Complexity)')
axes[0].set_ylabel('Accuracy Score')
axes[0].legend()
axes[0].grid(True)


# --- Model 2: Linear Regression (Train vs Validation Loss across Sample Sizes) ---
train_mse_lr, test_mse_lr = [], []
fractions = np.linspace(0.1, 1.0, 10)

for f in fractions:
    idx = max(2, int(len(X_train) * f))  # At least 2 samples to fit
    lr = LinearRegression()
    
    # Handle slicing whether X_train/y_train are DataFrames or NumPy Arrays
    if hasattr(X_train, 'iloc'):
        X_tr_slice, y_tr_slice = X_train.iloc[:idx], y_train.iloc[:idx]
    else:
        X_tr_slice, y_tr_slice = X_train[:idx], y_train[:idx]
        
    lr.fit(X_tr_slice, y_tr_slice)
    
    train_mse_lr.append(mean_squared_error(y_tr_slice, lr.predict(X_tr_slice)))
    test_mse_lr.append(mean_squared_error(y_test, lr.predict(X_test)))

axes[1].plot(fractions * 100, train_mse_lr, label='Training Loss (MSE)', marker='o', color='blue')
axes[1].plot(fractions * 100, test_mse_lr, label='Validation Loss (MSE)', marker='s', color='orange')
axes[1].set_title('Linear Regression: Training vs Validation Loss')
axes[1].set_xlabel('Training Data Used (%)')
axes[1].set_ylabel('Loss (MSE)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()



# STEP 3: MODEL SUMMARY (BOTH MODELS)

print("="*45)
print("             MODEL SUMMARY               ")
print("="*45)

print("\n- Model 1: Decision Tree Classifier -")
print(f"Algorithm Type : DecisionTreeClassifier")
print(f"Parameters     : {dt_model.get_params()}")

print("\n--- Model 2: Linear Regression ---")
print(f"Algorithm Type : LinearRegression")
lr_intercept = float(np.ravel(lr_model.intercept_)[0])
print(f"Intercept      : {lr_intercept:.4f}")
print(f"Coefficients   : {np.round(lr_model.coef_, 4)}")
print(f"Parameters     : {lr_model.get_params()}")







#--------------------- STEP 5: COMPARATIVE ANALYTICS (BOTH MODELS)

print("\n" + "="*45)
print("        5. COMPARATIVE ANALYTICS         ")
print("="*45)

acc_dt = accuracy_score(y_test, y_pred_dt)
acc_lr = accuracy_score(y_test, y_pred_lr_binary)

mse_dt = mean_squared_error(y_test, y_pred_dt)
mse_lr = mean_squared_error(y_test, y_pred_lr_raw)

r2_dt = r2_score(y_test, y_pred_dt)
r2_lr = r2_score(y_test, y_pred_lr_raw)

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Mean Squared Error (MSE)', 'R2 Score'],
    'Decision Tree': [f"{acc_dt:.4%}", f"{mse_dt:.4f}", f"{r2_dt:.4f}"],
    'Linear Regression': [f"{acc_lr:.4%}", f"{mse_lr:.4f}", f"{r2_lr:.4f}"]
})

print(comparison_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
plot_data = pd.DataFrame({
    'Model': ['Decision Tree', 'Linear Regression'],
    'Accuracy Score': [acc_dt, acc_lr]
})

sns.barplot(data=plot_data, x='Model', y='Accuracy Score', palette=['steelblue', 'seagreen'], ax=ax)
ax.set_title('Comparative Analytics: Model Accuracy Comparison', fontsize=12)
ax.set_ylim(0, 1.0)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.show()
# -------------------------STEP 4: TESTING RESULTS (SINGLE ROW TEST)

# Get the exact feature order from training
feature_order = list(X_train.columns)

manual_values = {
    'gender': 0,             # 0=Female,1=Male,2=Other
    'age': 80,
    'hypertension': 0,
    'heart_disease': 1,
    'smoking_history': 4,    # 0=never,1=former,2=current,3=not known
    'bmi': 25.19,
    'HbA1c_level': 6.6,
    'blood_glucose_level': 140
}

# Build DataFrame in correct order
manual_row = pd.DataFrame([[manual_values[feat] for feat in feature_order]],
                          columns=feature_order)

# Predictions
dt_pred = dt_model.predict(manual_row)[0]
lr_raw = lr_model.predict(manual_row)[0]
lr_pred = 1 if lr_raw >= 0.5 else 0

print("\n Runtime Prediction ")
print("Input Features:", manual_values)
print(f"Decision Tree Prediction : {dt_pred} ({'Diabetes' if dt_pred==1 else 'No Diabetes'})")
print(f"Linear Regression Prediction : {lr_pred} ({'Diabetes' if lr_pred==1 else 'No Diabetes'})")