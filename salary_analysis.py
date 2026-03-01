import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the data
df = pd.read_csv('Salary_Data.csv')

print("Dataset Overview:")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nBasic statistics:")
print(df.describe())

# Prepare features and target
X = df[['YearsExperience']].values
y = df['Salary'].values

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate performance metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

# Print model results
print("\n" + "="*50)
print("LINEAR REGRESSION MODEL RESULTS")
print("="*50)
print(f"\nModel Equation: Salary = {model.intercept_:.2f} + {model.coef_[0]:.2f} × Years")
print(f"\nIntercept: ${model.intercept_:,.2f}")
print(f"Coefficient: ${model.coef_[0]:,.2f} per year")

print("\n" + "-"*50)
print("Performance Metrics:")
print("-"*50)
print(f"Training R² Score:   {train_r2:.4f}")
print(f"Testing R² Score:    {test_r2:.4f}")
print(f"\nTraining RMSE:       ${train_rmse:,.2f}")
print(f"Testing RMSE:        ${test_rmse:,.2f}")
print(f"\nTraining MAE:        ${train_mae:,.2f}")
print(f"Testing MAE:         ${test_mae:,.2f}")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Scatter plot with regression line
axes[0].scatter(X_train, y_train, color='blue', alpha=0.6, label='Training Data')
axes[0].scatter(X_test, y_test, color='green', alpha=0.6, label='Testing Data')
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range = model.predict(X_range)
axes[0].plot(X_range, y_range, color='red', linewidth=2, label='Regression Line')
axes[0].set_xlabel('Years of Experience')
axes[0].set_ylabel('Salary ($)')
axes[0].set_title('Salary vs Years of Experience')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Residuals plot
residuals_test = y_test - y_test_pred
axes[1].scatter(y_test_pred, residuals_test, color='purple', alpha=0.6)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Salary ($)')
axes[1].set_ylabel('Residuals ($)')
axes[1].set_title('Residual Plot')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('salary_regression_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'salary_regression_analysis.png'")
plt.show()

# Example predictions
print("\n" + "="*50)
print("EXAMPLE PREDICTIONS")
print("="*50)
example_years = np.array([[3], [5], [7], [10]])
predictions = model.predict(example_years)
for years, pred in zip(example_years.flatten(), predictions):
    print(f"{years} years experience → Predicted salary: ${pred:,.2f}")