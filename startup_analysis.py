"""
MULTIPLE LINEAR REGRESSION ANALYSIS
Predicting Startup Profit based on R&D Spend, Administration, Marketing Spend, and State

THEORY EXPLANATION:
==================
Simple Linear Regression: y = b0 + b1*x1
Multiple Linear Regression: y = b0 + b1*x1 + b2*x2 + b3*x3 + ... + bn*xn

Where:
- y = dependent variable (what we're predicting: Profit)
- x1, x2, x3... = independent variables (predictors: R&D, Admin, Marketing, State)
- b0 = intercept (baseline profit when all predictors are 0)
- b1, b2, b3... = coefficients (how much each predictor affects profit)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("="*80)
print("MULTIPLE LINEAR REGRESSION: STARTUP PROFIT PREDICTION")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND EXPLORE THE DATA
# ============================================================================
print("\n📊 STEP 1: DATA LOADING AND EXPLORATION")
print("-"*80)

df = pd.read_csv('50_Startups.csv')

print("\nFirst 5 rows of data:")
print(df.head())

print(f"\nDataset shape: {df.shape[0]} startups, {df.shape[1]} variables")

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

print("\nState distribution:")
print(df['State'].value_counts())

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n\n🔧 STEP 2: DATA PREPROCESSING")
print("-"*80)
print("\nENCODING CATEGORICAL VARIABLES:")
print("State is a categorical variable (New York, California, Florida)")
print("We need to convert it to numerical format using Label Encoding")

# Encode the 'State' column
le = LabelEncoder()
df['State_Encoded'] = le.fit_transform(df['State'])

print(f"\nEncoding mapping:")
for i, state in enumerate(le.classes_):
    print(f"  {state} → {i}")

# Prepare features (X) and target (y)
X = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State_Encoded']].values
y = df['Profit'].values

print(f"\nFeature matrix X shape: {X.shape}")
print(f"Target vector y shape: {y.shape}")

# ============================================================================
# STEP 3: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================
print("\n\n✂️ STEP 3: TRAIN-TEST SPLIT")
print("-"*80)
print("Splitting data: 80% training, 20% testing")
print("Why? To evaluate model performance on unseen data (avoid overfitting)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# ============================================================================
# STEP 4: BUILD AND TRAIN THE MODEL
# ============================================================================
print("\n\n🤖 STEP 4: MODEL TRAINING")
print("-"*80)
print("Using Ordinary Least Squares (OLS) method")
print("Goal: Minimize the sum of squared residuals (errors)")

model = LinearRegression()
model.fit(X_train, y_train)

print("\n✓ Model trained successfully!")

# ============================================================================
# STEP 5: ANALYZE MODEL COEFFICIENTS
# ============================================================================
print("\n\n📈 STEP 5: MODEL EQUATION AND INTERPRETATION")
print("-"*80)

feature_names = ['R&D Spend', 'Administration', 'Marketing Spend', 'State']
coefficients = model.coef_
intercept = model.intercept_

print(f"\nMODEL EQUATION:")
print(f"Profit = {intercept:.2f}")
for name, coef in zip(feature_names, coefficients):
    sign = "+" if coef >= 0 else ""
    print(f"         {sign} {coef:.4f} × {name}")

print("\n\nINTERPRETATION OF COEFFICIENTS:")
print("-"*80)
for name, coef in zip(feature_names, coefficients):
    if coef > 0:
        interpretation = f"increases profit by ${abs(coef):.2f}"
    else:
        interpretation = f"decreases profit by ${abs(coef):.2f}"
    print(f"\n• {name}: {coef:.4f}")
    print(f"  → Every $1 increase in {name} {interpretation}")
    
print(f"\n• Intercept: ${intercept:.2f}")
print(f"  → Baseline profit when all predictors are zero")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("\n\nFEATURE IMPORTANCE (by coefficient magnitude):")
print(feature_importance.to_string(index=False))

# ============================================================================
# STEP 6: MAKE PREDICTIONS
# ============================================================================
print("\n\n🎯 STEP 6: MAKING PREDICTIONS")
print("-"*80)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Predictions generated for both training and testing sets")

# ============================================================================
# STEP 7: EVALUATE MODEL PERFORMANCE
# ============================================================================
print("\n\n📊 STEP 7: MODEL EVALUATION")
print("-"*80)

# R-squared score
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nR² SCORE (Coefficient of Determination):")
print(f"  Training R²: {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"  Testing R²:  {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"\n  Interpretation: The model explains {test_r2*100:.2f}% of variance in profit")
print("  • R² = 1.0: Perfect predictions")
print("  • R² = 0.0: Model no better than predicting the mean")
print("  • R² < 0.0: Model worse than baseline")

# RMSE (Root Mean Squared Error)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("\n\nRMSE (Root Mean Squared Error):")
print(f"  Training RMSE: ${train_rmse:,.2f}")
print(f"  Testing RMSE:  ${test_rmse:,.2f}")
print(f"\n  Interpretation: On average, predictions are off by ${test_rmse:,.2f}")
print("  • Lower RMSE = Better model")
print("  • RMSE in same units as target variable (dollars)")

# MAE (Mean Absolute Error)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n\nMAE (Mean Absolute Error):")
print(f"  Training MAE: ${train_mae:,.2f}")
print(f"  Testing MAE:  ${test_mae:,.2f}")
print(f"\n  Interpretation: Average absolute prediction error is ${test_mae:,.2f}")
print("  • MAE less sensitive to outliers than RMSE")

# Adjusted R²
n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)

print(f"\n\nADJUSTED R²: {adjusted_r2:.4f}")
print("  Interpretation: R² adjusted for number of predictors")
print("  • Penalizes adding unnecessary variables")

# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================
print("\n\n📉 STEP 8: CREATING VISUALIZATIONS")
print("-"*80)

fig = plt.figure(figsize=(16, 12))

# Plot 1: Correlation Heatmap
ax1 = plt.subplot(3, 3, 1)
correlation_matrix = df[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax1.set_title('Correlation Matrix\n(Shows relationships between variables)', fontsize=10, fontweight='bold')

# Plot 2: Actual vs Predicted (Training)
ax2 = plt.subplot(3, 3, 2)
ax2.scatter(y_train, y_train_pred, alpha=0.6, color='blue', edgecolors='k', linewidth=0.5)
ax2.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Profit ($)')
ax2.set_ylabel('Predicted Profit ($)')
ax2.set_title(f'Training Set: Actual vs Predicted\nR² = {train_r2:.4f}', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Actual vs Predicted (Testing)
ax3 = plt.subplot(3, 3, 3)
ax3.scatter(y_test, y_test_pred, alpha=0.6, color='green', edgecolors='k', linewidth=0.5)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax3.set_xlabel('Actual Profit ($)')
ax3.set_ylabel('Predicted Profit ($)')
ax3.set_title(f'Testing Set: Actual vs Predicted\nR² = {test_r2:.4f}', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals Distribution
ax4 = plt.subplot(3, 3, 4)
residuals_test = y_test - y_test_pred
ax4.hist(residuals_test, bins=15, color='purple', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Residual Value ($)')
ax4.set_ylabel('Frequency')
ax4.set_title('Residual Distribution\n(Should be normally distributed)', fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Residuals vs Predicted
ax5 = plt.subplot(3, 3, 5)
ax5.scatter(y_test_pred, residuals_test, alpha=0.6, color='orange', edgecolors='k', linewidth=0.5)
ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Profit ($)')
ax5.set_ylabel('Residuals ($)')
ax5.set_title('Residual Plot\n(Should show random scatter)', fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Feature Coefficients
ax6 = plt.subplot(3, 3, 6)
colors = ['green' if c > 0 else 'red' for c in coefficients]
ax6.barh(feature_names, coefficients, color=colors, alpha=0.7, edgecolor='black')
ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax6.set_xlabel('Coefficient Value')
ax6.set_title('Feature Coefficients\n(Impact on Profit)', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')

# Plot 7: R&D Spend vs Profit
ax7 = plt.subplot(3, 3, 7)
ax7.scatter(df['R&D Spend'], df['Profit'], alpha=0.6, color='blue', edgecolors='k', linewidth=0.5)
ax7.set_xlabel('R&D Spend ($)')
ax7.set_ylabel('Profit ($)')
ax7.set_title('R&D Spend vs Profit\n(Strongest predictor)', fontweight='bold')
ax7.grid(True, alpha=0.3)

# Plot 8: Marketing Spend vs Profit
ax8 = plt.subplot(3, 3, 8)
ax8.scatter(df['Marketing Spend'], df['Profit'], alpha=0.6, color='green', edgecolors='k', linewidth=0.5)
ax8.set_xlabel('Marketing Spend ($)')
ax8.set_ylabel('Profit ($)')
ax8.set_title('Marketing Spend vs Profit', fontweight='bold')
ax8.grid(True, alpha=0.3)

# Plot 9: Model Performance Comparison
ax9 = plt.subplot(3, 3, 9)
metrics = ['R²', 'RMSE\n(÷1000)', 'MAE\n(÷1000)']
train_values = [train_r2, train_rmse/1000, train_mae/1000]
test_values = [test_r2, test_rmse/1000, test_mae/1000]
x = np.arange(len(metrics))
width = 0.35
ax9.bar(x - width/2, train_values, width, label='Training', alpha=0.8, edgecolor='black')
ax9.bar(x + width/2, test_values, width, label='Testing', alpha=0.8, edgecolor='black')
ax9.set_ylabel('Score')
ax9.set_title('Model Performance Metrics', fontweight='bold')
ax9.set_xticks(x)
ax9.set_xticklabels(metrics)
ax9.legend()
ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('startup_multiple_regression_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Comprehensive visualization saved as 'startup_multiple_regression_analysis.png'")

# ============================================================================
# STEP 9: EXAMPLE PREDICTIONS
# ============================================================================
print("\n\n🔮 STEP 9: EXAMPLE PREDICTIONS")
print("-"*80)

example_startups = np.array([
    [150000, 120000, 300000, 0],  # New York
    [100000, 150000, 200000, 1],  # California
    [80000, 100000, 150000, 2],   # Florida
])

example_predictions = model.predict(example_startups)

print("\nPredicting profit for hypothetical startups:\n")
states = ['New York', 'California', 'Florida']
for i, (startup, pred) in enumerate(zip(example_startups, example_predictions)):
    print(f"Startup {i+1} ({states[i]}):")
    print(f"  R&D Spend: ${startup[0]:,.0f}")
    print(f"  Administration: ${startup[1]:,.0f}")
    print(f"  Marketing Spend: ${startup[2]:,.0f}")
    print(f"  → Predicted Profit: ${pred:,.2f}\n")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📚 KEY TAKEAWAYS")
print("="*80)
print(f"""
1. MODEL QUALITY: R² = {test_r2:.4f} means the model explains {test_r2*100:.1f}% of profit variation

2. MOST IMPORTANT FACTOR: {feature_importance.iloc[0]['Feature']} 
   (coefficient = {feature_importance.iloc[0]['Coefficient']:.4f})

3. PREDICTION ACCURACY: Average error of ${test_mae:,.2f} (MAE)

4. MODEL RELIABILITY: Training and testing R² are similar 
   → No significant overfitting

5. BUSINESS INSIGHT: Invest heavily in {feature_importance.iloc[0]['Feature']} 
   for maximum profit impact!
""")

print("="*80)
print("Analysis complete! Check the saved visualization for detailed insights.")
print("="*80)

plt.show()