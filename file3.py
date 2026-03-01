"""

Complete Feature Engineering & Preprocessing Example

Dataset: Bike Sharing Demand

Goal: Predict hourly bike rental count


This demonstrates:

- Data loading and exploration

- Missing value handling

- Outlier detection

- Feature scaling

- Categorical encoding

- Date/time feature engineering

- Interaction features

- Polynomial features

- Feature selection

- Model comparison

"""


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.feature_selection import SelectKBest, f_regression, RFE

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import warnings

warnings.filterwarnings('ignore')


# ============================================================================

# 1. LOAD AND EXPLORE DATA

# ============================================================================


# Load bike sharing dataset

# Download from: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset

# For this example, we'll create sample data similar to the real dataset


np.random.seed(42)


# Create sample data (in practice, load real CSV with pd.read_csv('hour.csv'))

n_samples = 5000

data = pd.DataFrame({

'datetime': pd.date_range('2011-01-01', periods=n_samples, freq='H'),

'season': np.random.choice([1, 2, 3, 4], n_samples), # 1:spring, 2:summer, 3:fall, 4:winter

'holiday': np.random.choice([0, 1], n_samples, p=[0.97, 0.03]),

'workingday': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),

'weather': np.random.choice([1, 2, 3, 4], n_samples, p=[0.7, 0.2, 0.08, 0.02]),

'temp': np.random.uniform(0, 41, n_samples), # Temperature in Celsius

'atemp': np.random.uniform(0, 45, n_samples), # "Feels like" temperature

'humidity': np.random.uniform(0, 100, n_samples),

'windspeed': np.random.uniform(0, 67, n_samples),

})


# Create target variable (count) with some relationship to features

data['count'] = (

100 + 

data['temp'] * 8 +

data['atemp'] * 5 -

data['humidity'] * 2 +

data['workingday'] * 50 +

(data['season'] == 3) * 100 + # Fall has more rentals

np.random.normal(0, 50, n_samples)

).clip(0)


print("="*80)

print("DATASET OVERVIEW")

print("="*80)

print(f"\nDataset shape: {data.shape}")

print(f"\nFirst few rows:")

print(data.head())

print(f"\nData types:")

print(data.dtypes)

print(f"\nBasic statistics:")

print(data.describe())


# ============================================================================

# 2. MISSING VALUE ANALYSIS

# ============================================================================


print("\n" + "="*80)

print("MISSING VALUE HANDLING")

print("="*80)


# Introduce some missing values for demonstration

data_with_missing = data.copy()

missing_indices = np.random.choice(data_with_missing.index, size=100, replace=False)

data_with_missing.loc[missing_indices, 'temp'] = np.nan

data_with_missing.loc[missing_indices[:50], 'humidity'] = np.nan


print(f"\nMissing values:")

print(data_with_missing.isnull().sum())

print(f"\nMissing percentage:")

print((data_with_missing.isnull().sum() / len(data_with_missing) * 100).round(2))


# Strategy 1: Simple imputation

imputer_mean = SimpleImputer(strategy='mean')

data_with_missing['temp_mean_imputed'] = imputer_mean.fit_transform(

data_with_missing[['temp']]

)


# Strategy 2: Median (better for skewed data)

imputer_median = SimpleImputer(strategy='median')

data_with_missing['temp_median_imputed'] = imputer_median.fit_transform(

data_with_missing[['temp']]

)


# Strategy 3: KNN imputation (considers relationships)

imputer_knn = KNNImputer(n_neighbors=5)

data_with_missing[['temp_knn_imputed', 'humidity_knn_imputed']] = imputer_knn.fit_transform(

data_with_missing[['temp', 'humidity']]

)


print(f"\nComparison of imputation methods (first 5 missing values):")

comparison = data_with_missing[data_with_missing['temp'].isnull()].head()[

['temp', 'temp_mean_imputed', 'temp_median_imputed', 'temp_knn_imputed']

]

print(comparison)


# Use the original data without missing values for rest of analysis

data = data.copy()


# ============================================================================

# 3. OUTLIER DETECTION

# ============================================================================


print("\n" + "="*80)

print("OUTLIER DETECTION")

print("="*80)


def detect_outliers_iqr(df, column):

"""Detect outliers using IQR method"""

Q1 = df[column].quantile(0.25)

Q3 = df[column].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

return outliers, lower_bound, upper_bound


def detect_outliers_zscore(df, column, threshold=3):

"""Detect outliers using Z-score method"""

z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())

outliers = df[z_scores > threshold]

return outliers


# Detect outliers in 'count'

outliers_iqr, lower, upper = detect_outliers_iqr(data, 'count')

outliers_zscore = detect_outliers_zscore(data, 'count')


print(f"\nOutliers in 'count' using IQR method: {len(outliers_iqr)} ({len(outliers_iqr)/len(data)*100:.2f}%)")

print(f"IQR bounds: [{lower:.2f}, {upper:.2f}]")

print(f"\nOutliers in 'count' using Z-score method: {len(outliers_zscore)} ({len(outliers_zscore)/len(data)*100:.2f}%)")


# Treatment: Cap outliers (Winsorization)

data['count_capped'] = data['count'].clip(lower=lower, upper=upper)


print(f"\nOriginal count range: [{data['count'].min():.2f}, {data['count'].max():.2f}]")

print(f"Capped count range: [{data['count_capped'].min():.2f}, {data['count_capped'].max():.2f}]")


# ============================================================================

# 4. DATE/TIME FEATURE ENGINEERING

# ============================================================================


print("\n" + "="*80)

print("DATE/TIME FEATURE ENGINEERING")

print("="*80)


# Extract basic date components

data['year'] = data['datetime'].dt.year

data['month'] = data['datetime'].dt.month

data['day'] = data['datetime'].dt.day

data['hour'] = data['datetime'].dt.hour

data['dayofweek'] = data['datetime'].dt.dayofweek # 0=Monday, 6=Sunday

data['quarter'] = data['datetime'].dt.quarter

data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)


# Cyclical encoding for hour (0-23)

data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)

data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)


# Cyclical encoding for month (1-12)

data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)

data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)


# Cyclical encoding for day of week (0-6)

data['dayofweek_sin'] = np.sin(2 * np.pi * data['dayofweek'] / 7)

data['dayofweek_cos'] = np.cos(2 * np.pi * data['dayofweek'] / 7)


# Peak hours (rush hours)

data['is_rush_hour'] = data['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)


# Time of day categories

data['time_of_day'] = pd.cut(data['hour'], 

bins=[0, 6, 12, 18, 24], 

labels=['Night', 'Morning', 'Afternoon', 'Evening'],

include_lowest=True)


print(f"\nNew time-based features created:")

time_features = ['year', 'month', 'hour', 'dayofweek', 'is_weekend', 

'hour_sin', 'hour_cos', 'is_rush_hour', 'time_of_day']

print(data[time_features].head())


# ============================================================================

# 5. CATEGORICAL ENCODING

# ============================================================================


print("\n" + "="*80)

print("CATEGORICAL ENCODING")

print("="*80)


# One-Hot Encoding for 'season'

data_encoded = pd.get_dummies(data, columns=['season'], prefix='season', drop_first=True)

print(f"\nOne-hot encoded 'season':")

print(data_encoded[[col for col in data_encoded.columns if 'season' in col]].head())


# Label Encoding for 'weather' (ordinal: 1=best, 4=worst)

data_encoded['weather_label'] = data['weather']


# One-Hot Encoding for 'time_of_day'

data_encoded = pd.get_dummies(data_encoded, columns=['time_of_day'], prefix='time', drop_first=True)


print(f"\nOne-hot encoded 'time_of_day':")

print(data_encoded[[col for col in data_encoded.columns if 'time_' in col]].head())


# Frequency encoding for 'hour'

hour_freq = data['hour'].value_counts() / len(data)

data_encoded['hour_frequency'] = data['hour'].map(hour_freq)


print(f"\nFrequency encoded 'hour':")

print(data_encoded[['hour', 'hour_frequency']].head(10))


# ============================================================================

# 6. NUMERICAL FEATURE TRANSFORMATIONS

# ============================================================================


print("\n" + "="*80)

print("NUMERICAL TRANSFORMATIONS")

print("="*80)


# Log transformation for skewed features

data_encoded['count_log'] = np.log1p(data_encoded['count']) # log(1 + x) to handle zeros


# Square root transformation

data_encoded['count_sqrt'] = np.sqrt(data_encoded['count'])


# Binning temperature

data_encoded['temp_bin'] = pd.cut(data['temp'], 

bins=[0, 10, 20, 30, 41], 

labels=['Cold', 'Cool', 'Warm', 'Hot'])


# Convert to dummy variables

data_encoded = pd.get_dummies(data_encoded, columns=['temp_bin'], prefix='temp', drop_first=True)


print(f"\nTransformed count variable:")

print(data_encoded[['count', 'count_log', 'count_sqrt']].describe())


# ============================================================================

# 7. INTERACTION FEATURES

# ============================================================================


print("\n" + "="*80)

print("INTERACTION FEATURES")

print("="*80)


# Temperature × Working Day (temp matters more on working days)

data_encoded['temp_workingday'] = data_encoded['temp'] * data_encoded['workingday']


# Temperature × Hour (temp effect varies by time)

data_encoded['temp_hour'] = data_encoded['temp'] * data_encoded['hour']


# Humidity × Temperature (discomfort index)

data_encoded['humidity_temp'] = data_encoded['humidity'] * data_encoded['temp']


# Weekend × Hour (different patterns on weekends)

data_encoded['weekend_hour'] = data_encoded['is_weekend'] * data_encoded['hour']


# Weather × Temperature

data_encoded['weather_temp'] = data_encoded['weather_label'] * data_encoded['temp']


print(f"\nInteraction features created:")

interaction_features = ['temp_workingday', 'temp_hour', 'humidity_temp', 

'weekend_hour', 'weather_temp']

print(data_encoded[interaction_features].describe())


# ============================================================================

# 8. POLYNOMIAL FEATURES

# ============================================================================


print("\n" + "="*80)

print("POLYNOMIAL FEATURES")

print("="*80)


# Create polynomial features for key variables

data_encoded['temp_squared'] = data_encoded['temp'] ** 2

data_encoded['temp_cubed'] = data_encoded['temp'] ** 3

data_encoded['humidity_squared'] = data_encoded['humidity'] ** 2


print(f"\nPolynomial features:")

print(data_encoded[['temp', 'temp_squared', 'temp_cubed']].head())


# ============================================================================

# 9. AGGREGATION FEATURES

# ============================================================================


print("\n" + "="*80)

print("AGGREGATION FEATURES")

print("="*80)


# Rolling statistics (moving average)

data_encoded = data_encoded.sort_values('datetime')

data_encoded['count_rolling_mean_3h'] = data_encoded['count'].rolling(window=3, min_periods=1).mean()

data_encoded['count_rolling_std_3h'] = data_encoded['count'].rolling(window=3, min_periods=1).std()

data_encoded['count_rolling_max_3h'] = data_encoded['count'].rolling(window=3, min_periods=1).max()


# Lag features (previous hour's count)

data_encoded['count_lag_1h'] = data_encoded['count'].shift(1)

data_encoded['count_lag_24h'] = data_encoded['count'].shift(24) # Same hour yesterday


# Fill NaN from rolling/lag with 0

data_encoded = data_encoded.fillna(0)


print(f"\nTime series features:")

print(data_encoded[['count', 'count_rolling_mean_3h', 'count_lag_1h', 'count_lag_24h']].head(30))


# ============================================================================

# 10. PREPARE FOR MODELING

# ============================================================================


print("\n" + "="*80)

print("PREPARE FOR MODELING")

print("="*80)


# Select features for modeling

feature_cols = [col for col in data_encoded.columns if col not in 

['datetime', 'count', 'count_log', 'count_sqrt', 'count_capped']]


X = data_encoded[feature_cols]

y = data_encoded['count']


print(f"\nTotal features created: {len(feature_cols)}")

print(f"Feature names: {feature_cols[:20]}... (showing first 20)")


# Train-test split (BEFORE any scaling!)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(f"\nTrain set size: {X_train.shape}")

print(f"Test set size: {X_test.shape}")


# ============================================================================

# 11. FEATURE SCALING

# ============================================================================


print("\n" + "="*80)

print("FEATURE SCALING")

print("="*80)


# Standardization (Z-score)

scaler_standard = StandardScaler()

X_train_standard = scaler_standard.fit_transform(X_train)

X_test_standard = scaler_standard.transform(X_test)


# Min-Max Scaling

scaler_minmax = MinMaxScaler()

X_train_minmax = scaler_minmax.fit_transform(X_train)

X_test_minmax = scaler_minmax.transform(X_test)


# Robust Scaling (good with outliers)

scaler_robust = RobustScaler()

X_train_robust = scaler_robust.fit_transform(X_train)

X_test_robust = scaler_robust.transform(X_test)


print(f"\nOriginal data statistics (first feature):")

print(f"Mean: {X_train.iloc[:, 0].mean():.2f}, Std: {X_train.iloc[:, 0].std():.2f}")

print(f"Min: {X_train.iloc[:, 0].min():.2f}, Max: {X_train.iloc[:, 0].max():.2f}")


print(f"\nStandardized data statistics (first feature):")

print(f"Mean: {X_train_standard[:, 0].mean():.2f}, Std: {X_train_standard[:, 0].std():.2f}")

print(f"Min: {X_train_standard[:, 0].min():.2f}, Max: {X_train_standard[:, 0].max():.2f}")


# ============================================================================

# 12. FEATURE SELECTION

# ============================================================================


print("\n" + "="*80)

print("FEATURE SELECTION")

print("="*80)


# Method 1: SelectKBest with F-statistic

selector_kbest = SelectKBest(score_func=f_regression, k=20)

X_train_kbest = selector_kbest.fit_transform(X_train, y_train)

X_test_kbest = selector_kbest.transform(X_test)


# Get selected feature names

selected_features_kbest = X.columns[selector_kbest.get_support()].tolist()

print(f"\nTop 20 features (SelectKBest):")

print(selected_features_kbest)


# Method 2: Feature importance from Random Forest

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

rf.fit(X_train, y_train)


feature_importance = pd.DataFrame({

'feature': X.columns,

'importance': rf.feature_importances_

}).sort_values('importance', ascending=False)


print(f"\nTop 20 features (Random Forest Importance):")

print(feature_importance.head(20))


# Method 3: L1 Regularization (Lasso)

lasso = Lasso(alpha=0.1, random_state=42)

lasso.fit(X_train_standard, y_train)


lasso_features = pd.DataFrame({

'feature': X.columns,

'coefficient': np.abs(lasso.coef_)

}).sort_values('coefficient', ascending=False)


selected_features_lasso = lasso_features[lasso_features['coefficient'] > 0]['feature'].tolist()

print(f"\nFeatures selected by Lasso: {len(selected_features_lasso)}")

print(selected_features_lasso[:20])


# ============================================================================

# 13. MODEL COMPARISON

# ============================================================================


print("\n" + "="*80)

print("MODEL COMPARISON")

print("="*80)


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):

"""Train and evaluate a model"""

model.fit(X_train, y_train)


# Predictions

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)


# Metrics

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_r2 = r2_score(y_train, y_train_pred)

test_r2 = r2_score(y_test, y_test_pred)

train_mae = mean_absolute_error(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)


print(f"\n{model_name}:")

print(f" Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")

print(f" Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")

print(f" Train MAE: {train_mae:.2f} | Test MAE: {test_mae:.2f}")


return test_r2, test_rmse


results = {}


# Baseline: Linear Regression with all features (unscaled)

lr = LinearRegression()

results['Linear Regression (All Features)'] = evaluate_model(

lr, X_train, X_test, y_train, y_test, 

"Linear Regression (All Features)"

)


# Linear Regression with standardized features

lr_standard = LinearRegression()

results['Linear Regression (Standardized)'] = evaluate_model(

lr_standard, X_train_standard, X_test_standard, y_train, y_test,

"Linear Regression (Standardized)"

)


# Ridge Regression (L2 regularization)

ridge = Ridge(alpha=1.0)

results['Ridge Regression'] = evaluate_model(

ridge, X_train_standard, X_test_standard, y_train, y_test,

"Ridge Regression (Standardized)"

)


# Lasso Regression (L1 regularization + feature selection)

lasso_model = Lasso(alpha=0.1)

results['Lasso Regression'] = evaluate_model(

lasso_model, X_train_standard, X_test_standard, y_train, y_test,

"Lasso Regression (Standardized)"

)


# Linear Regression with selected features (SelectKBest)

lr_kbest = LinearRegression()

results['Linear Regression (SelectKBest)'] = evaluate_model(

lr_kbest, X_train_kbest, X_test_kbest, y_train, y_test,

"Linear Regression (SelectKBest 20 features)"

)


# Random Forest (doesn't need scaling)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)

results['Random Forest'] = evaluate_model(

rf_model, X_train, X_test, y_train, y_test,

"Random Forest"

)


# ============================================================================

# 14. SUMMARY

# ============================================================================


print("\n" + "="*80)

print("SUMMARY OF TECHNIQUES DEMONSTRATED")

print("="*80)


techniques = """

✓ Missing Value Handling:

- Mean imputation

- Median imputation

- KNN imputation


✓ Outlier Detection:

- IQR method

- Z-score method

- Winsorization (capping)


✓ Date/Time Features:

- Basic extraction (year, month, hour, etc.)

- Cyclical encoding (sin/cos)

- Time-based categories (rush hour, weekend, time of day)


✓ Categorical Encoding:

- One-hot encoding

- Label encoding

- Frequency encoding


✓ Numerical Transformations:

- Log transformation

- Square root transformation

- Binning/discretization


✓ Feature Engineering:

- Interaction features (temp × workingday, etc.)

- Polynomial features (squared, cubed)

- Rolling statistics (moving averages)

- Lag features (previous values)


✓ Feature Scaling:

- Standardization (Z-score)

- Min-Max scaling

- Robust scaling


✓ Feature Selection:

- SelectKBest (statistical)

- Random Forest importance

- L1 regularization (Lasso)


✓ Model Comparison:

- Linear Regression

- Ridge Regression (L2)

- Lasso Regression (L1)

- Random Forest

"""


print(techniques)


print("\n" + "="*80)

print("KEY TAKEAWAYS")

print("="*80)


print("""

1. Always split data BEFORE preprocessing to avoid leakage

2. Fit scalers/imputers only on training data

3. Feature engineering can dramatically improve performance

4. Tree-based models don't need scaling but benefit from good features

5. Regularization helps with high-dimensional data

6. Domain knowledge guides the best feature engineering

7. Always validate on held-out test data

8. Monitor for overfitting (train vs test performance)

""")


print("\n" + "="*80)

print("BEST MODEL PERFORMANCE")

print("="*80)


best_model = max(results.items(), key=lambda x: x[1][0])

print(f"\nBest Model: {best_model[0]}")

print(f"Test R²: {best_model[1][0]:.4f}")

print(f"Test RMSE: {best_model[1][1]:.2f}")


print("\n" + "="*80)

print("EXERCISE COMPLETE!")

print("="*80)

print("\nYou now have a complete pipeline demonstrating:")

print("- Data preprocessing")

print("- Feature engineering") 

print("- Feature selection")

print("- Model training and evaluation")

print("\nTry modifying features and parameters to improve performance!")