import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
from scipy import stats

# Load the dataset
df = pd.read_csv('FOAI-assignment2-1.csv')

# Filter to Data Scientist roles
df_ds = df[df['job_title'] == 'Data Scientist']

# Handle missing values with interpolation
df_ds_interpolated = df_ds.interpolate(method='linear', axis=0, limit_direction='forward', inplace=False)
df_ds_interpolated.to_csv('data_scientist_interpolated.csv', index=False)

# Reload interpolated data
df = pd.read_csv('data_scientist_interpolated.csv')

# Set seaborn style
sns.set(style="whitegrid")

# Figure 1: Distribution of Salary in USD
plt.figure(figsize=(10, 6))
sns.histplot(df['salary_in_usd'].dropna(), bins=30, kde=True, color='blue')
plt.title('Distribution of Salary in USD', fontsize=16)
plt.xlabel('Salary in USD', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig('figure1_salary_distribution.png')
plt.close()

# Figure 2: Salary Distribution for Data Scientists
plt.figure(figsize=(10, 6))
sns.histplot(df['salary_in_usd'].dropna(), bins=20, kde=True, color='green')
plt.title('Salary Distribution for Data Scientists', fontsize=16)
plt.xlabel('Salary in USD', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig('figure2_ds_salary_distribution.png')
plt.close()

# Figure 3: Remote Ratio vs Salary in USD
plt.figure(figsize=(10, 6))
sns.scatterplot(x='remote_ratio', y='salary_in_usd', data=df, alpha=0.6, hue='experience_level')
plt.title('Remote Ratio vs Salary in USD', fontsize=16)
plt.xlabel('Remote Ratio (%)', fontsize=12)
plt.ylabel('Salary in USD', fontsize=12)
plt.legend(title='Experience Level', loc='upper left')
plt.savefig('figure3_remote_vs_salary.png')
plt.close()

# Encode categorical for correlation
categorical_columns = df.select_dtypes(include=['object']).columns
encoded_data = df.copy()
for col in categorical_columns:
    encoded_data[col] = encoded_data[col].astype('category').cat.codes

numeric_data = encoded_data.select_dtypes(include=['number'])

corr_matrix = numeric_data.corr(method='pearson')

# Figure 4: Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Pearson Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('figure4_correlation_matrix.png')
plt.close()

# Figure 5: Company Size vs Salary Boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='company_size', y='salary_in_usd', palette='Set2')
plt.title('Company Size vs Salary in USD')
plt.xlabel('Company Size')
plt.ylabel('Salary in USD')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.savefig('figure5_company_size_boxplot.png')
plt.close()

# Figure 6: Salary by Experience Level Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='experience_level', y='salary_in_usd', data=df, palette='Set2')
plt.title('Salary Distribution by Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Salary (USD)')
plt.savefig('figure6_experience_boxplot.png')
plt.close()

# Model Training Section - Enhanced Feature Engineering
X = df.drop(['salary_in_usd', 'salary', 'salary_currency', 'job_title'], axis=1)
y = df['salary_in_usd']

# Enhanced outlier handling using IQR method
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
mask = (y >= lower_bound) & (y <= upper_bound)
X, y = X[mask], y[mask]

print(f"Data shape after outlier removal: {X.shape}")
print(f"Outliers removed: {len(df) - len(X)}")

# Log transform the target for better distribution
y_log = np.log1p(y)

categorical_features = ['experience_level', 'employment_type', 'employee_residence', 'company_location', 'company_size']
numeric_features = ['work_year', 'remote_ratio']

# Enhanced preprocessing with feature selection
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('scaler', RobustScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('selector', SelectKBest(f_regression, k=15))
    ]), numeric_features),
    ('cat', Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('selector', SelectKBest(f_regression, k=20))
    ]), categorical_features)
])

# Enhanced ensemble with more diverse models
base_models = [
    ('ridge', Ridge(alpha=1.0)),
    ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5)),
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)),
    ('et', ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42))
]

# Meta-learner for stacking
meta_learner = Ridge(alpha=0.1)
stack_model = StackingRegressor(
    estimators=base_models, 
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', stack_model)
])

# Enhanced train-test split with stratification based on salary quantiles
y_quantiles = pd.qcut(y_log, q=4, labels=False)
X_train, X_test, y_train, y_test, y_train_quantiles, y_test_quantiles = train_test_split(
    X, y_log, y_quantiles, test_size=0.2, random_state=42, stratify=y_quantiles
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Cross-validation before final training
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean CV R² score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Predictions (convert back from log scale)
y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Convert back from log scale
y_test_original = np.expm1(y_test)  # Convert back from log scale

# Clip predictions to reasonable range
y_pred = np.clip(y_pred, 20000, 500000)

mse = mean_squared_error(y_test_original, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred)
mae = mean_absolute_error(y_test_original, y_pred)

print(f"\n=== Model Performance ===")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Additional metrics
mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
print(f"MAPE: {mape:.2f}%")

# Enhanced Hyperparameter Tuning with more comprehensive grid
param_grid = {
    'prep__num__selector__k': [10, 15, 20],
    'prep__cat__selector__k': [15, 20, 25],
    'model__final_estimator__alpha': [0.01, 0.1, 1.0],
    'model__rf__n_estimators': [100, 200],
    'model__rf__max_depth': [8, 10, 12],
    'model__gb__learning_rate': [0.05, 0.1],
    'model__gb__n_estimators': [150, 200],
    'model__gb__max_depth': [5, 6]
}

print("\n=== Starting Hyperparameter Tuning ===")
print("This may take several minutes...")

# Use fewer CV folds and limit n_jobs to prevent memory issues
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=3,  # Reduced from 5 to 3 for faster execution
    scoring='r2',
    n_jobs=2,  # Limited parallelism
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

final_model = grid_search.best_estimator_

# Final model evaluation
final_y_pred_log = final_model.predict(X_test)
final_y_pred = np.expm1(final_y_pred_log)
final_y_pred = np.clip(final_y_pred, 20000, 500000)

final_mse = mean_squared_error(y_test_original, final_y_pred)
final_rmse = np.sqrt(final_mse)
final_r2 = r2_score(y_test_original, final_y_pred)
final_mae = mean_absolute_error(y_test_original, final_y_pred)
final_mape = np.mean(np.abs((y_test_original - final_y_pred) / y_test_original)) * 100

print(f"\n=== Final Tuned Model Performance ===")
print(f"MSE: {final_mse:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"MAE: {final_mae:.4f}")
print(f"R²: {final_r2:.4f}")
print(f"MAPE: {final_mape:.2f}%")

# Save the best model
joblib.dump(final_model, 'salary_model_optimized.pkl')
print("Optimized model saved as 'salary_model_optimized.pkl'")


# Figure 7,8,10: Predictions Visualization - Updated to use optimized model
try:
    # Create a test prediction plot using actual test data
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Predicted vs Actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_original, final_y_pred, alpha=0.6, color='blue')
    plt.plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
    plt.xlabel('Actual Salary (USD)')
    plt.ylabel('Predicted Salary (USD)')
    plt.title('Actual vs Predicted Salaries')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Residuals plot
    plt.subplot(1, 2, 2)
    residuals = y_test_original - final_y_pred
    plt.scatter(final_y_pred, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Salary (USD)')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure_model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Model evaluation plots saved.")
    
except Exception as e:
    print(f"Plotting failed: {e}")

# Feature importance with error handling
try:
    # Get feature importance from the best Random Forest in the ensemble
    rf_model = None
    gb_model = None
    
    for name, model in final_model.named_steps['model'].named_estimators_.items():
        if 'rf' in name and hasattr(model, 'feature_importances_'):
            rf_model = model
        elif 'gb' in name and hasattr(model, 'feature_importances_'):
            gb_model = model
    
    if rf_model is not None:
        # Get the preprocessor from the final model
        preprocessor_fitted = final_model.named_steps['prep']
        
        # Try to get feature names
        try:
            feature_names = preprocessor_fitted.get_feature_names_out()
            importances = rf_model.feature_importances_
            
            if len(feature_names) == len(importances):
                # Create feature importance DataFrame
                feat_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(20)
                
                # Plot feature importance
                plt.figure(figsize=(12, 8))
                sns.barplot(data=feat_importance, y='Feature', x='Importance', palette='viridis')
                plt.title('Top 20 Feature Importances (Random Forest)', fontsize=14)
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save to CSV
                feat_importance.to_csv('feature_importance_optimized.csv', index=False)
                print("Feature importance analysis completed and saved.")
            else:
                print(f"Feature name mismatch: {len(feature_names)} vs {len(importances)}")
        except Exception as e:
            print(f"Could not extract feature names: {e}")
    else:
        print("No Random Forest model found for feature importance.")
        
except Exception as e:
    print(f"Feature importance extraction failed: {e}")

# Model comparison summary
print(f"\n=== Model Performance Summary ===")
print(f"Initial Model R²: {r2:.4f}")
print(f"Optimized Model R²: {final_r2:.4f}")
print(f"Improvement: {(final_r2 - r2):.4f}")
print(f"RMSE Reduction: ${rmse - final_rmse:.0f}")

print("\n=== Script completed successfully! ===")
print("Files generated:")
print("- salary_model_optimized.pkl (optimized model)")
print("- feature_importance_optimized.csv (feature importance)")
print("- figure_model_evaluation.png (model evaluation plots)")
print("- feature_importance_plot.png (feature importance visualization)")
print("- All original visualization figures")