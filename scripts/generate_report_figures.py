import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
import joblib
from joblib import load
import os

# Create output directory for report figures
os.makedirs('figures/report_figures', exist_ok=True)

print("Loading and preprocessing data...")

# Load the entire dataset
df = pd.read_csv('data/FOAI-assignment2-1.csv')

# Handle missing values with interpolation (addressing FutureWarning)
df = df.infer_objects(copy=False)
df_interpolated = df.interpolate(method='linear', axis=0, limit_direction='forward', inplace=False)

# Save interpolated data to data folder
os.makedirs('data', exist_ok=True)
df_interpolated.to_csv('data/interpolated_dataset.csv', index=False)

# Reload interpolated data
df = pd.read_csv('data/interpolated_dataset.csv')

# Set seaborn style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print(f"Dataset loaded: {len(df)} total records")
print(f"Data Scientists: {len(df[df['job_title'] == 'Data Scientist'])} records")

# Figure 1: Distribution of Salary in USD (whole dataset)
print("Generating Figure 1: Overall Salary Distribution...")
plt.figure(figsize=(10, 6))
sns.histplot(df['salary_in_usd'].dropna(), bins=30, kde=True, color='steelblue', alpha=0.7)
plt.title('Distribution of Salary in USD (All Positions)', fontsize=16, fontweight='bold')
plt.xlabel('Salary in USD', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/report_figures/figure1_salary_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Salary Distribution for Data Scientists (subset)
print("Generating Figure 2: Data Scientist Salary Distribution...")
df_ds = df[df['job_title'] == 'Data Scientist']
plt.figure(figsize=(10, 6))
sns.histplot(df_ds['salary_in_usd'].dropna(), bins=20, kde=True, color='green', alpha=0.7)
plt.title('Salary Distribution for Data Scientists', fontsize=16, fontweight='bold')
plt.xlabel('Salary in USD', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
# Add mean and median lines
mean_sal = df_ds['salary_in_usd'].mean()
median_sal = df_ds['salary_in_usd'].median()
plt.axvline(mean_sal, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_sal:,.0f}')
plt.axvline(median_sal, color='orange', linestyle='--', linewidth=2, label=f'Median: ${median_sal:,.0f}')
plt.legend()
plt.tight_layout()
plt.savefig('figures/report_figures/figure2_ds_salary_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Remote Ratio vs Salary in USD (Data Scientists)
print("Generating Figure 3: Remote Work vs Salary...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='remote_ratio', y='salary_in_usd', data=df_ds, alpha=0.6, hue='experience_level', s=60)
plt.title('Remote Work Ratio vs Salary (Data Scientists)', fontsize=16, fontweight='bold')
plt.xlabel('Remote Ratio (%)', fontsize=12)
plt.ylabel('Salary in USD', fontsize=12)
plt.legend(title='Experience Level', loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/report_figures/figure3_remote_vs_salary.png', dpi=300, bbox_inches='tight')
plt.close()

# Encode all categorical columns for correlation (whole dataset)
print("Generating Figure 4: Correlation Matrix...")
categorical_columns = df.select_dtypes(include=['object']).columns
encoded_data = df.copy()
for col in categorical_columns:
    encoded_data[col] = encoded_data[col].astype('category').cat.codes

# Select numerical and encoded categorical data (whole dataset)
numeric_data = encoded_data.select_dtypes(include=['number'])

# Figure 4: Correlation Matrix (whole dataset)
corr_matrix = numeric_data.corr(method='pearson')
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show only lower triangle
sns.heatmap(corr_matrix, annot=True, cmap="RdYlBu_r", fmt=".2f", 
           cbar_kws={'label': 'Correlation Coefficient'}, mask=mask, square=True)
plt.title('Feature Correlation Matrix (All Data)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/report_figures/figure4_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Company Size vs Salary Boxplot (Data Scientists)
print("Generating Figure 5: Company Size Analysis...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_ds, x='company_size', y='salary_in_usd', palette='Set2')
plt.title('Salary by Company Size (Data Scientists)', fontsize=16, fontweight='bold')
plt.xlabel('Company Size', fontsize=12)
plt.ylabel('Salary in USD', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.3)

# Add median annotations
for i, size in enumerate(['S', 'M', 'L']):
    if size in df_ds['company_size'].values:
        median_val = df_ds[df_ds['company_size'] == size]['salary_in_usd'].median()
        plt.text(i, median_val + 8000, f'${median_val:,.0f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('figures/report_figures/figure5_company_size_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 6: Salary by Experience Level Boxplot (Data Scientists)
print("Generating Figure 6: Experience Level Analysis...")
plt.figure(figsize=(10, 6))
experience_order = ['EN', 'MI', 'SE', 'EX']
sns.boxplot(x='experience_level', y='salary_in_usd', data=df_ds, 
           order=experience_order, palette='viridis')
plt.title('Salary Distribution by Experience Level (Data Scientists)', fontsize=16, fontweight='bold')
plt.xlabel('Experience Level', fontsize=12)
plt.ylabel('Salary (USD)', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.3)

# Add median annotations
for i, exp in enumerate(experience_order):
    if exp in df_ds['experience_level'].values:
        median_val = df_ds[df_ds['experience_level'] == exp]['salary_in_usd'].median()
        plt.text(i, median_val + 10000, f'${median_val:,.0f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('figures/report_figures/figure6_experience_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# Model Training Section (Data Scientists)
print("\nStarting model training...")
X = df_ds.drop(['salary_in_usd', 'salary', 'salary_currency', 'job_title'], axis=1)
y = df_ds['salary_in_usd']

# Handle outliers with capping
upper, lower = np.percentile(y, 99), np.percentile(y, 1)
mask = (y >= lower) & (y <= upper)
X, y = X[mask], y[mask]

print(f"After outlier removal: {len(X)} samples remaining")

categorical_features = ['experience_level', 'employment_type', 'employee_residence', 'company_location', 'company_size']
numeric_features = ['work_year', 'remote_ratio']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=2))]), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

base_models = [
    ('ridge', Ridge(alpha=0.1)),
    ('gb', GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_depth=5))
]

stack_model = StackingRegressor(estimators=base_models, final_estimator=Ridge(alpha=10.0))

pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', stack_model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training initial model...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nInitial Model Performance:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Hyperparameter Tuning
print("\nStarting hyperparameter tuning...")
param_grid = {
    'model__final_estimator__alpha': [0.1, 1.0, 10.0],
    'model__gb__learning_rate': [0.01, 0.05],
    'model__gb__max_depth': [3, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=2, verbose=1)
grid_search.fit(X_train, y_train)

final_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Save final model
os.makedirs('models', exist_ok=True)
joblib.dump(final_model, 'models/salary_model_report.pkl')
print("Model saved to models/salary_model_report.pkl")

# Final model evaluation
y_pred_final = final_model.predict(X_test)
final_r2 = r2_score(y_test, y_pred_final)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
final_mae = mean_absolute_error(y_test, y_pred_final)

print(f"\nFinal Model Performance:")
print(f"R²: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"MAE: {final_mae:.4f}")

# Figure 7: Actual vs. Predicted on Test Set
print("Generating Figure 7: Model Performance Visualization...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_final, alpha=0.6, color='blue', s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', 
         linewidth=2, label='Perfect Prediction')
plt.title(f'Actual vs. Predicted Salaries (R² = {final_r2:.3f})', fontsize=16, fontweight='bold')
plt.xlabel('Actual Salary (USD)', fontsize=12)
plt.ylabel('Predicted Salary (USD)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/report_figures/figure7_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()

# Additional Analysis Figures

# Figure 8: Experience Level and Company Size Interaction
print("Generating Figure 8: Experience vs Company Size Analysis...")
plt.figure(figsize=(12, 6))
sns.boxplot(x='experience_level', y='salary_in_usd', hue='company_size', data=df_ds, 
           order=['EN', 'MI', 'SE', 'EX'])
plt.title('Salary by Experience Level and Company Size', fontsize=16, fontweight='bold')
plt.xlabel('Experience Level', fontsize=12)
plt.ylabel('Salary (USD)', fontsize=12)
plt.legend(title='Company Size')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/report_figures/figure8_experience_company_interaction.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 9: Salary Trend Over Years
print("Generating Figure 9: Temporal Trends...")
year_stats = df_ds.groupby('work_year')['salary_in_usd'].agg(['mean', 'median', 'count']).reset_index()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(year_stats['work_year'], year_stats['mean'], marker='o', linewidth=2, markersize=8, label='Mean')
plt.plot(year_stats['work_year'], year_stats['median'], marker='s', linewidth=2, markersize=8, label='Median')
plt.title('Salary Trends (2020-2023)', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Salary (USD)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.subplot(1, 2, 2)
plt.bar(year_stats['work_year'], year_stats['count'], alpha=0.7, color='lightcoral')
plt.title('Sample Size by Year', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Records', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(year_stats['count']):
    plt.text(year_stats['work_year'].iloc[i], v + 1, str(v), 
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/report_figures/figure9_temporal_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature Importance Analysis
print("Analyzing feature importance...")
try:
    gb_model = final_model.named_steps['model'].estimators_[1][1]  # Get GradientBoosting model
    
    # Get feature names after preprocessing
    preprocessor_fitted = final_model.named_steps['prep']
    
    # Get numeric feature names (after polynomial transformation)
    numeric_transformer = preprocessor_fitted.named_transformers_['num']
    poly_features = numeric_transformer.named_steps['poly'].get_feature_names_out(numeric_features)
    
    # Get categorical feature names (after one-hot encoding)
    cat_transformer = preprocessor_fitted.named_transformers_['cat']
    cat_features = cat_transformer.get_feature_names_out(categorical_features)
    
    # Combine all feature names
    all_feature_names = list(poly_features) + list(cat_features)
    
    importances = gb_model.feature_importances_
    
    if len(all_feature_names) == len(importances):
        feat_importance = pd.DataFrame({
            'Feature': all_feature_names, 
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feat_importance.head(10))
        
        # Save feature importance
        os.makedirs('analysis', exist_ok=True)
        feat_importance.to_csv('analysis/feature_importance_report.csv', index=False)
        print("Feature importance saved to analysis/feature_importance_report.csv")
        
        # Plot top 15 features
        plt.figure(figsize=(10, 8))
        top_features = feat_importance.head(15)
        sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis')
        plt.title('Top 15 Feature Importances (Gradient Boosting)', fontsize=16, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig('figures/report_figures/figure10_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Figure 10: Feature Importance plot saved")
        
    else:
        print(f"Warning: Feature names ({len(all_feature_names)}) and importances ({len(importances)}) length mismatch.")
        
except Exception as e:
    print(f"Error in feature importance analysis: {e}")

# Print summary statistics for report
print(f"\n{'='*60}")
print("SUMMARY STATISTICS FOR DATA EXPLORATION SECTION")
print(f"{'='*60}")

print(f"\n📊 DATASET OVERVIEW:")
print(f"• Total records: {len(df):,}")
print(f"• Data Scientist records: {len(df_ds):,} ({len(df_ds)/len(df)*100:.1f}%)")
print(f"• Years covered: {df_ds['work_year'].min()}-{df_ds['work_year'].max()}")
print(f"• Countries represented: {df_ds['company_location'].nunique()}")

print(f"\n💰 SALARY STATISTICS (Data Scientists):")
print(f"• Mean salary: ${df_ds['salary_in_usd'].mean():,.0f}")
print(f"• Median salary: ${df_ds['salary_in_usd'].median():,.0f}")
print(f"• Salary range: ${df_ds['salary_in_usd'].min():,.0f} - ${df_ds['salary_in_usd'].max():,.0f}")
print(f"• Standard deviation: ${df_ds['salary_in_usd'].std():,.0f}")

print(f"\n📈 MODEL PERFORMANCE:")
print(f"• Final R² Score: {final_r2:.4f}")
print(f"• RMSE: ${final_rmse:,.0f}")
print(f"• MAE: ${final_mae:,.0f}")

print(f"\n✅ All report figures saved to: figures/report_figures/")
print("📁 Generated files:")
for i in range(1, 11):
    print(f"  • figure{i}_*.png")

print(f"\nAnalysis complete! Your Data Exploration section now has supporting visualizations.")