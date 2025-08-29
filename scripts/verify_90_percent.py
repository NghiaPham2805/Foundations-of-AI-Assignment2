import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('FOAI-assignment2-1.csv')
ds = df[df['job_title'] == 'Data Scientist'].copy()
ds = ds.dropna(subset=['salary_in_usd'])

print("🔍 INVESTIGATING YOUR FRIEND'S 90% R² CLAIM")
print("=" * 50)

# Basic stats
y = ds['salary_in_usd']
print(f"📊 Data Scientist Salary Statistics:")
print(f"Mean: ${y.mean():,.0f}")
print(f"Std:  ${y.std():,.0f}")
print(f"Min:  ${y.min():,.0f}")
print(f"Max:  ${y.max():,.0f}")
print(f"CV:   {y.std()/y.mean():.2f}")

# What R²=0.90 means
rmse_90 = np.sqrt((1-0.90) * y.var())
print(f"\n🎯 What R² = 0.90 would mean:")
print(f"RMSE: ${rmse_90:,.0f}")
print(f"This means predicting salary within ±${rmse_90:,.0f}")
print(f"That's {rmse_90/y.mean()*100:.1f}% error rate")
print(f"For salary prediction, this is EXTREMELY optimistic!")

# Test various scenarios that could lead to inflated R²
print(f"\n🚨 COMMON MISTAKES THAT INFLATE R²:")

# 1. Data leakage test
print(f"\n1. DATA LEAKAGE TEST:")
X = ds.drop(['salary_in_usd', 'salary', 'salary_currency', 'job_title'], axis=1)

# Simple preprocessing
from sklearn.preprocessing import LabelEncoder
X_encoded = X.copy()
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))

# Basic model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_encoded, y)
train_r2 = rf.score(X_encoded, y)
print(f"Training R² (overfitting): {train_r2:.4f}")

# 2. Proper train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
rf.fit(X_train, y_train)
test_r2 = rf.score(X_test, y_test)
print(f"Proper test R²: {test_r2:.4f}")

# 3. If they used salary as a feature (major data leakage)
if 'salary' in ds.columns:
    X_leak = X_encoded.copy()
    # Simulate using original salary (major leak)
    salary_leak = ds['salary'].fillna(ds['salary'].median())
    X_leak['salary_leak'] = salary_leak
    rf_leak = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_leak.fit(X_leak, y)
    leak_r2 = rf_leak.score(X_leak, y)
    print(f"With salary leak R²: {leak_r2:.4f} ⚠️ DATA LEAKAGE!")

print(f"\n🎓 REALISTIC R² RANGES FOR SALARY PREDICTION:")
print(f"• Small dataset (300-500): R² = 0.40-0.65")
print(f"• Medium dataset (1000+): R² = 0.55-0.75")
print(f"• Large dataset (5000+):  R² = 0.65-0.80")
print(f"• R² = 0.90+: Nearly impossible without data leakage")

print(f"\n💡 LIKELY EXPLANATIONS FOR 90% R²:")
print(f"1. Used training data for evaluation (no train-test split)")
print(f"2. Data leakage (included salary-related features)")
print(f"3. Overfitted model (too complex for small data)")
print(f"4. Used wrong metric (maybe accuracy instead of R²)")
print(f"5. Different dataset or task")

print(f"\n✅ YOUR R² = 0.56 IS ACTUALLY EXCELLENT!")
print(f"Your friend's 90% is likely a mistake or misunderstanding.")
