# DAY 70

## Linear Regression: Hyperplane & Implementation

A short, human-friendly guide to understanding hyperplanes and building a complete linear regression model from EDA to predictions.

### What is a Hyperplane?

A hyperplane is a flat surface that separates or fits data in multi-dimensional space.

- **1D (line):** y = mx + b (simple line)
- **2D (plane):** z = a·x + b·y + c (flat surface)
- **3D+ (hyperplane):** extends to any number of features

**In regression:** The hyperplane is the best-fit surface that minimizes prediction error.

### Why Hyperplanes Matter

- They generalize to any number of features.
- Foundation for understanding linear regression, SVM, and neural networks.
- The more features you have, the higher the dimensional "plane."

### Full Workflow: Insurance Data Example

**Goal:** Predict insurance charges based on age, BMI, smoking status, region, etc.

#### Step 1: EDA (Exploratory Data Analysis)

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("insurance.csv")
df.head()
df.info()
df.describe()
df.isnull().sum()

# Visualize distributions
sns.histplot(df['charges'], kde=True)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
```

**Key findings:** Check for missing values, outliers, and correlations with target (charges).

#### Step 2: Data Cleaning & Preprocessing

```python
df_cleaned = df.copy()
df_cleaned.drop_duplicates(inplace=True)

# Encode categorical variables
df_cleaned['sex'] = df_cleaned['sex'].map({'male': 0, 'female': 1})
df_cleaned['smoker'] = df_cleaned['smoker'].map({'no': 0, 'yes': 1})

# One-hot encode regions
df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True)
```

#### Step 3: Feature Engineering

```python
# Create BMI categories
df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)
df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=True)
```

#### Step 4: Feature Scaling (for Linear Regression)

```python
from sklearn.preprocessing import StandardScaler

cols_to_scale = ['age', 'bmi', 'children']
scaler = StandardScaler()
df_cleaned[cols_to_scale] = scaler.fit_transform(df_cleaned[cols_to_scale])
```

**Why scale?** Helps the model train faster and improves interpretability.

#### Step 5: Feature Selection

```python
from scipy.stats import pearsonr, chi2_contingency

# Numerical: Use Pearson correlation
correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in numerical_features
}

# Categorical: Use Chi-square test
# Keep features with p-value < 0.05
```

**Result:** Keep only features that matter.

#### Step 6: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X = final_df.drop('charges', axis=1)
y = final_df['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
```

#### Step 7: Build & Train the Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
```

#### Step 8: Evaluate

```python
from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Adjusted R²
n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")
print(f"RMSE: ${rmse:.2f}")
```

### Key Takeaways

- **Hyperplane:** The decision surface in multi-dimensional space.
- **Workflow:** EDA → Clean → Engineer → Scale → Select → Split → Train → Evaluate.
- **R²:** Explains variance; higher is better (0 to 1).
- **Adjusted R²:** Penalizes too many features.
