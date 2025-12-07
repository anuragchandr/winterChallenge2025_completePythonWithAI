# DAY 68

## Introduction to Regression & Linear Regression

A short, human-friendly guide to understanding regression, its assumptions, and how to apply linear regression in practice.

### What is Regression?

Regression is a statistical technique to model the relationship between:
- **Independent variable(s)** (features, predictors) — X
- **Dependent variable** (target, response) — y

**Goal:** Predict or understand how y changes as X changes.

### Types of Regression (quick overview)

- **Linear Regression:** y = mx + b (straight line fit)
- **Polynomial Regression:** curved relationships (y = ax² + bx + c)
- **Multiple Regression:** multiple X variables predicting one y
- **Logistic Regression:** classification (y = 0 or 1, not continuous)

### Linear Regression: The Basics

**Equation:** y = β₀ + β₁·x + ε

Where:
- β₀ = intercept (y-value when x=0)
- β₁ = slope (change in y per unit change in x)
- ε = error term (residual)

**Fit method:** Ordinary Least Squares (OLS) — minimize the sum of squared errors.

### Key Assumptions of Linear Regression

1. **Linearity:** The relationship between X and y is linear.
   - Check: scatter plot should show a roughly straight-line trend.

2. **Independence:** Observations are independent (no autocorrelation).
   - Example: daily stock prices violate this; random samples do not.

3. **Homoscedasticity:** Constant variance of errors across all X values.
   - Check: residuals plot should show even spread (no funnel shape).

4. **Normality:** Error terms are normally distributed.
   - Check: Q-Q plot or histogram of residuals should look bell-shaped.

5. **No Multicollinearity:** Predictor variables are not highly correlated (for multiple regression).
   - Check: correlation matrix; VIF (Variance Inflation Factor) < 10.

6. **No outliers or leverage points:** Extreme values can distort the fit.
   - Check: residuals plot for large outliers.

### Simple Linear Regression Workflow

````python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 6])

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Coefficients
print(f"Intercept (β₀): {model.intercept_}")
print(f"Slope (β₁): {model.coef_[0]}")
print(f"R² Score: {r2_score(y, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred))}")

# Plot
plt.scatter(X, y, label='Actual')
plt.plot(X, y_pred, 'r-', label='Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
````