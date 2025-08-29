import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv('expanded_synthetic_daily_expenses.csv')

df = load_data()

# Data Processing
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['DayOfYear'] = df['Date'].dt.dayofyear
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

df = pd.get_dummies(df, columns=['Category', 'City', 'Place'], drop_first=True)
X = df.drop(columns=['Amount', 'Date', 'Time'])
y = df['Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Streamlit App Layout
st.title("Daily Expenses Prediction Dashboard")
st.markdown("Analyze and predict daily expenses with advanced models.")

# Sidebar options
st.sidebar.header("Model Parameters")

# Model Selection
model_option = st.sidebar.selectbox(
    "Choose Model",
    ("Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression", "Random Forest")
)

# Model Parameters
if model_option == "Polynomial Regression":
    degree = st.sidebar.slider("Degree of Polynomial", min_value=2, max_value=5, value=2)
if model_option in ["Ridge Regression", "Lasso Regression"]:
    alpha = st.sidebar.slider("Regularization Parameter (Alpha)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
if model_option == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", min_value=10, max_value=200, value=100, step=10)

# Model Training and Prediction
if model_option == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

elif model_option == "Polynomial Regression":
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

elif model_option == "Ridge Regression":
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

elif model_option == "Lasso Regression":
    model = Lasso(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

elif model_option == "Random Forest":
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader(f"{model_option} Evaluation Metrics")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Plot: Actual vs Predicted Expenses
st.subheader("Actual vs Predicted Expenses")
fig = go.Figure()
fig.add_trace(go.Scatter(y=y_test, mode='markers', name='Actual', marker=dict(color='blue')))
fig.add_trace(go.Scatter(y=y_pred, mode='markers', name='Predicted', marker=dict(color='red')))
fig.update_layout(title="Actual vs Predicted Daily Expenses", xaxis_title="Sample Index", yaxis_title="Expense Amount")
st.plotly_chart(fig)

# Monthly Expense Bar Chart
df['Prediction'] = model.predict(scaler.transform(X) if model_option != "Random Forest" else model.predict(X))
monthly_expense = df.groupby(df['Date'].dt.month)['Prediction'].sum()
st.subheader("Predicted Monthly Expenses")
fig_monthly = px.bar(x=monthly_expense.index, y=monthly_expense.values, labels={'x': 'Month', 'y': 'Predicted Monthly Expense ($)'})
fig_monthly.update_layout(title="Predicted Monthly Expenses", xaxis=dict(tickmode="array", tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']))
st.plotly_chart(fig_monthly)

# Feature Importance (only for Random Forest)
if model_option == "Random Forest":
    st.subheader("Feature Importance (Random Forest)")
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_importance = px.bar(feature_importance, labels={'index': 'Features', 'value': 'Importance Score'})
    fig_importance.update_layout(title="Feature Importance in Random Forest Model")
    st.plotly_chart(fig_importance)

st.markdown("This dashboard provides an interactive experience for exploring different predictive models and understanding key factors driving daily expenses.")
