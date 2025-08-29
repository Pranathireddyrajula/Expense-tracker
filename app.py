import pandas as pd
import numpy as np
import streamlit as st
import joblib  # To save and load models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # For interactive plots

# Set Streamlit page configurations
st.set_page_config(page_title="Expense Prediction", layout="wide")

# Title and description
st.title("Daily Expense Prediction App")
st.write("This app analyzes and predicts daily expenses based on a dataset you upload. "
         "Ensure your CSV contains the following columns: **Date**, **Amount**, and any relevant categorical features (e.g., Category, City, Place).")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Data validation
    required_columns = ['Date', 'Amount']
    if not all(col in df.columns for col in required_columns):
        st.error("CSV file must contain the following columns: Date, Amount.")
        st.stop()

    st.success("CSV file successfully uploaded!")
    st.write("### Full Dataset")
    st.dataframe(df)

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        st.write("### Missing Values")
        st.write(df.isnull().sum())

    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if df['Date'].isnull().all():
        st.error("Date parsing failed. Please check your Date column format.")
        st.stop()

    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    # Handle categorical columns safely
    categorical_columns = [col for col in ['Category', 'City', 'Place'] if col in df.columns]
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Features and target
    X = df.drop(columns=['Amount', 'Date', 'Time'], errors='ignore')
    y = df['Amount']

    # Train/test split
    test_size = st.slider("Select Test Size (%)", 10, 50, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=st.slider("Ridge Alpha", 0.01, 10.0, 1.0)),
        "Lasso Regression": Lasso(alpha=st.slider("Lasso Alpha", 0.01, 10.0, 1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=int(st.slider("Random Forest Estimators", 10, 200, 100)), random_state=42)
    }

    selected_model_name = st.selectbox("Select Model", list(models.keys()))

    # Polynomial features (only for linear models)
    if selected_model_name != "Random Forest":
        poly_degree = st.slider("Select Polynomial Degree", 1, 5, 2)
        poly = PolynomialFeatures(degree=poly_degree)
        X_train_trans = poly.fit_transform(X_train_scaled)
        X_test_trans = poly.transform(X_test_scaled)
    else:
        poly = None
        X_train_trans, X_test_trans = X_train_scaled, X_test_scaled

    # Train model
    model = models[selected_model_name]
    model.fit(X_train_trans, y_train)
    y_pred = model.predict(X_test_trans)

    # Metrics
    st.write(f"### Model Evaluation - {selected_model_name}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"R^2 Score: {r2_score(y_test, y_pred):.2f}")

    # Predict for full dataset
    if poly is not None:
        X_all_trans = poly.transform(scaler.transform(X))
    else:
        X_all_trans = scaler.transform(X)
    df['Prediction'] = model.predict(X_all_trans)

    # Download predictions
    predictions_df = df[['Date', 'Amount', 'Prediction']]
    csv_data = predictions_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", data=csv_data,
                       file_name="predictions.csv", mime="text/csv")

    # Save model (with scaler + poly)
    if st.button("Save Model"):
        joblib.dump({"model": model, "scaler": scaler, "poly": poly}, "expense_model.pkl")
        st.success("Model + Preprocessing saved as expense_model.pkl!")

    # Visualizations
    # 1. Distribution
    st.write("### Distribution of Daily Expenses")
    fig, ax = plt.subplots()
    sns.histplot(df['Amount'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Daily Expenses')
    st.pyplot(fig)

    # 2. Actual vs Predicted
    st.write("### Actual vs Predicted Daily Expenses")
    fig, ax = plt.subplots()
    ax.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
    ax.scatter(range(len(y_pred)), y_pred, color='red', alpha=0.6, label='Predicted')
    ax.legend()
    st.pyplot(fig)

    # 3. Monthly Predicted
    st.write("### Predicted Monthly Expenses")
    df['Month'] = df['Date'].dt.month
    monthly_expense = df.groupby('Month')['Prediction'].sum()
    fig, ax = plt.subplots()
    sns.barplot(x=monthly_expense.index, y=monthly_expense.values, palette="viridis", ax=ax)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    st.pyplot(fig)

    # 4. Feedback
    feedback = st.text_area("### Feedback on Predictions")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

    # 5. Monthly Trend (Plotly)
    st.write("### Monthly Expense Trend (Interactive)")
    monthly_expense_trend = df.groupby('Month')['Amount'].sum().reset_index()
    fig = px.line(monthly_expense_trend, x='Month', y='Amount', title='Monthly Expenses Trend')
    st.plotly_chart(fig)

    # 6. Feature Importance
    st.write("### Feature Importance in Expense Prediction")
    if hasattr(model, "coef_"):
        feature_importance = pd.Series(model.coef_, index=poly.get_feature_names_out(X.columns) if poly else X.columns)
    elif hasattr(model, "feature_importances_"):
        feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    else:
        feature_importance = pd.Series(dtype=float)

    if not feature_importance.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        feature_importance.sort_values(ascending=False).head(20).plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title("Top Feature Importances")
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model.")

    # 7. Time Series
    st.write("### Time Series of Daily Expenses")
    daily_expense = df.groupby('Date')['Amount'].sum()
    fig, ax = plt.subplots()
    daily_expense.plot(ax=ax, marker='o')
    st.pyplot(fig)

    # 8. Category-wise Expense
    if any("Category_" in col for col in df.columns):
        st.write("### Category-wise Expense Over Time")
        category_columns = [col for col in df.columns if 'Category_' in col]
        category_expense = df[category_columns].multiply(df['Amount'], axis=0).groupby(df['Date']).sum()
        fig, ax = plt.subplots(figsize=(12, 6))
        category_expense.plot(ax=ax)
        st.pyplot(fig)
