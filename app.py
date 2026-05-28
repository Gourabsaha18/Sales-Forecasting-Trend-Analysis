import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose

import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="Sales Dashboard", layout="wide")
st.title("📊 Sales Forecasting & Trend Analysis Dashboard")

# ---------------------------------
# FUNCTIONS
# ---------------------------------

def apply_filters(df):
    st.sidebar.subheader("🎛 Advanced Filters")

    df = df.copy()
    df.columns = df.columns.str.strip()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        min_date, max_date = df['Date'].min(), df['Date'].max()

        date_range = st.sidebar.date_input("📅 Date Range", [min_date, max_date])

        if len(date_range) == 2:
            df = df[(df['Date'] >= pd.to_datetime(date_range[0])) &
                    (df['Date'] <= pd.to_datetime(date_range[1]))]

    store_col = next((c for c in df.columns if 'store' in c.lower()), None)
    region_col = next((c for c in df.columns if 'region' in c.lower()), None)
    category_col = next((c for c in df.columns if 'category' in c.lower()), None)
    sales_col = next((c for c in df.columns if 'sales' in c.lower()), None)

    def multi_filter(column, name):
        values = df[column].dropna().unique()
        selected = st.sidebar.multiselect(name, values, default=values)
        return df[df[column].isin(selected)]

    if store_col:
        df = multi_filter(store_col, "🏪 Store")
    if region_col:
        df = multi_filter(region_col, "🌍 Region")
    if category_col:
        df = multi_filter(category_col, "📦 Category")

    if sales_col:
        df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
        min_val, max_val = float(df[sales_col].min()), float(df[sales_col].max())

        sales_range = st.sidebar.slider("💰 Sales Range", min_val, max_val, (min_val, max_val))
        df = df[(df[sales_col] >= sales_range[0]) &
                (df[sales_col] <= sales_range[1])]

    return df


def preprocess(df):
    df.columns = df.columns.str.strip()

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])

    sales_col = next((c for c in df.columns if 'sales' in c.lower()), None)

    if sales_col is None:
        st.error("❌ Sales column not found")
        st.stop()

    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')

    sales_data = df.groupby('Date')[sales_col].sum().reset_index()
    sales_data.rename(columns={sales_col: 'Weekly_Sales'}, inplace=True)

    return sales_data, sales_col


# ---------------------------------
# MENU
# ---------------------------------
menu = st.sidebar.selectbox("Menu", [

    "Upload Data",
    "Data Preprocessing",
    "EDA Analysis",
    "Heatmaps",
    "Forecasting (SARIMA)",
])

# ---------------------------------
# UPLOAD
# ---------------------------------
if menu == "Upload Data":
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        st.session_state['df'] = df
        st.success("✅ Data Loaded")
        st.write(df.head())
        
# ---------------------------------
# DATA PREPROCESSING
# ---------------------------------
elif menu == "Data Preprocessing":

    st.header("🛠 Data Preprocessing Dashboard")

    if 'df' not in st.session_state:
        st.warning("Upload data first!")

    else:

        df = st.session_state['df'].copy()

        st.subheader("📄 Raw Dataset")

        st.dataframe(df.head())

        # ---------------------------------
        # DATASET INFO
        # ---------------------------------
        st.subheader("📌 Dataset Information")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Rows",
            df.shape[0]
        )

        col2.metric(
            "Columns",
            df.shape[1]
        )

        col3.metric(
            "Duplicate Rows",
            df.duplicated().sum()
        )

        # ---------------------------------
        # COLUMN TYPES
        # ---------------------------------
        st.subheader("🧾 Column Data Types")

        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Datatype": df.dtypes.astype(str)
        })

        st.dataframe(dtype_df)

        # ---------------------------------
        # MISSING VALUES
        # ---------------------------------
        st.subheader("❌ Missing Values Analysis")

        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isnull().sum(),
            "Missing %": (
                df.isnull().sum() / len(df)
            ) * 100
        })

        st.dataframe(missing_df)

        fig = px.bar(
            missing_df,
            x="Column",
            y="Missing Values",
            title="Missing Values Per Column"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ---------------------------------
        # DUPLICATE VALUES
        # ---------------------------------
        st.subheader("🔁 Duplicate Data Check")

        duplicate_count = df.duplicated().sum()

        st.metric(
            "Duplicate Rows",
            duplicate_count
        )

        if duplicate_count > 0:

            st.warning(
                "Dataset contains duplicate rows."
            )

            if st.button("Remove Duplicates"):

                df = df.drop_duplicates()

                st.session_state['df'] = df

                st.success(
                    "Duplicates removed successfully!"
                )

        else:

            st.success(
                "No duplicate rows found."
            )

        # ---------------------------------
        # NULL VALUE HANDLING
        # ---------------------------------
        st.subheader("🧹 Handle Missing Values")

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        fill_option = st.selectbox(
            "Select Fill Method",
            [
                "Mean",
                "Median",
                "Mode",
                "Forward Fill",
                "Backward Fill"
            ]
        )

        if st.button("Apply Missing Value Treatment"):

            for col in numeric_cols:

                if fill_option == "Mean":

                    df[col] = df[col].fillna(
                        df[col].mean()
                    )

                elif fill_option == "Median":

                    df[col] = df[col].fillna(
                        df[col].median()
                    )

                elif fill_option == "Mode":

                    df[col] = df[col].fillna(
                        df[col].mode()[0]
                    )

                elif fill_option == "Forward Fill":

                    df[col] = df[col].ffill()

                elif fill_option == "Backward Fill":

                    df[col] = df[col].bfill()

            st.session_state['df'] = df

            st.success(
                "Missing values handled successfully!"
            )

        # ---------------------------------
        # OUTLIER DETECTION
        # ---------------------------------
        st.subheader("🚨 Outlier Detection")

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:

            selected_col = st.selectbox(
                "Select Numeric Column",
                numeric_columns
            )

            fig = px.box(
                df,
                y=selected_col,
                title=f"Outlier Detection - {selected_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ---------------------------------
        # DATA DISTRIBUTION
        # ---------------------------------
        st.subheader("📊 Data Distribution")

        if numeric_columns:

            hist_col = st.selectbox(
                "Select Column for Histogram",
                numeric_columns,
                key="hist"
            )

            fig = px.histogram(
                df,
                x=hist_col,
                nbins=30,
                title=f"Distribution of {hist_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ---------------------------------
        # CORRELATION MATRIX
        # ---------------------------------
        st.subheader("🔗 Correlation Matrix")

        if len(numeric_columns) > 1:

            corr = df[numeric_columns].corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                title="Correlation Heatmap"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ---------------------------------
        # DATE CONVERSION
        # ---------------------------------
        st.subheader("📅 Date Column Conversion")

        possible_dates = [
            c for c in df.columns
            if 'date' in c.lower()
        ]

        if possible_dates:

            date_col = st.selectbox(
                "Select Date Column",
                possible_dates
            )

            if st.button("Convert to Datetime"):

                df[date_col] = pd.to_datetime(
                    df[date_col],
                    errors='coerce'
                )

                st.session_state['df'] = df

                st.success(
                    "Date column converted successfully!"
                )

        # ---------------------------------
        # FEATURE ENGINEERING
        # ---------------------------------
        st.subheader("⚙ Feature Engineering")

        if 'Date' in df.columns:

            df['Date'] = pd.to_datetime(
                df['Date'],
                errors='coerce'
            )

            if st.button("Generate Date Features"):

                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Day'] = df['Date'].dt.day
                df['Weekday'] = df['Date'].dt.day_name()
                df['Quarter'] = df['Date'].dt.quarter

                st.session_state['df'] = df

                st.success(
                    "Date features generated!"
                )

                st.dataframe(
                    df.head()
                )

        # ---------------------------------
        # DATA EXPORT
        # ---------------------------------
        st.subheader("📥 Export Cleaned Dataset")

        csv = df.to_csv(index=False)

        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

# ---------------------------------
# EDA (UPGRADED)
# ---------------------------------
elif menu == "EDA Analysis":

    if 'df' not in st.session_state:
        st.warning("Upload data first!")
    else:
        df = apply_filters(st.session_state['df'])
        sales_data, sales_col = preprocess(df)

        # KPI
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sales", f"{sales_data['Weekly_Sales'].sum():,.0f}")
        col2.metric("Avg Sales", f"{sales_data['Weekly_Sales'].mean():,.0f}")
        col3.metric("Max Sales", f"{sales_data['Weekly_Sales'].max():,.0f}")

        # Growth Rate
        sales_data['Growth %'] = sales_data['Weekly_Sales'].pct_change()*100

        fig = px.line(sales_data, x='Date', y='Growth %', title="📈 Growth Rate %")
        st.plotly_chart(fig, use_container_width=True)

        # Rolling Average
        sales_data['Rolling Avg'] = sales_data['Weekly_Sales'].rolling(4).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sales_data['Date'], y=sales_data['Weekly_Sales'], name='Actual'))
        fig.add_trace(go.Scatter(x=sales_data['Date'], y=sales_data['Rolling Avg'], name='Rolling Avg'))
        st.plotly_chart(fig, use_container_width=True)

        # Day of Week
        sales_data['Day'] = sales_data['Date'].dt.day_name()

        fig = px.box(sales_data, x='Day', y='Weekly_Sales', title="📅 Day-wise Sales")
        st.plotly_chart(fig, use_container_width=True)

        # Outliers
        fig = px.box(sales_data, y='Weekly_Sales', title="🚨 Outlier Detection")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # PIE CHARTS
    # -------------------------------
    st.subheader("🥧 Contribution Analysis")

    store_col = next((c for c in df.columns if 'store' in c.lower()), None)
    region_col = next((c for c in df.columns if 'region' in c.lower()), None)
    category_col = next((c for c in df.columns if 'category' in c.lower()), None)

    if store_col:
        store_share = df.groupby(store_col)[sales_col].sum().reset_index()
        fig = px.pie(store_share, names=store_col, values=sales_col, title="Store Contribution")
        st.plotly_chart(fig, use_container_width=True)

    if region_col:
        region_share = df.groupby(region_col)[sales_col].sum().reset_index()
        fig = px.pie(region_share, names=region_col, values=sales_col, title="Region Contribution")
        st.plotly_chart(fig, use_container_width=True)

    if category_col:
        category_share = df.groupby(category_col)[sales_col].sum().reset_index()
        fig = px.pie(category_share, names=category_col, values=sales_col, title="Category Contribution")
        st.plotly_chart(fig, use_container_width=True)
        
    # -------------------------------
    # BAR ANALYSIS
    # -------------------------------
    st.subheader("📊 Comparative Analysis")

    if category_col:
        cat_sales = df.groupby(category_col)[sales_col].sum().reset_index()
        fig = px.bar(cat_sales, x=category_col, y=sales_col, title="Category-wise Sales")
        st.plotly_chart(fig, use_container_width=True)

    if region_col:
        reg_sales = df.groupby(region_col)[sales_col].sum().reset_index()
        fig = px.bar(reg_sales, x=region_col, y=sales_col, title="Region-wise Sales")
        st.plotly_chart(fig, use_container_width=True)
    # ---------------------------------
    # YEAR OVER YEAR GROWTH
    # ---------------------------------
    st.subheader("📈 Year-over-Year Growth Analysis")

    # Create Year column
    sales_data['Year'] = sales_data['Date'].dt.year

    # Yearly sales
    yearly_sales = sales_data.groupby('Year')['Weekly_Sales'].sum().reset_index()

    # YoY Growth %
    yearly_sales['YoY Growth %'] = yearly_sales['Weekly_Sales'].pct_change() * 100

    # Show table
    st.write(yearly_sales)

    # YoY Growth Chart
    fig = px.bar(
        yearly_sales,
        x='Year',
        y='YoY Growth %',
        text_auto='.2f',
        title="📊 Year-over-Year Growth (%)"
    )

    st.plotly_chart(fig, use_container_width=True)
    # -------------------------------
    # CUMULATIVE SALES
    # -------------------------------
    st.subheader("📈 Cumulative Sales Growth")

    sales_data['Cumulative Sales'] = sales_data['Weekly_Sales'].cumsum()

    fig = px.line(sales_data, x='Date', y='Cumulative Sales',
                title="Cumulative Sales Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # SEASONALITY
    # -------------------------------

    st.subheader("📅 Monthly Seasonality")

    # Ensure Month column exists
    sales_data['Month'] = sales_data['Date'].dt.month

    monthly_avg = sales_data.groupby('Month')['Weekly_Sales'].mean().reset_index()

    fig = px.bar(
        monthly_avg,
        x='Month',
        y='Weekly_Sales',
        title="Average Monthly Sales"
    )

    st.plotly_chart(fig, use_container_width=True)
    # -------------------------------
    # CATEGORY TREND
    # -------------------------------
    if category_col:
        st.subheader("📦 Category Trend Over Time")

        cat_trend = df.groupby(['Date', category_col])[sales_col].sum().reset_index()

        fig = px.line(cat_trend, x='Date', y=sales_col, color=category_col)
        st.plotly_chart(fig, use_container_width=True)
    # -------------------------------
    # SCATTER ANALYSIS
    # -------------------------------
    st.subheader("🔍 Sales vs Time Scatter")

    fig = px.scatter(sales_data, x='Date', y='Weekly_Sales',
                    title="Sales Distribution Over Time")
    st.plotly_chart(fig, use_container_width=True)
    
    # ---------------------------------
    # BUSINESS INSIGHTS PANEL
    # ---------------------------------
    st.subheader("🤖 Automated EDA Insights")

    avg_sales = sales_data['Weekly_Sales'].mean()
    max_sales = sales_data['Weekly_Sales'].max()
    min_sales = sales_data['Weekly_Sales'].min()

    latest_sales = sales_data['Weekly_Sales'].iloc[-1]

    if latest_sales > avg_sales:
        st.success(
            f"📈 Current sales ({latest_sales:.2f}) are ABOVE average sales ({avg_sales:.2f})."
        )
    else:
        st.warning(
            f"📉 Current sales ({latest_sales:.2f}) are BELOW average sales ({avg_sales:.2f})."
        )

    st.info(
        f"🔥 Highest recorded sales: {max_sales:.2f}"
    )

    st.info(
        f"📉 Lowest recorded sales: {min_sales:.2f}"
    )

    # ---------------------------------
    # SALES VOLATILITY ANALYSIS
    # ---------------------------------
    st.subheader("📊 Sales Volatility Analysis")

    volatility = sales_data['Weekly_Sales'].std()

    st.metric(
        "Sales Volatility (Std Dev)",
        f"{volatility:.2f}"
    )

    if volatility > avg_sales * 0.5:
        st.warning(
            "Sales show HIGH volatility. Demand fluctuates significantly."
        )
    else:
        st.success(
            "Sales are relatively stable over time."
        )

    # ---------------------------------
    # TOP SALES PERIODS
    # ---------------------------------
    st.subheader("🏆 Top Performing Time Periods")

    top_5 = sales_data.nlargest(5, 'Weekly_Sales')

    fig = px.bar(
        top_5,
        x='Date',
        y='Weekly_Sales',
        title="Top 5 Sales Dates",
        text_auto=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------
    # LOWEST SALES PERIODS
    # ---------------------------------
    st.subheader("📉 Lowest Performing Time Periods")

    bottom_5 = sales_data.nsmallest(5, 'Weekly_Sales')

    fig = px.bar(
        bottom_5,
        x='Date',
        y='Weekly_Sales',
        title="Lowest 5 Sales Dates",
        text_auto=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------
    # SALES TREND STRENGTH
    # ---------------------------------
    st.subheader("📈 Trend Strength Analysis")

    sales_data['Trend'] = (
        sales_data['Weekly_Sales']
        .rolling(window=12)
        .mean()
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sales_data['Date'],
        y=sales_data['Weekly_Sales'],
        mode='lines',
        name='Actual Sales'
    ))

    fig.add_trace(go.Scatter(
        x=sales_data['Date'],
        y=sales_data['Trend'],
        mode='lines',
        name='Trend Line'
    ))

    fig.update_layout(
        title="Sales Trend Strength",
        xaxis_title="Date",
        yaxis_title="Sales"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------
    # MONTHLY GROWTH ANALYSIS
    # ---------------------------------
    st.subheader("📅 Monthly Growth Analysis")

    sales_data['Month_Name'] = sales_data['Date'].dt.strftime('%b')

    monthly_growth = (
        sales_data
        .groupby('Month_Name')['Growth %']
        .mean()
        .reset_index()
    )

    fig = px.line(
        monthly_growth,
        x='Month_Name',
        y='Growth %',
        markers=True,
        title="Average Monthly Growth %"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------
    # SALES DISTRIBUTION HISTOGRAM
    # ---------------------------------
    st.subheader("📦 Sales Frequency Distribution")

    fig = px.histogram(
        sales_data,
        x='Weekly_Sales',
        nbins=25,
        title="Sales Frequency Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------
    # STORE PERFORMANCE RANKING
    # ---------------------------------
    if store_col:

        st.subheader("🏪 Store Performance Ranking")

        store_rank = (
            df.groupby(store_col)[sales_col]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.funnel(
            store_rank,
            x=sales_col,
            y=store_col,
            title="Store Ranking Funnel"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------------------------
    # CATEGORY SHARE TREND
    # ---------------------------------
    if category_col:

        st.subheader("📦 Category Share Over Time")

        category_time = (
            df.groupby(['Date', category_col])[sales_col]
            .sum()
            .reset_index()
        )

        fig = px.area(
            category_time,
            x='Date',
            y=sales_col,
            color=category_col,
            title="Category Contribution Trend"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------------------------
    # SALES CORRELATION INSIGHTS
    # ---------------------------------
    st.subheader("🔍 Correlation Insights")

    sales_data['Month'] = sales_data['Date'].dt.month
    sales_data['Year'] = sales_data['Date'].dt.year

    corr_matrix = sales_data[
        ['Weekly_Sales', 'Month', 'Year']
    ].corr()

    st.write(corr_matrix)

    correlation = corr_matrix.loc['Weekly_Sales', 'Month']

    if correlation > 0:
        st.success(
            "Sales positively correlate with month progression."
        )
    else:
        st.warning(
            "Sales negatively correlate with month progression."
        )

    # ---------------------------------
    # DATA QUALITY CHECK
    # ---------------------------------
    st.subheader("🛠 Data Quality Check")

    missing_values = df.isnull().sum()

    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Values': missing_values.values
    })

    st.dataframe(missing_df)

    if missing_values.sum() == 0:
        st.success("✅ No missing values detected.")
    else:
        st.warning("⚠ Dataset contains missing values.")


# ---------------------------------
# HEATMAPS (same)
# ---------------------------------
elif menu == "Heatmaps":

    if 'df' not in st.session_state:
        st.warning("Upload data first!")
    else:
        df = apply_filters(st.session_state['df'])
        sales_data, _ = preprocess(df)

        sales_data['Month'] = sales_data['Date'].dt.month
        sales_data['Year'] = sales_data['Date'].dt.year

        corr = sales_data[['Weekly_Sales','Month','Year']].corr()
        st.plotly_chart(px.imshow(corr, text_auto=True), use_container_width=True)

        pivot = sales_data.pivot_table(values='Weekly_Sales',
                                       index='Month', columns='Year', aggfunc='mean')

        st.plotly_chart(px.imshow(pivot, text_auto=True), use_container_width=True)


# ---------------------------------
# FORECASTING (SARIMA)
# ---------------------------------
elif menu == "Forecasting (SARIMA)":

    if 'df' not in st.session_state:
        st.warning("Upload data first!")

    else:
        df = apply_filters(st.session_state['df'])
        sales_data, _ = preprocess(df)

        # Set index
        sales_data.set_index('Date', inplace=True)
        sales_data = sales_data.sort_index().asfreq('W-FRI')

        # Fill missing values
        sales_data['Weekly_Sales'] = sales_data['Weekly_Sales'].ffill()

        # ---------------------------------
        # USER FORECAST DATE INPUT
        # ---------------------------------
        st.sidebar.subheader("🔮 Forecast Settings")

        last_date = sales_data.index.max()

        future_date = st.sidebar.date_input(
            "Select Forecast End Date",
            value=last_date + pd.DateOffset(weeks=12),
            min_value=last_date
        )

        # Calculate forecast steps
        forecast_steps = int(
            (pd.to_datetime(future_date) - last_date).days / 7
        )

        if forecast_steps <= 0:
            st.warning("Please select a future date.")
            st.stop()

        # ---------------------------------
        # TRAIN TEST SPLIT
        # ---------------------------------
        train = sales_data.iloc[:int(0.8 * len(sales_data))]
        test = sales_data.iloc[int(0.8 * len(sales_data)):]

        # ---------------------------------
        # SARIMA MODEL
        # ---------------------------------
        model = SARIMAX(
            train['Weekly_Sales'],
            order=(1,1,1),
            seasonal_order=(1,1,1,52)
        )

        model_fit = model.fit(disp=False)

        # ---------------------------------
        # TEST FORECAST
        # ---------------------------------
        forecast_obj = model_fit.get_forecast(steps=len(test))

        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()

        # ---------------------------------
        # FORECAST VS ACTUAL
        # ---------------------------------
        st.subheader("📈 Forecast vs Actual")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=test.index,
            y=test['Weekly_Sales'],
            mode='lines',
            name='Actual'
        ))

        fig.add_trace(go.Scatter(
            x=test.index,
            y=forecast,
            mode='lines',
            name='Forecast'
        ))

        # Confidence Interval
        fig.add_trace(go.Scatter(
            x=test.index,
            y=conf_int.iloc[:, 0],
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=test.index,
            y=conf_int.iloc[:, 1],
            fill='tonexty',
            name='Confidence Interval'
        ))

        fig.update_layout(
            title="Forecast vs Actual with Confidence Interval",
            xaxis_title="Date",
            yaxis_title="Sales"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------
        # RESIDUAL ANALYSIS
        # ---------------------------------
        st.subheader("📉 Residual Distribution")

        residuals = test['Weekly_Sales'] - forecast

        fig = px.histogram(
            residuals,
            nbins=30,
            title="Residual Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------
        # FUTURE FORECAST
        # ---------------------------------
        st.subheader("🔮 Future Forecast")

        future_forecast_obj = model_fit.get_forecast(steps=forecast_steps)

        future_forecast = future_forecast_obj.predicted_mean
        future_conf = future_forecast_obj.conf_int()

        # Forecast dates
        future_dates = pd.date_range(
            start=last_date,
            periods=forecast_steps + 1,
            freq='W-FRI'
        )[1:]

        # Forecast dataframe
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': future_forecast.values,
            'Lower CI': future_conf.iloc[:, 0].values,
            'Upper CI': future_conf.iloc[:, 1].values
        })

        # ---------------------------------
        # FUTURE FORECAST GRAPH
        # ---------------------------------
        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(
            x=sales_data.index,
            y=sales_data['Weekly_Sales'],
            mode='lines',
            name='Historical Sales'
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=future_df['Date'],
            y=future_df['Forecast'],
            mode='lines',
            name='Future Forecast'
        ))

        # Confidence Interval
        fig.add_trace(go.Scatter(
            x=future_df['Date'],
            y=future_df['Lower CI'],
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=future_df['Date'],
            y=future_df['Upper CI'],
            fill='tonexty',
            name='Confidence Interval'
        ))

        fig.update_layout(
            title="Future Sales Forecast",
            xaxis_title="Date",
            yaxis_title="Sales"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------
        # FORECAST TABLE
        # ---------------------------------
        st.subheader("📋 Forecast Data")

        st.dataframe(future_df)
        
        # ---------------------------------
        # METRICS
        # ---------------------------------
        mae = mean_absolute_error(test['Weekly_Sales'], forecast)
        rmse = np.sqrt(mean_squared_error(test['Weekly_Sales'], forecast))

        # Accuracy %
        accuracy = 100 - (mae / test['Weekly_Sales'].mean()) * 100

        st.subheader("📏 Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("Accuracy", f"{accuracy:.2f}%")

        # ---------------------------------
        # METRICS + ADVANCED ANALYTICS
        # PLACE THIS AT THE VERY END OF
        # FORECASTING (SARIMA) SECTION
        # ---------------------------------

        # ---------------------------------
        # METRICS
        # ---------------------------------
        mae = mean_absolute_error(test['Weekly_Sales'], forecast)

        rmse = np.sqrt(
            mean_squared_error(
                test['Weekly_Sales'],
                forecast
            )
        )

        accuracy = 100 - (
            mae / test['Weekly_Sales'].mean()
        ) * 100

        st.subheader("📏 Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "MAE",
            f"{mae:.2f}"
        )

        col2.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

        col3.metric(
            "Accuracy",
            f"{accuracy:.2f}%"
        )

        # ---------------------------------
        # BEST & WORST SALES DAY
        # ---------------------------------
        st.subheader("🔥 Best & Worst Sales Days")

        best_day = sales_data.loc[
            sales_data['Weekly_Sales'].idxmax()
        ]

        worst_day = sales_data.loc[
            sales_data['Weekly_Sales'].idxmin()
        ]

        c1, c2 = st.columns(2)

        c1.success(
            f"""
        🔥 Best Sales Day

        Date: {best_day.name.strftime('%Y-%m-%d')}

        Sales: {best_day['Weekly_Sales']:.2f}
        """
        )

        c2.error(
            f"""
        📉 Worst Sales Day

        Date: {worst_day.name.strftime('%Y-%m-%d')}

        Sales: {worst_day['Weekly_Sales']:.2f}
        """
        )

        # ---------------------------------
        # MOVING AVERAGE ANALYSIS
        # ---------------------------------
        st.subheader("📊 Moving Average Analysis")

        sales_data['7 Week MA'] = (
            sales_data['Weekly_Sales']
            .rolling(7)
            .mean()
        )

        sales_data['30 Week MA'] = (
            sales_data['Weekly_Sales']
            .rolling(30)
            .mean()
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sales_data.index,
            y=sales_data['Weekly_Sales'],
            mode='lines',
            name='Actual Sales'
        ))

        fig.add_trace(go.Scatter(
            x=sales_data.index,
            y=sales_data['7 Week MA'],
            mode='lines',
            name='7 Week Moving Avg'
        ))

        fig.add_trace(go.Scatter(
            x=sales_data.index,
            y=sales_data['30 Week MA'],
            mode='lines',
            name='30 Week Moving Avg'
        ))

        fig.update_layout(
            title="Moving Average Trend Analysis",
            xaxis_title="Date",
            yaxis_title="Sales"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ---------------------------------
        # FORECAST TREND INDICATOR
        # ---------------------------------
        st.subheader("📈 Forecast Trend Indicator")

        if future_df['Forecast'].iloc[-1] > future_df['Forecast'].iloc[0]:

            st.success(
                "📈 Forecast indicates an UPWARD sales trend."
            )

        else:

            st.error(
                "📉 Forecast indicates a DOWNWARD sales trend."
            )

        # ---------------------------------
        # ANOMALY DETECTION
        # ---------------------------------
        st.subheader("🚨 Sales Anomaly Detection")

        threshold = (
            sales_data['Weekly_Sales'].mean()
            +
            2 * sales_data['Weekly_Sales'].std()
        )

        anomalies = sales_data[
            sales_data['Weekly_Sales'] > threshold
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sales_data.index,
            y=sales_data['Weekly_Sales'],
            mode='lines',
            name='Sales'
        ))

        fig.add_trace(go.Scatter(
            x=anomalies.index,
            y=anomalies['Weekly_Sales'],
            mode='markers',
            name='Anomalies'
        ))

        fig.update_layout(
            title="Sales Anomaly Detection",
            xaxis_title="Date",
            yaxis_title="Sales"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ---------------------------------
        # QUARTERLY SALES ANALYSIS
        # ---------------------------------
        st.subheader("📅 Quarterly Sales Analysis")

        quarterly_data = sales_data.copy()

        quarterly_data['Quarter'] = (
            quarterly_data.index.quarter
        )

        quarterly_sales = (
            quarterly_data
            .groupby('Quarter')['Weekly_Sales']
            .mean()
            .reset_index()
        )

        fig = px.bar(
            quarterly_sales,
            x='Quarter',
            y='Weekly_Sales',
            title="Average Quarterly Sales"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ---------------------------------
        # SALES DISTRIBUTION CURVE
        # ---------------------------------
        st.subheader("📉 Sales Distribution Curve")

        fig = px.violin(
            sales_data,
            y='Weekly_Sales',
            box=True,
            title="Sales Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ---------------------------------
        # DOWNLOAD FORECAST
        # ---------------------------------
        st.subheader("📥 Download Forecast Report")

        csv = future_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name='sales_forecast.csv',
            mime='text/csv'
        )

        # ---------------------------------
        # AUTOMATED BUSINESS INSIGHTS
        # ---------------------------------
        st.subheader("🤖 Automated Business Insights")

        avg_sales = sales_data[
            'Weekly_Sales'
        ].mean()

        latest_sales = sales_data[
            'Weekly_Sales'
        ].iloc[-1]

        if latest_sales > avg_sales:

            st.success(
                "Sales are currently ABOVE average levels."
            )

        else:

            st.warning(
                "Sales are currently BELOW average levels."
            )

        growth_rate = (
            (
                future_df['Forecast'].iloc[-1]
                -
                future_df['Forecast'].iloc[0]
            )
            /
            future_df['Forecast'].iloc[0]
        ) * 100

        st.info(
            f"Expected forecast growth rate: {growth_rate:.2f}%"
        )

        # ---------------------------------
        # FORECAST SUMMARY TABLE
        # ---------------------------------
        st.subheader("📋 Forecast Summary Statistics")

        summary_df = pd.DataFrame({

            "Metric": [

                "Average Forecast",
                "Maximum Forecast",
                "Minimum Forecast",
                "Forecast Growth %",
                "Forecast Accuracy %"

            ],

            "Value": [

                round(
                    future_df['Forecast'].mean(),
                    2
                ),

                round(
                    future_df['Forecast'].max(),
                    2
                ),

                round(
                    future_df['Forecast'].min(),
                    2
                ),

                round(
                    growth_rate,
                    2
                ),

                round(
                    accuracy,
                    2
                )
            ]
        })

        st.dataframe(summary_df)
