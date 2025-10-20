"""
Stock Market Analytics Dashboard
==================================

A Streamlit web application for analyzing historical stock data with interactive visualizations.

Installation:
-------------
pip install streamlit yfinance pandas numpy plotly scikit-learn

Usage:
------
streamlit run app.py

Features:
---------
- Multi-stock selection and custom date range
- Interactive Plotly charts
- Moving averages and technical indicators
- Correlation analysis and volatility metrics
- CSV export functionality
- Regression model evaluation
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ========================================
# Configuration and Constants
# ========================================

# Default stock tickers
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMZN']

# Page configuration
st.set_page_config(
    page_title="Stock Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# Helper Functions
# ========================================

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    
    Parameters:
    -----------
    tickers : list
        List of stock ticker symbols
    start_date : datetime
        Start date for data fetch
    end_date : datetime
        End date for data fetch
    
    Returns:
    --------
    pd.DataFrame or tuple : Data and adjusted close prices
    """
    try:
        # Convert single ticker to list format for consistent handling
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Download data
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        if data.empty:
            return None, None
        
        # Extract adjusted close prices based on data structure
        adj_close = None
        
        if len(tickers) == 1:
            # Single ticker returns simple DataFrame
            if 'Adj Close' in data.columns:
                adj_close = data['Adj Close']
            elif 'Close' in data.columns:
                # Fallback to Close if Adj Close not available
                adj_close = data['Close']
        else:
            # Multiple tickers - handle MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                # Try to find the adjusted close column
                price_types = data.columns.get_level_values(0).unique().tolist()
                
                if 'Adj Close' in price_types:
                    adj_close = data['Adj Close']
                elif 'Close' in price_types:
                    adj_close = data['Close']
            else:
                # Shouldn't happen, but handle it
                if 'Adj Close' in data.columns:
                    adj_close = data[['Adj Close']]
                elif 'Close' in data.columns:
                    adj_close = data[['Close']]
        
        return data, adj_close
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None


def calculate_moving_averages(df, periods=[20, 50]):
    """
    Calculate moving averages for the given periods.
    
    Parameters:
    -----------
    df : pd.DataFrame or pd.Series
        DataFrame or Series with price data
    periods : list
        List of periods for moving averages
    
    Returns:
    --------
    pd.DataFrame : DataFrame with moving averages added
    """
    # Convert Series to DataFrame if needed
    if isinstance(df, pd.Series):
        result = df.to_frame(name='Price')
    else:
        result = df.copy()
    
    # Calculate moving averages
    for period in periods:
        if isinstance(df, pd.Series):
            result[f'MA_{period}'] = df.rolling(window=period).mean()
        else:
            result[f'MA_{period}'] = df.rolling(window=period).mean()
    
    return result


def calculate_daily_returns(df):
    """
    Calculate daily percentage returns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    
    Returns:
    --------
    pd.Series : Daily returns as percentages
    """
    return df.pct_change() * 100


def calculate_volatility(returns, window=20):
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns
    window : int
        Rolling window size
    
    Returns:
    --------
    pd.Series : Rolling volatility
    """
    return returns.rolling(window=window).std()


def calculate_cumulative_returns(df):
    """
    Calculate cumulative returns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    
    Returns:
    --------
    pd.Series : Cumulative returns as percentages
    """
    returns = df.pct_change()
    cumulative = (1 + returns).cumprod() - 1
    return cumulative * 100


def prepare_regression_data(price_series, n_lags=30):
    """
    Prepare data for linear regression model.
    
    Parameters:
    -----------
    price_series : pd.Series
        Time series of prices
    n_lags : int
        Number of lagged features
    
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test, dates_test)
    """
    # Create lagged features
    feature_df = pd.DataFrame(index=price_series.index)
    
    for i in range(n_lags, 0, -1):
        feature_df[f'lag_{i}'] = price_series.shift(i)
    
    # Target: next day's price
    feature_df['target'] = price_series.shift(-1)
    
    # Drop NaN values
    feature_df.dropna(inplace=True)
    
    # Split features and target
    X = feature_df.drop('target', axis=1)
    y = feature_df['target']
    
    # Train-test split (80-20)
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    dates_test = X_test.index
    
    return X_train, X_test, y_train, y_test, dates_test


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Train a linear regression model and evaluate it.
    
    Parameters:
    -----------
    X_train, X_test, y_train, y_test : DataFrames/Series
        Training and testing data
    
    Returns:
    --------
    dict : Dictionary containing model, predictions, and metrics
    """
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train': {
            'MAE': mean_absolute_error(y_train, y_pred_train),
            'MSE': mean_squared_error(y_train, y_pred_train),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'R²': r2_score(y_train, y_pred_train)
        },
        'test': {
            'MAE': mean_absolute_error(y_test, y_pred_test),
            'MSE': mean_squared_error(y_test, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'R²': r2_score(y_test, y_pred_test)
        }
    }
    
    return {
        'model': model,
        'predictions_train': y_pred_train,
        'predictions_test': y_pred_test,
        'metrics': metrics
    }


# ========================================
# Visualization Functions
# ========================================

def plot_price_chart(data, tickers):
    """
    Create an interactive line chart of adjusted close prices.
    """
    fig = go.Figure()
    
    if len(tickers) == 1:
        # Single ticker
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name=tickers[0],
            line=dict(width=2)
        ))
        title = f"{tickers[0]} - Adjusted Close Price"
    else:
        # Multiple tickers
        for ticker in tickers:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[ticker].values,
                mode='lines',
                name=ticker,
                line=dict(width=2)
            ))
        title = "Stock Price Comparison - Adjusted Close"
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_moving_averages(price_data, ticker):
    """
    Create a chart with price and moving averages.
    """
    # Calculate moving averages
    ma_data = calculate_moving_averages(price_data, periods=[20, 50])
    
    fig = go.Figure()
    
    # Price line - handle both DataFrame and the converted structure
    price_col = 'Price' if 'Price' in ma_data.columns else ma_data.columns[0]
    
    fig.add_trace(go.Scatter(
        x=ma_data.index,
        y=ma_data[price_col],
        mode='lines',
        name='Price',
        line=dict(color='blue', width=2)
    ))
    
    # 20-day MA
    fig.add_trace(go.Scatter(
        x=ma_data.index,
        y=ma_data['MA_20'],
        mode='lines',
        name='MA 20',
        line=dict(color='orange', width=1.5, dash='dash')
    ))
    
    # 50-day MA
    fig.add_trace(go.Scatter(
        x=ma_data.index,
        y=ma_data['MA_50'],
        mode='lines',
        name='MA 50',
        line=dict(color='green', width=1.5, dash='dash')
    ))
    
    fig.update_layout(
        title=f"{ticker} - Price with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_daily_returns(returns, ticker):
    """
    Create a chart of daily returns.
    """
    fig = go.Figure()
    
    colors = ['red' if val < 0 else 'green' for val in returns.values]
    
    fig.add_trace(go.Bar(
        x=returns.index,
        y=returns.values,
        marker_color=colors,
        name='Daily Returns'
    ))
    
    fig.update_layout(
        title=f"{ticker} - Daily Returns (%)",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def plot_correlation_heatmap(data, tickers):
    """
    Create a correlation heatmap for multiple stocks.
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=tickers,
        y=tickers,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Stock Price Correlation Matrix",
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_cumulative_returns(data, tickers):
    """
    Create a chart of cumulative returns.
    """
    fig = go.Figure()
    
    if len(tickers) == 1:
        cum_returns = calculate_cumulative_returns(data)
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values,
            mode='lines',
            name=tickers[0],
            line=dict(width=2),
            fill='tozeroy'
        ))
    else:
        for ticker in tickers:
            cum_returns = calculate_cumulative_returns(data[ticker])
            fig.add_trace(go.Scatter(
                x=cum_returns.index,
                y=cum_returns.values,
                mode='lines',
                name=ticker,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Cumulative Returns Comparison (%)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_prediction_vs_actual(dates, actual, predicted, ticker):
    """
    Create a chart comparing actual vs predicted prices.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted,
        mode='lines',
        name='Predicted',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f"{ticker} - Actual vs Predicted Prices",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig


# ========================================
# Main Application
# ========================================

def main():
    # Header
    st.title("📈 Stock Market Analytics Dashboard")
    st.markdown("*Analyze historical stock data with interactive visualizations and predictive models*")
    st.markdown("---")
    
    # Sidebar for user inputs
    st.sidebar.header("🔧 Configuration")
    st.sidebar.markdown("Select stocks and date range to analyze")
    
    # Stock ticker selection
    selected_tickers = st.sidebar.multiselect(
        "Select Stock Tickers:",
        options=DEFAULT_TICKERS,
        default=['AAPL', 'MSFT', 'GOOGL'],
        help="Choose one or multiple stocks to analyze"
    )
    
    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    
    default_end = datetime.now()
    default_start = default_end - timedelta(days=5*365)
    
    start_date = st.sidebar.date_input(
        "Start Date:",
        value=default_start,
        max_value=datetime.now()
    )
    
    end_date = st.sidebar.date_input(
        "End Date:",
        value=default_end,
        max_value=datetime.now()
    )
    
    # Advanced options
    st.sidebar.subheader("⚙️ Advanced Options")
    show_regression = st.sidebar.checkbox("Show Regression Analysis", value=True)
    show_cumulative = st.sidebar.checkbox("Show Cumulative Returns", value=True)
    
    # Validate inputs
    if not selected_tickers:
        st.warning("⚠️ Please select at least one stock ticker from the sidebar.")
        st.stop()
    
    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        st.stop()
    
    # Fetch data button
    if st.sidebar.button("🔄 Fetch Data", type="primary"):
        st.session_state.fetch_triggered = True
    
    # Initialize session state
    if 'fetch_triggered' not in st.session_state:
        st.info("👈 Configure your settings in the sidebar and click **Fetch Data** to begin.")
        st.stop()
    
    # Fetch data
    with st.spinner("Downloading stock data..."):
        data, adj_close = fetch_stock_data(selected_tickers, start_date, end_date)
    
    if data is None or adj_close is None:
        st.error("❌ Failed to fetch stock data. Please try again with different settings.")
        st.stop()
    
    # Success message
    st.success(f"✅ Successfully loaded {len(adj_close)} days of data for {len(selected_tickers)} stock(s)")
    
    # Display basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stocks Selected", len(selected_tickers))
    with col2:
        st.metric("Data Points", len(adj_close))
    with col3:
        st.metric("Date Range", f"{(end_date - start_date).days} days")
    with col4:
        if len(selected_tickers) == 1:
            latest_price = adj_close.iloc[-1]
        else:
            latest_price = adj_close.iloc[-1].mean()
        st.metric("Avg Latest Price", f"${latest_price:.2f}")
    
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Price Charts", 
        "📈 Technical Analysis", 
        "🔗 Correlation Analysis", 
        "📉 Returns Analysis",
        "🤖 Predictive Model"
    ])
    
    # ========================================
    # Tab 1: Price Charts
    # ========================================
    with tab1:
        st.header("Stock Price History")
        
        # Main price chart
        price_fig = plot_price_chart(adj_close, selected_tickers)
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        if len(selected_tickers) == 1:
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
                'Value': [
                    f"${adj_close.mean():.2f}",
                    f"${adj_close.median():.2f}",
                    f"${adj_close.std():.2f}",
                    f"${adj_close.min():.2f}",
                    f"${adj_close.max():.2f}",
                    f"${adj_close.max() - adj_close.min():.2f}"
                ]
            })
        else:
            stats_data = []
            for ticker in selected_tickers:
                stats_data.append({
                    'Ticker': ticker,
                    'Mean': f"${adj_close[ticker].mean():.2f}",
                    'Median': f"${adj_close[ticker].median():.2f}",
                    'Std Dev': f"${adj_close[ticker].std():.2f}",
                    'Min': f"${adj_close[ticker].min():.2f}",
                    'Max': f"${adj_close[ticker].max():.2f}"
                })
            stats_df = pd.DataFrame(stats_data)
        
        st.dataframe(stats_df, use_container_width=True)
    
    # ========================================
    # Tab 2: Technical Analysis
    # ========================================
    with tab2:
        st.header("Technical Indicators")
        
        # Select ticker for technical analysis
        if len(selected_tickers) > 1:
            analysis_ticker = st.selectbox("Select stock for technical analysis:", selected_tickers)
            ticker_data = adj_close[analysis_ticker]
        else:
            analysis_ticker = selected_tickers[0]
            ticker_data = adj_close
        
        # Moving averages chart
        st.subheader(f"Moving Averages - {analysis_ticker}")
        ma_fig = plot_moving_averages(ticker_data, analysis_ticker)
        st.plotly_chart(ma_fig, use_container_width=True)
        
        # Daily returns chart
        st.subheader(f"Daily Returns - {analysis_ticker}")
        returns = calculate_daily_returns(ticker_data)
        returns_fig = plot_daily_returns(returns, analysis_ticker)
        st.plotly_chart(returns_fig, use_container_width=True)
        
        # Volatility analysis
        st.subheader("📊 Volatility Metrics")
        volatility = calculate_volatility(returns, window=20)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("20-Day Volatility", f"{volatility.iloc[-1]:.2f}%")
        with col2:
            st.metric("Mean Volatility", f"{volatility.mean():.2f}%")
        with col3:
            st.metric("Max Volatility", f"{volatility.max():.2f}%")
    
    # ========================================
    # Tab 3: Correlation Analysis
    # ========================================
    with tab3:
        st.header("Correlation Analysis")
        
        if len(selected_tickers) < 2:
            st.info("ℹ️ Select at least 2 stocks to view correlation analysis.")
        else:
            st.subheader("Correlation Heatmap")
            corr_fig = plot_correlation_heatmap(adj_close, selected_tickers)
            st.plotly_chart(corr_fig, use_container_width=True)
            
            # Correlation insights
            st.subheader("📊 Correlation Insights")
            corr_matrix = adj_close.corr()
            
            # Find highest and lowest correlations
            corr_pairs = []
            for i in range(len(selected_tickers)):
                for j in range(i+1, len(selected_tickers)):
                    corr_pairs.append({
                        'Stock 1': selected_tickers[i],
                        'Stock 2': selected_tickers[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Highest Correlations:**")
                st.dataframe(corr_df.head(3), use_container_width=True)
            with col2:
                st.write("**Lowest Correlations:**")
                st.dataframe(corr_df.tail(3), use_container_width=True)
    
    # ========================================
    # Tab 4: Returns Analysis
    # ========================================
    with tab4:
        st.header("Returns Analysis")
        
        if show_cumulative:
            st.subheader("Cumulative Returns")
            cum_returns_fig = plot_cumulative_returns(adj_close, selected_tickers)
            st.plotly_chart(cum_returns_fig, use_container_width=True)
        
        # Returns statistics
        st.subheader("📊 Returns Statistics")
        
        returns_stats = []
        for ticker in selected_tickers:
            if len(selected_tickers) == 1:
                ticker_returns = calculate_daily_returns(adj_close)
                cum_return = calculate_cumulative_returns(adj_close).iloc[-1]
            else:
                ticker_returns = calculate_daily_returns(adj_close[ticker])
                cum_return = calculate_cumulative_returns(adj_close[ticker]).iloc[-1]
            
            returns_stats.append({
                'Ticker': ticker,
                'Mean Daily Return (%)': f"{ticker_returns.mean():.3f}",
                'Std Daily Return (%)': f"{ticker_returns.std():.3f}",
                'Cumulative Return (%)': f"{cum_return:.2f}",
                'Best Day (%)': f"{ticker_returns.max():.2f}",
                'Worst Day (%)': f"{ticker_returns.min():.2f}"
            })
        
        returns_stats_df = pd.DataFrame(returns_stats)
        st.dataframe(returns_stats_df, use_container_width=True)
    
    # ========================================
    # Tab 5: Predictive Model
    # ========================================
    with tab5:
        st.header("Predictive Model (Linear Regression)")
        
        if not show_regression:
            st.info("ℹ️ Enable 'Show Regression Analysis' in the sidebar to view this section.")
        else:
            # Select ticker for prediction
            if len(selected_tickers) > 1:
                pred_ticker = st.selectbox("Select stock for prediction:", selected_tickers, key='pred_ticker')
                pred_data = adj_close[pred_ticker]
            else:
                pred_ticker = selected_tickers[0]
                pred_data = adj_close
            
            st.info(f"🤖 Training model to predict next day's closing price for **{pred_ticker}** using 30-day lagged features...")
            
            # Prepare data and train model
            with st.spinner("Training model..."):
                X_train, X_test, y_train, y_test, dates_test = prepare_regression_data(pred_data, n_lags=30)
                results = train_and_evaluate_model(X_train, X_test, y_train, y_test)
            
            # Display metrics
            st.subheader("📊 Model Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Training Set:**")
                train_metrics_df = pd.DataFrame({
                    'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
                    'Value': [
                        f"${results['metrics']['train']['MAE']:.2f}",
                        f"${results['metrics']['train']['MSE']:.2f}",
                        f"${results['metrics']['train']['RMSE']:.2f}",
                        f"{results['metrics']['train']['R²']:.4f}"
                    ]
                })
                st.dataframe(train_metrics_df, use_container_width=True)
            
            with col2:
                st.write("**Test Set:**")
                test_metrics_df = pd.DataFrame({
                    'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
                    'Value': [
                        f"${results['metrics']['test']['MAE']:.2f}",
                        f"${results['metrics']['test']['MSE']:.2f}",
                        f"${results['metrics']['test']['RMSE']:.2f}",
                        f"{results['metrics']['test']['R²']:.4f}"
                    ]
                })
                st.dataframe(test_metrics_df, use_container_width=True)
            
            # Model interpretation
            r2_test = results['metrics']['test']['R²']
            if r2_test > 0.9:
                interpretation = "🎯 Excellent: Model explains >90% of price variation"
            elif r2_test > 0.8:
                interpretation = "✅ Good: Model has strong predictive power"
            elif r2_test > 0.7:
                interpretation = "👍 Fair: Model shows decent predictive ability"
            else:
                interpretation = "⚠️ Limited: Model may need improvement"
            
            st.info(interpretation)
            
            # Prediction visualization
            st.subheader("Actual vs Predicted Prices (Test Set)")
            pred_fig = plot_prediction_vs_actual(
                dates_test,
                y_test.values,
                results['predictions_test'],
                pred_ticker
            )
            st.plotly_chart(pred_fig, use_container_width=True)
            
            # Residuals analysis
            st.subheader("Prediction Errors (Residuals)")
            residuals = y_test.values - results['predictions_test']
            
            residuals_fig = go.Figure()
            residuals_fig.add_trace(go.Scatter(
                x=dates_test,
                y=residuals,
                mode='markers',
                marker=dict(
                    color=residuals,
                    colorscale='RdYlGn_r',
                    size=6,
                    showscale=True,
                    colorbar=dict(title="Error ($)")
                ),
                name='Residuals'
            ))
            residuals_fig.add_hline(y=0, line_dash="dash", line_color="red")
            
            residuals_fig.update_layout(
                title="Prediction Errors Over Time",
                xaxis_title="Date",
                yaxis_title="Prediction Error ($)",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(residuals_fig, use_container_width=True)
    
    # ========================================
    # Download Section
    # ========================================
    st.markdown("---")
    st.header("📥 Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download raw data
        csv_data = adj_close.to_csv()
        st.download_button(
            label="📄 Download Raw Data (CSV)",
            data=csv_data,
            file_name=f"stock_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download the adjusted close prices for selected stocks"
        )
    
    with col2:
        # Download returns data
        if len(selected_tickers) == 1:
            returns_data = calculate_daily_returns(adj_close).to_frame(name='Daily_Returns_%')
        else:
            returns_data = adj_close.pct_change() * 100
            returns_data.columns = [f"{col}_Returns_%" for col in returns_data.columns]
        
        returns_csv = returns_data.to_csv()
        st.download_button(
            label="📊 Download Returns Data (CSV)",
            data=returns_csv,
            file_name=f"stock_returns_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download daily returns data for selected stocks"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Built with ❤️ using Streamlit | Data powered by Yahoo Finance</p>
            <p style='font-size: 12px;'>⚠️ Disclaimer: This tool is for educational purposes only. 
            Not financial advice. Always do your own research before making investment decisions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ========================================
# Run Application
# ========================================
if __name__ == "__main__":
    main()
