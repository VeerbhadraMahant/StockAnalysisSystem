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
    # Ensure we have a clean Series
    if isinstance(price_series, pd.DataFrame):
        price_series = price_series.iloc[:, 0]
    
    # Remove any NaN or infinite values from input
    price_series = price_series.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Check if we have enough data
    if len(price_series) < n_lags + 100:  # Need at least n_lags + reasonable train/test split
        raise ValueError(f"Insufficient data for regression. Need at least {n_lags + 100} data points, got {len(price_series)}")
    
    # Create lagged features
    feature_df = pd.DataFrame(index=price_series.index)
    
    for i in range(n_lags, 0, -1):
        feature_df[f'lag_{i}'] = price_series.shift(i)
    
    # Target: next day's price
    feature_df['target'] = price_series.shift(-1)
    
    # Drop NaN values
    feature_df = feature_df.dropna()
    
    # Double-check for any remaining NaN or infinite values
    if feature_df.isnull().any().any() or np.isinf(feature_df.values).any():
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Split features and target
    X = feature_df.drop('target', axis=1).values  # Convert to numpy array
    y = feature_df['target'].values  # Convert to numpy array
    
    # Train-test split (80-20)
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    dates_test = feature_df.index[split_idx:]
    
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


def predict_future_prices(model, price_series, n_days, n_lags=30):
    """
    Predict future stock prices for the next n_days.
    
    Parameters:
    -----------
    model : sklearn model
        Trained regression model
    price_series : pd.Series or pd.DataFrame
        Historical price data
    n_days : int
        Number of days to predict into the future
    n_lags : int
        Number of lagged features used in the model
    
    Returns:
    --------
    tuple : (future_dates, predictions)
    """
    # Ensure we're working with a Series (flatten if DataFrame)
    if isinstance(price_series, pd.DataFrame):
        price_series = price_series.iloc[:, 0]
    
    # Get the last n_lags prices as a flat list
    last_prices = price_series.values[-n_lags:].flatten().tolist()
    
    # Generate future dates (excluding weekends for business days)
    last_date = price_series.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=n_days)
    
    predictions = []
    
    # Iteratively predict future prices
    for _ in range(n_days):
        # Create feature vector from last n_lags prices
        features = np.array(last_prices[-n_lags:]).reshape(1, -1)
        
        # Predict next price
        next_price = model.predict(features)[0]
        predictions.append(next_price)
        
        # Update the list of prices with the new prediction
        last_prices.append(next_price)
    
    return future_dates, np.array(predictions)


def calculate_rolling_metrics(price_series, windows=[20, 30]):
    """
    Calculate rolling metrics (mean, std, min, max) for different windows.
    
    Parameters:
    -----------
    price_series : pd.Series
        Price data
    windows : list
        List of window sizes
    
    Returns:
    --------
    dict : Dictionary of rolling metrics for each window
    """
    metrics = {}
    for window in windows:
        metrics[window] = {
            'mean': price_series.rolling(window=window).mean(),
            'std': price_series.rolling(window=window).std(),
            'min': price_series.rolling(window=window).min(),
            'max': price_series.rolling(window=window).max()
        }
    return metrics


def analyze_trading_signals(actual, predicted):
    """
    Analyze trading signal accuracy (directional prediction).
    
    Parameters:
    -----------
    actual : np.array
        Actual prices
    predicted : np.array
        Predicted prices
    
    Returns:
    --------
    dict : Trading signal statistics
    """
    # Calculate directional changes
    true_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))
    
    # Check if directions match
    correct = true_dir == pred_dir
    accuracy = np.mean(correct) * 100
    
    # Separate up and down moves
    ups = true_dir > 0
    downs = true_dir < 0
    
    up_acc = np.mean(correct[ups]) * 100 if np.any(ups) else 0
    down_acc = np.mean(correct[downs]) * 100 if np.any(downs) else 0
    
    # Rolling accuracy
    rolling_acc = pd.Series(correct.astype(float)).rolling(window=20).mean() * 100
    
    return {
        'overall_accuracy': accuracy,
        'up_accuracy': up_acc,
        'down_accuracy': down_acc,
        'correct_predictions': np.sum(correct),
        'total_predictions': len(correct),
        'rolling_accuracy': rolling_acc,
        'correct_flags': correct
    }


def detect_outliers(price_series, method='iqr'):
    """
    Detect outliers in price data using IQR or Z-score method.
    
    Parameters:
    -----------
    price_series : pd.Series
        Price data
    method : str
        'iqr' for Interquartile Range or 'zscore' for Z-score method
    
    Returns:
    --------
    dict : Outlier information including indices, values, and statistics
    """
    if method == 'iqr':
        # IQR method
        Q1 = price_series.quantile(0.25)
        Q3 = price_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (price_series < lower_bound) | (price_series > upper_bound)
        
        return {
            'outlier_mask': outlier_mask,
            'outlier_values': price_series[outlier_mask],
            'outlier_count': outlier_mask.sum(),
            'outlier_percentage': (outlier_mask.sum() / len(price_series)) * 100,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'method': 'IQR'
        }
    elif method == 'zscore':
        # Z-score method (|z| > 3)
        mean = price_series.mean()
        std = price_series.std()
        z_scores = np.abs((price_series - mean) / std)
        
        outlier_mask = z_scores > 3
        
        return {
            'outlier_mask': outlier_mask,
            'outlier_values': price_series[outlier_mask],
            'outlier_count': outlier_mask.sum(),
            'outlier_percentage': (outlier_mask.sum() / len(price_series)) * 100,
            'mean': mean,
            'std': std,
            'z_threshold': 3,
            'method': 'Z-score'
        }
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")


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


def plot_rolling_metrics(price_series, ticker, windows=[20, 30]):
    """
    Plot rolling metrics (mean, std) for multiple windows.
    """
    metrics = calculate_rolling_metrics(price_series, windows)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Rolling Mean', 'Rolling Standard Deviation'),
        vertical_spacing=0.12
    )
    
    # Plot price and rolling means
    fig.add_trace(go.Scatter(
        x=price_series.index, y=price_series.values,
        mode='lines', name='Price',
        line=dict(color='gray', width=1), opacity=0.5
    ), row=1, col=1)
    
    colors = ['blue', 'green', 'orange']
    for i, window in enumerate(windows):
        fig.add_trace(go.Scatter(
            x=metrics[window]['mean'].index,
            y=metrics[window]['mean'].values,
            mode='lines',
            name=f'{window}-Day MA',
            line=dict(color=colors[i % len(colors)], width=2)
        ), row=1, col=1)
    
    # Plot rolling standard deviations
    for i, window in enumerate(windows):
        fig.add_trace(go.Scatter(
            x=metrics[window]['std'].index,
            y=metrics[window]['std'].values,
            mode='lines',
            name=f'{window}-Day Std',
            line=dict(color=colors[i % len(colors)], width=2),
            showlegend=True
        ), row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Std Deviation ($)", row=2, col=1)
    
    fig.update_layout(
        title=f"{ticker} - Rolling Metrics Analysis",
        height=700,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def plot_trading_signals(actual, predicted, ticker):
    """
    Visualize trading signal analysis.
    """
    signals = analyze_trading_signals(actual, predicted)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cumulative Correct Predictions',
            '20-Day Rolling Accuracy',
            'Directional Accuracy by Move Type',
            'Prediction Breakdown'
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Cumulative correct predictions
    cumulative = np.cumsum(signals['correct_flags'])
    fig.add_trace(go.Scatter(
        y=cumulative,
        mode='lines',
        name='Cumulative Correct',
        line=dict(color='green', width=2),
        fill='tozeroy'
    ), row=1, col=1)
    
    # Rolling accuracy
    fig.add_trace(go.Scatter(
        y=signals['rolling_accuracy'].values,
        mode='lines',
        name='Rolling Accuracy',
        line=dict(color='purple', width=2)
    ), row=1, col=2)
    
    fig.add_hline(y=50, line_dash="dash", line_color="red",
                  annotation_text="Random (50%)", row=1, col=2)
    
    # Directional accuracy
    fig.add_trace(go.Bar(
        x=['Up Moves', 'Down Moves'],
        y=[signals['up_accuracy'], signals['down_accuracy']],
        marker_color=['green', 'red'],
        name='Accuracy',
        text=[f"{signals['up_accuracy']:.1f}%", f"{signals['down_accuracy']:.1f}%"],
        textposition='outside'
    ), row=2, col=1)
    
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Breakdown
    correct_count = signals['correct_predictions']
    wrong_count = signals['total_predictions'] - correct_count
    
    fig.add_trace(go.Bar(
        x=['Correct', 'Wrong'],
        y=[correct_count, wrong_count],
        marker_color=['lightgreen', 'lightcoral'],
        name='Count',
        text=[str(correct_count), str(wrong_count)],
        textposition='outside'
    ), row=2, col=2)
    
    fig.update_xaxes(title_text="Sample Index", row=1, col=1)
    fig.update_xaxes(title_text="Sample Index", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    fig.update_layout(
        title=f"{ticker} - Trading Signal Analysis (Overall: {signals['overall_accuracy']:.1f}%)",
        height=800,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def plot_residuals_analysis(actual, predicted, ticker):
    """
    Create residual analysis visualization.
    """
    residuals = actual - predicted
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Residuals vs Predicted',
            'Residual Distribution',
            'Residuals Over Time',
            'Absolute Residuals Over Time'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    # Residuals vs Predicted
    fig.add_trace(go.Scatter(
        x=predicted,
        y=residuals,
        mode='markers',
        marker=dict(size=5, color='steelblue', opacity=0.6),
        name='Residuals'
    ), row=1, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        marker_color='steelblue',
        name='Distribution'
    ), row=1, col=2)
    
    # Residuals over time
    fig.add_trace(go.Scatter(
        y=residuals,
        mode='lines',
        line=dict(color='coral', width=1),
        fill='tozeroy',
        name='Residuals',
        opacity=0.7
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
    
    # Absolute residuals
    fig.add_trace(go.Scatter(
        y=np.abs(residuals),
        mode='lines',
        line=dict(color='orange', width=1),
        name='|Residuals|'
    ), row=2, col=2)
    
    fig.update_xaxes(title_text="Predicted Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Residual ($)", row=1, col=2)
    fig.update_xaxes(title_text="Sample Index", row=2, col=1)
    fig.update_xaxes(title_text="Sample Index", row=2, col=2)
    
    fig.update_yaxes(title_text="Residual ($)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Residual ($)", row=2, col=1)
    fig.update_yaxes(title_text="|Residual| ($)", row=2, col=2)
    
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))
    
    fig.update_layout(
        title=f"{ticker} - Residual Analysis (Mean: ${mean_res:.2f}, Std: ${std_res:.2f})",
        height=700,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def plot_outlier_analysis(price_series, ticker, method='iqr'):
    """
    Visualize outlier detection using box plots and scatter plots.
    
    Parameters:
    -----------
    price_series : pd.Series
        Price data
    ticker : str
        Stock ticker symbol
    method : str
        'iqr' or 'zscore'
    
    Returns:
    --------
    fig : plotly figure
    """
    outlier_info = detect_outliers(price_series, method=method)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'Box Plot - {method.upper()} Method',
            'Price Over Time with Outliers Highlighted'
        ),
        horizontal_spacing=0.12
    )
    
    # Box plot
    fig.add_trace(go.Box(
        y=price_series,
        name='Price',
        marker_color='lightblue',
        boxmean='sd'
    ), row=1, col=1)
    
    # Scatter plot with outliers highlighted
    fig.add_trace(go.Scatter(
        x=price_series.index,
        y=price_series,
        mode='markers',
        name='Normal',
        marker=dict(size=5, color='steelblue', opacity=0.6)
    ), row=1, col=2)
    
    # Highlight outliers
    if outlier_info['outlier_count'] > 0:
        outlier_dates = outlier_info['outlier_values'].index
        outlier_values = outlier_info['outlier_values'].values
        
        fig.add_trace(go.Scatter(
            x=outlier_dates,
            y=outlier_values,
            mode='markers',
            name='Outliers',
            marker=dict(size=10, color='red', symbol='x', line=dict(width=2))
        ), row=1, col=2)
    
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=2)
    
    method_label = outlier_info['method']
    fig.update_layout(
        title=f"{ticker} - Outlier Detection ({method_label} Method): {outlier_info['outlier_count']} outliers ({outlier_info['outlier_percentage']:.2f}%)",
        height=500,
        template='plotly_white',
        hovermode='x unified'
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
            # For single ticker, get the scalar value
            if isinstance(adj_close.iloc[-1], pd.Series):
                latest_price = float(adj_close.iloc[-1].values[0])
            else:
                latest_price = float(adj_close.iloc[-1])
        else:
            # For multiple tickers, calculate mean
            latest_price = float(adj_close.iloc[-1].mean())
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
            # Convert to ensure we're working with a Series
            if isinstance(adj_close, pd.DataFrame):
                price_series = adj_close.iloc[:, 0]
            else:
                price_series = adj_close
            
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
                'Value': [
                    f"${float(price_series.mean()):.2f}",
                    f"${float(price_series.median()):.2f}",
                    f"${float(price_series.std()):.2f}",
                    f"${float(price_series.min()):.2f}",
                    f"${float(price_series.max()):.2f}",
                    f"${float(price_series.max() - price_series.min()):.2f}"
                ]
            })
        else:
            stats_data = []
            for ticker in selected_tickers:
                stats_data.append({
                    'Ticker': ticker,
                    'Mean': f"${float(adj_close[ticker].mean()):.2f}",
                    'Median': f"${float(adj_close[ticker].median()):.2f}",
                    'Std Dev': f"${float(adj_close[ticker].std()):.2f}",
                    'Min': f"${float(adj_close[ticker].min()):.2f}",
                    'Max': f"${float(adj_close[ticker].max()):.2f}"
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
            st.metric("20-Day Volatility", f"{float(volatility.iloc[-1]):.2f}%")
        with col2:
            st.metric("Mean Volatility", f"{float(volatility.mean()):.2f}%")
        with col3:
            st.metric("Max Volatility", f"{float(volatility.max()):.2f}%")
        
        # Rolling Metrics Analysis
        st.markdown("---")
        st.subheader(f"📈 Rolling Metrics Analysis - {analysis_ticker}")
        st.info("This section shows 20-day and 30-day rolling averages and standard deviations to help identify trends and volatility patterns.")
        
        rolling_fig = plot_rolling_metrics(ticker_data, analysis_ticker, windows=[20, 30])
        st.plotly_chart(rolling_fig, use_container_width=True)
        
        # Rolling metrics statistics
        metrics_20 = calculate_rolling_metrics(ticker_data, windows=[20])
        metrics_30 = calculate_rolling_metrics(ticker_data, windows=[30])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("20-Day MA (Latest)", f"${float(metrics_20[20]['mean'].iloc[-1]):.2f}")
        with col2:
            st.metric("20-Day Std (Latest)", f"${float(metrics_20[20]['std'].iloc[-1]):.2f}")
        with col3:
            st.metric("30-Day MA (Latest)", f"${float(metrics_30[30]['mean'].iloc[-1]):.2f}")
        with col4:
            st.metric("30-Day Std (Latest)", f"${float(metrics_30[30]['std'].iloc[-1]):.2f}")
        
        # Outlier Analysis
        st.markdown("---")
        st.subheader(f"🔍 Outlier Analysis - {analysis_ticker}")
        st.info("Outliers are data points that significantly differ from other observations. This analysis helps identify unusual price movements.")
        
        # Method selection
        outlier_method = st.radio(
            "Select outlier detection method:",
            options=['iqr', 'zscore'],
            format_func=lambda x: 'IQR (Interquartile Range)' if x == 'iqr' else 'Z-score (Standard Deviations)',
            horizontal=True
        )
        
        # Detect outliers
        outlier_info = detect_outliers(ticker_data, method=outlier_method)
        
        # Display outlier statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data Points", len(ticker_data))
        with col2:
            st.metric("Outliers Detected", outlier_info['outlier_count'])
        with col3:
            st.metric("Outlier Percentage", f"{outlier_info['outlier_percentage']:.2f}%")
        
        # Method-specific statistics
        if outlier_method == 'iqr':
            st.write("**IQR Method Statistics:**")
            iqr_col1, iqr_col2, iqr_col3, iqr_col4 = st.columns(4)
            with iqr_col1:
                st.metric("Q1 (25th percentile)", f"${float(outlier_info['Q1']):.2f}")
            with iqr_col2:
                st.metric("Q3 (75th percentile)", f"${float(outlier_info['Q3']):.2f}")
            with iqr_col3:
                st.metric("IQR (Q3 - Q1)", f"${float(outlier_info['IQR']):.2f}")
            with iqr_col4:
                st.metric("Bounds", f"${float(outlier_info['lower_bound']):.2f} - ${float(outlier_info['upper_bound']):.2f}")
        else:
            st.write("**Z-score Method Statistics:**")
            z_col1, z_col2, z_col3 = st.columns(3)
            with z_col1:
                st.metric("Mean Price", f"${float(outlier_info['mean']):.2f}")
            with z_col2:
                st.metric("Std Deviation", f"${float(outlier_info['std']):.2f}")
            with z_col3:
                st.metric("Z-score Threshold", f"±{outlier_info['z_threshold']}")
        
        # Outlier visualization
        outlier_fig = plot_outlier_analysis(ticker_data, analysis_ticker, method=outlier_method)
        st.plotly_chart(outlier_fig, use_container_width=True)
        
        # Show outlier details if any exist
        if outlier_info['outlier_count'] > 0:
            with st.expander("📋 View Outlier Details"):
                outlier_df = pd.DataFrame({
                    'Date': outlier_info['outlier_values'].index.strftime('%Y-%m-%d'),
                    'Price': [f"${val:.2f}" for val in outlier_info['outlier_values'].values]
                })
                st.dataframe(outlier_df, use_container_width=True)
        else:
            st.success("✅ No outliers detected in the price data using the selected method.")
    
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
                'Mean Daily Return (%)': f"{float(ticker_returns.mean()):.3f}",
                'Std Daily Return (%)': f"{float(ticker_returns.std()):.3f}",
                'Cumulative Return (%)': f"{float(cum_return):.2f}",
                'Best Day (%)': f"{float(ticker_returns.max()):.2f}",
                'Worst Day (%)': f"{float(ticker_returns.min()):.2f}"
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
            try:
                with st.spinner("Training model..."):
                    X_train, X_test, y_train, y_test, dates_test = prepare_regression_data(pred_data, n_lags=30)
                    results = train_and_evaluate_model(X_train, X_test, y_train, y_test)
            except ValueError as e:
                st.error(f"❌ Error preparing data: {str(e)}")
                st.warning("💡 Try selecting a longer date range to get more data for model training.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")
                st.stop()
            
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
                y_test,
                results['predictions_test'],
                pred_ticker
            )
            st.plotly_chart(pred_fig, use_container_width=True)
            
            # Residuals analysis
            st.subheader("Prediction Errors (Residuals)")
            residuals = y_test - results['predictions_test']
            
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
            # Advanced Model Analysis
            # ========================================
            st.markdown("---")
            st.subheader("🔬 Advanced Model Analysis")
            
            # Create tabs for advanced analysis
            adv_tab1, adv_tab2, adv_tab3 = st.tabs([
                "📊 Detailed Residual Analysis",
                "🎯 Trading Signal Analysis",
                "📈 Performance Metrics"
            ])
            
            with adv_tab1:
                st.info("Residual analysis helps identify patterns in prediction errors and validate model assumptions.")
                residual_analysis_fig = plot_residuals_analysis(
                    y_test,
                    results['predictions_test'],
                    pred_ticker
                )
                st.plotly_chart(residual_analysis_fig, use_container_width=True)
            
            with adv_tab2:
                st.info("Trading signal analysis evaluates how well the model predicts price movement directions, which is crucial for trading strategies.")
                signal_fig = plot_trading_signals(
                    y_test,
                    results['predictions_test'],
                    pred_ticker
                )
                st.plotly_chart(signal_fig, use_container_width=True)
                
                # Display trading signal stats
                signals = analyze_trading_signals(y_test, results['predictions_test'])
                
                st.markdown("### 📊 Signal Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Overall Accuracy", f"{signals['overall_accuracy']:.1f}%")
                with col2:
                    st.metric("Up Move Accuracy", f"{signals['up_accuracy']:.1f}%")
                with col3:
                    st.metric("Down Move Accuracy", f"{signals['down_accuracy']:.1f}%")
                with col4:
                    st.metric("Correct Predictions", 
                             f"{signals['correct_predictions']}/{signals['total_predictions']}")
            
            with adv_tab3:
                st.info("Extended performance metrics provide deeper insights into model behavior.")
                
                # Additional metrics
                mape = np.mean(np.abs((y_test - results['predictions_test']) / y_test)) * 100
                
                metrics_extended = pd.DataFrame({
                    'Metric': [
                        'Mean Absolute Error (MAE)',
                        'Root Mean Squared Error (RMSE)',
                        'R² Score',
                        'Mean Absolute Percentage Error (MAPE)',
                        'Max Error',
                        'Min Error',
                        'Mean Residual',
                        'Std Residual'
                    ],
                    'Value': [
                        f"${results['metrics']['test']['MAE']:.2f}",
                        f"${results['metrics']['test']['RMSE']:.2f}",
                        f"{results['metrics']['test']['R²']:.4f}",
                        f"{mape:.2f}%",
                        f"${float(np.max(residuals)):.2f}",
                        f"${float(np.min(residuals)):.2f}",
                        f"${float(np.mean(residuals)):.2f}",
                        f"${float(np.std(residuals)):.2f}"
                    ]
                })
                
                st.dataframe(metrics_extended, use_container_width=True, hide_index=True)
            
            # ========================================
            # Future Price Prediction
            # ========================================
            st.markdown("---")
            st.subheader("🔮 Future Price Prediction")
            st.info("⚠️ **Disclaimer:** Future predictions are based on historical patterns and should not be used as the sole basis for investment decisions. Stock markets are influenced by many unpredictable factors.")
            
            # User input for prediction range
            col1, col2 = st.columns([1, 3])
            
            with col1:
                n_days = st.slider(
                    "Days to predict:",
                    min_value=1,
                    max_value=30,
                    value=7,
                    help="Select number of business days to predict into the future"
                )
            
            with col2:
                if st.button("🚀 Generate Future Predictions", type="primary"):
                    with st.spinner("Generating predictions..."):
                        # Get future predictions
                        future_dates, future_predictions = predict_future_prices(
                            results['model'],
                            pred_data,
                            n_days,
                            n_lags=30
                        )
                        
                        # Create visualization
                        st.subheader(f"📈 {pred_ticker} - {n_days} Day Price Forecast")
                        
                        # Combine historical and future data for plotting
                        historical_dates = pred_data.index[-60:]  # Last 60 days
                        historical_prices = pred_data.values[-60:]
                        
                        future_fig = go.Figure()
                        
                        # Historical prices
                        future_fig.add_trace(go.Scatter(
                            x=historical_dates,
                            y=historical_prices,
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Predicted future prices
                        future_fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=future_predictions,
                            mode='lines+markers',
                            name='Predicted',
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(size=6)
                        ))
                        
                        # Add connecting line between last historical and first prediction
                        future_fig.add_trace(go.Scatter(
                            x=[historical_dates[-1], future_dates[0]],
                            y=[historical_prices[-1], future_predictions[0]],
                            mode='lines',
                            line=dict(color='gray', width=1, dash='dot'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        future_fig.update_layout(
                            title=f"{pred_ticker} - Price Forecast for Next {n_days} Business Days",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            hovermode='x unified',
                            height=500,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(future_fig, use_container_width=True)
                        
                        # Display prediction table
                        st.subheader("📊 Detailed Predictions")
                        
                        predictions_df = pd.DataFrame({
                            'Date': future_dates.strftime('%Y-%m-%d'),
                            'Predicted Price': [f"${price:.2f}" for price in future_predictions],
                            'Change from Previous': [''] + [f"{((future_predictions[i] - future_predictions[i-1]) / future_predictions[i-1] * 100):+.2f}%" 
                                                             for i in range(1, len(future_predictions))]
                        })
                        
                        # Add change from last known price for first prediction
                        # Ensure last_price is a scalar value
                        if isinstance(pred_data, pd.DataFrame):
                            last_price = float(pred_data.iloc[-1, 0])
                        else:
                            last_price = float(pred_data.values[-1])
                        
                        predictions_df.loc[0, 'Change from Previous'] = f"{((future_predictions[0] - last_price) / last_price * 100):+.2f}%"
                        
                        st.dataframe(predictions_df, use_container_width=True, hide_index=True)
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Starting Price",
                                f"${last_price:.2f}",
                                help="Last known historical price"
                            )
                        
                        with col2:
                            st.metric(
                                "Predicted End Price",
                                f"${future_predictions[-1]:.2f}",
                                f"{((future_predictions[-1] - last_price) / last_price * 100):+.2f}%"
                            )
                        
                        with col3:
                            predicted_high = future_predictions.max()
                            st.metric(
                                "Predicted High",
                                f"${predicted_high:.2f}",
                                f"{((predicted_high - last_price) / last_price * 100):+.2f}%"
                            )
                        
                        with col4:
                            predicted_low = future_predictions.min()
                            st.metric(
                                "Predicted Low",
                                f"${predicted_low:.2f}",
                                f"{((predicted_low - last_price) / last_price * 100):+.2f}%"
                            )
                        
                        # Trend analysis
                        overall_trend = "📈 Upward" if future_predictions[-1] > last_price else "📉 Downward"
                        volatility = np.std(np.diff(future_predictions) / future_predictions[:-1]) * 100
                        
                        st.markdown(f"""
                        **Prediction Summary:**
                        - **Overall Trend:** {overall_trend}
                        - **Predicted Volatility:** {volatility:.2f}%
                        - **Price Range:** ${predicted_low:.2f} - ${predicted_high:.2f}
                        """)
    
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
        returns_data = calculate_daily_returns(adj_close)
        
        # Handle Series vs DataFrame
        if isinstance(returns_data, pd.Series):
            returns_data = returns_data.to_frame(name='Daily_Returns_%')
        else:
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
