# 📈 Stock Market Analytics Dashboard

A powerful Streamlit web application for analyzing historical stock market data with interactive visualizations, technical indicators, and predictive modeling.

## ✨ Features

- **Multi-Stock Selection**: Analyze multiple stocks simultaneously
- **Custom Date Ranges**: Select any historical date range
- **Interactive Charts**: Powered by Plotly for full interactivity
- **Technical Indicators**:
  - Moving Averages (20-day, 50-day)
  - Daily Returns
  - Volatility Metrics
- **Correlation Analysis**: Heatmaps showing relationships between stocks
- **Predictive Modeling**: Linear regression model for price prediction
- **Performance Metrics**: MAE, MSE, RMSE, R² scores
- **Data Export**: Download analyzed data as CSV
- **Clean UI**: Organized with tabs and sidebar controls

## 🚀 Quick Start

### Installation

1. **Clone the repository** (or navigate to your project folder):
   ```bash
   cd StockPrediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install streamlit yfinance pandas numpy plotly scikit-learn
   ```

### Running the App

Run the following command in your terminal:

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## 📖 How to Use

1. **Select Stocks**: Use the sidebar to choose one or multiple stock tickers
2. **Set Date Range**: Pick start and end dates for your analysis
3. **Configure Options**: Toggle advanced features like regression analysis
4. **Fetch Data**: Click the "Fetch Data" button to load stock data
5. **Explore Tabs**: Navigate through different analysis sections
6. **Download Results**: Export your data using the download buttons

## 📊 Dashboard Sections

### 1. Price Charts
- Historical price trends
- Summary statistics for each stock
- Comparison across multiple stocks

### 2. Technical Analysis
- Moving averages overlay
- Daily returns visualization
- Volatility metrics

### 3. Correlation Analysis
- Correlation heatmap
- Highest/lowest correlated pairs
- Multi-stock relationships

### 4. Returns Analysis
- Cumulative returns comparison
- Returns statistics table
- Best/worst day performance

### 5. Predictive Model
- Linear regression predictions
- Model performance metrics
- Actual vs predicted visualization
- Residuals analysis

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Source**: Yahoo Finance (via yfinance)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Machine Learning**: Scikit-learn

## 📋 Default Stock Tickers

- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Google)
- NVDA (Nvidia)
- META (Meta)
- TSLA (Tesla)
- AMZN (Amazon)

## ⚙️ Configuration

You can modify the default tickers in `app.py`:

```python
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMZN']
```

## 🔧 Troubleshooting

### Port Already in Use
If port 8501 is busy, specify a different port:
```bash
streamlit run app.py --server.port 8502
```

### Data Not Loading
- Check your internet connection
- Verify ticker symbols are correct
- Ensure date range is valid (not future dates)

### Installation Issues
If you encounter issues with specific packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt --upgrade
```

## 📝 Notes

- Data is cached for 1 hour to improve performance
- Large date ranges may take longer to load
- Regression model uses 30-day lagged features
- Test set is 20% of available data

## ⚠️ Disclaimer

This application is for **educational purposes only**. It is not intended as financial advice. Always conduct your own research and consult with financial professionals before making investment decisions.

## 📄 License

This project is open source and available for educational use.

## 🤝 Contributing

Suggestions and improvements are welcome! Feel free to fork and submit pull requests.

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Built with ❤️ using Streamlit | Data powered by Yahoo Finance**
