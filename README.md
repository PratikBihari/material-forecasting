# Material Forecasting System

![Material Forecasting](https://images.unsplash.com/photo-1460925895917-afdab827c52f?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80)

## Overview

The Material Forecasting System is an advanced web application designed to predict future material needs based on historical consumption patterns while accounting for existing stock levels. This tool helps businesses optimize inventory management, reduce carrying costs, and prevent stockouts.

Designed and developed by **Pratik Bihari**.

## Features

- **Consumption Forecasting**: Predict future consumption based on historical patterns using advanced time series models
- **Material Forecasting**: Calculate required new materials by subtracting existing stock from consumption forecasts
- **Multiple Forecasting Models**: Choose from ARIMA, Exponential Smoothing, and SARIMA models
- **Financial Year Planning**: Automatically generate forecasts for the next financial year (April 1st to March 31st)
- **Stock Optimization**: Account for current stock levels and pending deliveries
- **Interactive Visualizations**: View forecasts through beautiful, interactive charts
- **CSV Data Import**: Easily upload your historical consumption data in CSV format
- **Performance Metrics**: Evaluate forecast accuracy with MAPE and RMSE metrics

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/material-forecasting.git
   cd material-forecasting
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```
   python run.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5002
   ```

3. Upload your CSV file containing historical consumption data. The file should include:
   - A date/time column
   - A consumption/demand column
   - Recent stock levels (optional, but recommended)
   - Pending deliveries (optional)

4. Configure the forecasting parameters:
   - Select the date column
   - Select the consumption column
   - Select the stock column (if available)
   - Select the delivery column (if available)
   - Choose the forecasting model (ARIMA, Exponential Smoothing, or SARIMA)
   - Set the number of forecast periods

5. Generate and view your material forecast

## Technical Details

The application is built with:

- **Flask**: Web framework
- **Pandas**: Data manipulation and analysis
- **Statsmodels**: Time series analysis and forecasting
- **Scikit-learn**: Machine learning metrics
- **Plotly**: Interactive visualizations
- **Bootstrap**: Responsive UI design

## Forecasting Models

### ARIMA (AutoRegressive Integrated Moving Average)
Suitable for time series data with trends but no seasonality.

### Exponential Smoothing
Handles both trend and seasonality with weighted averages of past observations.

### SARIMA (Seasonal ARIMA)
Extends ARIMA to support time series with both trend and seasonal components.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Designed and developed by Pratik Bihari
- Uses open-source libraries and frameworks
- Inspired by the need for better inventory management solutions

---

© 2025 Pratik Bihari | Advanced Material Forecasting System 