import os
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Create Flask app
app = Flask(__name__)
app.secret_key = 'demand_forecasting_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the CSV file
        try:
            df = pd.read_csv(filepath)
            
            # Store column names in session
            session['columns'] = df.columns.tolist()
            session['filename'] = filename
            
            # Preview data
            preview = df.head(5).to_html(classes='table table-striped table-bordered')
            
            return render_template('configure.html', 
                                  preview=preview, 
                                  columns=session['columns'])
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(request.url)
    else:
        flash('File type not allowed. Please upload a CSV file.')
        return redirect(request.url)

@app.route('/configure', methods=['POST'])
def configure():
    date_column = request.form.get('date_column')
    target_column = request.form.get('target_column')
    stock_column = request.form.get('stock_column', None)
    delivery_column = request.form.get('delivery_column', None)
    forecast_periods = int(request.form.get('forecast_periods', 12))
    model_type = request.form.get('model_type', 'arima')
    
    # Store configuration in session
    session['date_column'] = date_column
    session['target_column'] = target_column
    session['stock_column'] = stock_column
    session['delivery_column'] = delivery_column
    session['forecast_periods'] = forecast_periods
    session['model_type'] = model_type
    
    return redirect(url_for('forecast'))

@app.route('/forecast')
def forecast():
    # Get configuration from session
    filename = session.get('filename')
    date_column = session.get('date_column')
    target_column = session.get('target_column')
    stock_column = session.get('stock_column')
    delivery_column = session.get('delivery_column')
    forecast_periods = session.get('forecast_periods', 12)
    model_type = session.get('model_type', 'arima')
    
    if not all([filename, date_column, target_column]):
        flash('Missing configuration parameters')
        return redirect(url_for('index'))
    
    # Load data
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    
    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date
    df = df.sort_values(by=date_column)
    
    # Set date as index
    df.set_index(date_column, inplace=True)
    
    # Get the target data (consumption)
    data = df[target_column]
    
    # Train-test split (use last 20% for testing)
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    # Get additional data if available
    stock_data = df[stock_column] if stock_column else None
    delivery_data = df[delivery_column] if delivery_column else None
    
    # Generate material forecast
    forecast_result, forecast_dates, mape, rmse = generate_forecast(
        train, test, model_type, forecast_periods, stock_data, delivery_data
    )
    
    # Create visualization
    fig = create_forecast_plot(data, forecast_result, forecast_dates)
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Create additional visualizations if stock and delivery data are available
    stock_delivery_graph_json = None
    if stock_data is not None and delivery_data is not None:
        stock_delivery_fig = create_stock_delivery_plot(df, stock_column, delivery_column, forecast_result)
        stock_delivery_graph_json = json.dumps(stock_delivery_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('forecast.html', 
                          graph_json=graph_json,
                          stock_delivery_graph_json=stock_delivery_graph_json,
                          mape=mape,
                          rmse=rmse,
                          forecast_table=forecast_result.to_html(classes='table table-striped table-bordered', 
                                                               index_names=False, 
                                                               float_format=lambda x: f"{x:.2f}"))

def generate_forecast(train, test, model_type, forecast_periods, stock_data=None, delivery_data=None):
    """Generate forecast using the specified model for the next financial year"""
    
    # Fit model on training data
    if model_type == 'arima':
        model = ARIMA(train, order=(5,1,0))
        model_fit = model.fit()
    elif model_type == 'exponential_smoothing':
        model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
        model_fit = model.fit()
    elif model_type == 'sarima':
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
    else:
        # Default to ARIMA
        model = ARIMA(train, order=(5,1,0))
        model_fit = model.fit()
    
    # Forecast for test period
    test_forecast = model_fit.forecast(steps=len(test))
    
    # Make sure test and test_forecast have the same length and index
    if len(test) > 0 and len(test_forecast) > 0:
        # Align the indices
        test_forecast = pd.Series(test_forecast.values, index=test.index[:len(test_forecast)])
        test = test[:len(test_forecast)]
        
        # Calculate error metrics
        mape = np.mean(np.abs((test - test_forecast) / test)) * 100
        rmse = np.sqrt(mean_squared_error(test, test_forecast))
    else:
        mape = 0
        rmse = 0
    
    # Determine the next financial year dates (April 1st to March 31st)
    today = datetime.now()
    if today.month < 4:  # Before April
        next_fy_start = datetime(today.year, 4, 1)
    else:  # April or later
        next_fy_start = datetime(today.year + 1, 4, 1)
    
    next_fy_end = datetime(next_fy_start.year + 1, 3, 31)
    
    # Create forecast dates for the next financial year - use MS (Month Start) frequency
    # This will create dates for the 1st of each month from April to March
    forecast_dates = pd.date_range(start=next_fy_start, end=next_fy_end, freq='MS')
    
    # Forecast for the next financial year
    # Calculate how many periods to forecast from the last data point to the end of next FY
    last_date = train.index[-1] if len(test) == 0 else test.index[-1]
    periods_to_forecast = (next_fy_end.year - last_date.year) * 12 + (next_fy_end.month - last_date.month)
    
    # Ensure we forecast at least the number of periods in the next financial year
    if periods_to_forecast < len(forecast_dates):
        periods_to_forecast = len(forecast_dates)
    
    future_forecast = model_fit.forecast(steps=periods_to_forecast)
    
    # Extract only the forecasts for the next financial year
    # Calculate the offset from the last date to the start of next FY
    months_offset = (next_fy_start.year - last_date.year) * 12 + (next_fy_start.month - last_date.month)
    if months_offset < 0:
        months_offset = 0
    
    # Extract the relevant forecast values
    fy_forecast = future_forecast[months_offset:months_offset+len(forecast_dates)]
    
    # Ensure fy_forecast and forecast_dates have the same length
    min_length = min(len(fy_forecast), len(forecast_dates))
    fy_forecast = fy_forecast[:min_length]
    forecast_dates = forecast_dates[:min_length]
    
    # Debug print to check lengths
    print(f"Length of forecast_dates: {len(forecast_dates)}")
    print(f"Length of fy_forecast: {len(fy_forecast)}")
    
    # First create a consumption forecast dataframe (before stock adjustment)
    consumption_forecast = pd.DataFrame({
        'Consumption_Forecast': fy_forecast
    }, index=forecast_dates)
    
    # Then calculate material forecast by subtracting recent stock
    if stock_data is not None:
        # Get the most recent stock value
        latest_stock = stock_data.iloc[-1] if not stock_data.empty else 0
        print(f"Latest stock: {latest_stock}")
        
        # Calculate average monthly consumption from historical data
        avg_monthly_consumption = train.mean()
        print(f"Average monthly consumption: {avg_monthly_consumption}")
        
        # Create a material forecast that accounts for existing stock
        material_forecast = fy_forecast.copy()
        
        # Reduce the material forecast by the amount of stock available
        # Distribute the stock reduction across the first few months
        remaining_stock = latest_stock
        for i in range(len(material_forecast)):
            if remaining_stock > 0:
                # Calculate how much stock to use this month (up to the forecast amount)
                stock_used = min(remaining_stock, material_forecast[i])
                # Reduce the material forecast by the amount of stock used
                material_forecast[i] = max(0, material_forecast[i] - stock_used)
                # Update remaining stock
                remaining_stock -= stock_used
                print(f"Month {i+1}: Used {stock_used} from stock, remaining stock: {remaining_stock}")
        
        # Create a combined dataframe with both consumption forecast and material forecast
        forecast_df = pd.DataFrame({
            'Consumption_Forecast': fy_forecast,
            'Material_Forecast': material_forecast
        }, index=forecast_dates)
    else:
        # If no stock data, material forecast equals consumption forecast
        forecast_df = pd.DataFrame({
            'Consumption_Forecast': fy_forecast,
            'Material_Forecast': fy_forecast
        }, index=forecast_dates)
    
    return forecast_df, forecast_dates, mape, rmse

def create_forecast_plot(historical_data, forecast_data, forecast_dates):
    """Create a plotly visualization of historical data and forecast"""
    
    # Create dataframe for plotting
    historical_df = historical_data.reset_index()
    historical_df.columns = ['Date', 'Value']
    historical_df['Source'] = 'Historical Consumption'
    
    # Create separate dataframes for consumption and material forecasts
    consumption_forecast_df = forecast_data['Consumption_Forecast'].reset_index()
    consumption_forecast_df.columns = ['Date', 'Value']
    consumption_forecast_df['Source'] = 'Consumption Forecast'
    
    material_forecast_df = forecast_data['Material_Forecast'].reset_index()
    material_forecast_df.columns = ['Date', 'Value']
    material_forecast_df['Source'] = 'Material Forecast'
    
    # Combine data
    plot_df = pd.concat([historical_df, consumption_forecast_df, material_forecast_df])
    
    # Create plot
    fig = px.line(plot_df, x='Date', y='Value', color='Source',
                 title='Material Forecast for Next Financial Year (April 1 - March 31)',
                 labels={'Value': 'Quantity', 'Date': 'Date'})
    
    # Add vertical lines for financial year boundaries
    for year in range(historical_df['Date'].dt.year.min(), material_forecast_df['Date'].dt.year.max() + 2):
        fig.add_vline(x=f"{year}-04-01", line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add annotation explaining the difference between forecasts
    fig.add_annotation(
        x=0.5, y=1.05,
        xref="paper", yref="paper",
        text="Material Forecast = Consumption Forecast - Recent Stock",
        showarrow=False,
        font=dict(size=12),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    return fig

def create_stock_delivery_plot(df, stock_column, delivery_column, forecast_df):
    """Create a visualization of stock, delivery, and forecast data"""
    
    # Create dataframe for plotting
    stock_df = df[stock_column].reset_index()
    stock_df.columns = ['Date', 'Value']
    stock_df['Type'] = 'Recent Stock'
    
    delivery_df = df[delivery_column].reset_index()
    delivery_df.columns = ['Date', 'Value']
    delivery_df['Type'] = 'Pending Delivery'
    
    # Use material forecast for this plot
    material_forecast_df = forecast_df['Material_Forecast'].reset_index()
    material_forecast_df.columns = ['Date', 'Value']
    material_forecast_df['Type'] = 'Material Forecast'
    
    # Combine data
    plot_df = pd.concat([stock_df, delivery_df, material_forecast_df])
    
    # Create plot
    fig = px.line(plot_df, x='Date', y='Value', color='Type',
                 title='Stock, Delivery, and Material Forecast',
                 labels={'Value': 'Quantity', 'Date': 'Date'})
    
    # Add annotation explaining the material forecast
    fig.add_annotation(
        x=0.5, y=1.05,
        xref="paper", yref="paper",
        text="Material Forecast: Projected material needs after accounting for existing stock",
        showarrow=False,
        font=dict(size=12),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True) 