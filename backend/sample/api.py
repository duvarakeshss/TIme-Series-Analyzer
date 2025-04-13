import os
import sys
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import logging
import io
import warnings
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt
import base64
from fastapi import Response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import classifier components
from classifiers.classifier import TimeSeriesClassifier
from classifiers.FeatureExtractor import FeatureExtractor

# Create FastAPI app
app = FastAPI(
    title="Time Series Analysis API",
    description="API for time series prediction, forecasting and anomaly detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifiers", "models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

# Load model paths
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')] if os.path.exists(MODEL_DIR) else []
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, model_files[0]) if model_files else None

# Load classifier on startup if model exists
classifier = None
if DEFAULT_MODEL_PATH and os.path.exists(DEFAULT_MODEL_PATH):
    try:
        classifier = TimeSeriesClassifier(model_path=DEFAULT_MODEL_PATH)
        logger.info(f"Loaded model from {DEFAULT_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")


def detect_anomalies(original_values, predicted_values, threshold=0.3):
    """
    Detect anomalies in the time series by comparing original and predicted values
    
    Args:
        original_values (list): Original values
        predicted_values (list): Predicted values
        threshold (float): Threshold for anomaly detection (relative error)
        
    Returns:
        list: List of booleans indicating anomalies
    """
    anomalies = []
    
    for orig, pred in zip(original_values, predicted_values):
        if orig == 0 or np.isnan(orig):
            # Avoid division by zero or NaN values
            is_anomaly = abs(pred) > threshold if not np.isnan(pred) else False
        else:
            # Calculate relative error
            rel_error = abs((pred - orig) / orig)
            is_anomaly = rel_error > threshold
            
        anomalies.append(is_anomaly)
        
    return anomalies


def process_time_series(df):
    """
    Process the time series data from dataframe
    
    Args:
        df (pd.DataFrame): DataFrame with time series data
        
    Returns:
        pd.Series: Time series with datetime index
    """
    # Check required columns
    timestamp_col = None
    value_col = None
    
    for col in df.columns:
        if "timestamp" in col.lower() or "date" in col.lower() or "time" in col.lower():
            timestamp_col = col
        elif "value" in col.lower() or "point" in col.lower() or "data" in col.lower():
            value_col = col
    
    if timestamp_col is None or value_col is None:
        # If we can't find appropriate columns, assume the first column is timestamp and second is value
        if len(df.columns) >= 2:
            timestamp_col = df.columns[0]
            value_col = df.columns[1]
        else:
            raise ValueError("Cannot identify timestamp and value columns in the data")
    
    # Convert timestamp to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Convert value column to numeric
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    # Create time series
    series = pd.Series(df[value_col].values, index=df[timestamp_col])
    
    # Sort by index
    series = series.sort_index()
    
    # Handle missing values
    series = series.interpolate(method='time').ffill().bfill()
    
    return series


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Time Series Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "model": DEFAULT_MODEL_PATH
    }


@app.get("/models")
async def list_models():
    """List available models"""
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')] if os.path.exists(MODEL_DIR) else []
    return {"models": model_files}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    forecast_horizon: Optional[int] = Query(30, description="Number of steps to forecast"),
    model_type: Optional[str] = Query(None, description="Model type to use (arima, sarima, ets, prophet)"),
    anomaly_threshold: Optional[float] = Query(0.3, description="Threshold for anomaly detection")
):
    """
    Process time series data from a file and return predictions with anomaly detection
    
    Args:
        file: CSV file with time series data (should have timestamp and value columns)
        forecast_horizon: Number of steps to forecast
        model_type: Type of model to use
        anomaly_threshold: Threshold for anomaly detection
        
    Returns:
        JSON response with predictions and anomaly detection
    """
    global classifier
    
    # Check if classifier is loaded
    if classifier is None:
        try:
            classifier = TimeSeriesClassifier()
            logger.info("Created new TimeSeriesClassifier")
        except Exception as e:
            logger.error(f"Error creating classifier: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error creating classifier: {str(e)}")
    
    start_time = time.time()
    
    try:
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Process time series
        time_series = process_time_series(df)
        
        # Check if time series is valid
        if len(time_series) < 10:
            raise ValueError(f"Time series too short ({len(time_series)} points). Need at least 10 points.")
            
        if time_series.nunique() <= 1:
            raise ValueError("Time series has constant values.")
        
        # Make predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Get best model if not specified
            best_model = model_type
            if best_model is None:
                best_model = classifier.select_best_time_series_model(time_series)
            
            # Generate forecasts
            forecasts = classifier.forecast_with_best_model(
                time_series, 
                horizon=forecast_horizon,
                best_model=best_model
            )
            
            # Calculate fitted values for anomaly detection
            original_values = time_series.tolist()
            fitted_values = []
            
            # For existing timestamps, use the forecast model to fit the data
            if best_model == "arima":
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(time_series, order=(1, 1, 1))
                model_fit = model.fit()
                fitted_values = model_fit.fittedvalues.tolist()
                # Fill the first values that are NaN due to differencing
                fitted_values = [original_values[0]] * (len(original_values) - len(fitted_values)) + fitted_values
            elif best_model == "sarima":
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                model_fit = model.fit(disp=False)
                fitted_values = model_fit.fittedvalues.tolist()
                # Fill the first values that are NaN due to differencing
                fitted_values = [original_values[0]] * (len(original_values) - len(fitted_values)) + fitted_values
            elif best_model == "ets":
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                model = ExponentialSmoothing(time_series, trend='add', seasonal=None, damped=True)
                model_fit = model.fit()
                fitted_values = model_fit.fittedvalues.tolist()
            elif best_model == "prophet":
                from prophet import Prophet
                df_prophet = pd.DataFrame({
                    'ds': time_series.index, 
                    'y': time_series.values
                })
                model = Prophet()
                model.fit(df_prophet)
                future = model.make_future_dataframe(periods=0)
                forecast = model.predict(future)
                fitted_values = forecast['yhat'].tolist()
            else:
                # Fallback to simple moving average
                fitted_values = time_series.rolling(window=3, min_periods=1).mean().tolist()
            
            # Detect anomalies
            anomalies = detect_anomalies(original_values, fitted_values, threshold=anomaly_threshold)
            
            # Create result objects
            results = []
            for i, (timestamp, value) in enumerate(zip(time_series.index, time_series.values)):
                predicted = fitted_values[i]
                is_anomaly = "yes" if anomalies[i] else "no"
                
                results.append({
                    "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "point_value": float(value),
                    "predicted": float(predicted),
                    "is_anomaly": is_anomaly
                })
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            with np.errstate(divide='ignore', invalid='ignore'):
                mape_values = np.abs((np.array(original_values) - np.array(fitted_values[:len(original_values)])) / np.array(original_values))
                mape_values = mape_values[~np.isnan(mape_values) & ~np.isinf(mape_values)]  # Filter out NaN and inf
                mape = np.mean(mape_values) if len(mape_values) > 0 else 0
            
            # Calculate forecastability score (inverse of variability, normalized to 0-10 scale)
            variability = np.std(time_series) / np.abs(np.mean(time_series)) if np.mean(time_series) != 0 else np.std(time_series)
            forecastability_score = min(10, 10 * np.exp(-variability))
            
            # Calculate time taken
            execution_time = time.time() - start_time
            avg_time_per_fit = execution_time / len(time_series) if len(time_series) > 0 else 0
            
            # Prepare response
            response = {
                "forecastability_score": round(forecastability_score, 1),
                "number_of_batch_fits": len(time_series),
                "mape": round(mape, 1),
                "avg_time_taken_per_fit_in_seconds": round(avg_time_per_fit, 1),
                "best_model": best_model,
                "results": results
            }
            
            return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast")
async def forecast(
    file: UploadFile = File(...),
    horizon: Optional[int] = Query(30, description="Number of steps to forecast"),
    model_type: Optional[str] = Query(None, description="Model type to use (arima, sarima, ets, prophet)"),
):
    """
    Generate forecasts for future time periods based on input time series data
    
    Args:
        file: CSV file with time series data (should have timestamp and value columns)
        horizon: Number of steps to forecast
        model_type: Type of model to use
        
    Returns:
        JSON response with forecasts
    """
    global classifier
    
    # Check if classifier is loaded
    if classifier is None:
        try:
            classifier = TimeSeriesClassifier()
            logger.info("Created new TimeSeriesClassifier")
        except Exception as e:
            logger.error(f"Error creating classifier: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error creating classifier: {str(e)}")
    
    try:
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Process time series
        time_series = process_time_series(df)
        
        # Check if time series is valid
        if len(time_series) < 10:
            raise ValueError(f"Time series too short ({len(time_series)} points). Need at least 10 points.")
            
        if time_series.nunique() <= 1:
            raise ValueError("Time series has constant values.")
        
        # Make forecast
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Get best model if not specified
            best_model = model_type
            if best_model is None:
                best_model = classifier.select_best_time_series_model(time_series)
            
            # Generate forecasts
            forecasts = classifier.forecast_with_best_model(
                time_series, 
                horizon=horizon,
                best_model=best_model
            )
            
            # Convert forecasts to list of dictionaries
            forecast_data = []
            for timestamp, value in zip(forecasts.index, forecasts.values):
                forecast_data.append({
                    "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "forecast_value": float(value)
                })
            
            # Prepare response
            response = {
                "best_model": best_model,
                "horizon": horizon,
                "forecasts": forecast_data
            }
            
            return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error during forecasting: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
):
    """
    Classify time series data based on trained model
    
    Args:
        file: CSV file with time series data (should have timestamp and value columns)
        
    Returns:
        JSON response with classification results
    """
    global classifier
    
    # Check if classifier is loaded
    if classifier is None:
        try:
            classifier = TimeSeriesClassifier()
            logger.info("Created new TimeSeriesClassifier")
        except Exception as e:
            logger.error(f"Error creating classifier: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error creating classifier: {str(e)}")
    
    try:
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Process time series
        time_series = process_time_series(df)
        
        # Check if time series is valid
        if len(time_series) < 10:
            raise ValueError(f"Time series too short ({len(time_series)} points). Need at least 10 points.")
            
        if time_series.nunique() <= 1:
            raise ValueError("Time series has constant values.")
        
        # Make classification
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Get prediction
            prediction = classifier.predict(time_series)
            
            # Get prediction probabilities
            probabilities = classifier.predict_proba(time_series)
            
            # Prepare response
            response = {
                "predicted_class": str(prediction),
                "probabilities": probabilities
            }
            
            return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plot")
async def plot_time_series(
    file: UploadFile = File(...),
    model_type: Optional[str] = Query(None, description="Model type to use (arima, sarima, ets, prophet)"),
    forecast_horizon: Optional[int] = Query(10, description="Number of steps to forecast"),
    plot_type: Optional[str] = Query("all", description="Type of plot (original, forecast, anomaly, all)")
):
    """
    Generate a plot of the time series data with optional forecasts and anomaly detection
    
    Args:
        file: CSV file with time series data
        model_type: Type of model to use
        forecast_horizon: Number of steps to forecast
        plot_type: Type of plot to generate
        
    Returns:
        HTML response with embedded plot images
    """
    global classifier
    
    # Check if classifier is loaded
    if classifier is None:
        try:
            classifier = TimeSeriesClassifier()
            logger.info("Created new TimeSeriesClassifier")
        except Exception as e:
            logger.error(f"Error creating classifier: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error creating classifier: {str(e)}")
    
    try:
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Process time series
        time_series = process_time_series(df)
        
        # Check if time series is valid
        if len(time_series) < 10:
            raise ValueError(f"Time series too short ({len(time_series)} points). Need at least 10 points.")
            
        if time_series.nunique() <= 1:
            raise ValueError("Time series has constant values.")
        
        # Make predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Get best model if not specified
            best_model = model_type
            if best_model is None:
                best_model = classifier.select_best_time_series_model(time_series)
            
            # Generate forecasts
            forecasts = classifier.forecast_with_best_model(
                time_series, 
                horizon=forecast_horizon,
                best_model=best_model
            )
            
            # Calculate fitted values for anomaly detection
            original_values = time_series.tolist()
            fitted_values = []
            
            # For existing timestamps, use the forecast model to fit the data
            if best_model == "arima":
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(time_series, order=(1, 1, 1))
                model_fit = model.fit()
                fitted_values = model_fit.fittedvalues.tolist()
                # Fill the first values that are NaN due to differencing
                fitted_values = [original_values[0]] * (len(original_values) - len(fitted_values)) + fitted_values
            elif best_model == "sarima":
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                model_fit = model.fit(disp=False)
                fitted_values = model_fit.fittedvalues.tolist()
                # Fill the first values that are NaN due to differencing
                fitted_values = [original_values[0]] * (len(original_values) - len(fitted_values)) + fitted_values
            elif best_model == "ets":
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                model = ExponentialSmoothing(time_series, trend='add', seasonal=None, damped=True)
                model_fit = model.fit()
                fitted_values = model_fit.fittedvalues.tolist()
            elif best_model == "prophet":
                from prophet import Prophet
                df_prophet = pd.DataFrame({
                    'ds': time_series.index, 
                    'y': time_series.values
                })
                model = Prophet()
                model.fit(df_prophet)
                future = model.make_future_dataframe(periods=0)
                forecast = model.predict(future)
                fitted_values = forecast['yhat'].tolist()
            else:
                # Fallback to simple moving average
                fitted_values = time_series.rolling(window=3, min_periods=1).mean().tolist()
            
            # Detect anomalies (threshold: 0.3)
            anomalies = detect_anomalies(original_values, fitted_values, threshold=0.3)
            
            # Create plots
            plt.figure(figsize=(12, 8))
            plt.style.use('ggplot')
            
            # Original data
            if plot_type in ["original", "all"]:
                plt.plot(time_series.index, time_series.values, 'b-', label='Original Data', linewidth=2)
            
            # Fitted values
            if plot_type in ["forecast", "all"]:
                plt.plot(time_series.index, fitted_values, 'g--', label=f'Fitted Values ({best_model.upper()})', linewidth=2)
            
            # Forecasts
            if plot_type in ["forecast", "all"]:
                plt.plot(forecasts.index, forecasts.values, 'r--', label=f'Forecasts ({forecast_horizon} steps)', linewidth=2)
            
            # Anomalies
            if plot_type in ["anomaly", "all"]:
                # Plot anomalies
                anomaly_indices = [i for i, is_anomaly in enumerate(anomalies) if is_anomaly]
                if anomaly_indices:
                    plt.scatter(
                        [time_series.index[i] for i in anomaly_indices],
                        [time_series.values[i] for i in anomaly_indices],
                        color='red', marker='o', s=100, label='Anomalies'
                    )
            
            # Add labels and title
            plt.title(f'Time Series Analysis with {best_model.upper()} Model', fontsize=16)
            plt.xlabel('Timestamp', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.legend(loc='best', fontsize=12)
            plt.grid(True)
            
            # Format the plot
            plt.tight_layout()
            
            # Save the plot to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            # Encode the image to base64
            plot_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Create an HTML response with the embedded image
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Time Series Plot</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        text-align: center;
                    }}
                    .container {{
                        max-width: 1000px;
                        margin: 0 auto;
                    }}
                    h1 {{
                        color: #333;
                    }}
                    .plot {{
                        margin: 20px 0;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    }}
                    .info {{
                        background-color: #f5f5f5;
                        padding: 15px;
                        border-radius: 5px;
                        margin: 20px 0;
                        text-align: left;
                    }}
                    .info-item {{
                        margin: 8px 0;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Time Series Analysis</h1>
                    
                    <div class="info">
                        <div class="info-item"><strong>Model:</strong> {best_model.upper()}</div>
                        <div class="info-item"><strong>Data Points:</strong> {len(time_series)}</div>
                        <div class="info-item"><strong>Forecast Horizon:</strong> {forecast_horizon}</div>
                        <div class="info-item"><strong>Anomalies Detected:</strong> {sum(anomalies)}</div>
                        <div class="info-item"><strong>Date Range:</strong> {time_series.index[0]} to {time_series.index[-1]}</div>
                    </div>
                    
                    <div class="plot">
                        <img src="data:image/png;base64,{plot_img}" alt="Time Series Plot">
                    </div>
                </div>
            </body>
            </html>
            """
            
            return HTMLResponse(content=html_content)
            
    except Exception as e:
        logger.error(f"Error generating plot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plot_image")
async def plot_time_series_image(
    file: UploadFile = File(...),
    model_type: Optional[str] = Query(None, description="Model type to use (arima, sarima, ets, prophet)"),
    forecast_horizon: Optional[int] = Query(10, description="Number of steps to forecast"),
    plot_type: Optional[str] = Query("all", description="Type of plot (original, forecast, anomaly, all)")
):
    """
    Generate a plot of the time series data and return it as an image
    
    Args:
        file: CSV file with time series data
        model_type: Type of model to use
        forecast_horizon: Number of steps to forecast
        plot_type: Type of plot to generate
        
    Returns:
        PNG image of the plot
    """
    global classifier
    
    # Check if classifier is loaded
    if classifier is None:
        try:
            classifier = TimeSeriesClassifier()
            logger.info("Created new TimeSeriesClassifier")
        except Exception as e:
            logger.error(f"Error creating classifier: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error creating classifier: {str(e)}")
    
    try:
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Process time series
        time_series = process_time_series(df)
        
        # Check if time series is valid
        if len(time_series) < 10:
            raise ValueError(f"Time series too short ({len(time_series)} points). Need at least 10 points.")
            
        if time_series.nunique() <= 1:
            raise ValueError("Time series has constant values.")
        
        # Make predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Get best model if not specified
            best_model = model_type
            if best_model is None:
                best_model = classifier.select_best_time_series_model(time_series)
            
            # Generate forecasts
            forecasts = classifier.forecast_with_best_model(
                time_series, 
                horizon=forecast_horizon,
                best_model=best_model
            )
            
            # Calculate fitted values for anomaly detection
            original_values = time_series.tolist()
            fitted_values = []
            
            # For existing timestamps, use the forecast model to fit the data
            if best_model == "arima":
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(time_series, order=(1, 1, 1))
                model_fit = model.fit()
                fitted_values = model_fit.fittedvalues.tolist()
                # Fill the first values that are NaN due to differencing
                fitted_values = [original_values[0]] * (len(original_values) - len(fitted_values)) + fitted_values
            elif best_model == "sarima":
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                model_fit = model.fit(disp=False)
                fitted_values = model_fit.fittedvalues.tolist()
                # Fill the first values that are NaN due to differencing
                fitted_values = [original_values[0]] * (len(original_values) - len(fitted_values)) + fitted_values
            elif best_model == "ets":
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                model = ExponentialSmoothing(time_series, trend='add', seasonal=None, damped=True)
                model_fit = model.fit()
                fitted_values = model_fit.fittedvalues.tolist()
            elif best_model == "prophet":
                from prophet import Prophet
                df_prophet = pd.DataFrame({
                    'ds': time_series.index, 
                    'y': time_series.values
                })
                model = Prophet()
                model.fit(df_prophet)
                future = model.make_future_dataframe(periods=0)
                forecast = model.predict(future)
                fitted_values = forecast['yhat'].tolist()
            else:
                # Fallback to simple moving average
                fitted_values = time_series.rolling(window=3, min_periods=1).mean().tolist()
            
            # Detect anomalies (threshold: 0.3)
            anomalies = detect_anomalies(original_values, fitted_values, threshold=0.3)
            
            # Create plots
            plt.figure(figsize=(12, 8))
            plt.style.use('ggplot')
            
            # Original data
            if plot_type in ["original", "all"]:
                plt.plot(time_series.index, time_series.values, 'b-', label='Original Data', linewidth=2)
            
            # Fitted values
            if plot_type in ["forecast", "all"]:
                plt.plot(time_series.index, fitted_values, 'g--', label=f'Fitted Values ({best_model.upper()})', linewidth=2)
            
            # Forecasts
            if plot_type in ["forecast", "all"]:
                plt.plot(forecasts.index, forecasts.values, 'r--', label=f'Forecasts ({forecast_horizon} steps)', linewidth=2)
            
            # Anomalies
            if plot_type in ["anomaly", "all"]:
                # Plot anomalies
                anomaly_indices = [i for i, is_anomaly in enumerate(anomalies) if is_anomaly]
                if anomaly_indices:
                    plt.scatter(
                        [time_series.index[i] for i in anomaly_indices],
                        [time_series.values[i] for i in anomaly_indices],
                        color='red', marker='o', s=100, label='Anomalies'
                    )
            
            # Add info text
            info_text = (
                f"Model: {best_model.upper()}\n"
                f"Data Points: {len(time_series)}\n"
                f"Forecast Horizon: {forecast_horizon}\n"
                f"Anomalies: {sum(anomalies)}\n"
                f"Date Range: {time_series.index[0].strftime('%Y-%m-%d')} to {time_series.index[-1].strftime('%Y-%m-%d')}"
            )
            plt.figtext(0.02, 0.02, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            # Add labels and title
            plt.title(f'Time Series Analysis with {best_model.upper()} Model', fontsize=16)
            plt.xlabel('Timestamp', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.legend(loc='best', fontsize=12)
            plt.grid(True)
            
            # Format the plot
            plt.tight_layout()
            
            # Save the plot to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            buf.seek(0)
            
            # Return the image directly
            return Response(content=buf.getvalue(), media_type="image/png")
            
    except Exception as e:
        logger.error(f"Error generating plot image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 