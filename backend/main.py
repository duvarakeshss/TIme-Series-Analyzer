import os
import sys
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Body
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import logging
import io
import warnings
import time
from datetime import datetime, timedelta
import base64
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import uvicorn
from classifiers.classifier import TimeSeriesClassifier
from classifiers.FeatureExtractor import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Create FastAPI app
app = FastAPI(
    title="Time Series Analysis API",
    description="API for time series forecasting and anomaly detection",
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
else:
    try:
        classifier = TimeSeriesClassifier()
        logger.info("Created new TimeSeriesClassifier without pre-trained model")
    except Exception as e:
        logger.error(f"Error creating classifier: {str(e)}")


def process_time_series(df):
    """Process DataFrame into time series with datetime index"""
    timestamp_col = None
    value_col = None
    
    for col in df.columns:
        if "timestamp" in col.lower() or "date" in col.lower() or "time" in col.lower():
            timestamp_col = col
        elif "value" in col.lower() or "point" in col.lower() or "data" in col.lower():
            value_col = col
    
    if timestamp_col is None or value_col is None:
        if len(df.columns) >= 2:
            timestamp_col = df.columns[0]
            value_col = df.columns[1]
        else:
            raise ValueError("Cannot identify timestamp and value columns in the data")
    
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    series = pd.Series(df[value_col].values, index=df[timestamp_col])
    series = series.sort_index()
    series = series.interpolate(method='time').ffill().bfill()
    
    return series


def infer_time_series_frequency(time_series):
    """Infer the frequency of a time series"""
    freq = pd.infer_freq(time_series.index)
    if freq is None:
        if len(time_series) >= 2:
            avg_seconds = (time_series.index[-1] - time_series.index[0]).total_seconds() / (len(time_series) - 1)
            if avg_seconds >= 86400:  # daily
                freq = 'D'
            elif avg_seconds >= 3600:  # hourly
                freq = 'H'
            else:  # minutes
                freq = 'T'
        else:
            freq = 'D'  # default
    return freq


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Time Series Analysis API",
        "version": "1.0.0",
        "docs": "/docs"
    }


def generate_plot(time_series, forecast_data=None, anomalies=None, past_forecast=None, plot_type="matplotlib"):
    
    """
    Generate a plot of time series data with optional forecasts and anomalies
    
    Args:
        time_series: Time series data
        forecast_data: Future forecast data points (optional)
        anomalies: List of anomaly points (optional)
        past_forecast: Past forecast data points (optional)
        plot_type: "matplotlib" or "plotly"
    
    Returns:
        encoded_image: Base64 encoded image or plotly JSON
    """
    
    try:
        if plot_type == "plotly":
            fig = go.Figure()
            
            # Ensure all values are numeric
            y_values = []
            for val in time_series.values:
                try:
                    y_values.append(float(val))
                except (ValueError, TypeError):
                    y_values.append(None)  # Use None for non-numeric values
            
            # Convert all timestamp indices to strings to avoid type mixing
            x_values = [str(idx) for idx in time_series.index]
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,  # Use converted numeric values
                mode='lines',
                name='Historical Data',
                line=dict(color='blue', width=2)
            ))
            
            # Add forecast if available
            if forecast_data is not None:
                try:
                    # Convert all timestamps to strings and values to float
                    forecast_x = []
                    forecast_y = []
                    
                    for item in forecast_data:
                        try:
                            # Always convert timestamp to string to avoid type mixing
                            forecast_x.append(str(item['timestamp']))
                            
                            # Ensure value is numeric
                            if isinstance(item['value'], str):
                                forecast_y.append(float(item['value']))
                            else:
                                forecast_y.append(float(item['value']))
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Skipping forecast point due to conversion error: {e}")
                            continue
                    
                    # Add a vertical line to mark the forecast start
                    if len(time_series.index) > 0 and len(forecast_x) > 0:
                        # Make sure we're using string for the index
                        last_timestamp = str(time_series.index[-1])
                        fig.add_vline(
                            x=last_timestamp, 
                            line=dict(color="gray", width=1, dash="dash"),
                            annotation_text="Forecast Start",
                            annotation_position="top right"
                        )
                    
                    # Only add trace if we have valid data
                    if forecast_x and forecast_y:
                        fig.add_trace(go.Scatter(
                            x=forecast_x,
                            y=forecast_y,
                            mode='lines',
                            name='Future Forecast',
                            line=dict(color='green', width=2.5, dash='dash')
                        ))
                except Exception as e:
                    logger.error(f"Error adding forecast to plot: {str(e)}")
            
            # Add past forecast if available
            if past_forecast is not None:
                try:
                    # Convert all timestamps to strings and values to float
                    past_x = []
                    past_y = []
                    
                    for item in past_forecast:
                        try:
                            # Always convert timestamp to string to avoid type mixing
                            past_x.append(str(item['timestamp']))
                            
                            # Ensure value is numeric
                            if isinstance(item['value'], str):
                                past_y.append(float(item['value']))
                            else:
                                past_y.append(float(item['value']))
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Skipping past forecast point due to conversion error: {e}")
                            continue
                    
                    # Only add trace if we have valid data
                    if past_x and past_y:
                        fig.add_trace(go.Scatter(
                            x=past_x,
                            y=past_y,
                            mode='lines',
                            name='Past Forecast',
                            line=dict(color='purple', dash='dash')
                        ))
                except Exception as e:
                    logger.error(f"Error adding past forecast to plot: {str(e)}")
            
            # Add anomalies if available
            if anomalies is not None:
                try:
                    # Convert all timestamps to strings and values to float
                    anomaly_x = []
                    anomaly_y = []
                    
                    for item in anomalies:
                        if item.get('is_anomaly') == 'yes':
                            try:
                                # Always convert timestamp to string to avoid type mixing
                                anomaly_x.append(str(item['timestamp']))
                                
                                # Ensure point_value is numeric
                                if isinstance(item['point_value'], str):
                                    anomaly_y.append(float(item['point_value']))
                                else:
                                    anomaly_y.append(float(item['point_value']))
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Skipping anomaly point due to conversion error: {e}")
                                continue
                    
                    # Only add trace if we have valid data
                    if anomaly_x and anomaly_y:
                        fig.add_trace(go.Scatter(
                            x=anomaly_x,
                            y=anomaly_y,
                            mode='markers',
                            name='Anomalies',
                            marker=dict(color='red', size=10, symbol='circle')
                        ))
                except Exception as e:
                    logger.error(f"Error adding anomalies to plot: {str(e)}")
            
            fig.update_layout(
                title='Time Series Analysis',
                xaxis_title='Time',
                yaxis_title='Value',
                legend_title='Legend',
                template='plotly_white',
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Convert to JSON and ensure all values are serializable
            import json
            plot_json = fig.to_json()
            # Validate JSON
            try:
                # Additional validation and formatting for client compatibility
                plot_data = json.loads(plot_json)
                
                # Ensure data and layout are properly structured
                if 'data' not in plot_data or not isinstance(plot_data['data'], list):
                    logger.warning("Plotly data structure missing 'data' array")
                    # Add empty data array if missing
                    if 'data' not in plot_data:
                        plot_data['data'] = []
                
                # Ensure layout exists
                if 'layout' not in plot_data:
                    plot_data['layout'] = {
                        'title': 'Time Series Analysis',
                        'height': 500,
                        'margin': {'l': 50, 'r': 20, 't': 50, 'b': 50}
                    }
                
                # Convert back to JSON
                plot_json = json.dumps(plot_data)
                
                logger.info(f"Generated valid Plotly JSON with {len(plot_data.get('data', []))} traces")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in plot data: {e}")
                # Create a simple error plot
                error_plot = {
                    "data": [{"type": "scatter", "x": [], "y": []}],
                    "layout": {"title": f"Error generating plot: {str(e)}"}
                }
                plot_json = json.dumps(error_plot)
            
            return plot_json
        else:  # matplotlib
            plt.figure(figsize=(12, 6))
            
            # Ensure all values are numeric
            y_values = []
            x_values = []
            for i, val in enumerate(time_series.values):
                try:
                    y_values.append(float(val))
                    x_values.append(time_series.index[i])
                except (ValueError, TypeError):
                    continue  # Skip non-numeric values
                    
            # Plot historical data
            plt.plot(x_values, y_values, 'b-', label='Historical Data')
            
            # Plot forecast if available
            if forecast_data is not None:
                try:
                    forecast_x = []
                    forecast_y = []
                    
                    for item in forecast_data:
                        try:
                            # Convert timestamp string to datetime
                            x = pd.to_datetime(item['timestamp'])
                            
                            # Ensure value is numeric
                            if isinstance(item['value'], str):
                                y = float(item['value'])
                            else:
                                y = float(item['value'])
                                
                            forecast_x.append(x)
                            forecast_y.append(y)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Skipping forecast point due to conversion error: {e}")
                            continue
                    
                    if forecast_x and forecast_y:
                        plt.plot(forecast_x, forecast_y, 'g--', label='Future Forecast', linewidth=2)
                    
                    # Add a vertical line to mark the forecast start
                    if len(time_series.index) > 0:
                        plt.axvline(x=time_series.index[-1], color='gray', linestyle='--', alpha=0.7)
                        plt.text(time_series.index[-1], plt.ylim()[1], 'Forecast Start', 
                                ha='right', va='top', color='gray', alpha=0.7)
                except Exception as e:
                    logger.error(f"Error adding forecast to matplotlib plot: {str(e)}")
            
            # Plot past forecast if available
            if past_forecast is not None:
                try:
                    past_x = []
                    past_y = []
                    
                    for item in past_forecast:
                        try:
                            # Convert timestamp string to datetime 
                            x = pd.to_datetime(item['timestamp'])
                            
                            # Ensure value is numeric
                            if isinstance(item['value'], str):
                                y = float(item['value'])
                            else:
                                y = float(item['value'])
                                
                            past_x.append(x)
                            past_y.append(y)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Skipping past forecast point due to conversion error: {e}")
                            continue
                    
                    if past_x and past_y:
                        plt.plot(past_x, past_y, 'm--', label='Past Forecast')
                except Exception as e:
                    logger.error(f"Error adding past forecast to matplotlib plot: {str(e)}")
            
            # Plot anomalies if available
            if anomalies is not None:
                try:
                    anomaly_x = []
                    anomaly_y = []
                    
                    for item in anomalies:
                        if item.get('is_anomaly') == 'yes':
                            try:
                                # Convert timestamp string to datetime
                                x = pd.to_datetime(item['timestamp'])
                                
                                # Ensure point_value is numeric
                                if isinstance(item['point_value'], str):
                                    y = float(item['point_value'])
                                else:
                                    y = float(item['point_value'])
                                    
                                anomaly_x.append(x)
                                anomaly_y.append(y)
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Skipping anomaly point due to conversion error: {e}")
                                continue
                    
                    if anomaly_x and anomaly_y:  # Only plot if there are anomalies
                        plt.scatter(anomaly_x, anomaly_y, color='red', s=50, label='Anomalies')
                except Exception as e:
                    logger.error(f"Error adding anomalies to matplotlib plot: {str(e)}")
            
            plt.title('Time Series Analysis')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            # Encode
            buf.seek(0)
            encoded_image = base64.b64encode(buf.read()).decode('utf-8')
            
            return encoded_image
    except Exception as e:
        logger.error(f"Error in generate_plot: {str(e)}")
        if plot_type == "plotly":
            # Return a simple fallback plot
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error generating plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig.to_json()
        else:
            # Generate a simple error image
            plt.figure(figsize=(8, 4))
            plt.text(0.5, 0.5, f"Error generating plot: {str(e)}", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')


@app.post("/plot")
async def get_plot(
    file: Optional[UploadFile] = File(None),
    data: Optional[List[Dict[str, Any]]] = Body(None),
    horizon: Optional[int] = Query(30, description="Number of steps to forecast"),
    detect_anomalies: Optional[bool] = Query(False, description="Detect anomalies in the data"),
    past_horizon: Optional[int] = Query(0, description="Number of steps to forecast into the past"),
    plot_type: Optional[str] = Query("matplotlib", description="Plot type (matplotlib or plotly)")
):
    """Generate time series plot with optional forecasts and anomalies"""
    try:
        if file is not None:
            content = await file.read()
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif data is not None:
            time_data = [item["timestamp"] for item in data]
            value_data = [item["point_value"] for item in data]
            
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(time_data),
                "point_value": value_data
            })
        else:
            raise HTTPException(status_code=400, detail="Either file or data must be provided")
        
        time_series = process_time_series(df)
        
        if len(time_series) < 10:
            raise ValueError(f"Time series too short ({len(time_series)} points). Need at least 10 points.")
            
        if time_series.nunique() <= 1:
            raise ValueError("Time series has constant values.")
        
        # Variables to store forecast and anomaly data
        forecast_data = None
        past_forecast_data = None
        anomaly_results = None
        
        # Calculate future forecast if requested
        if horizon > 0:
            alpha = 0.3
            smoothed_values = [time_series.iloc[0]]
            for i in range(1, len(time_series)):
                smoothed_value = alpha * time_series.iloc[i] + (1 - alpha) * smoothed_values[-1]
                smoothed_values.append(smoothed_value)
            
            n_for_trend = min(10, len(time_series) // 3)
            if n_for_trend < 2:
                n_for_trend = 2
                
            last_n_values = smoothed_values[-n_for_trend:]
            trend = (last_n_values[-1] - last_n_values[0]) / (n_for_trend - 1)
            
            forecast_values = []
            last_value = smoothed_values[-1]
            
            for i in range(horizon):
                next_value = last_value + trend
                forecast_values.append(next_value)
                last_value = next_value
            
            freq = infer_time_series_frequency(time_series)
            
            # Create forecast dates with explicit timestamp conversion
            try:
                # Start date for forecast (make sure it's a datetime)
                start_date = pd.to_datetime(time_series.index[-1])
                
                # Add a small time delta to avoid duplicates
                start_date = start_date + pd.Timedelta(seconds=1)
                
                # Generate date range
                forecast_dates = pd.date_range(
                    start=start_date,
                    periods=horizon,
                    freq=freq
                )
                
                # Convert all dates to string in ISO format for consistency
                forecast_dates_str = [d.isoformat() for d in forecast_dates]
                
                # Create forecast data with explicit string dates and float values
                forecast_data = []
                for ts, val in zip(forecast_dates_str, forecast_values):
                    forecast_data.append({
                        "timestamp": ts,
                        "value": float(val)
                    })
                
                logger.info(f"Generated {len(forecast_data)} forecast points")
            except Exception as e:
                logger.error(f"Error generating forecast dates: {str(e)}")
                forecast_data = None
        
        # Calculate past forecast if requested
        if past_horizon > 0:
            x = np.arange(len(time_series))
            y = time_series.values
            slope, intercept = np.polyfit(x, y, 1)
            
            # Create past forecast dates with explicit timestamp conversion
            try:
                # First date of the time series (ensure it's datetime)
                first_date = pd.to_datetime(time_series.index[0])
                
                # Subtract a small time delta to avoid duplicates
                first_date = first_date - pd.Timedelta(seconds=1)
                
                freq = infer_time_series_frequency(time_series)
                
                # Generate date range for past
                past_dates = pd.date_range(
                    end=first_date,
                    periods=past_horizon,
                    freq=freq
                )
                
                # Convert all dates to string in ISO format
                past_dates_str = [d.isoformat() for d in past_dates]
                
                # Calculate values for past forecast
                past_values = []
                for i in range(past_horizon):
                    steps_from_start = past_horizon - i
                    past_values.append(intercept + slope * (-steps_from_start))
                
                # Create past forecast data with explicit string dates and float values
                past_forecast_data = []
                for ts, val in zip(past_dates_str, past_values):
                    past_forecast_data.append({
                        "timestamp": ts,
                        "value": float(val)
                    })
                
                logger.info(f"Generated {len(past_forecast_data)} past forecast points")
            except Exception as e:
                logger.error(f"Error generating past forecast dates: {str(e)}")
                past_forecast_data = None
        
        # Calculate anomalies if requested
        if detect_anomalies:
            window_size = min(5, len(time_series) // 3)
            if window_size < 2:
                window_size = 2
                
            fitted_values = time_series.rolling(window=window_size, min_periods=1).mean().tolist()
            
            std_dev = np.std(time_series.values)
            threshold = 2.0
            
            # Process anomalies with explicit timestamp conversion
            try:
                anomaly_results = []
                
                # Convert timestamps to ISO format strings
                for i, (ts, actual, predicted) in enumerate(zip(time_series.index, time_series.values, fitted_values)):
                    # Convert timestamp to string in ISO format
                    ts_str = pd.to_datetime(ts).isoformat()
                    
                    # Ensure values are numeric
                    try:
                        actual_val = float(actual)
                        predicted_val = float(predicted)
                        
                        # Calculate error and check if it's an anomaly
                        error = abs(actual_val - predicted_val)
                        is_anomaly = "yes" if error > threshold * std_dev else "no"
                        
                        # Add to results
                        anomaly_results.append({
                            "timestamp": ts_str,
                            "point_value": actual_val,
                            "predicted": predicted_val,
                            "is_anomaly": is_anomaly
                        })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping anomaly at index {i} due to conversion error: {e}")
                
                logger.info(f"Generated {len(anomaly_results)} anomaly data points")
            except Exception as e:
                logger.error(f"Error generating anomalies: {str(e)}")
                anomaly_results = None
        
        # Generate plot
        plot_data = generate_plot(
            time_series=time_series, 
            forecast_data=forecast_data, 
            anomalies=anomaly_results,
            past_forecast=past_forecast_data,
            plot_type=plot_type
        )
        
        if plot_type == "plotly":
            return JSONResponse({
                "plot_type": "plotly",
                "plot_data": plot_data
            })
        else:
            return JSONResponse({
                "plot_type": "image",
                "image_data": f"data:image/png;base64,{plot_data}"
            })
            
    except Exception as e:
        logger.error(f"Error generating plot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze(
    file: Optional[UploadFile] = File(None),
    data: Optional[List[Dict[str, Any]]] = Body(None),
    horizon: Optional[int] = Query(30, description="Number of steps to forecast"),
    detect_anomalies: Optional[bool] = Query(True, description="Detect anomalies in the data"),
    include_plot: Optional[bool] = Query(False, description="Include plot data in response"),
    plot_type: Optional[str] = Query("matplotlib", description="Plot type (matplotlib or plotly)"),
    simplify: Optional[bool] = Query(False, description="Simplify data for faster processing")
):
    """Time series analysis with forecasting and optional anomaly detection"""
    try:
        start_time = time.time()
        
        if file is not None:
            content = await file.read()
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif data is not None:
            time_data = [item["timestamp"] for item in data]
            value_data = [item["point_value"] for item in data]
            
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(time_data),
                "point_value": value_data
            })
        else:
            raise HTTPException(status_code=400, detail="Either file or data must be provided")
        
        # Handle large datasets by downsampling if requested
        original_size = len(df)
        if simplify and original_size > 500:
            sample_size = min(500, original_size // 2)
            logger.info(f"Simplifying dataset from {original_size} to {sample_size} points")
            
            # Instead of random sampling, take evenly spaced rows to preserve time series pattern
            step = original_size // sample_size
            indices = list(range(0, original_size, step))
            if len(indices) > sample_size:
                indices = indices[:sample_size]
            
            df = df.iloc[indices].reset_index(drop=True)
            logger.info(f"Dataset simplified to {len(df)} points")
        
        time_series = process_time_series(df)
        
        if len(time_series) < 10:
            raise ValueError(f"Time series too short ({len(time_series)} points). Need at least 10 points.")
            
        if time_series.nunique() <= 1:
            raise ValueError("Time series has constant values.")
        
        # Always do anomaly detection regardless of the detect_anomalies parameter
        batch_fits = 0
        total_fit_time = 0
        fit_start_time = time.time()
        
        window_size = min(5, len(time_series) // 3)
        if window_size < 2:
            window_size = 2
                
        fitted_values = time_series.rolling(window=window_size, min_periods=1).mean().tolist()
        batch_fits += 1
        
        fit_end_time = time.time()
        total_fit_time += (fit_end_time - fit_start_time)
        avg_time_per_fit = total_fit_time / max(1, batch_fits)
        
        std_dev = np.std(time_series.values)
        threshold = 2.0
        
        # For large datasets, analyze fewer points to improve performance
        results_limit = len(time_series)
        if simplify and results_limit > 500:
            results_limit = 500
            logger.info(f"Limiting anomaly results to {results_limit} points")
        
        anomaly_results = []
        for i, (ts, actual, predicted) in enumerate(list(zip(time_series.index, time_series.values, fitted_values))[:results_limit]):
            error = abs(actual - predicted)
            is_anomaly = "yes" if error > threshold * std_dev else "no"
            
            anomaly_results.append({
                "timestamp": str(ts),
                "point_value": float(actual),
                "predicted": float(predicted),
                "is_anomaly": is_anomaly
            })
        
        actual_array = np.array(time_series.values)
        predicted_array = np.array(fitted_values)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape_array = np.abs((actual_array - predicted_array) / actual_array)
            mape_array = mape_array[~np.isnan(mape_array) & ~np.isinf(mape_array)]
            mape = np.mean(mape_array) if len(mape_array) > 0 else 0
        
        r_squared = 1 - (np.sum((actual_array - predicted_array)**2) / 
                        np.sum((actual_array - np.mean(actual_array))**2))
        forecastability_score = max(0, min(10, 10 * r_squared))
        
        # Generate forecast data
        alpha = 0.3
        smoothed_values = [time_series.iloc[0]]
        for i in range(1, len(time_series)):
            smoothed_value = alpha * time_series.iloc[i] + (1 - alpha) * smoothed_values[-1]
            smoothed_values.append(smoothed_value)
        
        n_for_trend = min(10, len(time_series) // 3)
        if n_for_trend < 2:
            n_for_trend = 2
            
        last_n_values = smoothed_values[-n_for_trend:]
        trend = (last_n_values[-1] - last_n_values[0]) / (n_for_trend - 1)
        
        forecast_values = []
        last_value = smoothed_values[-1]
        
        for i in range(horizon):
            next_value = last_value + trend
            forecast_values.append(next_value)
            last_value = next_value
        
        freq = infer_time_series_frequency(time_series)
        
        # Generate forecast dates with proper ISO formatting
        try:
            # Start date for forecast (make sure it's a datetime)
            start_date = pd.to_datetime(time_series.index[-1])
            
            # Add a small time delta to avoid duplicates
            start_date = start_date + pd.Timedelta(seconds=1)
            
            # Generate date range
            forecast_dates = pd.date_range(
                start=start_date,
                periods=horizon,
                freq=freq
            )
            
            # Convert all dates to string in ISO format for consistency
            forecast_dates_str = [d.isoformat() for d in forecast_dates]
            
            # Create forecast data with explicit string dates and float values
            forecast_data = []
            for ts, val in zip(forecast_dates_str, forecast_values):
                forecast_data.append({
                    "timestamp": ts,
                    "value": float(val)
                })
            
            logger.info(f"Generated {len(forecast_data)} forecast points in analyze endpoint")
        except Exception as e:
            logger.error(f"Error generating forecast dates in analyze endpoint: {str(e)}")
            # Fall back to simple string conversion if ISO format fails
            forecast_data = [{"timestamp": str(ts), "value": float(val)} for ts, val in zip(forecast_dates, forecast_values)]
        
        result = {
            "forecastability_score": round(forecastability_score, 1),
            "number_of_batch_fits": batch_fits,
            "mape": round(mape, 1),
            "avg_time_taken_per_fit_in_seconds": round(avg_time_per_fit, 1),
            "results": anomaly_results,
            "forecast": forecast_data
        }
        
        # Include plot if requested
        if include_plot:
            # For large datasets, always simplify the plot data
            if simplify or len(time_series) > 1000:
                # Downsample for plotting while maintaining pattern
                logger.info("Simplifying plot data for better performance")
                # Determine how many points to sample (max 300 for smooth plot without overwhelming browser)
                sample_size = min(300, len(time_series))
                step = max(1, len(time_series) // sample_size)
                
                # Create simplified series for plotting
                plot_indices = list(range(0, len(time_series), step))
                if plot_indices[-1] != len(time_series) - 1:
                    plot_indices.append(len(time_series) - 1)  # Always include last point
                
                plot_series = pd.Series(
                    [time_series.iloc[i] for i in plot_indices],
                    index=[time_series.index[i] for i in plot_indices]
                )
                
                # Only highlight key anomalies in the plot
                plot_anomalies = None
                if anomaly_results:
                    anomalies_yes = [item for item in anomaly_results if item['is_anomaly'] == 'yes']
                    if len(anomalies_yes) > 20:
                        # Too many anomalies, only show the most significant ones
                        anomalies_yes.sort(key=lambda x: abs(float(x['point_value']) - float(x['predicted'])), reverse=True)
                        plot_anomalies = anomalies_yes[:20]  # Top 20 anomalies
                    else:
                        plot_anomalies = anomalies_yes
                
                plot_data = generate_plot(
                    time_series=plot_series,
                    anomalies=plot_anomalies,
                    forecast_data=forecast_data,
                    plot_type=plot_type
                )
            else:
                plot_data = generate_plot(
                    time_series=time_series, 
                    anomalies=anomaly_results,
                    forecast_data=forecast_data,
                    plot_type=plot_type
                )
            
            if plot_type == "plotly":
                result["plot"] = {
                    "plot_type": "plotly",
                    "plot_data": plot_data
                }
            else:
                result["plot"] = {
                    "plot_type": "image",
                    "image_data": f"data:image/png;base64,{plot_data}"
                }
        
        return JSONResponse(result)
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast")
async def forecast(
    file: Optional[UploadFile] = File(None),
    data: Optional[List[Dict[str, Any]]] = Body(None),
    past_horizon: Optional[int] = Query(30, description="Number of steps to forecast into the past"),
    future_horizon: Optional[int] = Query(30, description="Number of steps to forecast into the future"),
    include_plot: Optional[bool] = Query(False, description="Include plot data in response"),
    plot_type: Optional[str] = Query("matplotlib", description="Plot type (matplotlib or plotly)")
):
    """Generate forecasts for both past and future time periods"""
    try:
        if file is not None:
            content = await file.read()
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif data is not None:
            time_data = [item["timestamp"] for item in data]
            value_data = [item["point_value"] for item in data]
            
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(time_data),
                "point_value": value_data
            })
        else:
            raise HTTPException(status_code=400, detail="Either file or data must be provided")
        
        time_series = process_time_series(df)
        
        if len(time_series) < 10:
            raise ValueError(f"Time series too short ({len(time_series)} points). Need at least 10 points.")
            
        if time_series.nunique() <= 1:
            raise ValueError("Time series has constant values.")
        
        feature_dict = {
            "mean": float(np.mean(time_series.values)),
            "std": float(np.std(time_series.values)),
            "min": float(np.min(time_series.values)),
            "max": float(np.max(time_series.values)),
            "median": float(np.median(time_series.values)),
        }
        
        freq = infer_time_series_frequency(time_series)
        
        x = np.arange(len(time_series))
        y = time_series.values
        slope, intercept = np.polyfit(x, y, 1)
        
        future_values = []
        for i in range(future_horizon):
            future_values.append(intercept + slope * (len(time_series) + i))
        
        # Generate future dates with explicit timestamp conversion
        try:
            # Ensure last timestamp is datetime
            last_timestamp = pd.to_datetime(time_series.index[-1])
            
            # Add small time delta to avoid duplicates
            start_date = last_timestamp + pd.Timedelta(seconds=1)
            
            freq = infer_time_series_frequency(time_series)
            
            future_dates = pd.date_range(
                start=start_date,
                periods=future_horizon,
                freq=freq
            )
            
            # Convert dates to ISO format strings
            future_dates_str = [d.isoformat() for d in future_dates]
            
            # Create forecast data with explicit types
            future_forecast_data = []
            for ts, val in zip(future_dates_str, future_values):
                future_forecast_data.append({
                    "timestamp": ts,
                    "value": float(val)
                })
            
            logger.info(f"Generated {len(future_forecast_data)} future forecast points")
        except Exception as e:
            logger.error(f"Error generating future forecast dates: {str(e)}")
            future_forecast_data = []
        
        first_date = time_series.index[0]
        
        # Generate past dates with explicit timestamp conversion
        try:
            # Ensure first timestamp is datetime
            first_timestamp = pd.to_datetime(time_series.index[0])
            
            # Subtract small time delta to avoid duplicates
            end_date = first_timestamp - pd.Timedelta(seconds=1)
            
            past_dates = pd.date_range(
                end=end_date,
                periods=past_horizon,
                freq=freq
            )
            
            # Calculate past values
            past_values = []
            for i in range(past_horizon):
                steps_from_start = past_horizon - i
                past_values.append(intercept + slope * (-steps_from_start))
            
            # Convert dates to ISO format strings
            past_dates_str = [d.isoformat() for d in past_dates]
            
            # Create past forecast data with explicit types
            past_forecast_data = []
            for ts, val in zip(past_dates_str, past_values):
                past_forecast_data.append({
                    "timestamp": ts, 
                    "value": float(val)
                })
        except Exception as e:
            logger.error(f"Error generating past forecast dates: {str(e)}")
            past_forecast_data = []
        
        # Convert historical data with explicit timestamp handling
        try:
            historical_data = []
            for ts, val in zip(time_series.index, time_series.values):
                # Convert timestamp to ISO format string
                ts_str = pd.to_datetime(ts).isoformat()
                
                # Ensure value is numeric
                val_float = float(val)
                
                historical_data.append({
                    "timestamp": ts_str,
                    "value": val_float
                })
        except Exception as e:
            logger.error(f"Error processing historical data: {str(e)}")
            # Fallback with string conversion
            historical_data = [{"timestamp": str(ts), "value": float(val)} for ts, val in zip(time_series.index, time_series.values)]
        
        result = {
            "features": feature_dict,
            "historical_data": historical_data,
            "past_forecast": past_forecast_data,
            "future_forecast": future_forecast_data,
            "num_historical_points": len(historical_data),
            "num_past_forecast_points": len(past_forecast_data),
            "num_future_forecast_points": len(future_forecast_data)
        }
        
        # Include plot if requested
        if include_plot:
            plot_data = generate_plot(
                time_series=time_series, 
                forecast_data=future_forecast_data,
                past_forecast=past_forecast_data,
                plot_type=plot_type
            )
            
            if plot_type == "plotly":
                result["plot"] = {
                    "plot_type": "plotly",
                    "plot_data": plot_data
                }
            else:
                result["plot"] = {
                    "plot_type": "image",
                    "image_data": f"data:image/png;base64,{plot_data}"
                }
        
        return JSONResponse(result)
            
    except Exception as e:
        logger.error(f"Error during forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
                            