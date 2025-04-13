import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

API_URL = "http://localhost:8000"

def generate_sample_data(num_points=100, freq='H'):
    """Generate sample time series data for testing"""
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=num_points) if freq == 'H' else end_date - timedelta(days=num_points)
    
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    trend = np.linspace(500, 600, num=len(dates))
    seasonality = 50 * np.sin(np.linspace(0, 10 * np.pi, num=len(dates)))
    noise = np.random.normal(0, 20, size=len(dates))
    
    values = trend + seasonality + noise
    
    df = pd.DataFrame({
        'timestamp': dates,
        'point_value': values
    })
    
    return df

def save_sample_data(df, filename='sample_data.csv'):
    """Save sample data to a CSV file"""
    df.to_csv(filename, index=False)
    print(f"Sample data saved to {filename}")
    return filename

def generate_anomaly_sample_data(num_points=20):
    """Generate sample time series data with anomalies for testing"""
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=num_points)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    base_values = np.random.uniform(120, 140, size=len(dates))
    
    anomaly_indices = [4, 9, 15]
    for idx in anomaly_indices:
        if idx < len(base_values):
            base_values[idx] = base_values[idx] * 2.5
    
    data = []
    for i, (date, value) in enumerate(zip(dates, base_values)):
        data.append({
            "timestamp": date.strftime("%Y-%m-%dT%H:%M:%S"),
            "point_value": float(round(value, 1))
        })
    
    return data

def test_analyze_endpoint(file_path=None, data=None, horizon=30, detect_anomalies=False):
    """Test the analyze endpoint"""
    print(f"\nTesting analyze endpoint with detect_anomalies={detect_anomalies}...")
    
    url = f"{API_URL}/analyze"
    
    params = {
        'horizon': horizon,
        'detect_anomalies': detect_anomalies
    }
    
    if file_path:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'text/csv')}
            response = requests.post(url, files=files, params=params)
    elif data:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=data, headers=headers, params=params)
    else:
        print("Error: Either file_path or data must be provided")
        return None
    
    if response.status_code == 200:
        result = response.json()
        
        if detect_anomalies:
            print(f"Anomaly detection successful:")
            print(f"Forecastability score: {result.get('forecastability_score')}")
            print(f"Number of batch fits: {result.get('number_of_batch_fits')}")
            print(f"MAPE: {result.get('mape')}")
            
            anomalies = [r for r in result['results'] if r['is_anomaly'] == 'yes']
            print(f"Detected {len(anomalies)} anomalies out of {len(result['results'])} points")
            
            if anomalies:
                print("Sample anomalies:")
                for a in anomalies[:2]:
                    print(f"  Timestamp: {a['timestamp']}, Value: {a['point_value']}, Predicted: {a['predicted']}")
        else:
            print(f"Forecast successful:")
            print(f"Horizon: {result.get('horizon')} steps")
            print(f"Number of data points: {result.get('num_data_points')}")
            print(f"First 3 forecast points: {result['forecast'][:3]}")
        
        filename = "analysis_result"
        if detect_anomalies:
            filename += "_with_anomalies"
        filename += ".json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Full results saved to {filename}")
        
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def test_forecast(file_path=None, data=None, past_horizon=30, future_horizon=30):
    """Test the forecast endpoint"""
    print(f"\nTesting forecast endpoint with past_horizon={past_horizon}, future_horizon={future_horizon}...")
    
    url = f"{API_URL}/forecast"
    
    params = {
        'past_horizon': past_horizon,
        'future_horizon': future_horizon
    }
    
    if file_path:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'text/csv')}
            response = requests.post(url, files=files, params=params)
    elif data:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=data, headers=headers, params=params)
    else:
        print("Error: Either file_path or data must be provided")
        return None
    
    if response.status_code == 200:
        result = response.json()
        print(f"Forecast successful:")
        print(f"Historical data points: {result['num_historical_points']}")
        print(f"Past forecast points: {result['num_past_forecast_points']}")
        print(f"Future forecast points: {result['num_future_forecast_points']}")
        
        print("\nSample of extracted features:")
        feature_items = list(result['features'].items())
        for name, value in feature_items:
            print(f"  {name}: {value}")
        
        filename = "forecast_result.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Full results saved to {filename}")
        
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def main():
    print("Generating sample time series data...")
    sample_data = generate_sample_data(num_points=200, freq='H')
    file_path = save_sample_data(sample_data)
    
    anomaly_data = generate_anomaly_sample_data(20)
    
    print("\n===== Testing with CSV file =====")
    
    test_analyze_endpoint(file_path=file_path, horizon=60, detect_anomalies=False)
    test_analyze_endpoint(file_path=file_path, detect_anomalies=True)
    test_forecast(file_path=file_path, past_horizon=30, future_horizon=60)
    
    print("\n===== Testing with JSON data =====")
    
    test_analyze_endpoint(data=anomaly_data, detect_anomalies=True)
    test_forecast(data=anomaly_data, past_horizon=10, future_horizon=20)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 