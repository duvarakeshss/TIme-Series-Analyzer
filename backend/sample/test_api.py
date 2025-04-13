#!/usr/bin/env python
"""
Test script for the Time Series Analysis API
This script tests all the endpoints of the API using the requests library
"""

import requests
import json
import time
import sys
import os

# API Base URL
BASE_URL = "http://localhost:8000"
TEST_FILE = "test_sample_data.csv"

def print_response(response, endpoint_name):
    """Print API response in a formatted way"""
    print(f"\n{'=' * 50}")
    print(f"TESTING: {endpoint_name}")
    print(f"STATUS CODE: {response.status_code}")
    print(f"RESPONSE TIME: {response.elapsed.total_seconds():.4f} seconds")
    
    try:
        data = response.json()
        print("RESPONSE:")
        print(json.dumps(data, indent=2))
    except json.JSONDecodeError:
        print("RESPONSE: (not JSON)")
        print(response.text[:200] + "..." if len(response.text) > 200 else response.text)
    
    print(f"{'=' * 50}\n")

def test_root_endpoint():
    """Test the root endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/")
        print_response(response, "Root Endpoint (GET /)")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing root endpoint: {str(e)}")
        return False

def test_models_endpoint():
    """Test the models endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/models")
        print_response(response, "Models Endpoint (GET /models)")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing models endpoint: {str(e)}")
        return False

def test_predict_endpoint(model_type=None, anomaly_threshold=0.3):
    """Test the predict endpoint"""
    try:
        # Make sure the test file exists
        if not os.path.exists(TEST_FILE):
            print(f"Error: Test file {TEST_FILE} not found")
            return False
        
        # Prepare the files and data
        files = {'file': open(TEST_FILE, 'rb')}
        data = {}
        
        if model_type:
            data['model_type'] = model_type
        
        data['anomaly_threshold'] = str(anomaly_threshold)
        
        # Make the request
        print(f"Testing predict endpoint with model_type={model_type}, anomaly_threshold={anomaly_threshold}")
        response = requests.post(f"{BASE_URL}/predict", files=files, data=data)
        
        endpoint_name = f"Predict Endpoint (POST /predict) with model_type={model_type}, anomaly_threshold={anomaly_threshold}"
        print_response(response, endpoint_name)
        
        # Close the file
        files['file'].close()
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing predict endpoint: {str(e)}")
        return False

def test_forecast_endpoint(model_type=None, horizon=10):
    """Test the forecast endpoint"""
    try:
        # Make sure the test file exists
        if not os.path.exists(TEST_FILE):
            print(f"Error: Test file {TEST_FILE} not found")
            return False
        
        # Prepare the files and data
        files = {'file': open(TEST_FILE, 'rb')}
        data = {}
        
        if model_type:
            data['model_type'] = model_type
        
        data['horizon'] = str(horizon)
        
        # Make the request
        print(f"Testing forecast endpoint with model_type={model_type}, horizon={horizon}")
        response = requests.post(f"{BASE_URL}/forecast", files=files, data=data)
        
        endpoint_name = f"Forecast Endpoint (POST /forecast) with model_type={model_type}, horizon={horizon}"
        print_response(response, endpoint_name)
        
        # Close the file
        files['file'].close()
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing forecast endpoint: {str(e)}")
        return False

def test_classify_endpoint():
    """Test the classify endpoint"""
    try:
        # Make sure the test file exists
        if not os.path.exists(TEST_FILE):
            print(f"Error: Test file {TEST_FILE} not found")
            return False
        
        # Prepare the files and data
        files = {'file': open(TEST_FILE, 'rb')}
        
        # Make the request
        print("Testing classify endpoint")
        response = requests.post(f"{BASE_URL}/classify", files=files)
        
        print_response(response, "Classify Endpoint (POST /classify)")
        
        # Close the file
        files['file'].close()
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing classify endpoint: {str(e)}")
        return False

def test_plot_image_endpoint():
    """
    Test the /plot_image endpoint that returns a PNG image of the time series plot
    """
    print("\n=== Testing /plot_image endpoint ===")
    
    # Check if test file exists
    test_file_path = 'test_sample_data.csv'
    if not os.path.exists(test_file_path):
        print(f"ERROR: Test file {test_file_path} not found. Please create a test file with time series data.")
        return

    try:
        # Try different model types
        model_types = [None, 'arima', 'sarima', 'ets', 'prophet']
        plot_types = ['all', 'original', 'forecast', 'anomaly']
        
        for model_type in model_types[:2]:  # Test only a subset to keep tests short
            for plot_type in plot_types[:2]:  # Test only a subset to keep tests short
                # Start timing
                start_time = time.time()
                
                # Prepare request
                model_param = f"&model_type={model_type}" if model_type else ""
                plot_param = f"&plot_type={plot_type}"
                
                url = f"{BASE_URL}/plot_image?forecast_horizon=10{model_param}{plot_param}"
                print(f"\nTesting plot_image with model={model_type}, plot_type={plot_type}")
                
                # Open file and send request
                with open(test_file_path, 'rb') as f:
                    files = {'file': (test_file_path, f, 'text/csv')}
                    response = requests.post(url, files=files)
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Print response information
                print(f"Status code: {response.status_code}")
                print(f"Response time: {response_time:.2f} seconds")
                
                if response.status_code == 200:
                    print(f"Successfully received plot image ({len(response.content)} bytes)")
                    
                    # Optionally save the image to a file
                    with open(f"plot_{model_type or 'auto'}_{plot_type}.png", 'wb') as f:
                        f.write(response.content)
                    print(f"Saved plot image to plot_{model_type or 'auto'}_{plot_type}.png")
                else:
                    print(f"Error: {response.text}")
        
        print("\n✅ plot_image endpoint testing completed")
        return True
        
    except Exception as e:
        print(f"ERROR: Exception while testing plot_image endpoint: {str(e)}")
        return False

def main():
    """
    Main function to run all tests
    """
    # Store test results
    test_results = {}
    
    # Test endpoints
    test_results["root"] = test_root_endpoint()
    test_results["models"] = test_models_endpoint()
    test_results["predict"] = test_predict_endpoint()
    test_results["forecast"] = test_forecast_endpoint()
    test_results["classify"] = test_classify_endpoint()
    test_results["plot_image"] = test_plot_image_endpoint()
    
    # Print summary
    print("\n=== Test Summary ===")
    for test, result in test_results.items():
        status = "✅ Passed" if result else "❌ Failed"
        print(f"{test}: {status}")

if __name__ == "__main__":
    main() 