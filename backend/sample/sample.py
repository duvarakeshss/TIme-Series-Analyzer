import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os

# Time series modeling packages
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from statsmodels.tsa.stattools import acf, pacf, adfuller
from scipy import stats
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

class TimeSeriesFeatureExtractor:
    """Extract features from time series data for model selection"""
    
    @staticmethod
    def extract_features(time_series):
        """
        Extract statistical and time series specific features from a time series
        
        Args:
            time_series (pandas.Series): The time series data
            
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        # Handle potential NaN values
        ts = time_series.dropna()
        
        if len(ts) < 4:  # Minimum length for meaningful features
            raise ValueError("Time series too short for feature extraction")
        
        # Basic statistical features
        features['mean'] = ts.mean()
        features['std'] = ts.std()
        features['min'] = ts.min()
        features['max'] = ts.max()
        features['range'] = ts.max() - ts.min()
        features['median'] = ts.median()
        features['skewness'] = stats.skew(ts)
        features['kurtosis'] = stats.kurtosis(ts)
        features['iqr'] = np.percentile(ts, 75) - np.percentile(ts, 25)
        
        # Trend features
        x = np.arange(len(ts))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts)
        features['trend_slope'] = slope
        features['trend_r_squared'] = r_value**2
        
        # Stationarity features
        try:
            adf_result = adfuller(ts)
            features['adf_statistic'] = adf_result[0]
            features['adf_pvalue'] = adf_result[1]
        except:
            features['adf_statistic'] = 0
            features['adf_pvalue'] = 1
        
        # Autocorrelation features
        try:
            acf_values = acf(ts, nlags=min(10, len(ts)//2), fft=True)
            features['acf_lag_1'] = acf_values[1] if len(acf_values) > 1 else 0
            features['acf_lag_2'] = acf_values[2] if len(acf_values) > 2 else 0
            features['acf_lag_3'] = acf_values[3] if len(acf_values) > 3 else 0
            
            pacf_values = pacf(ts, nlags=min(10, len(ts)//2))
            features['pacf_lag_1'] = pacf_values[1] if len(pacf_values) > 1 else 0
            features['pacf_lag_2'] = pacf_values[2] if len(pacf_values) > 2 else 0
            features['pacf_lag_3'] = pacf_values[3] if len(pacf_values) > 3 else 0
        except:
            for lag in range(1, 4):
                features[f'acf_lag_{lag}'] = 0
                features[f'pacf_lag_{lag}'] = 0
        
        # Seasonality detection
        try:
            # Using autocorrelation to detect seasonality
            acf_values = acf(ts, nlags=min(len(ts)//2, 50), fft=True)
            # Find the first peak in ACF (excluding lag 0)
            peaks = [i for i in range(1, len(acf_values)) if i > 1 and 
                    acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]]
            
            if peaks:
                features['seasonal_period'] = peaks[0]
                features['seasonal_strength'] = acf_values[peaks[0]]
            else:
                features['seasonal_period'] = 0
                features['seasonal_strength'] = 0
        except:
            features['seasonal_period'] = 0
            features['seasonal_strength'] = 0
        
        # Volatility features
        features['volatility'] = ts.diff().std()
        
        # Complexity features
        features['entropy'] = stats.entropy(pd.cut(ts, bins=10).value_counts())
        
        # Outlier features
        z_scores = abs(stats.zscore(ts))
        features['outlier_count'] = np.sum(z_scores > 3)
        
        return features

class TimeSeriesModels:
    """Class to fit and evaluate various time series forecasting models"""
    
    @staticmethod
    def fit_arima(train_data, test_data, order=(1, 1, 1)):
        """Fit ARIMA model and evaluate on test data"""
        try:
            model = ARIMA(train_data, order=order)
            model_fit = model.fit()
            
            # Generate forecasts
            forecast = model_fit.forecast(steps=len(test_data))
            
            # Calculate MAPE
            mape = mean_absolute_percentage_error(test_data, forecast)
            return mape, forecast
        except:
            return float('inf'), None
    
    @staticmethod
    def fit_sarima(train_data, test_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """Fit SARIMA model and evaluate on test data"""
        try:
            model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            
            # Generate forecasts
            forecast = model_fit.forecast(steps=len(test_data))
            
            # Calculate MAPE
            mape = mean_absolute_percentage_error(test_data, forecast)
            return mape, forecast
        except:
            return float('inf'), None
    
    @staticmethod
    def fit_exponential_smoothing(train_data, test_data, trend=None, seasonal=None, seasonal_periods=None):
        """Fit Exponential Smoothing model and evaluate on test data"""
        try:
            model = ExponentialSmoothing(
                train_data, 
                trend=trend, 
                seasonal=seasonal, 
                seasonal_periods=seasonal_periods
            )
            model_fit = model.fit()
            
            # Generate forecasts
            forecast = model_fit.forecast(steps=len(test_data))
            
            # Calculate MAPE
            mape = mean_absolute_percentage_error(test_data, forecast)
            return mape, forecast
        except:
            return float('inf'), None
    
    @staticmethod
    def fit_prophet(train_data, test_data):
        """Fit Prophet model and evaluate on test data"""
        try:
            # Create a dataframe for Prophet
            df = pd.DataFrame({
                'ds': train_data.index,
                'y': train_data.values
            })
            
            # Initialize and fit the model
            model = Prophet()
            model.fit(df)
            
            # Create a dataframe for future predictions
            future = pd.DataFrame({'ds': test_data.index})
            
            # Make predictions
            forecast = model.predict(future)
            
            # Calculate MAPE
            mape = mean_absolute_percentage_error(test_data.values, forecast['yhat'].values)
            return mape, forecast['yhat']
        except:
            return float('inf'), None

class ModelSelector:
    """Class to select the best time series model for a given dataset"""
    
    def __init__(self, classifier_path=None):
        """
        Initialize the model selector
        
        Args:
            classifier_path (str, optional): Path to a pre-trained classifier model
        """
        self.feature_extractor = TimeSeriesFeatureExtractor()
        
        if classifier_path and os.path.exists(classifier_path):
            self.classifier = joblib.load(classifier_path)
        else:
            self.classifier = None
    
    def train_classifier(self, time_series_collection, test_size=0.2, random_state=42):
        """
        Train a classifier to predict the best time series model
        
        Args:
            time_series_collection (list): List of time series data
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: Trained classifier and evaluation metrics
        """
        # Extract features and determine best models for each time series
        features_list = []
        best_models = []
        
        for ts in time_series_collection:
            # Split into train/test
            train_size = int(len(ts) * 0.8)
            train_data = ts[:train_size]
            test_data = ts[train_size:]
            
            # Extract features from training data
            features = self.feature_extractor.extract_features(train_data)
            features_list.append(features)
            
            # Evaluate different models
            model_performances = {}
            
            # ARIMA
            model_performances['ARIMA'] = TimeSeriesModels.fit_arima(train_data, test_data)[0]
            
            # Auto-detect seasonality for seasonal models
            seasonal_period = features.get('seasonal_period', 0)
            if seasonal_period > 1:
                # SARIMA
                model_performances['SARIMA'] = TimeSeriesModels.fit_sarima(
                    train_data, test_data, seasonal_order=(1, 1, 1, seasonal_period)
                )[0]
                
                # Exponential Smoothing
                model_performances['ETS'] = TimeSeriesModels.fit_exponential_smoothing(
                    train_data, test_data, 
                    trend='add', seasonal='add', 
                    seasonal_periods=seasonal_period
                )[0]
            else:
                # Non-seasonal models
                model_performances['SARIMA'] = float('inf')
                model_performances['ETS'] = TimeSeriesModels.fit_exponential_smoothing(
                    train_data, test_data, trend='add'
                )[0]
            
            # Prophet
            model_performances['Prophet'] = TimeSeriesModels.fit_prophet(train_data, test_data)[0]
            
            # Determine best model
            best_model = min(model_performances, key=model_performances.get)
            best_models.append(best_model)
        
        # Prepare data for classifier
        X = pd.DataFrame(features_list)
        y = best_models
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Build classification pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=random_state))
        ])
        
        # Parameter grid for GridSearchCV
        param_grid = {
            'classifier__n_estimators': [50, 100, 150],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline, param_grid=param_grid, cv=5, scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)
        
        # Best classifier
        self.classifier = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Print results
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Test set accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.classifier, accuracy
    
    def save_classifier(self, path="model_selector_classifier.joblib"):
        """Save the trained classifier to disk"""
        if self.classifier:
            joblib.dump(self.classifier, path)
            print(f"Classifier saved to {path}")
        else:
            print("No classifier to save. Train a classifier first.")
    
    def predict_best_model(self, time_series):
        """
        Predict the best time series model for a given time series
        
        Args:
            time_series (pandas.Series): The time series data
            
        Returns:
            str: Name of the best time series model
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained or loaded. Train or load a classifier first.")
        
        # Extract features
        features = self.feature_extractor.extract_features(time_series)
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Predict best model
        best_model = self.classifier.predict(X)[0]
        
        return best_model
    
    def generate_predictions(self, time_series, date_from=None, date_to=None, period=0):
        """
        Generate predictions using the best model for a given time series
        
        Args:
            time_series (pandas.Series): The time series data
            date_from (str): Start date for predictions
            date_to (str): End date for predictions
            period (int): Number of periods to forecast
            
        Returns:
            dict: Dictionary containing predictions, model name, and MAPE
        """
        # Determine the best model
        best_model = self.predict_best_model(time_series)
        
        # Split the time series into train and test
        if date_from and date_to:
            try:
                # Convert to datetime if not already
                if not isinstance(time_series.index, pd.DatetimeIndex):
                    raise ValueError("Time series index is not a DatetimeIndex")
                
                train_data = time_series[time_series.index < date_from]
                test_data = time_series[(time_series.index >= date_from) & (time_series.index <= date_to)]
                
                # Check if we have enough training data
                if len(train_data) < 2:
                    raise ValueError("Not enough training data")
                
            except Exception as e:
                # Fallback to percentage split if date filtering fails
                print(f"Date filtering failed: {e}. Using percentage split instead.")
                train_size = int(len(time_series) * 0.8)
                train_data = time_series[:train_size]
                test_data = time_series[train_size:]
        else:
            # Default: Use 80% for training, 20% for testing
            train_size = int(len(time_series) * 0.8)
            train_data = time_series[:train_size]
            test_data = time_series[train_size:]
        
        # Determine forecast horizon
        if period > 0:
            forecast_steps = period
        else:
            forecast_steps = len(test_data)
        
        # Generate predictions using the best model
        if best_model == 'ARIMA':
            mape, predictions = TimeSeriesModels.fit_arima(train_data, test_data)
        elif best_model == 'SARIMA':
            # Extract seasonal period
            features = self.feature_extractor.extract_features(train_data)
            seasonal_period = features.get('seasonal_period', 12)  # Default to 12 if not found
            seasonal_period = max(seasonal_period, 2)  # Ensure seasonal_period is at least 2
            
            mape, predictions = TimeSeriesModels.fit_sarima(
                train_data, test_data, 
                seasonal_order=(1, 1, 1, seasonal_period)
            )
        elif best_model == 'ETS':
            # Extract seasonal period
            features = self.feature_extractor.extract_features(train_data)
            seasonal_period = features.get('seasonal_period', 0)
            
            if seasonal_period > 1:
                mape, predictions = TimeSeriesModels.fit_exponential_smoothing(
                    train_data, test_data,
                    trend='add', seasonal='add', seasonal_periods=seasonal_period
                )
            else:
                mape, predictions = TimeSeriesModels.fit_exponential_smoothing(
                    train_data, test_data, trend='add'
                )
        elif best_model == 'Prophet':
            mape, predictions = TimeSeriesModels.fit_prophet(train_data, test_data)
        else:
            # Fallback to ARIMA if something went wrong
            mape, predictions = TimeSeriesModels.fit_arima(train_data, test_data)
            best_model = 'ARIMA'
        
        # Handle case where model fitting failed
        if predictions is None:
            # Try fallback to ARIMA
            mape, predictions = TimeSeriesModels.fit_arima(train_data, test_data, order=(1,0,0))
            best_model = 'ARIMA (fallback)'
            
            # If still failing, use naive forecast
            if predictions is None:
                predictions = np.full(len(test_data), train_data.iloc[-1])
                mape = mean_absolute_percentage_error(test_data, predictions)
                best_model = 'Naive forecast (fallback)'
        
        return {
            'best_model': best_model,
            'mape': mape,
            'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
            'test_data': test_data.tolist() if hasattr(test_data, 'tolist') else test_data
        }


# Example usage script
if __name__ == "__main__":
    # Example of training the classifier with synthetic data
    # In practice, you would use real time series datasets
    
    # Generate synthetic time series
    np.random.seed(42)
    time_series_collection = []
    
    # Generate 20 time series with different characteristics
    for i in range(20):
        # Time series with trend
        trend_series = pd.Series(np.linspace(0, 10, 100) + np.random.normal(0, 1, 100))
        
        # Time series with seasonality
        seasonal_series = pd.Series(
            np.sin(np.linspace(0, 10*np.pi, 100)) + np.random.normal(0, 0.2, 100)
        )
        
        # Time series with trend and seasonality
        trend_seasonal_series = pd.Series(
            np.linspace(0, 5, 100) + np.sin(np.linspace(0, 10*np.pi, 100)) + 
            np.random.normal(0, 0.3, 100)
        )
        
        # Random walk time series
        random_walk = pd.Series(np.cumsum(np.random.normal(0, 1, 100)))
        
        # AR process
        ar_series = pd.Series(np.zeros(100))
        for t in range(1, 100):
            ar_series[t] = 0.8 * ar_series[t-1] + np.random.normal(0, 1)
        
        time_series_collection.extend([
            trend_series, seasonal_series, trend_seasonal_series, random_walk, ar_series
        ])
    
    # Create model selector and train classifier
    model_selector = ModelSelector()
    classifier, accuracy = model_selector.train_classifier(time_series_collection)
    
    # Save the trained classifier
    model_selector.save_classifier("model_selector_classifier.joblib")
    
    # Example of using the trained classifier
    new_time_series = pd.Series(
        np.linspace(0, 5, 100) + np.sin(np.linspace(0, 10*np.pi, 100)) + 
        np.random.normal(0, 0.3, 100)
    )
    
    # Predict best model
    best_model = model_selector.predict_best_model(new_time_series)
    print(f"Best model for new time series: {best_model}")
    
    # Generate predictions
    result = model_selector.generate_predictions(new_time_series)
    print(f"Predictions generated using {result['best_model']}")
    print(f"MAPE: {result['mape']:.4f}")