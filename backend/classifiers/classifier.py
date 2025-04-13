import os
import logging
from typing import Dict, List, Union
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_percentage_error)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from classifiers.FeatureExtractor import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesClassifier:
    def __init__(self, model_path=None):
        """Initialize with optional model path"""
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.feature_names = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.initialize_model()
    
    def initialize_model(self):
        """Create RandomForest model pipeline"""
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1
            ))
        ])
    
    def prepare_data(self, time_series_list, labels):
        """Extract features from time series for training"""
        features_list = []
        valid_labels = []
        
        for i, time_series in enumerate(time_series_list):
            try:
                if not isinstance(time_series, pd.Series):
                    logger.warning(f"Time series {i} is not a pandas Series. Skipping.")
                    continue
                
                if not isinstance(time_series.index, pd.DatetimeIndex):
                    logger.warning(f"Time series {i} does not have datetime index. Skipping.")
                    continue
                
                preprocessed_series = self.feature_extractor.preprocess_data(time_series)
                features = self.feature_extractor.extract_features(preprocessed_series)
                features_list.append(features)
                valid_labels.append(labels[i])
            except Exception as e:
                logger.error(f"Error processing time series {i}: {str(e)}")
                continue
        
        if not features_list:
            raise ValueError("No valid time series data after preprocessing")
        
        self.feature_names = self.feature_extractor.feature_names
        X = pd.DataFrame(features_list, columns=self.feature_names)
        y = np.array(valid_labels)
        
        if X.isna().any().any():
            logger.warning("Found NaN values in features. Replacing with 0.")
            X = X.fillna(0)
        
        return X, y
    
    def optimize_model(self, X, y, cv=5):
        """Tune RandomForest hyperparameters"""
        logger.info("Optimizing Random Forest hyperparameters...")
        
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2', None]
        }
        
        rf_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])
        
        stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            rf_model, 
            param_grid=param_grid, 
            cv=stratified_cv, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train(self, time_series_list, labels, optimize=True, test_size=0.2, random_state=42):
        """Train classifier and evaluate performance"""
        X, y = self.prepare_data(time_series_list, labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        if optimize:
            logger.info("Optimizing model hyperparameters...")
            self.model = self.optimize_model(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logger.info(f"\nModel: RandomForest")
        logger.info(f"Test set accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(report)
        
        self.plot_confusion_matrix(conf_matrix, list(np.unique(y)))
        self.plot_feature_importance()
        
        return accuracy, report
    
    def plot_confusion_matrix(self, conf_matrix, class_names):
        """Plot confusion matrix using Plotly"""
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            showscale=True,
            text=conf_matrix,
            texttemplate="%{text}",
            textfont={"size": 14}
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='True',
            xaxis={'side': 'bottom'},
            width=800,
            height=700
        )
        
        fig.show()
        logger.info("Displayed confusion matrix plot")
    
    def predict(self, time_series, use_time_series_models=True):
        """Predict class for new time series"""
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        try:
            if use_time_series_models:
                prediction, _, _ = self.predict_with_time_series_models(time_series)
                return prediction
            else:
                preprocessed_series = self.feature_extractor.preprocess_data(time_series)
                features = self.feature_extractor.extract_features(preprocessed_series)
                
                if self.feature_names:
                    X = pd.DataFrame([features], columns=self.feature_names)
                else:
                    X = pd.DataFrame([features])
                
                X = X.fillna(0)
                prediction = self.model.predict(X)[0]
                
                return prediction
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_proba(self, time_series, use_time_series_models=True):
        """Predict class probabilities for a new time series"""
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        try:
            if use_time_series_models:
                _, class_probabilities, _ = self.predict_with_time_series_models(time_series)
                return class_probabilities
            else:
                preprocessed_series = self.feature_extractor.preprocess_data(time_series)
                features = self.feature_extractor.extract_features(preprocessed_series)
                
                if self.feature_names:
                    X = pd.DataFrame([features], columns=self.feature_names)
                else:
                    X = pd.DataFrame([features])
                
                X = X.fillna(0)
                probabilities = self.model.predict_proba(X)[0]
                
                classes = self.model.classes_
                
                class_probabilities = {
                    str(class_name): float(prob) for class_name, prob in 
                    zip(classes, probabilities)
                }
                
                return class_probabilities
        except Exception as e:
            logger.error(f"Error during probability prediction: {str(e)}")
            raise
    
    def save_model(self, path):
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        
        try:
            joblib.dump(model_data, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path):
        """Load model from disk"""
        try:
            model_data = joblib.load(path)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_feature_importance(self):
        """Extract feature importance from model"""
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        classifier = self.model.named_steps['classifier']
        
        importance = np.zeros(len(self.feature_names)) if self.feature_names else []
        
        try:
            importance = classifier.feature_importances_
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            })
        else:
            importance_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, top_n=15):
        """Plot top feature importances using Plotly"""
        try:
            importance_df = self.get_feature_importance()
            df_plot = importance_df.head(top_n).sort_values('importance')
            
            fig = px.bar(
                df_plot, 
                y='feature', 
                x='importance',
                orientation='h',
                title='Top Feature Importance (RandomForest)',
                labels={'feature': 'Feature', 'importance': 'Importance'}
            )
            
            fig.update_layout(
                width=900,
                height=600,
                xaxis_title='Importance',
                yaxis_title='Feature'
            )
            
            fig.show()
            logger.info("Displayed feature importance plot")
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            
    def evaluate_with_cross_validation(self, time_series_list, labels, cv=5):
        """Evaluate model with cross-validation"""
        X, y = self.prepare_data(time_series_list, labels)
        
        stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(
            self.model, X, y, cv=stratified_cv, scoring='accuracy'
        )
        
        results = {
            'mean_accuracy': float(cv_scores.mean()),
            'std_accuracy': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist()
        }
        
        logger.info(f"Cross-validation results: Mean accuracy = {results['mean_accuracy']:.4f}, Std = {results['std_accuracy']:.4f}")
        
        return results

    def select_best_time_series_model(self, time_series):
        """Select best forecasting model using MAPE evaluation"""
        logger.info("Selecting best time series model using MAPE evaluation")
        
        series_length = len(time_series)
        has_seasonality = self.check_seasonality(time_series)
        has_trend = self.check_trend(time_series)
        is_stationary = self.check_stationarity(time_series)
        
        models_to_test = []
        models_to_test.append("arima")
        
        if series_length >= 100:
            if has_seasonality:
                models_to_test.append("sarima")
            if not is_stationary or has_trend:
                models_to_test.append("ets")
            if series_length >= 200:
                models_to_test.append("prophet")
        else:
            if not is_stationary:
                models_to_test.append("ets")
        
        models_to_test = list(set(models_to_test))
        
        if len(models_to_test) == 1:
            return models_to_test[0]
        
        train_size = int(len(time_series) * 0.8)
        if train_size < 10:
            train_size = max(3, len(time_series) - 2)
        
        train = time_series.iloc[:train_size]
        test = time_series.iloc[train_size:]
        
        if len(test) == 0:
            if series_length < 100:
                return "arima" if is_stationary else "ets"
            else:
                return "sarima" if has_seasonality and has_trend else "arima"
        
        results = {}
        for model_name in models_to_test:
            try:
                horizon = len(test)
                if model_name == "arima":
                    forecast = self.forecast_arima(train, horizon)
                elif model_name == "sarima":
                    forecast = self.forecast_sarima(train, horizon)
                elif model_name == "ets":
                    forecast = self.forecast_ets(train, horizon)
                elif model_name == "prophet":
                    forecast = self.forecast_prophet(train, horizon)
                else:
                    continue
                
                mape = self.calculate_mape(test, forecast)
                results[model_name] = mape
                logger.info(f"Model {model_name} MAPE: {mape:.4f}")
            except Exception as e:
                logger.warning(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        if not results:
            logger.warning("No models could be evaluated. Falling back to rule-based selection.")
            if series_length < 100:
                return "arima" if is_stationary else "ets"
            else:
                return "sarima" if has_seasonality and has_trend else "prophet" if has_trend else "arima"
        
        best_model = min(results, key=results.get)
        logger.info(f"Selected best model: {best_model} with MAPE: {results[best_model]:.4f}")
        
        return best_model

    def calculate_mape(self, actual, forecast):
        """Calculate Mean Absolute Percentage Error using sklearn"""
        from sklearn.metrics import mean_absolute_percentage_error
        
        if isinstance(actual.index, pd.DatetimeIndex) and isinstance(forecast.index, pd.DatetimeIndex):
            forecast = forecast.reindex(actual.index, method='nearest')
        else:
            forecast = forecast.iloc[:len(actual)]
        
        try:
            mape = mean_absolute_percentage_error(actual, forecast) * 100
            return mape
        except Exception as e:
            logger.error(f"Error calculating MAPE: {str(e)}")
            return float('inf')

    def check_seasonality(self, time_series):
        """Check for seasonal patterns in time series"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            if len(time_series) >= 10:
                result = seasonal_decompose(time_series, model='additive', extrapolate_trend='freq')
                return np.std(result.seasonal) > 0.1 * np.std(time_series)
            return False
        except Exception:
            return False

    def check_trend(self, time_series):
        """Check for trend in time series"""
        try:
            x = np.arange(len(time_series))
            y = time_series.values
            z = np.polyfit(x, y, 1)
            slope = z[0]
            
            data_range = np.max(y) - np.min(y)
            return abs(slope * len(time_series)) > 0.1 * data_range
        except Exception:
            return False

    def check_stationarity(self, time_series):
        """Check if time series is stationary"""
        try:
            from statsmodels.tsa.stattools import adfuller
            if len(time_series) >= 20:
                result = adfuller(time_series.values)
                return result[1] < 0.05
            return False
        except Exception:
            return False

    def forecast_with_best_model(self, time_series, horizon=30, best_model=None):
        """Generate forecasts using selected model"""
        if best_model is None:
            best_model = self.select_best_time_series_model(time_series)
        
        logger.info(f"Forecasting with {best_model} model")
        
        try:
            if best_model == "arima":
                return self.forecast_arima(time_series, horizon)
            elif best_model == "sarima":
                return self.forecast_sarima(time_series, horizon)
            elif best_model == "ets":
                return self.forecast_ets(time_series, horizon)
            elif best_model == "prophet":
                return self.forecast_prophet(time_series, horizon)
            else:
                return self.forecast_arima(time_series, horizon)
        except Exception as e:
            logger.error(f"Error in forecasting: {str(e)}")
            return self.naive_forecast(time_series, horizon)

    def forecast_arima(self, time_series, horizon):
        """Generate ARIMA forecast"""
        try:
            model = ARIMA(time_series, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=horizon)
            return forecast
        except Exception as e:
            logger.error(f"ARIMA forecasting error: {str(e)}")
            return self.naive_forecast(time_series, horizon)

    def forecast_sarima(self, time_series, horizon):
        """Generate SARIMA forecast"""
        try:
            freq = pd.infer_freq(time_series.index)
            seasonal_period = 12 if freq in ['M', 'MS'] else 4 if freq in ['Q', 'QS'] else 7 if freq in ['D'] else 24 if freq in ['H'] else 12
            
            model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=horizon)
            return forecast
        except Exception as e:
            logger.error(f"SARIMA forecasting error: {str(e)}")
            return self.forecast_arima(time_series, horizon)

    def forecast_ets(self, time_series, horizon):
        """Generate ETS forecast"""
        try:
            model = ExponentialSmoothing(time_series, trend='add', seasonal=None, damped=True)
            model_fit = model.fit()
            forecast = model_fit.forecast(horizon)
            return forecast
        except Exception as e:
            logger.error(f"ETS forecasting error: {str(e)}")
            return self.naive_forecast(time_series, horizon)

    def forecast_prophet(self, time_series, horizon):
        """Generate Prophet forecast"""
        try:
            df = pd.DataFrame({'ds': time_series.index, 'y': time_series.values})
            
            model = Prophet()
            model.fit(df)
            
            if isinstance(time_series.index, pd.DatetimeIndex):
                freq = pd.infer_freq(time_series.index)
                future = model.make_future_dataframe(periods=horizon, freq=freq)
            else:
                future = model.make_future_dataframe(periods=horizon)
            
            forecast = model.predict(future)
            
            if isinstance(time_series.index, pd.DatetimeIndex):
                result = pd.Series(forecast['yhat'].values[-horizon:], 
                                 index=pd.date_range(start=time_series.index[-1], 
                                                  periods=horizon+1, 
                                                  freq=freq)[1:])
            else:
                result = pd.Series(forecast['yhat'].values[-horizon:], 
                                 index=range(len(time_series), len(time_series) + horizon))
            
            return result
        except Exception as e:
            logger.error(f"Prophet forecasting error: {str(e)}")
            return self.forecast_arima(time_series, horizon)

    def naive_forecast(self, time_series, horizon):
        """Generate naive forecast (last value repeated)"""
        logger.warning("Using naive forecasting as fallback")
        last_value = time_series.iloc[-1]
        
        if isinstance(time_series.index, pd.DatetimeIndex):
            try:
                freq = pd.infer_freq(time_series.index)
                forecast_index = pd.date_range(start=time_series.index[-1], 
                                            periods=horizon+1, 
                                            freq=freq)[1:]
            except Exception:
                forecast_index = range(len(time_series), len(time_series) + horizon)
        else:
            forecast_index = range(len(time_series), len(time_series) + horizon)
        
        forecast = pd.Series([last_value] * horizon, index=forecast_index)
        return forecast

    def predict_with_time_series_models(self, time_series):
        """Predict class using the best time series model"""
        logger.info("Predicting with time series models")
        
        best_model = self.select_best_time_series_model(time_series)
        
        preprocessed_series = self.feature_extractor.preprocess_data(time_series)
        features = self.feature_extractor.extract_features(preprocessed_series)
        
        if self.feature_names:
            X = pd.DataFrame([features], columns=self.feature_names)
        else:
            X = pd.DataFrame([features])
        
        X = X.fillna(0)
        
        prediction = self.model.predict(X)[0]
        
        probabilities = self.model.predict_proba(X)[0]
        
        classes = self.model.classes_
        
        class_probabilities = {
            str(class_name): float(prob) for class_name, prob in 
            zip(classes, probabilities)
        }
        
        forecast = self.forecast_with_best_model(time_series, best_model=best_model)
        
        return prediction, class_probabilities, forecast

    def forecast(self, time_series, horizon=30):
        """Generate forecasts using best model"""
        best_model = self.select_best_time_series_model(time_series)
        forecast = self.forecast_with_best_model(time_series, horizon=horizon, best_model=best_model)
        return forecast
    
    def plot_forecast(self, time_series, forecast, title='Time Series Forecast'):
        """Plot time series forecast using Plotly"""
        try:
            # Create DataFrames for historical and forecast data
            historical = pd.DataFrame({
                'time': time_series.index,
                'value': time_series.values,
                'type': 'Historical'
            })
            
            forecast_df = pd.DataFrame({
                'time': forecast.index,
                'value': forecast.values,
                'type': 'Forecast'
            })
            
            # Concatenate historical and forecast data
            df = pd.concat([historical, forecast_df])
            
            # Create the plot
            fig = go.Figure()
            
            # Add historical data trace
            fig.add_trace(go.Scatter(
                x=historical['time'], 
                y=historical['value'],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Add forecast data trace
            fig.add_trace(go.Scatter(
                x=forecast_df['time'],
                y=forecast_df['value'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title='Time',
                yaxis_title='Value',
                width=1000,
                height=600,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Show the plot
            fig.show()
            logger.info("Displayed forecast plot")
            
            return fig
        except Exception as e:
            logger.error(f"Error plotting forecast: {str(e)}")
            return None

    