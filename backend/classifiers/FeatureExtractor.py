import os
import warnings
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import periodogram
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

class FeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def _clean_data(self, data):
        data = np.array(data)
        data[np.isinf(data)] = np.nan
        data[np.abs(data) > 1e10] = np.nan
        if np.isnan(data).any():
            data[np.isnan(data)] = np.nanmean(data)
        return data
    
    def _validate_features(self, features):
        features = np.array(features)
        if np.isinf(features).any() or np.any(np.abs(features) > 1e10) or np.isnan(features).any():
            features = self._clean_data(features)
        return features
    
    def check_missing_values(self, time_series):
        if not isinstance(time_series, pd.Series):
            raise ValueError("Input must be a pandas Series")
            
        total_missing = time_series.isna().sum()
        missing_percentage = (total_missing / len(time_series)) * 100
        
        missing_mask = time_series.isna()
        consecutive_missing = 0
        current_consecutive = 0
        
        for is_missing in missing_mask:
            if is_missing:
                current_consecutive += 1
                consecutive_missing = max(consecutive_missing, current_consecutive)
            else:
                current_consecutive = 0
                
        missing_patterns = {}
        if total_missing > 0:
            missing_indices = np.where(missing_mask)[0]
            if len(missing_indices) > 1:
                gaps = np.diff(missing_indices)
                unique_gaps, gap_counts = np.unique(gaps, return_counts=True)
                for gap, count in zip(unique_gaps, gap_counts):
                    missing_patterns[f'gap_{gap}'] = int(count)
                    
        return {
            'total_missing': int(total_missing),
            'missing_percentage': float(missing_percentage),
            'consecutive_missing': int(consecutive_missing),
            'missing_patterns': missing_patterns
        }
    
    def handle_missing_values(self, time_series, method='interpolate'):
        missing_stats = self.check_missing_values(time_series)
        
        if missing_stats['total_missing'] == 0:
            return time_series
            
        if method == 'interpolate':
            time_series = time_series.interpolate(method='linear')
        elif method == 'ffill':
            time_series = time_series.ffill()
        elif method == 'bfill':
            time_series = time_series.bfill()
        elif method == 'mean':
            time_series = time_series.fillna(time_series.mean())
        else:
            raise ValueError(f"Unknown method: {method}")
            
        if time_series.isna().any():
            time_series = time_series.ffill().bfill()
            
        return time_series
    
    def extract_features(self, time_series):
        features = []
        self.feature_names = []
        
        # Basic statistical features
        features.extend([
            float(np.mean(time_series)),
            float(np.std(time_series)),
            float(np.var(time_series)),
            float(np.min(time_series)),
            float(np.max(time_series)),
            float(np.median(time_series)),
            float(stats.skew(time_series)),
            float(stats.kurtosis(time_series))
        ])
        self.feature_names.extend([
            'mean', 'std', 'var', 'min', 'max', 
            'median', 'skew', 'kurtosis'
        ])
        
        # Trend features
        x = np.arange(len(time_series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, time_series)
        features.extend([float(slope), float(r_value**2)])
        self.feature_names.extend(['trend_slope', 'trend_r_squared'])
        
        # Time series specific features
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_result = adfuller(time_series)
                features.extend([float(adf_result[0]), float(adf_result[1])])
                self.feature_names.extend(['adf_stat', 'adf_pvalue'])
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_result = kpss(time_series, regression='ct', nlags='auto')
                features.extend([float(kpss_result[0]), float(kpss_result[1])])
                self.feature_names.extend(['kpss_stat', 'kpss_pvalue'])
        except Exception as e:
            warnings.warn(f"Error in statistical tests: {str(e)}")
            features.extend([0.0, 1.0, 0.0, 1.0])
            self.feature_names.extend(['adf_stat', 'adf_pvalue', 'kpss_stat', 'kpss_pvalue'])
        
        # Volatility features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            returns = np.diff(time_series) / time_series[:-1]
            returns = returns[np.isfinite(returns)]
            if len(returns) > 0:
                features.extend([
                    float(np.std(returns)),
                    float(np.mean(np.abs(returns))),
                    float(np.percentile(returns, 95) - np.percentile(returns, 5))
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        self.feature_names.extend(['volatility', 'mean_abs_change', 'range_90'])
        
        # Seasonality features
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                acf_values = acf(time_series, nlags=min(len(time_series)//2, 50), fft=True)
                peaks = [i for i in range(1, len(acf_values)-1) if acf_values[i] > acf_values[i-1] and 
                        acf_values[i] > acf_values[i+1]]
                if peaks:
                    seasonal_period = peaks[0]
                    seasonal_strength = float(acf_values[peaks[0]])
                else:
                    seasonal_period = 0
                    seasonal_strength = 0.0
                    
                features.extend([float(seasonal_period), seasonal_strength])
                self.feature_names.extend(['seasonal_period', 'seasonal_strength'])
        except Exception as e:
            warnings.warn(f"Error in seasonality detection: {str(e)}")
            features.extend([0.0, 0.0])
            self.feature_names.extend(['seasonal_period', 'seasonal_strength'])
        
        # Entropy and complexity features
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hist, _ = np.histogram(time_series, bins='auto')
                hist = hist + 1e-10
                entropy = float(stats.entropy(hist))
                features.append(entropy)
                self.feature_names.append('entropy')
                
                zero_crossings = float(np.sum(np.diff(np.signbit(time_series - np.mean(time_series)))))
                features.append(zero_crossings)
                self.feature_names.append('zero_crossings')
        except Exception as e:
            warnings.warn(f"Error in entropy calculation: {str(e)}")
            features.extend([0.0, 0.0])
            self.feature_names.extend(['entropy', 'zero_crossings'])
        
        features = np.array(features)
        features[~np.isfinite(features)] = 0.0
        
        return features.tolist()
    
    def preprocess_data(self, time_series):
        if not isinstance(time_series, pd.Series):
            raise ValueError("Input must be a pandas Series")
            
        if not isinstance(time_series.index, pd.DatetimeIndex):
            raise ValueError("Series must have a datetime index")
            
        series = time_series.copy()
        series = series.sort_index()
        
        if series.isna().any():
            series = series.interpolate(method='time')
            series = series.ffill().bfill()
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z_scores = np.abs(stats.zscore(series))
            series[z_scores > 3] = np.nan
            series = series.interpolate(method='time').ffill().bfill()
        
        return series

    def preprocess_time_series(self, time_series):
        """Preprocess time series data, including handling missing values and regularizing timeseries"""
        # Handle NaN values
        if time_series.isnull().any():
            # Forward fill
            time_series = time_series.ffill()
            
            # Backward fill any remaining NaNs
            time_series = time_series.bfill()
        
        # If series is still empty or contains NaNs, raise an error
        if time_series.empty or time_series.isnull().any():
            raise ValueError("Time series still contains NaN values after preprocessing")
            
        return time_series
        
    def process_time_series(self, df):
        """Extract time series from dataframe"""
        # Handle dataframe with datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            time_series = df.iloc[:, 0] if len(df.columns) >= 1 else None
        
        # Handle dataframe with separate datetime column
        elif len(df.columns) >= 2:
            # Assume first column is datetime and second is value
            timestamp_col = df.columns[0]
            value_col = df.columns[1]
            
            # Convert to datetime
            if df[timestamp_col].dtype != 'datetime64[ns]':
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Create time series
            time_series = pd.Series(df[value_col].values, index=df[timestamp_col])
            time_series = time_series.sort_index().ffill().bfill()
        else:
            raise ValueError("Input DataFrame must have at least 2 columns (datetime and value)")
        
        if time_series is None or len(time_series) < 10:
            raise ValueError("Time series is too short (minimum 10 data points required)")
            
        return time_series
