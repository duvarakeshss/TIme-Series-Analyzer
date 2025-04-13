import os
import sys
import argparse
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifiers.classifier import TimeSeriesClassifier
from classifiers.FeatureExtractor import FeatureExtractor

def load_time_series_data(data_dir):
    
    # Load time series data from the specified directory

    time_series_data = {}
    
    frequency_labels = {
        'hourly': 'hourly',
        'daily': 'daily',
        'weekly': 'weekly',
        'monthly': 'monthly',
        'externalDataset': 'external'  # Added external dataset
    }
    
    for freq_dir, label in frequency_labels.items():
        freq_path = os.path.join(data_dir, freq_dir)
        if not os.path.exists(freq_path):
            logger.warning(f"Directory {freq_path} not found. Skipping...")
            continue
            
        # Handle subdirectories in externalDataset
        if freq_dir == 'externalDataset':
            # Get all subdirectories
            subdirs = [d for d in os.listdir(freq_path) if os.path.isdir(os.path.join(freq_path, d))]
            
            if subdirs:
                # Process each subdirectory
                for subdir in subdirs:
                    subdir_path = os.path.join(freq_path, subdir)
                    subdir_label = f"external_{subdir}"
                    
                    csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
                    
                    if not csv_files:
                        logger.warning(f"No CSV files found in {subdir_path}. Skipping...")
                        continue
                    
                    for csv_file in tqdm(csv_files, desc=f"Loading {subdir_label} data"):
                        file_path = os.path.join(subdir_path, csv_file)
                        process_csv_file(file_path, csv_file, subdir_label, time_series_data)
            
            # Also check for CSV files directly in externalDataset folder
            csv_files = [f for f in os.listdir(freq_path) if f.endswith('.csv')]
            
            if csv_files:
                for csv_file in tqdm(csv_files, desc=f"Loading {label} data"):
                    file_path = os.path.join(freq_path, csv_file)
                    process_csv_file(file_path, csv_file, label, time_series_data)
        else:
            # Standard processing for regular frequency directories
            csv_files = [f for f in os.listdir(freq_path) if f.endswith('.csv')]
            
            if not csv_files:
                logger.warning(f"No CSV files found in {freq_path}. Skipping...")
                continue
                
            for csv_file in tqdm(csv_files, desc=f"Loading {label} data"):
                file_path = os.path.join(freq_path, csv_file)
                process_csv_file(file_path, csv_file, label, time_series_data)
    
    return time_series_data

def process_csv_file(file_path, csv_file, label, time_series_data):
    
    # Process a CSV file and add it to the time series data dictionary
    
    try:
        df = pd.read_csv(file_path, sep=None, engine='python')  # Try to automatically detect delimiter
        logger.info(f"\nProcessing {csv_file}:")
        logger.debug(f"Raw data shape: {df.shape}")
        logger.debug(f"Columns: {df.columns.tolist()}")
        
        # Try to identify time and value columns if not standard names
        time_col = None
        value_col = None
        
        # Check for standard column names
        if 'point_timestamp' in df.columns and 'point_value' in df.columns:
            time_col = 'point_timestamp'
            value_col = 'point_value'
        else:
            # Try to identify time column (looking for date-like strings)
            for col in df.columns:
                if col.lower().find('time') >= 0 or col.lower().find('date') >= 0:
                    time_col = col
                    break
            
            # If still not found, take the first column as a fallback
            if time_col is None and len(df.columns) > 0:
                time_col = df.columns[0]
            
            # Try to identify value column
            for col in df.columns:
                if col.lower().find('value') >= 0 or col.lower().find('price') >= 0 or col.lower().find('amount') >= 0:
                    value_col = col
                    break
            
            # If still not found, take the second column as a fallback
            if value_col is None and len(df.columns) > 1:
                value_col = df.columns[1]
        
        if time_col is None or value_col is None:
            logger.warning(f"{csv_file} missing required columns. Skipping...")
            return
        
        logger.debug(f"Using columns: time={time_col}, value={value_col}")
        
        try:
            # Check if the time column contains combined time and value
            if df[time_col].dtype == object and df[time_col].str.contains('\\s{2,}').any():
                # Split the combined column into time and value
                logger.info(f"Detected combined time and value in column {time_col}")
                
                # Split the column by multiple spaces
                split_data = df[time_col].str.split(r'\s{2,}', expand=True)
                
                # Create new time and value columns
                df['parsed_time'] = split_data[0]
                df['parsed_value'] = split_data[1].astype(float)
                
                # Update column references
                time_col = 'parsed_time'
                value_col = 'parsed_value'
            
            # Convert time column to datetime
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # Remove rows with NaT
            if df[time_col].isna().any():
                logger.warning(f"Removed {df[time_col].isna().sum()} rows with invalid dates")
                df = df.dropna(subset=[time_col])
            
            if df.empty:
                logger.warning(f"{csv_file} is empty after date parsing. Skipping...")
                return
            
            series = pd.Series(df[value_col].values, index=df[time_col])
            logger.debug(f"Timestamp range: {series.index.min()} to {series.index.max()}")
        except Exception as e:
            logger.warning(f"Could not parse datetime in {csv_file}: {str(e)}. Trying alternative approach...")
            
            try:
                # extract time and value columns manually if the file is space-delimited
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                times = []
                values = []
                
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            # parse time (could be in first two parts) and value
                            try:
                                time_str = ' '.join(parts[0:2]) if len(parts) > 2 else parts[0]
                                time = pd.to_datetime(time_str)
                                value = float(parts[-1])
                                times.append(time)
                                values.append(value)
                            except:
                                continue
                
                if times and values:
                    series = pd.Series(values, index=times)
                    logger.info(f"Successfully parsed {len(series)} points using alternative method")
                else:
                    logger.warning(f"Could not parse data from {csv_file} using alternative method. Skipping...")
                    return
            except Exception as e:
                logger.warning(f"Failed alternative parsing for {csv_file}: {str(e)}. Skipping...")
                return
        
        missing_count = series.isna().sum()
        missing_pct = missing_count/len(series)*100
        logger.debug(f"Missing values: {missing_count} ({missing_pct:.2f}%)")
        
        if series.empty:
            logger.warning(f"{csv_file} is empty. Skipping...")
            return
            
        if missing_pct > 50:
            logger.warning(f"{csv_file} has too many missing values. Skipping...")
            return
            
        if series.nunique() <= 1:
            logger.warning(f"{csv_file} has constant values. Skipping...")
            return
        
        time_series_data[f"{label}_{csv_file}"] = {
            'series': series,
            'label': label
        }
    except Exception as e:
        logger.error(f"Error loading {csv_file}: {str(e)}")

def analyze_features(time_series_data):
    
    # Analyze feature distributions across categories and return DataFrame of extracted features
    
    feature_extractor = FeatureExtractor()
    all_features = []
    
    warnings.filterwarnings("ignore", message=".*Invalid input, x is constant.*")
    
    for name, data in time_series_data.items():
        try:
            series = data['series']
            label = data['label']
            
            if series.nunique() <= 1 or len(series) < 10:
                continue
            
            preprocessed_series = feature_extractor.preprocess_data(series)
            features = feature_extractor.extract_features(preprocessed_series)
            feature_names = feature_extractor.feature_names
            
            feature_dict = {fname: features[i] for i, fname in enumerate(feature_names)}
            feature_dict['series_name'] = name
            feature_dict['label'] = label
            
            all_features.append(feature_dict)
            
        except Exception as e:
            logger.error(f"Error processing {name}: {str(e)}")
            continue
    
    if not all_features:
        return pd.DataFrame()
        
    features_df = pd.DataFrame(all_features)
    
    if not features_df.empty:
        logger.info("\nFeature Statistics by Category:")
        for label in features_df['label'].unique():
            category_df = features_df[features_df['label'] == label]
            numeric_cols = category_df.select_dtypes(include=[np.number]).columns
            stats = category_df[numeric_cols].describe().T
            
            for feature in numeric_cols:
                if feature in stats.index:
                    logger.info(f"{feature}: mean={stats.loc[feature, 'mean']:.4f}, std={stats.loc[feature, 'std']:.4f}")
    
    return features_df

def train_and_evaluate_classifier(time_series_data, optimize=True):
    
    """
    Train and evaluate a time series classifier
    
    Args:
        time_series_data (dict): Dictionary of time series data
        optimize (bool): Whether to optimize hyperparameters
        
    Returns:
        TimeSeriesClassifier: Trained classifier
    """
    
    time_series_list = []
    labels = []
    
    for name, data in time_series_data.items():
        series = data['series']
        if series.nunique() <= 1:
            continue
        if len(series) < 10:
            continue
            
        time_series_list.append(series)
        labels.append(data['label'])
    
    # Count samples per label
    label_counts = {}
    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    # Filter out classes with fewer than 2 samples
    valid_labels = [label for label, count in label_counts.items() if count >= 2]
    logger.info(f"Valid labels with at least 2 samples: {valid_labels}")
    
    if len(valid_labels) < 2:
        logger.error("Not enough classes with sufficient samples for classification (need at least 2)")
        return None
    
    # Filter time series to only include those with valid labels
    filtered_time_series = []
    filtered_labels = []
    for ts, label in zip(time_series_list, labels):
        if label in valid_labels:
            filtered_time_series.append(ts)
            filtered_labels.append(label)
    
    logger.info(f"Training with {len(filtered_time_series)} time series across {len(valid_labels)} classes")
    
    unique_labels = set(filtered_labels)
    if len(unique_labels) < 2:
        logger.error("Not enough different class labels for classification (need at least 2)")
        return None
    
    classifier = TimeSeriesClassifier()
    
    logger.info("\nTraining classifier...")
    try:
        accuracy, report = classifier.train(filtered_time_series, filtered_labels, optimize=optimize)
        
        logger.info("\nEvaluating with cross-validation...")
        cv_results = classifier.evaluate_with_cross_validation(filtered_time_series, filtered_labels)
        
        model_dir = 'models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        model_path = os.path.join(model_dir, "randomforest_classifier.joblib")
        classifier.save_model(model_path)
        
        return classifier
    except Exception as e:
        logger.error(f"Error training classifier: {str(e)}")
        return None

def visualize_results(features_df, classifier, time_series, class_names):
    
    # visualizations of the classification results
    
    if classifier is None:
        logger.error("Cannot create visualizations: No trained classifier provided")
        return
    
    if features_df.empty:
        logger.error("Cannot create visualizations: No feature data available")
        return
    
    viz_dir = 'visualizations'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    try:
        importance_df = classifier.get_feature_importance()
        top_n = min(15, len(importance_df))
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importance_df.head(top_n))
        plt.title(f'Top {top_n} Important Features (RandomForest)')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'feature_importance.png'))
        plt.close()
        importance_df.to_csv(os.path.join(viz_dir, 'feature_importance.csv'), index=False)
        logger.info(f"Feature importance saved to {os.path.join(viz_dir, 'feature_importance.png')}")
    except Exception as e:
        logger.error(f"Error creating feature importance visualization: {str(e)}")
    
    try:
        top_features = importance_df['feature'].head(6).tolist() if not importance_df.empty else []
        if not top_features:
            logger.warning("No features available for distribution visualization")
            return
        
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(top_features):
            if feature in features_df.columns:
                plt.subplot(2, 3, i+1)
                sns.boxplot(x='label', y=feature, data=features_df)
                plt.title(feature)
                plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'feature_distributions.png'))
        plt.close()
        logger.info(f"Feature distributions saved to {os.path.join(viz_dir, 'feature_distributions.png')}")
    except Exception as e:
        logger.error(f"Error creating feature distributions visualization: {str(e)}")
    
    try:
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric columns for correlation matrix")
            return
        
        corr_matrix = features_df[numeric_cols].corr()
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'correlation_matrix.png'))
        plt.close()
        logger.info(f"Correlation matrix saved to {os.path.join(viz_dir, 'correlation_matrix.png')}")
    except Exception as e:
        logger.error(f"Error creating correlation matrix: {str(e)}")

    try:
        # Prepare data for splitting - Remove 'series_name' and 'label' from feature columns
        X = features_df.drop(columns=['series_name', 'label']) if 'series_name' in features_df.columns else features_df.drop(columns=['label'])
        y = features_df['label']
        
        if len(set(y)) >= 2:  # Ensure we have at least 2 classes for train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            y_pred = classifier.model.predict(X_test)
            y_true = y_test
    
            conf_matrix = confusion_matrix(y_true, y_pred)
            classifier.plot_confusion_matrix(conf_matrix, class_names)
            logger.info(f"Confusion matrix displayed")
            
            # Save the confusion matrix as an image
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'))
            plt.close()
            logger.info(f"Confusion matrix saved to {os.path.join(viz_dir, 'confusion_matrix.png')}")
        else:
            logger.warning("Not enough classes for confusion matrix visualization")
    except Exception as e:
        logger.error(f"Error creating confusion matrix: {str(e)}")
        
    # Forecast is now handled separately in the main function to avoid duplication

def main():
    try:
        parser = argparse.ArgumentParser(description='Train and evaluate time series classifiers')
        parser.add_argument('--model', type=str, default='RandomForest', 
                            choices=['RandomForest'],  # Add more models here when supported
                            help='Type of model to use')
        parser.add_argument('--optimize', action='store_true', help='Optimize model hyperparameters')
        parser.add_argument('--data-dir', type=str, default=None, help='Path to data directory')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings')
        parser.add_argument('--include-external', action='store_true', help='Include external datasets even if not enough samples')
        
        args = parser.parse_args()
        
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            
        if args.no_warnings:
            warnings.filterwarnings("ignore")
        
        if args.data_dir:
            data_dir = args.data_dir
        else:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
        
        logger.info(f"Using data directory: {data_dir}")
        logger.info(f"Using model: {args.model}")
        logger.info(f"Hyperparameter optimization: {'Enabled' if args.optimize else 'Disabled'}")
        
        logger.info("Loading time series data...")
        time_series_data = load_time_series_data(data_dir)
        
        if not time_series_data:
            logger.error("No data found. Please check the data directory.")
            return
        
        logger.info(f"Loaded {len(time_series_data)} time series")
        
        # Get all available labels
        all_labels = list(set(data['label'] for data in time_series_data.values()))
        logger.info(f"Found {len(all_labels)} unique labels: {all_labels}")
        
        logger.info("\nAnalyzing features...")
        features_df = analyze_features(time_series_data)
        
        if features_df.empty:
            logger.error("No valid features could be extracted. Exiting.")
            return
        
        logger.info("\nTraining and evaluating classifier...")
        classifier = train_and_evaluate_classifier(
            time_series_data,
            optimize=args.optimize
        )
        
        if classifier is None:
            logger.error("Classifier training failed. Exiting.")
            return
        
        # For visualization, use only the labels that were used in training
        trained_labels = set(classifier.model.classes_)
        logger.info(f"Trained on classes: {trained_labels}")
        
        logger.info("\nCreating visualizations...")
        
        # Find a representative time series for each trained label
        viz_time_series = {}
        for name, data in time_series_data.items():
            label = data['label']
            if label in trained_labels and label not in viz_time_series:
                viz_time_series[label] = data['series']
        
        # Choose the first time series for forecast visualization
        if viz_time_series:
            time_series = list(viz_time_series.values())[0]
            forecast = classifier.forecast(time_series, horizon=60)
            classifier.plot_forecast(time_series, forecast, title='Time Series Forecast (60 Days)')
        
        trained_features_df = features_df[features_df['label'].isin(trained_labels)]
        
        class_names = list(trained_labels)
        visualize_results(trained_features_df, classifier, time_series, class_names)
        
        logger.info("\ncompleted")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
