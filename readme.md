# Time series analysis

Time series analysis and forecasting application that helps analyze temporal data patterns, detect anomalies, and generate forecasts.

## Features

- **Time Series Analysis**
  - Automatic frequency detection
  - Anomaly detection
  - Forecastability scoring
  - Statistical feature extraction

- **Forecasting**
  - Future value predictions
  - Confidence intervals
  - Multiple forecasting horizons
  - Historical trend analysis

- **Visualization**
  - Interactive Plotly charts
  - Anomaly highlighting
  - Forecast plotting
  - Historical data trends

- **Data Processing**
  - CSV file support
  - Automatic timestamp parsing
  - Missing value handling
  - Data validation


## **Random Forest**: 
It classifies which statistical models should use for the given data.

## Statistical Models Used

The application leverages the following statistical models and techniques for time series analysis and forecasting:

- **ARIMA (AutoRegrezssive Integrated Moving Average)**: For univariate time series forecasting with trends and seasonality.

- **Prophet**: For handling seasonality and holiday effects in time series forecasting.

- **Exponential Smoothing (ETS)**: For smoothing and forecasting time series data.

- **Anomaly Detection Models**: Techniques like Z-scores or Isolation Forest for identifying anomalies in the data.

- **Statistical Feature Extraction**: Methods like autocorrelation, partial autocorrelation, and Fourier transforms to extract meaningful features.

These models are selected based on the nature of the data and the specific analysis or forecasting requirements.

## Project Structure

```
datagenie/
├── backend/
│   ├── classifiers/         # ML models
│   │   ├── FeatureExtractor.py
│   │   ├── classidier.py
│   │   ├── build.py
│   │   ├── models/
│   │   │   └── randomforest_classifier.joblib
│   ├── main.py             # FastAPI backend server
│   └── requirements.txt    # Python dependencies
├── client/
│   ├── src/
│   │   ├── components/    # React components
│   │   │   ├── FileUpload.jsx
│   │   │   └── ResultDisplay.jsx           
│   │   └── App.jsx        # Main React application
│   ├── package.json       # Node dependencies
│   └── vite.config.js     # Vite configuration
└── README.md
```

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The backend server will start at `http://localhost:8000`
or `https://datagenie-533a.onrender.com`

### Frontend Setup

```bash
cd client
npm install
npm run dev
```

The frontend development server will start at `http://localhost:5173`
or `https://time-series-analysis.vercel.app/`

## Usage

1. Upload a CSV file containing time series data
2. The file should have:
   - A timestamp/date column
   - A numeric value column
   - At least 10 data points
   - Proper CSV formatting

3. The application will automatically:
   - Analyze the data patterns
   - Detect anomalies
   - Generate forecasts
   - Display interactive visualizations

## API Endpoints

- `POST /analyze` - Analyze time series data
- `POST /forecast` - Generate forecasts
- `POST /plot` - Create visualizations

## Technologies Used

- **Backend**
  - FastAPI
  - scikit-learn
  - pandas
  - numpy
  - plotly

- **Frontend**
  - React
  - Vite
  - Tailwind CSS
  - Plotly.js


## Acknowledgments

- Plotly.js for interactive visualizations
- scikit-learn for machine learning capabilities
- FastAPI for the high-performance backend