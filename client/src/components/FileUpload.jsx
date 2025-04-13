import { useState } from 'react'
import axios from 'axios'

function FileUpload({ onAnalysisComplete }) {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  
  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0])
      setError(null)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!file) {
      setError('Please select a CSV file')
      return
    }

    setLoading(true)
    setError(null)
    
    const formData = new FormData()
    formData.append('file', file)
    
    try {
      console.log("Sending request to backend...")
      
      // Common parameters for both requests
      const commonParams = '?horizon=20&simplify=true&detect_anomalies=true'
      
      // First get the analysis data without plot
      const analysisResult = await axios.post(`https://datagenie-533a.onrender.com/analyze${commonParams}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 60000
      })
      
      console.log("Received analysis from server", {
        status: analysisResult.status,
        hasData: !!analysisResult.data
      })
      
      // Now get a separate plot with consistent parameters
      const plotResult = await axios.post(`http://localhost:8000/plot${commonParams}&plot_type=plotly`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 30000
      })
      
      console.log("Received plot from server", {
        status: plotResult.status,
        hasPlotData: !!plotResult.data,
        plotType: plotResult.data?.plot_type
      })
      
      // Create a properly structured object for ResultsDisplay
      const combinedData = {
        ...analysisResult.data
      }
      
      // Add the plot data exactly as it comes from the server
      if (plotResult.data && plotResult.data.plot_type === 'plotly' && plotResult.data.plot_data) {
        // Create proper plot format expected by the component
        combinedData.plot = {
          plot_type: 'plotly',
          plot_data: plotResult.data.plot_data
        };
      }
      
      console.log("Combined data structure:", {
        hasForecast: !!combinedData.forecast,
        hasPlot: !!combinedData.plot,
        plotType: combinedData.plot?.plot_type
      })
      
      // Pass the combined data to the parent component
      onAnalysisComplete(combinedData)
    } catch (err) {
      console.error("Error details:", err)
      // Show a more user-friendly error message
      if (err.code === 'ECONNABORTED') {
        setError('Request timed out. Your file might be too large or have too many data points. Try a smaller sample.')
      } else if (err.response?.status === 500) {
        setError('Server error: The file may have an invalid format or contain problematic data')
      } else {
        setError(err.response?.data?.detail || 'Error uploading file: ' + err.message)
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-md mx-auto bg-gray-600 rounded-lg shadow-lg p-6 transition-all duration-1000 ease-in-out">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-200">
            Upload CSV File
          </label>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-300
              file:mr-4 file:py-2 file:px-4
              file:rounded-md file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-500 file:text-white
              hover:file:bg-blue-600
              transition-all duration-200"
          />
          {file && (
            <p className="mt-1 text-sm text-green-400">Selected: {file.name}</p>
          )}
          <div className="text-xs text-gray-400 mt-2">
            <p>Upload a CSV file with the following requirements:</p>
            <ul className="list-disc pl-5 mt-1 space-y-1">
              <li>Must include a timestamp/date column and a numeric value column</li>
            </ul>
          </div>
        </div>
        
        <button
          type="submit"
          disabled={loading}
          className={`w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white 
            ${loading ? 'bg-gray-500' : 'bg-blue-500 hover:bg-blue-600'} 
            transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
        >
          {loading ? 'Processing...' : 'Analyze Data'}
        </button>
      </form>
      
      {error && (
        <div className="mt-4 p-3 bg-red-900/50 text-red-200 rounded-md text-sm">
          {error}
        </div>
      )}
    </div>
  )
}

export default FileUpload 
