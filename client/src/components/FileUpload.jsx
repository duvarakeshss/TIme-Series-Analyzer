import { useState, useRef } from 'react'
import axios from 'axios'

function FileUpload({ onAnalysisComplete }) {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef(null)
  
  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0])
      setError(null)
    }
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0]
      if (droppedFile.type === "text/csv" || droppedFile.name.endsWith('.csv')) {
        setFile(droppedFile)
        setError(null)
      } else {
        setError("Please upload a CSV file")
      }
    }
  }

  const handleButtonClick = () => {
    inputRef.current.click()
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!file) {
      setError('Please select a CSV file')
      return
    }

    setLoading(true)
    setError(null)
    setUploadProgress(0)
    
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
        timeout: 60000,
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          setUploadProgress(percentCompleted)
        }
      })
      
      console.log("Received analysis from server", {
        status: analysisResult.status,
        hasData: !!analysisResult.data
      })
      
      // Now get a separate plot with consistent parameters
      setUploadProgress(0) // Reset progress for next request
      const plotResult = await axios.post(`https://datagenie-533a.onrender.com/plot${commonParams}&plot_type=plotly`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 30000,
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          setUploadProgress(percentCompleted)
        }
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
        combinedData.plot = {
          plot_type: 'plotly',
          plot_data: plotResult.data.plot_data
        };
      }
      
      console.log("Combined data structure:", {
        hasPlot: !!combinedData.plot,
        plotType: combinedData.plot?.plot_type
      })
      
      onAnalysisComplete(combinedData)
    } catch (err) {
      console.error("Error details:", err)
      if (err.code === 'ECONNABORTED') {
        setError('Request timed out. Your file might be too large or have too many data points. Try a smaller sample.')
      } else if (err.response?.status === 500) {
        setError('Server error: The file may have an invalid format or contain problematic data')
      } else {
        setError(err.response?.data?.detail || 'Error uploading file: ' + err.message)
      }
    } finally {
      setLoading(false)
      setUploadProgress(0)
    }
  }

  // Format file size in KB or MB
  const formatFileSize = (bytes) => {
    if (!bytes) return '0 Bytes';
    if (bytes < 1024) return bytes + ' Bytes';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
  }

  return (
    <div className="max-w-xl mx-auto p-6 bg-gray-800 rounded-lg shadow-xl">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* File Upload Section */}
        <div className="space-y-4">
          <label className="block text-lg font-semibold text-white mb-2">
            Upload CSV File
          </label>
          <div 
            className={`flex flex-col items-center justify-center w-full h-40 border-2 border-dashed rounded-lg cursor-pointer transition-all duration-200 ${
              dragActive 
                ? "border-blue-500 bg-blue-900/20" 
                : file 
                  ? "border-green-500 bg-green-900/10" 
                  : "border-gray-600 bg-gray-700 hover:bg-gray-600"
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={handleButtonClick}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
            />
            
            {!file ? (
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <svg className="w-10 h-10 mb-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="mb-2 text-sm text-gray-400">
                  <span className="font-semibold">Click to upload</span> or drag and drop
                </p>
                <p className="text-xs text-gray-400">CSV files only</p>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center px-4 py-6">
                <div className="flex items-center mb-2">
                  <svg className="w-8 h-8 mr-2 text-green-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-lg font-medium text-green-400">{file.name}</span>
                </div>
                <p className="text-sm text-gray-400">
                  {formatFileSize(file.size)} • Click to replace
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="p-4 bg-red-900/50 border border-red-500 rounded-lg">
            <div className="flex">
              <svg className="w-5 h-5 text-red-400 mr-2 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading || !file}
          className={`w-full flex items-center justify-center py-3 px-4 rounded-lg text-sm font-semibold transition-colors duration-200 ${
            !file
              ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
              : loading 
                ? 'bg-blue-800 text-white cursor-wait' 
                : 'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 text-white'
          }`}
        >
          {loading ? (
            <>
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing...
            </>
          ) : (
            'Analyze Data'
          )}
        </button>

        {/* Progress Bar (only shown during loading) */}
        {loading && uploadProgress > 0 && (
          <div className="w-full bg-gray-700 rounded-full h-2.5 mt-2">
            <div 
              className="bg-blue-600 h-2.5 rounded-full transition-all duration-300 ease-in-out" 
              style={{ width: `${uploadProgress}%` }}
            ></div>
            <p className="text-xs text-gray-400 text-center mt-1">Uploading... {uploadProgress}%</p>
          </div>
        )}
      </form>
    </div>
  )
}

export default FileUpload 
