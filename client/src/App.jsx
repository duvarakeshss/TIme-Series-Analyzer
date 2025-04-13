import { useState, useEffect } from 'react'
import './App.css'
import FileUpload from './components/FileUpload'
import ResultsDisplay from './components/ResultsDisplay'
import ErrorBoundary from './components/ErrorBoundary'

function App() {
  const [animate, setAnimate] = useState(false)
  const [analysisResults, setAnalysisResults] = useState(null)
  
  useEffect(() => {
    // Trigger animation after component mounts
    setTimeout(() => setAnimate(true), 100)
  }, [])

  const handleAnalysisComplete = (results) => {
    setAnalysisResults(results)
  }

  return (
    <div className='bg-gray-800 min-h-screen w-full overflow-auto absolute inset-0'>
      <div className={`bg-gradient-to-r from-purple-700 to-purple-400 w-full py-6 flex justify-center items-center transition-all duration-700 ease-in-out ${animate ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-10'}`}>
        <h1 className={`text-3xl font-bold text-white transition-all duration-1000 ease-in-out ${animate ? 'opacity-100 scale-100' : 'opacity-0 scale-90'}`}>
          Time Series Analysis
        </h1>
      </div>
      <div className='container mx-auto px-4 py-8 pb-16 transition-all duration-700 ease-in-out'>
        <FileUpload onAnalysisComplete={handleAnalysisComplete} />
        <ErrorBoundary>
          <ResultsDisplay results={analysisResults} />
        </ErrorBoundary>
      </div>
    </div>
  )
}

export default App
