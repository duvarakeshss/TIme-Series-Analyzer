import { useEffect, useRef, useState, useCallback } from 'react';

// Load Plotly directly in the script
const loadPlotlyScript = () => {
  return new Promise((resolve, reject) => {
    // Check if Plotly is already loaded
    if (window.Plotly) {
      console.log("Plotly already loaded, using existing instance");
      resolve(window.Plotly);
      return;
    }
    
    console.log("Loading Plotly from CDN");
    
    try {
      // Create script element
      const script = document.createElement('script');
      script.src = 'https://cdn.plot.ly/plotly-2.24.1.min.js';
      script.async = true;
      script.crossOrigin = "anonymous";
      
      // Set up load and error handlers
      script.onload = () => {
        console.log("Plotly script loaded successfully");
        if (window.Plotly) {
          resolve(window.Plotly);
        } else {
          reject(new Error("Plotly not found after script load"));
        }
      };
      
      script.onerror = (e) => {
        console.error("Failed to load Plotly script:", e);
        reject(new Error("Failed to load Plotly"));
      };
      
      // Add to document
      document.head.appendChild(script);
    } catch (err) {
      console.error("Error setting up Plotly script:", err);
      reject(err);
    }
  });
};

function ResultsDisplay({ results }) {
  const plotContainerRef = useRef(null);
  const [plotError, setPlotError] = useState(null);
  const [plotLoading, setPlotLoading] = useState(true);
  const [plotlyLoaded, setPlotlyLoaded] = useState(false);

  // Function to create a fallback plot with static data
  const createFallbackPlot = useCallback(() => {
    try {
      if (!plotContainerRef.current || !window.Plotly) {
        console.error("Cannot create fallback plot: missing container or Plotly");
        return false;
      }
      
      console.log("Creating fallback plot");
      
      // Clear container
      plotContainerRef.current.innerHTML = '';
      
      // Create simple fallback data
      const data = [
        {
          x: ['Point 1', 'Point 2', 'Point 3', 'Point 4', 'Point 5'],
          y: [5, 7, 2, 8, 4],
          type: 'scatter',
          mode: 'lines+markers',
          marker: { color: 'blue' }
        }
      ];
      
      const layout = {
        title: 'Fallback Plot - Error with actual data',
        annotations: [{
          x: 0.5,
          y: 0.5,
          xref: 'paper',
          yref: 'paper',
          text: 'Could not render actual data. Using placeholder.',
          showarrow: false,
          font: {
            size: 14,
            color: 'red'
          }
        }]
      };
      
      window.Plotly.newPlot(plotContainerRef.current, data, layout, {responsive: true});
      setPlotLoading(false);
      return true;
    } catch (err) {
      console.error("Failed to create fallback plot:", err);
      return false;
    }
  }, []);

  // Create a simple error message display as fallback when Plotly fails
  const displayErrorMessage = useCallback((message) => {
    if (!plotContainerRef.current) return false;
    
    try {
      // Clear the container
      plotContainerRef.current.innerHTML = '';
      
      // Create a simple error display
      const errorContainer = document.createElement('div');
      errorContainer.style.width = '100%';
      errorContainer.style.height = '100%';
      errorContainer.style.display = 'flex';
      errorContainer.style.flexDirection = 'column';
      errorContainer.style.alignItems = 'center';
      errorContainer.style.justifyContent = 'center';
      errorContainer.style.backgroundColor = '#f8f9fa';
      errorContainer.style.color = '#dc3545';
      errorContainer.style.padding = '20px';
      errorContainer.style.borderRadius = '8px';
      
      // Error icon
      const icon = document.createElement('div');
      icon.innerHTML = '⚠️';
      icon.style.fontSize = '32px';
      icon.style.marginBottom = '10px';
      
      // Error message
      const text = document.createElement('div');
      text.textContent = message || 'Error displaying plot';
      text.style.textAlign = 'center';
      text.style.maxWidth = '400px';
      
      // Add elements to container
      errorContainer.appendChild(icon);
      errorContainer.appendChild(text);
      plotContainerRef.current.appendChild(errorContainer);
      
      // Update state
      setPlotLoading(false);
      setPlotError(message);
      return true;
    } catch (err) {
      console.error("Failed to display error message:", err);
      return false;
    }
  }, []);

  // Function to render the plot data wrapped in useCallback
  const safeCleanupPlotly = useCallback((container) => {
    try {
      // Don't proceed if container is invalid
      if (!container || !document.contains(container)) {
        return;
      }
  
      // First, clean up any Plotly instances
      if (window.Plotly?.purge) {
        // Safely purge container
        try {
          window.Plotly.purge(container);
        } catch (e) {
          console.warn("Error purging container:", e);
        }
      }
  
      // Clear contents using a safer method
      while (container.firstChild) {
        container.firstChild.remove();
      }
    } catch (e) {
      console.warn("Error in safeCleanupPlotly:", e);
    }
  }, []);
  
  // Replace the renderPlotData function with this simplified version
const renderPlotData = useCallback(() => {
  if (!results?.plot || !plotContainerRef.current) {
    return;
  }

  try {
    setPlotLoading(true);
    setPlotError(null);

    // Get plot data
    const plotData = results.plot.plot_data;
    if (!plotData) {
      throw new Error("No plot data found in response");
    }

    // Parse the data if needed
    const plotObj = typeof plotData === 'string' ? JSON.parse(plotData) : plotData;
    
    if (!plotObj || !plotObj.data || !Array.isArray(plotObj.data)) {
      throw new Error("Invalid plot data format");
    }

    // Clean up existing plot
    if (window.Plotly?.purge) {
      window.Plotly.purge(plotContainerRef.current);
    }

    // Create new plot
    window.Plotly.newPlot(
      plotContainerRef.current,
      plotObj.data,
      plotObj.layout || { 
        title: "Time Series Analysis",
        autosize: true,
        height: 500
      },
      { 
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
      }
    ).then(() => {
      setPlotLoading(false);
      console.log("Plot rendered successfully");
    }).catch((err) => {
      console.error("Error in Plotly.newPlot:", err);
      setPlotError(`Failed to render chart: ${err.message}`);
      setPlotLoading(false);
    });

  } catch (err) {
    console.error("Error rendering plot:", err);
    setPlotError(`Failed to render chart: ${err.message}`);
    setPlotLoading(false);
  }
}, [results]);

// Add this effect to handle plot updates
useEffect(() => {
  if (plotlyLoaded && results?.plot) {
    renderPlotData();
  }
  
  // Cleanup function
  return () => {
    if (plotContainerRef.current && window.Plotly?.purge) {
      try {
        window.Plotly.purge(plotContainerRef.current);
      } catch (e) {
        console.warn("Error during plot cleanup:", e);
      }
    }
  };
}, [plotlyLoaded, results, renderPlotData]);

  // Update the handleRetryClick function to be simpler
  const handleRetryClick = useCallback(() => {
    setPlotLoading(true);
    setPlotError(null);
    
    try {
      if (!plotContainerRef.current || !results?.plot?.plot_data) {
        throw new Error("Missing plot container or data");
      }
  
      // Clean up existing plot if any
      if (window.Plotly?.purge) {
        window.Plotly.purge(plotContainerRef.current);
      }
  
      // Parse and render plot data
      const plotObj = typeof results.plot.plot_data === 'string' 
        ? JSON.parse(results.plot.plot_data) 
        : results.plot.plot_data;
  
      window.Plotly.newPlot(
        plotContainerRef.current,
        plotObj.data,
        plotObj.layout || { title: "Time Series Analysis" },
        { responsive: true }
      );
  
      setPlotLoading(false);
    } catch (err) {
      console.error("Error during plot retry:", err);
      setPlotError("Failed to retry: " + err.message);
      setPlotLoading(false);
    }
  }, [results]);

  // Load Plotly on component mount
  useEffect(() => {
    let isMounted = true;
    
    const loadPlotly = async () => {
      console.log("Starting Plotly library load");
      try {
        // Set loading state
        setPlotLoading(true);
        
        // Try to load Plotly
        await loadPlotlyScript();
        
        // Check if component is still mounted before updating state
        if (isMounted) {
          console.log("Successfully loaded Plotly");
          setPlotlyLoaded(true);
        }
      } catch (err) {
        console.error("Failed to load Plotly library:", err);
        
        // Only update state if still mounted
        if (isMounted) {
          setPlotError("Failed to load plotting library. Please try refreshing the page.");
          setPlotLoading(false);
        }
      }
    };
    
    loadPlotly();
    
    // Cleanup function to handle component unmounting
    return () => {
      console.log("Component unmounting, canceling Plotly load");
      isMounted = false;
    };
  }, []);

  // Debug the results structure
  useEffect(() => {
    if (results && results.plot) {
      console.log("Plot data structure:", {
        type: typeof results.plot,
        isObject: typeof results.plot === 'object',
        keys: typeof results.plot === 'object' ? Object.keys(results.plot) : null,
        plotType: results.plot.plot_type,
        hasPlotData: !!(results.plot.plot_data),
        dataLength: typeof results.plot.plot_data === 'string' ? results.plot.plot_data.length : 'not a string',
        forecastData: results.forecast ? results.forecast.length : 0
      });

      console.log("Forecast data:", results.forecast?.slice(0, 3));

      // Extra validation for the plot_data
      if (results.plot.plot_data && typeof results.plot.plot_data === 'string') {
        try {
          const plotObj = JSON.parse(results.plot.plot_data);
          console.log("Plot data parsed successfully:", {
            hasData: Array.isArray(plotObj.data),
            dataCount: plotObj.data?.length,
            hasLayout: !!plotObj.layout
          });
        } catch (err) {
          console.error("Failed to parse plot_data:", err);
        }
      }
    }
  }, [results]);

  // Add a safety timeout to prevent infinite loading
  useEffect(() => {
    if (plotLoading) {
      const safetyTimer = setTimeout(() => {
        if (plotLoading) {
          console.warn("Safety timeout triggered - plot still loading after 10 seconds");
          setPlotError("Chart load timeout. Try clicking 'Retry Loading Chart'");
          setPlotLoading(false);
        }
      }, 10000); // 10 second timeout
      
      return () => clearTimeout(safetyTimer);
    }
  }, [plotLoading]);

  // Add a thorough cleanup function to ensure all Plotly elements are removed
  useEffect(() => {
    // Function to safely clean up all Plotly elements
    function thoroughCleanup() {
      try {
        // Clean up all Plotly plots in the DOM, not just our container
        if (window.Plotly && window.Plotly.purge) {
          // Clean up plotly elements in our container first
          if (plotContainerRef.current) {
            try {
              window.Plotly.purge(plotContainerRef.current);
              
              // Also purge any Plotly elements inside our container
              const plotlyElements = plotContainerRef.current.querySelectorAll('[id^="plotly-"]');
              Array.from(plotlyElements).forEach(element => {
                try {
                  window.Plotly.purge(element);
                } catch (e) {
                  // Ignore errors on individual elements
                }
              });
              
              // Clear inner HTML as a last resort
              plotContainerRef.current.innerHTML = '';
            } catch (e) {
              console.warn("Error cleaning plotly container:", e);
            }
          }
          
          // Additionally purge any plotly elements in the document as a safety net
          const allPlotlyElements = document.querySelectorAll('[id^="plotly-"]');
          Array.from(allPlotlyElements).forEach(element => {
            try {
              window.Plotly.purge(element);
            } catch (e) {
              // Ignore errors on individual elements
            }
          });
        }
      } catch (e) {
        console.warn("Error in thoroughCleanup:", e);
      }
    }
    
    // Return cleanup function
    return thoroughCleanup;
  }, []);

  // Add a DOM mutation observer to protect against React/manual DOM conflicts
  useEffect(() => {
    if (!plotContainerRef.current) return;
    
    // Helper function to safely check if node is a child before removal
    const safeRemoveChild = (parent, child) => {
      try {
        // First check if both parent and child exist and are valid nodes
        if (!parent || !child || !(parent instanceof Node) || !(child instanceof Node)) {
          console.warn('Invalid parent or child node');
          return false;
        }
    
        // Check if the node is already detached
        if (!child.parentNode) {
          console.warn('Node is already detached');
          return true;
        }
    
        // Check if child is actually a child of parent before removing
        if (!parent.contains(child)) {
          console.warn('Child is not a descendant of parent');
          return false;
        }
    
        // Ensure the nodes are still in the DOM
        if (!document.contains(parent) || !document.contains(child)) {
          console.warn('Nodes are no longer in the DOM');
          return false;
        }
    
        // Remove the child using the parent node reference
        child.parentNode.removeChild(child);
        return true;
      } catch (e) {
        console.warn("Error in safeRemoveChild:", e);
        return false;
      }
    };
    
    // Create a mutation observer to watch for DOM changes
    const observer = new MutationObserver((mutations) => {
      // Skip if component is unmounting
      if (!document.contains(plotContainerRef.current)) return;
      
      mutations.forEach(mutation => {
        if (mutation.type === 'childList') {
          mutation.removedNodes.forEach(node => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              try {
                // Only handle Plotly elements
                if (node.id?.startsWith('plotly-')) {
                  if (window.Plotly?.purge && document.contains(node)) {
                    window.Plotly.purge(node);
                  }
                }
              } catch (e) {
                console.warn(`Error handling node removal: ${e.message}`);
              }
            }
          });
        }
      });
    });
    
    // Start observing the plot container with all needed options
    observer.observe(plotContainerRef.current, {
      childList: true,
      subtree: true
    });
    
    // Cleanup observer when component unmounts
    return () => {
      try {
        // Make sure we disable the observer before doing any cleanup
        observer.disconnect();
        console.log("Successfully disconnected MutationObserver");
      } catch (e) {
        console.warn("Error disconnecting MutationObserver:", e);
      }
    };
  }, []);
  
  // Add special handling for when component is about to unmount
  useEffect(() => {
    return () => {
      try {
        if (observer) {
          observer.disconnect();
        }
        if (plotContainerRef.current && window.Plotly?.purge) {
          window.Plotly.purge(plotContainerRef.current);
        }
      } catch (e) {
        console.warn('Error during cleanup:', e);
      }
    };
  }, [safeCleanupPlotly]);

  // Early return if no results - AFTER all hooks are defined
  if (!results) return null;
  
  // Calculate derived values AFTER all hooks
  const anomalyCount = results.results ? 
    results.results.filter(item => item.is_anomaly === "yes").length : 0;
  const anomalies = results.results ? 
    results.results.filter(item => item.is_anomaly === "yes") : [];

  return (
    <div className="mt-6 max-w-5xl mx-auto">
      <h2 className="text-xl font-bold text-white mb-4">Analysis Results</h2>
      
      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-700 rounded-lg p-4 shadow-md hover:shadow-lg transition-all duration-300 transform hover:scale-105 hover:bg-gray-600 cursor-pointer">
          <div className="text-gray-300 text-sm mb-1 font-medium">Forecastability Score</div>
          <div className="text-2xl font-bold text-blue-400">{results.forecastability_score}/10</div>
          <div className="text-xs text-gray-400 mt-1">Higher is better</div>
        </div>
        <div className="bg-gray-700 rounded-lg p-4 shadow-md hover:shadow-lg transition-all duration-300 transform hover:scale-105 hover:bg-gray-600 cursor-pointer">
          <div className="text-gray-300 text-sm mb-1 font-medium">Mean Abs % Error</div>
          <div className="text-2xl font-bold text-blue-400">{results.mape}</div>
          <div className="text-xs text-gray-400 mt-1">Lower is better</div>
        </div>
        <div className="bg-gray-700 rounded-lg p-4 shadow-md hover:shadow-lg transition-all duration-300 transform hover:scale-105 hover:bg-gray-600 cursor-pointer">
          <div className="text-gray-300 text-sm mb-1 font-medium">Data Points</div>
          <div className="text-2xl font-bold text-blue-400">{results.results?.length || 0}</div>
          <div className="text-xs text-gray-400 mt-1">Number of data points</div>
        </div>
        <div className="bg-gray-700 rounded-lg p-4 shadow-md hover:shadow-lg transition-all duration-300 transform hover:scale-105 hover:bg-gray-600 cursor-pointer">
          <div className="text-gray-300 text-sm mb-1 font-medium">Anomalies Detected</div>
          <div className="text-2xl font-bold text-red-400">{anomalyCount}</div>
          <div className="text-xs text-gray-400 mt-1">Unusual data points</div>
        </div>
      </div>
      
      {/* Plot Container */}
      <div className="mb-6 bg-gray-800 rounded-lg shadow-lg p-6 border-2 border-gray-700 hover:border-blue-500 transition-colors duration-300">
        <h3 className="text-lg font-medium text-white mb-4 border-b border-gray-700 pb-2">Data Visualization</h3>
        <div className="w-full h-[500px] relative rounded-lg overflow-hidden" ref={plotContainerRef}>
          {plotLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900/75 backdrop-blur-sm text-gray-300 rounded-lg z-10">
              <div className="text-center">
                <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-solid border-blue-500 border-r-transparent align-[-0.125em] motion-reduce:animate-[spin_1.5s_linear_infinite] mb-3"></div>
                <p className="text-lg">Loading chart...</p>
              </div>
            </div>
          )}
          
          {plotError && !plotLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900/75 backdrop-blur-sm text-gray-300 rounded-lg z-10">
              <div className="text-center max-w-md p-6 bg-gray-800 rounded-lg shadow-lg border border-red-500/50">
                <svg className="mx-auto h-16 w-16 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <h3 className="mt-4 text-lg font-medium text-white">Chart Error</h3>
                <p className="mt-2 text-sm text-gray-400">
                  {plotError}
                </p>
                <div className="mt-4">
                  <button 
                    onClick={handleRetryClick}
                    className="py-2 px-6 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-colors duration-300 focus:ring-2 focus:ring-blue-400 focus:outline-none"
                  >
                    Retry Loading Chart
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Tabs for different data views */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Future Forecast Section */}
        {results.forecast && results.forecast.length > 0 && (
          <div className="bg-gray-800 rounded-lg p-6 shadow-lg hover:shadow-xl transition-all duration-300 border-2 border-gray-700 hover:border-blue-500">
            <h3 className="text-lg font-medium text-white mb-4 border-b border-gray-700 pb-2 flex items-center">
              <svg className="w-5 h-5 mr-2 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
              Future Forecast <span className="ml-2 text-sm font-normal text-gray-400">({results.forecast.length} periods)</span>
            </h3>
            <div className="overflow-x-auto rounded-lg">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="bg-gray-700">
                    <th className="text-left py-3 px-4 text-gray-300 font-medium rounded-tl-lg">Timestamp</th>
                    <th className="text-right py-3 px-4 text-gray-300 font-medium rounded-tr-lg">Forecasted Value</th>
                  </tr>
                </thead>
                <tbody>
                  {results.forecast.slice(0, 10).map((item, index) => (
                    <tr key={index} className={`border-t border-gray-700 hover:bg-gray-700/50 transition-colors duration-200`}>
                      <td className="py-3 px-4 text-gray-300">{item.timestamp}</td>
                      <td className="text-right py-3 px-4 text-blue-400 font-medium">
                        {typeof item.value === 'number' ? item.value.toFixed(2) : item.value}
                      </td>
                    </tr>
                  ))}
                  {results.forecast.length > 10 && (
                    <tr className="border-t border-gray-700 bg-gray-800/80">
                      <td colSpan={2} className="py-3 px-4 text-gray-400 text-center font-medium">
                        {results.forecast.length - 10} more periods forecasted...
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      
        {/* Anomalies Section */}
        <div className="bg-gray-800 rounded-lg p-6 shadow-lg hover:shadow-xl transition-all duration-300 border-2 border-gray-700 hover:border-red-500">
          <h3 className="text-lg font-medium text-white mb-4 border-b border-gray-700 pb-2 flex items-center">
            <svg className="w-5 h-5 mr-2 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            Detected Anomalies <span className="ml-2 text-sm font-normal text-gray-400">{anomalyCount > 0 ? `(${anomalyCount})` : '(None)'}</span>
          </h3>
          
          {anomalyCount > 0 ? (
            <div className="overflow-x-auto rounded-lg">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="bg-gray-700">
                    <th className="text-left py-3 px-4 text-gray-300 font-medium rounded-tl-lg">Timestamp</th>
                    <th className="text-right py-3 px-4 text-gray-300 font-medium">Actual Value</th>
                    <th className="text-right py-3 px-4 text-gray-300 font-medium rounded-tr-lg">Expected Value</th>
                  </tr>
                </thead>
                <tbody>
                  {anomalies.slice(0, 10).map((item, index) => (
                    <tr key={index} className={`border-t border-gray-700 hover:bg-gray-700/50 transition-colors duration-200`}>
                      <td className="py-3 px-4 text-gray-300">{item.timestamp}</td>
                      <td className="text-right py-3 px-4 text-red-400 font-medium">
                        {typeof item.point_value === 'number' ? item.point_value.toFixed(2) : item.point_value}
                      </td>
                      <td className="text-right py-3 px-4 text-blue-400 font-medium">
                        {typeof item.predicted === 'number' ? item.predicted.toFixed(2) : item.predicted}
                      </td>
                    </tr>
                  ))}
                  {anomalies.length > 10 && (
                    <tr className="border-t border-gray-700 bg-gray-800/80">
                      <td colSpan={3} className="py-3 px-4 text-gray-400 text-center font-medium">
                        {anomalies.length - 10} more anomalies detected...
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-10 text-gray-400 bg-gray-700/30 rounded-lg">
              <svg className="w-12 h-12 mb-3 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-lg">No anomalies detected in the data</p>
              <p className="text-sm mt-1">Your time series appears to be stable</p>
            </div>
          )}
        </div>
      </div>
      
      {/* Performance Metrics */}
      <div className="mt-6 bg-gray-800 rounded-lg p-6 shadow-lg hover:shadow-xl transition-all duration-300 border-2 border-gray-700 hover:border-green-500">
        <h3 className="text-lg font-medium text-white mb-4 border-b border-gray-700 pb-2 flex items-center">
          <svg className="w-5 h-5 mr-2 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Technical Performance
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-700/50 p-4 rounded-lg hover:bg-gray-700 transition-all duration-300">
            <div className="text-gray-400 mb-1 font-medium">Number of batch fits</div>
            <div className="text-2xl font-semibold text-blue-400">{results.number_of_batch_fits}</div>
            <div className="text-xs text-gray-500 mt-1">Model training iterations</div>
          </div>
          <div className="bg-gray-700/50 p-4 rounded-lg hover:bg-gray-700 transition-all duration-300">
            <div className="text-gray-400 mb-1 font-medium">Average time per fit</div>
            <div className="text-2xl font-semibold text-blue-400">{results.avg_time_taken_per_fit_in_seconds} seconds</div>
            <div className="text-xs text-gray-500 mt-1">Processing efficiency</div>
          </div>
        </div>

        {/* Additional Metrics Section */}
        {results.metrics && (
          <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.entries(results.metrics).map(([key, value], index) => (
              <div key={index} className="bg-gray-700/50 p-3 rounded-lg hover:bg-gray-700 transition-colors duration-300">
                <div className="text-sm text-gray-400 font-medium truncate">{key.replace(/_/g, ' ')}</div>
                <div className="text-lg font-medium text-blue-400 mt-1">{typeof value === 'number' ? value.toFixed(4) : value}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default ResultsDisplay