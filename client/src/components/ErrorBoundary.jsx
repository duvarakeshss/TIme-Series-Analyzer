import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error to console
    console.error("Error caught by ErrorBoundary:", error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      // Custom fallback UI
      return (
        <div className="mt-6 max-w-5xl mx-auto bg-gray-700 rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold text-white mb-4">Something went wrong</h2>
          <div className="bg-red-800/30 p-4 rounded-lg text-red-200 mb-4">
            <p className="mb-2">An error occurred in the results display component:</p>
            <pre className="bg-gray-900 p-3 rounded text-sm overflow-auto max-h-[200px]">
              {this.state.error && this.state.error.toString()}
            </pre>
          </div>
          <button 
            onClick={() => window.location.reload()}
            className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded"
          >
            Reload Page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary; 