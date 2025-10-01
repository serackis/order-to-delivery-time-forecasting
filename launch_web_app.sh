#!/bin/bash

# Order Delivery Time Forecasting Web Application Launcher
# This script activates the virtual environment and starts the Flask web application

echo "🚀 Starting Order Delivery Time Forecasting Web Application..."
echo "================================================================"

# Check if virtual environment exists
if [ ! -d "delivery_forecast_env" ]; then
    echo "❌ Virtual environment 'delivery_forecast_env' not found!"
    echo "Please run the setup first or create the virtual environment manually."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source delivery_forecast_env/bin/activate

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import flask, pandas, numpy, sklearn, xgboost, matplotlib, seaborn, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Some required packages are missing!"
    echo "Installing missing packages..."
    pip install -r requirements.txt
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs models templates static

# Check if data directory exists and has required files
echo "📊 Checking data availability..."
if [ ! -d "data" ]; then
    echo "⚠️  Data directory not found. Creating it..."
    mkdir -p data
    echo "📋 Please download the required dataset files to the 'data' directory."
    echo "   See data/data_download_instructions.txt for details."
fi

# Start the Flask application
echo "🌐 Starting Flask web application on port 5003..."
echo "   Access the application at: http://localhost:5003"
echo "   Press Ctrl+C to stop the server"
echo "================================================================"

# Run the Flask app
python app.py

