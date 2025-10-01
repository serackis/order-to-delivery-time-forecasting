#!/usr/bin/env python3
"""
Order Delivery Time Forecasting Web Application
A Flask web interface for running and visualizing ML models for delivery time prediction
"""

import os
import sys
import json
import time
import subprocess
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.utils
from io import BytesIO
import base64

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)

# Global variables to store execution status and results
execution_status = {
    'running': False,
    'current_task': None,
    'progress': 0,
    'logs': [],
    'results': None,
    'error': None
}

def log_message(message):
    """Add a message to the execution logs"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    execution_status['logs'].append(log_entry)
    print(log_entry)

def run_ml_pipeline(pipeline_type, use_intel=False):
    """Run the ML pipeline in a separate thread"""
    global execution_status
    
    try:
        execution_status['running'] = True
        execution_status['current_task'] = pipeline_type
        execution_status['progress'] = 0
        execution_status['error'] = None
        execution_status['logs'] = []
        
        log_message(f"Starting {pipeline_type} pipeline...")
        
        # Check if data exists
        data_files = [
            'data/olist_orders_dataset.csv',
            'data/olist_order_items_dataset.csv',
            'data/olist_customers_dataset.csv',
            'data/olist_sellers_dataset.csv',
            'data/olist_geolocation_dataset.csv',
            'data/olist_products_dataset.csv'
        ]
        
        missing_files = [f for f in data_files if not os.path.exists(f)]
        if missing_files:
            execution_status['error'] = f"Missing data files: {', '.join(missing_files)}. Please download the dataset first."
            execution_status['running'] = False
            return
        
        execution_status['progress'] = 10
        log_message("Data files found. Starting preprocessing...")
        
        # Prepare command
        script_name = f"run_benchmarks_{pipeline_type}.py"
        log_file = f"logs/{pipeline_type}_{'intel' if use_intel else 'stock'}.log"
        model_file = f"models/{pipeline_type}_{'intel' if use_intel else 'stock'}.pkl"
        
        # Create directories if they don't exist
        os.makedirs('logs', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        cmd = [
            sys.executable, 
            f"src/{script_name}",
            "-l", log_file,
            "-m", model_file
        ]
        
        if use_intel:
            cmd.append("-i")
        
        execution_status['progress'] = 20
        log_message(f"Executing command: {' '.join(cmd)}")
        
        # Run the pipeline
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Monitor progress
        while process.poll() is None:
            time.sleep(1)
            execution_status['progress'] = min(execution_status['progress'] + 2, 90)
        
        # Wait for completion
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            execution_status['progress'] = 100
            log_message(f"{pipeline_type} pipeline completed successfully!")
            
            # Parse results from log file
            results = parse_log_results(log_file)
            execution_status['results'] = results
            
        else:
            execution_status['error'] = f"Pipeline failed with return code {process.returncode}: {stderr or stdout}"
            log_message(f"Pipeline failed: {execution_status['error']}")
            
    except Exception as e:
        execution_status['error'] = str(e)
        log_message(f"Error running pipeline: {str(e)}")
    finally:
        execution_status['running'] = False
        execution_status['current_task'] = None

def parse_log_results(log_file):
    """Parse results from the log file"""
    results = {
        'hyperparameter_times': {},
        'training_times': {},
        'inference_results': {},
        'streaming_times': {}
    }
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Parse hyperparameter tuning times
            if 'Hyperparameter tuning time for for:' in line:
                if 'XGB:' in line:
                    time_val = extract_time_from_line(line)
                    results['hyperparameter_times']['XGB'] = time_val
                elif 'RF:' in line:
                    time_val = extract_time_from_line(line)
                    results['hyperparameter_times']['RF'] = time_val
                elif 'SV:' in line:
                    time_val = extract_time_from_line(line)
                    results['hyperparameter_times']['SV'] = time_val
                elif 'ensemble model:' in line:
                    time_val = extract_time_from_line(line)
                    results['hyperparameter_times']['Ensemble'] = time_val
            
            # Parse training times
            elif 'Training time for for: ensemble model:' in line:
                time_val = extract_time_from_line(line)
                results['training_times']['Ensemble'] = time_val
            
            # Parse inference results
            elif 'Inference time and MSE for for' in line or 'Inference time and MSE for for' in line:
                if 'XGB:' in line:
                    time_val, mse_val = extract_time_mse_from_line(line)
                    results['inference_results']['XGB'] = {'time': time_val, 'mse': mse_val}
                elif 'RF:' in line:
                    time_val, mse_val = extract_time_mse_from_line(line)
                    results['inference_results']['RF'] = {'time': time_val, 'mse': mse_val}
                elif 'SV:' in line:
                    time_val, mse_val = extract_time_mse_from_line(line)
                    results['inference_results']['SV'] = {'time': time_val, 'mse': mse_val}
                elif 'Voting model:' in line:
                    time_val, mse_val = extract_time_mse_from_line(line)
                    results['inference_results']['Ensemble'] = {'time': time_val, 'mse': mse_val}
            
            # Parse streaming times
            elif 'Average Streaming Inference Time for' in line:
                time_val = extract_time_from_line(line)
                if 'Voting Regressor' in line:
                    results['streaming_times']['Regression'] = time_val
                elif 'Voting Classifier' in line:
                    results['streaming_times']['Classification'] = time_val
                    
    except Exception as e:
        log_message(f"Error parsing log results: {str(e)}")
    
    return results

def extract_time_from_line(line):
    """Extract time value from a log line"""
    try:
        # Look for time pattern like "time: 1.23" or "time: 1.23s"
        import re
        match = re.search(r'time[:\s]+([\d.]+)', line)
        if match:
            return float(match.group(1))
    except:
        pass
    return 0.0

def extract_time_mse_from_line(line):
    """Extract time and MSE values from a log line"""
    try:
        import re
        # Look for pattern like "time: 1.23, mse: 4.56"
        match = re.search(r'time[:\s]+([\d.]+).*?mse[:\s]+([\d.]+)', line)
        if match:
            return float(match.group(1)), float(match.group(2))
    except:
        pass
    return 0.0, 0.0

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get current execution status"""
    return jsonify(execution_status)

@app.route('/api/run', methods=['POST'])
def run_pipeline():
    """Start running a pipeline"""
    global execution_status
    
    if execution_status['running']:
        return jsonify({'error': 'Pipeline is already running'}), 400
    
    data = request.get_json()
    pipeline_type = data.get('pipeline_type', 'regression')
    use_intel = data.get('use_intel', False)
    
    if pipeline_type not in ['regression', 'classification']:
        return jsonify({'error': 'Invalid pipeline type'}), 400
    
    # Start pipeline in a separate thread
    thread = threading.Thread(target=run_ml_pipeline, args=(pipeline_type, use_intel))
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': f'{pipeline_type} pipeline started'})

@app.route('/api/stop', methods=['POST'])
def stop_pipeline():
    """Stop the current pipeline"""
    global execution_status
    
    if not execution_status['running']:
        return jsonify({'error': 'No pipeline is currently running'}), 400
    
    execution_status['running'] = False
    execution_status['current_task'] = None
    execution_status['error'] = 'Pipeline stopped by user'
    
    return jsonify({'message': 'Pipeline stopped'})

@app.route('/api/results')
def get_results():
    """Get the latest results"""
    return jsonify(execution_status.get('results', {}))

@app.route('/api/visualize')
def visualize_results():
    """Generate visualization of results"""
    results = execution_status.get('results', {})
    
    if not results:
        return jsonify({'error': 'No results available'}), 400
    
    # Create visualizations
    plots = {}
    
    # Hyperparameter tuning times
    if 'hyperparameter_times' in results and results['hyperparameter_times']:
        fig, ax = plt.subplots(figsize=(10, 6))
        models = list(results['hyperparameter_times'].keys())
        times = list(results['hyperparameter_times'].values())
        
        bars = ax.bar(models, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_title('Hyperparameter Tuning Times')
        ax.set_ylabel('Time (seconds)')
        ax.set_xlabel('Models')
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{time:.2f}s', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        plots['hyperparameter_times'] = plot_data
    
    # Inference results comparison
    if 'inference_results' in results and results['inference_results']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        models = list(results['inference_results'].keys())
        times = [results['inference_results'][m]['time'] for m in models]
        mses = [results['inference_results'][m]['mse'] for m in models]
        
        # Time comparison
        bars1 = ax1.bar(models, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('Inference Times')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xlabel('Models')
        
        for bar, time in zip(bars1, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{time:.4f}s', ha='center', va='bottom', fontsize=8)
        
        # MSE comparison
        bars2 = ax2.bar(models, mses, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_title('Mean Squared Error')
        ax2.set_ylabel('MSE')
        ax2.set_xlabel('Models')
        
        for bar, mse in zip(bars2, mses):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{mse:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        plots['inference_comparison'] = plot_data
    
    return jsonify(plots)

@app.route('/api/data_status')
def data_status():
    """Check if required data files exist"""
    data_files = [
        'olist_orders_dataset.csv',
        'olist_order_items_dataset.csv',
        'olist_customers_dataset.csv',
        'olist_sellers_dataset.csv',
        'olist_geolocation_dataset.csv',
        'olist_products_dataset.csv'
    ]
    
    status = {}
    for file in data_files:
        file_path = os.path.join('data', file)
        status[file] = os.path.exists(file_path)
    
    return jsonify(status)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("Starting Order Delivery Time Forecasting Web Application...")
    print("Access the application at: http://localhost:5003")
    print("Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5003, debug=True)

