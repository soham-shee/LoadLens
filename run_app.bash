#!/bin/bash

# Exit script on any error
set -e

# Activate virtual environment if available (optional)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run App.py
