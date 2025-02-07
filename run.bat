@echo off

:: Check if virtual environment exists and activate it
if exist venv (
    call venv\Scripts\activate
)

:: Install dependencies
pip install -r requirements.txt

:: Run Streamlit app
streamlit run App.py

pause