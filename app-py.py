"""
Rename the original file to app.py for easier deployment on Streamlit Cloud.
This is a simple redirection to the main application file.
"""

import streamlit as st
import os
import sys

# Add the current directory to the path to find the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main function from the resume enhancement app
try:
    from resume_enhancement_app import main
    main()
except ImportError:
    st.error("Could not import the main application. Check file naming and paths.")
