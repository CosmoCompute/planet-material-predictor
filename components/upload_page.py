import streamlit as st
import pandas as pd
import duckdb
import os
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def header():
    st.markdown("""
        <div class="main-header-upload-section">
            <h1 class="header-title-upload-section">Professional Data Analytics Platform</h1>
            <p class="header-subtitle-upload-section">Advanced file upload, analysis, and data modeling capabilities</p>
        </div>
    """, unsafe_allow_html=True)

def upload():
    header()
    
    with st.sidebar:
        st.header("⚙️ Settings")


#upload page update