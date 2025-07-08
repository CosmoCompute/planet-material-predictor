import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Climate Analysis Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
    }
    .trend-up { color: #e74c3c; font-weight: bold; }
    .trend-down { color: #3498db; font-weight: bold; }
    .alert-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Directory setup
MODEL_DIR = 'models'
DATA_DIR = 'data/data_temp'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

try:
    from utils import db_utils
    from models.temp_model import load_model
except ImportError:
    st.warning("Some modules not found. Running in demo mode.")
    db_utils = None
    load_model = None

def load_and_prepare_data():
    """Load and prepare climate data with enhanced features"""
    try:
        if db_utils:
            directory = Path("data/data_temp")
            file_dict = {f.stem: f.name for f in directory.iterdir() if f.is_file()}
            selected_name = st.selectbox("📁 Select Dataset:", list(file_dict.keys()))
            file_name = file_dict[selected_name]
            df = db_utils.load_db(file_name)
        else:
            # Demo data generation
            st.info("Running in demo mode with synthetic data")
            dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='D')
            np.random.seed(42)
            
            # Generate realistic temperature data with seasonal patterns
            day_of_year = dates.dayofyear
            seasonal_temp = 20 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            noise = np.random.normal(0, 3, len(dates))
            trend = 0.002 * np.arange(len(dates))  # Small warming trend
            
            df = pd.DataFrame({
                'Date': dates,
                'Maximum Temperature': seasonal_temp + noise + trend,
                'Minimum Temperature': seasonal_temp - 8 + noise * 0.8 + trend,
                'Wind Speed': np.random.normal(15, 5, len(dates)),
                'Pressure': np.random.normal(1013, 10, len(dates))
            })
            
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        # Essential feature engineering
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day_of_Year'] = df.index.dayofyear
        df['Temperature_Range'] = df['Maximum Temperature'] - df['Minimum Temperature']
        df['Temperature_Average'] = (df['Maximum Temperature'] + df['Minimum Temperature']) / 2
        
        # Key rolling statistics
        df['Max_Temp_30D'] = df['Maximum Temperature'].rolling(window=30, center=True).mean()
        df['Max_Temp_365D'] = df['Maximum Temperature'].rolling(window=365, center=True).mean()
        
        # Extreme weather indicators
        df['Is_Extreme_Hot'] = df['Maximum Temperature'] > df['Maximum Temperature'].quantile(0.95)
        df['Is_Extreme_Cold'] = df['Maximum Temperature'] < df['Maximum Temperature'].quantile(0.05)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_simple_model(df):
    """Create a simple but effective prediction model"""
    try:
        features = ['Month', 'Day_of_Year']
        target = 'Maximum Temperature'
        
        # Prepare clean data
        model_data = df[features + [target]].dropna()
        X = model_data[features]
        y = model_data[target]
        
        # Use 80% for training, 20% for testing
        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Train simple model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model': model,
            'metrics': {'MAE': mae, 'RMSE': rmse, 'R2': r2},
            'test_data': (X_test, y_test, y_pred)
        }
        
    except Exception as e:
        st.error(f"Error creating model: {e}")
        return None

def main_dashboard():
    """Main dashboard function"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌡️ Climate Analysis Dashboard</h1>
        <p>Professional climate data analysis with trend detection and forecasting</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading climate data..."):
        df = load_and_prepare_data()
    
    if df is None:
        st.error("Failed to load climate data.")
        return
    
    # Data overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Data Points</h3>
            <h2>{len(df):,}</h2>
            <p>Total records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        years_span = df.index.year.max() - df.index.year.min() + 1
        st.markdown(f"""
        <div class="metric-card">
            <h3>📅 Time Span</h3>
            <h2>{years_span}</h2>
            <p>Years of data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_temp = df['Maximum Temperature'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌡️ Avg Temperature</h3>
            <h2>{avg_temp:.1f}°C</h2>
            <p>Maximum daily</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        completeness = (df['Maximum Temperature'].notna().sum() / len(df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>✅ Data Quality</h3>
            <h2>{completeness:.1f}%</h2>
            <p>Complete records</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Climate Trend Analysis
    st.markdown("## 📈 Climate Trend Analysis")
    
    # Calculate trends
    yearly_stats = df.groupby('Year').agg({
        'Maximum Temperature': 'mean',
        'Minimum Temperature': 'mean',
        'Temperature_Average': 'mean'
    }).reset_index()
    
    # Statistical analysis
    years = yearly_stats['Year'].values
    avg_temps = yearly_stats['Temperature_Average'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, avg_temps)
    
    # Display trend results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trend_class = "trend-up" if slope > 0 else "trend-down"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Temperature Trend</h4>
            <h3 class="{trend_class}">{slope:.4f}°C/year</h3>
            <p>Average change rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        significance = "Significant" if p_value < 0.05 else "Not Significant"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Statistical Significance</h4>
            <h3>{significance}</h3>
            <p>p-value: {p_value:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_change = slope * (years.max() - years.min())
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Change</h4>
            <h3>{total_change:.2f}°C</h3>
            <p>Over {years.max() - years.min()} years</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Trend visualization
    fig_trend = go.Figure()
    
    # Temperature data
    fig_trend.add_trace(go.Scatter(
        x=df.index, y=df['Maximum Temperature'],
        mode='lines', name='Daily Max Temperature',
        line=dict(color='lightblue', width=1), opacity=0.6
    ))
    
    # Annual average
    fig_trend.add_trace(go.Scatter(
        x=df.index, y=df['Max_Temp_365D'],
        mode='lines', name='Annual Average',
        line=dict(color='blue', width=3)
    ))
    
    # Trend line
    trend_line = slope * df.index.year + intercept
    fig_trend.add_trace(go.Scatter(
        x=df.index, y=trend_line,
        mode='lines', name=f'Trend ({slope:.4f}°C/year)',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig_trend.update_layout(
        title="Climate Trend Analysis",
        xaxis_title="Year",
        yaxis_title="Temperature (°C)",
        height=500
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Climate assessment
    if p_value < 0.05:
        if abs(slope) > 0.02:
            alert_type = "warming" if slope > 0 else "cooling"
            st.markdown(f"""
            <div class="alert-box">
                <h4>⚠️ Climate Alert: Significant {alert_type.title()} Trend Detected</h4>
                <p>Rate: {slope:.4f}°C/year | Statistical confidence: {(1-p_value)*100:.1f}%</p>
                <p>This trend is statistically significant and represents notable climate change.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-box">
                <h4>📊 Moderate Climate Change Detected</h4>
                <p>Rate: {slope:.4f}°C/year | Statistical confidence: {(1-p_value)*100:.1f}%</p>
                <p>Changes are statistically significant but within moderate ranges.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-box">
            <h4>📊 No Significant Climate Trend</h4>
            <p>Rate: {slope:.4f}°C/year | Statistical confidence: {(1-p_value)*100:.1f}%</p>
            <p>Temperature variations appear to be within natural fluctuation ranges.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Simple Prediction Model
    st.markdown("## 🔮 Temperature Predictions")
    
    model_result = create_simple_model(df)
    
    if model_result:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            r2_score = model_result['metrics']['R2']
            accuracy = "High" if r2_score > 0.8 else "Medium" if r2_score > 0.6 else "Low"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Model Accuracy</h4>
                <h3>{accuracy}</h3>
                <p>R² Score: {r2_score:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            mae = model_result['metrics']['MAE']
            st.markdown(f"""
            <div class="metric-card">
                <h4>Prediction Error</h4>
                <h3>{mae:.2f}°C</h3>
                <p>Mean Absolute Error</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            rmse = model_result['metrics']['RMSE']
            st.markdown(f"""
            <div class="metric-card">
                <h4>RMSE</h4>
                <h3>{rmse:.2f}°C</h3>
                <p>Root Mean Square Error</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Prediction vs Actual plot
        X_test, y_test, y_pred = model_result['test_data']
        
        fig_pred = go.Figure()
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig_pred.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        # Actual vs Predicted
        fig_pred.add_trace(go.Scatter(
            x=y_test, y=y_pred,
            mode='markers', name='Predictions',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        fig_pred.update_layout(
            title="Model Performance: Predicted vs Actual Temperature",
            xaxis_title="Actual Temperature (°C)",
            yaxis_title="Predicted Temperature (°C)",
            height=500
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Simple future forecast
        st.markdown("### 📅 Short-term Forecast")
        
        # Generate next 30 days forecast
        last_date = df.index.max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
        
        future_features = pd.DataFrame({
            'Month': future_dates.month,
            'Day_of_Year': future_dates.dayofyear
        })
        
        future_predictions = model_result['model'].predict(future_features)
        
        # Plot forecast
        fig_forecast = go.Figure()
        
        # Recent historical data
        recent_data = df.tail(60)
        fig_forecast.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data['Maximum Temperature'],
            mode='lines', name='Historical Data',
            line=dict(color='blue', width=2)
        ))
        
        # Future predictions
        fig_forecast.add_trace(go.Scatter(
            x=future_dates, y=future_predictions,
            mode='lines+markers', name='30-Day Forecast',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig_forecast.update_layout(
            title="30-Day Temperature Forecast",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            height=400
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Extreme Weather Analysis
    st.markdown("## ⚡ Extreme Weather Analysis")
    
    # Calculate extreme events
    extreme_hot = df[df['Is_Extreme_Hot']]
    extreme_cold = df[df['Is_Extreme_Cold']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        hot_days_per_year = len(extreme_hot) / df.index.year.nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔥 Extreme Hot Days</h4>
            <h3>{len(extreme_hot)} total</h3>
            <p>{hot_days_per_year:.1f} days/year average</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        cold_days_per_year = len(extreme_cold) / df.index.year.nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h4>❄️ Extreme Cold Days</h4>
            <h3>{len(extreme_cold)} total</h3>
            <p>{cold_days_per_year:.1f} days/year average</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Extreme events over time
    yearly_extremes = df.groupby('Year').agg({
        'Is_Extreme_Hot': 'sum',
        'Is_Extreme_Cold': 'sum'
    }).reset_index()
    
    fig_extreme = go.Figure()
    
    fig_extreme.add_trace(go.Scatter(
        x=yearly_extremes['Year'], y=yearly_extremes['Is_Extreme_Hot'],
        mode='lines+markers', name='Extreme Hot Days',
        line=dict(color='red', width=2)
    ))
    
    fig_extreme.add_trace(go.Scatter(
        x=yearly_extremes['Year'], y=yearly_extremes['Is_Extreme_Cold'],
        mode='lines+markers', name='Extreme Cold Days',
        line=dict(color='blue', width=2)
    ))
    
    fig_extreme.update_layout(
        title="Extreme Weather Events Over Time",
        xaxis_title="Year",
        yaxis_title="Number of Days",
        height=400
    )
    
    st.plotly_chart(fig_extreme, use_container_width=True)
    
    # Summary
    st.markdown("## 📋 Summary")
    
    # Generate key insights
    insights = []
    
    if p_value < 0.05 and abs(slope) > 0.01:
        trend_type = "warming" if slope > 0 else "cooling"
        insights.append(f"• Significant {trend_type} trend: {slope:.4f}°C/year")
    
    if model_result and model_result['metrics']['R2'] > 0.7:
        insights.append(f"• High prediction accuracy achieved (R² = {model_result['metrics']['R2']:.3f})")
    
    if hot_days_per_year > 10:
        insights.append(f"• High frequency of extreme hot days: {hot_days_per_year:.1f} per year")
    
    if len(insights) > 0:
        insight_text = "\n".join(insights)
        st.markdown(f"""
        <div class="alert-box">
            <h4>🎯 Key Insights</h4>
            <p>{insight_text}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-box">
            <h4>📊 Analysis Summary</h4>
            <p>• Climate data shows normal variability patterns<br>
            • No significant extreme trends detected<br>
            • Prediction models show reasonable accuracy</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main_dashboard()