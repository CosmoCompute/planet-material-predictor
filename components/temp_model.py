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
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

# Directory setup
MODEL_DIR = 'models'
DATA_DIR = 'data/data_temp'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils import db_utils
from models.temp_model import load_model

def load_and_prepare_data():
    try:
        directory = Path("data/data_temp/")
        file_dict = {f.stem: f.name for f in directory.iterdir() if f.is_file()}
        selected_name = st.selectbox("Select a file:", list(file_dict.keys()))

        file_name = file_dict[selected_name]
        df = db_utils.load_db(file_name, "temp")
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        # Enhanced feature engineering
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day_of_Year'] = df.index.dayofyear
        df['Temperature_Range'] = df['Maximum Temperature'] - df['Minimum Temperature']
        df['Temperature_Average'] = (df['Maximum Temperature'] + df['Minimum Temperature']) / 2
        
        # Advanced rolling statistics
        for window in [7, 30, 90, 365]:
            df[f'Max_Temp_{window}D'] = df['Maximum Temperature'].rolling(window=window, center=True).mean()
            df[f'Min_Temp_{window}D'] = df['Minimum Temperature'].rolling(window=window, center=True).mean()
            df[f'Temp_Std_{window}D'] = df['Maximum Temperature'].rolling(window=window, center=True).std()
        
        # Seasonal indicators
        df['Season'] = df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        # Climate anomalies (deviations from long-term average)
        long_term_avg = df['Maximum Temperature'].mean()
        df['Temperature_Anomaly'] = df['Maximum Temperature'] - long_term_avg
        
        # Extreme weather indicators
        df['Is_Extreme_Hot'] = df['Maximum Temperature'] > df['Maximum Temperature'].quantile(0.95)
        df['Is_Extreme_Cold'] = df['Maximum Temperature'] < df['Maximum Temperature'].quantile(0.05)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_synthetic_model_data(df):
    """Create synthetic model data for demonstration if no trained model exists"""
    try:
        # Simulate model training results
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        # Prepare features for modeling
        features = ['Month', 'Day_of_Year', 'Wind Speed', 'Pressure']
        target = 'Maximum Temperature'
        
        # Remove NaN values
        model_data = df[features + [target]].dropna()
        X = model_data[features]
        y = model_data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train multiple models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = -np.inf
        model_results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            model_results[name] = {
                'model': model,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'predictions': y_pred,
                'actual': y_test
            }
            
            if r2 > best_score:
                best_score = r2
                best_model = model
        
        # Create synthetic SARIMAX-like results
        return {
            'fitted_model': best_model,
            'model_info': {
                'order': '(2,1,2)',
                'seasonal_order': '(1,1,1,12)',
                'aic': 15432.5,
                'n_observations': len(model_data),
                'evaluation': model_results['Random Forest']
            },
            'model_results': model_results,
            'features': features,
            'test_data': (X_test, y_test)
        }
        
    except Exception as e:
        st.error(f"Error creating model data: {e}")
        return None

def load_trained_model():
    """Load the trained model or create synthetic data"""
    try:  
        model = load_model()
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def generate_sparkline_svg(values, color):
    if len(values) == 0:
        return ""
    val_min, val_max = min(values), max(values)
    val_range = (val_max - val_min) if val_max != val_min else 1
    points = []
    for idx, val in enumerate(values):
        x = (idx / (len(values) - 1)) * 100 if len(values) > 1 else 50
        y = 35 - ((val - val_min) / val_range) * 30
        points.append(f"{x},{y}")
    points_str = " ".join(points)
    return f"""
    <svg width="100" height="40" style="vertical-align: middle;">
        <polyline fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="{points_str}" />
    </svg>
    """

def climate_trend_analysis(df):
    """Enhanced climate trend analysis with statistical significance"""
    st.markdown('<div class="dashboard-section-header">📈 Climate Trend Analysis</div>', unsafe_allow_html=True)
    
    # Calculate yearly statistics
    yearly_stats = df.groupby('Year').agg({
        'Maximum Temperature': ['mean', 'max', 'min', 'std'],
        'Minimum Temperature': ['mean', 'max', 'min'],
        'Temperature_Average': 'mean',
        'Temperature_Anomaly': 'mean',
        'Wind Speed': 'mean',
        'Pressure': 'mean'
    }).round(3)
    
    yearly_stats.columns = ['_'.join(col).strip() for col in yearly_stats.columns]
    yearly_stats = yearly_stats.reset_index()
    
    # Statistical trend analysis
    years = yearly_stats['Year'].values
    max_temp_slope, max_temp_intercept, max_temp_r, max_temp_p, _ = stats.linregress(years, yearly_stats['Maximum Temperature_mean'])
    min_temp_slope, min_temp_intercept, min_temp_r, min_temp_p, _ = stats.linregress(years, yearly_stats['Minimum Temperature_mean'])
    avg_temp_slope, avg_temp_intercept, avg_temp_r, avg_temp_p, _ = stats.linregress(years, yearly_stats['Temperature_Average_mean'])
    
    # Generate sparkline SVGs
    max_spark = generate_sparkline_svg(yearly_stats['Maximum Temperature_mean'].values, '#ef4444')
    min_spark = generate_sparkline_svg(yearly_stats['Minimum Temperature_mean'].values, '#3b82f6')
    avg_spark = generate_sparkline_svg(yearly_stats['Temperature_Average_mean'].values, '#10b981')
    change_spark = generate_sparkline_svg(yearly_stats['Temperature_Average_mean'].values, '#f97316')

    # Display enhanced trend metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        significance = "Significant" if max_temp_p < 0.05 else "Not Significant"
        trend_class = "trend-up" if max_temp_slope > 0 else "trend-down"
        arrow = "↑" if max_temp_slope > 0 else "↓"
        st.markdown(f'<div class="metric-container"><div style="display: flex; justify-content: space-between; align-items: flex-start;"><div><h4>Max Temperature Trend</h4><p class="{trend_class}">{max_temp_slope:.4f}°C/yr {arrow}</p></div><div style="background: #fef2f2; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem;">🌡️</div></div><div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 12px;"><div style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">R² = {max_temp_r**2:.3f}<br>{significance} (p={max_temp_p:.3f})</div><div class="metric-sparkline">{max_spark}</div></div></div>', unsafe_allow_html=True)
    
    with col2:
        significance = "Significant" if min_temp_p < 0.05 else "Not Significant"
        trend_class = "trend-up" if min_temp_slope > 0 else "trend-down"
        arrow = "↑" if min_temp_slope > 0 else "↓"
        st.markdown(f'<div class="metric-container"><div style="display: flex; justify-content: space-between; align-items: flex-start;"><div><h4>Min Temperature Trend</h4><p class="{trend_class}">{min_temp_slope:.4f}°C/yr {arrow}</p></div><div style="background: #eff6ff; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem;">❄️</div></div><div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 12px;"><div style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">R² = {min_temp_r**2:.3f}<br>{significance} (p={min_temp_p:.3f})</div><div class="metric-sparkline">{min_spark}</div></div></div>', unsafe_allow_html=True)
    
    with col3:
        significance = "Significant" if avg_temp_p < 0.05 else "Not Significant"
        trend_class = "trend-up" if avg_temp_slope > 0 else "trend-down"
        arrow = "↑" if avg_temp_slope > 0 else "↓"
        st.markdown(f'<div class="metric-container"><div style="display: flex; justify-content: space-between; align-items: flex-start;"><div><h4>Average Temp Trend</h4><p class="{trend_class}">{avg_temp_slope:.4f}°C/yr {arrow}</p></div><div style="background: #f0fdf4; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem;">🌡️</div></div><div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 12px;"><div style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">R² = {avg_temp_r**2:.3f}<br>{significance} (p={avg_temp_p:.3f})</div><div class="metric-sparkline">{avg_spark}</div></div></div>', unsafe_allow_html=True)
    
    with col4:
        years_span = years.max() - years.min()
        total_change = avg_temp_slope * years_span
        trend_class = "trend-up" if total_change > 0 else "trend-down"
        arrow = "↑" if total_change > 0 else "↓"
        st.markdown(f'<div class="metric-container"><div style="display: flex; justify-content: space-between; align-items: flex-start;"><div><h4>Total Change</h4><p class="{trend_class}">{total_change:.3f}°C {arrow}</p></div><div style="background: #fff7ed; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem;">📈</div></div><div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 12px;"><div style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">Over {years_span} years<br>Data Quality: High</div><div class="metric-sparkline">{change_spark}</div></div></div>', unsafe_allow_html=True)
    
    # Enhanced climate assessment with status-bar-container
    if avg_temp_p < 0.05:  # Statistically significant
        if avg_temp_slope > 0.02:
            status_icon = "🚨"
            status_title = "Climate Alert: Statistically Significant Warming Detected"
            status_bg = "#fef2f2"
            status_color = "#ef4444"
            status_assessment = f"This warming rate is significant and exceeds normal baseline variability. At this rate, temperatures could increase by {avg_temp_slope*50:.2f}°C over the next 50 years."
        elif avg_temp_slope < -0.02:
            status_icon = "❄️"
            status_title = "Climate Alert: Statistically Significant Cooling Detected"
            status_bg = "#eff6ff"
            status_color = "#3b82f6"
            status_assessment = "A statistically significant cooling trend has been identified, which is unusual in current long-term global climate projections."
        else:
            status_icon = "📊"
            status_title = "Climate Status: Mild but Significant Temperature Change"
            status_bg = "#f0fdf4"
            status_color = "#10b981"
            status_assessment = "Temperature changes are statistically significant but remain within moderate, manageable limits."
    else:
        status_icon = "🛡️"
        status_title = "Climate Status: No Statistically Significant Trend"
        status_bg = "#f0f9ff"
        status_color = "#0284c7"
        status_assessment = "Temperature variations appear to be within natural fluctuation ranges. No critical trend is detected."

    st.markdown(f'<div class="status-bar-container"><div class="status-left"><div class="status-icon-wrapper" style="background: {status_bg}; color: {status_color};">{status_icon}</div><div class="status-info"><span class="status-title" style="color: {status_color}; font-weight: 600;">{status_title}</span><span class="status-sub">Rate: {avg_temp_slope:.4f}°C/year (p-value: {avg_temp_p:.4f})</span></div></div><div class="status-assessment"><strong>Assessment:</strong> {status_assessment}</div></div>', unsafe_allow_html=True)
    
    return yearly_stats, max_temp_slope, min_temp_slope, avg_temp_slope

def apply_plotly_theme(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, -apple-system, sans-serif', size=11, color='#475569'),
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#f1f5f9',
            linecolor='#cbd5e1',
            tickfont=dict(color='#64748b', size=10)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#f1f5f9',
            linecolor='#cbd5e1',
            tickfont=dict(color='#64748b', size=10)
        )
    )

def create_comprehensive_plots(df, yearly_stats):
    """Create enhanced comprehensive visualization plots"""
    st.markdown('<div class="dashboard-section-header visualizations">📊 Comprehensive Climate Visualizations</div>', unsafe_allow_html=True)
    
    # 1. Enhanced Temperature Trend Analysis
    fig1 = go.Figure()
    
    # Calculate confidence interval for trend line
    years_full = df.index.year
    slope, intercept, _, _, _ = stats.linregress(years_full, df['Maximum Temperature'].ffill())
    trend_line = slope * years_full + intercept
    
    # Statistical CI estimation around trend line
    residuals = df['Maximum Temperature'].ffill() - trend_line
    std_err_resid = np.std(residuals)
    ci_upper = trend_line + 1.96 * std_err_resid
    ci_lower = trend_line - 1.96 * std_err_resid
    
    # Shade CI (Upper bound and then filled lower bound)
    fig1.add_trace(go.Scatter(
        x=df.index, y=ci_upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig1.add_trace(go.Scatter(
        x=df.index, y=ci_lower, mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(239, 68, 68, 0.08)', name='95% Confidence Interval'
    ))
    
    # Temperature trends
    fig1.add_trace(go.Scatter(
        x=df.index, y=df['Maximum Temperature'], name='Max Temperature',
        line=dict(color='#ef4444', width=1), opacity=0.5
    ))
    fig1.add_trace(go.Scatter(
        x=df.index, y=df['Max_Temp_365D'], name='Annual Moving Average',
        line=dict(color='#991b1b', width=2.5)
    ))
    fig1.add_trace(go.Scatter(
        x=df.index, y=trend_line, name=f'Trend Line ({slope:.4f}°C/year)',
        line=dict(color='#1e293b', dash='dash', width=2)
    ))
    fig1.update_layout(title="")
    apply_plotly_theme(fig1)
    
    # Temperature anomalies
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df.index, y=df['Temperature_Anomaly'], name='Temperature Anomaly',
        line=dict(color='#8b5cf6', width=1.5), fill='tonexty', fillcolor='rgba(139, 92, 246, 0.15)'
    ))
    fig2.add_hline(y=0, line_dash="dash", line_color="#475569", opacity=0.5)
    fig2.update_layout(title="")
    apply_plotly_theme(fig2)
    
    # Tab Layout for better UI grouping
    tab1, tab2 = st.tabs(["📈 Temperature Analysis", "🔍 Detailed Insights"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Long-term Temperature Trends with Confidence Intervals")
            st.plotly_chart(fig1, width='stretch')
        with col2:
            st.markdown("### Temperature Anomalies Over Time")
            st.plotly_chart(fig2, width='stretch')
            
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Seasonal patterns
            fig3 = go.Figure()
            season_colors = {'Spring': '#10b981', 'Summer': '#f97316', 'Autumn': '#b45309', 'Winter': '#3b82f6'}
            for season in ['Spring', 'Summer', 'Autumn', 'Winter']:
                season_data = df[df['Season'] == season]['Maximum Temperature']
                fig3.add_trace(go.Box(y=season_data, name=season, boxpoints='outliers', marker_color=season_colors[season]))
            fig3.update_layout(title="")
            apply_plotly_theme(fig3)
            st.markdown("### Seasonal Temperature Patterns")
            st.plotly_chart(fig3, width='stretch')
            
        with col2:
            # Temperature variability
            fig4 = go.Figure()
            monthly_variability = df.groupby('Month')['Temperature_Range'].mean()
            fig4.add_trace(go.Scatter(
                x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                y=monthly_variability.values, mode='lines+markers', name='Monthly Temp Range',
                line=dict(color='#f97316', width=3), marker=dict(size=8)
            ))
            fig4.update_layout(title="")
            apply_plotly_theme(fig4)
            st.markdown("### Temperature Variability Analysis")
            st.plotly_chart(fig4, width='stretch')
            
        # 2. Advanced Statistical Analysis (Distribution and Correlation)
        st.markdown('<div class="dashboard-section-header">📊 Statistical Climate Insights</div>', unsafe_allow_html=True)
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            # Temperature distribution analysis
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=df['Maximum Temperature'], name='Temperature Distribution',
                nbinsx=50, opacity=0.6, marker_color='#3b82f6'
            ))
            
            # Add normal distribution overlay
            mean_temp = df['Maximum Temperature'].mean()
            std_temp = df['Maximum Temperature'].std()
            x_norm = np.linspace(df['Maximum Temperature'].min(), df['Maximum Temperature'].max(), 100)
            y_norm = stats.norm.pdf(x_norm, mean_temp, std_temp) * len(df) * (df['Maximum Temperature'].max() - df['Maximum Temperature'].min()) / 50
            
            fig_dist.add_trace(go.Scatter(
                x=x_norm, y=y_norm, name='Normal Distribution Fit',
                line=dict(color='#ef4444', width=2)
            ))
            
            fig_dist.update_layout(
                title="",
                xaxis_title="Temperature (°C)",
                yaxis_title="Frequency"
            )
            apply_plotly_theme(fig_dist)
            st.markdown("### Temperature Distribution Analysis")
            st.plotly_chart(fig_dist, width='stretch')
            
        with col_stat2:
            # Correlation heatmap
            correlation_vars = ['Maximum Temperature', 'Minimum Temperature', 'Wind Speed', 'Pressure', 'Temperature_Range']
            corr_matrix = df[correlation_vars].corr()
            
            fig_corr = px.imshow(
                corr_matrix, title="",
                color_continuous_scale='RdBu_r', text_auto=True
            )
            # Style the px heatmap cleanly
            fig_corr.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter, -apple-system, sans-serif', size=11),
                margin=dict(l=40, r=20, t=50, b=40)
            )
            st.markdown("### Weather Parameters Correlation Matrix")
            st.plotly_chart(fig_corr, width='stretch')

def create_enhanced_prediction_analysis(df, model_data):
    """Create enhanced prediction analysis with multiple models and validation"""
    st.markdown('<div class="section-header">🔮 Advanced Prediction Analysis</div>', unsafe_allow_html=True)
    
    if model_data is None:
        # Create synthetic model data for demonstration
        model_data = create_synthetic_model_data(df)
    
    if model_data is None:
        st.warning("Unable to create model data for predictions.")
        return
    
    # Model Performance Dashboard
    st.subheader("🎯 Model Performance Dashboard")
    
    if 'model_results' in model_data:
        model_results = model_data['model_results']
        
        # Display results for all models
        for model_name, results in model_results.items():
            st.markdown(f"### {model_name} Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Determine accuracy level
            r2_score = results['R2']
            if r2_score > 0.8:
                accuracy_class = "accuracy-high"
                accuracy_text = "Excellent"
            elif r2_score > 0.6:
                accuracy_class = "accuracy-medium"
                accuracy_text = "Good"
            else:
                accuracy_class = "accuracy-low"
                accuracy_text = "Poor"
            
            with col1:
                st.markdown(f'<div class="metric-container"><h4>R² Score</h4><p class="{accuracy_class}">{r2_score:.4f}</p><small>{accuracy_text} Fit</small></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="metric-container"><h4>MAE</h4><p>{results["MAE"]:.3f}°C</p><small>Mean Absolute Error</small></div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'<div class="metric-container"><h4>RMSE</h4><p>{results["RMSE"]:.3f}°C</p><small>Root Mean Square Error</small></div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown(f'<div class="metric-container"><h4>MAPE</h4><p>{results["MAPE"]:.2f}%</p><small>Mean Absolute % Error</small></div>', unsafe_allow_html=True)
            
            # Prediction vs Actual plot
            fig_pred = go.Figure()
            
            # Perfect prediction line
            min_val = min(results['actual'].min(), results['predictions'].min())
            max_val = max(results['actual'].max(), results['predictions'].max())
            fig_pred.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                        mode='lines', name='Perfect Prediction',
                                        line=dict(color='red', dash='dash')))
            
            # Actual vs Predicted scatter
            fig_pred.add_trace(go.Scatter(x=results['actual'], y=results['predictions'],
                                        mode='markers', name='Predictions',
                                        marker=dict(color='blue', opacity=0.6)))
            
            fig_pred.update_layout(title="")
            apply_plotly_theme(fig_pred)
            st.markdown(f"#### {model_name}: Predicted vs Actual Temperature")
            st.plotly_chart(fig_pred, width='stretch')
    
    # Future Predictions
    st.subheader("🔮 Future Climate Projections")
    
    prediction_years = st.slider("Select prediction horizon (years)", 1, 10, 5)
    
    # Create future predictions based on trends
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                periods=prediction_years * 365, freq='D')
    
    # Use historical trends for projection
    yearly_avg = df.groupby(df.index.year)['Maximum Temperature'].mean()
    years = yearly_avg.index.values
    temps = yearly_avg.values
    
    # Fit trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, temps)
    
    # Project future temperatures
    future_years = future_dates.year
    base_prediction = slope * future_years + intercept
    
    # Add seasonal variation
    seasonal_pattern = df.groupby(df.index.dayofyear)['Maximum Temperature'].mean()
    day_of_year = future_dates.dayofyear
    
    # Handle leap years
    seasonal_adjustment = []
    for doy in day_of_year:
        if doy <= 365:
            seasonal_adjustment.append(seasonal_pattern.iloc[doy-1] - seasonal_pattern.mean())
        else:  # Day 366 (leap year)
            seasonal_adjustment.append(seasonal_pattern.iloc[-1] - seasonal_pattern.mean())
    
    future_predictions = base_prediction + np.array(seasonal_adjustment)
    
    # Add uncertainty bounds
    prediction_std = np.sqrt(std_err**2 * (future_years - years.mean())**2 + df['Maximum Temperature'].std()**2)
    upper_bound = future_predictions + 1.96 * prediction_std
    lower_bound = future_predictions - 1.96 * prediction_std
    
    # Plot future predictions
    fig_future = go.Figure()
    
    # Historical data (last 3 years)
    recent_data = df.tail(3*365)
    fig_future.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Maximum Temperature'],
                                  name='Historical Data', line=dict(color='blue')))
    
    # Future predictions
    fig_future.add_trace(go.Scatter(x=future_dates, y=future_predictions,
                                  name='Future Predictions', line=dict(color='red', dash='dash')))
    
    # Confidence intervals
    fig_future.add_trace(go.Scatter(x=future_dates, y=upper_bound,
                                  fill=None, mode='lines', line_color='rgba(0,0,0,0)',
                                  showlegend=False))
    fig_future.add_trace(go.Scatter(x=future_dates, y=lower_bound,
                                  fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                                  name='95% Confidence Interval',
                                  fillcolor='rgba(255,0,0,0.2)'))
    
    fig_future.update_layout(title="",
                           xaxis_title="Date", yaxis_title="Temperature (°C)",
                           height=600)
    apply_plotly_theme(fig_future)
    st.markdown(f"### Temperature Projections for Next {prediction_years} Years")
    st.plotly_chart(fig_future, width='stretch')
    
    # Climate Impact Assessment
    st.subheader("🌍 Climate Impact Assessment")
    
    future_avg_change = slope * prediction_years
    confidence_level = "High" if abs(r_value) > 0.7 else "Medium" if abs(r_value) > 0.5 else "Low"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'<div class="metric-container"><h4>Projected Change</h4><p class="{"trend-up" if future_avg_change > 0 else "trend-down"}">{future_avg_change:.3f}°C</p><small>Over {prediction_years} years</small></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-container"><h4>Trend Confidence</h4><p>{confidence_level}</p><small>R² = {r_value**2:.3f}</small></div>', unsafe_allow_html=True)
    
    with col3:
        significance = "Statistically Significant" if p_value < 0.05 else "Not Statistically Significant"
        st.markdown(f'<div class="metric-container"><h4>Statistical Significance</h4><p>{significance}</p><small>p-value = {p_value:.4f}</small></div>', unsafe_allow_html=True)

def create_extreme_weather_analysis(df):
    """Enhanced extreme weather analysis with return periods"""
    st.markdown('<div class="section-header">⚡ Advanced Extreme Weather Analysis</div>', unsafe_allow_html=True)
    
    # Define extreme thresholds using statistical methods
    temp_95th = df['Maximum Temperature'].quantile(0.95)
    temp_99th = df['Maximum Temperature'].quantile(0.99)
    temp_5th = df['Maximum Temperature'].quantile(0.05)
    temp_1st = df['Maximum Temperature'].quantile(0.01)
    
    # Calculate extreme events
    extreme_hot_days = df[df['Maximum Temperature'] > temp_95th]
    severe_hot_days = df[df['Maximum Temperature'] > temp_99th]
    extreme_cold_days = df[df['Maximum Temperature'] < temp_5th]
    severe_cold_days = df[df['Maximum Temperature'] < temp_1st]
    
    # Annual extreme event counts
    yearly_extremes = df.groupby('Year').agg({
        'Is_Extreme_Hot': 'sum',
        'Is_Extreme_Cold': 'sum',
        'Maximum Temperature': ['max', 'min'],
        'Temperature_Range': ['max', 'mean']
    }).round(2)
    
    yearly_extremes.columns = ['_'.join(col).strip() for col in yearly_extremes.columns]
    yearly_extremes = yearly_extremes.reset_index()
    
    # Display extreme weather metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        hot_trend = yearly_extremes['Is_Extreme_Hot_sum'].corr(yearly_extremes['Year'])
        trend_class = "trend-up" if hot_trend > 0.3 else "trend-down" if hot_trend < -0.3 else ""
        st.markdown(f'<div class="metric-container"><h4>🔥 Extreme Hot Days</h4><p>{len(extreme_hot_days)} days</p><small>>{temp_95th:.1f}°C (95th percentile)</small><br><small class="{trend_class}">Trend: {hot_trend:.3f}</small></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-container"><h4>🌡️ Record Hot Days</h4><p>{len(severe_hot_days)} days</p><small>>{temp_99th:.1f}°C (99th percentile)</small><br><small>Most Severe Events</small></div>', unsafe_allow_html=True)
    
    with col3:
        cold_trend = yearly_extremes['Is_Extreme_Cold_sum'].corr(yearly_extremes['Year'])
        trend_class = "trend-up" if cold_trend > 0.3 else "trend-down" if cold_trend < -0.3 else ""
        st.markdown(f'<div class="metric-container"><h4>❄️ Extreme Cold Days</h4><p>{len(extreme_cold_days)} days</p><small><{temp_5th:.1f}°C (5th percentile)</small><br><small class="{trend_class}">Trend: {cold_trend:.3f}</small></div>', unsafe_allow_html=True)
    
    with col4:
        max_temp_ever = df['Maximum Temperature'].max()
        min_temp_ever = df['Maximum Temperature'].min()
        st.markdown(f'<div class="metric-container"><h4>🏆 Temperature Records</h4><p>Max: {max_temp_ever:.1f}°C</p><p>Min: {min_temp_ever:.1f}°C</p><small>All-time extremes</small></div>', unsafe_allow_html=True)
    
    # Extreme weather trends visualization
    fig_extreme = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Annual Extreme Hot Days Trend', 'Temperature Extremes by Year',
                       'Extreme Events Calendar', 'Return Period Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Annual extreme hot days trend
    fig_extreme.add_trace(
        go.Scatter(x=yearly_extremes['Year'], y=yearly_extremes['Is_Extreme_Hot_sum'],
                  mode='lines+markers', name='Extreme Hot Days',
                  line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Add trend line
    years = yearly_extremes['Year'].values
    hot_days = yearly_extremes['Is_Extreme_Hot_sum'].values
    z = np.polyfit(years, hot_days, 1)
    trend_line = np.poly1d(z)(years)
    fig_extreme.add_trace(
        go.Scatter(x=years, y=trend_line, name='Trend Line',
                  line=dict(color='darkred', dash='dash')),
        row=1, col=1
    )
    
    # Temperature extremes by year
    fig_extreme.add_trace(
        go.Scatter(x=yearly_extremes['Year'], y=yearly_extremes['Maximum Temperature_max'],
                  mode='markers', name='Annual Max', marker=dict(color='red', size=8)),
        row=1, col=2
    )
    fig_extreme.add_trace(
        go.Scatter(x=yearly_extremes['Year'], y=yearly_extremes['Maximum Temperature_min'],
                  mode='markers', name='Annual Min', marker=dict(color='blue', size=8)),
        row=1, col=2
    )
    
    # Extreme events calendar (sample for recent year)
    recent_year = df.index.year.max()
    recent_data = df[df.index.year == recent_year]
    extreme_days = recent_data[recent_data['Is_Extreme_Hot'] | recent_data['Is_Extreme_Cold']]
    
    if len(extreme_days) > 0:
        colors = ['red' if x else 'blue' for x in extreme_days['Is_Extreme_Hot']]
        fig_extreme.add_trace(
            go.Scatter(x=extreme_days.index.dayofyear, y=extreme_days['Maximum Temperature'],
                      mode='markers', name=f'Extreme Days {recent_year}',
                      marker=dict(color=colors, size=10)),
            row=2, col=1
        )
    
    # Return period analysis (simplified)
    sorted_temps = np.sort(df['Maximum Temperature'].dropna())[::-1]  # Descending order
    n = len(sorted_temps)
    return_periods = n / (np.arange(1, n+1))  # Simplified return period calculation
    
    fig_extreme.add_trace(
        go.Scatter(x=return_periods[:100], y=sorted_temps[:100],  # Top 100 values
                  mode='markers', name='Return Period',
                  marker=dict(color='purple', size=6)),
        row=2, col=2
    )
    
    fig_extreme.update_layout(height=800, title_text="")
    fig_extreme.update_xaxes(title_text="Year", row=1, col=1)
    fig_extreme.update_xaxes(title_text="Year", row=1, col=2)
    fig_extreme.update_xaxes(title_text="Day of Year", row=2, col=1)
    fig_extreme.update_xaxes(title_text="Return Period (years)", type="log", row=2, col=2)
    fig_extreme.update_yaxes(title_text="Number of Days", row=1, col=1)
    fig_extreme.update_yaxes(title_text="Temperature (°C)", row=1, col=2)
    fig_extreme.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig_extreme.update_yaxes(title_text="Temperature (°C)", row=2, col=2)
    
    apply_plotly_theme(fig_extreme)
    st.markdown("### Extreme Weather Analysis Dashboard")
    st.plotly_chart(fig_extreme, width='stretch')
    
    # Climate Risk Assessment
    st.subheader("🚨 Climate Risk Assessment")
    
    # Calculate risk metrics
    hot_days_per_year = yearly_extremes['Is_Extreme_Hot_sum'].mean()
    hot_days_trend = np.corrcoef(yearly_extremes['Year'], yearly_extremes['Is_Extreme_Hot_sum'])[0, 1]
    
    # Risk level determination
    if hot_days_per_year > 20 and hot_days_trend > 0.5:
        risk_level = "🔴 HIGH RISK"
        risk_class = "extreme-alert"
    elif hot_days_per_year > 10 or hot_days_trend > 0.3:
        risk_level = "🟡 MODERATE RISK"
        risk_class = "warning-box"
    else:
        risk_level = "🟢 LOW RISK"
        risk_class = "insight-box"
    
    st.markdown(f"""
    <div class="{risk_class}">
        <h4>Climate Risk Level: {risk_level}</h4>
        <p><strong>Average extreme hot days per year:</strong> {hot_days_per_year:.1f}</p>
        <p><strong>Trend correlation:</strong> {hot_days_trend:.3f}</p>
        <p><strong>Risk factors:</strong></p>
        <ul>
            <li>Increasing frequency of extreme events: {'Yes' if hot_days_trend > 0.3 else 'No'}</li>
            <li>High baseline extreme days: {'Yes' if hot_days_per_year > 15 else 'No'}</li>
            <li>Temperature variability increasing: {'Yes' if yearly_extremes['Temperature_Range_mean'].corr(yearly_extremes['Year']) > 0.3 else 'No'}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Extreme weather summary table
    st.subheader("📊 Extreme Weather Summary")
    
    extreme_summary = pd.DataFrame({
        'Event Type': ['Extreme Hot (>95th)', 'Severe Hot (>99th)', 'Extreme Cold (<5th)', 'Severe Cold (<1st)'],
        'Threshold (°C)': [temp_95th, temp_99th, temp_5th, temp_1st],
        'Total Days': [len(extreme_hot_days), len(severe_hot_days), len(extreme_cold_days), len(severe_cold_days)],
        'Avg per Year': [len(extreme_hot_days)/len(yearly_extremes), len(severe_hot_days)/len(yearly_extremes),
                        len(extreme_cold_days)/len(yearly_extremes), len(severe_cold_days)/len(yearly_extremes)],
        'Last Occurrence': [extreme_hot_days.index[-1].strftime('%Y-%m-%d') if len(extreme_hot_days) > 0 else 'N/A',
                           severe_hot_days.index[-1].strftime('%Y-%m-%d') if len(severe_hot_days) > 0 else 'N/A',
                           extreme_cold_days.index[-1].strftime('%Y-%m-%d') if len(extreme_cold_days) > 0 else 'N/A',
                           severe_cold_days.index[-1].strftime('%Y-%m-%d') if len(severe_cold_days) > 0 else 'N/A']
    }).round(2)
    
    st.dataframe(extreme_summary, width='stretch')

def create_seasonal_climate_patterns(df):
    """Analyze detailed seasonal climate patterns and changes"""
    st.markdown('<div class="section-header">🌱 Seasonal Climate Pattern Analysis</div>', unsafe_allow_html=True)
    
    # Calculate seasonal statistics
    seasonal_stats = df.groupby(['Year', 'Season']).agg({
        'Maximum Temperature': ['mean', 'max', 'min', 'std'],
        'Minimum Temperature': ['mean', 'max', 'min'],
        'Wind Speed': 'mean',
        'Pressure': 'mean',
        'Temperature_Range': 'mean'
    }).round(3)
    
    seasonal_stats.columns = ['_'.join(col).strip() for col in seasonal_stats.columns]
    seasonal_stats = seasonal_stats.reset_index()
    
    # Seasonal trend analysis
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    seasonal_trends = {}
    
    for season in seasons:
        season_data = seasonal_stats[seasonal_stats['Season'] == season]
        if len(season_data) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                season_data['Year'], season_data['Maximum Temperature_mean']
            )
            seasonal_trends[season] = {
                'slope': slope,
                'r_value': r_value,
                'p_value': p_value,
                'total_change': slope * (season_data['Year'].max() - season_data['Year'].min())
            }
    
    # Display seasonal trends
    st.subheader("🌡️ Seasonal Temperature Trends")
    
    cols = st.columns(4)
    season_colors = {'Spring': '#90EE90', 'Summer': '#FFB347', 'Autumn': '#DEB887', 'Winter': '#87CEEB'}
    
    for i, season in enumerate(seasons):
        with cols[i]:
            if season in seasonal_trends:
                trend = seasonal_trends[season]
                significance = "Significant" if trend['p_value'] < 0.05 else "Not Significant"
                trend_class = "trend-up" if trend['slope'] > 0 else "trend-down"
                
                st.markdown(f'<div class="metric-container" style="border-left-color: {season_colors[season]};"><h4>{season}</h4><p class="{trend_class}">{trend["slope"]:.4f}°C/year</p><small>Total: {trend["total_change"]:.2f}°C</small><br><small>{significance}</small><br><small>R² = {trend["r_value"]**2:.3f}</small></div>', unsafe_allow_html=True)
    
    # Seasonal pattern visualization
    fig_seasonal = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Seasonal Temperature Trends Over Time', 'Monthly Temperature Patterns',
                       'Seasonal Variability Changes', 'Temperature Range by Season'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "box"}]]
    )
    
    # Seasonal trends over time
    for season in seasons:
        season_data = seasonal_stats[seasonal_stats['Season'] == season]
        fig_seasonal.add_trace(
            go.Scatter(x=season_data['Year'], y=season_data['Maximum Temperature_mean'],
                      mode='lines+markers', name=f'{season} Avg Temp',
                      line=dict(color=season_colors[season], width=2)),
            row=1, col=1
        )
    
    # Monthly patterns
    monthly_avg = df.groupby('Month')['Maximum Temperature'].mean()
    monthly_std = df.groupby('Month')['Maximum Temperature'].std()
    
    fig_seasonal.add_trace(
        go.Scatter(x=list(range(1, 13)), y=monthly_avg.values,
                  error_y=dict(type='data', array=monthly_std.values),
                  mode='lines+markers', name='Monthly Average',
                  line=dict(color='blue', width=3)),
        row=1, col=2
    )
    
    # Seasonal variability changes
    seasonal_variability = seasonal_stats.groupby(['Year', 'Season'])['Maximum Temperature_std'].mean().reset_index()
    for season in seasons:
        season_var = seasonal_variability[seasonal_variability['Season'] == season]
        fig_seasonal.add_trace(
            go.Scatter(x=season_var['Year'], y=season_var['Maximum Temperature_std'],
                      mode='lines', name=f'{season} Variability',
                      line=dict(color=season_colors[season], width=2)),
            row=2, col=1
        )
    
    # Temperature range by season
    for season in seasons:
        season_data = df[df['Season'] == season]['Temperature_Range']
        fig_seasonal.add_trace(
            go.Box(y=season_data, name=season, boxpoints='outliers',
                  marker_color=season_colors[season]),
            row=2, col=2
        )
    
    fig_seasonal.update_layout(height=800, title_text="")
    apply_plotly_theme(fig_seasonal)
    st.markdown("### Seasonal Climate Analysis")
    st.plotly_chart(fig_seasonal, width='stretch')
    
    # Season shift analysis
    st.subheader("🔄 Seasonal Shift Analysis")
    
    # Calculate when seasons typically start based on temperature patterns
    def find_seasonal_transitions(df):
        monthly_temps = df.groupby('Month')['Maximum Temperature'].mean()
        
        # Find temperature-based season boundaries
        winter_temp = monthly_temps[[12, 1, 2]].mean()
        summer_temp = monthly_temps[[6, 7, 8]].mean()
        threshold = (winter_temp + summer_temp) / 2
        
        transitions = {}
        for month in range(1, 13):
            if monthly_temps[month] > threshold:
                transitions[month] = 'Warm'
            else:
                transitions[month] = 'Cool'
        
        return transitions, threshold
    
    transitions, temp_threshold = find_seasonal_transitions(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
            <h4>🌡️ Temperature-Based Season Definition</h4>
            <p><strong>Threshold Temperature:</strong> {temp_threshold:.1f}°C</p>
            <p><strong>Warm Months:</strong> {', '.join([str(k) for k, v in transitions.items() if v == 'Warm'])}</p>
            <p><strong>Cool Months:</strong> {', '.join([str(k) for k, v in transitions.items() if v == 'Cool'])}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Calculate seasonal length changes over time
        yearly_seasonal = df.groupby(['Year', 'Month'])['Maximum Temperature'].mean().reset_index()
        warm_months_per_year = []
        years = sorted(df['Year'].unique())
        
        for year in years:
            year_data = yearly_seasonal[yearly_seasonal['Year'] == year]
            warm_months = len(year_data[year_data['Maximum Temperature'] > temp_threshold])
            warm_months_per_year.append(warm_months)
        
        warm_season_trend = np.corrcoef(years, warm_months_per_year)[0, 1] if len(years) > 1 else 0
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>📈 Seasonal Length Trends</h4>
            <p><strong>Average warm months per year:</strong> {np.mean(warm_months_per_year):.1f}</p>
            <p><strong>Warm season trend:</strong> {warm_season_trend:.3f}</p>
            <p><strong>Interpretation:</strong> {'Lengthening warm seasons' if warm_season_trend > 0.3 else 'Stable seasonal patterns' if abs(warm_season_trend) < 0.3 else 'Shortening warm seasons'}</p>
        </div>
        """, unsafe_allow_html=True)

def render_report_header(df):
    if df is None:
        return
    start_year = df.index.year.min()
    end_year = df.index.year.max()
    st.markdown(f'<div class="report-header-container"><div class="report-header-left"><div class="report-header-icon">🌍</div><div class="report-header-title"><h1>Comprehensive Climate Report</h1><p>Advanced Climate Trend Analysis & Visualization</p></div></div><div class="report-header-right"><div class="header-info-card blue"><div class="header-info-icon">📅</div><div class="header-info-text"><span class="header-info-label">Analysis Period</span><span class="header-info-val">{start_year} - {end_year}</span></div></div></div></div>', unsafe_allow_html=True)

def render_metadata_footer(df):
    if df is None:
        return
    years_span = df.index.year.max() - df.index.year.min() + 1
    total_points = len(df)
    st.markdown(f"""
    <div class="metadata-footer-row">
        <div class="metadata-footer-card purple">
            <div class="metadata-footer-icon-wrapper">📅</div>
            <div class="metadata-footer-info">
                <span class="metadata-footer-label">Analysis Duration</span>
                <span class="metadata-footer-val">{years_span} Years</span>
                <span class="metadata-footer-sub">{df.index.year.min()} - {df.index.year.max()}</span>
            </div>
        </div>
        <div class="metadata-footer-card blue">
            <div class="metadata-footer-icon-wrapper">📊</div>
            <div class="metadata-footer-info">
                <span class="metadata-footer-label">Total Data Points</span>
                <span class="metadata-footer-val">{total_points:,}+</span>
                <span class="metadata-footer-sub">Daily Observations</span>
            </div>
        </div>
        <div class="metadata-footer-card green">
            <div class="metadata-footer-icon-wrapper">⚙️</div>
            <div class="metadata-footer-info">
                <span class="metadata-footer-label">Analysis Methods</span>
                <span class="metadata-footer-val">Statistical + ML</span>
                <span class="metadata-footer-sub">Advanced Algorithms</span>
            </div>
        </div>
        <div class="metadata-footer-card orange">
            <div class="metadata-footer-icon-wrapper">🛡️</div>
            <div class="metadata-footer-info">
                <span class="metadata-footer-label">Confidence Level</span>
                <span class="metadata-footer-val">95%</span>
                <span class="metadata-footer-sub">Statistical Confidence</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function with enhanced structure"""
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["📈 Climate Trends", "🔮 Prediction Analysis", "⚡ Extreme Weather", "🌱 Seasonal Patterns", "📊 Full Report"]
    )
    
    # Load data
    with st.spinner("Loading climate data..."):
        df = load_and_prepare_data()
    
    if df is None:
        st.error("Failed to load climate data. Please check your data source.")
        return
    
    # Load model data
    model_data = load_trained_model()
    
    # Data overview
    st.sidebar.markdown("### 📋 Data Overview")
    st.sidebar.info(f"""
    **Date Range:** {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}
    
    **Total Records:** {len(df):,}
    
    **Data Quality:** {((df['Maximum Temperature'].notna().sum() / len(df)) * 100):.1f}% complete
    
    **Years Covered:** {df.index.year.nunique()}
    """)
    
    # Render modern header
    render_report_header(df)
    
    # Main analysis based on selection
    if analysis_type == "📈 Climate Trends":
        yearly_stats, max_slope, min_slope, avg_slope = climate_trend_analysis(df)
        create_comprehensive_plots(df, yearly_stats)
        
    elif analysis_type == "🔮 Prediction Analysis":
        create_enhanced_prediction_analysis(df, model_data)
        
    elif analysis_type == "⚡ Extreme Weather":
        create_extreme_weather_analysis(df)
        
    elif analysis_type == "🌱 Seasonal Patterns":
        create_seasonal_climate_patterns(df)
        
    elif analysis_type == "📊 Full Report":
        # Run all analyses
        yearly_stats, max_slope, min_slope, avg_slope = climate_trend_analysis(df)
        create_comprehensive_plots(df, yearly_stats)
        create_enhanced_prediction_analysis(df, model_data)
        create_extreme_weather_analysis(df)
        create_seasonal_climate_patterns(df)
        
        # Executive summary
        st.markdown('<div class="dashboard-section-header">📋 Executive Summary</div>', unsafe_allow_html=True)
        
        # Generate automated insights
        insights = []
        
        if abs(avg_slope) > 0.02:
            trend_direction = "warming" if avg_slope > 0 else "cooling"
            insights.append(f"🌡️ Significant {trend_direction} trend detected at {avg_slope:.4f}°C per year")
        
        extreme_hot_days = len(df[df['Is_Extreme_Hot']])
        if extreme_hot_days > len(df) * 0.1:  # More than 10% extreme days
            insights.append(f"🔥 High frequency of extreme hot days: {extreme_hot_days} days ({(extreme_hot_days/len(df)*100):.1f}%)")
        
        temp_variability = df['Maximum Temperature'].std()
        if temp_variability > 10:
            insights.append(f"📊 High temperature variability detected: {temp_variability:.2f}°C standard deviation")
        
        if model_data and 'model_results' in model_data:
            best_r2 = max([results['R2'] for results in model_data['model_results'].values()])
            if best_r2 > 0.8:
                insights.append(f"🎯 High prediction accuracy achieved: R² = {best_r2:.3f}")
        
        if insights:
            for insight in insights:
                st.markdown(f"""
                <div class="insight-box">
                    <p>{insight}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box">
                <p>📊 Climate data shows normal variability patterns with no significant extreme trends detected.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Render modern metadata footer
    render_metadata_footer(df)
    
    # Footer
    st.markdown("---")
    st.markdown("*Advanced Climate Analysis Dashboard - Powered by Machine Learning and Statistical Analysis*")