import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils.model_utils import (
    load_mars_models, 
    load_mars_models_no_cache, 
    predict_future_temperatures, 
    get_model_info, 
    check_models_exist, 
    retrain_models
)
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('default')
sns.set_palette("Set2")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def create_feature_importance_plot(model, features, title):
    """Create feature importance plot"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    bars = ax.bar(range(len(features)), importance[indices], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_title(f'Feature Importance - {title}', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Features', fontsize=10)
    ax.set_ylabel('Importance', fontsize=10)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([features[i] for i in indices], rotation=45, ha='right', fontsize=9)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{importance[indices[i]]:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def create_prediction_analysis_plot(y_true, y_pred, title, color='blue'):
    """Create comprehensive prediction analysis plot"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    residuals = y_pred - y_true
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Actual vs Predicted
    ax1.scatter(y_true, y_pred, alpha=0.7, color=color, s=30, edgecolors='white', linewidth=0.5)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Temperature (°C)', fontsize=10)
    ax1.set_ylabel('Predicted Temperature (°C)', fontsize=10)
    ax1.set_title(f'Actual vs Predicted\nR² = {r2:.3f}', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    ax2.scatter(y_true, residuals, alpha=0.7, color='orange', s=25)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Actual Temperature (°C)', fontsize=10)
    ax2.set_ylabel('Residuals (°C)', fontsize=10)
    ax2.set_title('Residuals Plot', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Prediction distribution
    ax3.hist(y_pred, bins=15, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Predicted Temperature (°C)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Prediction Distribution', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Residuals distribution
    ax4.hist(residuals, bins=15, alpha=0.7, color='orange', edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Residuals (°C)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Residuals Distribution', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    fig.suptitle(f'{title} - RMSE: {rmse:.2f}°C, MAE: {mae:.2f}°C', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def create_actual_vs_predicted_plot(y_true, y_pred, title, color='blue'):
    """Create a focused actual vs predicted plot"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, color=color, s=40, edgecolors='white', linewidth=0.5)
    
    # Perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Labels and title
    ax.set_xlabel('Actual Temperature (°C)', fontsize=12)
    ax.set_ylabel('Predicted Temperature (°C)', fontsize=12)
    ax.set_title(f'{title}\nR² = {r2:.3f}, RMSE = {rmse:.2f}°C, MAE = {mae:.2f}°C', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig

def create_predictions_over_time_plot(sample_data, models, features, title_prefix):
    """Create predictions over time plot for training data"""
    if sample_data is None or len(sample_data) == 0:
        return None
    
    # Get a subset of data for visualization (last 200 points)
    subset_data = sample_data.tail(200).copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Min Temperature predictions over time
    if 'min_temp' in models and 'min_temp' in subset_data.columns:
        X_subset = subset_data[features]
        y_actual_min = subset_data['min_temp']
        y_pred_min = models['min_temp'].predict(X_subset)
        
        ax1.plot(range(len(y_actual_min)), y_actual_min, 'b-', label='Actual Min Temp', linewidth=2, alpha=0.8)
        ax1.plot(range(len(y_pred_min)), y_pred_min, 'r--', label='Predicted Min Temp', linewidth=2, alpha=0.8)
        ax1.set_title(f'{title_prefix} Min Temperature - Actual vs Predicted (Last 200 records)', 
                     fontsize=11, fontweight='bold')
        ax1.set_ylabel('Temperature (°C)', fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
    
    # Max Temperature predictions over time
    if 'max_temp' in models and 'max_temp' in subset_data.columns:
        X_subset = subset_data[features]
        y_actual_max = subset_data['max_temp']
        y_pred_max = models['max_temp'].predict(X_subset)
        
        ax2.plot(range(len(y_actual_max)), y_actual_max, 'b-', label='Actual Max Temp', linewidth=2, alpha=0.8)
        ax2.plot(range(len(y_pred_max)), y_pred_max, 'r--', label='Predicted Max Temp', linewidth=2, alpha=0.8)
        ax2.set_title(f'{title_prefix} Max Temperature - Actual vs Predicted (Last 200 records)', 
                     fontsize=11, fontweight='bold')
        ax2.set_xlabel('Time Index', fontsize=10)
        ax2.set_ylabel('Temperature (°C)', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_data_exploration_plots(sample_data):
    """Create data exploration visualizations"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Temperature distribution
    ax1.hist(sample_data['min_temp'], bins=20, alpha=0.7, color='#2E86AB', 
             edgecolor='black', linewidth=0.5, label='Min Temp')
    ax1.hist(sample_data['max_temp'], bins=20, alpha=0.7, color='#F24236', 
             edgecolor='black', linewidth=0.5, label='Max Temp')
    ax1.set_title('Temperature Distribution', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Temperature (°C)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Seasonal patterns
    seasonal_stats = sample_data.groupby('season').agg({
        'min_temp': 'mean',
        'max_temp': 'mean'
    }).round(1)
    
    ax2.plot(seasonal_stats.index, seasonal_stats['min_temp'], 'o-', 
             color='#2E86AB', label='Avg Min Temp', linewidth=2.5, markersize=6)
    ax2.plot(seasonal_stats.index, seasonal_stats['max_temp'], 'o-', 
             color='#F24236', label='Avg Max Temp', linewidth=2.5, markersize=6)
    ax2.set_title('Temperature by Season', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Season', fontsize=10)
    ax2.set_ylabel('Temperature (°C)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Pressure distribution
    ax3.hist(sample_data['pressure'], bins=20, alpha=0.8, color='green', 
             edgecolor='black', linewidth=0.5)
    ax3.set_title('Pressure Distribution', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Pressure (Pa)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Temperature range vs pressure
    ax4.scatter(sample_data['pressure'], sample_data['temp_range'], 
                alpha=0.6, color='purple', s=30)
    ax4.set_title('Temperature Range vs Pressure', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Pressure (Pa)', fontsize=10)
    ax4.set_ylabel('Temperature Range (°C)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_future_predictions_plot(future_predictions, days_ahead):
    """Create future predictions plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(future_predictions['date'], future_predictions['predicted_min_temp'], 
            'b-', label='Predicted Min Temp', linewidth=2.5, alpha=0.8)
    ax.plot(future_predictions['date'], future_predictions['predicted_max_temp'], 
            'r-', label='Predicted Max Temp', linewidth=2.5, alpha=0.8)
    
    ax.fill_between(future_predictions['date'], 
                    future_predictions['predicted_min_temp'], 
                    future_predictions['predicted_max_temp'], 
                    alpha=0.2, color='gray', label='Temperature Range')
    
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Temperature (°C)', fontsize=10)
    ax.set_title(f'Mars Temperature Predictions - Next {days_ahead} Days', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    plt.tight_layout()
    return fig

def safe_get_metric(metrics_dict, key):
    """Safely get metric value with fallback"""
    try:
        return metrics_dict[key]
    except (KeyError, TypeError):
        return {'R2': 0.0, 'RMSE': 0.0, 'MAE': 0.0}

def safe_format_number(value, decimals=1):
    """Safely format a number value that might be a string"""
    try:
        if isinstance(value, str):
            value = float(value)
        return f"{value:.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"

def safe_format_range(range_list, decimals=1):
    """Safely format a range that might contain strings"""
    try:
        if len(range_list) >= 2:
            min_val = float(range_list[0])
            max_val = float(range_list[1])
            return f"{min_val:.{decimals}f} to {max_val:.{decimals}f}"
        return "N/A"
    except (ValueError, TypeError, IndexError):
        return "N/A"

def render_mars_weather_page():
    """Main function to render the Mars Weather Predictor page"""
    st.markdown('<h1 class="main-header">🌌 Mars Weather Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### 🚀 Predicting Martian Temperatures using Pre-trained Random Forest Models")

    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #FF6B35;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .sub-header {
            font-size: 1.8rem;
            color: #2E86AB;
            margin: 1rem 0;
            border-bottom: 2px solid #2E86AB;
            padding-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if models exist
    models_exist, missing_files = check_models_exist()
    
    if not models_exist:
        st.error("❌ Pre-trained models not found!")
        st.write("**Missing files:**")
        for file in missing_files:
            st.write(f"- {file}")
        
        st.info("💡 **Solution:** Please train the models first by running the training script.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.code("cd train_models\npython randomforest.py", language="bash")
        
        with col2:
            if st.button("🔄 Retrain Models Now", type="primary"):
                with st.spinner("🤖 Training models... This may take a few minutes."):
                    success = retrain_models()
                    if success:
                        st.success("✅ Models trained successfully! Please refresh the page.")
                        st.rerun()
                    else:
                        st.error("❌ Training failed. Please check the console for errors.")
        
        # Debug information
        with st.expander("🔍 Debug Information"):
            from utils.model_utils import get_project_structure
            structure = get_project_structure()
            st.json(structure)
        
        st.stop()
    
    # Load models with better error handling
    with st.spinner("🔄 Loading pre-trained models..."):
        try:
            models_data = load_mars_models()
        except Exception as e:
            st.warning("⚠️ Cache loading failed, trying alternative method...")
            models_data = load_mars_models_no_cache()
    
    if models_data is None:
        st.error("❌ Failed to load models. Please check the console for errors.")
        st.stop()
    
    model_info = get_model_info(models_data)
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "📈 Model Performance", "🔮 Predictions", "📊 Data Exploration"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">📋 Model Overview</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Model Status", "✅ Ready")
        with col2:
            st.metric("📊 Dataset Size", f"{model_info.get('dataset_size', 0):,}")
        with col3:
            min_metrics = safe_get_metric(model_info.get('metrics', {}), 'min_temp')
            st.metric("❄️ Min Temp R²", f"{min_metrics['R2']:.3f}")
        with col4:
            max_metrics = safe_get_metric(model_info.get('metrics', {}), 'max_temp')
            st.metric("🔥 Max Temp R²", f"{max_metrics['R2']:.3f}")
        
        # Performance summary
        st.markdown('<h2 class="sub-header">🎯 Performance Summary</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        metrics = model_info.get('metrics', {})
        temp_types = ['min_temp', 'max_temp']
        
        for i, temp_type in enumerate(temp_types):
            with col1 if i == 0 else col2:
                emoji = "❄️" if temp_type == "min_temp" else "🔥"
                title = temp_type.replace('_', ' ').title()
                st.markdown(f"#### {emoji} {title} Model")
                
                temp_metrics = safe_get_metric(metrics, temp_type)
                sub_cols = st.columns(3)
                with sub_cols[0]:
                    st.metric("R²", f"{temp_metrics['R2']:.3f}")
                with sub_cols[1]:
                    st.metric("RMSE", f"{temp_metrics['RMSE']:.1f}°C")
                with sub_cols[2]:
                    st.metric("MAE", f"{temp_metrics['MAE']:.1f}°C")
        
        # Features and sample data
        st.markdown("#### 🔍 Model Features")
        features = model_info.get('features', [])
        st.info(f"**Features:** {', '.join(features)}")
        
        st.markdown("#### 📋 Sample Data")
        if 'sample_data' in models_data.get('metadata', {}) and models_data['metadata']['sample_data'] is not None:
            sample_df = models_data['metadata']['sample_data'][['sol', 'season', 'min_temp', 'max_temp', 'pressure']].head(10)
            st.dataframe(sample_df, use_container_width=True)
        else:
            st.warning("⚠️ Sample data not available")
    
    with tab2:
        st.markdown('<h2 class="sub-header">📈 Model Performance Analysis</h2>', unsafe_allow_html=True)
        
        models = models_data.get('models', {})
        metadata = models_data.get('metadata', {})
        features = model_info.get('features', [])
        sample_data = metadata.get('sample_data')
        
        # Feature importance
        st.markdown("#### 🔍 Feature Importance")
        
        if 'feature_importance' in metadata:
            col1, col2 = st.columns(2)
            
            for i, temp_type in enumerate(['min_temp', 'max_temp']):
                title = temp_type.replace('_', ' ').title()
                with col1 if i == 0 else col2:
                    if temp_type in models and temp_type in metadata['feature_importance']:
                        fig = create_feature_importance_plot(models[temp_type], features, title)
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.warning(f"Feature importance data not available for {title}")
        
        # NEW: Actual vs Predicted plots for both models
        st.markdown("#### 🎯 Model Accuracy - Actual vs Predicted")
        
        if sample_data is not None and len(sample_data) > 0:
            col1, col2 = st.columns(2)
            
            # Get test predictions for visualization
            test_size = min(500, len(sample_data))  # Use last 500 points for visualization
            test_data = sample_data.tail(test_size)
            X_test = test_data[features]
            
            with col1:
                if 'min_temp' in models and 'min_temp' in test_data.columns:
                    y_true_min = test_data['min_temp']
                    y_pred_min = models['min_temp'].predict(X_test)
                    fig_min = create_actual_vs_predicted_plot(y_true_min, y_pred_min, 
                                                            "Min Temperature Model", color='#2E86AB')
                    st.pyplot(fig_min, use_container_width=True)
                else:
                    st.warning("Min temperature model or data not available")
            
            with col2:
                if 'max_temp' in models and 'max_temp' in test_data.columns:
                    y_true_max = test_data['max_temp']
                    y_pred_max = models['max_temp'].predict(X_test)
                    fig_max = create_actual_vs_predicted_plot(y_true_max, y_pred_max, 
                                                            "Max Temperature Model", color='#F24236')
                    st.pyplot(fig_max, use_container_width=True)
                else:
                    st.warning("Max temperature model or data not available")
        
        # NEW: Predictions over time
        st.markdown("#### 📈 Model Predictions Over Time")
        
        if sample_data is not None and len(sample_data) > 0 and models:
            fig_time = create_predictions_over_time_plot(sample_data, models, features, "Mars Weather")
            if fig_time:
                st.pyplot(fig_time, use_container_width=True)
            else:
                st.warning("Could not generate predictions over time plot")
        
        # Performance metrics display
        st.markdown("#### 📊 Model Metrics Summary")
        
        metrics = model_info.get('metrics', {})
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            min_metrics = safe_get_metric(metrics, 'min_temp')
            max_metrics = safe_get_metric(metrics, 'max_temp')
            
            with col1:
                st.metric("❄️ Min R²", f"{min_metrics['R2']:.3f}")
            with col2:
                st.metric("❄️ Min RMSE", f"{min_metrics['RMSE']:.1f}°C")
            with col3:
                st.metric("🔥 Max R²", f"{max_metrics['R2']:.3f}")
            with col4:
                st.metric("🔥 Max RMSE", f"{max_metrics['RMSE']:.1f}°C")
        
        # Model information
        st.markdown("#### ℹ️ Model Information")
        model_info_data = model_info.get('model_info', {})
        if model_info_data:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Algorithm:** {model_info_data.get('algorithm', 'N/A')}")
                st.write(f"**Estimators:** {model_info_data.get('n_estimators', 'N/A')}")
            with col2:
                st.write(f"**Max Depth:** {model_info_data.get('max_depth', 'N/A')}")
                training_date = model_info_data.get('training_date', '')
                if training_date:
                    st.write(f"**Training Date:** {training_date[:10]}")
                else:
                    st.write("**Training Date:** N/A")
    
    with tab3:
        st.markdown('<h2 class="sub-header">🚀 Future Predictions</h2>', unsafe_allow_html=True)
        
        # Input controls
        col1, col2 = st.columns([2, 1])
        with col1:
            days_ahead = st.slider("📅 Prediction period (days):", 7, 90, 30, 7)
        with col2:
            st.markdown("### 🎯 Quick Select")
            quick_options = [("1 Week", 7), ("1 Month", 30), ("3 Months", 90)]
            for label, value in quick_options:
                if st.button(label, key=f"quick_{value}"):
                    days_ahead = value
        
        # Generate predictions
        with st.spinner(f"🔮 Generating {days_ahead}-day forecast..."):
            future_predictions = predict_future_temperatures(models_data, days_ahead)
        
        if future_predictions is not None:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📅 Period", f"{days_ahead} days")
            with col2:
                st.metric("🌡️ Min Temp", f"{future_predictions['predicted_min_temp'].min():.1f}°C")
            with col3:
                st.metric("🌡️ Max Temp", f"{future_predictions['predicted_max_temp'].max():.1f}°C")
            with col4:
                avg_range = (future_predictions['predicted_max_temp'] - future_predictions['predicted_min_temp']).mean()
                st.metric("📊 Avg Range", f"{avg_range:.1f}°C")
            
            # Predictions plot
            st.markdown("#### 📈 Temperature Forecast")
            fig = create_future_predictions_plot(future_predictions, days_ahead)
            st.pyplot(fig, use_container_width=True)
            
            # Predictions table and download
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 📅 Detailed Forecast")
                display_df = future_predictions.head(14).copy()
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                display_df = display_df.round(1)
                display_df['daily_range'] = (display_df['predicted_max_temp'] - display_df['predicted_min_temp']).round(1)
                
                st.dataframe(
                    display_df.rename(columns={
                        'date': '📅 Date',
                        'predicted_min_temp': '❄️ Min (°C)',
                        'predicted_max_temp': '🔥 Max (°C)',
                        'daily_range': '📊 Range (°C)'
                    }),
                    use_container_width=True,
                    height=400
                )
            
            with col2:
                st.markdown("#### 📥 Export Data")
                csv = future_predictions.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"mars_forecast_{days_ahead}days.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                st.markdown("#### 📊 Statistics")
                stats = {
                    "Coldest": f"{future_predictions['predicted_min_temp'].min():.1f}°C",
                    "Warmest": f"{future_predictions['predicted_max_temp'].max():.1f}°C",
                    "Avg Min": f"{future_predictions['predicted_min_temp'].mean():.1f}°C",
                    "Avg Max": f"{future_predictions['predicted_max_temp'].mean():.1f}°C"
                }
                for label, value in stats.items():
                    st.write(f"**{label}:** {value}")
        else:
            st.error("❌ Failed to generate predictions. Please check the console for errors.")
    
    with tab4:
        st.markdown('<h2 class="sub-header">📊 Data Exploration</h2>', unsafe_allow_html=True)
        
        if 'sample_data' in models_data.get('metadata', {}) and models_data['metadata']['sample_data'] is not None:
            sample_data = models_data['metadata']['sample_data']
            
            st.markdown("#### 🔍 Dataset Insights")
            fig = create_data_exploration_plots(sample_data)
            st.pyplot(fig, use_container_width=True)
            
            # Additional statistics
            col1, col2, col3 = st.columns(3)
            
            data_info = model_info.get('data_info', {})
            temp_stats = data_info.get('temp_stats', {})
            
            with col1:
                st.markdown("#### 🌡️ Temperature Stats")
                min_range = temp_stats.get('min_temp_range', [0, 0])
                max_range = temp_stats.get('max_temp_range', [0, 0])
                avg_range = temp_stats.get('avg_temp_range', 0)
                
                # Safe formatting to handle string values from JSON
                st.write(f"**Min Range:** {safe_format_range(min_range)}°C")
                st.write(f"**Max Range:** {safe_format_range(max_range)}°C")
                st.write(f"**Avg Daily Range:** {safe_format_number(avg_range)}°C")
            
            with col2:
                st.markdown("#### 📊 Pressure Stats")
                pressure_stats = data_info.get('pressure_stats', {})
                
                # Safe formatting for pressure stats
                min_pressure = safe_format_number(pressure_stats.get('min', 0), 0)
                max_pressure = safe_format_number(pressure_stats.get('max', 0), 0)
                mean_pressure = safe_format_number(pressure_stats.get('mean', 0), 0)
                
                st.write(f"**Range:** {min_pressure} - {max_pressure} Pa")
                st.write(f"**Average:** {mean_pressure} Pa")
                
                # Safe formatting for standard deviation from sample data
                if 'pressure' in sample_data.columns:
                    std_pressure = safe_format_number(sample_data['pressure'].std(), 0)
                    st.write(f"**Std Dev:** {std_pressure} Pa")
                else:
                    st.write("**Std Dev:** N/A Pa")
            
            with col3:
                st.markdown("#### 📅 Dataset Info")
                dataset_size = model_info.get('dataset_size', 0)
                features_count = len(model_info.get('features', []))
                
                # Format dataset size safely
                if isinstance(dataset_size, str):
                    try:
                        dataset_size = int(float(dataset_size))
                    except (ValueError, TypeError):
                        dataset_size = 0
                
                st.write(f"**Total Records:** {dataset_size:,}")
                st.write(f"**Features:** {features_count}")
                
                training_date = model_info.get('model_info', {}).get('training_date', '')
                if training_date:
                    # Extract just the date part if it's a full timestamp
                    date_part = str(training_date)[:10] if training_date else 'N/A'
                    st.write(f"**Model Trained:** {date_part}")
                else:
                    st.write("**Model Trained:** N/A")
        else:
            st.warning("⚠️ Sample data not available for exploration.")
    
    # Footer
    st.markdown("---")
    st.markdown("🌌 **Mars Weather Predictor** - Powered by Pre-trained Random Forest Models")