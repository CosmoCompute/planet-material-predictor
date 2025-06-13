import streamlit as st
import pandas as pd
import json
import pickle
from pathlib import Path
import sys
import os
import warnings


# Add the train_models directory to the Python path
current_dir = Path(__file__).parent.parent
train_models_dir = current_dir / 'train_models'
models_dir = current_dir / 'models'
data_dir = current_dir / 'data'

sys.path.append(str(train_models_dir))

try:
    from train_models.randomforest import MarsWeatherPredictor
except ImportError:
    st.error("❌ Could not import MarsWeatherPredictor. Please check the train_models directory.")

@st.cache_resource  # Changed from st.cache_data to st.cache_resource
def load_mars_models():
    """Load pre-trained Mars weather models and metadata"""
    try:
        predictor = MarsWeatherPredictor()
        
        # Load models
        if not predictor.load_models():
            st.error("❌ Failed to load models. Please train the models first.")
            return None
        
        # Load metadata
        metadata = predictor.get_model_summary()
        if metadata is None:
            st.error("❌ Failed to load model metadata.")
            return None
        
        # Load sample data for exploration
        sample_data_path = models_dir / 'mars_weather_sample_data.csv'
        sample_data = None
        if sample_data_path.exists():
            sample_data = pd.read_csv(sample_data_path)
            # Add temperature range calculation
            if 'temp_range' not in sample_data.columns:
                sample_data['temp_range'] = sample_data['max_temp'] - sample_data['min_temp']
        else:
            st.warning(f"⚠️ Sample data not found at {sample_data_path}")
        
        return {
            'models': predictor.models,
            'metadata': {
                **metadata,
                'sample_data': sample_data
            },
            'predictor': predictor
        }
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

@st.cache_data  # This is fine for serializable data
def load_sample_data():
    """Load sample data separately for caching"""
    try:
        sample_data_path = models_dir / 'mars_weather_sample_data.csv'
        if sample_data_path.exists():
            sample_data = pd.read_csv(sample_data_path)
            # Add temperature range calculation
            if 'temp_range' not in sample_data.columns:
                sample_data['temp_range'] = sample_data['max_temp'] - sample_data['min_temp']
            return sample_data
        return None
    except Exception as e:
        st.error(f"❌ Error loading sample data: {str(e)}")
        return None

@st.cache_data  # This is fine for serializable data
def load_metadata():
    """Load metadata separately for caching"""
    try:
        metadata_path = models_dir / 'mars_weather_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        return None
    except Exception as e:
        st.error(f"❌ Error loading metadata: {str(e)}")
        return None

def predict_future_temperatures(models_data, days_ahead=30):
    """Generate future temperature predictions"""
    try:
        if 'predictor' not in models_data:
            st.error("❌ Predictor not available in models data.")
            return None
        
        predictor = models_data['predictor']
        return predictor.predict_future_temperatures(days_ahead)
        
    except Exception as e:
        st.error(f"❌ Error generating predictions: {str(e)}")
        return None

def get_model_info(models_data):
    """Extract model information for display"""
    try:
        metadata = models_data.get('metadata', {})
        
        return {
            'features': metadata.get('features', []),
            'metrics': metadata.get('metrics', {}),
            'dataset_size': metadata.get('data_info', {}).get('dataset_size', 0),
            'data_info': metadata.get('data_info', {}),
            'model_info': metadata.get('model_info', {})
        }
        
    except Exception as e:
        st.error(f"❌ Error extracting model info: {str(e)}")
        return {}

def check_models_exist():
    """Check if trained models exist"""
    model_files = [
        'mars_weather_min_temp_model.pkl',
        'mars_weather_max_temp_model.pkl',
        'mars_weather_metadata.json'
    ]
    
    missing_files = []
    for filename in model_files:
        filepath = models_dir / filename
        if not filepath.exists():
            missing_files.append(filename)
    
    return len(missing_files) == 0, missing_files

def retrain_models():
    """Retrain the models (can be called from Streamlit)"""
    try:
        # Clear the cache to force reload
        load_mars_models.clear()
        load_metadata.clear()
        load_sample_data.clear()
        
        # Import and run training
        from train_models.randomforest import train_and_save_models
        train_and_save_models()
        return True
    except Exception as e:
        st.error(f"❌ Error retraining models: {str(e)}")
        return False

def get_project_structure():
    """Get the current project structure for debugging"""
    structure = {
        'current_dir': str(current_dir),
        'train_models_dir': str(train_models_dir),
        'models_dir': str(models_dir),
        'data_dir': str(data_dir),
        'train_models_exists': train_models_dir.exists(),
        'models_exists': models_dir.exists(),
        'data_exists': data_dir.exists()
    }
    return structure

# Alternative approach: Load models without caching for the predictor object
def load_mars_models_no_cache():
    """Load models without caching - use when cache causes issues"""
    try:
        predictor = MarsWeatherPredictor()
        
        # Load models
        if not predictor.load_models():
            return None
        
        # Load metadata and sample data separately using cached functions
        metadata = load_metadata()
        sample_data = load_sample_data()
        
        if metadata is None:
            return None
        
        return {
            'models': predictor.models,
            'metadata': {
                **metadata,
                'sample_data': sample_data
            },
            'predictor': predictor
        }
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None