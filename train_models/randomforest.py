import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class MarsWeatherPredictor:
    """Mars Weather Random Forest Model Trainer and Predictor"""
    
    def __init__(self, data_path='data/cleaned_marsWeather.csv'):
        self.data_path = data_path
        self.models = {}
        self.features = ['sol', 'ls', 'season', 'pressure', 'year', 'month', 'day', 'day_of_year']
        self.target_columns = ['min_temp', 'max_temp']
        # Set model directory to models/ folder
        self.model_dir = Path(__file__).parent.parent / 'models'
        # Create models directory if it doesn't exist
        self.model_dir.mkdir(exist_ok=True)
        self.scaler = None
        
    def load_and_process_data(self):
        """Load and process the Mars weather data"""
        try:
            # Get the absolute path to the data file
            data_path = Path(__file__).parent.parent / self.data_path
            df = pd.read_csv(data_path)
            print(f"✅ Loaded {len(df)} records from {data_path}")
            
            # Convert terrestrial_date to datetime
            df['terrestrial_date'] = pd.to_datetime(df['terrestrial_date'], format='%d-%m-%Y')
            
            # Feature Engineering
            df['year'] = df['terrestrial_date'].dt.year
            df['month'] = df['terrestrial_date'].dt.month
            df['day'] = df['terrestrial_date'].dt.day
            df['day_of_year'] = df['terrestrial_date'].dt.dayofyear
            df['temp_range'] = df['max_temp'] - df['min_temp']
            
            # Remove any rows with missing values in required columns
            # Include temp_range in the columns to keep for metadata calculation
            required_cols = self.features + self.target_columns + ['temp_range']
            df_clean = df[required_cols + ['terrestrial_date']].dropna()
            
            print(f"✅ Processed data: {len(df_clean)} clean records")
            print(f"📊 Features: {self.features}")
            print(f"🎯 Targets: {self.target_columns}")
            
            return df, df_clean
            
        except FileNotFoundError:
            print(f"❌ Error: Could not find '{data_path}'")
            print(f"   Please ensure the file exists at: {data_path.absolute()}")
            return None, None
        except Exception as e:
            print(f"❌ Error processing data: {str(e)}")
            return None, None
    
    def train_models(self, df_clean, test_size=0.2, random_state=42):
        """Train Random Forest models for temperature prediction"""
        print("\n🤖 Training Random Forest models...")
        
        # Prepare features (exclude temp_range from training features)
        X = df_clean[self.features]
        
        # Store training results
        training_results = {
            'models': {},
            'metrics': {},
            'test_data': {},
            'feature_importance': {}
        }
        
        for target in self.target_columns:
            print(f"\n📈 Training model for {target}...")
            
            y = df_clean[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Create and train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            
            # Store results
            training_results['models'][target] = model
            training_results['metrics'][target] = metrics
            training_results['test_data'][target] = {
                'y_test': y_test,
                'y_pred': y_pred,
                'X_test': X_test
            }
            training_results['feature_importance'][target] = {
                'features': self.features,
                'importance': model.feature_importances_.tolist()
            }
            
            # Print results
            print(f"✅ {target} model trained successfully!")
            print(f"   R² Score: {metrics['R2']:.3f}")
            print(f"   RMSE: {metrics['RMSE']:.2f}°C")
            print(f"   MAE: {metrics['MAE']:.2f}°C")
        
        self.models = training_results['models']
        return training_results
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate model performance metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MSE': mse
        }
    
    def save_models(self, training_results, sample_data):
        """Save trained models and metadata"""
        print("\n💾 Saving models and metadata...")
        
        try:
            # Save individual models
            for target, model in training_results['models'].items():
                model_path = self.model_dir / f'mars_weather_{target}_model.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"✅ Saved {target} model to {model_path}")
            
            # Calculate average temp range safely
            if 'temp_range' in sample_data.columns:
                avg_temp_range = float(sample_data['temp_range'].mean())
            else:
                # Calculate temp_range if not available
                avg_temp_range = float((sample_data['max_temp'] - sample_data['min_temp']).mean())
            
            # Prepare metadata with explicit type conversion
            metadata = {
                'features': self.features,
                'target_columns': self.target_columns,
                'metrics': training_results['metrics'],
                'feature_importance': training_results['feature_importance'],
                'model_info': {
                    'algorithm': 'RandomForestRegressor',
                    'n_estimators': 100,
                    'max_depth': 10,
                    'training_date': pd.Timestamp.now().isoformat()
                },
                'data_info': {
                    'dataset_size': int(len(sample_data)),
                    'temp_stats': {
                        'min_temp_range': [float(sample_data['min_temp'].min()), float(sample_data['min_temp'].max())],
                        'max_temp_range': [float(sample_data['max_temp'].min()), float(sample_data['max_temp'].max())],
                        'avg_temp_range': avg_temp_range
                    },
                    'pressure_stats': {
                        'min': float(sample_data['pressure'].min()),
                        'max': float(sample_data['pressure'].max()),
                        'mean': float(sample_data['pressure'].mean())
                    }
                }
            }
            
            # Save metadata
            metadata_path = self.model_dir / 'mars_weather_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=float)  # Use float as default converter
            print(f"✅ Saved metadata to {metadata_path}")
            
            # Save sample data for exploration (ensure temp_range is included)
            sample_path = self.model_dir / 'mars_weather_sample_data.csv'
            sample_data_to_save = sample_data.head(1000).copy()
            
            # Ensure temp_range column exists in the saved data
            if 'temp_range' not in sample_data_to_save.columns:
                sample_data_to_save['temp_range'] = sample_data_to_save['max_temp'] - sample_data_to_save['min_temp']
            
            sample_data_to_save.to_csv(sample_path, index=False)
            print(f"✅ Saved sample data to {sample_path}")
            
            print("\n🎉 All models and metadata saved successfully!")
            print(f"📁 Models directory: {self.model_dir.absolute()}")
            
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def predict_future_temperatures(self, days_ahead=30):
        """Predict temperatures for future days"""
        if not self.models:
            print("❌ No models loaded. Please train or load models first.")
            return None
        
        try:
            # Load sample data to get reference values
            df, _ = self.load_and_process_data()
            if df is None:
                return None
            
            # Get the last date and sol
            last_date = df['terrestrial_date'].max()
            last_sol = df['sol'].max()
            
            # Create future dates
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=days_ahead, 
                freq='D'
            )
            
            # Create future features
            future_data = []
            for i, date in enumerate(future_dates):
                future_sol = last_sol + i + 1
                future_ls = (df['ls'].iloc[-1] + (i + 1)) % 360
                future_season = ((future_ls // 90) + 1) % 12 + 1
                avg_pressure = df['pressure'].mean()
                
                future_data.append({
                    'sol': future_sol,
                    'ls': future_ls,
                    'season': future_season,
                    'pressure': avg_pressure,
                    'year': date.year,
                    'month': date.month,
                    'day': date.day,
                    'day_of_year': date.dayofyear
                })
            
            future_df = pd.DataFrame(future_data)
            
            # Make predictions
            predictions = {}
            for target, model in self.models.items():
                predictions[f'predicted_{target}'] = model.predict(future_df[self.features])
            
            # Create results dataframe
            results = pd.DataFrame({
                'date': future_dates,
                **predictions
            })
            
            return results
            
        except Exception as e:
            print(f"❌ Error making predictions: {str(e)}")
            return None
    
    def load_models(self):
        """Load saved models"""
        try:
            self.models = {}
            for target in self.target_columns:
                model_path = self.model_dir / f'mars_weather_{target}_model.pkl'
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[target] = pickle.load(f)
                    print(f"✅ Loaded {target} model from {model_path}")
                else:
                    print(f"❌ Model file not found: {model_path}")
                    return False
            
            print(f"🎉 Successfully loaded {len(self.models)} models")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            return False
    
    def get_model_summary(self):
        """Get summary of loaded models"""
        if not self.models:
            return None
        
        try:
            metadata_path = self.model_dir / 'mars_weather_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata
            else:
                print(f"❌ Metadata file not found: {metadata_path}")
                return None
                
        except Exception as e:
            print(f"❌ Error loading metadata: {str(e)}")
            return None

def train_and_save_models():
    """Main function to train and save models"""
    print("🚀 Starting Mars Weather Model Training...")
    
    # Initialize predictor
    predictor = MarsWeatherPredictor()
    
    # Load and process data
    df, df_clean = predictor.load_and_process_data()
    if df_clean is None:
        return
    
    # Train models
    training_results = predictor.train_models(df_clean)
    
    # Save models and metadata
    predictor.save_models(training_results, df_clean)
    
    print("\n🎉 Training completed successfully!")
    print(f"📁 Models saved in: {predictor.model_dir}")

if __name__ == "__main__":
    train_and_save_models()