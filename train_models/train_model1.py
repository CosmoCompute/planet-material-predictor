# temp_model_trainer.py
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import pickle
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Get the path to the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the root directory (one level up)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils import db_utils

MODEL_DIR = os.path.join(project_root, 'models')

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

class SARIMAXTemperatureModel:
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.data = None
        self.train_data = None
        self.test_data = None
        self.model_info = {}
        
    def load_and_preprocess_data(self, db_path="Earth.duckdb"):
        """Load and preprocess the data"""
        print("Loading data...")
        df = db_utils.load_db(db_path)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df.set_index('Date', inplace=True)
        
        # Sort by date and handle missing values
        df = df.sort_index()
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        self.data = df
        print(f"Data loaded: {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df
    
    def split_data(self, train_ratio=0.8):
        """Split data into train and test sets"""
        split_idx = int(len(self.data) * train_ratio)
        self.train_data = self.data.iloc[:split_idx]
        self.test_data = self.data.iloc[split_idx:]
        
        print(f"Training data: {len(self.train_data)} records")
        print(f"Testing data: {len(self.test_data)} records")
    
    def find_best_sarimax_params(self):
        """Find best SARIMAX parameters using grid search"""
        print("Searching for optimal SARIMAX parameters...")
        
        y_train = self.train_data['Maximum Temperature']
        
        # Parameter combinations to test
        param_combinations = [
            ((1,1,1), (1,1,1,12)),  # Original parameters
            ((2,1,2), (1,1,1,12)),
            ((1,1,2), (2,1,1,12)),
            ((2,1,1), (1,1,2,12)),
            ((1,0,1), (1,1,1,12)),
            ((2,1,0), (1,1,1,12)),
            ((0,1,1), (1,1,1,12)),
            ((1,1,1), (2,1,1,12)),
            ((1,1,1), (1,1,2,12)),
        ]
        
        best_aic = float('inf')
        best_params = None
        best_model = None
        
        for order, seasonal_order in param_combinations:
            try:
                print(f"Testing SARIMAX{order}x{seasonal_order}...")
                model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
                results = model.fit(disp=False)
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (order, seasonal_order)
                    best_model = results
                    
                print(f"  AIC: {results.aic:.2f}")
                
            except Exception as e:
                print(f"  Failed: {str(e)}")
                continue
        
        if best_model is not None:
            self.fitted_model = best_model
            self.model_info = {
                'order': best_params[0],
                'seasonal_order': best_params[1],
                'aic': best_aic,
                'training_period': (self.train_data.index.min(), self.train_data.index.max()),
                'n_observations': len(self.train_data)
            }
            print(f"\nBest SARIMAX parameters: {best_params}")
            print(f"Best AIC: {best_aic:.2f}")
            return True
        else:
            print("No suitable SARIMAX model found!")
            return False
    
    def train_sarimax_model(self, order=None, seasonal_order=None):
        """Train SARIMAX model with specified or optimal parameters"""
        if order is None or seasonal_order is None:
            # Use grid search to find best parameters
            return self.find_best_sarimax_params()
        else:
            # Use provided parameters
            print(f"Training SARIMAX{order}x{seasonal_order}...")
            y_train = self.train_data['Maximum Temperature']
            
            try:
                model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
                self.fitted_model = model.fit(disp=False)
                
                self.model_info = {
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'aic': self.fitted_model.aic,
                    'training_period': (self.train_data.index.min(), self.train_data.index.max()),
                    'n_observations': len(self.train_data)
                }
                
                print(f"Model trained successfully. AIC: {self.fitted_model.aic:.2f}")
                return True
            except Exception as e:
                print(f"Error training model: {e}")
                return False
    
    def evaluate_model(self):
        """Evaluate the trained model on test data"""
        if self.fitted_model is None:
            print("No trained model available for evaluation!")
            return None
        
        print("Evaluating model on test data...")
        
        try:
            # Get predictions for test period
            forecast = self.fitted_model.forecast(steps=len(self.test_data))
            y_true = self.test_data['Maximum Temperature']
            
            # Calculate metrics
            mae = mean_absolute_error(y_true, forecast)
            mse = mean_squared_error(y_true, forecast)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, forecast)
            
            # Calculate mean absolute percentage error
            mape = np.mean(np.abs((y_true - forecast) / y_true)) * 100
            
            evaluation_results = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'forecast': forecast,
                'actual': y_true,
                'test_period': (self.test_data.index.min(), self.test_data.index.max())
            }
            
            # Add to model info
            self.model_info['evaluation'] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            }
            
            print(f"\nModel Evaluation Results:")
            print(f"MAE (Mean Absolute Error): {mae:.2f}°C")
            print(f"RMSE (Root Mean Square Error): {rmse:.2f}°C")
            print(f"R² Score: {r2:.3f}")
            print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
            
            return evaluation_results
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None
    
    def predict_future(self, days_ahead=30):
        """Predict future temperatures"""
        if self.fitted_model is None:
            print("No trained model available for prediction!")
            return None
        
        print(f"Predicting temperatures for next {days_ahead} days...")
        
        try:
            # Generate future dates
            last_date = self.data.index.max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=days_ahead,
                freq='D'
            )
            
            # Make predictions
            forecast = self.fitted_model.forecast(steps=days_ahead)
            
            # Get confidence intervals
            forecast_ci = self.fitted_model.get_forecast(steps=days_ahead).conf_int()
            
            # Create results DataFrame
            future_predictions = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Max_Temp': forecast,
                'Lower_CI': forecast_ci.iloc[:, 0],
                'Upper_CI': forecast_ci.iloc[:, 1]
            })
            future_predictions.set_index('Date', inplace=True)
            
            print(f"\nFuture Temperature Predictions (Next {days_ahead} days):")
            print(future_predictions.round(2))
            
            return future_predictions
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def get_model_summary(self):
        """Get detailed model summary"""
        if self.fitted_model is None:
            print("No trained model available!")
            return None
        
        print("\nModel Summary:")
        print(self.fitted_model.summary())
        return self.fitted_model.summary()
    
    def save_model(self, filename='temp_sarimax_model.pkl'):
        """Save the complete model object"""
        if self.fitted_model is None:
            print("No trained model to save!")
            return False
        
        model_path = os.path.join(MODEL_DIR, filename)
        
        # Prepare data to save
        model_data = {
            'fitted_model': self.fitted_model,
            'model_info': self.model_info,
            'data_info': {
                'total_records': len(self.data),
                'date_range': (self.data.index.min(), self.data.index.max()),
                'training_records': len(self.train_data),
                'test_records': len(self.test_data)
            },
            'last_date': self.data.index.max(),
            'training_data': self.train_data['Maximum Temperature'],
            'test_data': self.test_data['Maximum Temperature'] if self.test_data is not None else None
        }
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Model saved successfully to: {model_path}")
            print(f"Model parameters: SARIMAX{self.model_info['order']}x{self.model_info['seasonal_order']}")
            print(f"AIC: {self.model_info['aic']:.2f}")
            
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    @staticmethod
    def load_model(filename='temp_sarimax_model.pkl', model_dir=None):
        """Load a saved model"""
        if model_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..'))
            model_dir = os.path.join(project_root, 'models')
        
        model_path = os.path.join(model_dir, filename)
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            print(f"Model loaded successfully from: {model_path}")
            return model_data
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


def main():
    """Main function to train the SARIMAX temperature prediction model"""
    print("=== SARIMAX Temperature Prediction Model Training ===")
    
    # Initialize the model
    sarimax_model = SARIMAXTemperatureModel()
    
    # Load and preprocess data
    sarimax_model.load_and_preprocess_data("Earth.duckdb")
    
    # Split data into train/test
    sarimax_model.split_data(train_ratio=0.8)
    
    # Train the model (will find optimal parameters automatically)
    success = sarimax_model.train_sarimax_model()
    
    if success:
        # Evaluate the model
        evaluation_results = sarimax_model.evaluate_model()
        
        # Show model summary
        sarimax_model.get_model_summary()
        
        # Make future predictions
        future_predictions = sarimax_model.predict_future(days_ahead=30)
        
        # Save the model
        sarimax_model.save_model('temp_sarimax_model.pkl')
        
        print("\n=== Training Completed Successfully! ===")
        print(f"Model saved in: {MODEL_DIR}")
        print("Use the saved model file for plotting and further analysis.")
        
    else:
        print("Model training failed!")


# Utility class for loading and using the saved model
class SARIMAXPredictor:
    """Simple class for making predictions with saved SARIMAX model"""
    
    def __init__(self, model_path=None):
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..'))
            model_path = os.path.join(project_root, 'models', 'temp_sarimax_model.pkl')
        
        self.model_path = model_path
        self.model_data = None
    
    def load_model(self):
        """Load the saved model"""
        self.model_data = SARIMAXTemperatureModel.load_model()
        return self.model_data is not None
    
    def predict(self, days_ahead=7):
        """Make temperature predictions"""
        if self.model_data is None:
            if not self.load_model():
                return None
        
        try:
            fitted_model = self.model_data['fitted_model']
            forecast = fitted_model.forecast(steps=days_ahead)
            
            last_date = self.model_data['last_date']
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=days_ahead,
                freq='D'
            )
            
            predictions_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Max_Temp': forecast
            })
            predictions_df.set_index('Date', inplace=True)
            
            return predictions_df
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def get_model_info(self):
        """Get model information"""
        if self.model_data is None:
            if not self.load_model():
                return None
        
        return self.model_data['model_info']
    
    def get_evaluation_results(self):
        """Get model evaluation results"""
        if self.model_data is None:
            if not self.load_model():
                return None
        
        return self.model_data['model_info'].get('evaluation', None)


if __name__ == "__main__":
    main()


# Example usage for later:
"""
# To use the saved model for predictions:
predictor = SARIMAXPredictor()
future_temps = predictor.predict(days_ahead=14)
model_info = predictor.get_model_info()
evaluation = predictor.get_evaluation_results()

# To load model data for plotting:
model_data = SARIMAXTemperatureModel.load_model('temp_sarimax_model.pkl')
fitted_model = model_data['fitted_model']
model_info = model_data['model_info']
"""
