import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class BloodDonationPredictor:
    def __init__(self):
        self.demand_model = None
        self.donor_availability_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load all datasets"""
        self.donors_df = pd.read_csv('donors_data.csv')
        self.donations_df = pd.read_csv('donations_data.csv')
        self.inventory_df = pd.read_csv('inventory_data.csv')
        self.requests_df = pd.read_csv('requests_data.csv')
        
        # Convert date columns
        self.donations_df['donation_date'] = pd.to_datetime(self.donations_df['donation_date'])
        self.inventory_df['date'] = pd.to_datetime(self.inventory_df['date'])
        self.requests_df['request_date'] = pd.to_datetime(self.requests_df['request_date'])
        self.donors_df['registration_date'] = pd.to_datetime(self.donors_df['registration_date'])
        
        print("Data loaded successfully!")
        
    def prepare_demand_prediction_data(self):
        """Prepare data for demand prediction model"""
        # Aggregate daily demand by region and blood type
        daily_demand = self.requests_df.groupby(['request_date', 'region', 'blood_type']).agg({
            'units_requested': 'sum',
            'request_id': 'count'
        }).reset_index()
        daily_demand.rename(columns={'request_id': 'num_requests'}, inplace=True)
        
        # Add time-based features
        daily_demand['day_of_week'] = daily_demand['request_date'].dt.dayofweek
        daily_demand['month'] = daily_demand['request_date'].dt.month
        daily_demand['day_of_year'] = daily_demand['request_date'].dt.dayofyear
        daily_demand['is_weekend'] = daily_demand['day_of_week'].isin([5, 6]).astype(int)
        
        # Add historical demand features (rolling averages)
        daily_demand = daily_demand.sort_values(['region', 'blood_type', 'request_date'])
        daily_demand['demand_7d_avg'] = daily_demand.groupby(['region', 'blood_type'])['units_requested'].rolling(7, min_periods=1).mean().values
        daily_demand['demand_30d_avg'] = daily_demand.groupby(['region', 'blood_type'])['units_requested'].rolling(30, min_periods=1).mean().values
        
        # Encode categorical variables
        for col in ['region', 'blood_type']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                daily_demand[f'{col}_encoded'] = self.label_encoders[col].fit_transform(daily_demand[col])
            else:
                daily_demand[f'{col}_encoded'] = self.label_encoders[col].transform(daily_demand[col])
        
        return daily_demand
    
    def prepare_donor_availability_data(self):
        """Prepare data for donor availability prediction"""
        # Get last donation date for each donor
        last_donations = self.donations_df.groupby('donor_id')['donation_date'].max().reset_index()
        last_donations.rename(columns={'donation_date': 'last_donation_date'}, inplace=True)
        
        # Merge with donor information
        donor_features = self.donors_df.merge(last_donations, on='donor_id', how='left')
        
        # Calculate days since last donation
        current_date = datetime.now()
        donor_features['days_since_last_donation'] = (current_date - donor_features['last_donation_date']).dt.days
        donor_features['days_since_last_donation'].fillna(365, inplace=True)  # New donors
        
        # Calculate donation frequency
        donation_counts = self.donations_df.groupby('donor_id').size().reset_index(name='total_donations')
        donor_features = donor_features.merge(donation_counts, on='donor_id', how='left')
        donor_features['total_donations'].fillna(0, inplace=True)
        
        # Calculate donation success rate
        success_rates = self.donations_df.groupby('donor_id')['success'].mean().reset_index(name='success_rate')
        donor_features = donor_features.merge(success_rates, on='donor_id', how='left')
        donor_features['success_rate'].fillna(1.0, inplace=True)
        
        # Create target variable (likelihood to donate in next 30 days)
        # Based on donation patterns and eligibility
        donor_features['eligible_to_donate'] = (donor_features['days_since_last_donation'] >= 56).astype(int)
        donor_features['likely_to_donate'] = (
            (donor_features['eligible_to_donate'] == 1) & 
            (donor_features['availability_score'] > 0.5) &
            (donor_features['health_score'] > 0.6)
        ).astype(int)
        
        # Encode categorical variables
        for col in ['gender', 'blood_type', 'region']:
            if f'{col}_donor' not in self.label_encoders:
                self.label_encoders[f'{col}_donor'] = LabelEncoder()
                donor_features[f'{col}_encoded'] = self.label_encoders[f'{col}_donor'].fit_transform(donor_features[col])
            else:
                donor_features[f'{col}_encoded'] = self.label_encoders[f'{col}_donor'].transform(donor_features[col])
        
        return donor_features
    
    def train_demand_prediction_model(self):
        """Train the demand prediction model"""
        print("Training demand prediction model...")
        
        demand_data = self.prepare_demand_prediction_data()
        
        # Features for demand prediction
        feature_cols = ['day_of_week', 'month', 'day_of_year', 'is_weekend', 
                       'demand_7d_avg', 'demand_30d_avg', 'region_encoded', 'blood_type_encoded']
        
        X = demand_data[feature_cols].fillna(0)
        y = demand_data['units_requested']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.demand_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.demand_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.demand_model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Demand Prediction Model Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.demand_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'feature_importance': feature_importance,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def train_donor_availability_model(self):
        """Train the donor availability prediction model"""
        print("\nTraining donor availability model...")
        
        donor_data = self.prepare_donor_availability_data()
        
        # Features for donor availability prediction
        feature_cols = ['age', 'days_since_last_donation', 'total_donations', 
                       'success_rate', 'health_score', 'availability_score',
                       'gender_encoded', 'blood_type_encoded', 'region_encoded']
        
        X = donor_data[feature_cols].fillna(0)
        y = donor_data['likely_to_donate']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest Classifier
        self.donor_availability_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.donor_availability_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.donor_availability_model.predict(X_test)
        y_pred_proba = self.donor_availability_model.predict_proba(X_test)[:, 1]
        
        print(f"Donor Availability Model Performance:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.donor_availability_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': feature_importance,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def predict_demand(self, region, blood_type, date, historical_avg_7d=None, historical_avg_30d=None):
        """Predict demand for specific region, blood type, and date"""
        if self.demand_model is None:
            raise ValueError("Demand model not trained yet!")
        
        # Prepare features
        day_of_week = date.weekday()
        month = date.month
        day_of_year = date.timetuple().tm_yday
        is_weekend = 1 if day_of_week in [5, 6] else 0
        
        # Use provided historical averages or defaults
        demand_7d_avg = historical_avg_7d if historical_avg_7d is not None else 5.0
        demand_30d_avg = historical_avg_30d if historical_avg_30d is not None else 5.0
        
        # Encode categorical variables
        region_encoded = self.label_encoders['region'].transform([region])[0]
        blood_type_encoded = self.label_encoders['blood_type'].transform([blood_type])[0]
        
        features = np.array([[day_of_week, month, day_of_year, is_weekend,
                            demand_7d_avg, demand_30d_avg, region_encoded, blood_type_encoded]])
        
        features_scaled = self.scaler.transform(features)
        prediction = self.demand_model.predict(features_scaled)[0]
        
        return max(0, round(prediction))
    
    def predict_donor_availability(self, donor_data):
        """Predict donor availability probability"""
        if self.donor_availability_model is None:
            raise ValueError("Donor availability model not trained yet!")
        
        # Prepare features (assuming donor_data is a dictionary with required fields)
        features = np.array([[
            donor_data['age'],
            donor_data['days_since_last_donation'],
            donor_data['total_donations'],
            donor_data['success_rate'],
            donor_data['health_score'],
            donor_data['availability_score'],
            donor_data['gender_encoded'],
            donor_data['blood_type_encoded'],
            donor_data['region_encoded']
        ]])
        
        probability = self.donor_availability_model.predict_proba(features)[0, 1]
        return probability
    
    def save_models(self):
        """Save trained models"""
        joblib.dump(self.demand_model, 'demand_model.pkl')
        joblib.dump(self.donor_availability_model, 'donor_availability_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        print("Models saved successfully!")
    
    def load_models(self):
        """Load trained models"""
        self.demand_model = joblib.load('demand_model.pkl')
        self.donor_availability_model = joblib.load('donor_availability_model.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.label_encoders = joblib.load('label_encoders.pkl')
        print("Models loaded successfully!")

def create_visualizations(demand_results, donor_results):
    """Create visualizations for model results"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Demand prediction: Actual vs Predicted
    axes[0, 0].scatter(demand_results['y_test'], demand_results['y_pred'], alpha=0.6)
    axes[0, 0].plot([demand_results['y_test'].min(), demand_results['y_test'].max()], 
                    [demand_results['y_test'].min(), demand_results['y_test'].max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Demand')
    axes[0, 0].set_ylabel('Predicted Demand')
    axes[0, 0].set_title('Demand Prediction: Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Demand prediction: Feature importance
    top_features = demand_results['feature_importance'].head(6)
    axes[0, 1].barh(top_features['feature'], top_features['importance'])
    axes[0, 1].set_xlabel('Importance')
    axes[0, 1].set_title('Demand Model: Feature Importance')
    
    # Donor availability: Confusion matrix
    cm = confusion_matrix(donor_results['y_test'], donor_results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Donor Availability: Confusion Matrix')
    
    # Donor availability: Feature importance
    top_features_donor = donor_results['feature_importance'].head(6)
    axes[1, 1].barh(top_features_donor['feature'], top_features_donor['importance'])
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title('Donor Availability Model: Feature Importance')
    
    plt.tight_layout()
    plt.savefig('ml_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return 'ml_model_results.png'

if __name__ == "__main__":
    # Initialize predictor
    predictor = BloodDonationPredictor()
    
    # Load data
    predictor.load_data()
    
    # Train models
    demand_results = predictor.train_demand_prediction_model()
    donor_results = predictor.train_donor_availability_model()
    
    # Save models
    predictor.save_models()
    
    # Create visualizations
    viz_path = create_visualizations(demand_results, donor_results)
    print(f"\nVisualization saved as: {viz_path}")
    
    # Example predictions
    print("\n=== EXAMPLE PREDICTIONS ===")
    
    # Predict demand for tomorrow
    tomorrow = datetime.now() + timedelta(days=1)
    demand_pred = predictor.predict_demand('North', 'O+', tomorrow, 5.2, 4.8)
    print(f"Predicted demand for O+ blood in North region tomorrow: {demand_pred} units")
    
    # Predict donor availability
    sample_donor = {
        'age': 35,
        'days_since_last_donation': 70,
        'total_donations': 5,
        'success_rate': 1.0,
        'health_score': 0.8,
        'availability_score': 0.7,
        'gender_encoded': 0,  # Male
        'blood_type_encoded': 0,  # O+
        'region_encoded': 0  # North
    }
    
    availability_prob = predictor.predict_donor_availability(sample_donor)
    print(f"Donor availability probability: {availability_prob:.3f}")

