import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_data():
    """Generate comprehensive synthetic blood donation data"""
    
    # Blood types distribution (realistic percentages)
    blood_types = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']
    blood_type_probs = [0.374, 0.357, 0.085, 0.034, 0.066, 0.063, 0.015, 0.006]
    
    # Generate donor data
    n_donors = 10000
    donors_data = []
    
    for i in range(n_donors):
        donor_id = f"D{i+1:05d}"
        age = np.random.normal(35, 12)
        age = max(18, min(65, age))  # Constrain age between 18-65
        
        # Gender distribution
        gender = np.random.choice(['M', 'F'], p=[0.52, 0.48])
        
        # Blood type
        blood_type = np.random.choice(blood_types, p=blood_type_probs)
        
        # Location (simplified to regions)
        region = np.random.choice(['North', 'South', 'East', 'West', 'Central'], 
                                p=[0.2, 0.25, 0.2, 0.15, 0.2])
        
        # Registration date (last 2 years)
        reg_date = datetime.now() - timedelta(days=np.random.randint(1, 730))
        
        # Donor characteristics affecting donation likelihood
        health_score = np.random.normal(0.8, 0.15)
        health_score = max(0.3, min(1.0, health_score))
        
        availability_score = np.random.normal(0.7, 0.2)
        availability_score = max(0.1, min(1.0, availability_score))
        
        donors_data.append({
            'donor_id': donor_id,
            'age': round(age),
            'gender': gender,
            'blood_type': blood_type,
            'region': region,
            'registration_date': reg_date,
            'health_score': round(health_score, 2),
            'availability_score': round(availability_score, 2)
        })
    
    donors_df = pd.DataFrame(donors_data)
    
    # Generate donation history
    donations_data = []
    donation_id = 1
    
    for _, donor in donors_df.iterrows():
        # Number of donations based on donor characteristics
        base_donations = max(1, int(np.random.poisson(3)))
        num_donations = int(base_donations * donor['health_score'] * donor['availability_score'])
        
        last_donation = donor['registration_date']
        
        for j in range(num_donations):
            # Time between donations (minimum 56 days)
            days_gap = max(56, np.random.exponential(90))
            donation_date = last_donation + timedelta(days=days_gap)
            
            if donation_date > datetime.now():
                break
                
            # Donation success rate
            success = np.random.choice([True, False], p=[0.95, 0.05])
            
            # Blood component donated
            component = np.random.choice(['Whole Blood', 'Platelets', 'Plasma'], 
                                       p=[0.7, 0.2, 0.1])
            
            donations_data.append({
                'donation_id': f"DON{donation_id:06d}",
                'donor_id': donor['donor_id'],
                'donation_date': donation_date,
                'blood_type': donor['blood_type'],
                'component': component,
                'success': success,
                'region': donor['region']
            })
            
            donation_id += 1
            last_donation = donation_date
    
    donations_df = pd.DataFrame(donations_data)
    
    # Generate blood bank inventory data
    blood_banks = ['BB001', 'BB002', 'BB003', 'BB004', 'BB005']
    regions_map = {'BB001': 'North', 'BB002': 'South', 'BB003': 'East', 
                   'BB004': 'West', 'BB005': 'Central'}
    
    inventory_data = []
    
    # Generate daily inventory for last 365 days
    for days_back in range(365):
        date = datetime.now() - timedelta(days=days_back)
        
        for bb in blood_banks:
            for bt in blood_types:
                for component in ['Whole Blood', 'PRBC', 'FFP', 'Platelets']:
                    # Base inventory with seasonal and weekly patterns
                    base_inventory = 50
                    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * days_back / 365)
                    weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * days_back / 7)
                    
                    # Random variation
                    random_factor = np.random.normal(1, 0.3)
                    
                    inventory = max(0, int(base_inventory * seasonal_factor * 
                                         weekly_factor * random_factor))
                    
                    # Demand (units requested)
                    demand = max(0, int(np.random.poisson(inventory * 0.3)))
                    
                    inventory_data.append({
                        'date': date,
                        'blood_bank_id': bb,
                        'region': regions_map[bb],
                        'blood_type': bt,
                        'component': component,
                        'inventory': inventory,
                        'demand': demand,
                        'expiry_days': np.random.randint(1, 42)  # Days until expiry
                    })
    
    inventory_df = pd.DataFrame(inventory_data)
    
    # Generate hospital request data
    hospitals = ['H001', 'H002', 'H003', 'H004', 'H005', 'H006', 'H007', 'H008']
    hospital_regions = {
        'H001': 'North', 'H002': 'North', 'H003': 'South', 'H004': 'South',
        'H005': 'East', 'H006': 'East', 'H007': 'West', 'H008': 'Central'
    }
    
    requests_data = []
    request_id = 1
    
    for days_back in range(180):  # Last 6 months
        date = datetime.now() - timedelta(days=days_back)
        
        # Number of requests per day varies
        num_requests = np.random.poisson(5)
        
        for _ in range(num_requests):
            hospital = np.random.choice(hospitals)
            blood_type = np.random.choice(blood_types, p=blood_type_probs)
            component = np.random.choice(['Whole Blood', 'PRBC', 'FFP', 'Platelets'],
                                       p=[0.3, 0.4, 0.2, 0.1])
            
            # Urgency levels
            urgency = np.random.choice(['Low', 'Medium', 'High', 'Critical'],
                                     p=[0.3, 0.4, 0.2, 0.1])
            
            # Units requested
            if urgency == 'Critical':
                units = np.random.randint(5, 20)
            elif urgency == 'High':
                units = np.random.randint(3, 10)
            elif urgency == 'Medium':
                units = np.random.randint(2, 6)
            else:
                units = np.random.randint(1, 4)
            
            # Fulfillment status
            fulfillment = np.random.choice(['Fulfilled', 'Partial', 'Pending', 'Cancelled'],
                                         p=[0.7, 0.15, 0.1, 0.05])
            
            requests_data.append({
                'request_id': f"REQ{request_id:06d}",
                'hospital_id': hospital,
                'region': hospital_regions[hospital],
                'request_date': date,
                'blood_type': blood_type,
                'component': component,
                'units_requested': units,
                'urgency': urgency,
                'fulfillment_status': fulfillment
            })
            
            request_id += 1
    
    requests_df = pd.DataFrame(requests_data)
    
    return donors_df, donations_df, inventory_df, requests_df

def save_datasets(donors_df, donations_df, inventory_df, requests_df):
    """Save all datasets to CSV files"""
    donors_df.to_csv('donors_data.csv', index=False)
    donations_df.to_csv('donations_data.csv', index=False)
    inventory_df.to_csv('inventory_data.csv', index=False)
    requests_df.to_csv('requests_data.csv', index=False)
    
    print("Datasets saved successfully!")
    print(f"Donors: {len(donors_df)} records")
    print(f"Donations: {len(donations_df)} records")
    print(f"Inventory: {len(inventory_df)} records")
    print(f"Requests: {len(requests_df)} records")

if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic blood donation data...")
    donors_df, donations_df, inventory_df, requests_df = generate_synthetic_data()
    
    # Save datasets
    save_datasets(donors_df, donations_df, inventory_df, requests_df)
    
    # Display basic statistics
    print("\n=== DATASET OVERVIEW ===")
    print("\nDonors Dataset:")
    print(donors_df.head())
    print(f"\nBlood type distribution:")
    print(donors_df['blood_type'].value_counts().sort_index())
    
    print("\nDonations Dataset:")
    print(donations_df.head())
    
    print("\nInventory Dataset:")
    print(inventory_df.head())
    
    print("\nRequests Dataset:")
    print(requests_df.head())

