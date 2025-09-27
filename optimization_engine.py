import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import geodesic
import random
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class BloodMatchingOptimizer:
    def __init__(self):
        self.compatibility_matrix = {
            'O-': ['O-', 'O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+'],
            'O+': ['O+', 'A+', 'B+', 'AB+'],
            'A-': ['A-', 'A+', 'AB-', 'AB+'],
            'A+': ['A+', 'AB+'],
            'B-': ['B-', 'B+', 'AB-', 'AB+'],
            'B+': ['B+', 'AB+'],
            'AB-': ['AB-', 'AB+'],
            'AB+': ['AB+']
        }
        
    def load_data(self):
        """Load all necessary data"""
        self.donors_df = pd.read_csv('donors_data.csv')
        self.donations_df = pd.read_csv('donations_data.csv')
        self.inventory_df = pd.read_csv('inventory_data.csv')
        self.requests_df = pd.read_csv('requests_data.csv')
        
        # Convert date columns
        self.donations_df['donation_date'] = pd.to_datetime(self.donations_df['donation_date'])
        self.inventory_df['date'] = pd.to_datetime(self.inventory_df['date'])
        self.requests_df['request_date'] = pd.to_datetime(self.requests_df['request_date'])
        
    def calculate_donor_score(self, donor_id: str, blood_type_needed: str, urgency: str) -> float:
        """Calculate a score for how suitable a donor is for a specific request"""
        donor = self.donors_df[self.donors_df['donor_id'] == donor_id].iloc[0]
        
        # Base compatibility score
        if donor['blood_type'] not in self.get_compatible_donors(blood_type_needed):
            return 0.0
        
        score = 1.0
        
        # Health and availability scores
        score *= donor['health_score']
        score *= donor['availability_score']
        
        # Check eligibility (days since last donation)
        last_donations = self.donations_df[self.donations_df['donor_id'] == donor_id]
        if not last_donations.empty:
            last_donation_date = last_donations['donation_date'].max()
            days_since_last = (datetime.now() - last_donation_date).days
            if days_since_last < 56:  # Not eligible
                return 0.0
            elif days_since_last < 84:  # Recently eligible
                score *= 0.8
        
        # Urgency multiplier
        urgency_multipliers = {'Critical': 1.5, 'High': 1.2, 'Medium': 1.0, 'Low': 0.8}
        score *= urgency_multipliers.get(urgency, 1.0)
        
        # Blood type rarity bonus (rarer types get higher priority)
        rarity_bonus = {'AB+': 1.0, 'AB-': 1.3, 'B+': 1.1, 'B-': 1.2, 
                       'A+': 1.0, 'A-': 1.1, 'O+': 1.0, 'O-': 1.4}
        score *= rarity_bonus.get(donor['blood_type'], 1.0)
        
        return min(score, 2.0)  # Cap at 2.0
    
    def get_compatible_donors(self, blood_type_needed: str) -> List[str]:
        """Get list of compatible donor blood types for a needed blood type"""
        compatible = []
        for donor_type, can_donate_to in self.compatibility_matrix.items():
            if blood_type_needed in can_donate_to:
                compatible.append(donor_type)
        return compatible
    
    def find_optimal_donors(self, blood_type_needed: str, region: str, 
                          urgency: str, units_needed: int) -> List[Dict]:
        """Find optimal donors for a specific blood request"""
        
        # Get compatible donor blood types
        compatible_types = self.get_compatible_donors(blood_type_needed)
        
        # Filter donors by region and blood type compatibility
        eligible_donors = self.donors_df[
            (self.donors_df['region'] == region) & 
            (self.donors_df['blood_type'].isin(compatible_types))
        ].copy()
        
        # Calculate scores for each donor
        donor_scores = []
        for _, donor in eligible_donors.iterrows():
            score = self.calculate_donor_score(donor['donor_id'], blood_type_needed, urgency)
            if score > 0:
                donor_scores.append({
                    'donor_id': donor['donor_id'],
                    'blood_type': donor['blood_type'],
                    'score': score,
                    'region': donor['region'],
                    'age': donor['age'],
                    'gender': donor['gender']
                })
        
        # Sort by score (descending)
        donor_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top donors (typically need more donors than units due to no-shows)
        return donor_scores[:min(units_needed * 2, len(donor_scores))]
    
    def optimize_inventory_distribution(self) -> Dict:
        """Optimize blood inventory distribution across regions"""
        
        # Get current inventory (latest date)
        latest_date = self.inventory_df['date'].max()
        current_inventory = self.inventory_df[self.inventory_df['date'] == latest_date]
        
        # Get recent demand (last 30 days)
        recent_date = datetime.now() - timedelta(days=30)
        recent_requests = self.requests_df[self.requests_df['request_date'] >= recent_date]
        
        # Calculate demand by region and blood type
        demand_summary = recent_requests.groupby(['region', 'blood_type']).agg({
            'units_requested': 'sum',
            'request_id': 'count'
        }).reset_index()
        demand_summary.rename(columns={'request_id': 'num_requests'}, inplace=True)
        
        # Calculate supply by region and blood type
        supply_summary = current_inventory.groupby(['region', 'blood_type']).agg({
            'inventory': 'sum'
        }).reset_index()
        
        # Merge supply and demand
        optimization_data = supply_summary.merge(
            demand_summary, on=['region', 'blood_type'], how='outer'
        ).fillna(0)
        
        # Calculate supply-demand ratio
        optimization_data['supply_demand_ratio'] = np.where(
            optimization_data['units_requested'] > 0,
            optimization_data['inventory'] / optimization_data['units_requested'],
            np.inf
        )
        
        # Identify surpluses and shortages
        surpluses = optimization_data[optimization_data['supply_demand_ratio'] > 2.0]
        shortages = optimization_data[optimization_data['supply_demand_ratio'] < 1.0]
        
        # Generate transfer recommendations
        transfer_recommendations = []
        
        for _, shortage in shortages.iterrows():
            # Find potential surplus regions for the same blood type
            potential_sources = surpluses[
                (surpluses['blood_type'] == shortage['blood_type']) &
                (surpluses['region'] != shortage['region'])
            ]
            
            if not potential_sources.empty:
                # Choose the region with highest surplus
                best_source = potential_sources.loc[potential_sources['supply_demand_ratio'].idxmax()]
                
                # Calculate transfer amount
                shortage_amount = max(0, shortage['units_requested'] - shortage['inventory'])
                available_surplus = max(0, best_source['inventory'] - best_source['units_requested'])
                transfer_amount = min(shortage_amount, available_surplus * 0.5)  # Transfer up to 50% of surplus
                
                if transfer_amount > 0:
                    transfer_recommendations.append({
                        'from_region': best_source['region'],
                        'to_region': shortage['region'],
                        'blood_type': shortage['blood_type'],
                        'transfer_amount': int(transfer_amount),
                        'priority': 'High' if shortage['supply_demand_ratio'] < 0.5 else 'Medium'
                    })
        
        return {
            'optimization_data': optimization_data,
            'transfer_recommendations': transfer_recommendations,
            'surpluses': surpluses,
            'shortages': shortages
        }
    
    def simulate_emergency_response(self, emergency_type: str, affected_region: str) -> Dict:
        """Simulate emergency response scenario"""
        
        # Define emergency scenarios
        emergency_scenarios = {
            'mass_casualty': {
                'O+': 20, 'O-': 15, 'A+': 12, 'A-': 8, 'B+': 8, 'B-': 5, 'AB+': 3, 'AB-': 2
            },
            'natural_disaster': {
                'O+': 25, 'O-': 20, 'A+': 15, 'A-': 10, 'B+': 10, 'B-': 8, 'AB+': 5, 'AB-': 3
            },
            'pandemic_surge': {
                'O+': 15, 'O-': 12, 'A+': 10, 'A-': 8, 'B+': 8, 'B-': 6, 'AB+': 4, 'AB-': 3
            }
        }
        
        if emergency_type not in emergency_scenarios:
            return {'error': 'Unknown emergency type'}
        
        emergency_demand = emergency_scenarios[emergency_type]
        
        # Get current inventory for affected region
        latest_date = self.inventory_df['date'].max()
        region_inventory = self.inventory_df[
            (self.inventory_df['date'] == latest_date) & 
            (self.inventory_df['region'] == affected_region)
        ]
        
        current_stock = region_inventory.groupby('blood_type')['inventory'].sum().to_dict()
        
        # Calculate shortfalls
        shortfalls = {}
        for blood_type, needed in emergency_demand.items():
            available = current_stock.get(blood_type, 0)
            if available < needed:
                shortfalls[blood_type] = needed - available
        
        # Find donors for emergency response
        emergency_donors = {}
        for blood_type, shortage in shortfalls.items():
            donors = self.find_optimal_donors(blood_type, affected_region, 'Critical', shortage)
            emergency_donors[blood_type] = donors
        
        # Calculate response time estimate
        total_donors_needed = sum(len(donors) for donors in emergency_donors.values())
        estimated_response_time = min(4, max(1, total_donors_needed / 10))  # 1-4 hours
        
        return {
            'emergency_type': emergency_type,
            'affected_region': affected_region,
            'emergency_demand': emergency_demand,
            'current_stock': current_stock,
            'shortfalls': shortfalls,
            'emergency_donors': emergency_donors,
            'estimated_response_time_hours': estimated_response_time,
            'total_donors_mobilized': total_donors_needed
        }

def create_optimization_visualizations(optimizer):
    """Create visualizations for optimization results"""
    
    # Get optimization data
    opt_results = optimizer.optimize_inventory_distribution()
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Supply-Demand Ratio by Region
    opt_data = opt_results['optimization_data']
    region_ratios = opt_data.groupby('region')['supply_demand_ratio'].mean()
    
    axes[0, 0].bar(region_ratios.index, region_ratios.values, color='skyblue')
    axes[0, 0].set_title('Average Supply-Demand Ratio by Region')
    axes[0, 0].set_ylabel('Supply/Demand Ratio')
    axes[0, 0].axhline(y=1.0, color='red', linestyle='--', label='Ideal Ratio')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Blood Type Distribution in Shortages
    shortages = opt_results['shortages']
    if not shortages.empty:
        shortage_by_type = shortages.groupby('blood_type')['units_requested'].sum()
        axes[0, 1].pie(shortage_by_type.values, labels=shortage_by_type.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Blood Type Distribution in Shortages')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Shortages Detected', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Blood Type Distribution in Shortages')
    
    # 3. Transfer Recommendations Heatmap
    transfers = opt_results['transfer_recommendations']
    if transfers:
        transfer_df = pd.DataFrame(transfers)
        transfer_matrix = transfer_df.pivot_table(
            values='transfer_amount', 
            index='from_region', 
            columns='to_region', 
            aggfunc='sum', 
            fill_value=0
        )
        sns.heatmap(transfer_matrix, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0])
        axes[1, 0].set_title('Recommended Blood Transfers Between Regions')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Transfers Needed', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Recommended Blood Transfers Between Regions')
    
    # 4. Emergency Response Simulation
    emergency_result = optimizer.simulate_emergency_response('mass_casualty', 'North')
    
    emergency_demand = emergency_result['emergency_demand']
    current_stock = emergency_result['current_stock']
    
    blood_types = list(emergency_demand.keys())
    demand_values = [emergency_demand[bt] for bt in blood_types]
    stock_values = [current_stock.get(bt, 0) for bt in blood_types]
    
    x = np.arange(len(blood_types))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, demand_values, width, label='Emergency Demand', color='red', alpha=0.7)
    axes[1, 1].bar(x + width/2, stock_values, width, label='Current Stock', color='blue', alpha=0.7)
    axes[1, 1].set_xlabel('Blood Type')
    axes[1, 1].set_ylabel('Units')
    axes[1, 1].set_title('Emergency Response: Mass Casualty (North Region)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(blood_types)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return 'optimization_results.png'

if __name__ == "__main__":
    # Initialize optimizer
    optimizer = BloodMatchingOptimizer()
    optimizer.load_data()
    
    print("=== BLOOD MATCHING AND OPTIMIZATION SYSTEM ===\n")
    
    # Example 1: Find optimal donors for a request
    print("1. Finding optimal donors for O+ blood in North region (High urgency, 5 units)")
    optimal_donors = optimizer.find_optimal_donors('O+', 'North', 'High', 5)
    
    print(f"Found {len(optimal_donors)} potential donors:")
    for i, donor in enumerate(optimal_donors[:5]):  # Show top 5
        print(f"  {i+1}. Donor {donor['donor_id']} ({donor['blood_type']}) - Score: {donor['score']:.3f}")
    
    # Example 2: Inventory optimization
    print("\n2. Inventory Distribution Optimization")
    opt_results = optimizer.optimize_inventory_distribution()
    
    print(f"Transfer recommendations: {len(opt_results['transfer_recommendations'])}")
    for transfer in opt_results['transfer_recommendations'][:3]:  # Show top 3
        print(f"  Transfer {transfer['transfer_amount']} units of {transfer['blood_type']} "
              f"from {transfer['from_region']} to {transfer['to_region']} (Priority: {transfer['priority']})")
    
    # Example 3: Emergency response simulation
    print("\n3. Emergency Response Simulation")
    emergency_result = optimizer.simulate_emergency_response('mass_casualty', 'North')
    
    print(f"Emergency Type: {emergency_result['emergency_type']}")
    print(f"Affected Region: {emergency_result['affected_region']}")
    print(f"Estimated Response Time: {emergency_result['estimated_response_time_hours']:.1f} hours")
    print(f"Total Donors to Mobilize: {emergency_result['total_donors_mobilized']}")
    
    if emergency_result['shortfalls']:
        print("Blood shortfalls identified:")
        for blood_type, shortage in emergency_result['shortfalls'].items():
            print(f"  {blood_type}: {shortage} units short")
    else:
        print("No shortfalls - sufficient inventory available!")
    
    # Create visualizations
    print("\n4. Generating optimization visualizations...")
    viz_path = create_optimization_visualizations(optimizer)
    print(f"Visualizations saved as: {viz_path}")

