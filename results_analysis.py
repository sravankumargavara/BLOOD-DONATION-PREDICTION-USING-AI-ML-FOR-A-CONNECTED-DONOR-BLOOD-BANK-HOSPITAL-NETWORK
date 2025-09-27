import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ResultsAnalyzer:
    def __init__(self):
        self.load_data()
        
    def load_data(self):
        """Load all datasets and processed results"""
        self.donors_df = pd.read_csv('donors_data.csv')
        self.donations_df = pd.read_csv('donations_data.csv')
        self.inventory_df = pd.read_csv('inventory_data.csv')
        self.requests_df = pd.read_csv('requests_data.csv')
        
        # Convert date columns
        self.donations_df['donation_date'] = pd.to_datetime(self.donations_df['donation_date'])
        self.inventory_df['date'] = pd.to_datetime(self.inventory_df['date'])
        self.requests_df['request_date'] = pd.to_datetime(self.requests_df['request_date'])
        
        print("Data loaded for results analysis")
    
    def analyze_donor_demographics(self):
        """Analyze donor demographics and patterns"""
        print("Analyzing donor demographics...")
        
        # Age distribution analysis
        age_stats = {
            'mean_age': self.donors_df['age'].mean(),
            'median_age': self.donors_df['age'].median(),
            'age_std': self.donors_df['age'].std(),
            'age_range': (self.donors_df['age'].min(), self.donors_df['age'].max())
        }
        
        # Blood type distribution
        blood_type_dist = self.donors_df['blood_type'].value_counts(normalize=True)
        
        # Regional distribution
        regional_dist = self.donors_df['region'].value_counts(normalize=True)
        
        # Gender distribution
        gender_dist = self.donors_df['gender'].value_counts(normalize=True)
        
        # Health and availability scores
        health_stats = {
            'mean_health': self.donors_df['health_score'].mean(),
            'mean_availability': self.donors_df['availability_score'].mean(),
            'high_health_donors': (self.donors_df['health_score'] > 0.8).sum(),
            'high_availability_donors': (self.donors_df['availability_score'] > 0.8).sum()
        }
        
        return {
            'age_stats': age_stats,
            'blood_type_distribution': blood_type_dist,
            'regional_distribution': regional_dist,
            'gender_distribution': gender_dist,
            'health_stats': health_stats
        }
    
    def analyze_donation_patterns(self):
        """Analyze donation patterns and trends"""
        print("Analyzing donation patterns...")
        
        # Donation success rate
        success_rate = self.donations_df['success'].mean()
        
        # Donations by blood type
        donations_by_type = self.donations_df['blood_type'].value_counts()
        
        # Donations by component
        donations_by_component = self.donations_df['component'].value_counts()
        
        # Donations by region
        donations_by_region = self.donations_df['region'].value_counts()
        
        # Monthly donation trends
        self.donations_df['month'] = self.donations_df['donation_date'].dt.month
        monthly_donations = self.donations_df.groupby('month').size()
        
        # Weekly patterns
        self.donations_df['day_of_week'] = self.donations_df['donation_date'].dt.day_name()
        weekly_donations = self.donations_df.groupby('day_of_week').size()
        
        # Donor frequency analysis
        donor_frequency = self.donations_df.groupby('donor_id').size()
        frequency_stats = {
            'mean_donations_per_donor': donor_frequency.mean(),
            'median_donations_per_donor': donor_frequency.median(),
            'max_donations_per_donor': donor_frequency.max(),
            'single_time_donors': (donor_frequency == 1).sum(),
            'repeat_donors': (donor_frequency > 1).sum()
        }
        
        return {
            'success_rate': success_rate,
            'donations_by_type': donations_by_type,
            'donations_by_component': donations_by_component,
            'donations_by_region': donations_by_region,
            'monthly_trends': monthly_donations,
            'weekly_patterns': weekly_donations,
            'frequency_stats': frequency_stats
        }
    
    def analyze_demand_patterns(self):
        """Analyze blood demand patterns"""
        print("Analyzing demand patterns...")
        
        # Total demand by blood type
        demand_by_type = self.requests_df.groupby('blood_type')['units_requested'].sum()
        
        # Demand by urgency
        demand_by_urgency = self.requests_df.groupby('urgency')['units_requested'].sum()
        
        # Demand by region
        demand_by_region = self.requests_df.groupby('region')['units_requested'].sum()
        
        # Fulfillment analysis
        fulfillment_stats = self.requests_df['fulfillment_status'].value_counts(normalize=True)
        
        # Average request size by urgency
        avg_request_size = self.requests_df.groupby('urgency')['units_requested'].mean()
        
        # Monthly demand trends
        self.requests_df['month'] = self.requests_df['request_date'].dt.month
        monthly_demand = self.requests_df.groupby('month')['units_requested'].sum()
        
        # Component demand
        component_demand = self.requests_df.groupby('component')['units_requested'].sum()
        
        return {
            'demand_by_type': demand_by_type,
            'demand_by_urgency': demand_by_urgency,
            'demand_by_region': demand_by_region,
            'fulfillment_stats': fulfillment_stats,
            'avg_request_size': avg_request_size,
            'monthly_demand': monthly_demand,
            'component_demand': component_demand
        }
    
    def analyze_supply_demand_balance(self):
        """Analyze supply-demand balance across the system"""
        print("Analyzing supply-demand balance...")
        
        # Get latest inventory
        latest_date = self.inventory_df['date'].max()
        current_inventory = self.inventory_df[self.inventory_df['date'] == latest_date]
        
        # Total supply by blood type
        supply_by_type = current_inventory.groupby('blood_type')['inventory'].sum()
        
        # Recent demand (last 30 days)
        recent_date = self.requests_df['request_date'].max() - timedelta(days=30)
        recent_demand = self.requests_df[self.requests_df['request_date'] >= recent_date]
        demand_by_type = recent_demand.groupby('blood_type')['units_requested'].sum()
        
        # Calculate supply-demand ratios
        supply_demand_ratio = supply_by_type / demand_by_type.reindex(supply_by_type.index, fill_value=1)
        
        # Regional analysis
        supply_by_region = current_inventory.groupby('region')['inventory'].sum()
        demand_by_region = recent_demand.groupby('region')['units_requested'].sum()
        regional_ratio = supply_by_region / demand_by_region.reindex(supply_by_region.index, fill_value=1)
        
        # Identify critical shortages and surpluses
        critical_shortages = supply_demand_ratio[supply_demand_ratio < 0.5]
        high_surpluses = supply_demand_ratio[supply_demand_ratio > 3.0]
        
        return {
            'supply_by_type': supply_by_type,
            'demand_by_type': demand_by_type,
            'supply_demand_ratio': supply_demand_ratio,
            'regional_supply': supply_by_region,
            'regional_demand': demand_by_region,
            'regional_ratio': regional_ratio,
            'critical_shortages': critical_shortages,
            'high_surpluses': high_surpluses
        }
    
    def create_comprehensive_dashboard(self):
        """Create comprehensive results dashboard"""
        print("Creating comprehensive results dashboard...")
        
        # Set up the plot style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 24))
        
        # Create a grid layout
        gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)
        
        # Get analysis results
        donor_analysis = self.analyze_donor_demographics()
        donation_analysis = self.analyze_donation_patterns()
        demand_analysis = self.analyze_demand_patterns()
        balance_analysis = self.analyze_supply_demand_balance()
        
        # 1. Donor Age Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.donors_df['age'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_title('Donor Age Distribution')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Frequency')
        ax1.axvline(donor_analysis['age_stats']['mean_age'], color='red', linestyle='--', 
                   label=f"Mean: {donor_analysis['age_stats']['mean_age']:.1f}")
        ax1.legend()
        
        # 2. Blood Type Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        blood_types = donor_analysis['blood_type_distribution']
        ax2.pie(blood_types.values, labels=blood_types.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Blood Type Distribution')
        
        # 3. Regional Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        regional_dist = donor_analysis['regional_distribution']
        ax3.bar(regional_dist.index, regional_dist.values, color='lightgreen', alpha=0.7)
        ax3.set_title('Donors by Region')
        ax3.set_ylabel('Proportion')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Health vs Availability Scores
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.scatter(self.donors_df['health_score'], self.donors_df['availability_score'], 
                   alpha=0.6, color='purple')
        ax4.set_xlabel('Health Score')
        ax4.set_ylabel('Availability Score')
        ax4.set_title('Health vs Availability Scores')
        ax4.grid(True, alpha=0.3)
        
        # 5. Monthly Donation Trends
        ax5 = fig.add_subplot(gs[1, 0:2])
        monthly_donations = donation_analysis['monthly_trends']
        ax5.plot(monthly_donations.index, monthly_donations.values, marker='o', linewidth=2, markersize=6)
        ax5.set_title('Monthly Donation Trends')
        ax5.set_xlabel('Month')
        ax5.set_ylabel('Number of Donations')
        ax5.grid(True, alpha=0.3)
        
        # 6. Weekly Donation Patterns
        ax6 = fig.add_subplot(gs[1, 2:4])
        weekly_donations = donation_analysis['weekly_patterns']
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_ordered = weekly_donations.reindex(days_order)
        ax6.bar(range(len(weekly_ordered)), weekly_ordered.values, color='orange', alpha=0.7)
        ax6.set_title('Weekly Donation Patterns')
        ax6.set_xlabel('Day of Week')
        ax6.set_ylabel('Number of Donations')
        ax6.set_xticks(range(len(days_order)))
        ax6.set_xticklabels([day[:3] for day in days_order])
        
        # 7. Donation Success Rate
        ax7 = fig.add_subplot(gs[2, 0])
        success_rate = donation_analysis['success_rate']
        ax7.pie([success_rate, 1-success_rate], labels=['Successful', 'Failed'], 
               autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
        ax7.set_title(f'Donation Success Rate\n({success_rate:.1%})')
        
        # 8. Donations by Component
        ax8 = fig.add_subplot(gs[2, 1])
        component_donations = donation_analysis['donations_by_component']
        ax8.bar(component_donations.index, component_donations.values, color='lightcoral', alpha=0.7)
        ax8.set_title('Donations by Component')
        ax8.set_ylabel('Count')
        ax8.tick_params(axis='x', rotation=45)
        
        # 9. Demand by Urgency
        ax9 = fig.add_subplot(gs[2, 2])
        demand_urgency = demand_analysis['demand_by_urgency']
        colors = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'}
        bar_colors = [colors.get(urgency, 'gray') for urgency in demand_urgency.index]
        ax9.bar(demand_urgency.index, demand_urgency.values, color=bar_colors, alpha=0.7)
        ax9.set_title('Demand by Urgency Level')
        ax9.set_ylabel('Total Units Requested')
        ax9.tick_params(axis='x', rotation=45)
        
        # 10. Fulfillment Status
        ax10 = fig.add_subplot(gs[2, 3])
        fulfillment = demand_analysis['fulfillment_stats']
        ax10.pie(fulfillment.values, labels=fulfillment.index, autopct='%1.1f%%', startangle=90)
        ax10.set_title('Request Fulfillment Status')
        
        # 11. Supply-Demand Balance by Blood Type
        ax11 = fig.add_subplot(gs[3, 0:2])
        supply_demand = balance_analysis['supply_demand_ratio']
        x_pos = np.arange(len(supply_demand))
        bars = ax11.bar(x_pos, supply_demand.values, alpha=0.7)
        
        # Color bars based on ratio
        for i, (bar, ratio) in enumerate(zip(bars, supply_demand.values)):
            if ratio < 1:
                bar.set_color('red')
            elif ratio < 2:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        ax11.axhline(y=1, color='black', linestyle='--', label='Ideal Ratio')
        ax11.set_title('Supply-Demand Ratio by Blood Type')
        ax11.set_xlabel('Blood Type')
        ax11.set_ylabel('Supply/Demand Ratio')
        ax11.set_xticks(x_pos)
        ax11.set_xticklabels(supply_demand.index)
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        
        # 12. Regional Supply-Demand Balance
        ax12 = fig.add_subplot(gs[3, 2:4])
        regional_ratio = balance_analysis['regional_ratio']
        ax12.bar(regional_ratio.index, regional_ratio.values, color='lightblue', alpha=0.7)
        ax12.axhline(y=1, color='red', linestyle='--', label='Ideal Ratio')
        ax12.set_title('Regional Supply-Demand Balance')
        ax12.set_xlabel('Region')
        ax12.set_ylabel('Supply/Demand Ratio')
        ax12.legend()
        ax12.tick_params(axis='x', rotation=45)
        
        # 13. Donor Frequency Distribution
        ax13 = fig.add_subplot(gs[4, 0])
        donor_frequency = self.donations_df.groupby('donor_id').size()
        ax13.hist(donor_frequency, bins=range(1, donor_frequency.max()+2), 
                 color='mediumpurple', alpha=0.7, edgecolor='black')
        ax13.set_title('Donor Frequency Distribution')
        ax13.set_xlabel('Number of Donations')
        ax13.set_ylabel('Number of Donors')
        
        # 14. Average Request Size by Urgency
        ax14 = fig.add_subplot(gs[4, 1])
        avg_request = demand_analysis['avg_request_size']
        ax14.bar(avg_request.index, avg_request.values, color='gold', alpha=0.7)
        ax14.set_title('Average Request Size by Urgency')
        ax14.set_ylabel('Average Units')
        ax14.tick_params(axis='x', rotation=45)
        
        # 15. Monthly Demand Trends
        ax15 = fig.add_subplot(gs[4, 2:4])
        monthly_demand = demand_analysis['monthly_demand']
        ax15.plot(monthly_demand.index, monthly_demand.values, marker='s', 
                 linewidth=2, markersize=6, color='darkred')
        ax15.set_title('Monthly Demand Trends')
        ax15.set_xlabel('Month')
        ax15.set_ylabel('Total Units Requested')
        ax15.grid(True, alpha=0.3)
        
        # 16. System Performance Summary
        ax16 = fig.add_subplot(gs[5, 0:4])
        ax16.axis('off')
        
        # Create performance summary text
        summary_text = f"""
        BLOOD DONATION SYSTEM - PERFORMANCE SUMMARY
        
        DONOR METRICS:
        • Total Registered Donors: {len(self.donors_df):,}
        • Average Age: {donor_analysis['age_stats']['mean_age']:.1f} years
        • High Health Score Donors: {donor_analysis['health_stats']['high_health_donors']:,} ({donor_analysis['health_stats']['high_health_donors']/len(self.donors_df):.1%})
        • High Availability Donors: {donor_analysis['health_stats']['high_availability_donors']:,} ({donor_analysis['health_stats']['high_availability_donors']/len(self.donors_df):.1%})
        
        DONATION METRICS:
        • Total Donations: {len(self.donations_df):,}
        • Success Rate: {donation_analysis['success_rate']:.1%}
        • Average Donations per Donor: {donation_analysis['frequency_stats']['mean_donations_per_donor']:.1f}
        • Repeat Donors: {donation_analysis['frequency_stats']['repeat_donors']:,} ({donation_analysis['frequency_stats']['repeat_donors']/(donation_analysis['frequency_stats']['repeat_donors']+donation_analysis['frequency_stats']['single_time_donors']):.1%})
        
        DEMAND METRICS:
        • Total Requests: {len(self.requests_df):,}
        • Total Units Requested: {self.requests_df['units_requested'].sum():,}
        • Fulfillment Rate: {demand_analysis['fulfillment_stats'].get('Fulfilled', 0):.1%}
        • Critical Requests: {(self.requests_df['urgency'] == 'Critical').sum():,} ({(self.requests_df['urgency'] == 'Critical').mean():.1%})
        
        SYSTEM EFFICIENCY:
        • Blood Types in Shortage: {len(balance_analysis['critical_shortages'])}
        • Blood Types in Surplus: {len(balance_analysis['high_surpluses'])}
        • Overall System Balance: {'Balanced' if len(balance_analysis['critical_shortages']) == 0 else 'Needs Attention'}
        """
        
        ax16.text(0.05, 0.95, summary_text, transform=ax16.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Blood Donation System - Comprehensive Results Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig('comprehensive_results_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return 'comprehensive_results_dashboard.png'
    
    def generate_insights_report(self):
        """Generate key insights from the analysis"""
        print("Generating insights report...")
        
        donor_analysis = self.analyze_donor_demographics()
        donation_analysis = self.analyze_donation_patterns()
        demand_analysis = self.analyze_demand_patterns()
        balance_analysis = self.analyze_supply_demand_balance()
        
        insights = {
            'donor_insights': [
                f"The donor population has a mean age of {donor_analysis['age_stats']['mean_age']:.1f} years, indicating a mature donor base.",
                f"O+ blood type represents {donor_analysis['blood_type_distribution']['O+']:.1%} of donors, making it the most common type.",
                f"{donor_analysis['health_stats']['high_health_donors']} donors ({donor_analysis['health_stats']['high_health_donors']/len(self.donors_df):.1%}) have high health scores (>0.8).",
                f"Regional distribution shows {donor_analysis['regional_distribution'].index[0]} region has the highest donor concentration."
            ],
            'donation_insights': [
                f"Donation success rate is {donation_analysis['success_rate']:.1%}, indicating high reliability.",
                f"Whole Blood donations account for {donation_analysis['donations_by_component']['Whole Blood']/donation_analysis['donations_by_component'].sum():.1%} of all donations.",
                f"Average donor contributes {donation_analysis['frequency_stats']['mean_donations_per_donor']:.1f} donations.",
                f"{donation_analysis['frequency_stats']['repeat_donors']} donors ({donation_analysis['frequency_stats']['repeat_donors']/(donation_analysis['frequency_stats']['repeat_donors']+donation_analysis['frequency_stats']['single_time_donors']):.1%}) are repeat donors."
            ],
            'demand_insights': [
                f"Critical urgency requests represent {(self.requests_df['urgency'] == 'Critical').mean():.1%} of all requests.",
                f"Request fulfillment rate is {demand_analysis['fulfillment_stats'].get('Fulfilled', 0):.1%}.",
                f"Average critical request size is {demand_analysis['avg_request_size'].get('Critical', 0):.1f} units.",
                f"O+ blood type has the highest demand with {demand_analysis['demand_by_type'].get('O+', 0)} total units requested."
            ],
            'balance_insights': [
                f"{len(balance_analysis['critical_shortages'])} blood types are in critical shortage (ratio < 0.5).",
                f"{len(balance_analysis['high_surpluses'])} blood types have high surplus (ratio > 3.0).",
                f"Regional balance shows {balance_analysis['regional_ratio'].idxmax()} region has the best supply-demand ratio.",
                f"Overall system efficiency is {'high' if len(balance_analysis['critical_shortages']) == 0 else 'moderate'} with balanced inventory management."
            ]
        }
        
        return insights

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ResultsAnalyzer()
    
    # Create comprehensive dashboard
    dashboard_path = analyzer.create_comprehensive_dashboard()
    print(f"Comprehensive dashboard saved as: {dashboard_path}")
    
    # Generate insights
    insights = analyzer.generate_insights_report()
    
    print("\n=== KEY INSIGHTS ===")
    for category, insight_list in insights.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for i, insight in enumerate(insight_list, 1):
            print(f"  {i}. {insight}")
    
    # Save insights to file
    with open('system_insights.txt', 'w') as f:
        f.write("BLOOD DONATION SYSTEM - KEY INSIGHTS\n")
        f.write("="*50 + "\n\n")
        for category, insight_list in insights.items():
            f.write(f"{category.upper().replace('_', ' ')}:\n")
            for i, insight in enumerate(insight_list, 1):
                f.write(f"  {i}. {insight}\n")
            f.write("\n")
    
    print("\nInsights saved to: system_insights.txt")

