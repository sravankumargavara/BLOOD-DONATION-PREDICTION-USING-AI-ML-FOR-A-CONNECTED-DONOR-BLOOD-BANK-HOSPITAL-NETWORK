import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import unittest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class SystemTester:
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        
    def load_models_and_data(self):
        """Load trained models and test data"""
        try:
            # Load models
            self.demand_model = joblib.load('demand_model.pkl')
            self.donor_availability_model = joblib.load('donor_availability_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.label_encoders = joblib.load('label_encoders.pkl')
            
            # Load data
            self.donors_df = pd.read_csv('donors_data.csv')
            self.donations_df = pd.read_csv('donations_data.csv')
            self.inventory_df = pd.read_csv('inventory_data.csv')
            self.requests_df = pd.read_csv('requests_data.csv')
            
            print("✓ Models and data loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error loading models/data: {e}")
            return False
    
    def test_data_integrity(self):
        """Test data integrity and consistency"""
        print("\n=== DATA INTEGRITY TESTS ===")
        
        tests = []
        
        # Test 1: Check for missing critical fields
        critical_fields = {
            'donors': ['donor_id', 'blood_type', 'region'],
            'donations': ['donation_id', 'donor_id', 'donation_date'],
            'inventory': ['blood_bank_id', 'blood_type', 'inventory'],
            'requests': ['request_id', 'hospital_id', 'blood_type', 'units_requested']
        }
        
        for dataset_name, fields in critical_fields.items():
            df = getattr(self, f"{dataset_name}_df")
            missing_fields = [field for field in fields if field not in df.columns]
            test_passed = len(missing_fields) == 0
            tests.append({
                'test': f'{dataset_name.title()} - Critical Fields Present',
                'passed': test_passed,
                'details': f'Missing fields: {missing_fields}' if missing_fields else 'All fields present'
            })
        
        # Test 2: Check for data consistency
        # Verify all donations have valid donor IDs
        valid_donor_ids = set(self.donors_df['donor_id'])
        invalid_donations = self.donations_df[~self.donations_df['donor_id'].isin(valid_donor_ids)]
        test_passed = len(invalid_donations) == 0
        tests.append({
            'test': 'Donations - Valid Donor IDs',
            'passed': test_passed,
            'details': f'{len(invalid_donations)} invalid donor references' if not test_passed else 'All donor IDs valid'
        })
        
        # Test 3: Check blood type validity
        valid_blood_types = ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-']
        for dataset_name in ['donors', 'donations', 'inventory', 'requests']:
            df = getattr(self, f"{dataset_name}_df")
            if 'blood_type' in df.columns:
                invalid_types = df[~df['blood_type'].isin(valid_blood_types)]
                test_passed = len(invalid_types) == 0
                tests.append({
                    'test': f'{dataset_name.title()} - Valid Blood Types',
                    'passed': test_passed,
                    'details': f'{len(invalid_types)} invalid blood types' if not test_passed else 'All blood types valid'
                })
        
        # Test 4: Check for negative values where inappropriate
        negative_inventory = self.inventory_df[self.inventory_df['inventory'] < 0]
        test_passed = len(negative_inventory) == 0
        tests.append({
            'test': 'Inventory - Non-negative Values',
            'passed': test_passed,
            'details': f'{len(negative_inventory)} negative inventory values' if not test_passed else 'All inventory values non-negative'
        })
        
        self.test_results['data_integrity'] = tests
        
        # Print results
        for test in tests:
            status = "✓" if test['passed'] else "✗"
            print(f"{status} {test['test']}: {test['details']}")
        
        passed_tests = sum(1 for test in tests if test['passed'])
        print(f"\nData Integrity: {passed_tests}/{len(tests)} tests passed")
        
        return passed_tests / len(tests)
    
    def test_model_performance(self):
        """Test ML model performance and accuracy"""
        print("\n=== MODEL PERFORMANCE TESTS ===")
        
        tests = []
        
        # Test demand prediction model
        try:
            # Create test features
            test_features = np.array([[1, 6, 150, 0, 5.0, 4.5, 0, 0]])  # Monday, June, day 150, not weekend, etc.
            test_features_scaled = self.scaler.transform(test_features)
            prediction = self.demand_model.predict(test_features_scaled)[0]
            
            test_passed = 0 <= prediction <= 100  # Reasonable range for blood demand
            tests.append({
                'test': 'Demand Model - Prediction Range',
                'passed': test_passed,
                'details': f'Prediction: {prediction:.2f} units' if test_passed else f'Out of range: {prediction:.2f}'
            })
        except Exception as e:
            tests.append({
                'test': 'Demand Model - Prediction Range',
                'passed': False,
                'details': f'Error: {str(e)}'
            })
        
        # Test donor availability model
        try:
            test_donor = np.array([[35, 70, 5, 1.0, 0.8, 0.7, 0, 0, 0]])  # Sample donor features
            probability = self.donor_availability_model.predict_proba(test_donor)[0, 1]
            
            test_passed = 0 <= probability <= 1
            tests.append({
                'test': 'Donor Model - Probability Range',
                'passed': test_passed,
                'details': f'Probability: {probability:.3f}' if test_passed else f'Out of range: {probability:.3f}'
            })
        except Exception as e:
            tests.append({
                'test': 'Donor Model - Probability Range',
                'passed': False,
                'details': f'Error: {str(e)}'
            })
        
        # Test model consistency (same input should give same output)
        try:
            pred1 = self.demand_model.predict(test_features_scaled)[0]
            pred2 = self.demand_model.predict(test_features_scaled)[0]
            
            test_passed = abs(pred1 - pred2) < 1e-10
            tests.append({
                'test': 'Demand Model - Consistency',
                'passed': test_passed,
                'details': 'Consistent predictions' if test_passed else f'Inconsistent: {pred1} vs {pred2}'
            })
        except Exception as e:
            tests.append({
                'test': 'Demand Model - Consistency',
                'passed': False,
                'details': f'Error: {str(e)}'
            })
        
        self.test_results['model_performance'] = tests
        
        # Print results
        for test in tests:
            status = "✓" if test['passed'] else "✗"
            print(f"{status} {test['test']}: {test['details']}")
        
        passed_tests = sum(1 for test in tests if test['passed'])
        print(f"\nModel Performance: {passed_tests}/{len(tests)} tests passed")
        
        return passed_tests / len(tests)
    
    def test_system_performance(self):
        """Test system performance metrics"""
        print("\n=== SYSTEM PERFORMANCE TESTS ===")
        
        tests = []
        performance_data = {}
        
        # Test 1: Model prediction speed
        test_features = np.random.rand(100, 8)  # 100 test samples
        test_features_scaled = self.scaler.transform(test_features)
        
        start_time = time.time()
        predictions = self.demand_model.predict(test_features_scaled)
        end_time = time.time()
        
        prediction_time = (end_time - start_time) * 1000  # Convert to milliseconds
        performance_data['demand_prediction_time_ms'] = prediction_time
        
        test_passed = prediction_time < 100  # Should be under 100ms for 100 predictions
        tests.append({
            'test': 'Demand Model - Prediction Speed',
            'passed': test_passed,
            'details': f'{prediction_time:.2f}ms for 100 predictions'
        })
        
        # Test 2: Donor model prediction speed
        test_donor_features = np.random.rand(100, 9)
        
        start_time = time.time()
        probabilities = self.donor_availability_model.predict_proba(test_donor_features)
        end_time = time.time()
        
        donor_prediction_time = (end_time - start_time) * 1000
        performance_data['donor_prediction_time_ms'] = donor_prediction_time
        
        test_passed = donor_prediction_time < 100
        tests.append({
            'test': 'Donor Model - Prediction Speed',
            'passed': test_passed,
            'details': f'{donor_prediction_time:.2f}ms for 100 predictions'
        })
        
        # Test 3: Memory usage (approximate)
        import sys
        model_size = sys.getsizeof(self.demand_model) + sys.getsizeof(self.donor_availability_model)
        performance_data['model_memory_bytes'] = model_size
        
        test_passed = model_size < 50 * 1024 * 1024  # Should be under 50MB
        tests.append({
            'test': 'Models - Memory Usage',
            'passed': test_passed,
            'details': f'{model_size / (1024*1024):.2f}MB'
        })
        
        # Test 4: Data processing speed
        start_time = time.time()
        # Simulate processing 1000 donor records
        for _ in range(1000):
            sample_donor = self.donors_df.sample(1).iloc[0]
            # Simple processing simulation
            _ = sample_donor['age'] * sample_donor['health_score']
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000
        performance_data['data_processing_time_ms'] = processing_time
        
        test_passed = processing_time < 1000  # Should be under 1 second
        tests.append({
            'test': 'Data Processing - Speed',
            'passed': test_passed,
            'details': f'{processing_time:.2f}ms for 1000 records'
        })
        
        self.test_results['system_performance'] = tests
        self.performance_metrics = performance_data
        
        # Print results
        for test in tests:
            status = "✓" if test['passed'] else "✗"
            print(f"{status} {test['test']}: {test['details']}")
        
        passed_tests = sum(1 for test in tests if test['passed'])
        print(f"\nSystem Performance: {passed_tests}/{len(tests)} tests passed")
        
        return passed_tests / len(tests)
    
    def test_edge_cases(self):
        """Test system behavior with edge cases"""
        print("\n=== EDGE CASE TESTS ===")
        
        tests = []
        
        # Test 1: Empty input handling
        try:
            empty_features = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])
            empty_scaled = self.scaler.transform(empty_features)
            prediction = self.demand_model.predict(empty_scaled)[0]
            
            test_passed = not np.isnan(prediction) and not np.isinf(prediction)
            tests.append({
                'test': 'Demand Model - Empty Input',
                'passed': test_passed,
                'details': f'Prediction: {prediction:.2f}' if test_passed else 'Invalid prediction'
            })
        except Exception as e:
            tests.append({
                'test': 'Demand Model - Empty Input',
                'passed': False,
                'details': f'Error: {str(e)}'
            })
        
        # Test 2: Extreme values
        try:
            extreme_features = np.array([[999, 999, 999, 1, 999, 999, 999, 999]])
            extreme_scaled = self.scaler.transform(extreme_features)
            prediction = self.demand_model.predict(extreme_scaled)[0]
            
            test_passed = not np.isnan(prediction) and not np.isinf(prediction)
            tests.append({
                'test': 'Demand Model - Extreme Values',
                'passed': test_passed,
                'details': f'Prediction: {prediction:.2f}' if test_passed else 'Invalid prediction'
            })
        except Exception as e:
            tests.append({
                'test': 'Demand Model - Extreme Values',
                'passed': False,
                'details': f'Error: {str(e)}'
            })
        
        # Test 3: Negative values (where inappropriate)
        try:
            negative_features = np.array([[-1, -1, -1, 0, -1, -1, 0, 0]])
            negative_scaled = self.scaler.transform(negative_features)
            prediction = self.demand_model.predict(negative_scaled)[0]
            
            test_passed = not np.isnan(prediction) and not np.isinf(prediction)
            tests.append({
                'test': 'Demand Model - Negative Values',
                'passed': test_passed,
                'details': f'Prediction: {prediction:.2f}' if test_passed else 'Invalid prediction'
            })
        except Exception as e:
            tests.append({
                'test': 'Demand Model - Negative Values',
                'passed': False,
                'details': f'Error: {str(e)}'
            })
        
        self.test_results['edge_cases'] = tests
        
        # Print results
        for test in tests:
            status = "✓" if test['passed'] else "✗"
            print(f"{status} {test['test']}: {test['details']}")
        
        passed_tests = sum(1 for test in tests if test['passed'])
        print(f"\nEdge Cases: {passed_tests}/{len(tests)} tests passed")
        
        return passed_tests / len(tests)
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n=== COMPREHENSIVE TEST REPORT ===")
        
        # Calculate overall scores
        data_integrity_score = self.test_data_integrity()
        model_performance_score = self.test_model_performance()
        system_performance_score = self.test_system_performance()
        edge_case_score = self.test_edge_cases()
        
        overall_score = (data_integrity_score + model_performance_score + 
                        system_performance_score + edge_case_score) / 4
        
        print(f"\n=== OVERALL TEST RESULTS ===")
        print(f"Data Integrity: {data_integrity_score:.1%}")
        print(f"Model Performance: {model_performance_score:.1%}")
        print(f"System Performance: {system_performance_score:.1%}")
        print(f"Edge Case Handling: {edge_case_score:.1%}")
        print(f"Overall Score: {overall_score:.1%}")
        
        # Performance summary
        print(f"\n=== PERFORMANCE METRICS ===")
        for metric, value in self.performance_metrics.items():
            if 'time_ms' in metric:
                print(f"{metric.replace('_', ' ').title()}: {value:.2f}ms")
            elif 'memory_bytes' in metric:
                print(f"{metric.replace('_', ' ').title()}: {value / (1024*1024):.2f}MB")
        
        return {
            'overall_score': overall_score,
            'category_scores': {
                'data_integrity': data_integrity_score,
                'model_performance': model_performance_score,
                'system_performance': system_performance_score,
                'edge_cases': edge_case_score
            },
            'performance_metrics': self.performance_metrics,
            'detailed_results': self.test_results
        }

def create_testing_visualizations(test_report):
    """Create visualizations for testing results"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Test Category Scores
    categories = list(test_report['category_scores'].keys())
    scores = list(test_report['category_scores'].values())
    
    colors = ['green' if score >= 0.8 else 'orange' if score >= 0.6 else 'red' for score in scores]
    bars = axes[0, 0].bar(categories, scores, color=colors, alpha=0.7)
    axes[0, 0].set_title('Test Category Scores')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add score labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.1%}', ha='center', va='bottom')
    
    # 2. Performance Metrics
    perf_metrics = test_report['performance_metrics']
    time_metrics = {k: v for k, v in perf_metrics.items() if 'time_ms' in k}
    
    if time_metrics:
        metric_names = [k.replace('_time_ms', '').replace('_', ' ').title() for k in time_metrics.keys()]
        metric_values = list(time_metrics.values())
        
        axes[0, 1].bar(metric_names, metric_values, color='skyblue', alpha=0.7)
        axes[0, 1].set_title('Performance Metrics (Response Time)')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Overall Score Gauge
    overall_score = test_report['overall_score']
    
    # Create a simple gauge chart
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    axes[1, 0].plot(theta, r, 'k-', linewidth=3)
    
    # Score indicator
    score_angle = np.pi * (1 - overall_score)
    axes[1, 0].plot([score_angle, score_angle], [0, 1], 'r-', linewidth=4)
    
    axes[1, 0].set_xlim(0, np.pi)
    axes[1, 0].set_ylim(0, 1.2)
    axes[1, 0].set_title(f'Overall Test Score: {overall_score:.1%}')
    axes[1, 0].set_xticks([0, np.pi/2, np.pi])
    axes[1, 0].set_xticklabels(['100%', '50%', '0%'])
    axes[1, 0].set_yticks([])
    
    # 4. Test Results Summary
    all_tests = []
    for category, tests in test_report['detailed_results'].items():
        for test in tests:
            all_tests.append({
                'category': category,
                'test': test['test'],
                'passed': test['passed']
            })
    
    test_df = pd.DataFrame(all_tests)
    if not test_df.empty:
        pass_rate_by_category = test_df.groupby('category')['passed'].mean()
        
        axes[1, 1].bar(pass_rate_by_category.index, pass_rate_by_category.values, 
                      color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Pass Rate by Test Category')
        axes[1, 1].set_ylabel('Pass Rate')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add pass rate labels
        for i, (category, rate) in enumerate(pass_rate_by_category.items()):
            axes[1, 1].text(i, rate + 0.01, f'{rate:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('testing_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return 'testing_results.png'

if __name__ == "__main__":
    # Initialize tester
    tester = SystemTester()
    
    # Load models and data
    if not tester.load_models_and_data():
        print("Failed to load models and data. Exiting.")
        exit(1)
    
    # Run comprehensive tests
    test_report = tester.generate_test_report()
    
    # Create visualizations
    print("\nGenerating test visualizations...")
    viz_path = create_testing_visualizations(test_report)
    print(f"Test visualizations saved as: {viz_path}")
    
    # Save test report
    import json
    with open('test_report.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json_report = json.loads(json.dumps(test_report, default=convert_numpy))
        json.dump(json_report, f, indent=2)
    
    print("Test report saved as: test_report.json")

