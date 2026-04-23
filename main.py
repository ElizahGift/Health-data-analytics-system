import sys
import pandas as pd
from data_processor import HealthDataProcessor
from analytics_engine import HealthAnalyticsEngine
from visualization import HealthVisualizer
import warnings
warnings.filterwarnings('ignore')

class HealthAnalyticsSystem:
    def __init__(self):
        self.processor = HealthDataProcessor()
        self.engine = None
        self.visualizer = None
        self.data = None
        
    def initialize_system(self, num_patients: int = 100):
        """Initialize the system with synthetic data"""
        print("\n" + "="*60)
        print("INITIALIZING HEALTH DATA ANALYTICS SYSTEM")
        print("="*60)
        
        # Generate and process data
        self.processor.generate_synthetic_data(num_patients)
        self.data = self.processor.process_data()
        self.data = self.processor.clean_data(self.data)
        
        # Initialize analytics and visualization
        self.engine = HealthAnalyticsEngine(self.data)
        self.visualizer = HealthVisualizer(self.data)
        
        print(f"\nSystem initialized successfully!")
        print(f"Loaded {len(self.data['patients'])} patients")
        print(f"Loaded {len(self.data['vitals'])} vital sign records")
        print(f"Loaded {len(self.data['labs'])} lab results")
        
    def run_analysis(self):
        """Run comprehensive analysis"""
        print("\n" + "="*60)
        print("RUNNING POPULATION HEALTH ANALYSIS")
        print("="*60)
        
        # Population statistics
        stats = self.engine.calculate_population_stats()
        print("\nPOPULATION STATISTICS:")
        print("-" * 30)
        print(f"Total Patients: {stats['total_patients']}")
        print(f"Average Age: {stats['avg_age']:.1f} years")
        print(f"Average BMI: {stats['avg_bmi']:.1f}")
        print(f"Hypertension Prevalence: {stats['hypertension_prevalence']:.1f}%")
        print(f"Diabetes Prevalence: {stats['diabetes_prevalence']:.1f}%")
        
        # High-risk patients
        print("\n" + "="*60)
        print("HIGH-RISK PATIENT IDENTIFICATION")
        print("="*60)
        
        risk_df = self.engine.identify_high_risk_patients()
        high_risk = risk_df[risk_df['risk_level'] == 'High']
        
        print(f"\nFound {len(high_risk)} high-risk patients:")
        print("-" * 50)
        for _, patient in high_risk.head().iterrows():
            print(f"Patient {patient['patient_id']}: Score={patient['risk_score']}, "
                  f"Age={patient['age']}, BMI={patient['bmi']:.1f}")
            print(f"  Risk Factors: {patient['risk_factors'][:100]}...")
        
        # Lab correlations
        print("\n" + "="*60)
        print("LAB RESULT CORRELATIONS")
        print("="*60)
        
        correlations = self.engine.analyze_lab_correlations()
        print("\nCorrelations with Age and BMI:")
        print("-" * 50)
        print(correlations.to_string(index=False))
        
        # Patient clustering
        print("\n" + "="*60)
        print("PATIENT CLUSTERING ANALYSIS")
        print("="*60)
        
        clustered_patients, profiles = self.engine.cluster_patients(n_clusters=4)
        print("\nCluster Profiles:")
        print("-" * 50)
        for _, profile in profiles.iterrows():
            print(f"Cluster {profile['cluster']}:")
            print(f"  Size: {profile['size']} patients")
            print(f"  Avg Age: {profile['avg_age']:.1f}")
            print(f"  Avg BMI: {profile['avg_bmi']:.1f}")
            print(f"  Common Conditions: {profile['common_conditions']}")
            print()
        
    def generate_visualizations(self):
        """Generate and display visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Create population dashboard
        dashboard = self.visualizer.create_population_dashboard()
        dashboard.write_html("population_dashboard.html")
        print("\n✓ Population dashboard saved as 'population_dashboard.html'")
        
        # Sample patient visualization
        sample_patient = self.data['patients'].iloc[0]['patient_id']
        patient_timeline = self.visualizer.plot_vital_signs_timeline(sample_patient)
        patient_timeline.write_html(f"patient_{sample_patient}_timeline.html")
        print(f"✓ Patient timeline saved as 'patient_{sample_patient}_timeline.html'")
        
        # Risk matrix
        risk_df = self.engine.identify_high_risk_patients()
        risk_matrix = self.visualizer.create_risk_matrix(risk_df)
        risk_matrix.write_html("risk_matrix.html")
        print("✓ Risk matrix saved as 'risk_matrix.html'")
        
        # Generate static plots
        print("\nGenerating static matplotlib plots...")
        self.visualizer.plot_lab_value_distributions()
        
    def generate_individual_report(self, patient_id: str = None):
        """Generate report for a specific patient"""
        if patient_id is None:
            # Pick a high-risk patient for demonstration
            risk_df = self.engine.identify_high_risk_patients()
            patient_id = risk_df[risk_df['risk_level'] == 'High'].iloc[0]['patient_id']
        
        print("\n" + "="*60)
        print(f"GENERATING HEALTH REPORT FOR PATIENT {patient_id}")
        print("="*60)
        
        report = self.visualizer.generate_health_report(patient_id)
        print(report)
        
        # Save report to file
        with open(f"patient_{patient_id}_report.txt", "w") as f:
            f.write(report)
        print(f"\nReport saved as 'patient_{patient_id}_report.txt'")
        
    def interactive_menu(self):
        """Provide interactive menu for exploring data"""
        while True:
            print("\n" + "="*60)
            print("HEALTH ANALYTICS SYSTEM - INTERACTIVE MENU")
            print("="*60)
            print("1. View Population Statistics")
            print("2. List High-Risk Patients")
            print("3. Search Patient by ID")
            print("4. View Vital Signs Trends")
            print("5. Export Data to CSV")
            print("6. Generate All Reports")
            print("7. Exit")
            
            choice = input("\nEnter your choice (1-7): ")
            
            if choice == '1':
                stats = self.engine.calculate_population_stats()
                print("\nPOPULATION HEALTH METRICS:")
                print("-" * 40)
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
            
            elif choice == '2':
                risk_df = self.engine.identify_high_risk_patients()
                print("\nHIGH-RISK PATIENTS (Top 10):")
                print("-" * 80)
                print(risk_df[['patient_id', 'age', 'bmi', 'risk_score', 'risk_level']].head(10))
            
            elif choice == '3':
                patient_id = input("Enter patient ID (e.g., P0001): ")
                if patient_id in self.data['patients']['patient_id'].values:
                    patient = self.data['patients'][self.data['patients']['patient_id'] == patient_id].iloc[0]
                    print(f"\nPATIENT {patient_id} INFORMATION:")
                    print("-" * 40)
                    for col in patient.index:
                        print(f"{col}: {patient[col]}")
                else:
                    print(f"Patient {patient_id} not found!")
            
            elif choice == '4':
                patient_id = input("Enter patient ID: ")
                trends = self.engine.analyze_vital_trends(patient_id)
                print(f"\nVITAL SIGNS TRENDS FOR PATIENT {patient_id}:")
                print("-" * 40)
                for key, value in trends.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
            
            elif choice == '5':
                print("\nExporting data to CSV files...")
                self.data['patients'].to_csv('patients.csv', index=False)
                self.data['vitals'].to_csv('vitals.csv', index=False)
                self.data['labs'].to_csv('labs.csv', index=False)
                print("✓ Data exported to patients.csv, vitals.csv, labs.csv")
            
            elif choice == '6':
                self.run_analysis()
                self.generate_visualizations()
                self.generate_individual_report()
            
            elif choice == '7':
                print("\nExiting Health Analytics System. Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")

def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════╗
║     HEALTH DATA ANALYTICS SYSTEM - DEMONSTRATION        ║
║     A Comprehensive Healthcare Analytics Platform         ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize system
    system = HealthAnalyticsSystem()
    system.initialize_system(num_patients=100)
    
    # Run analysis
    system.run_analysis()
    
    # Generate visualizations
    system.generate_visualizations()
    
    # Generate sample report
    system.generate_individual_report()
    
    # Interactive menu
    system.interactive_menu()

if __name__ == "__main__":
    main()
