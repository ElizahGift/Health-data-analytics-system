import sys
import pandas as pd
from pathlib import Path
from data_processor import HealthDataProcessor
from analytics_engine import HealthAnalyticsEngine
from visualization import HealthVisualizer
import warnings
warnings.filterwarnings('ignore')

class HealthAnalyticsSystem:
    """Main system orchestrator for health analytics"""
    
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
        print(f"Age 95% CI: ({stats['age_confidence_interval_95'][0]:.1f}, {stats['age_confidence_interval_95'][1]:.1f})")
        print(f"Average BMI: {stats['avg_bmi']:.1f}")
        print(f"Hypertension Prevalence: {stats['hypertension_prevalence']:.1f}%")
        print(f"Diabetes Prevalence: {stats['diabetes_prevalence']:.1f}%")
        
        # Statistical tests
        print("\n" + "="*60)
        print("STATISTICAL ANALYSES")
        print("="*60)
        
        # T-test: Blood pressure by gender
        bp_ttest = self.engine.perform_ttest('gender', 'systolic_bp', 'M', 'F')
        print(f"\nT-test Results (Systolic BP: Male vs Female):")
        print(f"  t-statistic: {bp_ttest['t_statistic']:.3f}")
        print(f"  p-value: {bp_ttest['p_value']:.4f}")
        print(f"  {bp_ttest['interpretation']}")
        
        # ANOVA: BMI across BP categories
        if len(self.data['vitals']['bp_category'].unique()) >= 2:
            bmi_anova = self.engine.perform_anova('bp_category', 'bmi')
            print(f"\nANOVA Results (BMI across BP categories):")
            print(f"  F-statistic: {bmi_anova['f_statistic']:.3f}")
            print(f"  p-value: {bmi_anova['p_value']:.4f}")
            print(f"  {bmi_anova['interpretation']}")
        
        # Correlation
        age_bmi_corr = self.engine.calculate_correlation_with_pvalue('age', 'bmi')
        print(f"\nCorrelation (Age vs BMI):")
        print(f"  r = {age_bmi_corr['correlation']:.3f}")
        print(f"  p-value: {age_bmi_corr['p_value']:.4f}")
        print(f"  {age_bmi_corr['interpretation']}")
        
        # Outlier detection
        print("\n" + "="*60)
        print("OUTLIER DETECTION")
        print("="*60)
        bp_outliers = self.engine.detect_outliers_iqr(self.data['vitals'], 'systolic_bp')
        print(f"Found {len(bp_outliers)} outlier systolic BP readings")
        if len(bp_outliers) > 0:
            print(f"  Range considered normal: {bp_outliers['outlier_bound_lower'].iloc[0]:.0f} - {bp_outliers['outlier_bound_upper'].iloc[0]:.0f}")
        
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
        if not clustered_patients.empty:
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
            if len(risk_df[risk_df['risk_level'] == 'High']) > 0:
                patient_id = risk_df[risk_df['risk_level'] == 'High'].iloc[0]['patient_id']
            else:
                patient_id = self.data['patients'].iloc[0]['patient_id']
        
        print("\n" + "="*60)
        print(f"GENERATING HEALTH REPORT FOR PATIENT {patient_id}")
        print("="*60)
        
        risk_df = self.engine.identify_high_risk_patients()
        report = self.visualizer.generate_health_report(patient_id, risk_df)
        print(report)
        
        # Save report to file
        report_path = Path("patient_reports")
        report_path.mkdir(exist_ok=True)
        with open(report_path / f"patient_{patient_id}_report.txt", "w") as f:
            f.write(report)
        print(f"\nReport saved as 'patient_reports/patient_{patient_id}_report.txt'")
    
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
            print("7. Run Advanced Statistical Tests")
            print("8. Load External CSV Data")
            print("9. Export Analysis Results")
            print("10. Exit")
            
            choice = input("\nEnter your choice (1-10): ").strip()
            
            if choice == '1':
                stats = self.engine.calculate_population_stats()
                print("\nPOPULATION HEALTH METRICS:")
                print("-" * 40)
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        print(f"{key}: {value:.2f}")
                    elif isinstance(value, tuple):
                        print(f"{key}: {value[0]:.2f} - {value[1]:.2f}")
                    else:
                        print(f"{key}: {value}")
            
            elif choice == '2':
                risk_df = self.engine.identify_high_risk_patients()
                print("\nHIGH-RISK PATIENTS (Top 10):")
                print("-" * 80)
                print(risk_df[['patient_id', 'age', 'bmi', 'risk_score', 'risk_level']].head(10))
            
            elif choice == '3':
                patient_id = input("Enter patient ID (e.g., P0001): ").strip()
                if patient_id in self.data['patients']['patient_id'].values:
                    patient = self.data['patients'][self.data['patients']['patient_id'] == patient_id].iloc[0]
                    print(f"\nPATIENT {patient_id} INFORMATION:")
                    print("-" * 40)
                    for col in patient.index:
                        print(f"{col}: {patient[col]}")
                    
                    # Show z-scores for labs
                    zscore_labs = self.engine.calculate_zscore_for_labs(patient_id)
                    if not zscore_labs.empty:
                        print("\nLAB Z-SCORES:")
                        for _, lab in zscore_labs.iterrows():
                            status = "ABNORMAL" if lab['abnormal_zscore'] else "normal"
                            print(f"  {lab['test_name']}: z={lab['z_score']:.2f} ({status})")
                else:
                    print(f"Patient {patient_id} not found!")
            
            elif choice == '4':
                patient_id = input("Enter patient ID: ").strip()
                trends = self.engine.analyze_vital_trends(patient_id)
                print(f"\nVITAL SIGNS TRENDS FOR PATIENT {patient_id}:")
                print("-" * 40)
                for key, value in trends.items():
                    if isinstance(value, dict):
                        print(f"{key}: {value['direction']} (p={value.get('p_value', 'N/A'):.4f})" if value.get('p_value') else f"{key}: {value['direction']}")
                    elif isinstance(value, float):
                        print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
            
            elif choice == '5':
                print("\nExporting data to CSV files...")
                export_dir = Path("exported_data")
                export_dir.mkdir(exist_ok=True)
                self.data['patients'].to_csv(export_dir / 'patients.csv', index=False)
                self.data['vitals'].to_csv(export_dir / 'vitals.csv', index=False)
                self.data['labs'].to_csv(export_dir / 'labs.csv', index=False)
                print(f"✓ Data exported to {export_dir}/patients.csv, vitals.csv, labs.csv")
            
            elif choice == '6':
                self.run_analysis()
                self.generate_visualizations()
                self.generate_individual_report()
            
            elif choice == '7':
                print("\nADVANCED STATISTICAL TESTS:")
                print("-" * 40)
                print("1. T-test (Systolic BP: Male vs Female)")
                print("2. ANOVA (BMI across BP categories)")
                print("3. Correlation (Age vs BMI)")
                print("4. Outlier Detection (Systolic BP)")
                subchoice = input("Select test (1-4): ").strip()
                
                if subchoice == '1':
                    result = self.engine.perform_ttest('gender', 'systolic_bp', 'M', 'F')
                    print(f"\n{result['test_name']}")
                    print(f"Male mean: {result['group1_mean']:.2f}")
                    print(f"Female mean: {result['group2_mean']:.2f}")
                    print(f"t = {result['t_statistic']:.3f}, p = {result['p_value']:.4f}")
                    print(f"✓ {result['interpretation']}")
                elif subchoice == '2':
                    result = self.engine.perform_anova('bp_category', 'bmi')
                    print(f"\n{result['test_name']}")
                    print(f"F = {result['f_statistic']:.3f}, p = {result['p_value']:.4f}")
                    print(f"✓ {result['interpretation']}")
                elif subchoice == '3':
                    result = self.engine.calculate_correlation_with_pvalue('age', 'bmi')
                    print(f"\n{result['variables'][0]} vs {result['variables'][1]}")
                    print(f"r = {result['correlation']:.3f}, p = {result['p_value']:.4f}")
                    print(f"✓ {result['interpretation']}")
                elif subchoice == '4':
                    outliers = self.engine.detect_outliers_iqr(self.data['vitals'], 'systolic_bp')
                    print(f"\nFound {len(outliers)} outlier readings")
                    if len(outliers) > 0:
                        print(outliers[['patient_id', 'systolic_bp', 'outlier_type']].head())
                else:
                    print("Invalid choice")
            
            elif choice == '8':
                print("\nLOAD EXTERNAL CSV DATA:")
                print("-" * 40)
                patients_csv = input("Path to patients CSV (or press Enter to skip): ").strip()
                vitals_csv = input("Path to vitals CSV (or press Enter to skip): ").strip()
                labs_csv = input("Path to labs CSV (or press Enter to skip): ").strip()
                self.engine.load_external_data(
                    patients_csv if patients_csv else None,
                    vitals_csv if vitals_csv else None,
                    labs_csv if labs_csv else None
                )
                # Reinitialize visualizer with new data
                self.visualizer = HealthVisualizer(self.data)
            
            elif choice == '9':
                print("\nExporting analysis results...")
                self.engine.export_analysis_results("analysis_outputs")
            
            elif choice == '10':
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
║                                                           ║
║     Features:                                            ║
║     • Population health analytics                        ║
║     • Risk stratification                                ║
║     • Statistical testing (t-test, ANOVA, correlation)   ║
║     • Outlier detection                                  ║
║     • Patient clustering                                 ║
║     • Interactive visualizations                         ║
║     • CSV import/export                                  ║
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
