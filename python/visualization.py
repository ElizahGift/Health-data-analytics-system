import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict
import numpy as np

class HealthVisualizer:
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self.patients_df = data['patients']
        self.vitals_df = data['vitals']
        self.labs_df = data['labs']
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def create_population_dashboard(self) -> go.Figure:
        """Create an interactive population health dashboard"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Age Distribution', 'BMI Categories', 'Gender Distribution',
                          'Blood Pressure Categories', 'Top Medical Conditions', 'Lab Abnormalities'),
            specs=[[{'type': 'histogram'}, {'type': 'pie'}, {'type': 'pie'}],
                   [{'type': 'pie'}, {'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Age distribution
        fig.add_trace(
            go.Histogram(x=self.patients_df['age'], nbinsx=20, name='Age'),
            row=1, col=1
        )
        
        # BMI categories
        bmi_counts = self.patients_df['bmi_category'].value_counts()
        fig.add_trace(
            go.Pie(labels=bmi_counts.index, values=bmi_counts.values, name='BMI'),
            row=1, col=2
        )
        
        # Gender distribution
        gender_counts = self.patients_df['gender'].value_counts()
        fig.add_trace(
            go.Pie(labels=gender_counts.index, values=gender_counts.values, name='Gender'),
            row=1, col=3
        )
        
        # BP categories
        bp_counts = self.vitals_df['bp_category'].value_counts()
        fig.add_trace(
            go.Pie(labels=bp_counts.index, values=bp_counts.values, name='BP'),
            row=2, col=1
        )
        
        # Top conditions
        all_conditions = []
        for conditions in self.patients_df['medical_conditions']:
            if conditions != 'None':
                all_conditions.extend([c.strip() for c in conditions.split(',')])
        
        from collections import Counter
        condition_counts = Counter(all_conditions).most_common(10)
        fig.add_trace(
            go.Bar(x=[c[0] for c in condition_counts], 
                   y=[c[1] for c in condition_counts], name='Conditions'),
            row=2, col=2
        )
        
        # Lab abnormalities by test
        abnormal_rates = self.labs_df.groupby('test_name')['is_abnormal'].mean() * 100
        fig.add_trace(
            go.Bar(x=abnormal_rates.index, y=abnormal_rates.values, name='Abnormal Rate'),
            row=2, col=3
        )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Population Health Dashboard")
        return fig
    
    def plot_vital_signs_timeline(self, patient_id: str) -> go.Figure:
        """Plot vital signs timeline for a specific patient"""
        patient_vitals = self.vitals_df[self.vitals_df['patient_id'] == patient_id]
        
        if patient_vitals.empty:
            fig = go.Figure()
            fig.add_annotation(text="No vital signs data available", 
                             showarrow=False, font=dict(size=20))
            return fig
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Heart Rate', 'Blood Pressure', 'Oxygen Saturation',
                          'Temperature', 'Respiratory Rate', 'BP Category Distribution'),
            vertical_spacing=0.12
        )
        
        # Heart Rate
        fig.add_trace(
            go.Scatter(x=patient_vitals['timestamp'], y=patient_vitals['heart_rate'],
                      mode='lines+markers', name='Heart Rate',
                      line=dict(color='red')),
            row=1, col=1
        )
        
        # Blood Pressure
        fig.add_trace(
            go.Scatter(x=patient_vitals['timestamp'], y=patient_vitals['systolic_bp'],
                      mode='lines+markers', name='Systolic',
                      line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=patient_vitals['timestamp'], y=patient_vitals['diastolic_bp'],
                      mode='lines+markers', name='Diastolic',
                      line=dict(color='lightblue')),
            row=1, col=2
        )
        
        # Oxygen Saturation
        fig.add_trace(
            go.Scatter(x=patient_vitals['timestamp'], y=patient_vitals['oxygen_saturation'],
                      mode='lines+markers', name='O2 Sat',
                      line=dict(color='green')),
            row=2, col=1
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(x=patient_vitals['timestamp'], y=patient_vitals['temperature'],
                      mode='lines+markers', name='Temperature',
                      line=dict(color='orange')),
            row=2, col=2
        )
        
        # Respiratory Rate
        fig.add_trace(
            go.Scatter(x=patient_vitals['timestamp'], y=patient_vitals['respiratory_rate'],
                      mode='lines+markers', name='Resp Rate',
                      line=dict(color='purple')),
            row=3, col=1
        )
        
        # BP Categories
        bp_cats = patient_vitals['bp_category'].value_counts()
        fig.add_trace(
            go.Bar(x=bp_cats.index, y=bp_cats.values, name='BP Categories'),
            row=3, col=2
        )
        
        fig.update_layout(height=900, title_text=f"Vital Signs Timeline - Patient {patient_id}")
        return fig
    
    def create_risk_matrix(self, risk_df: pd.DataFrame) -> go.Figure:
        """Create a risk matrix visualization"""
        fig = px.scatter(
            risk_df, 
            x='age', 
            y='bmi',
            size='risk_score',
            color='risk_level',
            hover_data=['patient_id', 'conditions'],
            title='Patient Risk Matrix',
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        )
        
        fig.update_layout(
            xaxis_title="Age",
            yaxis_title="BMI",
            height=600
        )
        
        return fig
    
    def plot_lab_value_distributions(self) -> None:
        """Create static matplotlib plots for lab value distributions"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, test_name in enumerate(self.labs_df['test_name'].unique()):
            if idx >= 6:
                break
                
            test_data = self.labs_df[self.labs_df['test_name'] == test_name]
            
            ax = axes[idx]
            
            # Plot distribution
            sns.histplot(data=test_data, x='result_value', hue='is_abnormal', 
                        ax=ax, kde=True)
            
            # Add reference range lines
            ref_low = test_data['reference_range_low'].iloc[0]
            ref_high = test_data['reference_range_high'].iloc[0]
            ax.axvline(ref_low, color='green', linestyle='--', alpha=0.5, label='Ref Range')
            ax.axvline(ref_high, color='green', linestyle='--', alpha=0.5)
            
            ax.set_title(f'{test_name} Distribution')
            ax.set_xlabel(f'Value ({test_data["unit"].iloc[0]})')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def generate_health_report(self, patient_id: str) -> str:
        """Generate a text-based health report for a patient"""
        patient = self.patients_df[self.patients_df['patient_id'] == patient_id].iloc[0]
        vitals = self.vitals_df[self.vitals_df['patient_id'] == patient_id]
        labs = self.labs_df[self.labs_df['patient_id'] == patient_id]
        
        report = f"""
{'='*60}
HEALTH ANALYTICS REPORT - PATIENT {patient_id}
{'='*60}

DEMOGRAPHICS:
-------------
Age: {patient['age']}
Gender: {patient['gender']}
Blood Type: {patient['blood_type']}
Height: {patient['height_cm']} cm
Weight: {patient['weight_kg']} kg
BMI: {patient['bmi']:.1f} ({patient['bmi_category']})

MEDICAL CONDITIONS:
------------------
{patient['medical_conditions']}

VITAL SIGNS SUMMARY (Last 30 days):
---------------------------------
Average Heart Rate: {vitals['heart_rate'].mean():.0f} bpm
Average Blood Pressure: {vitals['systolic_bp'].mean():.0f}/{vitals['diastolic_bp'].mean():.0f} mmHg
Average Oxygen Saturation: {vitals['oxygen_saturation'].mean():.1f}%
Average Temperature: {vitals['temperature'].mean():.1f}°C

Blood Pressure Classification:
"""
        
        bp_cats = vitals['bp_category'].value_counts()
        for cat, count in bp_cats.items():
            percentage = (count / len(vitals)) * 100
            report += f"  - {cat}: {percentage:.1f}% of readings\n"
        
        report += "\nRECENT LAB RESULTS:\n-------------------\n"
        
        latest_labs = labs.sort_values('test_date').groupby('test_name').last()
        for test_name, result in latest_labs.iterrows():
            status = "ABNORMAL" if result['is_abnormal'] else "Normal"
            report += f"{test_name}: {result['result_value']:.1f} {result['unit']} "
            report += f"(Ref: {result['reference_range_low']:.0f}-{result['reference_range_high']:.0f}) "
            report += f"[{status}]\n"
        
        # Calculate risk assessment
        from analytics_engine import HealthAnalyticsEngine
        engine = HealthAnalyticsEngine(self.data)
        risk_df = engine.identify_high_risk_patients()
        patient_risk = risk_df[risk_df['patient_id'] == patient_id].iloc[0]
        
        report += f"""
RISK ASSESSMENT:
---------------
Risk Score: {patient_risk['risk_score']}
Risk Level: {patient_risk['risk_level']}
Risk Factors: {patient_risk['risk_factors']}

{'='*60}
"""
        return report
