import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
from typing import Dict, List
from models import Patient, VitalSigns, LabResult

class HealthDataProcessor:
    def __init__(self):
        self.fake = Faker()
        self.patients = []
        self.vital_signs = []
        self.lab_results = []
        
    def generate_synthetic_data(self, num_patients: int = 100):
        """Generate synthetic health data for demonstration"""
        print(f"Generating synthetic data for {num_patients} patients...")
        
        blood_types = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
        conditions_pool = [
            'Hypertension', 'Diabetes Type 2', 'Asthma', 'Arthritis',
            'Depression', 'Anxiety', 'Obesity', 'Heart Disease',
            'COPD', 'Thyroid Disorder', 'None'
        ]
        
        for i in range(num_patients):
            patient_id = f"P{str(i+1).zfill(4)}"
            
            # Generate patient demographics
            age = random.randint(18, 85)
            gender = random.choice(['M', 'F'])
            blood_type = random.choice(blood_types)
            
            # Height and weight with realistic distributions
            if gender == 'M':
                height = np.random.normal(175, 7)
            else:
                height = np.random.normal(162, 6)
            height = max(140, min(210, height))
            
            bmi_target = np.random.normal(26, 5)
            weight = bmi_target * ((height / 100) ** 2)
            weight = max(40, min(180, weight))
            
            # Medical conditions (1-3 per patient)
            num_conditions = random.randint(0, 3)
            conditions = random.sample(conditions_pool[:-1], num_conditions)
            if num_conditions == 0:
                conditions = ['None']
            
            patient = Patient(
                patient_id=patient_id,
                age=age,
                gender=gender,
                blood_type=blood_type,
                height_cm=round(height, 1),
                weight_kg=round(weight, 1),
                medical_conditions=conditions
            )
            self.patients.append(patient)
            
            # Generate vital signs for the last 30 days
            for day in range(30):
                timestamp = datetime.now() - timedelta(days=30-day)
                
                # Vitals with realistic variations
                heart_rate = int(np.random.normal(72, 10))
                heart_rate = max(50, min(120, heart_rate))
                
                # Blood pressure affected by age and conditions
                bp_baseline = 110 if age < 40 else 125
                if 'Hypertension' in conditions:
                    bp_baseline += 20
                systolic = int(np.random.normal(bp_baseline, 15))
                diastolic = int(np.random.normal(75, 10))
                
                temperature = np.random.normal(36.8, 0.5)
                temperature = round(max(35.5, min(39.0, temperature)), 1)
                
                oxygen_sat = int(np.random.normal(98, 2))
                oxygen_sat = max(90, min(100, oxygen_sat))
                
                resp_rate = int(np.random.normal(16, 3))
                resp_rate = max(10, min(25, resp_rate))
                
                vital = VitalSigns(
                    patient_id=patient_id,
                    timestamp=timestamp,
                    heart_rate=heart_rate,
                    systolic_bp=systolic,
                    diastolic_bp=diastolic,
                    temperature=temperature,
                    oxygen_saturation=oxygen_sat,
                    respiratory_rate=resp_rate
                )
                self.vital_signs.append(vital)
            
            # Generate lab results
            lab_tests = [
                ('Glucose', 70, 100, 'mg/dL'),
                ('Cholesterol', 125, 200, 'mg/dL'),
                ('HDL', 40, 60, 'mg/dL'),
                ('LDL', 0, 130, 'mg/dL'),
                ('Hemoglobin', 12, 16, 'g/dL')
            ]
            
            for test_name, low, high, unit in lab_tests:
                if test_name == 'Glucose' and 'Diabetes Type 2' in conditions:
                    result = np.random.normal(140, 30)
                else:
                    result = np.random.normal((low + high) / 2, (high - low) / 6)
                
                lab = LabResult(
                    patient_id=patient_id,
                    test_date=datetime.now() - timedelta(days=random.randint(1, 90)),
                    test_name=test_name,
                    result_value=round(result, 1),
                    unit=unit,
                    reference_range_low=low,
                    reference_range_high=high
                )
                self.lab_results.append(lab)
        
        print("Data generation complete!")
        
    def process_data(self) -> Dict[str, pd.DataFrame]:
        """Convert data to DataFrames for analysis"""
        patients_df = pd.DataFrame([{
            'patient_id': p.patient_id,
            'age': p.age,
            'gender': p.gender,
            'blood_type': p.blood_type,
            'height_cm': p.height_cm,
            'weight_kg': p.weight_kg,
            'bmi': p.bmi,
            'bmi_category': p.bmi_category,
            'medical_conditions': ', '.join(p.medical_conditions)
        } for p in self.patients])
        
        vitals_df = pd.DataFrame([{
            'patient_id': v.patient_id,
            'timestamp': v.timestamp,
            'heart_rate': v.heart_rate,
            'systolic_bp': v.systolic_bp,
            'diastolic_bp': v.diastolic_bp,
            'temperature': v.temperature,
            'oxygen_saturation': v.oxygen_saturation,
            'respiratory_rate': v.respiratory_rate,
            'bp_category': v.bp_category
        } for v in self.vital_signs])
        
        labs_df = pd.DataFrame([{
            'patient_id': l.patient_id,
            'test_date': l.test_date,
            'test_name': l.test_name,
            'result_value': l.result_value,
            'unit': l.unit,
            'reference_range_low': l.reference_range_low,
            'reference_range_high': l.reference_range_high,
            'is_abnormal': l.is_abnormal
        } for l in self.lab_results])
        
        return {
            'patients': patients_df,
            'vitals': vitals_df,
            'labs': labs_df
        }
    
    def clean_data(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Clean and validate data"""
        # Remove duplicates
        for key in dfs:
            dfs[key] = dfs[key].drop_duplicates()
        
        # Handle missing values
        dfs['vitals'] = dfs['vitals'].fillna(dfs['vitals'].mean(numeric_only=True))
        
        # Filter outliers
        vitals = dfs['vitals']
        vitals = vitals[vitals['heart_rate'].between(40, 200)]
        vitals = vitals[vitals['oxygen_saturation'].between(70, 100)]
        dfs['vitals'] = vitals
        
        return dfs
