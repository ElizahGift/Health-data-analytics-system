import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class HealthAnalyticsEngine:
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self.patients_df = data['patients']
        self.vitals_df = data['vitals']
        self.labs_df = data['labs']
        
    def calculate_population_stats(self) -> Dict:
        """Calculate key population health statistics"""
        stats = {
            'total_patients': len(self.patients_df),
            'avg_age': self.patients_df['age'].mean(),
            'age_distribution': self.patients_df['age'].describe().to_dict(),
            'gender_distribution': self.patients_df['gender'].value_counts().to_dict(),
            'bmi_categories': self.patients_df['bmi_category'].value_counts().to_dict(),
            'avg_bmi': self.patients_df['bmi'].mean(),
            'hypertension_prevalence': self.patients_df['medical_conditions'].str.contains(
                'Hypertension', na=False).sum() / len(self.patients_df) * 100,
            'diabetes_prevalence': self.patients_df['medical_conditions'].str.contains(
                'Diabetes', na=False).sum() / len(self.patients_df) * 100
        }
        return stats
    
    def analyze_vital_trends(self, patient_id: str = None) -> Dict:
        """Analyze vital signs trends over time"""
        if patient_id:
            vitals = self.vitals_df[self.vitals_df['patient_id'] == patient_id]
        else:
            vitals = self.vitals_df
            
        daily_avg = vitals.groupby(vitals['timestamp'].dt.date).agg({
            'heart_rate': 'mean',
            'systolic_bp': 'mean',
            'diastolic_bp': 'mean',
            'oxygen_saturation': 'mean'
        }).round(2)
        
        trends = {
            'heart_rate_trend': self._calculate_trend(daily_avg['heart_rate']),
            'bp_trend': self._calculate_trend(daily_avg['systolic_bp']),
            'avg_heart_rate': vitals['heart_rate'].mean(),
            'avg_systolic': vitals['systolic_bp'].mean(),
            'avg_diastolic': vitals['diastolic_bp'].mean(),
            'abnormal_bp_percentage': (vitals['bp_category'] != 'Normal').sum() / len(vitals) * 100
        }
        
        return trends
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction using linear regression"""
        if len(series) < 2:
            return "Insufficient data"
        
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        
        if slope > 0.1:
            return "Increasing"
        elif slope < -0.1:
            return "Decreasing"
        else:
            return "Stable"
    
    def identify_high_risk_patients(self) -> pd.DataFrame:
        """Identify patients at high risk using multiple factors"""
        risk_scores = []
        
        for _, patient in self.patients_df.iterrows():
            patient_id = patient['patient_id']
            risk_score = 0
            risk_factors = []
            
            # Age risk
            if patient['age'] > 65:
                risk_score += 3
                risk_factors.append("Advanced age")
            elif patient['age'] > 50:
                risk_score += 1
                risk_factors.append("Age > 50")
            
            # BMI risk
            if patient['bmi'] >= 30:
                risk_score += 2
                risk_factors.append("Obesity")
            elif patient['bmi'] >= 25:
                risk_score += 1
                risk_factors.append("Overweight")
            
            # Condition-based risk
            conditions = patient['medical_conditions']
            if 'Hypertension' in conditions:
                risk_score += 2
                risk_factors.append("Hypertension")
            if 'Diabetes Type 2' in conditions:
                risk_score += 2
                risk_factors.append("Diabetes")
            if 'Heart Disease' in conditions:
                risk_score += 3
                risk_factors.append("Heart Disease")
            
            # Vital signs risk
            patient_vitals = self.vitals_df[
                self.vitals_df['patient_id'] == patient_id
            ]
            if len(patient_vitals) > 0:
                avg_systolic = patient_vitals['systolic_bp'].mean()
                if avg_systolic >= 140:
                    risk_score += 2
                    risk_factors.append("Uncontrolled hypertension")
                
                avg_o2 = patient_vitals['oxygen_saturation'].mean()
                if avg_o2 < 95:
                    risk_score += 1
                    risk_factors.append("Low oxygen saturation")
            
            # Lab results risk
            patient_labs = self.labs_df[self.labs_df['patient_id'] == patient_id]
            abnormal_labs = patient_labs[patient_labs['is_abnormal'] == True]
            if len(abnormal_labs) > 0:
                risk_score += len(abnormal_labs)
                risk_factors.append(f"{len(abnormal_labs)} abnormal lab results")
            
            risk_scores.append({
                'patient_id': patient_id,
                'age': patient['age'],
                'gender': patient['gender'],
                'bmi': round(patient['bmi'], 1),
                'risk_score': risk_score,
                'risk_level': 'High' if risk_score >= 6 else 'Medium' if risk_score >= 3 else 'Low',
                'risk_factors': ', '.join(risk_factors),
                'conditions': conditions
            })
        
        risk_df = pd.DataFrame(risk_scores)
        return risk_df.sort_values('risk_score', ascending=False)
    
    def analyze_lab_correlations(self) -> pd.DataFrame:
        """Analyze correlations between lab results and patient characteristics"""
        # Merge patient data with lab results
        lab_patient = self.labs_df.merge(
            self.patients_df[['patient_id', 'age', 'bmi', 'gender']],
            on='patient_id'
        )
        
        correlations = []
        for test in lab_patient['test_name'].unique():
            test_data = lab_patient[lab_patient['test_name'] == test]
            
            # Calculate correlations
            age_corr = test_data['result_value'].corr(test_data['age'])
            bmi_corr = test_data['result_value'].corr(test_data['bmi'])
            
            # Gender comparison
            male_avg = test_data[test_data['gender'] == 'M']['result_value'].mean()
            female_avg = test_data[test_data['gender'] == 'F']['result_value'].mean()
            
            correlations.append({
                'test_name': test,
                'age_correlation': round(age_corr, 3),
                'bmi_correlation': round(bmi_corr, 3),
                'male_avg': round(male_avg, 2),
                'female_avg': round(female_avg, 2),
                'abnormal_rate': (test_data['is_abnormal'] == True).sum() / len(test_data) * 100
            })
        
        return pd.DataFrame(correlations)
    
    def cluster_patients(self, n_clusters: int = 4) -> Tuple[pd.DataFrame, KMeans]:
        """Cluster patients based on health metrics"""
        # Prepare features for clustering
        features = []
        patient_ids = []
        
        for _, patient in self.patients_df.iterrows():
            patient_id = patient['patient_id']
            patient_vitals = self.vitals_df[self.vitals_df['patient_id'] == patient_id]
            
            if len(patient_vitals) > 0:
                feature_vector = [
                    patient['age'],
                    patient['bmi'],
                    patient_vitals['heart_rate'].mean(),
                    patient_vitals['systolic_bp'].mean(),
                    patient_vitals['diastolic_bp'].mean(),
                    patient_vitals['oxygen_saturation'].mean()
                ]
                features.append(feature_vector)
                patient_ids.append(patient_id)
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Create results DataFrame
        cluster_results = pd.DataFrame({
            'patient_id': patient_ids,
            'cluster': clusters
        })
        
        # Add cluster descriptions
        cluster_profiles = []
        for cluster_id in range(n_clusters):
            cluster_patients = cluster_results[cluster_results['cluster'] == cluster_id]['patient_id']
            cluster_data = self.patients_df[self.patients_df['patient_id'].isin(cluster_patients)]
            
            profile = {
                'cluster': cluster_id,
                'size': len(cluster_data),
                'avg_age': cluster_data['age'].mean(),
                'avg_bmi': cluster_data['bmi'].mean(),
                'common_conditions': self._get_common_conditions(cluster_patients)
            }
            cluster_profiles.append(profile)
        
        return cluster_results.merge(self.patients_df, on='patient_id'), pd.DataFrame(cluster_profiles)
    
    def _get_common_conditions(self, patient_ids: pd.Series) -> str:
        """Get most common conditions in a group of patients"""
        cluster_patients = self.patients_df[
            self.patients_df['patient_id'].isin(patient_ids)
        ]
        
        all_conditions = []
        for conditions in cluster_patients['medical_conditions']:
            if conditions != 'None':
                all_conditions.extend([c.strip() for c in conditions.split(',')])
        
        if not all_conditions:
            return "No common conditions"
        
        from collections import Counter
        common = Counter(all_conditions).most_common(3)
        return ', '.join([f"{cond} ({count})" for cond, count in common])
