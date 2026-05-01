import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

class HealthAnalyticsEngine:
    """Core analytics engine for health data processing and statistical analysis"""
    
    # Risk score constants
    RISK_AGE_HIGH = 65
    RISK_AGE_MEDIUM = 50
    RISK_BMI_OBESE = 30
    RISK_BMI_OVERWEIGHT = 25
    RISK_HYPERTENSION = 2
    RISK_DIABETES = 2
    RISK_HEART_DISEASE = 3
    RISK_UNCONTROLLED_BP = 140
    RISK_LOW_O2 = 95
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Initialize analytics engine with data
        
        Args:
            data: Dictionary containing 'patients', 'vitals', 'labs' DataFrames
        """
        self.data = data
        self.patients_df = data['patients']
        self.vitals_df = data['vitals']
        self.labs_df = data['labs']
        self.analysis_log = []
        
    def _log_analysis(self, analysis_name: str, details: str):
        """Internal method to log analysis operations"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {analysis_name}: {details}"
        self.analysis_log.append(log_entry)
        print(f"✓ Logged: {analysis_name}")
    
    def calculate_population_stats(self) -> Dict:
        """Calculate key population health statistics with confidence intervals"""
        self._log_analysis("population_stats", "Computing population health metrics")
        
        # Basic statistics
        stats = {
            'total_patients': len(self.patients_df),
            'avg_age': self.patients_df['age'].mean(),
            'age_std': self.patients_df['age'].std(),
            'age_distribution': self.patients_df['age'].describe().to_dict(),
            'gender_distribution': self.patients_df['gender'].value_counts().to_dict(),
            'bmi_categories': self.patients_df['bmi_category'].value_counts().to_dict(),
            'avg_bmi': self.patients_df['bmi'].mean(),
            'hypertension_prevalence': self.patients_df['medical_conditions'].str.contains(
                'Hypertension', na=False).sum() / len(self.patients_df) * 100,
            'diabetes_prevalence': self.patients_df['medical_conditions'].str.contains(
                'Diabetes', na=False).sum() / len(self.patients_df) * 100
        }
        
        # Add confidence intervals for age
        age_data = self.patients_df['age']
        confidence_level = 0.95
        degrees_freedom = len(age_data) - 1
        sample_mean = age_data.mean()
        sample_standard_error = age_data.std() / np.sqrt(len(age_data))
        confidence_interval = stats.t.interval(
            confidence_level, 
            degrees_freedom, 
            sample_mean, 
            sample_standard_error
        )
        stats['age_confidence_interval_95'] = (confidence_interval[0], confidence_interval[1])
        
        return stats
    
    def perform_ttest(self, group_column: str, value_column: str, group1: str, group2: str) -> Dict:
        """
        Perform independent t-test between two groups
        
        Args:
            group_column: Column name for grouping (e.g., 'gender')
            value_column: Column name for values to compare (e.g., 'systolic_bp')
            group1: First group value (e.g., 'M')
            group2: Second group value (e.g., 'F')
        
        Returns:
            Dictionary with t-test results
        """
        # Merge vitals with patients if needed
        if value_column in self.vitals_df.columns:
            merged_data = self.vitals_df.merge(self.patients_df[['patient_id', group_column]], on='patient_id')
        else:
            merged_data = self.patients_df
        
        group1_data = merged_data[merged_data[group_column] == group1][value_column].dropna()
        group2_data = merged_data[merged_data[group_column] == group2][value_column].dropna()
        
        if len(group1_data) == 0 or len(group2_data) == 0:
            return {'error': 'Insufficient data for t-test'}
        
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.std()**2 + 
                              (len(group2_data) - 1) * group2_data.std()**2) / 
                             (len(group1_data) + len(group2_data) - 2))
        cohens_d = abs(group1_data.mean() - group2_data.mean()) / pooled_std if pooled_std > 0 else 0
        
        result = {
            'test_name': f"T-test: {value_column} between {group1} and {group2}",
            'group1_name': group1,
            'group2_name': group2,
            'group1_mean': group1_data.mean(),
            'group2_mean': group2_data.mean(),
            'group1_std': group1_data.std(),
            'group2_std': group2_data.std(),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'interpretation': f"{value_column} is {'significantly' if p_value < 0.05 else 'not significantly'} different between groups (p={p_value:.4f}, d={cohens_d:.2f})"
        }
        
        self._log_analysis("t_test", f"{value_column} ({group1} vs {group2}) - p={p_value:.4f}, d={cohens_d:.2f}")
        return result
    
    def perform_anova(self, group_column: str, value_column: str) -> Dict:
        """
        Perform one-way ANOVA across multiple groups
        
        Args:
            group_column: Column name for grouping (e.g., 'bp_category')
            value_column: Column name for values to compare (e.g., 'bmi')
        
        Returns:
            Dictionary with ANOVA results
        """
        # Merge data if needed
        if value_column in self.vitals_df.columns:
            merged_data = self.vitals_df.merge(self.patients_df[['patient_id', group_column]], on='patient_id')
        else:
            merged_data = self.patients_df
        
        groups = merged_data[group_column].dropna().unique()
        group_data = [merged_data[merged_data[group_column] == group][value_column].dropna() for group in groups]
        group_data = [g for g in group_data if len(g) > 0]
        
        if len(group_data) < 2:
            return {'error': 'Insufficient groups for ANOVA'}
        
        f_stat, p_value = stats.f_oneway(*group_data)
        
        # Calculate eta-squared (effect size)
        all_values = np.concatenate(group_data)
        grand_mean = all_values.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_data)
        ss_total = sum((val - grand_mean)**2 for val in all_values)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        result = {
            'test_name': f"ANOVA: {value_column} across {group_column}",
            'groups': list(groups),
            'group_counts': [len(g) for g in group_data],
            'group_means': [g.mean() for g in group_data],
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant': p_value < 0.05,
            'interpretation': f"There {'is' if p_value < 0.05 else 'is not'} a significant difference in {value_column} across {group_column} groups (F={f_stat:.3f}, p={p_value:.4f}, η²={eta_squared:.3f})"
        }
        
        self._log_analysis("anova", f"{value_column} across {group_column} - p={p_value:.4f}, η²={eta_squared:.3f}")
        return result
    
    def calculate_correlation_with_pvalue(self, var1: str, var2: str) -> Dict:
        """
        Calculate Pearson correlation with p-value
        
        Args:
            var1: First variable name (column in patients_df)
            var2: Second variable name (column in patients_df)
        
        Returns:
            Dictionary with correlation results
        """
        # Handle if variables are in vitals_df
        df_to_use = self.patients_df
        if var1 in self.vitals_df.columns:
            # Need to aggregate vitals per patient
            vitals_agg = self.vitals_df.groupby('patient_id')[var1].mean().reset_index()
            df_to_use = self.patients_df.merge(vitals_agg, on='patient_id', suffixes=('', '_vitals'))
            var1 = var1 + '_vitals' if var1 + '_vitals' in df_to_use.columns else var1
        
        if var2 in self.vitals_df.columns:
            vitals_agg = self.vitals_df.groupby('patient_id')[var2].mean().reset_index()
            df_to_use = df_to_use.merge(vitals_agg, on='patient_id', suffixes=('', '_vitals'))
            var2 = var2 + '_vitals' if var2 + '_vitals' in df_to_use.columns else var2
        
        data1 = df_to_use[var1].dropna()
        data2 = df_to_use[var2].dropna()
        
        # Align indices
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        if len(data1) < 3:
            return {'error': 'Insufficient data for correlation'}
        
        correlation, p_value = stats.pearsonr(data1, data2)
        
        result = {
            'variables': (var1.replace('_vitals', ''), var2.replace('_vitals', '')),
            'correlation': correlation,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'strength': self._interpret_correlation_strength(abs(correlation)),
            'interpretation': f"{self._interpret_correlation_strength(abs(correlation))} {'positive' if correlation > 0 else 'negative'} correlation (r={correlation:.3f}, p={p_value:.4f})"
        }
        
        self._log_analysis("correlation", f"{var1} vs {var2} - r={correlation:.3f}, p={p_value:.4f}")
        return result
    
    def _interpret_correlation_strength(self, abs_corr: float) -> str:
        """Interpret correlation strength"""
        if abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.4:
            return "Moderate"
        else:
            return "Weak"
    
    def detect_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Detect outliers using IQR method
        
        Args:
            df: DataFrame to analyze
            column: Column name to check for outliers
        
        Returns:
            DataFrame with outlier information
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        if len(outliers) > 0:
            result_df = outliers.copy()
            result_df['outlier_bound_lower'] = lower_bound
            result_df['outlier_bound_upper'] = upper_bound
            result_df['outlier_type'] = result_df[column].apply(
                lambda x: 'Below' if x < lower_bound else 'Above'
            )
        else:
            result_df = pd.DataFrame()
        
        self._log_analysis("outlier_detection", f"Found {len(outliers)} outliers in {column}")
        return result_df
    
    def calculate_zscore_for_labs(self, patient_id: str) -> pd.DataFrame:
        """
        Calculate z-scores for patient's lab results
        
        Args:
            patient_id: Patient identifier
        
        Returns:
            DataFrame with lab results and their z-scores
        """
        patient_labs = self.labs_df[self.labs_df['patient_id'] == patient_id].copy()
        
        if patient_labs.empty:
            return pd.DataFrame()
        
        z_scores = []
        percentiles = []
        
        for test_name in patient_labs['test_name'].unique():
            all_results = self.labs_df[self.labs_df['test_name'] == test_name]['result_value']
            mean_val = all_results.mean()
            std_val = all_results.std()
            
            patient_test = patient_labs[patient_labs['test_name'] == test_name]
            for idx, row in patient_test.iterrows():
                z_score = (row['result_value'] - mean_val) / std_val if std_val > 0 else 0
                z_scores.append(z_score)
                
                # Calculate percentile
                percentile = stats.percentileofscore(all_results, row['result_value'])
                percentiles.append(percentile)
        
        patient_labs['z_score'] = z_scores
        patient_labs['percentile'] = percentiles
        patient_labs['abnormal_zscore'] = abs(patient_labs['z_score']) > 2
        
        self._log_analysis("zscore_calculation", f"Calculated z-scores for {len(patient_labs)} lab results of patient {patient_id}")
        return patient_labs
    
    def analyze_vital_trends(self, patient_id: Optional[str] = None) -> Dict:
        """Analyze vital signs trends over time with significance testing"""
        if patient_id:
            vitals = self.vitals_df[self.vitals_df['patient_id'] == patient_id]
        else:
            vitals = self.vitals_df
            
        if len(vitals) == 0:
            return {'error': 'No vital signs data available'}
        
        daily_avg = vitals.groupby(vitals['timestamp'].dt.date).agg({
            'heart_rate': 'mean',
            'systolic_bp': 'mean',
            'diastolic_bp': 'mean',
            'oxygen_saturation': 'mean'
        }).round(2)
        
        trends = {
            'heart_rate_trend': self._calculate_trend_with_significance(daily_avg['heart_rate']),
            'systolic_bp_trend': self._calculate_trend_with_significance(daily_avg['systolic_bp']),
            'diastolic_bp_trend': self._calculate_trend_with_significance(daily_avg['diastolic_bp']),
            'oxygen_saturation_trend': self._calculate_trend_with_significance(daily_avg['oxygen_saturation']),
            'avg_heart_rate': vitals['heart_rate'].mean(),
            'avg_systolic': vitals['systolic_bp'].mean(),
            'avg_diastolic': vitals['diastolic_bp'].mean(),
            'oxygen_saturation_mean': vitals['oxygen_saturation'].mean(),
            'abnormal_bp_percentage': (vitals['bp_category'] != 'Normal').sum() / len(vitals) * 100
        }
        
        return trends
    
    def _calculate_trend_with_significance(self, series: pd.Series) -> Dict:
        """Calculate trend direction and significance using linear regression"""
        if len(series) < 2:
            return {"direction": "Insufficient data", "p_value": None, "significant": False, "slope": 0}
        
        # Remove NaN values
        series = series.dropna()
        if len(series) < 2:
            return {"direction": "Insufficient data", "p_value": None, "significant": False, "slope": 0}
        
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        
        # Calculate p-value for slope using scipy
        n = len(series)
        y_pred = model.predict(X)
        residuals = y - y_pred
        residual_std = np.std(residuals, ddof=2)
        slope_std_error = residual_std / np.sqrt(np.sum((X.flatten() - X.mean())**2))
        t_stat = slope / slope_std_error if slope_std_error > 0 else 0
        
        # Two-tailed test
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
        
        if slope > 0.1:
            direction = "Increasing"
        elif slope < -0.1:
            direction = "Decreasing"
        else:
            direction = "Stable"
        
        return {
            'direction': direction,
            'slope': slope,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def identify_high_risk_patients(self) -> pd.DataFrame:
        """Identify patients at high risk using multiple factors"""
        risk_scores = []
        
        for _, patient in self.patients_df.iterrows():
            patient_id = patient['patient_id']
            risk_score = 0
            risk_factors = []
            
            # Age risk
            if patient['age'] > self.RISK_AGE_HIGH:
                risk_score += 3
                risk_factors.append("Advanced age")
            elif patient['age'] > self.RISK_AGE_MEDIUM:
                risk_score += 1
                risk_factors.append("Age > 50")
            
            # BMI risk
            if patient['bmi'] >= self.RISK_BMI_OBESE:
                risk_score += 2
                risk_factors.append("Obesity")
            elif patient['bmi'] >= self.RISK_BMI_OVERWEIGHT:
                risk_score += 1
                risk_factors.append("Overweight")
            
            # Condition-based risk
            conditions = patient['medical_conditions']
            if 'Hypertension' in conditions:
                risk_score += self.RISK_HYPERTENSION
                risk_factors.append("Hypertension")
            if 'Diabetes Type 2' in conditions:
                risk_score += self.RISK_DIABETES
                risk_factors.append("Diabetes")
            if 'Heart Disease' in conditions:
                risk_score += self.RISK_HEART_DISEASE
                risk_factors.append("Heart Disease")
            
            # Vital signs risk
            patient_vitals = self.vitals_df[
                self.vitals_df['patient_id'] == patient_id
            ]
            if len(patient_vitals) > 0:
                avg_systolic = patient_vitals['systolic_bp'].mean()
                if avg_systolic >= self.RISK_UNCONTROLLED_BP:
                    risk_score += 2
                    risk_factors.append("Uncontrolled hypertension")
                
                avg_o2 = patient_vitals['oxygen_saturation'].mean()
                if avg_o2 < self.RISK_LOW_O2:
                    risk_score += 1
                    risk_factors.append("Low oxygen saturation")
            
            # Lab results risk
            patient_labs = self.labs_df[self.labs_df['patient_id'] == patient_id]
            abnormal_labs = patient_labs[patient_labs['is_abnormal'] == True]
            if len(abnormal_labs) > 0:
                risk_score += len(abnormal_labs)
                risk_factors.append(f"{len(abnormal_labs)} abnormal lab results")
            
            # Determine risk level
            if risk_score >= 6:
                risk_level = 'High'
            elif risk_score >= 3:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            risk_scores.append({
                'patient_id': patient_id,
                'age': patient['age'],
                'gender': patient['gender'],
                'bmi': round(patient['bmi'], 1),
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_factors': ', '.join(risk_factors) if risk_factors else 'None identified',
                'conditions': conditions
            })
        
        risk_df = pd.DataFrame(risk_scores)
        self._log_analysis("risk_assessment", f"Assessed {len(risk_df)} patients - {len(risk_df[risk_df['risk_level'] == 'High'])} high risk")
        return risk_df.sort_values('risk_score', ascending=False)
    
    def analyze_lab_correlations(self) -> pd.DataFrame:
        """Analyze correlations between lab results and patient characteristics"""
        lab_patient = self.labs_df.merge(
            self.patients_df[['patient_id', 'age', 'bmi', 'gender']],
            on='patient_id'
        )
        
        correlations = []
        for test in lab_patient['test_name'].unique():
            test_data = lab_patient[lab_patient['test_name'] == test]
            
            # Calculate correlations with p-values
            if len(test_data) > 2:
                age_corr, age_p = stats.pearsonr(test_data['result_value'], test_data['age'])
                bmi_corr, bmi_p = stats.pearsonr(test_data['result_value'], test_data['bmi'])
            else:
                age_corr, age_p = 0, 1
                bmi_corr, bmi_p = 0, 1
            
            # Gender comparison with t-test
            male_data = test_data[test_data['gender'] == 'M']['result_value']
            female_data = test_data[test_data['gender'] == 'F']['result_value']
            if len(male_data) > 0 and len(female_data) > 0:
                t_stat, p_value = stats.ttest_ind(male_data, female_data)
                gender_sig = p_value < 0.05
            else:
                p_value = 1
                gender_sig = False
            
            correlations.append({
                'test_name': test,
                'age_correlation': round(age_corr, 3),
                'age_p_value': round(age_p, 4),
                'bmi_correlation': round(bmi_corr, 3),
                'bmi_p_value': round(bmi_p, 4),
                'male_avg': round(male_data.mean(), 2) if len(male_data) > 0 else 0,
                'female_avg': round(female_data.mean(), 2) if len(female_data) > 0 else 0,
                'gender_diff_pvalue': round(p_value, 4),
                'gender_significant': gender_sig,
                'abnormal_rate': (test_data['is_abnormal'] == True).sum() / len(test_data) * 100
            })
        
        self._log_analysis("lab_correlations", f"Analyzed {len(correlations)} lab tests")
        return pd.DataFrame(correlations)
    
    def cluster_patients(self, n_clusters: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Cluster patients based on health metrics"""
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
        
        if not features:
            self._log_analysis("clustering", "No features available for clustering")
            return pd.DataFrame(), pd.DataFrame()
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        cluster_results = pd.DataFrame({
            'patient_id': patient_ids,
            'cluster': clusters
        })
        
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
        
        self._log_analysis("clustering", f"Created {n_clusters} patient clusters")
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
    
    def export_analysis_results(self, output_dir: str = "analysis_outputs"):
        """
        Export all analysis results to CSV and JSON files
        
        Args:
            output_dir: Directory to save outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export risk scores
        risk_df = self.identify_high_risk_patients()
        risk_df.to_csv(output_path / "risk_scores.csv", index=False)
        
        # Export lab correlations
        lab_corr = self.analyze_lab_correlations()
        lab_corr.to_csv(output_path / "lab_correlations.csv", index=False)
        
        # Export population stats
        stats = self.calculate_population_stats()
        with open(output_path / "population_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Export analysis log
        with open(output_path / "analysis_log.txt", "w") as f:
            f.write("\n".join(self.analysis_log))
        
        # Export cluster assignments if available
        try:
            clustered, profiles = self.cluster_patients(n_clusters=4)
            if not clustered.empty:
                clustered.to_csv(output_path / "patient_clusters.csv", index=False)
                profiles.to_csv(output_path / "cluster_profiles.csv", index=False)
        except Exception as e:
            print(f"Could not export clusters: {e}")
        
        print(f"✓ Analysis results exported to {output_dir}/")
        self._log_analysis("export", f"Exported results to {output_dir}")
    
    def load_external_data(self, patients_csv: str = None, vitals_csv: str = None, labs_csv: str = None):
        """
        Load external CSV data into the system
        
        Args:
            patients_csv: Path to patients CSV file
            vitals_csv: Path to vitals CSV file
            labs_csv: Path to labs CSV file
        """
        if patients_csv and Path(patients_csv).exists():
            external_patients = pd.read_csv(patients_csv)
            # Validate required columns
            required_cols = ['patient_id', 'age', 'gender', 'bmi', 'bmi_category', 'medical_conditions']
            if all(col in external_patients.columns for col in required_cols):
                self.patients_df = external_patients
                self.data['patients'] = external_patients
                print(f"✓ Loaded {len(external_patients)} patients from {patients_csv}")
            else:
                print(f"✗ Patients CSV missing required columns. Expected: {required_cols}")
        
        if vitals_csv and Path(vitals_csv).exists():
            external_vitals = pd.read_csv(vitals_csv)
            if 'timestamp' in external_vitals.columns:
                external_vitals['timestamp'] = pd.to_datetime(external_vitals['timestamp'])
            self.vitals_df = external_vitals
            self.data['vitals'] = external_vitals
            print(f"✓ Loaded {len(external_vitals)} vital sign records from {vitals_csv}")
        
        if labs_csv and Path(labs_csv).exists():
            external_labs = pd.read_csv(labs_csv)
            if 'test_date' in external_labs.columns:
                external_labs['test_date'] = pd.to_datetime(external_labs['test_date'])
            self.labs_df = external_labs
            self.data['labs'] = external_labs
            print(f"✓ Loaded {len(external_labs)} lab results from {labs_csv}")
        
        self._log_analysis("data_import", f"Imported external data")
