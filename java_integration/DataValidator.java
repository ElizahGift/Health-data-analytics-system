import java.util.*;
import java.util.stream.Collectors;

/**
 * Java-based data validator for health analytics system
 * Provides high-performance validation and clinical calculations
 */
public class DataValidator {
    
    // Clinical reference constants
    private static final int HEART_RATE_MIN = 40;
    private static final int HEART_RATE_MAX = 200;
    private static final int SYSTOLIC_BP_MIN = 70;
    private static final int SYSTOLIC_BP_MAX = 250;
    private static final int DIASTOLIC_BP_MIN = 40;
    private static final int DIASTOLIC_BP_MAX = 150;
    private static final double TEMP_MIN = 35.0;
    private static final double TEMP_MAX = 42.0;
    private static final int O2_SAT_MIN = 70;
    private static final int O2_SAT_MAX = 100;
    private static final int RESP_RATE_MIN = 5;
    private static final int RESP_RATE_MAX = 50;
    
    public static class ValidationResult {
        public boolean isValid;
        public List<String> errors;
        public List<String> warnings;
        
        public ValidationResult() {
            this.isValid = true;
            this.errors = new ArrayList<>();
            this.warnings = new ArrayList<>();
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("Validation Result: ").append(isValid ? "PASSED" : "FAILED").append("\n");
            if (!errors.isEmpty()) {
                sb.append("Errors:\n");
                for (String error : errors) {
                    sb.append("  - ").append(error).append("\n");
                }
            }
            if (!warnings.isEmpty()) {
                sb.append("Warnings:\n");
                for (String warning : warnings) {
                    sb.append("  - ").append(warning).append("\n");
                }
            }
            return sb.toString();
        }
    }
    
    /**
     * Validate vital signs with both errors and warnings
     */
    public ValidationResult validateVitalSigns(
            int heartRate, 
            int systolicBP, 
            int diastolicBP, 
            double temperature,
            int oxygenSaturation,
            int respiratoryRate) {
        
        ValidationResult result = new ValidationResult();
        
        // Heart rate validation
        if (heartRate < HEART_RATE_MIN || heartRate > HEART_RATE_MAX) {
            result.isValid = false;
            result.errors.add("Heart rate out of range: " + heartRate + 
                             " (normal: " + HEART_RATE_MIN + "-" + HEART_RATE_MAX + ")");
        } else if (heartRate < 60 || heartRate > 100) {
            result.warnings.add("Heart rate outside normal resting range: " + heartRate);
        }
        
        // Blood pressure validation
        if (systolicBP < SYSTOLIC_BP_MIN || systolicBP > SYSTOLIC_BP_MAX) {
            result.isValid = false;
            result.errors.add("Systolic BP out of range: " + systolicBP +
                             " (normal: " + SYSTOLIC_BP_MIN + "-" + SYSTOLIC_BP_MAX + ")");
        }
        
        if (diastolicBP < DIASTOLIC_BP_MIN || diastolicBP > DIASTOLIC_BP_MAX) {
            result.isValid = false;
            result.errors.add("Diastolic BP out of range: " + diastolicBP +
                             " (normal: " + DIASTOLIC_BP_MIN + "-" + DIASTOLIC_BP_MAX + ")");
        }
        
        if (systolicBP <= diastolicBP) {
            result.isValid = false;
            result.errors.add("Systolic BP (" + systolicBP + ") must be greater than diastolic BP (" + diastolicBP + ")");
        }
        
        // Temperature validation
        if (temperature < TEMP_MIN || temperature > TEMP_MAX) {
            result.isValid = false;
            result.errors.add("Temperature out of range: " + temperature +
                             " (normal: " + TEMP_MIN + "-" + TEMP_MAX + ")");
        } else if (temperature > 37.5) {
            result.warnings.add("Elevated temperature: " + temperature + "°C (possible fever)");
        } else if (temperature < 36.0) {
            result.warnings.add("Low temperature: " + temperature + "°C (possible hypothermia)");
        }
        
        // Oxygen saturation validation
        if (oxygenSaturation < O2_SAT_MIN || oxygenSaturation > O2_SAT_MAX) {
            result.isValid = false;
            result.errors.add("Oxygen saturation out of range: " + oxygenSaturation +
                             " (normal: " + O2_SAT_MIN + "-" + O2_SAT_MAX + ")");
        } else if (oxygenSaturation < 95) {
            result.warnings.add("Low oxygen saturation: " + oxygenSaturation + "% (consider supplemental oxygen)");
        }
        
        // Respiratory rate validation
        if (respiratoryRate < RESP_RATE_MIN || respiratoryRate > RESP_RATE_MAX) {
            result.isValid = false;
            result.errors.add("Respiratory rate out of range: " + respiratoryRate +
                             " (normal: " + RESP_RATE_MIN + "-" + RESP_RATE_MAX + ")");
        } else if (respiratoryRate > 20) {
            result.warnings.add("Elevated respiratory rate: " + respiratoryRate + " breaths/min");
        }
        
        return result;
    }
    
    /**
     * Overloaded method for backward compatibility (without respiratory rate)
     */
    public ValidationResult validateVitalSigns(
            int heartRate, 
            int systolicBP, 
            int diastolicBP, 
            double temperature,
            int oxygenSaturation) {
        return validateVitalSigns(heartRate, systolicBP, diastolicBP, temperature, oxygenSaturation, 16);
    }
    
    /**
     * Calculate Mean Arterial Pressure (MAP)
     */
    public double calculateMAP(int systolicBP, int diastolicBP) {
        return diastolicBP + ((systolicBP - diastolicBP) / 3.0);
    }
    
    /**
     * Calculate Pulse Pressure
     */
    public int calculatePulsePressure(int systolicBP, int diastolicBP) {
        return systolicBP - diastolicBP;
    }
    
    /**
     * Classify blood pressure according to JNC 8 guidelines
     */
    public String classifyBloodPressure(int systolicBP, int diastolicBP) {
        if (systolicBP < 120 && diastolicBP < 80) {
            return "Normal";
        } else if (systolicBP < 130 && diastolicBP < 80) {
            return "Elevated";
        } else if (systolicBP < 140 || diastolicBP < 90) {
            return "High Blood Pressure Stage 1";
        } else if (systolicBP >= 140 || diastolicBP >= 90) {
            return "High Blood Pressure Stage 2";
        } else if (systolicBP >= 180 || diastolicBP >= 120) {
            return "Hypertensive Crisis";
        } else {
            return "Unclassified";
        }
    }
    
    /**
     * Calculate body mass index
     */
    public double calculateBMI(double weightKg, double heightCm) {
        if (heightCm <= 0) return 0;
        double heightM = heightCm / 100.0;
        return weightKg / (heightM * heightM);
    }
    
    /**
     * Classify BMI according to WHO standards
     */
    public String classifyBMI(double bmi) {
        if (bmi < 18.5) {
            return "Underweight";
        } else if (bmi < 25) {
            return "Normal weight";
        } else if (bmi < 30) {
            return "Overweight";
        } else {
            return "Obese";
        }
    }
    
    /**
     * Calculate risk score for a single patient
     */
    public double calculatePatientRiskScore(Map<String, Object> patientData) {
        double score = 0.0;
        
        // Age risk
        int age = (int) patientData.getOrDefault("age", 0);
        if (age > 65) score += 3.0;
        else if (age > 50) score += 1.5;
        
        // BMI risk
        double bmi = (double) patientData.getOrDefault("bmi", 22.0);
        if (bmi >= 30) score += 2.5;
        else if (bmi >= 25) score += 1.0;
        
        // Blood pressure risk
        int systolic = (int) patientData.getOrDefault("systolicBP", 120);
        int diastolic = (int) patientData.getOrDefault("diastolicBP", 80);
        if (systolic >= 140 || diastolic >= 90) score += 2.0;
        
        // Condition-based risk
        @SuppressWarnings("unchecked")
        List<String> conditions = (List<String>) patientData.getOrDefault("conditions", new ArrayList<>());
        if (conditions.contains("Hypertension")) score += 2.0;
        if (conditions.contains("Diabetes Type 2")) score += 2.0;
        if (conditions.contains("Heart Disease")) score += 3.0;
        
        return score;
    }
    
    /**
     * Calculate risk scores for multiple patients
     */
    public Map<String, Double> calculateRiskScores(List<Map<String, Object>> patientData) {
        Map<String, Double> riskScores = new HashMap<>();
        
        for (Map<String, Object> data : patientData) {
            String patientId = (String) data.get("patientId");
            double score = calculatePatientRiskScore(data);
            riskScores.put(patientId, score);
        }
        
        return riskScores;
    }
    
    /**
     * Batch validate multiple vital signs records
     */
    public List<ValidationResult> batchValidateVitals(List<Map<String, Object>> vitalsList) {
        List<ValidationResult> results = new ArrayList<>();
        
        for (Map<String, Object> vitals : vitalsList) {
            int heartRate = (int) vitals.getOrDefault("heartRate", 70);
            int systolicBP = (int) vitals.getOrDefault("systolicBP", 120);
            int diastolicBP = (int) vitals.getOrDefault("diastolicBP", 80);
            double temperature = (double) vitals.getOrDefault("temperature", 36.8);
            int oxygenSaturation = (int) vitals.getOrDefault("oxygenSaturation", 98);
            int respiratoryRate = (int) vitals.getOrDefault("respiratoryRate", 16);
            
            ValidationResult result = validateVitalSigns(heartRate, systolicBP, diastolicBP, 
                                                          temperature, oxygenSaturation, respiratoryRate);
            results.add(result);
        }
        
        return results;
    }
    
    /**
     * Calculate trend from a list of values
     */
    public String calculateTrend(List<Double> values) {
        if (values == null || values.size() < 2) {
            return "Insufficient data";
        }
        
        // Simple linear regression
        int n = values.size();
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        
        for (int i = 0; i < n; i++) {
            double x = i;
            double y = values.get(i);
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
        }
        
        double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        
        if (slope > 0.1) {
            return "Increasing";
        } else if (slope < -0.1) {
            return "Decreasing";
        } else {
            return "Stable";
        }
    }
    
    /**
     * Calculate basic statistics for a list of values
     */
    public Map<String, Double> calculateStatistics(List<Double> values) {
        Map<String, Double> stats = new HashMap<>();
        
        if (values == null || values.isEmpty()) {
            stats.put("mean", 0.0);
            stats.put("std", 0.0);
            stats.put("min", 0.0);
            stats.put("max", 0.0);
            return stats;
        }
        
        double sum = 0;
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        
        for (double val : values) {
            sum += val;
            min = Math.min(min, val);
            max = Math.max(max, val);
        }
        
        double mean = sum / values.size();
        
        double variance = 0;
        for (double val : values) {
            variance += Math.pow(val - mean, 2);
        }
        double std = Math.sqrt(variance / values.size());
        
        stats.put("mean", mean);
        stats.put("std", std);
        stats.put("min", min);
        stats.put("max", max);
        
        return stats;
    }
    
    /**
     * Main method for testing
     */
    public static void main(String[] args) {
        DataValidator validator = new DataValidator();
        
        // Test vital signs validation
        ValidationResult result = validator.validateVitalSigns(75, 120, 80, 36.8, 98, 16);
        System.out.println(result);
        
        // Test BP classification
        String bpClass = validator.classifyBloodPressure(145, 92);
        System.out.println("BP Classification: " + bpClass);
        
        // Test MAP calculation
        double map = validator.calculateMAP(120, 80);
        System.out.println("Mean Arterial Pressure: " + map + " mmHg");
        
        // Test risk score calculation
        Map<String, Object> patient = new HashMap<>();
        patient.put("patientId", "P0001");
        patient.put("age", 68);
        patient.put("bmi", 31.5);
        patient.put("systolicBP", 145);
        patient.put("diastolicBP", 90);
        List<String> conditions = Arrays.asList("Hypertension", "Diabetes Type 2");
        patient.put("conditions", conditions);
        
        double riskScore = validator.calculatePatientRiskScore(patient);
        System.out.println("Risk Score: " + riskScore);
    }
}
