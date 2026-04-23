import java.util.*;
import java.util.stream.Collectors;

public class DataValidator {
    
    public static class ValidationResult {
        public boolean isValid;
        public List<String> errors;
        
        public ValidationResult() {
            this.isValid = true;
            this.errors = new ArrayList<>();
        }
    }
    
    public ValidationResult validateVitalSigns(
            int heartRate, 
            int systolicBP, 
            int diastolicBP, 
            double temperature,
            int oxygenSaturation) {
        
        ValidationResult result = new ValidationResult();
        
        // Heart rate validation
        if (heartRate < 40 || heartRate > 200) {
            result.isValid = false;
            result.errors.add("Heart rate out of range: " + heartRate);
        }
        
        // Blood pressure validation
        if (systolicBP < 70 || systolicBP > 250) {
            result.isValid = false;
            result.errors.add("Systolic BP out of range: " + systolicBP);
        }
        
        if (diastolicBP < 40 || diastolicBP > 150) {
            result.isValid = false;
            result.errors.add("Diastolic BP out of range: " + diastolicBP);
        }
        
        if (systolicBP <= diastolicBP) {
            result.isValid = false;
            result.errors.add("Systolic must be greater than diastolic");
        }
        
        // Temperature validation
        if (temperature < 35.0 || temperature > 42.0) {
            result.isValid = false;
            result.errors.add("Temperature out of range: " + temperature);
        }
        
        // Oxygen saturation validation
        if (oxygenSaturation < 70 || oxygenSaturation > 100) {
            result.isValid = false;
            result.errors.add("Oxygen saturation out of range: " + oxygenSaturation);
        }
        
        return result;
    }
    
    public double calculateMAP(int systolicBP, int diastolicBP) {
        // Mean Arterial Pressure
        return diastolicBP + ((systolicBP - diastolicBP) / 3.0);
    }
    
    public String classifyBloodPressure(int systolicBP, int diastolicBP) {
        if (systolicBP < 120 && diastolicBP < 80) {
            return "Normal";
        } else if (systolicBP < 130 && diastolicBP < 80) {
            return "Elevated";
        } else if (systolicBP < 140 || diastolicBP < 90) {
            return "High Blood Pressure Stage 1";
        } else if (systolicBP >= 140 || diastolicBP >= 90) {
            return "High Blood Pressure Stage 2";
        } else {
            return "Hypertensive Crisis";
        }
    }
    
    public Map<String, Double> calculateRiskScores(List<Map<String, Object>> patientData) {
        Map<String, Double> riskScores = new HashMap<>();
        
        for (Map<String, Object> data : patientData) {
            String patientId = (String) data.get("patientId");
            double score = 0.0;
            
            // Age risk
            int age = (int) data.get("age");
            if (age > 65) score += 3.0;
            else if (age > 50) score += 1.5;
            
            // BMI risk
            double bmi = (double) data.get("bmi");
            if (bmi >= 30) score += 2.5;
            else if (bmi >= 25) score += 1.0;
            
            // Blood pressure risk
            int systolic = (int) data.get("systolicBP");
            int diastolic = (int) data.get("diastolicBP");
            if (systolic >= 140 || diastolic >= 90) score += 2.0;
            
            riskScores.put(patientId, score);
        }
        
        return riskScores;
    }
}
