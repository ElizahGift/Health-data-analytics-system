from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import pandas as pd

@dataclass
class Patient:
    """Patient data model with validation"""
    patient_id: str
    age: int
    gender: str
    blood_type: str
    height_cm: float
    weight_kg: float
    medical_conditions: List[str]
    
    def __post_init__(self):
        """Validate patient data after initialization"""
        if self.age < 0 or self.age > 150:
            raise ValueError(f"Invalid age: {self.age}")
        if self.gender not in ['M', 'F']:
            raise ValueError(f"Invalid gender: {self.gender}")
        if self.height_cm <= 0 or self.height_cm > 300:
            raise ValueError(f"Invalid height: {self.height_cm} cm")
        if self.weight_kg <= 0 or self.weight_kg > 500:
            raise ValueError(f"Invalid weight: {self.weight_kg} kg")
    
    @property
    def bmi(self) -> float:
        """Calculate Body Mass Index"""
        return self.weight_kg / ((self.height_cm / 100) ** 2)
    
    @property
    def bmi_category(self) -> str:
        """Categorize BMI according to WHO standards"""
        bmi = self.bmi
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 25:
            return "Normal"
        elif 25 <= bmi < 30:
            return "Overweight"
        else:
            return "Obese"

@dataclass
class VitalSigns:
    """Vital signs data model with validation"""
    patient_id: str
    timestamp: datetime
    heart_rate: int
    systolic_bp: int
    diastolic_bp: int
    temperature: float
    oxygen_saturation: int
    respiratory_rate: int
    
    def __post_init__(self):
        """Validate vital signs"""
        if self.heart_rate < 30 or self.heart_rate > 250:
            raise ValueError(f"Invalid heart rate: {self.heart_rate}")
        if self.systolic_bp < 50 or self.systolic_bp > 300:
            raise ValueError(f"Invalid systolic BP: {self.systolic_bp}")
        if self.diastolic_bp < 30 or self.diastolic_bp > 200:
            raise ValueError(f"Invalid diastolic BP: {self.diastolic_bp}")
        if self.systolic_bp <= self.diastolic_bp:
            raise ValueError(f"Systolic ({self.systolic_bp}) must be > diastolic ({self.diastolic_bp})")
        if self.temperature < 34 or self.temperature > 43:
            raise ValueError(f"Invalid temperature: {self.temperature}")
        if self.oxygen_saturation < 50 or self.oxygen_saturation > 100:
            raise ValueError(f"Invalid oxygen saturation: {self.oxygen_saturation}")
        if self.respiratory_rate < 5 or self.respiratory_rate > 50:
            raise ValueError(f"Invalid respiratory rate: {self.respiratory_rate}")
    
    @property
    def bp_category(self) -> str:
        """Classify blood pressure according to JNC 8 guidelines"""
        if self.systolic_bp < 120 and self.diastolic_bp < 80:
            return "Normal"
        elif 120 <= self.systolic_bp < 130 and self.diastolic_bp < 80:
            return "Elevated"
        elif 130 <= self.systolic_bp < 140 or 80 <= self.diastolic_bp < 90:
            return "High BP Stage 1"
        elif self.systolic_bp >= 140 or self.diastolic_bp >= 90:
            return "High BP Stage 2"
        else:
            return "Hypertensive Crisis"
    
    @property
    def map_arterial_pressure(self) -> float:
        """Calculate Mean Arterial Pressure"""
        return self.diastolic_bp + (self.systolic_bp - self.diastolic_bp) / 3.0

@dataclass
class LabResult:
    """Lab result data model"""
    patient_id: str
    test_date: datetime
    test_name: str
    result_value: float
    unit: str
    reference_range_low: float
    reference_range_high: float
    
    def __post_init__(self):
        """Validate lab results"""
        if self.result_value < 0:
            raise ValueError(f"Invalid lab result value: {self.result_value}")
        if self.reference_range_low >= self.reference_range_high:
            raise ValueError(f"Invalid reference range: {self.reference_range_low} > {self.reference_range_high}")
    
    @property
    def is_abnormal(self) -> bool:
        """Check if result is outside reference range"""
        return self.result_value < self.reference_range_low or \
               self.result_value > self.reference_range_high
    
    @property
    def percent_of_normal(self) -> float:
        """Calculate result as percentage of normal range midpoint"""
        midpoint = (self.reference_range_low + self.reference_range_high) / 2
        return (self.result_value / midpoint) * 100 if midpoint > 0 else 0
