from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import pandas as pd

@dataclass
class Patient:
    patient_id: str
    age: int
    gender: str
    blood_type: str
    height_cm: float
    weight_kg: float
    medical_conditions: List[str]
    
    @property
    def bmi(self) -> float:
        return self.weight_kg / ((self.height_cm / 100) ** 2)
    
    @property
    def bmi_category(self) -> str:
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
    patient_id: str
    timestamp: datetime
    heart_rate: int
    systolic_bp: int
    diastolic_bp: int
    temperature: float
    oxygen_saturation: int
    respiratory_rate: int
    
    @property
    def bp_category(self) -> str:
        if self.systolic_bp < 120 and self.diastolic_bp < 80:
            return "Normal"
        elif 120 <= self.systolic_bp < 130 and self.diastolic_bp < 80:
            return "Elevated"
        elif 130 <= self.systolic_bp < 140 or 80 <= self.diastolic_bp < 90:
            return "High BP Stage 1"
        else:
            return "High BP Stage 2"

@dataclass
class LabResult:
    patient_id: str
    test_date: datetime
    test_name: str
    result_value: float
    unit: str
    reference_range_low: float
    reference_range_high: float
    
    @property
    def is_abnormal(self) -> bool:
        return self.result_value < self.reference_range_low or \
               self.result_value > self.reference_range_high
