# java_bridge.py
import jpype
import jpype.imports
from jpype.types import *

class JavaDataValidator:
    def __init__(self, jar_path: str = None):
        """Initialize Java bridge"""
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=['./java_integration'])
        
        # Import Java classes
        self.DataValidator = jpype.JClass('DataValidator')
        self.validator = self.DataValidator()
    
    def validate_vitals(self, heart_rate: int, systolic: int, 
                       diastolic: int, temp: float, o2: int) -> dict:
        """Validate vital signs using Java validator"""
        result = self.validator.validateVitalSigns(
            heart_rate, systolic, diastolic, temp, o2
        )
        
        return {
            'is_valid': result.isValid,
            'errors': list(result.errors)
        }
    
    def classify_bp(self, systolic: int, diastolic: int) -> str:
        """Classify blood pressure using Java method"""
        return self.validator.classifyBloodPressure(systolic, diastolic)
    
    def shutdown(self):
        """Shutdown JVM"""
        if jpype.isJVMStarted():
            jpype.shutdownJVM()
