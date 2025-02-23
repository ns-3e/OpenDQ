from typing import Any, Dict, List, Optional, Union, Pattern
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime

class TestSeverity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class TestResult:
    def __init__(self, passed: bool, message: str, severity: TestSeverity, details: Optional[Dict[str, Any]] = None):
        self.passed = passed
        self.message = message
        self.severity = severity
        self.details = details or {}
        self.timestamp = pd.Timestamp.now()

@dataclass
class TestCase:
    name: str
    description: str
    severity: TestSeverity
    
    def execute(self, df: pd.DataFrame) -> TestResult:
        raise NotImplementedError("Test case must implement execute method")

class NullCheck(TestCase):
    def __init__(self, column: str, threshold: float = 0.1, severity: TestSeverity = TestSeverity.HIGH):
        super().__init__(
            name=f"Null Check - {column}",
            description=f"Check for null values in column {column}",
            severity=severity
        )
        self.column = column
        self.threshold = threshold

    def execute(self, df: pd.DataFrame) -> TestResult:
        null_count = df[self.column].isnull().sum()
        null_ratio = null_count / len(df)
        
        passed = null_ratio <= self.threshold
        message = f"Column {self.column} has {null_count} null values ({null_ratio:.2%})"
        details = {
            "null_count": int(null_count),
            "null_ratio": float(null_ratio),
            "threshold": self.threshold
        }
        
        return TestResult(passed, message, self.severity, details)

class OutlierCheck(TestCase):
    def __init__(self, column: str, n_std: float = 3.0, severity: TestSeverity = TestSeverity.MEDIUM):
        super().__init__(
            name=f"Outlier Check - {column}",
            description=f"Check for outliers in column {column} using {n_std} standard deviations",
            severity=severity
        )
        self.column = column
        self.n_std = n_std

    def execute(self, df: pd.DataFrame) -> TestResult:
        series = df[self.column]
        mean = series.mean()
        std = series.std()
        
        lower_bound = mean - (self.n_std * std)
        upper_bound = mean + (self.n_std * std)
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_count = len(outliers)
        
        passed = outlier_count == 0
        message = f"Found {outlier_count} outliers in column {self.column}"
        details = {
            "outlier_count": outlier_count,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_indices": outliers.index.tolist()
        }
        
        return TestResult(passed, message, self.severity, details)

class UniquenessCheck(TestCase):
    def __init__(self, column: str, threshold: float = 1.0, severity: TestSeverity = TestSeverity.HIGH):
        super().__init__(
            name=f"Uniqueness Check - {column}",
            description=f"Check uniqueness of values in column {column}",
            severity=severity
        )
        self.column = column
        self.threshold = threshold

    def execute(self, df: pd.DataFrame) -> TestResult:
        unique_ratio = df[self.column].nunique() / len(df)
        passed = unique_ratio >= self.threshold
        
        message = f"Column {self.column} has {unique_ratio:.2%} unique values"
        details = {
            "unique_ratio": float(unique_ratio),
            "threshold": self.threshold,
            "unique_count": int(df[self.column].nunique())
        }
        
        return TestResult(passed, message, self.severity, details)

class RangeCheck(TestCase):
    def __init__(
        self, 
        column: str, 
        min_value: Optional[Union[int, float, datetime]] = None,
        max_value: Optional[Union[int, float, datetime]] = None,
        inclusive: bool = True,
        severity: TestSeverity = TestSeverity.HIGH
    ):
        super().__init__(
            name=f"Range Check - {column}",
            description=f"Check if values in {column} are between {min_value} and {max_value}",
            severity=severity
        )
        self.column = column
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def execute(self, df: pd.DataFrame) -> TestResult:
        series = df[self.column]
        violations = pd.Series(False, index=series.index)
        
        if self.min_value is not None:
            if self.inclusive:
                violations |= series < self.min_value
            else:
                violations |= series <= self.min_value
                
        if self.max_value is not None:
            if self.inclusive:
                violations |= series > self.max_value
            else:
                violations |= series >= self.max_value
        
        violation_count = violations.sum()
        passed = violation_count == 0
        
        message = f"Found {violation_count} values outside range in column {self.column}"
        details = {
            "violation_count": int(violation_count),
            "violation_indices": violations[violations].index.tolist(),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "inclusive": self.inclusive
        }
        
        return TestResult(passed, message, self.severity, details)

class ValueCheck(TestCase):
    def __init__(
        self,
        column: str,
        comparison: str,
        value: Any,
        severity: TestSeverity = TestSeverity.HIGH
    ):
        super().__init__(
            name=f"Value Check - {column} {comparison} {value}",
            description=f"Check if values in {column} are {comparison} {value}",
            severity=severity
        )
        self.column = column
        self.comparison = comparison
        self.value = value
        
        self._comparisons = {
            ">": lambda x: x > value,
            ">=": lambda x: x >= value,
            "<": lambda x: x < value,
            "<=": lambda x: x <= value,
            "==": lambda x: x == value,
            "!=": lambda x: x != value
        }
        
        if comparison not in self._comparisons:
            raise ValueError(f"Invalid comparison operator: {comparison}")

    def execute(self, df: pd.DataFrame) -> TestResult:
        series = df[self.column]
        compare_func = self._comparisons[self.comparison]
        violations = ~series.apply(compare_func)
        
        violation_count = violations.sum()
        passed = violation_count == 0
        
        message = f"Found {violation_count} values failing comparison {self.comparison} {self.value} in column {self.column}"
        details = {
            "violation_count": int(violation_count),
            "violation_indices": violations[violations].index.tolist(),
            "comparison": self.comparison,
            "value": self.value
        }
        
        return TestResult(passed, message, self.severity, details)

class ContainsCheck(TestCase):
    def __init__(
        self,
        column: str,
        values: List[Any],
        match_type: str = "any",  # "any" or "all"
        severity: TestSeverity = TestSeverity.MEDIUM
    ):
        super().__init__(
            name=f"Contains Check - {column}",
            description=f"Check if values in {column} contain {match_type} of {values}",
            severity=severity
        )
        self.column = column
        self.values = values
        self.match_type = match_type

    def execute(self, df: pd.DataFrame) -> TestResult:
        series = df[self.column]
        
        if self.match_type == "any":
            violations = ~series.apply(lambda x: any(val in str(x) for val in self.values))
        else:  # all
            violations = ~series.apply(lambda x: all(val in str(x) for val in self.values))
        
        violation_count = violations.sum()
        passed = violation_count == 0
        
        message = f"Found {violation_count} values not containing {self.match_type} of {self.values} in column {self.column}"
        details = {
            "violation_count": int(violation_count),
            "violation_indices": violations[violations].index.tolist(),
            "values": self.values,
            "match_type": self.match_type
        }
        
        return TestResult(passed, message, self.severity, details)

class RegexCheck(TestCase):
    def __init__(
        self,
        column: str,
        pattern: Union[str, Pattern],
        match_type: str = "full",  # "full" or "partial"
        severity: TestSeverity = TestSeverity.HIGH
    ):
        super().__init__(
            name=f"Regex Check - {column}",
            description=f"Check if values in {column} match pattern {pattern}",
            severity=severity
        )
        self.column = column
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.match_type = match_type

    def execute(self, df: pd.DataFrame) -> TestResult:
        series = df[self.column]
        
        if self.match_type == "full":
            violations = ~series.apply(lambda x: bool(self.pattern.fullmatch(str(x))))
        else:  # partial
            violations = ~series.apply(lambda x: bool(self.pattern.search(str(x))))
        
        violation_count = violations.sum()
        passed = violation_count == 0
        
        message = f"Found {violation_count} values not matching pattern in column {self.column}"
        details = {
            "violation_count": int(violation_count),
            "violation_indices": violations[violations].index.tolist(),
            "pattern": self.pattern.pattern,
            "match_type": self.match_type
        }
        
        return TestResult(passed, message, self.severity, details)

class CategoryCheck(TestCase):
    def __init__(
        self,
        column: str,
        allowed_values: List[Any],
        severity: TestSeverity = TestSeverity.HIGH
    ):
        super().__init__(
            name=f"Category Check - {column}",
            description=f"Check if values in {column} are within allowed categories {allowed_values}",
            severity=severity
        )
        self.column = column
        self.allowed_values = set(allowed_values)

    def execute(self, df: pd.DataFrame) -> TestResult:
        series = df[self.column]
        violations = ~series.isin(self.allowed_values)
        
        violation_count = violations.sum()
        invalid_values = set(series[violations].unique())
        passed = violation_count == 0
        
        message = f"Found {violation_count} values not in allowed categories in column {self.column}"
        details = {
            "violation_count": int(violation_count),
            "violation_indices": violations[violations].index.tolist(),
            "invalid_values": list(invalid_values),
            "allowed_values": list(self.allowed_values)
        }
        
        return TestResult(passed, message, self.severity, details)

class DataQualityTestSuite:
    def __init__(self, test_cases: List[TestCase]):
        self.test_cases = test_cases
        
    def run_tests(self, df: pd.DataFrame) -> List[TestResult]:
        results = []
        for test_case in self.test_cases:
            try:
                result = test_case.execute(df)
                results.append(result)
            except Exception as e:
                results.append(
                    TestResult(
                        passed=False,
                        message=f"Test case {test_case.name} failed with error: {str(e)}",
                        severity=test_case.severity,
                        details={"error": str(e)}
                    )
                )
        return results

    def generate_report(self, results: List[TestResult]) -> Dict[str, Any]:
        return {
            "summary": {
                "total_tests": len(results),
                "passed_tests": sum(1 for r in results if r.passed),
                "failed_tests": sum(1 for r in results if not r.passed),
            },
            "results": [
                {
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity.value,
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in results
            ]
        } 