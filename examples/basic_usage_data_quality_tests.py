import pandas as pd
import numpy as np
from datetime import datetime
from open_dq.quality.test_cases import (
    DataQualityTestSuite,
    NullCheck,
    OutlierCheck,
    UniquenessCheck,
    RangeCheck,
    ValueCheck,
    ContainsCheck,
    RegexCheck,
    CategoryCheck,
    TestSeverity
)

def main():
    # Create sample dataset
    data = {
        'id': range(1, 101),
        'value': [i + np.random.normal(0, 5) for i in range(1, 101)],
        'category': ['A'] * 30 + ['B'] * 30 + ['C'] * 30 + ['D'] * 10,  # D is invalid
        'email': [f'user{i}@example.com' if i % 10 != 0 else f'invalid-email-{i}' for i in range(1, 101)],
        'date': [datetime(2024, 1, i) if i <= 31 else datetime(2024, 2, i-31) for i in range(1, 101)],
        'text': [f'Sample text {i}' for i in range(1, 101)]
    }
    df = pd.DataFrame(data)

    # Create test cases
    test_cases = [
        # Basic checks
        NullCheck(
            column='category',
            threshold=0.05,
            severity=TestSeverity.HIGH
        ),
        
        OutlierCheck(
            column='value',
            n_std=3.0,
            severity=TestSeverity.MEDIUM
        ),
        
        UniquenessCheck(
            column='id',
            threshold=1.0,
            severity=TestSeverity.CRITICAL
        ),
        
        # Range checks
        RangeCheck(
            column='value',
            min_value=0,
            max_value=150,
            inclusive=True,
            severity=TestSeverity.HIGH
        ),
        
        RangeCheck(
            column='date',
            min_value=datetime(2024, 1, 1),
            max_value=datetime(2024, 2, 29),
            severity=TestSeverity.MEDIUM
        ),
        
        # Value comparison
        ValueCheck(
            column='value',
            comparison='>=',
            value=0,
            severity=TestSeverity.HIGH
        ),
        
        # Category validation
        CategoryCheck(
            column='category',
            allowed_values=['A', 'B', 'C'],
            severity=TestSeverity.HIGH
        ),
        
        # Text pattern checks
        ContainsCheck(
            column='text',
            values=['Sample', 'text'],
            match_type='all',
            severity=TestSeverity.LOW
        ),
        
        # Email validation
        RegexCheck(
            column='email',
            pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            match_type='full',
            severity=TestSeverity.HIGH
        )
    ]

    # Create and run test suite
    test_suite = DataQualityTestSuite(test_cases)
    results = test_suite.run_tests(df)

    # Generate and print report
    report = test_suite.generate_report(results)
    
    print("\nData Quality Test Results:")
    print("=" * 50)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed Tests: {report['summary']['passed_tests']}")
    print(f"Failed Tests: {report['summary']['failed_tests']}")
    print("\nDetailed Results:")
    print("=" * 50)
    
    for result in report['results']:
        status = "✓" if result['passed'] else "✗"
        print(f"\n{status} Severity: {result['severity']}")
        print(f"Message: {result['message']}")
        if not result['passed']:
            print("Details:", result['details'])

if __name__ == "__main__":
    main() 