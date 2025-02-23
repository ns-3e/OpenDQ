from setuptools import setup, find_packages

setup(
    name="open_dq",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "pyspark>=3.2.0",
        "pyodbc>=4.0.30",
        "JayDeBeApi>=1.2.3",
        "scikit-learn>=1.0.0",
        "pytest>=6.0.0",
        "python-dotenv>=0.19.0",
    ],
    author="Nick Smith",
    author_email="nsmith@skailr.io",
    description="A high-performance data quality library for Python supporting ODBC, JDBC, and Apache Spark",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ns-3e/OpenDQ",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 