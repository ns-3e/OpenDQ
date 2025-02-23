import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional, Any
import json
from pathlib import Path
import jinja2
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from .stats import BasicStats
from .distribution import DistributionAnalyzer
from .correlation import CorrelationAnalyzer
from .patterns import PatternDetector

class DataProfiler:
    """Main class for data profiling that orchestrates all profiling functionality."""
    
    def __init__(
        self, 
        data: Union[pd.DataFrame, 'pyspark.sql.DataFrame'],
        sample_size: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Initialize the DataProfiler with a dataset.
        
        Args:
            data: Input dataset as pandas DataFrame or Spark DataFrame
            sample_size: Number of rows to sample for profiling. If None, use entire dataset.
            random_seed: Random seed for reproducible sampling
        """
        self.original_data = data
        self._is_spark = not isinstance(data, pd.DataFrame)
        
        # Sample data if requested
        if sample_size is not None:
            if self._is_spark:
                total_count = data.count()
                if sample_size < total_count:
                    self.data = data.sample(False, sample_size/total_count, seed=random_seed)
                else:
                    self.data = data
            else:
                if sample_size < len(data):
                    self.data = data.sample(n=sample_size, random_state=random_seed)
                else:
                    self.data = data
        else:
            self.data = data
        
        # Initialize analyzers
        self.basic_stats = BasicStats(self.data)
        self.distribution = DistributionAnalyzer(self.data)
        self.correlation = CorrelationAnalyzer(self.data)
        self.patterns = PatternDetector(self.data)
        
        # Store profiling metadata
        self.profile_metadata = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': sample_size if sample_size is not None else 'full',
            'total_rows': len(data) if not self._is_spark else data.count(),
            'total_columns': len(data.columns),
            'is_spark': self._is_spark
        }
        
    def generate_profile(
        self, 
        include: Optional[List[str]] = None,
        correlation_threshold: float = 0.7,
        correlation_method: str = 'pearson'
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive profile of the dataset.
        
        Args:
            include: List of profile sections to include. If None, includes all.
                    Options: ['basic_stats', 'distribution', 'correlation', 'patterns']
            correlation_threshold: Threshold for identifying strong correlations
            correlation_method: Method for computing correlations ('pearson', 'spearman', or 'kendall')
        
        Returns:
            Dictionary containing all profiling results and metadata
        """
        if include is None:
            include = ['basic_stats', 'distribution', 'correlation', 'patterns']
            
        profile = {'metadata': self.profile_metadata}
        
        if 'basic_stats' in include:
            profile['basic_stats'] = self.basic_stats.compute_stats()
            
        if 'distribution' in include:
            profile['distribution'] = self.distribution.analyze()
            
        if 'correlation' in include:
            profile['correlation'] = self.correlation.analyze(
                method=correlation_method,
                threshold=correlation_threshold
            )
            
        if 'patterns' in include:
            profile['patterns'] = self.patterns.detect()
            
        return profile
    
    def to_json(self, filepath: str, pretty: bool = True):
        """
        Save profiling results to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
            pretty: Whether to format the JSON with indentation
        """
        profile = self.generate_profile()
        with open(filepath, 'w') as f:
            json.dump(profile, f, indent=2 if pretty else None)
    
    def to_html(
        self, 
        filepath: str,
        template_path: Optional[str] = None,
        include_plots: bool = True,
        plot_format: str = 'div'
    ):
        """
        Generate an HTML report of the profiling results.
        
        Args:
            filepath: Path to save the HTML report
            template_path: Path to custom Jinja2 template. If None, use default template.
            include_plots: Whether to include interactive plots in the report
            plot_format: Format for plots ('div' or 'json')
        """
        profile = self.generate_profile()
        
        if include_plots:
            plots = self._generate_plots(profile)
            profile['plots'] = plots
        
        # Load template
        if template_path is None:
            template_str = self._get_default_template()
            template = jinja2.Template(template_str)
        else:
            with open(template_path, 'r') as f:
                template = jinja2.Template(f.read())
        
        # Render HTML
        html = template.render(
            profile=profile,
            include_plots=include_plots,
            plot_format=plot_format
        )
        
        # Save to file
        with open(filepath, 'w') as f:
            f.write(html)
    
    def _generate_plots(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate interactive plots for the profile results."""
        plots = {}
        
        # Distribution plots
        if 'distribution' in profile:
            dist_plots = {}
            for column, dist in profile['distribution'].items():
                if 'histogram' in dist:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=dist['histogram']['bin_edges'][:-1],
                            y=dist['histogram']['counts'],
                            name='Histogram'
                        )
                    ])
                    fig.update_layout(
                        title=f'Distribution of {column}',
                        xaxis_title=column,
                        yaxis_title='Count'
                    )
                    dist_plots[column] = fig.to_html(full_html=False)
            plots['distribution'] = dist_plots
        
        # Correlation heatmap
        if 'correlation' in profile and 'correlation_matrix' in profile['correlation']:
            corr_data = profile['correlation']['correlation_matrix']
            if corr_data and 'values' in corr_data:
                fig = go.Figure(data=[
                    go.Heatmap(
                        z=corr_data['values'],
                        x=corr_data['columns'],
                        y=corr_data['columns'],
                        colorscale='RdBu',
                        zmid=0
                    )
                ])
                fig.update_layout(
                    title='Correlation Matrix',
                    xaxis_title='Features',
                    yaxis_title='Features'
                )
                plots['correlation_heatmap'] = fig.to_html(full_html=False)
        
        # Missing values plot
        missing_data = []
        for column, stats in profile['basic_stats'].items():
            missing_data.append({
                'column': column,
                'missing_percentage': stats['missing_percentage']
            })
        
        if missing_data:
            fig = px.bar(
                missing_data,
                x='column',
                y='missing_percentage',
                title='Missing Values by Column'
            )
            plots['missing_values'] = fig.to_html(full_html=False)
        
        return plots
    
    def _get_default_template(self) -> str:
        """Get the default HTML template for the profiling report."""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Data Profiling Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
        .plot { margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f5f5f5; }
    </style>
</head>
<body>
    <h1>Data Profiling Report</h1>
    
    <div class="section">
        <h2>Dataset Overview</h2>
        <table>
            <tr><th>Total Rows</th><td>{{ profile.metadata.total_rows }}</td></tr>
            <tr><th>Total Columns</th><td>{{ profile.metadata.total_columns }}</td></tr>
            <tr><th>Sample Size</th><td>{{ profile.metadata.sample_size }}</td></tr>
            <tr><th>Generated At</th><td>{{ profile.metadata.timestamp }}</td></tr>
        </table>
    </div>
    
    {% if profile.basic_stats %}
    <div class="section">
        <h2>Basic Statistics</h2>
        {% for column, stats in profile.basic_stats.items() %}
        <h3>{{ column }}</h3>
        <table>
            {% for stat, value in stats.items() %}
            <tr><th>{{ stat }}</th><td>{{ value }}</td></tr>
            {% endfor %}
        </table>
        {% endfor %}
    </div>
    {% endif %}
    
    {% if include_plots and profile.plots %}
    <div class="section">
        <h2>Visualizations</h2>
        {% if profile.plots.missing_values %}
        <div class="plot">
            <h3>Missing Values</h3>
            {{ profile.plots.missing_values | safe }}
        </div>
        {% endif %}
        
        {% if profile.plots.correlation_heatmap %}
        <div class="plot">
            <h3>Correlation Heatmap</h3>
            {{ profile.plots.correlation_heatmap | safe }}
        </div>
        {% endif %}
        
        {% if profile.plots.distribution %}
        <div class="plot">
            <h3>Distributions</h3>
            {% for column, plot in profile.plots.distribution.items() %}
            <h4>{{ column }}</h4>
            {{ plot | safe }}
            {% endfor %}
        </div>
        {% endif %}
    </div>
    {% endif %}
    
    {% if profile.patterns %}
    <div class="section">
        <h2>Pattern Detection</h2>
        {% for column, patterns in profile.patterns.items() %}
        <h3>{{ column }}</h3>
        <table>
            {% for pattern_type, pattern_data in patterns.items() %}
            <tr>
                <th>{{ pattern_type }}</th>
                <td><pre>{{ pattern_data | tojson(indent=2) }}</pre></td>
            </tr>
            {% endfor %}
        </table>
        {% endfor %}
    </div>
    {% endif %}
</body>
</html>
''' 