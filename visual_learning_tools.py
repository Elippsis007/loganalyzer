"""
Visual Learning Tools for Log Analysis Education
Provides enhanced visualizations with educational annotations and interactive learning components
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

@dataclass
class VisualAnnotation:
    """Annotation for educational visualizations"""
    x: float
    y: float
    text: str
    annotation_type: str  # 'pattern', 'warning', 'insight', 'tip'
    color: str = 'blue'
    arrow: bool = True

@dataclass
class PatternVisualization:
    """Visual representation of a log pattern"""
    pattern_name: str
    description: str
    chart_type: str  # 'timeline', 'scatter', 'bar', 'heatmap'
    sample_data: List[Dict[str, Any]]
    annotations: List[VisualAnnotation]
    learning_points: List[str]
    what_to_look_for: List[str]

class VisualLearningTools:
    """Enhanced visualizations with educational features"""
    
    def __init__(self):
        self.pattern_library = self._load_pattern_library()
        self.color_schemes = self._load_color_schemes()
        self.chart_templates = self._load_chart_templates()
    
    def create_educational_timeline(self, log_entries: List[Any], annotations: List[VisualAnnotation] = None) -> go.Figure:
        """Create an educational timeline with annotations"""
        if not log_entries:
            return self._create_empty_chart("No data available for timeline")
        
        # Extract data for timeline
        timeline_data = []
        for i, entry in enumerate(log_entries):
            try:
                timestamp = getattr(entry, 'timestamp', datetime.now())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                level = getattr(entry, 'level', 'INFO')
                message = getattr(entry, 'message', '') or getattr(entry, 'content', '')
                
                timeline_data.append({
                    'timestamp': timestamp,
                    'level': level,
                    'message': message,
                    'y_position': self._get_level_position(level),
                    'color': self._get_level_color(level),
                    'size': self._get_level_size(level),
                    'index': i
                })
            except Exception:
                continue
        
        if not timeline_data:
            return self._create_empty_chart("No valid timeline data")
        
        df = pd.DataFrame(timeline_data)
        
        # Create scatter plot for timeline
        fig = go.Figure()
        
        # Add traces for each log level
        for level in ['ERROR', 'WARN', 'INFO', 'DEBUG']:
            level_data = df[df['level'] == level]
            if not level_data.empty:
                fig.add_trace(go.Scatter(
                    x=level_data['timestamp'],
                    y=level_data['y_position'],
                    mode='markers',
                    marker=dict(
                        size=level_data['size'],
                        color=level_data['color'],
                        line=dict(width=2, color='white')
                    ),
                    text=level_data['message'],
                    hovertemplate='<b>%{text}</b><br>Time: %{x}<br>Level: ' + level + '<extra></extra>',
                    name=level,
                    showlegend=True
                ))
        
        # Add educational annotations
        if annotations:
            for annotation in annotations:
                fig.add_annotation(
                    x=annotation.x,
                    y=annotation.y,
                    text=annotation.text,
                    showarrow=annotation.arrow,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=annotation.color,
                    ax=0,
                    ay=-40,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor=annotation.color,
                    borderwidth=1,
                    font=dict(size=12, color=annotation.color)
                )
        
        # Add educational information
        fig.update_layout(
            title={
                'text': '📚 Educational Timeline: Understanding Log Patterns',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title='Time',
            yaxis_title='Log Level',
            yaxis=dict(
                tickvals=[1, 2, 3, 4],
                ticktext=['DEBUG', 'INFO', 'WARN', 'ERROR']
            ),
            hovermode='closest',
            showlegend=True,
            height=600,
            annotations=[
                dict(
                    text="💡 Learning Tip: Look for patterns in the timeline - clusters of errors often indicate cascading failures",
                    xref="paper", yref="paper",
                    x=0.5, y=1.1,
                    showarrow=False,
                    font=dict(size=14, color='blue'),
                    bgcolor='rgba(173,216,230,0.3)',
                    bordercolor='blue',
                    borderwidth=1
                )
            ]
        )
        
        return fig
    
    def create_system_health_narrative(self, log_entries: List[Any], time_window: str = "1h") -> go.Figure:
        """Create a system health narrative visualization"""
        if not log_entries:
            return self._create_empty_chart("No data available for system health narrative")
        
        # Process log entries to create health metrics
        health_data = self._process_health_data(log_entries, time_window)
        
        # Create subplot with multiple metrics
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('System Health Score', 'Error Rate', 'Activity Level'),
            shared_xaxes=True,
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Health score
        fig.add_trace(
            go.Scatter(
                x=health_data['timestamps'],
                y=health_data['health_scores'],
                mode='lines+markers',
                name='Health Score',
                line=dict(color='green', width=3),
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.1)'
            ),
            row=1, col=1
        )
        
        # Error rate
        fig.add_trace(
            go.Scatter(
                x=health_data['timestamps'],
                y=health_data['error_rates'],
                mode='lines+markers',
                name='Error Rate',
                line=dict(color='red', width=2),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)'
            ),
            row=2, col=1
        )
        
        # Activity level
        fig.add_trace(
            go.Scatter(
                x=health_data['timestamps'],
                y=health_data['activity_levels'],
                mode='lines+markers',
                name='Activity Level',
                line=dict(color='blue', width=2),
                fill='tonexty',
                fillcolor='rgba(0,0,255,0.1)'
            ),
            row=3, col=1
        )
        
        # Add narrative annotations
        narrative_annotations = self._generate_health_narrative(health_data)
        for annotation in narrative_annotations:
            fig.add_annotation(
                x=annotation.x,
                y=annotation.y,
                text=annotation.text,
                showarrow=True,
                arrowhead=2,
                arrowcolor=annotation.color,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor=annotation.color,
                borderwidth=1,
                font=dict(size=10, color=annotation.color),
                row=1, col=1
            )
        
        fig.update_layout(
            title={
                'text': '📊 System Health Narrative: Your System\'s Story',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_pattern_library_visualization(self, pattern_name: str) -> go.Figure:
        """Create visualization for a specific pattern from the library"""
        if pattern_name not in self.pattern_library:
            return self._create_empty_chart(f"Pattern '{pattern_name}' not found in library")
        
        pattern = self.pattern_library[pattern_name]
        
        if pattern.chart_type == 'timeline':
            return self._create_pattern_timeline(pattern)
        elif pattern.chart_type == 'scatter':
            return self._create_pattern_scatter(pattern)
        elif pattern.chart_type == 'bar':
            return self._create_pattern_bar(pattern)
        elif pattern.chart_type == 'heatmap':
            return self._create_pattern_heatmap(pattern)
        else:
            return self._create_empty_chart(f"Unknown chart type: {pattern.chart_type}")
    
    def create_learning_dashboard(self, log_entries: List[Any]) -> Dict[str, go.Figure]:
        """Create a comprehensive learning dashboard"""
        dashboard = {}
        
        # 1. Pattern Recognition Chart
        dashboard['pattern_recognition'] = self._create_pattern_recognition_chart(log_entries)
        
        # 2. Severity Distribution
        dashboard['severity_distribution'] = self._create_severity_pie_chart(log_entries)
        
        # 3. Timeline with Annotations
        dashboard['annotated_timeline'] = self.create_educational_timeline(log_entries)
        
        # 4. System Health Narrative
        dashboard['health_narrative'] = self.create_system_health_narrative(log_entries)
        
        # 5. Investigation Guide
        dashboard['investigation_guide'] = self._create_investigation_guide_chart(log_entries)
        
        return dashboard
    
    def _load_pattern_library(self) -> Dict[str, PatternVisualization]:
        """Load the visual pattern library"""
        return {
            'error_cascade': PatternVisualization(
                pattern_name='Error Cascade',
                description='Cascading failures across system components',
                chart_type='timeline',
                sample_data=[
                    {'timestamp': '10:00:00', 'level': 'ERROR', 'component': 'Database', 'message': 'Connection timeout'},
                    {'timestamp': '10:00:05', 'level': 'ERROR', 'component': 'API', 'message': 'Database unavailable'},
                    {'timestamp': '10:00:10', 'level': 'ERROR', 'component': 'Frontend', 'message': 'API request failed'},
                    {'timestamp': '10:00:15', 'level': 'ERROR', 'component': 'LoadBalancer', 'message': 'Backend unhealthy'}
                ],
                annotations=[
                    VisualAnnotation(
                        x=1, y=4,
                        text="Initial failure: Database connection timeout",
                        annotation_type='pattern',
                        color='red'
                    ),
                    VisualAnnotation(
                        x=2, y=4,
                        text="Cascade begins: API cannot reach database",
                        annotation_type='pattern',
                        color='orange'
                    ),
                    VisualAnnotation(
                        x=3, y=4,
                        text="Frontend affected: Cannot get data from API",
                        annotation_type='pattern',
                        color='orange'
                    )
                ],
                learning_points=[
                    "Cascades typically start with a single point of failure",
                    "Errors propagate through system dependencies",
                    "Time intervals between cascading errors are often very short"
                ],
                what_to_look_for=[
                    "Rapid succession of errors",
                    "Errors following dependency chains",
                    "Similar error messages across components"
                ]
            ),
            
            'performance_degradation': PatternVisualization(
                pattern_name='Performance Degradation',
                description='Gradual decline in system performance',
                chart_type='scatter',
                sample_data=[
                    {'timestamp': '09:00:00', 'response_time': 100, 'level': 'INFO'},
                    {'timestamp': '09:30:00', 'response_time': 200, 'level': 'INFO'},
                    {'timestamp': '10:00:00', 'response_time': 350, 'level': 'WARN'},
                    {'timestamp': '10:30:00', 'response_time': 500, 'level': 'WARN'},
                    {'timestamp': '11:00:00', 'response_time': 800, 'level': 'ERROR'}
                ],
                annotations=[
                    VisualAnnotation(
                        x=1, y=100,
                        text="Normal performance baseline",
                        annotation_type='insight',
                        color='green'
                    ),
                    VisualAnnotation(
                        x=3, y=350,
                        text="Performance threshold breached",
                        annotation_type='warning',
                        color='orange'
                    ),
                    VisualAnnotation(
                        x=5, y=800,
                        text="Critical performance degradation",
                        annotation_type='pattern',
                        color='red'
                    )
                ],
                learning_points=[
                    "Performance degradation is often gradual",
                    "Warning signs appear before critical failures",
                    "Thresholds help identify when intervention is needed"
                ],
                what_to_look_for=[
                    "Increasing response times over time",
                    "Threshold breaches in monitoring",
                    "Pattern of gradual degradation"
                ]
            ),
            
            'memory_leak': PatternVisualization(
                pattern_name='Memory Leak',
                description='Increasing memory usage over time',
                chart_type='bar',
                sample_data=[
                    {'timestamp': '08:00:00', 'memory_mb': 256, 'level': 'INFO'},
                    {'timestamp': '10:00:00', 'memory_mb': 512, 'level': 'INFO'},
                    {'timestamp': '12:00:00', 'memory_mb': 768, 'level': 'WARN'},
                    {'timestamp': '14:00:00', 'memory_mb': 1024, 'level': 'ERROR'},
                    {'timestamp': '16:00:00', 'memory_mb': 1280, 'level': 'ERROR'}
                ],
                annotations=[
                    VisualAnnotation(
                        x=1, y=256,
                        text="Normal memory usage",
                        annotation_type='insight',
                        color='green'
                    ),
                    VisualAnnotation(
                        x=3, y=768,
                        text="Memory usage trending upward",
                        annotation_type='warning',
                        color='orange'
                    ),
                    VisualAnnotation(
                        x=5, y=1280,
                        text="Memory exhaustion imminent",
                        annotation_type='pattern',
                        color='red'
                    )
                ],
                learning_points=[
                    "Memory leaks show consistent upward trends",
                    "Memory usage should typically fluctuate, not continuously increase",
                    "Garbage collection events may temporarily reduce usage"
                ],
                what_to_look_for=[
                    "Steadily increasing memory usage",
                    "Memory usage that doesn't return to baseline",
                    "OutOfMemory errors or warnings"
                ]
            )
        }
    
    def _load_color_schemes(self) -> Dict[str, Dict[str, str]]:
        """Load color schemes for different visualization types"""
        return {
            'log_levels': {
                'DEBUG': '#6c757d',
                'INFO': '#17a2b8',
                'WARN': '#ffc107',
                'ERROR': '#dc3545'
            },
            'health_status': {
                'healthy': '#28a745',
                'warning': '#ffc107',
                'critical': '#dc3545',
                'unknown': '#6c757d'
            },
            'educational': {
                'pattern': '#007bff',
                'warning': '#fd7e14',
                'insight': '#28a745',
                'tip': '#6f42c1'
            }
        }
    
    def _load_chart_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load chart templates for consistent styling"""
        return {
            'educational_timeline': {
                'layout': {
                    'font': {'family': 'Arial, sans-serif', 'size': 12},
                    'plot_bgcolor': 'white',
                    'paper_bgcolor': 'white',
                    'gridcolor': '#e0e0e0',
                    'title_font_size': 16,
                    'showlegend': True,
                    'legend': {'orientation': 'h', 'y': -0.2}
                }
            },
            'pattern_library': {
                'layout': {
                    'font': {'family': 'Arial, sans-serif', 'size': 11},
                    'plot_bgcolor': '#f8f9fa',
                    'paper_bgcolor': 'white',
                    'title_font_size': 14,
                    'showlegend': True
                }
            }
        }
    
    def _get_level_position(self, level: str) -> int:
        """Get Y position for log level"""
        positions = {'DEBUG': 1, 'INFO': 2, 'WARN': 3, 'ERROR': 4}
        return positions.get(level, 2)
    
    def _get_level_color(self, level: str) -> str:
        """Get color for log level"""
        return self.color_schemes['log_levels'].get(level, '#6c757d')
    
    def _get_level_size(self, level: str) -> int:
        """Get marker size for log level"""
        sizes = {'DEBUG': 8, 'INFO': 10, 'WARN': 12, 'ERROR': 14}
        return sizes.get(level, 10)
    
    def _process_health_data(self, log_entries: List[Any], time_window: str) -> Dict[str, List]:
        """Process log entries to create health metrics"""
        if not log_entries:
            return {'timestamps': [], 'health_scores': [], 'error_rates': [], 'activity_levels': []}
        
        # Group entries by time windows
        time_buckets = {}
        for entry in log_entries:
            try:
                timestamp = getattr(entry, 'timestamp', datetime.now())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                # Round to nearest minute for grouping
                bucket = timestamp.replace(second=0, microsecond=0)
                if bucket not in time_buckets:
                    time_buckets[bucket] = {'total': 0, 'errors': 0, 'warnings': 0}
                
                time_buckets[bucket]['total'] += 1
                level = getattr(entry, 'level', 'INFO')
                if level == 'ERROR':
                    time_buckets[bucket]['errors'] += 1
                elif level == 'WARN':
                    time_buckets[bucket]['warnings'] += 1
            except Exception:
                continue
        
        # Calculate metrics
        timestamps = sorted(time_buckets.keys())
        health_scores = []
        error_rates = []
        activity_levels = []
        
        for timestamp in timestamps:
            bucket = time_buckets[timestamp]
            total = bucket['total']
            errors = bucket['errors']
            warnings = bucket['warnings']
            
            # Health score: 100 - (error% * 100) - (warning% * 50)
            if total > 0:
                error_rate = (errors / total) * 100
                warning_rate = (warnings / total) * 100
                health_score = max(0, 100 - error_rate - (warning_rate * 0.5))
            else:
                error_rate = 0
                health_score = 100
            
            health_scores.append(health_score)
            error_rates.append(error_rate)
            activity_levels.append(total)
        
        return {
            'timestamps': timestamps,
            'health_scores': health_scores,
            'error_rates': error_rates,
            'activity_levels': activity_levels
        }
    
    def _generate_health_narrative(self, health_data: Dict[str, List]) -> List[VisualAnnotation]:
        """Generate narrative annotations for health visualization"""
        annotations = []
        
        if not health_data['timestamps']:
            return annotations
        
        health_scores = health_data['health_scores']
        timestamps = health_data['timestamps']
        error_rates = health_data['error_rates']
        
        # Find significant events
        for i, (timestamp, health_score, error_rate) in enumerate(zip(timestamps, health_scores, error_rates)):
            if health_score < 50:
                annotations.append(VisualAnnotation(
                    x=i,
                    y=health_score,
                    text=f"System unhealthy: {health_score:.1f}% health",
                    annotation_type='warning',
                    color='red'
                ))
            elif health_score < 80 and error_rate > 10:
                annotations.append(VisualAnnotation(
                    x=i,
                    y=health_score,
                    text=f"Degraded performance: {error_rate:.1f}% errors",
                    annotation_type='warning',
                    color='orange'
                ))
            elif i == 0:
                annotations.append(VisualAnnotation(
                    x=i,
                    y=health_score,
                    text=f"System baseline: {health_score:.1f}% health",
                    annotation_type='insight',
                    color='green'
                ))
        
        return annotations
    
    def _create_pattern_timeline(self, pattern: PatternVisualization) -> go.Figure:
        """Create timeline visualization for a pattern"""
        fig = go.Figure()
        
        # Add sample data
        for i, data_point in enumerate(pattern.sample_data):
            fig.add_trace(go.Scatter(
                x=[i],
                y=[self._get_level_position(data_point['level'])],
                mode='markers',
                marker=dict(
                    size=15,
                    color=self._get_level_color(data_point['level']),
                    line=dict(width=2, color='white')
                ),
                text=data_point['message'],
                name=data_point.get('component', 'System'),
                showlegend=True
            ))
        
        # Add annotations
        for annotation in pattern.annotations:
            fig.add_annotation(
                x=annotation.x,
                y=annotation.y,
                text=annotation.text,
                showarrow=annotation.arrow,
                arrowcolor=annotation.color,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=annotation.color,
                borderwidth=1,
                font=dict(color=annotation.color)
            )
        
        fig.update_layout(
            title=f"📚 Pattern: {pattern.pattern_name}",
            xaxis_title="Event Sequence",
            yaxis_title="Log Level",
            yaxis=dict(
                tickvals=[1, 2, 3, 4],
                ticktext=['DEBUG', 'INFO', 'WARN', 'ERROR']
            ),
            height=500
        )
        
        return fig
    
    def _create_pattern_scatter(self, pattern: PatternVisualization) -> go.Figure:
        """Create scatter plot for a pattern"""
        fig = go.Figure()
        
        x_values = []
        y_values = []
        colors = []
        
        for i, data_point in enumerate(pattern.sample_data):
            x_values.append(i)
            y_values.append(data_point.get('response_time', 0))
            colors.append(self._get_level_color(data_point['level']))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers+lines',
            marker=dict(
                size=12,
                color=colors,
                line=dict(width=2, color='white')
            ),
            line=dict(width=2, color='blue'),
            name=pattern.pattern_name
        ))
        
        # Add annotations
        for annotation in pattern.annotations:
            fig.add_annotation(
                x=annotation.x,
                y=annotation.y,
                text=annotation.text,
                showarrow=annotation.arrow,
                arrowcolor=annotation.color,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=annotation.color,
                borderwidth=1,
                font=dict(color=annotation.color)
            )
        
        fig.update_layout(
            title=f"📈 Pattern: {pattern.pattern_name}",
            xaxis_title="Time Sequence",
            yaxis_title="Response Time (ms)",
            height=500
        )
        
        return fig
    
    def _create_pattern_bar(self, pattern: PatternVisualization) -> go.Figure:
        """Create bar chart for a pattern"""
        fig = go.Figure()
        
        x_values = [dp.get('timestamp', f"T{i}") for i, dp in enumerate(pattern.sample_data)]
        y_values = [dp.get('memory_mb', 0) for dp in pattern.sample_data]
        colors = [self._get_level_color(dp['level']) for dp in pattern.sample_data]
        
        fig.add_trace(go.Bar(
            x=x_values,
            y=y_values,
            marker=dict(color=colors),
            name=pattern.pattern_name
        ))
        
        # Add annotations
        for annotation in pattern.annotations:
            fig.add_annotation(
                x=annotation.x,
                y=annotation.y,
                text=annotation.text,
                showarrow=annotation.arrow,
                arrowcolor=annotation.color,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=annotation.color,
                borderwidth=1,
                font=dict(color=annotation.color)
            )
        
        fig.update_layout(
            title=f"📊 Pattern: {pattern.pattern_name}",
            xaxis_title="Time",
            yaxis_title="Memory Usage (MB)",
            height=500
        )
        
        return fig
    
    def _create_pattern_heatmap(self, pattern: PatternVisualization) -> go.Figure:
        """Create heatmap for a pattern"""
        # This would create a heatmap visualization
        # For now, return a placeholder
        return self._create_empty_chart(f"Heatmap for {pattern.pattern_name} - Coming Soon")
    
    def _create_pattern_recognition_chart(self, log_entries: List[Any]) -> go.Figure:
        """Create a chart highlighting patterns in the log data"""
        if not log_entries:
            return self._create_empty_chart("No data for pattern recognition")
        
        # Analyze patterns
        patterns = self._analyze_patterns(log_entries)
        
        fig = go.Figure()
        
        for pattern_name, pattern_data in patterns.items():
            fig.add_trace(go.Scatter(
                x=pattern_data['timestamps'],
                y=pattern_data['values'],
                mode='lines+markers',
                name=pattern_name,
                line=dict(width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="🔍 Pattern Recognition in Your Logs",
            xaxis_title="Time",
            yaxis_title="Pattern Intensity",
            height=400
        )
        
        return fig
    
    def _create_severity_pie_chart(self, log_entries: List[Any]) -> go.Figure:
        """Create educational pie chart for severity distribution"""
        if not log_entries:
            return self._create_empty_chart("No data for severity distribution")
        
        # Count severity levels
        severity_counts = {'DEBUG': 0, 'INFO': 0, 'WARN': 0, 'ERROR': 0}
        
        for entry in log_entries:
            level = getattr(entry, 'level', 'INFO')
            if level in severity_counts:
                severity_counts[level] += 1
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(severity_counts.keys()),
            values=list(severity_counts.values()),
            marker=dict(colors=[self.color_schemes['log_levels'][level] for level in severity_counts.keys()])
        )])
        
        fig.update_layout(
            title="📊 Log Severity Distribution",
            annotations=[
                dict(
                    text="💡 Healthy systems typically show mostly INFO logs with few ERRORs",
                    x=0.5, y=-0.1,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=12, color='blue')
                )
            ],
            height=500
        )
        
        return fig
    
    def _create_investigation_guide_chart(self, log_entries: List[Any]) -> go.Figure:
        """Create a visual investigation guide"""
        if not log_entries:
            return self._create_empty_chart("No data for investigation guide")
        
        # Create a flowchart-style visualization
        fig = go.Figure()
        
        # Add investigation steps as a flowchart
        steps = [
            "1. Identify Problem",
            "2. Gather Context",
            "3. Analyze Patterns",
            "4. Find Root Cause",
            "5. Plan Solution"
        ]
        
        for i, step in enumerate(steps):
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode='markers+text',
                marker=dict(size=50, color='lightblue'),
                text=step,
                textposition="middle center",
                name=step,
                showlegend=False
            ))
        
        fig.update_layout(
            title="🔍 Investigation Guide: Step-by-Step Process",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=200
        )
        
        return fig
    
    def _analyze_patterns(self, log_entries: List[Any]) -> Dict[str, Dict[str, List]]:
        """Analyze log entries for patterns"""
        patterns = {}
        
        # Simple pattern analysis
        error_pattern = {'timestamps': [], 'values': []}
        warning_pattern = {'timestamps': [], 'values': []}
        
        for entry in log_entries:
            try:
                timestamp = getattr(entry, 'timestamp', datetime.now())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                level = getattr(entry, 'level', 'INFO')
                
                if level == 'ERROR':
                    error_pattern['timestamps'].append(timestamp)
                    error_pattern['values'].append(1)
                elif level == 'WARN':
                    warning_pattern['timestamps'].append(timestamp)
                    warning_pattern['values'].append(0.5)
            except Exception:
                continue
        
        if error_pattern['timestamps']:
            patterns['Error Pattern'] = error_pattern
        if warning_pattern['timestamps']:
            patterns['Warning Pattern'] = warning_pattern
        
        return patterns
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=message,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color='gray')
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400
        )
        return fig

# Export main class
__all__ = ['VisualLearningTools', 'VisualAnnotation', 'PatternVisualization'] 