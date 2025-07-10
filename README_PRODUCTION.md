# Production LogNarrator AI - Enhanced Log Analysis System

## 🚀 Overview

**Production LogNarrator AI** is a comprehensive, enterprise-ready log analysis system that transforms raw log data into actionable insights using advanced machine learning, temporal analysis, and intelligent pattern recognition.

### ✨ Key Features

- **🧠 ML-Based Confidence Scoring**: Real pattern recognition with weighted scoring algorithms
- **📝 Advanced Multi-line Parsing**: Stack trace reconstruction, JSON/XML extraction, log continuation detection
- **📊 Temporal Analysis**: Trend detection, degradation monitoring, time-based correlations
- **🗄️ Database Integration**: SQLite/PostgreSQL for storing analysis history, patterns, and user feedback
- **📋 Advanced Export System**: PDF reports, Excel with charts, custom templates
- **⚙️ Configuration Management**: Customizable rules, thresholds, and user preferences
- **🔍 Real-time Anomaly Detection**: Statistical anomaly detection with baseline metrics
- **🔗 Subsystem Correlation Analysis**: Detect cascading failures and cross-system dependencies
- **📈 Predictive Analytics**: Trend-based failure prediction and performance forecasting
- **🚨 Intelligent Alerting**: Comprehensive alerting with escalation rules

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- 4GB+ RAM recommended
- 1GB+ disk space for databases and exports

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd logAnalyzer

# Install dependencies
pip install -r requirements_production.txt

# Run the system
python production_lognarrator.py --help
```

### Dependencies

Core dependencies will be automatically installed:
- `pandas`, `numpy`, `scipy` - Data analysis and statistics
- `matplotlib`, `seaborn` - Visualization
- `reportlab`, `xlsxwriter` - Export capabilities
- `PyYAML` - Configuration management
- `streamlit`, `plotly` - Optional web interface

## 📖 Usage

### Basic Analysis

```bash
# Analyze a log file
python production_lognarrator.py logfile.txt

# Analyze with verbose output
python production_lognarrator.py logfile.txt --verbose

# Export to PDF
python production_lognarrator.py logfile.txt --export pdf --output report.pdf

# Export to Excel
python production_lognarrator.py logfile.txt --export excel --output analysis.xlsx
```

### Advanced Options

```bash
# Use custom configuration
python production_lognarrator.py logfile.txt --config custom_config.yaml

# Generate system report
python production_lognarrator.py --system-report --days 30

# Export system report
python production_lognarrator.py --system-report --days 30 --output system_report.json
```

### Configuration

Create a `lognarrator_config.yaml` file:

```yaml
analysis:
  confidence_threshold: 0.7
  learning_enabled: true
  anomaly_detection_threshold: 2.5
  trend_analysis_window_days: 30

performance:
  max_concurrent_analyses: 8
  processing_timeout_seconds: 600
  memory_limit_mb: 4096

alerts:
  degradation_threshold_percent: 25
  critical_threshold_percent: 50
  max_alerts_per_hour: 5

export:
  default_format: pdf
  include_charts: true
  include_trends: true
  pdf_page_size: A4
```

## 🏗️ System Architecture

### Core Components

1. **ProductionLogParser**: Advanced multi-line parsing with format detection
2. **ProductionLogCategorizer**: ML-based categorization with confidence scoring
3. **DatabaseManager**: Comprehensive data storage and retrieval
4. **TemporalAnalyzer**: Trend detection and time-based analysis
5. **AdvancedExporter**: Multi-format export with professional reporting
6. **ConfigManager**: Centralized configuration management

### Data Flow

```
Log File → Parser → Categorizer → Database → Temporal Analysis → Export
    ↓         ↓         ↓           ↓            ↓              ↓
Multi-line  ML        Historical  Trend       Correlation   PDF/Excel
Parsing   Confidence  Storage    Detection    Analysis      Reports
```

## 📊 Advanced Features

### Temporal Analysis

```python
from temporal_analyzer import get_temporal_analyzer

analyzer = get_temporal_analyzer()

# Analyze trends for a specific subsystem
trend_analysis = analyzer.analyze_trends('SEM', 'confidence_score', 30)

# Detect subsystem correlations
correlations = analyzer.detect_subsystem_correlations(7)

# Find temporal anomalies
anomalies = analyzer.detect_temporal_anomalies(24)
```

### Database Operations

```python
from database_manager import get_database_manager

db = get_database_manager()

# Get analysis history
history = db.get_analysis_history(limit=50, date_from=datetime.now() - timedelta(days=7))

# Store user feedback
feedback = UserFeedback(
    entry_hash="abc123",
    original_phase="ERROR",
    corrected_phase="RECOVERY",
    user_comment="This was actually a recovery attempt"
)
db.store_user_feedback(feedback)
```

### Export Capabilities

```python
from advanced_exporter import get_advanced_exporter, ExportRequest

exporter = get_advanced_exporter()

# Export comprehensive report
request = ExportRequest(
    export_type='pdf',
    output_path='comprehensive_report.pdf',
    include_charts=True,
    include_trends=True,
    include_correlations=True,
    time_window_days=30
)

success = exporter.export_analysis_report(request)
```

## 🔧 Configuration Options

### Analysis Settings

- `confidence_threshold`: Minimum confidence for accepting results (0.0-1.0)
- `learning_enabled`: Enable automatic pattern learning (true/false)
- `anomaly_detection_threshold`: Standard deviations for anomaly detection (1.0-5.0)
- `trend_analysis_window_days`: Days to analyze for trends (7-365)

### Performance Settings

- `max_concurrent_analyses`: Maximum parallel analyses (1-16)
- `processing_timeout_seconds`: Timeout for processing (60-3600)
- `memory_limit_mb`: Memory limit for analysis (512-8192)
- `cache_ttl_hours`: Cache time-to-live (1-168)

### Alert Settings

- `degradation_threshold_percent`: Threshold for degradation alerts (5-100)
- `critical_threshold_percent`: Threshold for critical alerts (10-100)
- `max_alerts_per_hour`: Maximum alerts per hour (1-100)

## 📈 Performance Metrics

The system provides comprehensive performance monitoring:

### Processing Performance
- **Entries per second**: Real-time processing speed
- **Memory usage**: RAM consumption during analysis
- **CPU utilization**: Processor usage patterns
- **Disk I/O**: File access patterns

### Analysis Quality
- **Confidence distribution**: Distribution of confidence scores
- **Pattern accuracy**: Success rate of pattern matching
- **Anomaly detection rate**: Percentage of anomalies detected
- **Trend prediction accuracy**: Accuracy of trend forecasting

## 🚨 Alerting System

### Alert Types

1. **Performance Degradation**: System performance declining
2. **Anomaly Detection**: Unusual patterns detected
3. **Trend Alerts**: Concerning trends identified
4. **System Health**: Resource utilization alerts

### Alert Severity Levels

- **LOW**: Minor issues, informational
- **MEDIUM**: Moderate issues, attention needed
- **HIGH**: Significant issues, prompt action required
- **CRITICAL**: Severe issues, immediate intervention needed

## 🔍 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Install missing dependencies
pip install -r requirements_production.txt

# Check Python version
python --version  # Should be 3.8+
```

#### Performance Issues
```bash
# Increase memory limit
python production_lognarrator.py --config config.yaml

# In config.yaml:
performance:
  memory_limit_mb: 8192
  max_concurrent_analyses: 2
```

#### Database Issues
```bash
# Reset database
rm lognarrator_production.db

# Check disk space
df -h .
```

### Logging

System logs are stored in `logs/lognarrator_YYYYMMDD.log`:

```bash
# View recent logs
tail -f logs/lognarrator_$(date +%Y%m%d).log

# Search for errors
grep ERROR logs/lognarrator_*.log
```

## 📚 API Reference

### Main System Class

```python
class ProductionLogNarrator:
    def __init__(self, config_file: Optional[str] = None)
    def analyze_log_file(self, file_path: str, export_options: Optional[Dict] = None) -> Dict[str, Any]
    def generate_system_report(self, time_window_days: int = 30) -> Dict[str, Any]
    def get_session_stats(self) -> Dict[str, Any]
```

### Analysis Results Structure

```python
{
    'session_id': int,
    'file_path': str,
    'analysis_timestamp': str,
    'processing_time': float,
    'total_entries': int,
    'parsing_metrics': {
        'format_detected': str,
        'encoding_used': str,
        'multi_line_entries': int,
        'stack_traces_found': int,
        'json_blocks_found': int
    },
    'performance_metrics': {
        'entries_per_second': float,
        'average_confidence': float,
        'parsing_time': float,
        'categorization_time': float
    },
    'risk_analysis': {
        'distribution': {'GREEN': int, 'YELLOW': int, 'RED': int},
        'percentages': {'GREEN': float, 'YELLOW': float, 'RED': float},
        'critical_subsystems': [str]
    },
    'temporal_analysis': {
        'trend_summary': dict,
        'correlation_summary': dict,
        'anomaly_summary': dict
    }
}
```

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

### Integration Tests
```bash
# Test with sample data
python production_lognarrator.py sample_logs/test_log.txt --verbose

# Test export functionality
python production_lognarrator.py sample_logs/test_log.txt --export pdf --output test_report.pdf
```

## 🤝 Contributing

### Code Style
- Use `black` for code formatting
- Use `flake8` for linting
- Add type hints for all functions
- Include docstrings for all classes and methods

### Development Setup
```bash
# Install development dependencies
pip install -r requirements_production.txt

# Format code
black .

# Lint code
flake8 .

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built upon the original LogNarrator AI foundation
- Uses advanced ML techniques for pattern recognition
- Incorporates statistical analysis for trend detection
- Leverages modern Python libraries for performance

## 🔮 Future Enhancements

- **Real-time Streaming**: Live log analysis with WebSocket support
- **Natural Language Queries**: Ask questions about log data in plain English
- **Advanced ML Models**: Deep learning for pattern recognition
- **Cloud Integration**: AWS/Azure/GCP native support
- **Predictive Maintenance**: ML-based failure prediction
- **Custom Dashboards**: Interactive web-based dashboards

---

**Production LogNarrator AI** - Transform your logs into actionable insights with enterprise-grade analysis and reporting. 