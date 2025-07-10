"""
Production Temporal Analysis System for LogNarrator AI

Real temporal pattern analysis including:
- Trend detection and forecasting
- Degradation monitoring
- Time-based correlations
- Anomaly detection over time
- Performance degradation alerts

Production functionality only - no demo data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
from collections import defaultdict, deque
import math
import statistics
from scipy import stats
import sqlite3

from database_manager import DatabaseManager, get_database_manager

logger = logging.getLogger(__name__)


class TrendType(Enum):
    """Types of trends that can be detected"""
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    CYCLICAL = "cyclical"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    metric_name: str
    subsystem: str
    trend_type: TrendType
    trend_strength: float  # 0-1 scale
    trend_direction: float  # -1 to 1 (negative = degrading, positive = improving)
    confidence: float  # 0-1 scale
    data_points: int
    time_span: timedelta
    slope: float
    r_squared: float
    seasonal_component: Optional[float]
    anomaly_score: float
    forecast_next_period: Optional[float]
    alert_threshold_breached: bool
    alert_severity: Optional[AlertSeverity]


@dataclass
class CorrelationResult:
    """Correlation analysis result"""
    subsystem_a: str
    subsystem_b: str
    correlation_coefficient: float
    p_value: float
    is_significant: bool
    lag_offset: int  # Time offset in analysis periods
    common_patterns: List[str]
    causality_score: float  # 0-1 scale indicating potential causality


@dataclass
class TemporalAnomaly:
    """Temporal anomaly detection result"""
    timestamp: datetime
    subsystem: str
    metric_name: str
    actual_value: float
    expected_value: float
    anomaly_score: float  # Standard deviations from expected
    anomaly_type: str  # 'spike', 'dip', 'trend_break', 'seasonal_deviation'
    context: Dict[str, Any]
    severity: AlertSeverity


@dataclass
class DegradationAlert:
    """System degradation alert"""
    subsystem: str
    metric_name: str
    alert_type: str  # 'performance', 'reliability', 'availability'
    severity: AlertSeverity
    current_value: float
    baseline_value: float
    degradation_percentage: float
    time_detected: datetime
    trend_duration: timedelta
    recommended_actions: List[str]


class TemporalAnalyzer:
    """Production temporal analysis system"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or get_database_manager()
        
        # Analysis parameters
        self.min_data_points = 5
        self.significance_threshold = 0.05
        self.anomaly_threshold = 2.0  # Standard deviations
        self.trend_strength_threshold = 0.7
        
        # Baseline calculations
        self.baseline_window_days = 30
        self.seasonal_detection_window = 7  # days
        
        # Performance thresholds
        self.degradation_thresholds = {
            'processing_time': {'warning': 0.2, 'critical': 0.5},  # 20% and 50% increase
            'confidence': {'warning': -0.1, 'critical': -0.2},    # 10% and 20% decrease
            'error_rate': {'warning': 0.1, 'critical': 0.3},      # 10% and 30% increase
            'throughput': {'warning': -0.15, 'critical': -0.3}    # 15% and 30% decrease
        }
        
        # Cache for performance
        self.analysis_cache = {}
        self.cache_ttl = timedelta(hours=1)
        
        logger.info("Temporal analyzer initialized")
    
    def analyze_trends(self, subsystem: str, metric_name: str, 
                      time_window_days: int = 30) -> TrendAnalysis:
        """Comprehensive trend analysis for a metric"""
        
        # Check cache first
        cache_key = f"{subsystem}:{metric_name}:{time_window_days}"
        if cache_key in self.analysis_cache:
            cached_result, cached_time = self.analysis_cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_result
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_window_days)
        
        trend_data = self.db_manager.get_trend_analysis(
            subsystem, metric_name, period='session', limit=1000
        )
        
        if len(trend_data) < self.min_data_points:
            return self._create_insufficient_data_result(subsystem, metric_name)
        
        # Convert to pandas for analysis
        df = pd.DataFrame([{
            'timestamp': td.measurement_timestamp,
            'value': td.metric_value
        } for td in trend_data])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Filter to time window
        df = df[df['timestamp'] >= start_date]
        
        if len(df) < self.min_data_points:
            return self._create_insufficient_data_result(subsystem, metric_name)
        
        # Perform comprehensive analysis
        analysis = self._perform_trend_analysis(df, subsystem, metric_name)
        
        # Cache result
        self.analysis_cache[cache_key] = (analysis, datetime.now())
        
        return analysis
    
    def _perform_trend_analysis(self, df: pd.DataFrame, subsystem: str, 
                               metric_name: str) -> TrendAnalysis:
        """Perform comprehensive trend analysis"""
        
        # Prepare data
        df['timestamp_numeric'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
        
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df['timestamp_numeric'], df['value']
        )
        
        r_squared = r_value ** 2
        
        # Determine trend type and strength
        trend_type, trend_strength = self._classify_trend(df, slope, r_squared)
        
        # Calculate trend direction (normalized)
        value_range = df['value'].max() - df['value'].min()
        if value_range > 0:
            trend_direction = slope / value_range
        else:
            trend_direction = 0.0
        
        # Detect seasonal patterns
        seasonal_component = self._detect_seasonal_patterns(df)
        
        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(df)
        
        # Forecast next period
        forecast_next_period = self._forecast_next_period(df, slope, intercept)
        
        # Check alert thresholds
        alert_threshold_breached, alert_severity = self._check_alert_thresholds(
            subsystem, metric_name, df, trend_type, trend_strength
        )
        
        # Calculate confidence
        confidence = self._calculate_trend_confidence(df, r_squared, len(df))
        
        return TrendAnalysis(
            metric_name=metric_name,
            subsystem=subsystem,
            trend_type=trend_type,
            trend_strength=trend_strength,
            trend_direction=max(-1.0, min(1.0, trend_direction)),
            confidence=confidence,
            data_points=len(df),
            time_span=df['timestamp'].max() - df['timestamp'].min(),
            slope=slope,
            r_squared=r_squared,
            seasonal_component=seasonal_component,
            anomaly_score=anomaly_score,
            forecast_next_period=forecast_next_period,
            alert_threshold_breached=alert_threshold_breached,
            alert_severity=alert_severity
        )
    
    def _classify_trend(self, df: pd.DataFrame, slope: float, r_squared: float) -> Tuple[TrendType, float]:
        """Classify trend type and calculate strength"""
        
        # Calculate volatility
        volatility = df['value'].std() / df['value'].mean() if df['value'].mean() > 0 else 0
        
        # Determine trend strength based on R-squared and volatility
        trend_strength = r_squared * (1 - min(volatility, 1.0))
        
        # Classify trend type
        if r_squared < 0.3:
            if volatility > 0.2:
                return TrendType.VOLATILE, trend_strength
            else:
                return TrendType.STABLE, trend_strength
        
        # Check for cyclical patterns
        if self._is_cyclical_pattern(df):
            return TrendType.CYCLICAL, trend_strength
        
        # Determine improving vs degrading based on slope and metric type
        if abs(slope) < 1e-6:
            return TrendType.STABLE, trend_strength
        elif slope > 0:
            # For some metrics, increasing is bad (e.g., error_rate, processing_time)
            if any(bad_metric in df.columns[1].lower() for bad_metric in ['error', 'time', 'failure']):
                return TrendType.DEGRADING, trend_strength
            else:
                return TrendType.IMPROVING, trend_strength
        else:
            if any(bad_metric in df.columns[1].lower() for bad_metric in ['error', 'time', 'failure']):
                return TrendType.IMPROVING, trend_strength
            else:
                return TrendType.DEGRADING, trend_strength
    
    def _is_cyclical_pattern(self, df: pd.DataFrame) -> bool:
        """Detect cyclical patterns using autocorrelation"""
        if len(df) < 12:  # Need enough data for cycle detection
            return False
        
        try:
            # Calculate autocorrelation
            values = df['value'].values
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Look for secondary peaks indicating cycles
            if len(autocorr) > 3:
                # Normalize
                autocorr = autocorr / autocorr[0]
                
                # Find peaks after the first one
                for i in range(2, min(len(autocorr), len(values) // 2)):
                    if autocorr[i] > 0.5:  # Strong correlation at lag i
                        return True
            
            return False
        except:
            return False
    
    def _detect_seasonal_patterns(self, df: pd.DataFrame) -> Optional[float]:
        """Detect seasonal patterns in the data"""
        if len(df) < 7:  # Need at least a week of data
            return None
        
        try:
            # Add time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Calculate variance by hour and day of week
            hourly_variance = df.groupby('hour')['value'].var().mean()
            daily_variance = df.groupby('day_of_week')['value'].var().mean()
            
            # If there's significant variance by time, there's seasonality
            overall_variance = df['value'].var()
            
            if overall_variance > 0:
                seasonal_strength = max(hourly_variance, daily_variance) / overall_variance
                return min(seasonal_strength, 1.0)
            
            return None
        except:
            return None
    
    def _calculate_anomaly_score(self, df: pd.DataFrame) -> float:
        """Calculate overall anomaly score for the dataset"""
        if len(df) < 3:
            return 0.0
        
        # Calculate z-scores for all values
        z_scores = np.abs(stats.zscore(df['value']))
        
        # Return the 95th percentile of anomaly scores
        return np.percentile(z_scores, 95)
    
    def _forecast_next_period(self, df: pd.DataFrame, slope: float, intercept: float) -> Optional[float]:
        """Forecast the next period value"""
        if len(df) < 3:
            return None
        
        # Simple linear forecast
        last_timestamp = df['timestamp_numeric'].max()
        next_timestamp = last_timestamp + (df['timestamp_numeric'].iloc[-1] - df['timestamp_numeric'].iloc[-2])
        
        forecast = slope * next_timestamp + intercept
        
        # Add some bounds checking
        recent_values = df['value'].tail(5)
        min_recent = recent_values.min()
        max_recent = recent_values.max()
        
        # Don't forecast beyond 2x the recent range
        if forecast > max_recent * 2:
            forecast = max_recent * 1.5
        elif forecast < min_recent * 0.5:
            forecast = min_recent * 0.5
        
        return forecast
    
    def _check_alert_thresholds(self, subsystem: str, metric_name: str, 
                               df: pd.DataFrame, trend_type: TrendType, 
                               trend_strength: float) -> Tuple[bool, Optional[AlertSeverity]]:
        """Check if alert thresholds are breached"""
        
        # Get baseline (first 30% of data or historical baseline)
        baseline_size = max(1, len(df) // 3)
        baseline_value = df['value'].head(baseline_size).mean()
        current_value = df['value'].tail(5).mean()  # Recent average
        
        if baseline_value == 0:
            return False, None
        
        # Calculate change percentage
        change_percentage = (current_value - baseline_value) / baseline_value
        
        # Check against thresholds
        metric_key = self._get_metric_key(metric_name)
        
        if metric_key in self.degradation_thresholds:
            thresholds = self.degradation_thresholds[metric_key]
            
            # Check critical threshold
            if abs(change_percentage) >= abs(thresholds['critical']):
                return True, AlertSeverity.CRITICAL
            
            # Check warning threshold
            elif abs(change_percentage) >= abs(thresholds['warning']):
                return True, AlertSeverity.HIGH
            
            # Check trend strength for medium alerts
            elif trend_type == TrendType.DEGRADING and trend_strength > self.trend_strength_threshold:
                return True, AlertSeverity.MEDIUM
        
        return False, None
    
    def _get_metric_key(self, metric_name: str) -> str:
        """Map metric name to standardized key"""
        metric_lower = metric_name.lower()
        
        if 'time' in metric_lower or 'duration' in metric_lower:
            return 'processing_time'
        elif 'confidence' in metric_lower:
            return 'confidence'
        elif 'error' in metric_lower or 'failure' in metric_lower:
            return 'error_rate'
        elif 'throughput' in metric_lower or 'rate' in metric_lower:
            return 'throughput'
        else:
            return 'generic'
    
    def _calculate_trend_confidence(self, df: pd.DataFrame, r_squared: float, data_points: int) -> float:
        """Calculate confidence in trend analysis"""
        
        # Base confidence on R-squared
        confidence = r_squared
        
        # Adjust for sample size
        if data_points < 10:
            confidence *= 0.7
        elif data_points < 20:
            confidence *= 0.85
        elif data_points > 50:
            confidence *= 1.1
        
        # Adjust for data quality
        if df['value'].std() / df['value'].mean() > 0.5:  # High volatility
            confidence *= 0.8
        
        return max(0.0, min(1.0, confidence))
    
    def _create_insufficient_data_result(self, subsystem: str, metric_name: str) -> TrendAnalysis:
        """Create result for insufficient data"""
        return TrendAnalysis(
            metric_name=metric_name,
            subsystem=subsystem,
            trend_type=TrendType.UNKNOWN,
            trend_strength=0.0,
            trend_direction=0.0,
            confidence=0.0,
            data_points=0,
            time_span=timedelta(0),
            slope=0.0,
            r_squared=0.0,
            seasonal_component=None,
            anomaly_score=0.0,
            forecast_next_period=None,
            alert_threshold_breached=False,
            alert_severity=None
        )
    
    def detect_subsystem_correlations(self, time_window_days: int = 7) -> List[CorrelationResult]:
        """Detect correlations between subsystems"""
        
        # Get correlation data
        correlation_data = self.db_manager.get_subsystem_correlation_data(time_window_days * 24)
        
        if len(correlation_data) < 2:
            return []
        
        correlations = []
        subsystems = list(correlation_data.keys())
        
        # Analyze each pair of subsystems
        for i in range(len(subsystems)):
            for j in range(i + 1, len(subsystems)):
                subsystem_a = subsystems[i]
                subsystem_b = subsystems[j]
                
                correlation = self._analyze_subsystem_correlation(
                    correlation_data[subsystem_a], 
                    correlation_data[subsystem_b],
                    subsystem_a, 
                    subsystem_b
                )
                
                if correlation.is_significant:
                    correlations.append(correlation)
        
        return sorted(correlations, key=lambda x: abs(x.correlation_coefficient), reverse=True)
    
    def _analyze_subsystem_correlation(self, data_a: List[Dict], data_b: List[Dict],
                                     subsystem_a: str, subsystem_b: str) -> CorrelationResult:
        """Analyze correlation between two subsystems"""
        
        # Convert to time series
        ts_a = self._convert_to_timeseries(data_a)
        ts_b = self._convert_to_timeseries(data_b)
        
        # Find common time range
        common_times = set(ts_a.keys()) & set(ts_b.keys())
        
        if len(common_times) < 3:
            return CorrelationResult(
                subsystem_a=subsystem_a,
                subsystem_b=subsystem_b,
                correlation_coefficient=0.0,
                p_value=1.0,
                is_significant=False,
                lag_offset=0,
                common_patterns=[],
                causality_score=0.0
            )
        
        # Extract aligned values
        times = sorted(common_times)
        values_a = [ts_a[t] for t in times]
        values_b = [ts_b[t] for t in times]
        
        # Calculate correlation
        correlation_coeff, p_value = stats.pearsonr(values_a, values_b)
        
        # Check for lagged correlations
        best_lag, best_corr = self._find_best_lag_correlation(values_a, values_b)
        
        # Use the best correlation found
        if abs(best_corr) > abs(correlation_coeff):
            correlation_coeff = best_corr
            lag_offset = best_lag
        else:
            lag_offset = 0
        
        # Determine significance
        is_significant = p_value < self.significance_threshold and abs(correlation_coeff) > 0.3
        
        # Find common patterns
        common_patterns = self._find_common_patterns(data_a, data_b)
        
        # Calculate causality score using Granger-like analysis
        causality_score = self._calculate_causality_score(values_a, values_b, lag_offset)
        
        return CorrelationResult(
            subsystem_a=subsystem_a,
            subsystem_b=subsystem_b,
            correlation_coefficient=correlation_coeff,
            p_value=p_value,
            is_significant=is_significant,
            lag_offset=lag_offset,
            common_patterns=common_patterns,
            causality_score=causality_score
        )
    
    def _convert_to_timeseries(self, data: List[Dict]) -> Dict[str, float]:
        """Convert subsystem data to time series"""
        ts = {}
        
        for event in data:
            # Convert risk to numeric value
            risk_value = {'GREEN': 0, 'YELLOW': 1, 'RED': 2}.get(event['risk'], 0)
            
            # Use session timestamp as key
            time_key = event['session_timestamp']
            
            # Aggregate by time (average if multiple events at same time)
            if time_key in ts:
                ts[time_key] = (ts[time_key] + risk_value) / 2
            else:
                ts[time_key] = risk_value
        
        return ts
    
    def _find_best_lag_correlation(self, values_a: List[float], values_b: List[float]) -> Tuple[int, float]:
        """Find the best lag correlation between two series"""
        
        max_lag = min(len(values_a) // 4, 10)  # Don't lag more than 25% of data or 10 points
        best_corr = 0.0
        best_lag = 0
        
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                continue
            
            try:
                if lag > 0:
                    # values_a leads values_b
                    corr, _ = stats.pearsonr(values_a[:-lag], values_b[lag:])
                else:
                    # values_b leads values_a
                    corr, _ = stats.pearsonr(values_a[-lag:], values_b[:lag])
                
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            except:
                continue
        
        return best_lag, best_corr
    
    def _find_common_patterns(self, data_a: List[Dict], data_b: List[Dict]) -> List[str]:
        """Find common patterns between two subsystems"""
        patterns = []
        
        # Check for simultaneous errors
        error_times_a = [d['session_timestamp'] for d in data_a if d['risk'] == 'RED']
        error_times_b = [d['session_timestamp'] for d in data_b if d['risk'] == 'RED']
        
        # Find overlapping error times (within 1 hour)
        simultaneous_errors = 0
        for time_a in error_times_a:
            for time_b in error_times_b:
                if abs((datetime.fromisoformat(time_a) - datetime.fromisoformat(time_b)).total_seconds()) < 3600:
                    simultaneous_errors += 1
                    break
        
        if simultaneous_errors > 0:
            patterns.append(f"simultaneous_errors_{simultaneous_errors}")
        
        # Check for cascading failures
        for i, time_a in enumerate(error_times_a):
            for j, time_b in enumerate(error_times_b):
                time_diff = (datetime.fromisoformat(time_b) - datetime.fromisoformat(time_a)).total_seconds()
                if 0 < time_diff < 1800:  # B follows A within 30 minutes
                    patterns.append("cascading_failure")
                    break
        
        return patterns
    
    def _calculate_causality_score(self, values_a: List[float], values_b: List[float], lag: int) -> float:
        """Calculate simplified causality score"""
        
        if len(values_a) < 5 or len(values_b) < 5:
            return 0.0
        
        try:
            # Simple causality test: does A's past help predict B's future?
            if lag > 0:
                # A causes B
                past_a = values_a[:-lag]
                future_b = values_b[lag:]
                
                # Calculate how well past A predicts future B
                correlation, p_value = stats.pearsonr(past_a, future_b)
                
                # Causality score based on correlation strength and significance
                if p_value < 0.05:
                    return min(abs(correlation), 1.0)
                else:
                    return abs(correlation) * 0.5
            
            elif lag < 0:
                # B causes A
                past_b = values_b[:lag]
                future_a = values_a[-lag:]
                
                correlation, p_value = stats.pearsonr(past_b, future_a)
                
                if p_value < 0.05:
                    return min(abs(correlation), 1.0)
                else:
                    return abs(correlation) * 0.5
            
            else:
                # No lag, just correlation
                correlation, p_value = stats.pearsonr(values_a, values_b)
                return abs(correlation) * 0.3  # Lower score for no causality
        
        except:
            return 0.0
    
    def detect_temporal_anomalies(self, time_window_hours: int = 24) -> List[TemporalAnomaly]:
        """Detect temporal anomalies in recent data"""
        
        anomalies = []
        
        # Get recent system metrics
        recent_metrics = self.db_manager.get_system_metrics(hours_back=time_window_hours)
        
        # Group by metric type
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric['metric_name']].append(metric)
        
        # Analyze each metric group
        for metric_name, metrics in metric_groups.items():
            if len(metrics) < 3:
                continue
            
            # Convert to time series
            df = pd.DataFrame(metrics)
            df['recorded_at'] = pd.to_datetime(df['recorded_at'])
            df = df.sort_values('recorded_at')
            
            # Detect anomalies
            metric_anomalies = self._detect_metric_anomalies(df, metric_name)
            anomalies.extend(metric_anomalies)
        
        return sorted(anomalies, key=lambda x: x.anomaly_score, reverse=True)
    
    def _detect_metric_anomalies(self, df: pd.DataFrame, metric_name: str) -> List[TemporalAnomaly]:
        """Detect anomalies in a single metric"""
        
        anomalies = []
        
        if len(df) < 5:
            return anomalies
        
        # Calculate rolling statistics
        window_size = min(10, len(df) // 2)
        df['rolling_mean'] = df['metric_value'].rolling(window=window_size).mean()
        df['rolling_std'] = df['metric_value'].rolling(window=window_size).std()
        
        # Calculate z-scores
        df['z_score'] = (df['metric_value'] - df['rolling_mean']) / df['rolling_std']
        
        # Find anomalies
        for idx, row in df.iterrows():
            if pd.isna(row['z_score']) or pd.isna(row['rolling_mean']):
                continue
            
            if abs(row['z_score']) > self.anomaly_threshold:
                
                # Determine anomaly type
                if row['z_score'] > self.anomaly_threshold:
                    anomaly_type = 'spike'
                else:
                    anomaly_type = 'dip'
                
                # Calculate severity
                severity = self._calculate_anomaly_severity(row['z_score'])
                
                # Create anomaly record
                anomaly = TemporalAnomaly(
                    timestamp=row['recorded_at'],
                    subsystem='SYSTEM',
                    metric_name=metric_name,
                    actual_value=row['metric_value'],
                    expected_value=row['rolling_mean'],
                    anomaly_score=abs(row['z_score']),
                    anomaly_type=anomaly_type,
                    context={'window_size': window_size, 'rolling_std': row['rolling_std']},
                    severity=severity
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_anomaly_severity(self, z_score: float) -> AlertSeverity:
        """Calculate anomaly severity based on z-score"""
        
        abs_z = abs(z_score)
        
        if abs_z >= 4.0:
            return AlertSeverity.CRITICAL
        elif abs_z >= 3.0:
            return AlertSeverity.HIGH
        elif abs_z >= 2.5:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def generate_degradation_alerts(self, time_window_days: int = 7) -> List[DegradationAlert]:
        """Generate system degradation alerts"""
        
        alerts = []
        
        # Get all active subsystems
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_window_days)
        
        # Get trend data for all subsystems
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute('''
                SELECT DISTINCT subsystem, metric_name 
                FROM trend_data 
                WHERE measurement_timestamp >= ?
                AND subsystem != 'SYSTEM'
            ''', (start_date.isoformat(),))
            
            subsystem_metrics = cursor.fetchall()
        
        # Analyze each subsystem-metric combination
        for row in subsystem_metrics:
            subsystem, metric_name = row['subsystem'], row['metric_name']
            
            # Get trend analysis
            trend_analysis = self.analyze_trends(subsystem, metric_name, time_window_days)
            
            # Check for degradation
            if (trend_analysis.trend_type == TrendType.DEGRADING and 
                trend_analysis.trend_strength > self.trend_strength_threshold):
                
                # Calculate degradation details
                degradation_alert = self._create_degradation_alert(
                    subsystem, metric_name, trend_analysis
                )
                
                if degradation_alert:
                    alerts.append(degradation_alert)
        
        return sorted(alerts, key=lambda x: x.severity.value, reverse=True)
    
    def _create_degradation_alert(self, subsystem: str, metric_name: str, 
                                 trend_analysis: TrendAnalysis) -> Optional[DegradationAlert]:
        """Create degradation alert from trend analysis"""
        
        if not trend_analysis.alert_threshold_breached:
            return None
        
        # Determine alert type
        if 'performance' in metric_name.lower() or 'time' in metric_name.lower():
            alert_type = 'performance'
        elif 'error' in metric_name.lower() or 'failure' in metric_name.lower():
            alert_type = 'reliability'
        else:
            alert_type = 'availability'
        
        # Calculate current and baseline values
        current_value = trend_analysis.forecast_next_period or 0.0
        baseline_value = current_value - (trend_analysis.slope * trend_analysis.time_span.total_seconds())
        
        # Calculate degradation percentage
        if baseline_value != 0:
            degradation_percentage = abs((current_value - baseline_value) / baseline_value) * 100
        else:
            degradation_percentage = 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(alert_type, subsystem, degradation_percentage)
        
        return DegradationAlert(
            subsystem=subsystem,
            metric_name=metric_name,
            alert_type=alert_type,
            severity=trend_analysis.alert_severity,
            current_value=current_value,
            baseline_value=baseline_value,
            degradation_percentage=degradation_percentage,
            time_detected=datetime.now(),
            trend_duration=trend_analysis.time_span,
            recommended_actions=recommendations
        )
    
    def _generate_recommendations(self, alert_type: str, subsystem: str, 
                                degradation_percentage: float) -> List[str]:
        """Generate recommended actions for degradation alerts"""
        
        recommendations = []
        
        if alert_type == 'performance':
            recommendations.extend([
                f"Monitor {subsystem} resource usage (CPU, Memory, I/O)",
                "Check for recent configuration changes",
                "Review system logs for bottlenecks",
                "Consider scaling resources if degradation > 30%"
            ])
        
        elif alert_type == 'reliability':
            recommendations.extend([
                f"Investigate error patterns in {subsystem}",
                "Check hardware health and connections",
                "Review error logs for root causes",
                "Consider failover procedures if critical"
            ])
        
        else:  # availability
            recommendations.extend([
                f"Check {subsystem} connectivity and status",
                "Verify service dependencies",
                "Review maintenance schedules",
                "Consider backup system activation"
            ])
        
        # Add severity-specific recommendations
        if degradation_percentage > 50:
            recommendations.append("URGENT: Consider immediate intervention")
        elif degradation_percentage > 25:
            recommendations.append("Schedule maintenance window for investigation")
        
        return recommendations
    
    def get_temporal_summary(self, time_window_days: int = 7) -> Dict[str, Any]:
        """Get comprehensive temporal analysis summary"""
        
        summary = {
            'analysis_period': f"{time_window_days} days",
            'generated_at': datetime.now().isoformat(),
            'trend_summary': {},
            'correlation_summary': {},
            'anomaly_summary': {},
            'degradation_summary': {}
        }
        
        # Get all subsystems with recent data
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute('''
                SELECT DISTINCT subsystem 
                FROM trend_data 
                WHERE measurement_timestamp >= ?
                AND subsystem != 'SYSTEM'
            ''', ((datetime.now() - timedelta(days=time_window_days)).isoformat(),))
            
            subsystems = [row['subsystem'] for row in cursor.fetchall()]
        
        # Analyze trends for each subsystem
        trend_counts = {'improving': 0, 'degrading': 0, 'stable': 0, 'volatile': 0}
        
        for subsystem in subsystems:
            # Get main metrics for this subsystem
            trend_analysis = self.analyze_trends(subsystem, 'overall_confidence', time_window_days)
            trend_type = trend_analysis.trend_type.value
            if trend_type not in trend_counts:
                trend_counts[trend_type] = 0
            trend_counts[trend_type] += 1
        
        summary['trend_summary'] = {
            'total_subsystems': len(subsystems),
            'trend_distribution': trend_counts,
            'systems_with_alerts': len([s for s in subsystems if self.analyze_trends(s, 'overall_confidence', time_window_days).alert_threshold_breached])
        }
        
        # Correlation analysis
        correlations = self.detect_subsystem_correlations(time_window_days)
        summary['correlation_summary'] = {
            'total_correlations_found': len(correlations),
            'strong_correlations': len([c for c in correlations if abs(c.correlation_coefficient) > 0.7]),
            'potential_causalities': len([c for c in correlations if c.causality_score > 0.6])
        }
        
        # Anomaly detection
        anomalies = self.detect_temporal_anomalies(time_window_days * 24)
        summary['anomaly_summary'] = {
            'total_anomalies': len(anomalies),
            'critical_anomalies': len([a for a in anomalies if a.severity == AlertSeverity.CRITICAL]),
            'anomaly_types': list(set([a.anomaly_type for a in anomalies]))
        }
        
        # Degradation alerts
        degradation_alerts = self.generate_degradation_alerts(time_window_days)
        summary['degradation_summary'] = {
            'total_alerts': len(degradation_alerts),
            'critical_alerts': len([a for a in degradation_alerts if a.severity == AlertSeverity.CRITICAL]),
            'affected_subsystems': list(set([a.subsystem for a in degradation_alerts]))
        }
        
        return summary
    
    def export_temporal_analysis(self, output_path: str, time_window_days: int = 30):
        """Export comprehensive temporal analysis to file"""
        
        analysis_data = {
            'export_timestamp': datetime.now().isoformat(),
            'analysis_window_days': time_window_days,
            'summary': self.get_temporal_summary(time_window_days),
            'detailed_trends': [],
            'correlations': [],
            'anomalies': [],
            'degradation_alerts': []
        }
        
        # Get detailed analysis for each subsystem
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute('''
                SELECT DISTINCT subsystem, metric_name 
                FROM trend_data 
                WHERE measurement_timestamp >= ?
                AND subsystem != 'SYSTEM'
            ''', ((datetime.now() - timedelta(days=time_window_days)).isoformat(),))
            
            for row in cursor.fetchall():
                subsystem, metric_name = row['subsystem'], row['metric_name']
                trend_analysis = self.analyze_trends(subsystem, metric_name, time_window_days)
                analysis_data['detailed_trends'].append(asdict(trend_analysis))
        
        # Add correlations
        correlations = self.detect_subsystem_correlations(time_window_days)
        analysis_data['correlations'] = [asdict(c) for c in correlations]
        
        # Add anomalies
        anomalies = self.detect_temporal_anomalies(time_window_days * 24)
        analysis_data['anomalies'] = [asdict(a) for a in anomalies]
        
        # Add degradation alerts
        degradation_alerts = self.generate_degradation_alerts(time_window_days)
        analysis_data['degradation_alerts'] = [asdict(a) for a in degradation_alerts]
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        logger.info(f"Temporal analysis exported to {output_path}")


# Global temporal analyzer instance
_temporal_analyzer = None

def get_temporal_analyzer() -> TemporalAnalyzer:
    """Get global temporal analyzer instance"""
    global _temporal_analyzer
    if _temporal_analyzer is None:
        _temporal_analyzer = TemporalAnalyzer()
    return _temporal_analyzer 