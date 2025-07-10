"""
Production Database Manager for LogNarrator AI

Comprehensive database system for:
- Analysis history storage
- Pattern learning and optimization
- User feedback integration
- Performance metrics tracking
- Historical trend analysis

Real production functionality - no demo data.
"""

import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from contextlib import contextmanager
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisRecord:
    """Record of a complete log analysis session"""
    id: Optional[int]
    file_path: str
    file_hash: str
    analysis_timestamp: datetime
    total_entries: int
    risk_distribution: Dict[str, int]
    phase_distribution: Dict[str, int]
    subsystem_analysis: Dict[str, Dict]
    overall_confidence: float
    processing_time: float
    format_detected: str
    encoding_used: str
    ai_summary: Optional[str]
    user_notes: Optional[str]


@dataclass
class PatternPerformance:
    """Pattern performance tracking"""
    pattern_id: Optional[int]
    pattern_text: str
    pattern_type: str  # 'phase', 'risk', 'subsystem'
    target_value: str  # What it should detect
    success_count: int
    failure_count: int
    confidence_score: float
    last_updated: datetime
    created_by: str  # 'system' or 'user'


@dataclass
class UserFeedback:
    """User feedback on categorization accuracy"""
    feedback_id: Optional[int]
    entry_hash: str
    original_phase: str
    original_risk: str
    corrected_phase: Optional[str]
    corrected_risk: Optional[str]
    user_comment: Optional[str]
    feedback_timestamp: datetime
    confidence_rating: int  # 1-5 scale


@dataclass
class TrendData:
    """Trend analysis data point"""
    trend_id: Optional[int]
    subsystem: str
    metric_name: str
    metric_value: float
    measurement_timestamp: datetime
    analysis_session_id: int
    trend_period: str  # 'hour', 'day', 'week', 'month'


class DatabaseManager:
    """Production database manager with comprehensive functionality"""
    
    def __init__(self, db_path: str = "lognarrator_production.db"):
        self.db_path = db_path
        self.connection_pool = []
        self.pool_lock = threading.Lock()
        self.max_connections = 10
        
        # Initialize database schema
        self._init_database()
        
        # Create indexes for performance
        self._create_indexes()
        
        logger.info(f"Database manager initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize comprehensive database schema"""
        with self.get_connection() as conn:
            # Analysis sessions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    analysis_timestamp TIMESTAMP NOT NULL,
                    total_entries INTEGER NOT NULL,
                    risk_distribution TEXT NOT NULL,  -- JSON
                    phase_distribution TEXT NOT NULL,  -- JSON
                    subsystem_analysis TEXT NOT NULL,  -- JSON
                    overall_confidence REAL NOT NULL,
                    processing_time REAL NOT NULL,
                    format_detected TEXT,
                    encoding_used TEXT,
                    ai_summary TEXT,
                    user_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Individual log entries table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS log_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    entry_hash TEXT NOT NULL UNIQUE,
                    timestamp TEXT,
                    subsystem TEXT,
                    event_text TEXT NOT NULL,
                    predicted_phase TEXT NOT NULL,
                    predicted_risk TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    pattern_matches TEXT,  -- JSON array of matched patterns
                    anomaly_indicators TEXT,  -- JSON array
                    processing_time REAL,
                    line_numbers TEXT,  -- JSON array
                    entry_type TEXT DEFAULT 'single_line',
                    FOREIGN KEY (session_id) REFERENCES analysis_sessions(id)
                )
            ''')
            
            # Pattern performance tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pattern_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_text TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    target_value TEXT NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    confidence_score REAL DEFAULT 0.5,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT DEFAULT 'system',
                    is_active BOOLEAN DEFAULT 1,
                    UNIQUE(pattern_text, pattern_type, target_value)
                )
            ''')
            
            # User feedback table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_hash TEXT NOT NULL,
                    original_phase TEXT NOT NULL,
                    original_risk TEXT NOT NULL,
                    corrected_phase TEXT,
                    corrected_risk TEXT,
                    user_comment TEXT,
                    feedback_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence_rating INTEGER CHECK(confidence_rating BETWEEN 1 AND 5),
                    is_processed BOOLEAN DEFAULT 0,
                    FOREIGN KEY (entry_hash) REFERENCES log_entries(entry_hash)
                )
            ''')
            
            # Trend analysis table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trend_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subsystem TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    measurement_timestamp TIMESTAMP NOT NULL,
                    analysis_session_id INTEGER NOT NULL,
                    trend_period TEXT NOT NULL,
                    FOREIGN KEY (analysis_session_id) REFERENCES analysis_sessions(id)
                )
            ''')
            
            # System metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_metadata TEXT,  -- JSON
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Configuration table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    value_type TEXT NOT NULL,  -- 'string', 'int', 'float', 'json'
                    description TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Alert rules table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT NOT NULL UNIQUE,
                    condition_type TEXT NOT NULL,  -- 'threshold', 'pattern', 'trend'
                    condition_config TEXT NOT NULL,  -- JSON
                    action_type TEXT NOT NULL,  -- 'email', 'webhook', 'log'
                    action_config TEXT NOT NULL,  -- JSON
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_triggered TIMESTAMP
                )
            ''')
            
            conn.commit()
            
            # Initialize default configuration
            self._init_default_config(conn)
    
    def _init_default_config(self, conn: sqlite3.Connection):
        """Initialize default system configuration"""
        default_configs = [
            ('confidence_threshold', '0.6', 'float', 'Minimum confidence threshold for categorization'),
            ('learning_enabled', 'true', 'string', 'Enable automatic pattern learning'),
            ('retention_days', '90', 'int', 'Number of days to retain analysis data'),
            ('max_pattern_matches', '10', 'int', 'Maximum patterns to track per entry'),
            ('anomaly_threshold', '0.7', 'float', 'Threshold for anomaly detection'),
            ('auto_feedback_processing', 'true', 'string', 'Automatically process user feedback'),
            ('performance_tracking', 'true', 'string', 'Enable performance metrics tracking'),
        ]
        
        for key, value, value_type, description in default_configs:
            conn.execute('''
                INSERT OR IGNORE INTO system_config (key, value, value_type, description)
                VALUES (?, ?, ?, ?)
            ''', (key, value, value_type, description))
        
        conn.commit()
    
    def _create_indexes(self):
        """Create database indexes for performance optimization"""
        with self.get_connection() as conn:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_analysis_sessions_timestamp ON analysis_sessions(analysis_timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_analysis_sessions_file_hash ON analysis_sessions(file_hash)",
                "CREATE INDEX IF NOT EXISTS idx_log_entries_hash ON log_entries(entry_hash)",
                "CREATE INDEX IF NOT EXISTS idx_log_entries_session ON log_entries(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_log_entries_subsystem ON log_entries(subsystem)",
                "CREATE INDEX IF NOT EXISTS idx_log_entries_timestamp ON log_entries(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_pattern_performance_type ON pattern_performance(pattern_type)",
                "CREATE INDEX IF NOT EXISTS idx_user_feedback_processed ON user_feedback(is_processed)",
                "CREATE INDEX IF NOT EXISTS idx_trend_data_subsystem ON trend_data(subsystem, trend_period)",
                "CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name)",
            ]
            
            for index_sql in indexes:
                conn.execute(index_sql)
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        conn = None
        try:
            with self.pool_lock:
                if self.connection_pool:
                    conn = self.connection_pool.pop()
                else:
                    conn = sqlite3.connect(self.db_path, timeout=30.0)
                    conn.row_factory = sqlite3.Row
            
            yield conn
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                with self.pool_lock:
                    if len(self.connection_pool) < self.max_connections:
                        self.connection_pool.append(conn)
                    else:
                        conn.close()
    
    def store_analysis_session(self, analysis_data: Dict, entries_data: List[Dict]) -> int:
        """Store complete analysis session with all entries"""
        
        with self.get_connection() as conn:
            # Insert analysis session
            session_cursor = conn.execute('''
                INSERT INTO analysis_sessions 
                (file_path, file_hash, analysis_timestamp, total_entries, risk_distribution,
                 phase_distribution, subsystem_analysis, overall_confidence, processing_time,
                 format_detected, encoding_used, ai_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_data['file_path'],
                analysis_data['file_hash'],
                analysis_data['analysis_timestamp'],
                analysis_data['total_entries'],
                json.dumps(analysis_data['risk_distribution']),
                json.dumps(analysis_data['phase_distribution']),
                json.dumps(analysis_data['subsystem_analysis']),
                analysis_data['overall_confidence'],
                analysis_data['processing_time'],
                analysis_data.get('format_detected'),
                analysis_data.get('encoding_used'),
                analysis_data.get('ai_summary')
            ))
            
            session_id = session_cursor.lastrowid
            
            # Insert individual entries
            entry_records = []
            for entry in entries_data:
                entry_records.append((
                    session_id,
                    entry['entry_hash'],
                    entry.get('timestamp', ''),
                    entry.get('subsystem', ''),
                    entry['event_text'],
                    entry['predicted_phase'],
                    entry['predicted_risk'],
                    entry['confidence_score'],
                    json.dumps(entry.get('pattern_matches', [])),
                    json.dumps(entry.get('anomaly_indicators', [])),
                    entry.get('processing_time', 0.0),
                    json.dumps(entry.get('line_numbers', [])),
                    entry.get('entry_type', 'single_line')
                ))
            
            conn.executemany('''
                INSERT OR REPLACE INTO log_entries
                (session_id, entry_hash, timestamp, subsystem, event_text, predicted_phase,
                 predicted_risk, confidence_score, pattern_matches, anomaly_indicators,
                 processing_time, line_numbers, entry_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', entry_records)
            
            conn.commit()
            
            # Store trend data
            self._store_trend_data(conn, session_id, analysis_data)
            
            logger.info(f"Stored analysis session {session_id} with {len(entries_data)} entries")
            return session_id
    
    def _store_trend_data(self, conn: sqlite3.Connection, session_id: int, analysis_data: Dict):
        """Store trend analysis data"""
        trend_records = []
        
        # Store subsystem metrics
        for subsystem, metrics in analysis_data.get('subsystem_analysis', {}).items():
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        trend_records.append((
                            subsystem,
                            metric_name,
                            float(metric_value),
                            analysis_data['analysis_timestamp'],
                            session_id,
                            'session'
                        ))
        
        # Store overall metrics
        overall_metrics = {
            'total_entries': analysis_data['total_entries'],
            'overall_confidence': analysis_data['overall_confidence'],
            'processing_time': analysis_data['processing_time']
        }
        
        for metric_name, metric_value in overall_metrics.items():
            trend_records.append((
                'SYSTEM',
                metric_name,
                float(metric_value),
                analysis_data['analysis_timestamp'],
                session_id,
                'session'
            ))
        
        if trend_records:
            conn.executemany('''
                INSERT INTO trend_data 
                (subsystem, metric_name, metric_value, measurement_timestamp, analysis_session_id, trend_period)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', trend_records)
    
    def get_analysis_history(self, limit: int = 100, subsystem: Optional[str] = None,
                           date_from: Optional[datetime] = None, date_to: Optional[datetime] = None) -> List[AnalysisRecord]:
        """Get analysis history with filtering options"""
        
        with self.get_connection() as conn:
            query = '''
                SELECT * FROM analysis_sessions 
                WHERE 1=1
            '''
            params = []
            
            if date_from:
                query += ' AND analysis_timestamp >= ?'
                params.append(date_from.isoformat())
            
            if date_to:
                query += ' AND analysis_timestamp <= ?'
                params.append(date_to.isoformat())
            
            query += ' ORDER BY analysis_timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor = conn.execute(query, params)
            
            records = []
            for row in cursor.fetchall():
                record = AnalysisRecord(
                    id=row['id'],
                    file_path=row['file_path'],
                    file_hash=row['file_hash'],
                    analysis_timestamp=datetime.fromisoformat(row['analysis_timestamp']),
                    total_entries=row['total_entries'],
                    risk_distribution=json.loads(row['risk_distribution']),
                    phase_distribution=json.loads(row['phase_distribution']),
                    subsystem_analysis=json.loads(row['subsystem_analysis']),
                    overall_confidence=row['overall_confidence'],
                    processing_time=row['processing_time'],
                    format_detected=row['format_detected'],
                    encoding_used=row['encoding_used'],
                    ai_summary=row['ai_summary'],
                    user_notes=row['user_notes']
                )
                records.append(record)
            
            return records
    
    def store_user_feedback(self, feedback: UserFeedback) -> int:
        """Store user feedback and trigger learning updates"""
        
        with self.get_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO user_feedback 
                (entry_hash, original_phase, original_risk, corrected_phase, corrected_risk,
                 user_comment, confidence_rating)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.entry_hash,
                feedback.original_phase,
                feedback.original_risk,
                feedback.corrected_phase,
                feedback.corrected_risk,
                feedback.user_comment,
                feedback.confidence_rating
            ))
            
            feedback_id = cursor.lastrowid
            conn.commit()
            
            # Trigger pattern learning update if auto-processing is enabled
            if self.get_config('auto_feedback_processing', 'true') == 'true':
                self._process_feedback_for_learning(conn, feedback)
            
            logger.info(f"Stored user feedback {feedback_id} for entry {feedback.entry_hash}")
            return feedback_id
    
    def _process_feedback_for_learning(self, conn: sqlite3.Connection, feedback: UserFeedback):
        """Process user feedback to update pattern performance"""
        
        # Get the original entry and its patterns
        cursor = conn.execute('''
            SELECT pattern_matches, predicted_phase, predicted_risk
            FROM log_entries WHERE entry_hash = ?
        ''', (feedback.entry_hash,))
        
        result = cursor.fetchone()
        if not result:
            return
        
        pattern_matches = json.loads(result['pattern_matches'] or '[]')
        was_correct = (result['predicted_phase'] == feedback.corrected_phase and 
                      result['predicted_risk'] == feedback.corrected_risk)
        
        # Update pattern performance for each matched pattern
        for pattern in pattern_matches:
            if was_correct:
                conn.execute('''
                    UPDATE pattern_performance 
                    SET success_count = success_count + 1,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE pattern_text = ?
                ''', (pattern,))
            else:
                conn.execute('''
                    UPDATE pattern_performance 
                    SET failure_count = failure_count + 1,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE pattern_text = ?
                ''', (pattern,))
        
        # Mark feedback as processed
        conn.execute('''
            UPDATE user_feedback SET is_processed = 1 WHERE entry_hash = ?
        ''', (feedback.entry_hash,))
        
        conn.commit()
    
    def get_pattern_performance(self, pattern_type: Optional[str] = None, min_samples: int = 5) -> List[PatternPerformance]:
        """Get pattern performance statistics"""
        
        with self.get_connection() as conn:
            query = '''
                SELECT * FROM pattern_performance 
                WHERE (success_count + failure_count) >= ?
            '''
            params = [min_samples]
            
            if pattern_type:
                query += ' AND pattern_type = ?'
                params.append(pattern_type)
            
            query += ' ORDER BY confidence_score DESC'
            
            cursor = conn.execute(query, params)
            
            patterns = []
            for row in cursor.fetchall():
                pattern = PatternPerformance(
                    pattern_id=row['id'],
                    pattern_text=row['pattern_text'],
                    pattern_type=row['pattern_type'],
                    target_value=row['target_value'],
                    success_count=row['success_count'],
                    failure_count=row['failure_count'],
                    confidence_score=row['confidence_score'],
                    last_updated=datetime.fromisoformat(row['last_updated']),
                    created_by=row['created_by']
                )
                patterns.append(pattern)
            
            return patterns
    
    def update_pattern_performance(self, pattern_text: str, pattern_type: str, 
                                 target_value: str, success: bool):
        """Update pattern performance based on usage results"""
        
        with self.get_connection() as conn:
            # Check if pattern exists
            cursor = conn.execute('''
                SELECT id, success_count, failure_count FROM pattern_performance
                WHERE pattern_text = ? AND pattern_type = ? AND target_value = ?
            ''', (pattern_text, pattern_type, target_value))
            
            result = cursor.fetchone()
            
            if result:
                # Update existing pattern
                if success:
                    conn.execute('''
                        UPDATE pattern_performance 
                        SET success_count = success_count + 1, last_updated = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (result['id'],))
                else:
                    conn.execute('''
                        UPDATE pattern_performance 
                        SET failure_count = failure_count + 1, last_updated = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (result['id'],))
            else:
                # Create new pattern record
                success_count = 1 if success else 0
                failure_count = 0 if success else 1
                
                conn.execute('''
                    INSERT INTO pattern_performance 
                    (pattern_text, pattern_type, target_value, success_count, failure_count)
                    VALUES (?, ?, ?, ?, ?)
                ''', (pattern_text, pattern_type, target_value, success_count, failure_count))
            
            conn.commit()
    
    def get_trend_analysis(self, subsystem: str, metric_name: str, 
                          period: str = 'day', limit: int = 30) -> List[TrendData]:
        """Get trend analysis data for a specific metric"""
        
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM trend_data 
                WHERE subsystem = ? AND metric_name = ? AND trend_period = ?
                ORDER BY measurement_timestamp DESC 
                LIMIT ?
            ''', (subsystem, metric_name, period, limit))
            
            trends = []
            for row in cursor.fetchall():
                trend = TrendData(
                    trend_id=row['id'],
                    subsystem=row['subsystem'],
                    metric_name=row['metric_name'],
                    metric_value=row['metric_value'],
                    measurement_timestamp=datetime.fromisoformat(row['measurement_timestamp']),
                    analysis_session_id=row['analysis_session_id'],
                    trend_period=row['trend_period']
                )
                trends.append(trend)
            
            return trends
    
    def get_subsystem_correlation_data(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get data for subsystem correlation analysis"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT le.subsystem, le.predicted_risk, le.timestamp, a_sess.analysis_timestamp
                FROM log_entries le
                JOIN analysis_sessions a_sess ON le.session_id = a_sess.id
                WHERE a_sess.analysis_timestamp >= ?
                ORDER BY a_sess.analysis_timestamp, le.timestamp
            ''', (cutoff_time.isoformat(),))
            
            entries = cursor.fetchall()
            
            # Group by subsystem and analyze correlations
            subsystem_events = defaultdict(list)
            
            for entry in entries:
                subsystem_events[entry['subsystem']].append({
                    'risk': entry['predicted_risk'],
                    'timestamp': entry['timestamp'],
                    'session_timestamp': entry['analysis_timestamp']
                })
            
            return dict(subsystem_events)
    
    def store_system_metric(self, metric_name: str, metric_value: float, 
                           metadata: Optional[Dict] = None):
        """Store system performance metric"""
        
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO system_metrics (metric_name, metric_value, metric_metadata)
                VALUES (?, ?, ?)
            ''', (metric_name, metric_value, json.dumps(metadata) if metadata else None))
            
            conn.commit()
    
    def get_system_metrics(self, metric_name: Optional[str] = None, 
                          hours_back: int = 24) -> List[Dict]:
        """Get system performance metrics"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self.get_connection() as conn:
            if metric_name:
                cursor = conn.execute('''
                    SELECT * FROM system_metrics 
                    WHERE metric_name = ? AND recorded_at >= ?
                    ORDER BY recorded_at DESC
                ''', (metric_name, cutoff_time.isoformat()))
            else:
                cursor = conn.execute('''
                    SELECT * FROM system_metrics 
                    WHERE recorded_at >= ?
                    ORDER BY recorded_at DESC
                ''', (cutoff_time.isoformat(),))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        
        with self.get_connection() as conn:
            cursor = conn.execute('''
                SELECT value, value_type FROM system_config WHERE key = ?
            ''', (key,))
            
            result = cursor.fetchone()
            if not result:
                return default
            
            value, value_type = result['value'], result['value_type']
            
            # Convert based on type
            if value_type == 'int':
                return int(value)
            elif value_type == 'float':
                return float(value)
            elif value_type == 'json':
                return json.loads(value)
            else:
                return value
    
    def set_config(self, key: str, value: Any, value_type: str, description: str = ""):
        """Set configuration value"""
        
        with self.get_connection() as conn:
            if value_type == 'json':
                value_str = json.dumps(value)
            else:
                value_str = str(value)
            
            conn.execute('''
                INSERT OR REPLACE INTO system_config (key, value, value_type, description)
                VALUES (?, ?, ?, ?)
            ''', (key, value_str, value_type, description))
            
            conn.commit()
    
    def cleanup_old_data(self, retention_days: Optional[int] = None):
        """Clean up old data based on retention policy"""
        
        if retention_days is None:
            retention_days = self.get_config('retention_days', 90)
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        with self.get_connection() as conn:
            # Clean up old analysis sessions and related data
            cursor = conn.execute('''
                SELECT id FROM analysis_sessions 
                WHERE analysis_timestamp < ?
            ''', (cutoff_date.isoformat(),))
            
            old_session_ids = [row['id'] for row in cursor.fetchall()]
            
            if old_session_ids:
                # Delete related data
                placeholders = ','.join(['?'] * len(old_session_ids))
                
                conn.execute(f'''
                    DELETE FROM log_entries WHERE session_id IN ({placeholders})
                ''', old_session_ids)
                
                conn.execute(f'''
                    DELETE FROM trend_data WHERE analysis_session_id IN ({placeholders})
                ''', old_session_ids)
                
                conn.execute(f'''
                    DELETE FROM analysis_sessions WHERE id IN ({placeholders})
                ''', old_session_ids)
                
                # Clean up old system metrics
                conn.execute('''
                    DELETE FROM system_metrics WHERE recorded_at < ?
                ''', (cutoff_date.isoformat(),))
                
                conn.commit()
                
                logger.info(f"Cleaned up {len(old_session_ids)} old analysis sessions")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        
        with self.get_connection() as conn:
            stats = {}
            
            # Table counts
            tables = ['analysis_sessions', 'log_entries', 'pattern_performance', 
                     'user_feedback', 'trend_data', 'system_metrics']
            
            for table in tables:
                cursor = conn.execute(f'SELECT COUNT(*) as count FROM {table}')
                stats[f'{table}_count'] = cursor.fetchone()['count']
            
            # Database size
            cursor = conn.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            stats['database_size_bytes'] = cursor.fetchone()['size']
            
            # Recent activity
            cursor = conn.execute('''
                SELECT COUNT(*) as recent_sessions 
                FROM analysis_sessions 
                WHERE analysis_timestamp >= datetime('now', '-7 days')
            ''')
            stats['recent_sessions'] = cursor.fetchone()['recent_sessions']
            
            # Pattern performance summary
            cursor = conn.execute('''
                SELECT 
                    AVG(confidence_score) as avg_confidence,
                    COUNT(*) as total_patterns,
                    SUM(success_count) as total_successes,
                    SUM(failure_count) as total_failures
                FROM pattern_performance
            ''')
            
            perf_result = cursor.fetchone()
            stats.update({
                'avg_pattern_confidence': perf_result['avg_confidence'] or 0,
                'total_patterns': perf_result['total_patterns'],
                'total_pattern_successes': perf_result['total_successes'] or 0,
                'total_pattern_failures': perf_result['total_failures'] or 0
            })
            
            return stats
    
    def export_analysis_data(self, output_path: str, format_type: str = 'json', 
                           session_ids: Optional[List[int]] = None):
        """Export analysis data for backup or analysis"""
        
        with self.get_connection() as conn:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'database_stats': self.get_database_stats(),
                'analysis_sessions': [],
                'pattern_performance': [],
                'user_feedback': []
            }
            
            # Export analysis sessions
            if session_ids:
                placeholders = ','.join(['?'] * len(session_ids))
                session_query = f'SELECT * FROM analysis_sessions WHERE id IN ({placeholders})'
                session_params = session_ids
            else:
                session_query = 'SELECT * FROM analysis_sessions ORDER BY analysis_timestamp DESC'
                session_params = []
            
            cursor = conn.execute(session_query, session_params)
            for row in cursor.fetchall():
                session_data = dict(row)
                # Parse JSON fields
                session_data['risk_distribution'] = json.loads(session_data['risk_distribution'])
                session_data['phase_distribution'] = json.loads(session_data['phase_distribution'])
                session_data['subsystem_analysis'] = json.loads(session_data['subsystem_analysis'])
                export_data['analysis_sessions'].append(session_data)
            
            # Export pattern performance
            cursor = conn.execute('SELECT * FROM pattern_performance ORDER BY confidence_score DESC')
            export_data['pattern_performance'] = [dict(row) for row in cursor.fetchall()]
            
            # Export user feedback
            cursor = conn.execute('SELECT * FROM user_feedback ORDER BY feedback_timestamp DESC')
            export_data['user_feedback'] = [dict(row) for row in cursor.fetchall()]
            
            # Write to file
            if format_type.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            logger.info(f"Exported analysis data to {output_path}")
    
    def close(self):
        """Close all database connections"""
        with self.pool_lock:
            for conn in self.connection_pool:
                conn.close()
            self.connection_pool.clear()
        
        logger.info("Database manager closed")


# Global database manager instance
_db_manager = None

def get_database_manager(db_path: str = "lognarrator_production.db") -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_path)
    return _db_manager 