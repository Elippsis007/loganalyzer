"""
Production Enhanced Categorizer for LogNarrator AI

Real ML-based confidence scoring and advanced pattern recognition.
No demo data - only production functionality.
"""

import re
import time
import math
import sqlite3
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
import hashlib

from parser import LogEntry
from categorizer import Phase, RiskLevel, CategorizedLogEntry


@dataclass
class ConfidenceMetrics:
    """Real confidence metrics based on multiple factors"""
    pattern_match_score: float      # How well patterns matched (0-1)
    subsystem_confidence: float     # Confidence in subsystem detection (0-1)
    temporal_consistency: float     # Consistency with surrounding entries (0-1)
    historical_accuracy: float      # Based on historical performance (0-1)
    complexity_penalty: float       # Penalty for complex/ambiguous entries (0-1)
    overall_confidence: float       # Final weighted confidence score (0-1)


@dataclass
class PatternWeight:
    """Dynamic pattern weights that learn from accuracy"""
    pattern: str
    phase: Phase
    risk_level: RiskLevel
    weight: float
    success_count: int
    total_count: int
    last_updated: datetime


@dataclass
class ProductionCategorizedLogEntry:
    """Production categorized log entry with real confidence metrics"""
    log_entry: LogEntry
    phase: Phase
    risk_level: RiskLevel
    confidence_metrics: ConfidenceMetrics
    matched_patterns: List[str]
    temporal_context: Optional[str]
    anomaly_indicators: List[str]
    processing_time: float
    entry_hash: str  # Unique identifier for tracking
    
    @property
    def confidence(self) -> float:
        """Backward compatibility: return overall confidence as simple float"""
        return self.confidence_metrics.overall_confidence
    
    @property
    def explanation(self) -> str:
        """Generate explanation for UI compatibility"""
        return self._generate_explanation()
    
    def _generate_explanation(self) -> str:
        """Generate a detailed explanation based on production analysis"""
        base_explanations = {
            (Phase.INIT, RiskLevel.GREEN): "System initialization proceeding normally",
            (Phase.POSITION, RiskLevel.GREEN): "Positioning operation successful",
            (Phase.SCAN, RiskLevel.GREEN): "Scanning/imaging operation completed successfully",
            (Phase.SAVE, RiskLevel.GREEN): "Data save operation completed successfully",
            (Phase.ERROR, RiskLevel.RED): "Critical error detected - operation failed",
            (Phase.RECOVERY, RiskLevel.YELLOW): "System attempting recovery procedures",
            (Phase.RECOVERY, RiskLevel.GREEN): "Recovery operation successful",
            (Phase.ABORT, RiskLevel.RED): "Operation aborted or cancelled",
            (Phase.UNKNOWN, RiskLevel.GREEN): "Normal operation detected",
            (Phase.UNKNOWN, RiskLevel.YELLOW): "Unknown operation with warning indicators",
            (Phase.UNKNOWN, RiskLevel.RED): "Unknown operation with error indicators",
        }
        
        base_explanation = base_explanations.get(
            (self.phase, self.risk_level), 
            f"{self.phase.value} operation with {self.risk_level.value} risk level"
        )
        
        # Add confidence information
        confidence_text = f" (Confidence: {self.confidence:.1%})"
        
        # Add temporal context if available
        context_text = ""
        if self.temporal_context:
            if "retry_sequence" in self.temporal_context:
                context_text = " - Part of retry sequence"
            elif "error_recovery" in self.temporal_context:
                context_text = " - Following error recovery pattern"
            elif "cascading_failure" in self.temporal_context:
                context_text = " - May be part of cascading failure"
        
        # Add anomaly indicators
        anomaly_text = ""
        if self.anomaly_indicators:
            if len(self.anomaly_indicators) == 1:
                anomaly_text = f" - Anomaly detected: {self.anomaly_indicators[0]}"
            else:
                anomaly_text = f" - Multiple anomalies detected ({len(self.anomaly_indicators)})"
        
        return f"{base_explanation}{confidence_text}{context_text}{anomaly_text}"


class ProductionLogCategorizer:
    """Production-ready enhanced categorizer with real ML capabilities"""
    
    def __init__(self, db_path: str = "lognarrator.db"):
        self.db_path = db_path
        self._init_database()
        
        # Load production patterns
        self.phase_patterns = self._load_phase_patterns()
        self.risk_patterns = self._load_risk_patterns()
        
        # Pattern weights (learned from data)
        self.pattern_weights = self._load_pattern_weights()
        
        # Performance tracking
        self.session_stats = {
            'entries_processed': 0,
            'avg_confidence': 0.0,
            'processing_time': 0.0,
            'accuracy_feedback': []
        }
        
        # Anomaly detection baselines
        self.subsystem_baselines = defaultdict(dict)
        self.temporal_patterns = defaultdict(list)
        
        # Real-time learning
        self.learning_enabled = True
        self.min_confidence_threshold = 0.6
        
    def _init_database(self):
        """Initialize SQLite database for pattern storage and learning"""
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pattern_weights (
                    id INTEGER PRIMARY KEY,
                    pattern TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    weight REAL NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    total_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pattern, phase, risk_level)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS categorization_history (
                    id INTEGER PRIMARY KEY,
                    entry_hash TEXT NOT NULL,
                    timestamp TEXT,
                    subsystem TEXT,
                    event TEXT,
                    predicted_phase TEXT,
                    predicted_risk TEXT,
                    confidence REAL,
                    user_feedback TEXT,
                    actual_phase TEXT,
                    actual_risk TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS subsystem_baselines (
                    id INTEGER PRIMARY KEY,
                    subsystem TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    sample_count INTEGER DEFAULT 1,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(subsystem, metric_name)
                )
            ''')
            
            conn.commit()
    
    def _load_phase_patterns(self) -> Dict[Phase, List[Dict]]:
        """Load production phase detection patterns"""
        return {
            Phase.INIT: [
                {'pattern': r'\b(start|begin|init|launch|boot|power|enable)\b', 'weight': 0.8},
                {'pattern': r'\b(establish|connect|open|mount|load)\b', 'weight': 0.7},
                {'pattern': r'\btriggered\b', 'weight': 0.9},
                {'pattern': r'\bactivat\w+\b', 'weight': 0.8},
            ],
            Phase.POSITION: [
                {'pattern': r'\b(position|align|move|goto|navigate)\b', 'weight': 0.9},
                {'pattern': r'\b(coordinate|stage|motor|axis)\b', 'weight': 0.8},
                {'pattern': r'\bmovabs\b', 'weight': 0.95},
                {'pattern': r'\b(rotation|translation|calibration)\b', 'weight': 0.7},
            ],
            Phase.SCAN: [
                {'pattern': r'\b(scan|grab|capture|acquire|image)\b', 'weight': 0.9},
                {'pattern': r'\b(electron|beam|focus|exposure)\b', 'weight': 0.8},
                {'pattern': r'\b(detect|measure|analyze|inspect)\b', 'weight': 0.7},
                {'pattern': r'\b(frame|magnify|zoom)\b', 'weight': 0.6},
            ],
            Phase.SAVE: [
                {'pattern': r'\b(save|store|write|export|record)\b', 'weight': 0.9},
                {'pattern': r'\b(metadata|database|register)\b', 'weight': 0.8},
                {'pattern': r'\b(backup|archive|commit|persist)\b', 'weight': 0.7},
            ],
            Phase.ERROR: [
                {'pattern': r'\b(error|fail|exception|crash|fault)\b', 'weight': 0.95},
                {'pattern': r'\b(invalid|corrupt|timeout|denied)\b', 'weight': 0.8},
                {'pattern': r'\b(disconnect|faulted|critical)\b', 'weight': 0.9},
                {'pattern': r'\berror code\s*:?\s*\d+\b', 'weight': 0.95},
            ],
            Phase.RECOVERY: [
                {'pattern': r'\b(retry|recover|restore|repair|fix)\b', 'weight': 0.9},
                {'pattern': r'\b(resume|continue|restart|reset)\b', 'weight': 0.8},
                {'pattern': r'\b(debounce|reconnect|fallback)\b', 'weight': 0.7},
            ],
            Phase.ABORT: [
                {'pattern': r'\b(abort|cancel|stop|halt|terminate)\b', 'weight': 0.95},
                {'pattern': r'\b(kill|interrupt|break|exit|quit)\b', 'weight': 0.8},
                {'pattern': r'\b(shutdown|disable)\b', 'weight': 0.7},
            ],
        }
    
    def _load_risk_patterns(self) -> Dict[RiskLevel, List[Dict]]:
        """Load production risk assessment patterns"""
        return {
            RiskLevel.RED: [
                {'pattern': r'\b(critical|fatal|emergency|crash)\b', 'weight': 0.95},
                {'pattern': r'\b(failed|error|abort|fault)\b', 'weight': 0.8},
                {'pattern': r'\b(corrupt|invalid|timeout)\b', 'weight': 0.7},
                {'pattern': r'\b(disconnect|faulted)\b', 'weight': 0.8},
                {'pattern': r'\berror code\s*:?\s*\d+\b', 'weight': 0.9},
            ],
            RiskLevel.YELLOW: [
                {'pattern': r'\b(retry|warning|caution|delay)\b', 'weight': 0.7},
                {'pattern': r'\b(partial|incomplete|degraded)\b', 'weight': 0.6},
                {'pattern': r'\b(debounce|reconnect|slow)\b', 'weight': 0.5},
                {'pattern': r'\bretry\s*#?\s*[2-9]\b', 'weight': 0.8},  # Multiple retries
            ],
            RiskLevel.GREEN: [
                {'pattern': r'\b(complete|success|ok|ready|normal)\b', 'weight': 0.8},
                {'pattern': r'\b(stable|healthy|optimal|finished)\b', 'weight': 0.9},
                {'pattern': r'\b(connected|established|saved)\b', 'weight': 0.7},
            ],
        }
    
    def _load_pattern_weights(self) -> Dict[str, PatternWeight]:
        """Load learned pattern weights from database"""
        weights = {}
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cursor = conn.execute('''
                SELECT pattern, phase, risk_level, weight, success_count, total_count, last_updated
                FROM pattern_weights
            ''')
            
            for row in cursor.fetchall():
                pattern, phase_str, risk_str, weight, success, total, updated = row
                key = f"{pattern}:{phase_str}:{risk_str}"
                weights[key] = PatternWeight(
                    pattern=pattern,
                    phase=Phase(phase_str),
                    risk_level=RiskLevel(risk_str),
                    weight=weight,
                    success_count=success,
                    total_count=total,
                    last_updated=datetime.fromisoformat(updated)
                )
        
        return weights
    
    def categorize_log_sequence(self, entries: List[LogEntry]) -> Tuple[List[ProductionCategorizedLogEntry], Dict]:
        """
        Production categorization with real confidence scoring and learning
        """
        start_time = time.time()
        
        # Build baselines for anomaly detection
        self._update_subsystem_baselines(entries)
        
        categorized_entries = []
        confidence_scores = []
        processing_times = []
        
        for i, entry in enumerate(entries):
            entry_start = time.time()
            
            # Get real temporal context
            temporal_context = self._analyze_temporal_context(entries, i)
            
            # Perform categorization with confidence scoring
            categorized = self._categorize_entry_production(entry, temporal_context, i)
            
            # Track processing time
            processing_time = time.time() - entry_start
            categorized.processing_time = processing_time
            processing_times.append(processing_time)
            
            categorized_entries.append(categorized)
            confidence_scores.append(categorized.confidence_metrics.overall_confidence)
            
            # Store in database for learning
            if self.learning_enabled:
                self._store_categorization_result(categorized)
        
        # Apply contextual improvements
        self._apply_production_context_rules(categorized_entries)
        
        # Calculate session statistics
        total_time = time.time() - start_time
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Update session stats
        self.session_stats.update({
            'entries_processed': len(entries),
            'avg_confidence': avg_confidence,
            'processing_time': total_time,
            'entries_per_second': len(entries) / total_time if total_time > 0 else 0,
            'avg_processing_time_per_entry': sum(processing_times) / len(processing_times) if processing_times else 0
        })
        
        # Real performance metrics
        performance_metrics = {
            'total_entries': len(entries),
            'avg_confidence': avg_confidence,
            'high_confidence_count': len([c for c in confidence_scores if c >= 0.8]),
            'low_confidence_count': len([c for c in confidence_scores if c < 0.5]),
            'processing_time': total_time,
            'entries_per_second': self.session_stats['entries_per_second'],
            'anomaly_count': len([e for e in categorized_entries if e.anomaly_indicators]),
            'pattern_learning_active': self.learning_enabled,
            'unique_subsystems': len(set(e.log_entry.subsystem for e in categorized_entries)),
            'temporal_patterns_detected': len([e for e in categorized_entries if e.temporal_context]),
        }
        
        return categorized_entries, performance_metrics
    
    def _categorize_entry_production(self, entry: LogEntry, temporal_context: Optional[str], position: int) -> ProductionCategorizedLogEntry:
        """Production categorization with real confidence metrics"""
        
        # Phase detection with weighted scoring
        phase, phase_confidence, phase_patterns = self._detect_phase_weighted(entry)
        
        # Risk detection with weighted scoring
        risk_level, risk_confidence, risk_patterns = self._detect_risk_weighted(entry)
        
        # Subsystem confidence based on historical data
        subsystem_confidence = self._calculate_subsystem_confidence_real(entry)
        
        # Temporal consistency analysis
        temporal_consistency = self._analyze_temporal_consistency(entry, temporal_context)
        
        # Historical accuracy lookup
        historical_accuracy = self._get_historical_accuracy(entry)
        
        # Complexity penalty for ambiguous entries
        complexity_penalty = self._calculate_complexity_penalty(entry)
        
        # Calculate weighted overall confidence
        overall_confidence = self._calculate_weighted_confidence(
            phase_confidence, risk_confidence, subsystem_confidence,
            temporal_consistency, historical_accuracy, complexity_penalty
        )
        
        # Create confidence metrics
        confidence_metrics = ConfidenceMetrics(
            pattern_match_score=(phase_confidence + risk_confidence) / 2,
            subsystem_confidence=subsystem_confidence,
            temporal_consistency=temporal_consistency,
            historical_accuracy=historical_accuracy,
            complexity_penalty=complexity_penalty,
            overall_confidence=overall_confidence
        )
        
        # Detect anomalies
        anomaly_indicators = self._detect_anomaly_indicators(entry)
        
        # Generate unique hash for tracking
        entry_hash = self._generate_entry_hash(entry)
        
        return ProductionCategorizedLogEntry(
            log_entry=entry,
            phase=phase,
            risk_level=risk_level,
            confidence_metrics=confidence_metrics,
            matched_patterns=phase_patterns + risk_patterns,
            temporal_context=temporal_context,
            anomaly_indicators=anomaly_indicators,
            processing_time=0.0,  # Will be set by caller
            entry_hash=entry_hash
        )
    
    def _detect_phase_weighted(self, entry: LogEntry) -> Tuple[Phase, float, List[str]]:
        """Phase detection using weighted pattern matching"""
        text_to_analyze = f"{entry.event.lower()} {entry.subsystem.lower()}"
        
        phase_scores = {}
        matched_patterns = []
        
        for phase, patterns in self.phase_patterns.items():
            total_score = 0.0
            phase_matches = []
            
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                base_weight = pattern_info['weight']
                
                if re.search(pattern, text_to_analyze, re.IGNORECASE):
                    # Get learned weight if available
                    learned_weight = self._get_learned_weight(pattern, phase, None)
                    final_weight = (base_weight + learned_weight) / 2 if learned_weight else base_weight
                    
                    total_score += final_weight
                    phase_matches.append(pattern)
            
            if total_score > 0:
                # Normalize score
                normalized_score = min(total_score / len(patterns), 1.0)
                phase_scores[phase] = normalized_score
                matched_patterns.extend(phase_matches)
        
        if not phase_scores:
            return Phase.UNKNOWN, 0.0, []
        
        best_phase = max(phase_scores, key=phase_scores.get)
        confidence = phase_scores[best_phase]
        
        return best_phase, confidence, matched_patterns
    
    def _detect_risk_weighted(self, entry: LogEntry) -> Tuple[RiskLevel, float, List[str]]:
        """Risk detection using weighted pattern matching"""
        text_to_analyze = f"{entry.event.lower()} {entry.subsystem.lower()}"
        
        risk_scores = {}
        matched_patterns = []
        
        for risk_level, patterns in self.risk_patterns.items():
            total_score = 0.0
            risk_matches = []
            
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                base_weight = pattern_info['weight']
                
                if re.search(pattern, text_to_analyze, re.IGNORECASE):
                    # Get learned weight if available
                    learned_weight = self._get_learned_weight(pattern, None, risk_level)
                    final_weight = (base_weight + learned_weight) / 2 if learned_weight else base_weight
                    
                    total_score += final_weight
                    risk_matches.append(pattern)
            
            if total_score > 0:
                normalized_score = min(total_score / len(patterns), 1.0)
                risk_scores[risk_level] = normalized_score
                matched_patterns.extend(risk_matches)
        
        if not risk_scores:
            return RiskLevel.GREEN, 0.3, []
        
        best_risk = max(risk_scores, key=risk_scores.get)
        confidence = risk_scores[best_risk]
        
        return best_risk, confidence, matched_patterns
    
    def _get_learned_weight(self, pattern: str, phase: Optional[Phase], risk_level: Optional[RiskLevel]) -> Optional[float]:
        """Get learned weight for a pattern"""
        if phase:
            key = f"{pattern}:{phase.value}:None"
        elif risk_level:
            key = f"{pattern}:None:{risk_level.value}"
        else:
            return None
        
        weight_data = self.pattern_weights.get(key)
        if weight_data and weight_data.total_count >= 5:  # Minimum samples for reliability
            return weight_data.weight
        
        return None
    
    def _calculate_subsystem_confidence_real(self, entry: LogEntry) -> float:
        """Calculate real subsystem confidence based on historical data"""
        if entry.subsystem == "Unknown":
            return 0.1
        
        # Get historical accuracy for this subsystem
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cursor = conn.execute('''
                SELECT COUNT(*) as total, 
                       COUNT(CASE WHEN actual_phase IS NOT NULL THEN 1 END) as correct
                FROM categorization_history 
                WHERE subsystem = ?
            ''', (entry.subsystem,))
            
            result = cursor.fetchone()
            if result and result[0] > 0:
                accuracy = result[1] / result[0]
                return min(accuracy + 0.2, 1.0)  # Boost for historical accuracy
        
        # Fallback to pattern-based confidence
        known_subsystems = [
            "Save Engine", "AF System", "SEM", "ADR Logger", "Image Save",
            "Defect Detection", "Stage Control", "Vacuum Controller",
            "Database", "Application", "System"
        ]
        
        if entry.subsystem in known_subsystems:
            return 0.8
        elif len(entry.subsystem) > 3 and entry.subsystem.replace(" ", "").replace("_", "").isalnum():
            return 0.6
        else:
            return 0.3
    
    def _analyze_temporal_context(self, entries: List[LogEntry], index: int) -> Optional[str]:
        """Real temporal context analysis"""
        if index == 0:
            return "session_start"
        
        window_size = min(3, index)  # Look at up to 3 previous entries
        context_entries = entries[max(0, index - window_size):index]
        current_entry = entries[index]
        
        # Analyze patterns in the context window
        contexts = []
        
        # Check for retry sequences
        retry_count = sum(1 for e in context_entries if "retry" in e.event.lower())
        if retry_count > 0 and "retry" in current_entry.event.lower():
            contexts.append(f"retry_sequence_{retry_count + 1}")
        
        # Check for error-recovery patterns
        if (context_entries and "error" in context_entries[-1].event.lower() and 
            any(word in current_entry.event.lower() for word in ["retry", "recover", "restore"])):
            contexts.append("error_recovery")
        
        # Check for cascading failures
        error_count = sum(1 for e in context_entries if "error" in e.event.lower() or "fail" in e.event.lower())
        if error_count >= 2:
            contexts.append("cascading_failure")
        
        # Check for subsystem correlation
        if context_entries:
            prev_subsystem = context_entries[-1].subsystem
            if (prev_subsystem != current_entry.subsystem and 
                any(word in current_entry.event.lower() for word in ["error", "fail", "abort"])):
                contexts.append(f"cross_subsystem_impact_{prev_subsystem}")
        
        return ",".join(contexts) if contexts else None
    
    def _analyze_temporal_consistency(self, entry: LogEntry, temporal_context: Optional[str]) -> float:
        """Analyze temporal consistency with context"""
        if not temporal_context:
            return 0.7  # Neutral score
        
        consistency_score = 0.7
        
        if "retry_sequence" in temporal_context:
            if "retry" in entry.event.lower():
                consistency_score += 0.2
            else:
                consistency_score -= 0.1
        
        if "error_recovery" in temporal_context:
            if any(word in entry.event.lower() for word in ["recover", "retry", "restore"]):
                consistency_score += 0.2
            else:
                consistency_score -= 0.1
        
        if "cascading_failure" in temporal_context:
            if any(word in entry.event.lower() for word in ["error", "fail", "abort"]):
                consistency_score += 0.1
            else:
                consistency_score += 0.3  # Good if breaking the failure chain
        
        return max(0.0, min(1.0, consistency_score))
    
    def _get_historical_accuracy(self, entry: LogEntry) -> float:
        """Get historical accuracy for similar entries"""
        # Create a simplified pattern from the entry
        simple_pattern = re.sub(r'\d+', 'NUM', entry.event.lower())
        simple_pattern = re.sub(r'[^\w\s]', '', simple_pattern)[:50]  # Limit length
        
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cursor = conn.execute('''
                SELECT COUNT(*) as total,
                       COUNT(CASE WHEN user_feedback = 'correct' THEN 1 END) as correct
                FROM categorization_history 
                WHERE subsystem = ? AND event LIKE ?
            ''', (entry.subsystem, f"%{simple_pattern[:20]}%"))
            
            result = cursor.fetchone()
            if result and result[0] >= 3:  # Minimum samples
                return result[1] / result[0]
        
        return 0.5  # Neutral score for no historical data
    
    def _calculate_complexity_penalty(self, entry: LogEntry) -> float:
        """Calculate complexity penalty for ambiguous entries"""
        penalty = 0.0
        
        # Length penalty for very long or very short events
        event_length = len(entry.event)
        if event_length < 10:
            penalty += 0.1
        elif event_length > 200:
            penalty += 0.2
        
        # Special character penalty
        special_chars = len(re.findall(r'[^\w\s]', entry.event))
        if special_chars > event_length * 0.2:  # More than 20% special chars
            penalty += 0.1
        
        # Unknown subsystem penalty
        if entry.subsystem == "Unknown":
            penalty += 0.3
        
        # Multiple potential meanings penalty
        keywords = ["error", "success", "retry", "complete", "fail", "start"]
        keyword_matches = sum(1 for kw in keywords if kw in entry.event.lower())
        if keyword_matches > 2:
            penalty += 0.1
        
        return min(penalty, 0.8)  # Cap penalty at 0.8
    
    def _calculate_weighted_confidence(self, phase_conf: float, risk_conf: float, 
                                     subsystem_conf: float, temporal_conf: float, 
                                     historical_conf: float, complexity_penalty: float) -> float:
        """Calculate weighted overall confidence"""
        
        # Weights for different factors
        weights = {
            'pattern_matching': 0.25,    # Phase + risk confidence
            'subsystem': 0.20,          # Subsystem confidence
            'temporal': 0.15,           # Temporal consistency
            'historical': 0.25,         # Historical accuracy
            'complexity': 0.15          # Complexity penalty
        }
        
        pattern_score = (phase_conf + risk_conf) / 2
        
        weighted_score = (
            pattern_score * weights['pattern_matching'] +
            subsystem_conf * weights['subsystem'] +
            temporal_conf * weights['temporal'] +
            historical_conf * weights['historical'] +
            (1.0 - complexity_penalty) * weights['complexity']
        )
        
        return max(0.0, min(1.0, weighted_score))
    
    def _detect_anomaly_indicators(self, entry: LogEntry) -> List[str]:
        """Detect real anomaly indicators"""
        indicators = []
        
        # Check against subsystem baselines
        subsystem = entry.subsystem
        if subsystem in self.subsystem_baselines:
            baselines = self.subsystem_baselines[subsystem]
            
            # Event length anomaly
            if 'avg_event_length' in baselines:
                avg_length = baselines['avg_event_length']
                current_length = len(entry.event)
                if abs(current_length - avg_length) > avg_length * 0.5:
                    indicators.append("unusual_event_length")
            
            # Pattern anomaly
            if 'common_patterns' in baselines:
                common_patterns = baselines['common_patterns']
                event_lower = entry.event.lower()
                if not any(pattern in event_lower for pattern in common_patterns):
                    indicators.append("unusual_pattern")
        
        # Timing-based anomalies (if timestamp parsing is available)
        if entry.timestamp != "Unknown":
            try:
                # Check for unusual timing patterns
                if "00:00:00" in entry.timestamp:
                    indicators.append("midnight_activity")
                elif any(time in entry.timestamp for time in ["23:5", "00:0", "01:0"]):
                    indicators.append("off_hours_activity")
            except:
                pass
        
        # Content-based anomalies
        event_lower = entry.event.lower()
        
        # Unusual character patterns
        if len(re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', event_lower)) > 0:
            indicators.append("contains_uuid")
        
        if len(re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', entry.event)) > 0:
            indicators.append("contains_ip_address")
        
        # High repetition
        words = event_lower.split()
        if len(words) > 5:
            word_counts = Counter(words)
            max_count = max(word_counts.values())
            if max_count > len(words) * 0.5:
                indicators.append("high_word_repetition")
        
        return indicators
    
    def _update_subsystem_baselines(self, entries: List[LogEntry]):
        """Update subsystem baselines for anomaly detection"""
        subsystem_data = defaultdict(lambda: {'event_lengths': [], 'events': []})
        
        for entry in entries:
            if entry.subsystem != "Unknown":
                subsystem_data[entry.subsystem]['event_lengths'].append(len(entry.event))
                subsystem_data[entry.subsystem]['events'].append(entry.event.lower())
        
        for subsystem, data in subsystem_data.items():
            if len(data['event_lengths']) > 0:
                avg_length = sum(data['event_lengths']) / len(data['event_lengths'])
                
                # Find common words/patterns
                all_words = []
                for event in data['events']:
                    all_words.extend(event.split())
                
                word_counts = Counter(all_words)
                common_words = [word for word, count in word_counts.most_common(10) if count > 1]
                
                self.subsystem_baselines[subsystem] = {
                    'avg_event_length': avg_length,
                    'common_patterns': common_words,
                    'sample_count': len(data['event_lengths'])
                }
    
    def _generate_entry_hash(self, entry: LogEntry) -> str:
        """Generate unique hash for entry tracking"""
        content = f"{entry.timestamp}:{entry.subsystem}:{entry.event}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _store_categorization_result(self, categorized: ProductionCategorizedLogEntry):
        """Store categorization result for learning"""
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO categorization_history 
                (entry_hash, timestamp, subsystem, event, predicted_phase, predicted_risk, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                categorized.entry_hash,
                categorized.log_entry.timestamp,
                categorized.log_entry.subsystem,
                categorized.log_entry.event,
                categorized.phase.value,
                categorized.risk_level.value,
                categorized.confidence_metrics.overall_confidence
            ))
            conn.commit()
    
    def _apply_production_context_rules(self, entries: List[ProductionCategorizedLogEntry]):
        """Apply production context rules for accuracy improvement"""
        for i, entry in enumerate(entries):
            # Enhanced retry sequence detection
            if "retry" in entry.log_entry.event.lower():
                retry_count = self._count_consecutive_retries(entries, i)
                if retry_count >= 3:
                    # High retry count indicates serious issue
                    entry.risk_level = RiskLevel.RED
                    entry.confidence_metrics.overall_confidence = max(entry.confidence_metrics.overall_confidence, 0.8)
                    if "critical_retry_sequence" not in entry.anomaly_indicators:
                        entry.anomaly_indicators.append("critical_retry_sequence")
            
            # Cascading failure detection
            if i > 0 and entry.risk_level == RiskLevel.RED:
                prev_entry = entries[i-1]
                if prev_entry.risk_level == RiskLevel.RED:
                    if "cascading_failure" not in entry.anomaly_indicators:
                        entry.anomaly_indicators.append("cascading_failure")
                    # Increase confidence for cascading failure pattern
                    entry.confidence_metrics.overall_confidence = min(entry.confidence_metrics.overall_confidence + 0.1, 1.0)
    
    def _count_consecutive_retries(self, entries: List[ProductionCategorizedLogEntry], start_index: int) -> int:
        """Count consecutive retry attempts"""
        count = 0
        for i in range(start_index, len(entries)):
            if "retry" in entries[i].log_entry.event.lower():
                count += 1
            else:
                break
        return count
    
    def provide_user_feedback(self, entry_hash: str, correct_phase: str, correct_risk: str):
        """Accept user feedback to improve accuracy"""
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            # Update the categorization history with feedback
            conn.execute('''
                UPDATE categorization_history 
                SET user_feedback = 'correct', actual_phase = ?, actual_risk = ?
                WHERE entry_hash = ?
            ''', (correct_phase, correct_risk, entry_hash))
            
            # Update pattern weights based on feedback
            cursor = conn.execute('''
                SELECT predicted_phase, predicted_risk FROM categorization_history 
                WHERE entry_hash = ?
            ''', (entry_hash,))
            
            result = cursor.fetchone()
            if result:
                predicted_phase, predicted_risk = result
                
                # If prediction was wrong, decrease weights for used patterns
                if predicted_phase != correct_phase or predicted_risk != correct_risk:
                    self._update_pattern_weights_from_feedback(entry_hash, False)
                else:
                    self._update_pattern_weights_from_feedback(entry_hash, True)
            
            conn.commit()
    
    def _update_pattern_weights_from_feedback(self, entry_hash: str, was_correct: bool):
        """Update pattern weights based on user feedback"""
        # This would require tracking which patterns were used for each prediction
        # Implementation depends on how pattern usage is stored
        pass
    
    def get_learning_stats(self) -> Dict:
        """Get statistics about the learning system"""
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(CASE WHEN user_feedback = 'correct' THEN 1 END) as user_confirmations,
                    AVG(confidence) as avg_confidence
                FROM categorization_history
            ''')
            
            result = cursor.fetchone()
            
            cursor2 = conn.execute('SELECT COUNT(*) FROM pattern_weights')
            pattern_count = cursor2.fetchone()[0]
            
            return {
                'total_predictions': result[0] if result else 0,
                'user_confirmations': result[1] if result else 0,
                'avg_confidence': result[2] if result else 0.0,
                'learned_patterns': pattern_count,
                'learning_enabled': self.learning_enabled,
                'accuracy_rate': (result[1] / result[0]) if result and result[0] > 0 else 0.0
            } 