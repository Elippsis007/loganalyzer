"""
Enhanced Categorizer for LogNarrator AI

This module provides improved categorization with confidence scoring,
performance metrics, and advanced pattern recognition.
"""

import time
import re
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
from parser import LogEntry
from categorizer import Phase, RiskLevel, CategorizedLogEntry


@dataclass
class PerformanceMetrics:
    """Performance metrics for categorization"""
    parsing_time: float
    categorization_time: float
    total_entries: int
    successful_categorizations: int
    average_confidence: float
    entries_per_second: float
    confidence_distribution: Dict[str, int]


@dataclass
class EnhancedCategorizedLogEntry:
    """Enhanced categorized log entry with confidence scoring"""
    log_entry: LogEntry
    phase: Phase
    risk_level: RiskLevel
    confidence: float
    explanation: str
    pattern_matches: List[str]  # Which patterns matched
    subsystem_confidence: float  # Confidence in subsystem detection
    temporal_context: Optional[str]  # Context from surrounding entries
    anomaly_score: float  # How unusual this entry is (0-1)


class EnhancedLogCategorizer:
    """Enhanced categorizer with advanced features"""
    
    def __init__(self):
        # Load existing patterns from base categorizer
        self.phase_keywords = {
            Phase.INIT: [
                "triggered", "start", "begin", "initialize", "init", "startup", 
                "launch", "activate", "enable", "power", "boot", "loading",
                "mounting", "connecting", "establishing", "opening"
            ],
            Phase.POSITION: [
                "position", "align", "move", "goto", "navigate", "locate", 
                "coordinate", "stage", "motor", "axis", "movabs", "rotation",
                "translation", "calibration", "homing"
            ],
            Phase.SCAN: [
                "scan", "grab", "capture", "acquire", "image", "frame", 
                "exposure", "focus", "zoom", "magnify", "detect", "measure",
                "analyze", "inspect", "review", "electron", "beam"
            ],
            Phase.SAVE: [
                "save", "store", "write", "export", "backup", "archive", 
                "commit", "persist", "record", "register", "metadata",
                "database", "file", "disk"
            ],
            Phase.ERROR: [
                "error", "fail", "failed", "exception", "crash", "fault", 
                "invalid", "corrupt", "timeout", "denied", "rejected",
                "disconnected", "faulted", "critical", "emergency"
            ],
            Phase.RECOVERY: [
                "retry", "recover", "restore", "repair", "fix", "correct", 
                "resume", "continue", "restart", "reset", "debounce",
                "reconnect", "fallback", "backup"
            ],
            Phase.ABORT: [
                "abort", "cancel", "stop", "halt", "terminate", "kill", 
                "interrupt", "break", "exit", "quit", "shutdown",
                "disable", "disconnect"
            ]
        }
        
        # Enhanced risk keywords with weights
        self.risk_keywords = {
            RiskLevel.RED: {
                "discarded": 0.9, "failed": 0.8, "error": 0.7, "abort": 0.9,
                "crash": 0.95, "fault": 0.8, "corrupt": 0.85, "invalid": 0.7,
                "denied": 0.6, "timeout": 0.7, "exception": 0.8, "critical": 0.95,
                "fatal": 0.95, "emergency": 0.9, "disconnected": 0.8, "faulted": 0.8
            },
            RiskLevel.YELLOW: {
                "retry": 0.6, "stall": 0.7, "delay": 0.5, "slow": 0.4,
                "warning": 0.6, "caution": 0.5, "partial": 0.4,
                "incomplete": 0.5, "degraded": 0.6, "limited": 0.4,
                "debounce": 0.5, "reconnect": 0.3
            },
            RiskLevel.GREEN: {
                "complete": 0.9, "success": 0.9, "ok": 0.8, "ready": 0.7,
                "normal": 0.8, "stable": 0.8, "healthy": 0.9, "optimal": 0.9,
                "finished": 0.9, "done": 0.9, "saved": 0.8, "connected": 0.7
            }
        }
        
        # Multi-line patterns for complex log entries
        self.multiline_patterns = [
            # Stack traces
            r'at\s+[\w\.]+\([\w\.]+:\d+\)',
            # Exception details
            r'Exception\s+in\s+thread',
            # SQL queries
            r'SELECT\s+.*\s+FROM\s+',
            # File paths
            r'[A-Za-z]:\\[\w\\\.\-]+',
            # URLs
            r'https?://[\w\.\-/]+',
            # JSON/XML data
            r'[\{\[].*[\}\]]',
        ]
        
        # Performance tracking
        self.performance_metrics = None
        self.pattern_cache = {}
        self.anomaly_baseline = defaultdict(list)
        
    def categorize_log_sequence(self, entries: List[LogEntry]) -> Tuple[List[EnhancedCategorizedLogEntry], PerformanceMetrics]:
        """
        Enhanced categorization with performance metrics
        
        Args:
            entries: List of LogEntry objects
            
        Returns:
            Tuple of (categorized entries, performance metrics)
        """
        start_time = time.time()
        
        # Build anomaly baseline
        self._build_anomaly_baseline(entries)
        
        categorized_entries = []
        successful_categorizations = 0
        total_confidence = 0.0
        confidence_distribution = {"High": 0, "Medium": 0, "Low": 0}
        
        categorization_start = time.time()
        
        for i, entry in enumerate(entries):
            enhanced_entry = self._categorize_entry_enhanced(entry, entries, i)
            categorized_entries.append(enhanced_entry)
            
            if enhanced_entry.confidence > 0.5:
                successful_categorizations += 1
            
            total_confidence += enhanced_entry.confidence
            
            # Track confidence distribution
            if enhanced_entry.confidence >= 0.8:
                confidence_distribution["High"] += 1
            elif enhanced_entry.confidence >= 0.5:
                confidence_distribution["Medium"] += 1
            else:
                confidence_distribution["Low"] += 1
        
        categorization_time = time.time() - categorization_start
        total_time = time.time() - start_time
        
        # Apply context-aware improvements
        self._apply_enhanced_context_rules(categorized_entries)
        
        # Calculate metrics
        avg_confidence = total_confidence / len(entries) if entries else 0
        entries_per_second = len(entries) / total_time if total_time > 0 else 0
        
        metrics = PerformanceMetrics(
            parsing_time=categorization_start - start_time,
            categorization_time=categorization_time,
            total_entries=len(entries),
            successful_categorizations=successful_categorizations,
            average_confidence=avg_confidence,
            entries_per_second=entries_per_second,
            confidence_distribution=confidence_distribution
        )
        
        return categorized_entries, metrics
    
    def _categorize_entry_enhanced(self, entry: LogEntry, all_entries: List[LogEntry], index: int) -> EnhancedCategorizedLogEntry:
        """Enhanced categorization for a single entry"""
        # Get temporal context
        temporal_context = self._get_temporal_context(all_entries, index)
        
        # Multi-line analysis
        is_multiline = self._detect_multiline_pattern(entry)
        
        # Phase detection with confidence
        phase, phase_confidence, phase_patterns = self._detect_phase_enhanced(entry)
        
        # Risk detection with confidence
        risk_level, risk_confidence, risk_patterns = self._detect_risk_enhanced(entry)
        
        # Subsystem confidence
        subsystem_confidence = self._calculate_subsystem_confidence(entry)
        
        # Overall confidence calculation
        overall_confidence = (phase_confidence + risk_confidence + subsystem_confidence) / 3
        
        # Anomaly detection
        anomaly_score = self._calculate_anomaly_score(entry)
        
        # Pattern matches
        pattern_matches = phase_patterns + risk_patterns
        
        # Enhanced explanation
        explanation = self._generate_enhanced_explanation(
            entry, phase, risk_level, overall_confidence, temporal_context, anomaly_score
        )
        
        return EnhancedCategorizedLogEntry(
            log_entry=entry,
            phase=phase,
            risk_level=risk_level,
            confidence=overall_confidence,
            explanation=explanation,
            pattern_matches=pattern_matches,
            subsystem_confidence=subsystem_confidence,
            temporal_context=temporal_context,
            anomaly_score=anomaly_score
        )
    
    def _detect_phase_enhanced(self, entry: LogEntry) -> Tuple[Phase, float, List[str]]:
        """Enhanced phase detection with confidence scoring"""
        event_lower = entry.event.lower()
        subsystem_lower = entry.subsystem.lower()
        text_to_analyze = f"{event_lower} {subsystem_lower}"
        
        phase_scores = {}
        pattern_matches = []
        
        for phase, keywords in self.phase_keywords.items():
            score = 0
            matches = []
            
            for keyword in keywords:
                if keyword in text_to_analyze:
                    score += 1
                    matches.append(keyword)
            
            if score > 0:
                # Normalize score and add bonus for multiple matches
                normalized_score = min(score / len(keywords), 1.0)
                if score > 1:
                    normalized_score *= 1.2  # Bonus for multiple matches
                
                phase_scores[phase] = min(normalized_score, 1.0)
                pattern_matches.extend(matches)
        
        if not phase_scores:
            return Phase.UNKNOWN, 0.0, []
        
        best_phase = max(phase_scores, key=phase_scores.get)
        confidence = phase_scores[best_phase]
        
        return best_phase, confidence, pattern_matches
    
    def _detect_risk_enhanced(self, entry: LogEntry) -> Tuple[RiskLevel, float, List[str]]:
        """Enhanced risk detection with weighted scoring"""
        event_lower = entry.event.lower()
        subsystem_lower = entry.subsystem.lower()
        text_to_analyze = f"{event_lower} {subsystem_lower}"
        
        risk_scores = {}
        pattern_matches = []
        
        for risk_level, keywords in self.risk_keywords.items():
            score = 0
            matches = []
            
            for keyword, weight in keywords.items():
                if keyword in text_to_analyze:
                    score += weight
                    matches.append(keyword)
            
            if score > 0:
                risk_scores[risk_level] = min(score, 1.0)
                pattern_matches.extend(matches)
        
        if not risk_scores:
            return RiskLevel.GREEN, 0.3, []  # Default to low-confidence green
        
        best_risk = max(risk_scores, key=risk_scores.get)
        confidence = risk_scores[best_risk]
        
        return best_risk, confidence, pattern_matches
    
    def _calculate_subsystem_confidence(self, entry: LogEntry) -> float:
        """Calculate confidence in subsystem detection"""
        if entry.subsystem == "Unknown":
            return 0.1
        
        # Higher confidence for known subsystems
        known_subsystems = [
            "Save Engine", "AF System", "SEM", "ADR Logger", "Image Save",
            "Defect Detection", "Stage Control", "Vacuum Controller"
        ]
        
        if entry.subsystem in known_subsystems:
            return 0.9
        
        # Medium confidence for reasonable-looking subsystems
        if len(entry.subsystem) > 3 and entry.subsystem.replace(" ", "").isalnum():
            return 0.7
        
        return 0.4
    
    def _get_temporal_context(self, entries: List[LogEntry], index: int) -> Optional[str]:
        """Get context from surrounding entries"""
        if index == 0:
            return "Session start"
        
        prev_entry = entries[index - 1]
        current_entry = entries[index]
        
        # Check for rapid succession
        if hasattr(prev_entry, 'timestamp') and hasattr(current_entry, 'timestamp'):
            # Simple heuristic for timing
            if "retry" in current_entry.event.lower() and "retry" in prev_entry.event.lower():
                return "Repeated retry sequence"
            elif "error" in prev_entry.event.lower() and "retry" in current_entry.event.lower():
                return "Error recovery attempt"
        
        return None
    
    def _detect_multiline_pattern(self, entry: LogEntry) -> bool:
        """Detect if entry is part of a multi-line pattern"""
        for pattern in self.multiline_patterns:
            if re.search(pattern, entry.event, re.IGNORECASE):
                return True
        return False
    
    def _build_anomaly_baseline(self, entries: List[LogEntry]):
        """Build baseline for anomaly detection"""
        self.anomaly_baseline.clear()
        
        for entry in entries:
            self.anomaly_baseline[entry.subsystem].append(len(entry.event))
    
    def _calculate_anomaly_score(self, entry: LogEntry) -> float:
        """Calculate how anomalous this entry is"""
        if entry.subsystem not in self.anomaly_baseline:
            return 0.5  # Unknown subsystem
        
        baseline_lengths = self.anomaly_baseline[entry.subsystem]
        if not baseline_lengths:
            return 0.5
        
        avg_length = sum(baseline_lengths) / len(baseline_lengths)
        current_length = len(entry.event)
        
        # Simple anomaly score based on length deviation
        deviation = abs(current_length - avg_length) / avg_length if avg_length > 0 else 0
        return min(deviation, 1.0)
    
    def _generate_enhanced_explanation(self, entry: LogEntry, phase: Phase, risk_level: RiskLevel, 
                                     confidence: float, temporal_context: Optional[str], 
                                     anomaly_score: float) -> str:
        """Generate enhanced explanation with confidence and context"""
        base_explanation = self._generate_base_explanation(phase, risk_level)
        
        # Add confidence information
        confidence_text = ""
        if confidence >= 0.8:
            confidence_text = " (High confidence)"
        elif confidence >= 0.5:
            confidence_text = " (Medium confidence)"
        else:
            confidence_text = " (Low confidence)"
        
        # Add temporal context
        context_text = ""
        if temporal_context:
            context_text = f" - {temporal_context}"
        
        # Add anomaly information
        anomaly_text = ""
        if anomaly_score > 0.7:
            anomaly_text = " [Unusual pattern detected]"
        
        return f"{base_explanation}{confidence_text}{context_text}{anomaly_text}"
    
    def _generate_base_explanation(self, phase: Phase, risk_level: RiskLevel) -> str:
        """Generate base explanation"""
        explanations = {
            (Phase.INIT, RiskLevel.GREEN): "System initialization proceeding normally",
            (Phase.POSITION, RiskLevel.GREEN): "Positioning operation successful",
            (Phase.SCAN, RiskLevel.GREEN): "Scanning/imaging operation completed",
            (Phase.SAVE, RiskLevel.GREEN): "Data save operation successful",
            (Phase.ERROR, RiskLevel.RED): "Error detected - operation failed",
            (Phase.RECOVERY, RiskLevel.YELLOW): "System attempting recovery",
            (Phase.RECOVERY, RiskLevel.GREEN): "Recovery operation successful",
            (Phase.ABORT, RiskLevel.RED): "Operation aborted or cancelled",
        }
        
        key = (phase, risk_level)
        if key in explanations:
            return explanations[key]
        
        # Generic explanations
        if risk_level == RiskLevel.RED:
            return "Critical issue detected requiring attention"
        elif risk_level == RiskLevel.YELLOW:
            return "Warning condition - monitor closely"
        else:
            return "Normal operation"
    
    def _apply_enhanced_context_rules(self, categorized_entries: List[EnhancedCategorizedLogEntry]):
        """Apply enhanced context-aware rules"""
        for i in range(len(categorized_entries)):
            current = categorized_entries[i]
            
            # Enhanced retry detection
            if "retry" in current.log_entry.event.lower():
                retry_count = self._count_retries_in_sequence(categorized_entries, i)
                if retry_count > 2:
                    current.risk_level = RiskLevel.RED
                    current.confidence = max(current.confidence, 0.8)
                    current.explanation = f"Critical: {retry_count} consecutive retries detected"
            
            # Cascading failure detection
            if i > 0 and current.risk_level == RiskLevel.RED:
                prev_entry = categorized_entries[i-1]
                if prev_entry.risk_level == RiskLevel.RED:
                    current.explanation += " [Part of cascading failure]"
    
    def _count_retries_in_sequence(self, entries: List[EnhancedCategorizedLogEntry], start_index: int) -> int:
        """Count consecutive retries from a starting position"""
        count = 0
        for i in range(start_index, len(entries)):
            if "retry" in entries[i].log_entry.event.lower():
                count += 1
            else:
                break
        return count
    
    def get_detailed_statistics(self, categorized_entries: List[EnhancedCategorizedLogEntry]) -> Dict:
        """Get detailed statistics about the categorization results"""
        if not categorized_entries:
            return {}
        
        stats = {
            "total_entries": len(categorized_entries),
            "phase_distribution": defaultdict(int),
            "risk_distribution": defaultdict(int),
            "confidence_stats": {
                "min": min(e.confidence for e in categorized_entries),
                "max": max(e.confidence for e in categorized_entries),
                "avg": sum(e.confidence for e in categorized_entries) / len(categorized_entries),
            },
            "anomaly_stats": {
                "high_anomaly_count": sum(1 for e in categorized_entries if e.anomaly_score > 0.7),
                "avg_anomaly_score": sum(e.anomaly_score for e in categorized_entries) / len(categorized_entries),
            },
            "pattern_usage": defaultdict(int),
            "subsystem_analysis": defaultdict(lambda: {"count": 0, "avg_confidence": 0}),
        }
        
        for entry in categorized_entries:
            stats["phase_distribution"][entry.phase.value] += 1
            stats["risk_distribution"][entry.risk_level.value] += 1
            
            for pattern in entry.pattern_matches:
                stats["pattern_usage"][pattern] += 1
            
            subsystem = entry.log_entry.subsystem
            stats["subsystem_analysis"][subsystem]["count"] += 1
            stats["subsystem_analysis"][subsystem]["avg_confidence"] += entry.confidence
        
        # Calculate average confidence per subsystem
        for subsystem_data in stats["subsystem_analysis"].values():
            if subsystem_data["count"] > 0:
                subsystem_data["avg_confidence"] /= subsystem_data["count"]
        
        return dict(stats) 