"""
Categorizer for LogNarrator AI

This module handles categorizing log entries into phases and risk levels.
"""

from typing import List, Dict, Tuple
from enum import Enum
from dataclasses import dataclass
from parser import LogEntry


class Phase(Enum):
    """Operational phases for log categorization"""
    INIT = "Init"
    POSITION = "Position"
    SCAN = "Scan"
    SAVE = "Save"
    ERROR = "Error"
    RECOVERY = "Recovery"
    ABORT = "Abort"
    UNKNOWN = "Unknown"


class RiskLevel(Enum):
    """Risk levels for log entries"""
    GREEN = "🟢"
    YELLOW = "🟡"
    RED = "🔴"


@dataclass
class CategorizedLogEntry:
    """Log entry with categorization information"""
    log_entry: LogEntry
    phase: Phase
    risk_level: RiskLevel
    confidence: float
    explanation: str


class LogCategorizer:
    """Categorizes log entries into phases and risk levels"""
    
    def __init__(self):
        # Phase detection keywords
        self.phase_keywords = {
            Phase.INIT: [
                "triggered", "start", "begin", "initialize", "init", "startup", 
                "launch", "activate", "enable", "power", "boot"
            ],
            Phase.POSITION: [
                "position", "align", "move", "goto", "navigate", "locate", 
                "coordinate", "stage", "motor", "axis"
            ],
            Phase.SCAN: [
                "scan", "grab", "capture", "acquire", "image", "frame", 
                "exposure", "focus", "zoom", "magnify"
            ],
            Phase.SAVE: [
                "save", "store", "write", "export", "backup", "archive", 
                "commit", "persist", "record"
            ],
            Phase.ERROR: [
                "error", "fail", "failed", "exception", "crash", "fault", 
                "invalid", "corrupt", "timeout", "denied", "rejected"
            ],
            Phase.RECOVERY: [
                "retry", "recover", "restore", "repair", "fix", "correct", 
                "resume", "continue", "restart", "reset"
            ],
            Phase.ABORT: [
                "abort", "cancel", "stop", "halt", "terminate", "kill", 
                "interrupt", "break", "exit", "quit"
            ]
        }
        
        # Risk level keywords
        self.risk_keywords = {
            RiskLevel.RED: [
                "discarded", "failed", "error", "abort", "crash", "fault", 
                "corrupt", "invalid", "denied", "timeout", "exception", 
                "critical", "fatal", "emergency"
            ],
            RiskLevel.YELLOW: [
                "retry", "stall", "delay", "slow", "recovered", "warning", 
                "caution", "partial", "incomplete", "degraded", "limited"
            ],
            RiskLevel.GREEN: [
                "complete", "success", "ok", "ready", "normal", "stable", 
                "healthy", "optimal", "finished", "done", "saved"
            ]
        }
        
        # Subsystem-specific rules
        self.subsystem_rules = {
            "AF System": {
                "retry": (Phase.RECOVERY, RiskLevel.YELLOW),
                "recovery": (Phase.RECOVERY, RiskLevel.GREEN),
                "failed": (Phase.ERROR, RiskLevel.RED)
            },
            "Save Engine": {
                "triggered": (Phase.SAVE, RiskLevel.GREEN),
                "complete": (Phase.SAVE, RiskLevel.GREEN),
                "failed": (Phase.ERROR, RiskLevel.RED)
            },
            "SEM": {
                "discarded": (Phase.ERROR, RiskLevel.RED),
                "captured": (Phase.SCAN, RiskLevel.GREEN),
                "scan": (Phase.SCAN, RiskLevel.GREEN)
            },
            # SEM-specific subsystems
            "ADR Logger": {
                "handlesave": (Phase.SAVE, RiskLevel.GREEN),
                "async": (Phase.SAVE, RiskLevel.GREEN),
                "error": (Phase.ERROR, RiskLevel.RED),
                "failed": (Phase.ERROR, RiskLevel.RED)
            },
            "Image Save": {
                "save": (Phase.SAVE, RiskLevel.GREEN),
                "complete": (Phase.SAVE, RiskLevel.GREEN),
                "failed": (Phase.ERROR, RiskLevel.RED)
            },
            "Defect Detection": {
                "detect": (Phase.SCAN, RiskLevel.GREEN),
                "analysis": (Phase.SCAN, RiskLevel.GREEN),
                "failed": (Phase.ERROR, RiskLevel.RED)
            },
            "Stage Control": {
                "move": (Phase.POSITION, RiskLevel.GREEN),
                "position": (Phase.POSITION, RiskLevel.GREEN),
                "align": (Phase.POSITION, RiskLevel.GREEN),
                "failed": (Phase.ERROR, RiskLevel.RED)
            }
        }
    
    def categorize_entry(self, entry: LogEntry) -> CategorizedLogEntry:
        """
        Categorize a single log entry
        
        Args:
            entry: LogEntry to categorize
            
        Returns:
            CategorizedLogEntry with phase and risk information
        """
        # Check subsystem-specific rules first
        phase, risk_level, confidence = self._apply_subsystem_rules(entry)
        
        # If no specific rule matched, use keyword matching
        if phase == Phase.UNKNOWN:
            phase, confidence_phase = self._detect_phase(entry)
            risk_level, confidence_risk = self._detect_risk_level(entry)
            confidence = (confidence_phase + confidence_risk) / 2
        
        # Generate explanation
        explanation = self._generate_explanation(entry, phase, risk_level)
        
        return CategorizedLogEntry(
            log_entry=entry,
            phase=phase,
            risk_level=risk_level,
            confidence=confidence,
            explanation=explanation
        )
    
    def _apply_subsystem_rules(self, entry: LogEntry) -> Tuple[Phase, RiskLevel, float]:
        """Apply subsystem-specific categorization rules"""
        subsystem = entry.subsystem
        event_lower = entry.event.lower()
        
        if subsystem in self.subsystem_rules:
            rules = self.subsystem_rules[subsystem]
            for keyword, (phase, risk) in rules.items():
                if keyword in event_lower:
                    return phase, risk, 0.9  # High confidence for specific rules
        
        return Phase.UNKNOWN, RiskLevel.GREEN, 0.0
    
    def _detect_phase(self, entry: LogEntry) -> Tuple[Phase, float]:
        """Detect phase based on keyword matching"""
        event_lower = entry.event.lower()
        subsystem_lower = entry.subsystem.lower()
        text_to_analyze = f"{event_lower} {subsystem_lower}"
        
        phase_scores = {}
        
        for phase, keywords in self.phase_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_to_analyze:
                    score += 1
            
            if score > 0:
                phase_scores[phase] = score / len(keywords)
        
        if not phase_scores:
            return Phase.UNKNOWN, 0.0
        
        best_phase = max(phase_scores, key=phase_scores.get)
        confidence = phase_scores[best_phase]
        
        return best_phase, confidence
    
    def _detect_risk_level(self, entry: LogEntry) -> Tuple[RiskLevel, float]:
        """Detect risk level based on keyword matching"""
        event_lower = entry.event.lower()
        subsystem_lower = entry.subsystem.lower()
        text_to_analyze = f"{event_lower} {subsystem_lower}"
        
        risk_scores = {}
        
        for risk, keywords in self.risk_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_to_analyze:
                    score += 1
            
            if score > 0:
                risk_scores[risk] = score / len(keywords)
        
        if not risk_scores:
            return RiskLevel.GREEN, 0.5  # Default to green with medium confidence
        
        best_risk = max(risk_scores, key=risk_scores.get)
        confidence = risk_scores[best_risk]
        
        return best_risk, confidence
    
    def _generate_explanation(self, entry: LogEntry, phase: Phase, risk_level: RiskLevel) -> str:
        """Generate a simple explanation for the categorization"""
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
    
    def categorize_log_sequence(self, entries: List[LogEntry]) -> List[CategorizedLogEntry]:
        """
        Categorize a sequence of log entries
        
        Args:
            entries: List of LogEntry objects
            
        Returns:
            List of CategorizedLogEntry objects
        """
        categorized_entries = []
        
        for entry in entries:
            categorized = self.categorize_entry(entry)
            categorized_entries.append(categorized)
        
        # Post-process to improve categorization based on context
        self._apply_context_rules(categorized_entries)
        
        return categorized_entries
    
    def _apply_context_rules(self, categorized_entries: List[CategorizedLogEntry]):
        """Apply context-aware rules to improve categorization"""
        for i in range(len(categorized_entries)):
            current = categorized_entries[i]
            
            # Look for retry patterns
            if "retry" in current.log_entry.event.lower():
                # Check if this is part of a sequence
                retry_count = self._count_retries_in_sequence(categorized_entries, i)
                if retry_count > 2:
                    current.risk_level = RiskLevel.RED
                    current.explanation = f"Multiple retries detected ({retry_count}) - system instability"
            
            # Look for recovery after errors
            if current.phase == Phase.RECOVERY and i > 0:
                prev_entry = categorized_entries[i-1]
                if prev_entry.phase == Phase.ERROR:
                    current.explanation = "Recovery attempt after error"
    
    def _count_retries_in_sequence(self, entries: List[CategorizedLogEntry], start_index: int) -> int:
        """Count consecutive retries from a starting position"""
        count = 0
        for i in range(start_index, len(entries)):
            if "retry" in entries[i].log_entry.event.lower():
                count += 1
            else:
                break
        return count
    
    def get_categorization_summary(self, categorized_entries: List[CategorizedLogEntry]) -> Dict[str, any]:
        """
        Get summary statistics for categorized entries
        
        Args:
            categorized_entries: List of categorized log entries
            
        Returns:
            Dictionary with summary statistics
        """
        if not categorized_entries:
            return {}
        
        phase_counts = {}
        risk_counts = {}
        
        for entry in categorized_entries:
            phase = entry.phase.value
            risk = entry.risk_level.value
            
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        total_entries = len(categorized_entries)
        
        return {
            "total_entries": total_entries,
            "phase_distribution": phase_counts,
            "risk_distribution": risk_counts,
            "error_rate": risk_counts.get("🔴", 0) / total_entries * 100,
            "warning_rate": risk_counts.get("🟡", 0) / total_entries * 100,
            "success_rate": risk_counts.get("🟢", 0) / total_entries * 100
        }


def demo_categorizer():
    """Demo function to show categorizer in action"""
    from parser import LogParser
    
    sample_log = """
00:39:24.243 Save Engine Async Save Triggered  
00:39:24.267 AF System Retry #1 Triggered  
00:39:26.214 SEM Image discarded
00:39:27.001 AF System Retry #2 Triggered  
00:39:28.520 AF System Recovery Complete
00:39:30.100 Save Engine Save Operation Complete
"""
    
    parser = LogParser()
    entries = parser.parse_text(sample_log)
    
    categorizer = LogCategorizer()
    categorized = categorizer.categorize_log_sequence(entries)
    
    print("Categorized Log Entries:")
    print("-" * 80)
    for entry in categorized:
        print(f"Time: {entry.log_entry.timestamp:15} | "
              f"System: {entry.log_entry.subsystem:12} | "
              f"Phase: {entry.phase.value:8} | "
              f"Risk: {entry.risk_level.value} | "
              f"Event: {entry.log_entry.event}")
        print(f"    Explanation: {entry.explanation}")
        print()
    
    print("Summary Statistics:")
    summary = categorizer.get_categorization_summary(categorized)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    demo_categorizer() 