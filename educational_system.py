"""
Educational System for Log Analysis
Helps users understand the story that logs tell through narrative explanations,
contextual education, and interactive learning features.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

class StoryType(Enum):
    """Types of stories that can be told from logs."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_CASCADE = "error_cascade"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SECURITY_INCIDENT = "security_incident"
    CONFIGURATION_CHANGE = "configuration_change"
    NORMAL_OPERATION = "normal_operation"
    RECOVERY_PROCESS = "recovery_process"
    MAINTENANCE_ACTIVITY = "maintenance_activity"

class LearningLevel(Enum):
    """Learning levels for adaptive explanations."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class LogStorySegment:
    """A segment of the log story with educational context."""
    timestamp: datetime
    technical_description: str
    business_impact: str
    story_narrative: str
    learning_points: List[str]
    severity_explanation: str
    what_to_look_for: str
    red_flags: List[str] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)

@dataclass
class LogStory:
    """Complete story derived from log analysis."""
    story_type: StoryType
    title: str
    executive_summary: str
    detailed_narrative: str
    segments: List[LogStorySegment]
    lessons_learned: List[str]
    investigation_methodology: str
    prevention_tips: List[str]
    related_case_studies: List[str] = field(default_factory=list)

@dataclass
class EducationalContext:
    """Educational context for a specific log pattern or event."""
    pattern_name: str
    what_it_means: str
    why_it_happens: str
    business_impact: str
    investigation_steps: List[str]
    red_flags: List[str]
    common_causes: List[str]
    remediation_steps: List[str]
    learning_level: LearningLevel = LearningLevel.BEGINNER

class StoryNarrativeEngine:
    """Converts technical log data into educational narratives."""
    
    def __init__(self):
        self.story_templates = self._load_story_templates()
        self.technical_glossary = self._load_technical_glossary()
        self.business_impact_mapping = self._load_business_impact_mapping()
        self.pattern_library = self._load_pattern_library()
    
    def generate_story(self, log_entries: List[Any], learning_level: LearningLevel = LearningLevel.BEGINNER) -> LogStory:
        """Generate a complete story from log entries."""
        story_type = self._identify_story_type(log_entries)
        segments = self._create_story_segments(log_entries, learning_level)
        
        return LogStory(
            story_type=story_type,
            title=self._generate_title(story_type, log_entries),
            executive_summary=self._generate_executive_summary(segments, story_type),
            detailed_narrative=self._generate_detailed_narrative(segments, story_type),
            segments=segments,
            lessons_learned=self._extract_lessons_learned(segments, story_type),
            investigation_methodology=self._generate_investigation_methodology(story_type),
            prevention_tips=self._generate_prevention_tips(story_type),
            related_case_studies=self._find_related_case_studies(story_type)
        )
    
    def _identify_story_type(self, log_entries: List[Any]) -> StoryType:
        """Identify the type of story based on log patterns."""
        error_count = sum(1 for entry in log_entries if hasattr(entry, 'level') and entry.level == 'ERROR')
        warning_count = sum(1 for entry in log_entries if hasattr(entry, 'level') and entry.level == 'WARN')
        
        # Look for specific patterns
        patterns = []
        for entry in log_entries:
            message = getattr(entry, 'message', '') or getattr(entry, 'content', '')
            if any(keyword in message.lower() for keyword in ['memory', 'cpu', 'disk', 'resource']):
                patterns.append('resource')
            if any(keyword in message.lower() for keyword in ['timeout', 'slow', 'performance']):
                patterns.append('performance')
            if any(keyword in message.lower() for keyword in ['startup', 'starting', 'initializing']):
                patterns.append('startup')
            if any(keyword in message.lower() for keyword in ['shutdown', 'stopping', 'terminating']):
                patterns.append('shutdown')
            if any(keyword in message.lower() for keyword in ['security', 'authentication', 'unauthorized']):
                patterns.append('security')
            if any(keyword in message.lower() for keyword in ['config', 'configuration', 'setting']):
                patterns.append('config')
        
        # Determine story type based on patterns and error density
        if error_count > len(log_entries) * 0.3:
            return StoryType.ERROR_CASCADE
        elif 'resource' in patterns and error_count > 0:
            return StoryType.RESOURCE_EXHAUSTION
        elif 'performance' in patterns:
            return StoryType.PERFORMANCE_DEGRADATION
        elif 'startup' in patterns:
            return StoryType.SYSTEM_STARTUP
        elif 'shutdown' in patterns:
            return StoryType.SYSTEM_SHUTDOWN
        elif 'security' in patterns:
            return StoryType.SECURITY_INCIDENT
        elif 'config' in patterns:
            return StoryType.CONFIGURATION_CHANGE
        elif error_count == 0 and warning_count == 0:
            return StoryType.NORMAL_OPERATION
        else:
            return StoryType.NORMAL_OPERATION
    
    def _create_story_segments(self, log_entries: List[Any], learning_level: LearningLevel) -> List[LogStorySegment]:
        """Create educational story segments from log entries."""
        segments = []
        
        for entry in log_entries:
            segment = self._create_segment_from_entry(entry, learning_level)
            if segment:
                segments.append(segment)
        
        return segments
    
    def _create_segment_from_entry(self, entry: Any, learning_level: LearningLevel) -> Optional[LogStorySegment]:
        """Create a story segment from a single log entry."""
        try:
            timestamp = getattr(entry, 'timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            message = getattr(entry, 'message', '') or getattr(entry, 'content', '')
            level = getattr(entry, 'level', 'INFO')
            
            # Generate educational context
            technical_desc = self._generate_technical_description(message, level)
            business_impact = self._generate_business_impact(message, level)
            story_narrative = self._generate_story_narrative(message, level, learning_level)
            learning_points = self._generate_learning_points(message, level, learning_level)
            severity_explanation = self._generate_severity_explanation(level, learning_level)
            what_to_look_for = self._generate_what_to_look_for(message, level)
            red_flags = self._identify_red_flags(message, level)
            related_patterns = self._find_related_patterns(message, level)
            
            return LogStorySegment(
                timestamp=timestamp,
                technical_description=technical_desc,
                business_impact=business_impact,
                story_narrative=story_narrative,
                learning_points=learning_points,
                severity_explanation=severity_explanation,
                what_to_look_for=what_to_look_for,
                red_flags=red_flags,
                related_patterns=related_patterns
            )
        except Exception as e:
            # Handle gracefully for malformed entries
            return None
    
    def _generate_technical_description(self, message: str, level: str) -> str:
        """Generate technical description with educational context."""
        base_desc = f"Log Level: {level} - {message}"
        
        # Add technical context based on content
        if 'connection' in message.lower():
            base_desc += "\n\nTechnical Context: This relates to network connectivity between system components."
        elif 'memory' in message.lower():
            base_desc += "\n\nTechnical Context: This indicates memory (RAM) usage patterns in the system."
        elif 'timeout' in message.lower():
            base_desc += "\n\nTechnical Context: This shows a process or request took longer than expected."
        elif 'error' in message.lower():
            base_desc += "\n\nTechnical Context: This indicates an exceptional condition that needs attention."
        
        return base_desc
    
    def _generate_business_impact(self, message: str, level: str) -> str:
        """Translate technical events to business impact."""
        impact_mapping = {
            'ERROR': 'High Impact',
            'WARN': 'Medium Impact',
            'INFO': 'Low Impact',
            'DEBUG': 'No Impact'
        }
        
        base_impact = impact_mapping.get(level, 'Unknown Impact')
        
        # Add specific business context
        if 'connection' in message.lower() and level == 'ERROR':
            return f"{base_impact}: Users may experience service unavailability or slow response times."
        elif 'memory' in message.lower() and level in ['ERROR', 'WARN']:
            return f"{base_impact}: System performance may degrade, affecting user experience."
        elif 'timeout' in message.lower():
            return f"{base_impact}: Operations may take longer than expected, affecting user satisfaction."
        elif 'authentication' in message.lower():
            return f"{base_impact}: User login processes may be affected."
        elif level == 'INFO':
            return f"{base_impact}: Normal system operation, no immediate business concern."
        else:
            return f"{base_impact}: Potential impact on system reliability and user experience."
    
    def _generate_story_narrative(self, message: str, level: str, learning_level: LearningLevel) -> str:
        """Generate narrative explanation of what's happening."""
        if learning_level == LearningLevel.BEGINNER:
            return self._generate_beginner_narrative(message, level)
        elif learning_level == LearningLevel.INTERMEDIATE:
            return self._generate_intermediate_narrative(message, level)
        else:
            return self._generate_advanced_narrative(message, level)
    
    def _generate_beginner_narrative(self, message: str, level: str) -> str:
        """Generate beginner-friendly narrative."""
        if 'connection' in message.lower():
            return "Think of this like a phone call between two system components. Something went wrong with the connection, like a busy signal or dropped call."
        elif 'memory' in message.lower():
            return "This is like your computer's workspace. When it gets too full, things slow down or stop working properly."
        elif 'timeout' in message.lower():
            return "This is like waiting too long for someone to answer the phone. The system gave up waiting and moved on."
        elif level == 'ERROR':
            return "This is like a red warning light on your car dashboard - something important needs immediate attention."
        elif level == 'WARN':
            return "This is like a yellow warning light - not critical right now, but worth monitoring."
        else:
            return "This is normal system activity, like your heart beating - it's supposed to happen."
    
    def _generate_intermediate_narrative(self, message: str, level: str) -> str:
        """Generate intermediate-level narrative."""
        if 'connection' in message.lower():
            return "Network connectivity issue detected. This could indicate network congestion, service unavailability, or configuration problems."
        elif 'memory' in message.lower():
            return "Memory utilization event. This suggests either normal allocation patterns or potential memory leaks requiring investigation."
        elif 'timeout' in message.lower():
            return "Request timeout occurred. This indicates performance bottlenecks or resource contention in the system."
        elif level == 'ERROR':
            return "Error condition detected. This requires immediate investigation to prevent service degradation."
        elif level == 'WARN':
            return "Warning condition identified. This should be monitored for patterns that might indicate developing issues."
        else:
            return "Normal operational event. Part of expected system behavior patterns."
    
    def _generate_advanced_narrative(self, message: str, level: str) -> str:
        """Generate advanced technical narrative."""
        if 'connection' in message.lower():
            return "TCP/IP layer or application protocol communication failure. Investigate network topology, firewall rules, and service availability."
        elif 'memory' in message.lower():
            return "Heap allocation or garbage collection event. Analyze memory usage patterns, object lifecycle, and potential memory leaks."
        elif 'timeout' in message.lower():
            return "Request exceeded configured timeout threshold. Examine I/O operations, thread pool utilization, and resource lock contention."
        elif level == 'ERROR':
            return "Exception condition requiring immediate remediation. Correlate with system metrics and downstream service dependencies."
        elif level == 'WARN':
            return "Threshold breach or anomalous condition. Monitor for escalation patterns and correlation with performance metrics."
        else:
            return "Standard operational telemetry. Part of normal system instrumentation and monitoring."
    
    def _generate_learning_points(self, message: str, level: str, learning_level: LearningLevel) -> List[str]:
        """Generate educational learning points."""
        learning_points = []
        
        if level == 'ERROR':
            learning_points.append("Errors indicate problems that need immediate attention")
            learning_points.append("Look for patterns in error messages to identify root causes")
        elif level == 'WARN':
            learning_points.append("Warnings are early indicators of potential problems")
            learning_points.append("Monitor warning trends to prevent escalation to errors")
        
        if 'connection' in message.lower():
            learning_points.append("Connection issues often indicate network or service problems")
            learning_points.append("Check both client and server sides when investigating connection failures")
        elif 'memory' in message.lower():
            learning_points.append("Memory issues can cascade to other system problems")
            learning_points.append("Track memory usage trends over time to identify leaks")
        elif 'timeout' in message.lower():
            learning_points.append("Timeouts often indicate performance bottlenecks")
            learning_points.append("Consider both response time and resource availability")
        
        return learning_points
    
    def _generate_severity_explanation(self, level: str, learning_level: LearningLevel) -> str:
        """Explain the severity level's meaning."""
        explanations = {
            'ERROR': {
                LearningLevel.BEGINNER: "ERROR means something is broken and needs immediate attention, like a flat tire on your car.",
                LearningLevel.INTERMEDIATE: "ERROR indicates a failure condition that impacts system functionality and requires remediation.",
                LearningLevel.ADVANCED: "ERROR represents an exception condition that violates expected system behavior and requires immediate investigation."
            },
            'WARN': {
                LearningLevel.BEGINNER: "WARN means there's a potential problem, like a warning light that something might go wrong soon.",
                LearningLevel.INTERMEDIATE: "WARN indicates a condition that should be monitored but doesn't immediately break functionality.",
                LearningLevel.ADVANCED: "WARN represents a threshold breach or anomalous condition that requires monitoring for escalation."
            },
            'INFO': {
                LearningLevel.BEGINNER: "INFO is just normal system activity, like your computer showing it's working properly.",
                LearningLevel.INTERMEDIATE: "INFO provides operational visibility into normal system behavior and state changes.",
                LearningLevel.ADVANCED: "INFO offers instrumentation data for operational visibility and audit trails."
            },
            'DEBUG': {
                LearningLevel.BEGINNER: "DEBUG is very detailed information, usually only needed when investigating specific problems.",
                LearningLevel.INTERMEDIATE: "DEBUG provides detailed diagnostic information for troubleshooting and development.",
                LearningLevel.ADVANCED: "DEBUG offers granular execution flow information for deep system analysis."
            }
        }
        
        return explanations.get(level, {}).get(learning_level, "Unknown severity level")
    
    def _generate_what_to_look_for(self, message: str, level: str) -> str:
        """Generate guidance on what to look for when investigating."""
        guidance = []
        
        if level == 'ERROR':
            guidance.append("Look for error patterns and frequency")
            guidance.append("Check for related errors in nearby timeframes")
            guidance.append("Examine system resources at the time of error")
        
        if 'connection' in message.lower():
            guidance.append("Check network connectivity and service availability")
            guidance.append("Verify firewall and security configurations")
        elif 'memory' in message.lower():
            guidance.append("Monitor memory usage trends and allocation patterns")
            guidance.append("Look for memory leaks or excessive garbage collection")
        elif 'timeout' in message.lower():
            guidance.append("Examine response times and performance metrics")
            guidance.append("Check for resource contention or blocking operations")
        
        return " • ".join(guidance) if guidance else "Monitor for patterns and correlations with other system events"
    
    def _identify_red_flags(self, message: str, level: str) -> List[str]:
        """Identify red flags that indicate serious problems."""
        red_flags = []
        
        if level == 'ERROR':
            red_flags.append("High error frequency")
            red_flags.append("Cascading failures")
        
        if any(keyword in message.lower() for keyword in ['out of memory', 'memory leak', 'heap space']):
            red_flags.append("Memory exhaustion")
        
        if any(keyword in message.lower() for keyword in ['connection refused', 'network unreachable']):
            red_flags.append("Service unavailability")
        
        if any(keyword in message.lower() for keyword in ['timeout', 'slow', 'performance']):
            red_flags.append("Performance degradation")
        
        if any(keyword in message.lower() for keyword in ['security', 'unauthorized', 'breach']):
            red_flags.append("Security incident")
        
        return red_flags
    
    def _find_related_patterns(self, message: str, level: str) -> List[str]:
        """Find related patterns to look for."""
        patterns = []
        
        if 'connection' in message.lower():
            patterns.extend(["Network timeouts", "Service unavailability", "Load balancer issues"])
        elif 'memory' in message.lower():
            patterns.extend(["Garbage collection events", "Performance degradation", "OutOfMemory errors"])
        elif 'timeout' in message.lower():
            patterns.extend(["Slow database queries", "Resource contention", "Thread pool exhaustion"])
        
        return patterns
    
    def _generate_title(self, story_type: StoryType, log_entries: List[Any]) -> str:
        """Generate a descriptive title for the story."""
        titles = {
            StoryType.PERFORMANCE_DEGRADATION: "System Performance Degradation Event",
            StoryType.ERROR_CASCADE: "Error Cascade Analysis",
            StoryType.SYSTEM_STARTUP: "System Startup Sequence",
            StoryType.SYSTEM_SHUTDOWN: "System Shutdown Process",
            StoryType.RESOURCE_EXHAUSTION: "Resource Exhaustion Incident",
            StoryType.SECURITY_INCIDENT: "Security Event Analysis",
            StoryType.CONFIGURATION_CHANGE: "Configuration Change Impact",
            StoryType.NORMAL_OPERATION: "Normal System Operation",
            StoryType.RECOVERY_PROCESS: "System Recovery Process",
            StoryType.MAINTENANCE_ACTIVITY: "Maintenance Activity Log"
        }
        
        base_title = titles.get(story_type, "System Event Analysis")
        
        # Add time context
        if log_entries:
            start_time = getattr(log_entries[0], 'timestamp', None)
            if start_time:
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                base_title += f" - {start_time.strftime('%Y-%m-%d %H:%M')}"
        
        return base_title
    
    def _generate_executive_summary(self, segments: List[LogStorySegment], story_type: StoryType) -> str:
        """Generate executive summary of the story."""
        if not segments:
            return "No significant events detected in the log analysis."
        
        error_count = sum(1 for segment in segments if 'error' in segment.technical_description.lower())
        warning_count = sum(1 for segment in segments if 'warn' in segment.technical_description.lower())
        
        timespan = segments[-1].timestamp - segments[0].timestamp if len(segments) > 1 else timedelta(0)
        
        summary = f"Analysis of {len(segments)} log events over {timespan}. "
        
        if error_count > 0:
            summary += f"Detected {error_count} error conditions requiring attention. "
        if warning_count > 0:
            summary += f"Identified {warning_count} warning conditions for monitoring. "
        
        # Add story-specific context
        if story_type == StoryType.ERROR_CASCADE:
            summary += "System experienced cascading failures that may impact service availability."
        elif story_type == StoryType.PERFORMANCE_DEGRADATION:
            summary += "System performance degradation detected, affecting user experience."
        elif story_type == StoryType.NORMAL_OPERATION:
            summary += "System operating within normal parameters."
        
        return summary
    
    def _generate_detailed_narrative(self, segments: List[LogStorySegment], story_type: StoryType) -> str:
        """Generate detailed narrative of the story."""
        if not segments:
            return "No events to analyze."
        
        narrative = f"This log analysis tells the story of {story_type.value.replace('_', ' ')} events.\n\n"
        
        # Timeline narrative
        narrative += "Timeline of Events:\n"
        for i, segment in enumerate(segments):
            narrative += f"{i+1}. {segment.timestamp.strftime('%H:%M:%S')} - {segment.story_narrative}\n"
        
        narrative += "\nKey Insights:\n"
        all_learning_points = []
        for segment in segments:
            all_learning_points.extend(segment.learning_points)
        
        # Deduplicate learning points
        unique_points = list(set(all_learning_points))
        for point in unique_points[:5]:  # Top 5 insights
            narrative += f"• {point}\n"
        
        return narrative
    
    def _extract_lessons_learned(self, segments: List[LogStorySegment], story_type: StoryType) -> List[str]:
        """Extract key lessons learned from the story."""
        lessons = []
        
        # Extract common lessons based on story type
        if story_type == StoryType.ERROR_CASCADE:
            lessons.append("Error cascades can be prevented with proper circuit breakers and fallback mechanisms")
            lessons.append("Monitor error rates and implement alerting to catch issues early")
        elif story_type == StoryType.PERFORMANCE_DEGRADATION:
            lessons.append("Performance issues often start small and escalate if not addressed")
            lessons.append("Implement performance monitoring and automated scaling")
        elif story_type == StoryType.RESOURCE_EXHAUSTION:
            lessons.append("Resource monitoring and limits can prevent exhaustion scenarios")
            lessons.append("Implement graceful degradation when resources are constrained")
        
        # Extract lessons from red flags
        all_red_flags = []
        for segment in segments:
            all_red_flags.extend(segment.red_flags)
        
        if 'Memory exhaustion' in all_red_flags:
            lessons.append("Memory leaks can cause cascading system failures")
        if 'Service unavailability' in all_red_flags:
            lessons.append("Service dependencies require health checks and fallback strategies")
        if 'Performance degradation' in all_red_flags:
            lessons.append("Performance monitoring should include both response times and resource utilization")
        
        return lessons
    
    def _generate_investigation_methodology(self, story_type: StoryType) -> str:
        """Generate investigation methodology for this type of story."""
        methodologies = {
            StoryType.ERROR_CASCADE: """
1. Identify the initial failure point
2. Trace the cascade path through system components
3. Analyze timing and correlation of failures
4. Examine error recovery mechanisms
5. Review circuit breaker and fallback configurations
""",
            StoryType.PERFORMANCE_DEGRADATION: """
1. Establish baseline performance metrics
2. Identify when degradation began
3. Correlate with system resource usage
4. Examine concurrent user load
5. Analyze database and network performance
""",
            StoryType.RESOURCE_EXHAUSTION: """
1. Identify which resource was exhausted
2. Analyze resource usage trends over time
3. Examine allocation and deallocation patterns
4. Review resource limits and configurations
5. Investigate potential resource leaks
""",
            StoryType.SECURITY_INCIDENT: """
1. Identify the security event type
2. Analyze attack vectors and entry points
3. Examine authentication and authorization logs
4. Review access patterns and anomalies
5. Assess potential data exposure
"""
        }
        
        return methodologies.get(story_type, """
1. Review log chronology and timing
2. Identify patterns and correlations
3. Analyze error frequencies and types
4. Examine system resource usage
5. Review configuration and environmental factors
""")
    
    def _generate_prevention_tips(self, story_type: StoryType) -> List[str]:
        """Generate prevention tips based on story type."""
        tips = {
            StoryType.ERROR_CASCADE: [
                "Implement circuit breakers to prevent cascade failures",
                "Use bulkhead patterns to isolate system components",
                "Set up proper retry mechanisms with exponential backoff",
                "Monitor error rates and implement automated alerting"
            ],
            StoryType.PERFORMANCE_DEGRADATION: [
                "Implement performance monitoring and alerting",
                "Use load testing to identify performance bottlenecks",
                "Set up auto-scaling for variable workloads",
                "Optimize database queries and caching strategies"
            ],
            StoryType.RESOURCE_EXHAUSTION: [
                "Set appropriate resource limits and quotas",
                "Implement resource monitoring and alerting",
                "Use connection pooling and resource recycling",
                "Implement graceful degradation under resource pressure"
            ],
            StoryType.SECURITY_INCIDENT: [
                "Implement comprehensive authentication and authorization",
                "Use security monitoring and anomaly detection",
                "Regular security audits and penetration testing",
                "Implement proper access logging and monitoring"
            ]
        }
        
        return tips.get(story_type, [
            "Implement comprehensive monitoring and alerting",
            "Use proper error handling and recovery mechanisms",
            "Regular system health checks and maintenance",
            "Implement proper logging and observability"
        ])
    
    def _find_related_case_studies(self, story_type: StoryType) -> List[str]:
        """Find related case studies for learning."""
        case_studies = {
            StoryType.ERROR_CASCADE: [
                "AWS S3 outage 2017: Cascading failures in dependent services",
                "Google Cloud 2019: Error cascade in load balancing systems",
                "Netflix 2008: Circuit breaker pattern implementation"
            ],
            StoryType.PERFORMANCE_DEGRADATION: [
                "Stack Overflow 2016: Database performance degradation",
                "GitHub 2018: Performance impact of database migrations",
                "Twitter 2008: Scaling challenges and performance optimization"
            ],
            StoryType.RESOURCE_EXHAUSTION: [
                "Slack 2017: Memory leak causing service degradation",
                "Reddit 2018: Database connection pool exhaustion",
                "Discord 2017: Memory management and garbage collection"
            ]
        }
        
        return case_studies.get(story_type, [
            "Generic system reliability case studies",
            "Performance optimization examples",
            "Error handling best practices"
        ])
    
    def _load_story_templates(self) -> Dict[str, str]:
        """Load story templates for different scenarios."""
        return {
            "error_cascade": "The system experienced a cascade of failures starting with {initial_error}",
            "performance_degradation": "System performance began degrading at {start_time} due to {cause}",
            "resource_exhaustion": "The system ran out of {resource} causing {impact}",
            "normal_operation": "The system operated normally with {event_count} routine events"
        }
    
    def _load_technical_glossary(self) -> Dict[str, str]:
        """Load technical terms and their explanations."""
        return {
            "timeout": "A situation where a process waits too long for a response",
            "connection": "A communication link between two system components",
            "memory": "The computer's working space for running programs",
            "error": "An unexpected condition that prevents normal operation",
            "warning": "A condition that should be monitored but doesn't stop operation"
        }
    
    def _load_business_impact_mapping(self) -> Dict[str, str]:
        """Load mapping of technical events to business impact."""
        return {
            "error": "May cause service disruption affecting customers",
            "timeout": "May cause slow response times affecting user experience",
            "memory": "May cause performance issues affecting system reliability",
            "connection": "May cause service unavailability affecting business operations"
        }
    
    def _load_pattern_library(self) -> Dict[str, EducationalContext]:
        """Load library of common patterns and their educational context."""
        return {
            "connection_timeout": EducationalContext(
                pattern_name="Connection Timeout",
                what_it_means="The system couldn't establish or maintain a connection within the expected time",
                why_it_happens="Network issues, service overload, or configuration problems",
                business_impact="Users may experience service unavailability or slow response times",
                investigation_steps=[
                    "Check network connectivity",
                    "Verify service availability",
                    "Examine timeout configurations",
                    "Review system resource usage"
                ],
                red_flags=["High frequency timeouts", "Cascading timeout failures"],
                common_causes=["Network congestion", "Service overload", "Misconfigured timeouts"],
                remediation_steps=[
                    "Adjust timeout values",
                    "Implement retry mechanisms",
                    "Scale affected services",
                    "Investigate network issues"
                ]
            ),
            "memory_leak": EducationalContext(
                pattern_name="Memory Leak",
                what_it_means="The system is using more memory over time without releasing it",
                why_it_happens="Code not properly cleaning up memory allocations",
                business_impact="System performance degrades and may eventually crash",
                investigation_steps=[
                    "Monitor memory usage trends",
                    "Analyze memory allocation patterns",
                    "Review code for memory leaks",
                    "Examine garbage collection logs"
                ],
                red_flags=["Continuously increasing memory usage", "OutOfMemory errors"],
                common_causes=["Unclosed resources", "Circular references", "Large object retention"],
                remediation_steps=[
                    "Fix memory leaks in code",
                    "Implement proper resource cleanup",
                    "Adjust garbage collection settings",
                    "Restart affected services"
                ]
            )
        }

class ContextualEducationSystem:
    """Provides contextual education and explanations for log analysis."""
    
    def __init__(self):
        self.narrative_engine = StoryNarrativeEngine()
        self.user_progress = {}
        self.learning_objectives = self._load_learning_objectives()
    
    def get_contextual_explanation(self, log_entry: Any, user_level: LearningLevel = LearningLevel.BEGINNER) -> EducationalContext:
        """Get contextual explanation for a specific log entry."""
        message = getattr(log_entry, 'message', '') or getattr(log_entry, 'content', '')
        level = getattr(log_entry, 'level', 'INFO')
        
        # Find matching pattern
        pattern_key = self._identify_pattern(message, level)
        base_context = self.narrative_engine.pattern_library.get(pattern_key)
        
        if base_context:
            return base_context
        
        # Generate context on the fly
        return EducationalContext(
            pattern_name=f"{level} Event",
            what_it_means=self._explain_what_it_means(message, level, user_level),
            why_it_happens=self._explain_why_it_happens(message, level),
            business_impact=self._explain_business_impact(message, level),
            investigation_steps=self._generate_investigation_steps(message, level),
            red_flags=self._identify_red_flags(message, level),
            common_causes=self._identify_common_causes(message, level),
            remediation_steps=self._generate_remediation_steps(message, level),
            learning_level=user_level
        )
    
    def _identify_pattern(self, message: str, level: str) -> str:
        """Identify the pattern type from message content."""
        message_lower = message.lower()
        
        if 'timeout' in message_lower or 'connection' in message_lower:
            return "connection_timeout"
        elif 'memory' in message_lower and ('leak' in message_lower or 'out of' in message_lower):
            return "memory_leak"
        elif 'error' in message_lower:
            return "generic_error"
        elif 'warning' in message_lower:
            return "generic_warning"
        else:
            return "generic_info"
    
    def _explain_what_it_means(self, message: str, level: str, user_level: LearningLevel) -> str:
        """Explain what the log entry means."""
        if user_level == LearningLevel.BEGINNER:
            return self._beginner_explanation(message, level)
        elif user_level == LearningLevel.INTERMEDIATE:
            return self._intermediate_explanation(message, level)
        else:
            return self._advanced_explanation(message, level)
    
    def _beginner_explanation(self, message: str, level: str) -> str:
        """Provide beginner-friendly explanation."""
        if level == 'ERROR':
            return "Something went wrong and the system couldn't complete what it was trying to do."
        elif level == 'WARN':
            return "The system noticed something unusual but can still continue working."
        elif level == 'INFO':
            return "The system is reporting normal activity or status updates."
        else:
            return "The system is providing detailed information about what it's doing."
    
    def _intermediate_explanation(self, message: str, level: str) -> str:
        """Provide intermediate-level explanation."""
        if level == 'ERROR':
            return "An exception or failure condition occurred that prevented successful operation completion."
        elif level == 'WARN':
            return "A potentially problematic condition was detected that warrants monitoring."
        elif level == 'INFO':
            return "Operational information about system state changes or routine activities."
        else:
            return "Detailed diagnostic information for development and troubleshooting purposes."
    
    def _advanced_explanation(self, message: str, level: str) -> str:
        """Provide advanced technical explanation."""
        if level == 'ERROR':
            return "Exception condition indicating a violation of expected system behavior requiring immediate remediation."
        elif level == 'WARN':
            return "Threshold breach or anomalous condition that may escalate without intervention."
        elif level == 'INFO':
            return "Instrumentation data providing operational visibility into system state and behavior."
        else:
            return "Granular execution flow information for detailed system analysis and debugging."
    
    def _explain_why_it_happens(self, message: str, level: str) -> str:
        """Explain why this type of event occurs."""
        if 'connection' in message.lower():
            return "Connection issues occur due to network problems, service unavailability, or configuration issues."
        elif 'memory' in message.lower():
            return "Memory issues happen when applications use more memory than available or don't clean up properly."
        elif 'timeout' in message.lower():
            return "Timeouts occur when operations take longer than expected, often due to resource constraints."
        elif level == 'ERROR':
            return "Errors happen when the system encounters unexpected conditions or failures."
        else:
            return "This is part of normal system operation and monitoring."
    
    def _explain_business_impact(self, message: str, level: str) -> str:
        """Explain the business impact of this event."""
        impact_levels = {
            'ERROR': 'High',
            'WARN': 'Medium',
            'INFO': 'Low',
            'DEBUG': 'None'
        }
        
        base_impact = impact_levels.get(level, 'Unknown')
        
        if 'connection' in message.lower() and level == 'ERROR':
            return f"{base_impact} Impact: Users cannot access services, affecting revenue and customer satisfaction."
        elif 'memory' in message.lower() and level in ['ERROR', 'WARN']:
            return f"{base_impact} Impact: System performance degrades, affecting user experience and productivity."
        elif 'timeout' in message.lower():
            return f"{base_impact} Impact: Operations take longer than expected, reducing user satisfaction."
        else:
            return f"{base_impact} Impact: Minimal immediate business impact."
    
    def _generate_investigation_steps(self, message: str, level: str) -> List[str]:
        """Generate investigation steps for this type of event."""
        steps = ["Review the log entry context and timing"]
        
        if level == 'ERROR':
            steps.extend([
                "Check for related errors in the same timeframe",
                "Examine system resource usage at the time of error",
                "Review configuration and recent changes"
            ])
        
        if 'connection' in message.lower():
            steps.extend([
                "Verify network connectivity",
                "Check service availability and health",
                "Review firewall and security configurations"
            ])
        elif 'memory' in message.lower():
            steps.extend([
                "Monitor memory usage patterns",
                "Check for memory leaks",
                "Review garbage collection logs"
            ])
        elif 'timeout' in message.lower():
            steps.extend([
                "Examine response times and performance metrics",
                "Check for resource contention",
                "Review timeout configurations"
            ])
        
        return steps
    
    def _identify_red_flags(self, message: str, level: str) -> List[str]:
        """Identify red flags associated with this type of event."""
        red_flags = []
        
        if level == 'ERROR':
            red_flags.extend(["High error frequency", "Cascading failures"])
        
        if 'connection' in message.lower():
            red_flags.extend(["Service unavailability", "Network connectivity issues"])
        elif 'memory' in message.lower():
            red_flags.extend(["Memory exhaustion", "Performance degradation"])
        elif 'timeout' in message.lower():
            red_flags.extend(["Performance bottlenecks", "Resource contention"])
        
        return red_flags
    
    def _identify_common_causes(self, message: str, level: str) -> List[str]:
        """Identify common causes for this type of event."""
        causes = []
        
        if 'connection' in message.lower():
            causes.extend(["Network congestion", "Service overload", "Configuration issues"])
        elif 'memory' in message.lower():
            causes.extend(["Memory leaks", "Excessive memory allocation", "Garbage collection issues"])
        elif 'timeout' in message.lower():
            causes.extend(["Slow operations", "Resource contention", "Network delays"])
        elif level == 'ERROR':
            causes.extend(["Code bugs", "Configuration errors", "Resource exhaustion"])
        
        return causes
    
    def _generate_remediation_steps(self, message: str, level: str) -> List[str]:
        """Generate remediation steps for this type of event."""
        steps = ["Monitor the situation for escalation"]
        
        if level == 'ERROR':
            steps.extend([
                "Investigate root cause immediately",
                "Implement temporary workarounds if needed",
                "Apply permanent fixes"
            ])
        
        if 'connection' in message.lower():
            steps.extend([
                "Check network connectivity",
                "Restart affected services",
                "Review configuration settings"
            ])
        elif 'memory' in message.lower():
            steps.extend([
                "Restart services if memory usage is high",
                "Implement memory monitoring",
                "Fix memory leaks in code"
            ])
        elif 'timeout' in message.lower():
            steps.extend([
                "Adjust timeout values",
                "Optimize slow operations",
                "Scale resources if needed"
            ])
        
        return steps
    
    def _load_learning_objectives(self) -> Dict[str, List[str]]:
        """Load learning objectives for different skill levels."""
        return {
            LearningLevel.BEGINNER.value: [
                "Understand different log levels and their meanings",
                "Identify when something needs attention",
                "Learn to read basic log patterns",
                "Understand the business impact of different events"
            ],
            LearningLevel.INTERMEDIATE.value: [
                "Correlate events across different log sources",
                "Identify patterns that indicate problems",
                "Understand system performance indicators",
                "Learn investigation methodologies"
            ],
            LearningLevel.ADVANCED.value: [
                "Implement advanced monitoring strategies",
                "Design effective alerting systems",
                "Optimize system performance based on log analysis",
                "Create custom analysis tools and dashboards"
            ]
        }

class PatternRecognitionTrainer:
    """Interactive training system for log pattern recognition."""
    
    def __init__(self):
        self.training_scenarios = self._load_training_scenarios()
        self.progress_tracker = {}
    
    def create_training_scenario(self, scenario_type: str, difficulty: str) -> Dict[str, Any]:
        """Create a training scenario for pattern recognition."""
        scenario = {
            'id': f"{scenario_type}_{difficulty}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'type': scenario_type,
            'difficulty': difficulty,
            'description': self._get_scenario_description(scenario_type, difficulty),
            'log_entries': self._generate_scenario_logs(scenario_type, difficulty),
            'questions': self._generate_scenario_questions(scenario_type, difficulty),
            'learning_objectives': self._get_learning_objectives(scenario_type, difficulty),
            'hints': self._generate_hints(scenario_type, difficulty),
            'expected_patterns': self._get_expected_patterns(scenario_type, difficulty)
        }
        return scenario
    
    def _load_training_scenarios(self) -> Dict[str, Any]:
        """Load pre-defined training scenarios."""
        return {
            'connection_timeout': {
                'beginner': {
                    'description': 'Learn to identify connection timeout patterns',
                    'learning_objectives': ['Recognize timeout error messages', 'Understand connection failures']
                },
                'intermediate': {
                    'description': 'Analyze cascading connection failures',
                    'learning_objectives': ['Identify failure propagation', 'Understand system dependencies']
                },
                'advanced': {
                    'description': 'Design monitoring for connection resilience',
                    'learning_objectives': ['Implement circuit breakers', 'Design failure recovery']
                }
            },
            'memory_leak': {
                'beginner': {
                    'description': 'Learn to spot memory usage patterns',
                    'learning_objectives': ['Identify memory warnings', 'Understand memory allocation']
                },
                'intermediate': {
                    'description': 'Trace memory leak sources',
                    'learning_objectives': ['Analyze memory trends', 'Identify leak sources']
                },
                'advanced': {
                    'description': 'Implement memory monitoring strategies',
                    'learning_objectives': ['Design memory profiling', 'Optimize memory usage']
                }
            },
            'performance_degradation': {
                'beginner': {
                    'description': 'Recognize performance warning signs',
                    'learning_objectives': ['Identify slow responses', 'Understand performance metrics']
                },
                'intermediate': {
                    'description': 'Analyze performance bottlenecks',
                    'learning_objectives': ['Identify bottleneck sources', 'Understand resource contention']
                },
                'advanced': {
                    'description': 'Design performance optimization strategies',
                    'learning_objectives': ['Implement performance monitoring', 'Optimize system architecture']
                }
            }
        }
    
    def _get_scenario_description(self, scenario_type: str, difficulty: str) -> str:
        """Get description for a training scenario."""
        descriptions = {
            'connection_timeout': {
                'beginner': 'A web service is experiencing connection timeouts. Learn to identify the problem.',
                'intermediate': 'Multiple services are failing due to connection issues. Analyze the cascade.',
                'advanced': 'Design a monitoring system to prevent connection timeout cascades.'
            },
            'memory_leak': {
                'beginner': 'An application is using more memory over time. Identify the warning signs.',
                'intermediate': 'Trace the source of a memory leak through log analysis.',
                'advanced': 'Implement comprehensive memory monitoring and alerting.'
            },
            'performance_degradation': {
                'beginner': 'Users are complaining about slow response times. Find the indicators.',
                'intermediate': 'Analyze the root cause of system performance degradation.',
                'advanced': 'Design proactive performance monitoring and optimization.'
            }
        }
        
        return descriptions.get(scenario_type, {}).get(difficulty, 'Training scenario')
    
    def _generate_scenario_logs(self, scenario_type: str, difficulty: str) -> List[Dict[str, Any]]:
        """Generate sample log entries for the scenario."""
        logs = []
        
        if scenario_type == 'connection_timeout':
            logs.extend([
                {
                    'timestamp': '2024-01-15T10:30:00Z',
                    'level': 'INFO',
                    'message': 'Starting connection to database service',
                    'component': 'web-service'
                },
                {
                    'timestamp': '2024-01-15T10:30:05Z',
                    'level': 'WARN',
                    'message': 'Database connection taking longer than expected',
                    'component': 'web-service'
                },
                {
                    'timestamp': '2024-01-15T10:30:10Z',
                    'level': 'ERROR',
                    'message': 'Connection timeout: Unable to connect to database after 10 seconds',
                    'component': 'web-service'
                }
            ])
            
            if difficulty in ['intermediate', 'advanced']:
                logs.extend([
                    {
                        'timestamp': '2024-01-15T10:30:12Z',
                        'level': 'ERROR',
                        'message': 'Service unavailable: Database connection failed',
                        'component': 'api-gateway'
                    },
                    {
                        'timestamp': '2024-01-15T10:30:15Z',
                        'level': 'ERROR',
                        'message': 'Circuit breaker opened: Too many database failures',
                        'component': 'web-service'
                    }
                ])
        
        elif scenario_type == 'memory_leak':
            logs.extend([
                {
                    'timestamp': '2024-01-15T10:00:00Z',
                    'level': 'INFO',
                    'message': 'Application startup complete. Memory usage: 256MB',
                    'component': 'app-server'
                },
                {
                    'timestamp': '2024-01-15T11:00:00Z',
                    'level': 'INFO',
                    'message': 'Memory usage: 512MB',
                    'component': 'app-server'
                },
                {
                    'timestamp': '2024-01-15T12:00:00Z',
                    'level': 'WARN',
                    'message': 'Memory usage: 768MB - above normal threshold',
                    'component': 'app-server'
                },
                {
                    'timestamp': '2024-01-15T13:00:00Z',
                    'level': 'ERROR',
                    'message': 'Memory usage: 1024MB - critical threshold exceeded',
                    'component': 'app-server'
                }
            ])
        
        elif scenario_type == 'performance_degradation':
            logs.extend([
                {
                    'timestamp': '2024-01-15T10:00:00Z',
                    'level': 'INFO',
                    'message': 'Request processed in 150ms',
                    'component': 'web-service'
                },
                {
                    'timestamp': '2024-01-15T10:15:00Z',
                    'level': 'WARN',
                    'message': 'Request processed in 450ms - slower than expected',
                    'component': 'web-service'
                },
                {
                    'timestamp': '2024-01-15T10:30:00Z',
                    'level': 'ERROR',
                    'message': 'Request processed in 2100ms - exceeds timeout threshold',
                    'component': 'web-service'
                }
            ])
        
        return logs
    
    def _generate_scenario_questions(self, scenario_type: str, difficulty: str) -> List[Dict[str, Any]]:
        """Generate questions for the training scenario."""
        questions = []
        
        if scenario_type == 'connection_timeout':
            questions.extend([
                {
                    'question': 'What is the first sign of connection problems?',
                    'type': 'multiple_choice',
                    'options': [
                        'The ERROR message about timeout',
                        'The WARN message about slow connection',
                        'The INFO message about starting connection',
                        'The database server logs'
                    ],
                    'correct_answer': 1,
                    'explanation': 'The WARN message indicates the connection is taking longer than expected, which is the first warning sign.'
                },
                {
                    'question': 'How long did the system wait before timing out?',
                    'type': 'short_answer',
                    'correct_answer': '10 seconds',
                    'explanation': 'The error message explicitly states "after 10 seconds".'
                }
            ])
        
        elif scenario_type == 'memory_leak':
            questions.extend([
                {
                    'question': 'What pattern indicates a potential memory leak?',
                    'type': 'multiple_choice',
                    'options': [
                        'Memory usage staying constant',
                        'Memory usage gradually increasing over time',
                        'Memory usage randomly fluctuating',
                        'Memory usage decreasing over time'
                    ],
                    'correct_answer': 1,
                    'explanation': 'Gradually increasing memory usage over time is a classic sign of a memory leak.'
                }
            ])
        
        elif scenario_type == 'performance_degradation':
            questions.extend([
                {
                    'question': 'What is the performance degradation pattern?',
                    'type': 'multiple_choice',
                    'options': [
                        'Sudden spike in response time',
                        'Gradual increase in response time',
                        'Random response time variations',
                        'Constant response time'
                    ],
                    'correct_answer': 1,
                    'explanation': 'The logs show response times gradually increasing from 150ms to 2100ms.'
                }
            ])
        
        return questions
    
    def _get_learning_objectives(self, scenario_type: str, difficulty: str) -> List[str]:
        """Get learning objectives for the scenario."""
        objectives = self.training_scenarios.get(scenario_type, {}).get(difficulty, {}).get('learning_objectives', [])
        return objectives
    
    def _generate_hints(self, scenario_type: str, difficulty: str) -> List[str]:
        """Generate hints for the training scenario."""
        hints = {
            'connection_timeout': [
                'Look for progression from INFO to WARN to ERROR',
                'Pay attention to timing and duration',
                'Notice the component generating the logs'
            ],
            'memory_leak': [
                'Compare memory usage over time',
                'Look for increasing trends',
                'Notice threshold warnings'
            ],
            'performance_degradation': [
                'Compare response times across entries',
                'Look for gradual increases',
                'Notice threshold breaches'
            ]
        }
        
        return hints.get(scenario_type, ['Look for patterns in the logs', 'Pay attention to timestamps', 'Notice severity levels'])
    
    def _get_expected_patterns(self, scenario_type: str, difficulty: str) -> List[str]:
        """Get expected patterns students should identify."""
        patterns = {
            'connection_timeout': [
                'Progressive timeout pattern',
                'Service dependency failure',
                'Error escalation sequence'
            ],
            'memory_leak': [
                'Increasing memory usage trend',
                'Threshold breach pattern',
                'Resource exhaustion warning'
            ],
            'performance_degradation': [
                'Response time degradation',
                'Performance threshold breach',
                'Gradual system slowdown'
            ]
        }
        
        return patterns.get(scenario_type, ['General log patterns'])

# Export main classes for use in other modules
__all__ = [
    'StoryNarrativeEngine',
    'ContextualEducationSystem', 
    'PatternRecognitionTrainer',
    'LogStory',
    'LogStorySegment',
    'EducationalContext',
    'StoryType',
    'LearningLevel'
] 