"""
Summarizer for LogNarrator AI

This module handles AI-powered natural language generation for log analysis.
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from categorizer import CategorizedLogEntry, Phase, RiskLevel
from parser import LogEntry

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic library not available. Install with: pip install anthropic")


@dataclass
class LogSummary:
    """Complete summary of a log sequence"""
    timeline_summary: str
    key_events: List[str]
    risk_assessment: str
    recommendations: List[str]
    technical_details: str
    overall_status: str


class LogSummarizer:
    """Generates AI-powered summaries of log sequences"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the summarizer
        
        Args:
            api_key: Anthropic API key. If None, will try to get from environment
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        
        if ANTHROPIC_AVAILABLE and self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize Anthropic client: {e}")
                self.client = None
    
    def generate_summary(self, categorized_entries: List[CategorizedLogEntry]) -> LogSummary:
        """
        Generate a comprehensive summary of the log sequence
        
        Args:
            categorized_entries: List of categorized log entries
            
        Returns:
            LogSummary object with various types of analysis
        """
        if not categorized_entries:
            return LogSummary(
                timeline_summary="No log entries to analyze",
                key_events=[],
                risk_assessment="No risks detected",
                recommendations=[],
                technical_details="No technical information available",
                overall_status="Unknown"
            )
        
        # Generate different types of summaries
        timeline_summary = self._generate_timeline_summary(categorized_entries)
        key_events = self._extract_key_events(categorized_entries)
        risk_assessment = self._generate_risk_assessment(categorized_entries)
        recommendations = self._generate_recommendations(categorized_entries)
        technical_details = self._generate_technical_details(categorized_entries)
        overall_status = self._determine_overall_status(categorized_entries)
        
        return LogSummary(
            timeline_summary=timeline_summary,
            key_events=key_events,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            technical_details=technical_details,
            overall_status=overall_status
        )
    
    def _generate_timeline_summary(self, entries: List[CategorizedLogEntry]) -> str:
        """Generate a narrative timeline summary"""
        if self.client:
            return self._generate_ai_timeline_summary(entries)
        else:
            return self._generate_fallback_timeline_summary(entries)
    
    def _generate_ai_timeline_summary(self, entries: List[CategorizedLogEntry]) -> str:
        """Generate AI-powered timeline summary using Claude"""
        # Prepare the log data for AI analysis
        log_data = []
        for entry in entries:
            log_data.append({
                "timestamp": entry.log_entry.timestamp,
                "subsystem": entry.log_entry.subsystem,
                "event": entry.log_entry.event,
                "phase": entry.phase.value,
                "risk": entry.risk_level.value,
                "explanation": entry.explanation
            })
        
        # Create the prompt
        prompt = self._create_timeline_prompt(log_data)
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            print(f"AI summarization failed: {e}")
            return self._generate_fallback_timeline_summary(entries)
    
    def _create_timeline_prompt(self, log_data: List[Dict]) -> str:
        """Create a prompt for AI timeline generation"""
        log_text = "\n".join([
            f"Time: {entry['timestamp']} | System: {entry['subsystem']} | "
            f"Phase: {entry['phase']} | Risk: {entry['risk']} | "
            f"Event: {entry['event']}"
            for entry in log_data
        ])
        
        prompt = f"""
Analyze this machine/tool log sequence and provide a clear, concise summary of what happened:

{log_text}

Please provide a 2-3 sentence summary that explains:
1. What the system was trying to do
2. What problems occurred (if any)
3. How the situation was resolved (if it was)

Use plain English that a non-technical person could understand. Focus on the operational story, not individual log entries.
"""
        return prompt
    
    def _generate_fallback_timeline_summary(self, entries: List[CategorizedLogEntry]) -> str:
        """Generate a basic timeline summary without AI"""
        if not entries:
            return "No events to summarize"
        
        # Count phases and risks
        phases = [entry.phase for entry in entries]
        risks = [entry.risk_level for entry in entries]
        
        error_count = sum(1 for r in risks if r == RiskLevel.RED)
        warning_count = sum(1 for r in risks if r == RiskLevel.YELLOW)
        
        # Basic narrative construction
        summary_parts = []
        
        # Start with the first meaningful event
        first_entry = entries[0]
        if first_entry.phase == Phase.SAVE:
            summary_parts.append("The system initiated a save operation")
        elif first_entry.phase == Phase.SCAN:
            summary_parts.append("The system began scanning/imaging")
        elif first_entry.phase == Phase.INIT:
            summary_parts.append("The system started initialization")
        else:
            summary_parts.append("The system began operation")
        
        # Add problems if any
        if error_count > 0:
            summary_parts.append(f"but encountered {error_count} error(s)")
        if warning_count > 0:
            summary_parts.append(f"with {warning_count} warning(s)")
        
        # Add recovery information
        recovery_entries = [e for e in entries if e.phase == Phase.RECOVERY]
        if recovery_entries:
            summary_parts.append("The system attempted recovery")
        
        # Determine outcome
        last_entry = entries[-1]
        if last_entry.risk_level == RiskLevel.GREEN:
            summary_parts.append("and completed successfully")
        elif last_entry.risk_level == RiskLevel.RED:
            summary_parts.append("but ultimately failed")
        else:
            summary_parts.append("with mixed results")
        
        return ". ".join(summary_parts) + "."
    
    def _extract_key_events(self, entries: List[CategorizedLogEntry]) -> List[str]:
        """Extract key events from the log sequence"""
        key_events = []
        
        for entry in entries:
            # Include high-risk events
            if entry.risk_level == RiskLevel.RED:
                key_events.append(f"🔴 {entry.log_entry.timestamp}: {entry.log_entry.event}")
            
            # Include significant phase changes
            elif entry.phase in [Phase.ERROR, Phase.RECOVERY, Phase.ABORT]:
                key_events.append(f"{entry.risk_level.value} {entry.log_entry.timestamp}: {entry.log_entry.event}")
        
        # If no key events, include the first and last
        if not key_events and entries:
            first = entries[0]
            last = entries[-1]
            key_events.append(f"▶️ {first.log_entry.timestamp}: {first.log_entry.event}")
            if len(entries) > 1:
                key_events.append(f"🏁 {last.log_entry.timestamp}: {last.log_entry.event}")
        
        return key_events
    
    def _generate_risk_assessment(self, entries: List[CategorizedLogEntry]) -> str:
        """Generate a risk assessment summary"""
        if not entries:
            return "No data available for risk assessment"
        
        risk_counts = {
            RiskLevel.RED: 0,
            RiskLevel.YELLOW: 0,
            RiskLevel.GREEN: 0
        }
        
        for entry in entries:
            risk_counts[entry.risk_level] += 1
        
        total = len(entries)
        red_pct = (risk_counts[RiskLevel.RED] / total) * 100
        yellow_pct = (risk_counts[RiskLevel.YELLOW] / total) * 100
        green_pct = (risk_counts[RiskLevel.GREEN] / total) * 100
        
        if red_pct > 30:
            return f"HIGH RISK: {red_pct:.1f}% of events were critical errors. System requires immediate attention."
        elif red_pct > 10:
            return f"MEDIUM RISK: {red_pct:.1f}% critical errors detected. Monitor system closely."
        elif yellow_pct > 50:
            return f"MEDIUM RISK: {yellow_pct:.1f}% of events had warnings. System may be unstable."
        elif green_pct > 80:
            return f"LOW RISK: {green_pct:.1f}% of events completed successfully. System operating normally."
        else:
            return f"MIXED RISK: {red_pct:.1f}% errors, {yellow_pct:.1f}% warnings, {green_pct:.1f}% successful."
    
    def _generate_recommendations(self, entries: List[CategorizedLogEntry]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Count different types of issues
        error_count = sum(1 for e in entries if e.risk_level == RiskLevel.RED)
        retry_count = sum(1 for e in entries if "retry" in e.log_entry.event.lower())
        
        if error_count > 0:
            recommendations.append("Investigate and resolve critical errors before continuing operation")
        
        if retry_count > 2:
            recommendations.append("Check system stability - multiple retries indicate potential hardware issues")
        
        # Look for specific patterns
        af_issues = sum(1 for e in entries if "AF System" in e.log_entry.subsystem and e.risk_level == RiskLevel.RED)
        if af_issues > 0:
            recommendations.append("Calibrate or service auto-focus system")
        
        sem_issues = sum(1 for e in entries if "SEM" in e.log_entry.subsystem and "discarded" in e.log_entry.event.lower())
        if sem_issues > 0:
            recommendations.append("Check imaging parameters and sample conditions")
        
        if not recommendations:
            recommendations.append("System appears to be operating normally - continue monitoring")
        
        return recommendations
    
    def _generate_technical_details(self, entries: List[CategorizedLogEntry]) -> str:
        """Generate technical details summary"""
        if not entries:
            return "No technical data available"
        
        subsystems = set(entry.log_entry.subsystem for entry in entries)
        phases = set(entry.phase for entry in entries)
        
        duration = "Unknown"
        if len(entries) > 1:
            first_time = entries[0].log_entry.timestamp
            last_time = entries[-1].log_entry.timestamp
            duration = f"{first_time} to {last_time}"
        
        details = f"Duration: {duration}\n"
        details += f"Subsystems involved: {', '.join(sorted(subsystems))}\n"
        details += f"Operational phases: {', '.join(sorted([p.value for p in phases]))}\n"
        details += f"Total events: {len(entries)}"
        
        return details
    
    def _determine_overall_status(self, entries: List[CategorizedLogEntry]) -> str:
        """Determine overall operational status"""
        if not entries:
            return "Unknown"
        
        error_count = sum(1 for e in entries if e.risk_level == RiskLevel.RED)
        warning_count = sum(1 for e in entries if e.risk_level == RiskLevel.YELLOW)
        
        if error_count == 0 and warning_count == 0:
            return "✅ Success"
        elif error_count == 0 and warning_count > 0:
            return "⚠️ Completed with Warnings"
        elif error_count > 0:
            last_entry = entries[-1]
            if last_entry.risk_level == RiskLevel.GREEN:
                return "🔄 Recovered"
            else:
                return "❌ Failed"
        else:
            return "❓ Unknown"


def demo_summarizer():
    """Demo function to show summarizer in action"""
    from parser import LogParser
    from categorizer import LogCategorizer
    
    sample_log = """
00:39:24.243 Save Engine Async Save Triggered  
00:39:24.267 AF System Retry #1 Triggered  
00:39:26.214 SEM Image discarded
00:39:27.001 AF System Retry #2 Triggered  
00:39:28.520 AF System Recovery Complete
00:39:30.100 Save Engine Save Operation Complete
"""
    
    # Parse and categorize
    parser = LogParser()
    entries = parser.parse_text(sample_log)
    
    categorizer = LogCategorizer()
    categorized = categorizer.categorize_log_sequence(entries)
    
    # Generate summary
    summarizer = LogSummarizer()
    summary = summarizer.generate_summary(categorized)
    
    print("=== LOG ANALYSIS SUMMARY ===")
    print(f"\nOverall Status: {summary.overall_status}")
    print(f"\nTimeline Summary:")
    print(summary.timeline_summary)
    
    print(f"\nKey Events:")
    for event in summary.key_events:
        print(f"  {event}")
    
    print(f"\nRisk Assessment:")
    print(summary.risk_assessment)
    
    print(f"\nRecommendations:")
    for rec in summary.recommendations:
        print(f"  • {rec}")
    
    print(f"\nTechnical Details:")
    print(summary.technical_details)


if __name__ == "__main__":
    demo_summarizer() 