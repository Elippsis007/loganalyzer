"""
AI Narrator compatibility layer for Production LogNarrator AI

Uses the existing summarizer as the AI narrator component.
"""

from summarizer import LogSummarizer, LogSummary
from typing import List

class AILogNarrator:
    """AI narrator that creates natural language narratives from log analysis"""
    
    def __init__(self):
        self.summarizer = LogSummarizer()
    
    def create_narrative(self, categorized_entries: List, summary: LogSummary) -> str:
        """
        Create a comprehensive narrative from categorized entries and summary
        
        Args:
            categorized_entries: List of categorized log entries
            summary: LogSummary object
            
        Returns:
            Formatted narrative string
        """
        
        narrative_parts = []
        
        # Title
        narrative_parts.append("🤖 AI LOG ANALYSIS NARRATIVE")
        narrative_parts.append("=" * 50)
        
        # Overall status
        narrative_parts.append(f"\n📊 OVERALL STATUS: {summary.overall_status}")
        
        # Timeline summary
        narrative_parts.append(f"\n📖 WHAT HAPPENED:")
        narrative_parts.append(summary.timeline_summary)
        
        # Key events
        if summary.key_events:
            narrative_parts.append(f"\n🔍 KEY EVENTS:")
            for event in summary.key_events:
                narrative_parts.append(f"  • {event}")
        
        # Risk assessment
        narrative_parts.append(f"\n⚠️ RISK ASSESSMENT:")
        narrative_parts.append(summary.risk_assessment)
        
        # Recommendations
        if summary.recommendations:
            narrative_parts.append(f"\n💡 RECOMMENDATIONS:")
            for rec in summary.recommendations:
                narrative_parts.append(f"  • {rec}")
        
        # Technical details
        narrative_parts.append(f"\n🔧 TECHNICAL DETAILS:")
        narrative_parts.append(summary.technical_details)
        
        return "\n".join(narrative_parts) 