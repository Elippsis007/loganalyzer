#!/usr/bin/env python3
"""
LogNarrator AI - Main CLI Interface

This is the primary entry point for the LogNarrator AI system.
Transform raw machine/tool logs into human-readable summaries with AI analysis.
"""

import os
import sys
import argparse
from typing import List, Optional
from pathlib import Path

# Rich for beautiful console output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: rich library not available. Install with: pip install rich")

# Import our modules
from parser import LogParser, LogEntry
from categorizer import LogCategorizer, CategorizedLogEntry
from summarizer import LogSummarizer, LogSummary


class LogNarratorCLI:
    """Main CLI interface for LogNarrator AI"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.parser = LogParser()
        self.categorizer = LogCategorizer()
        self.summarizer = LogSummarizer()
    
    def print_styled(self, text: str, style: str = ""):
        """Print text with styling if Rich is available"""
        if self.console:
            self.console.print(text, style=style)
        else:
            print(text)
    
    def print_panel(self, content: str, title: str = ""):
        """Print content in a panel if Rich is available"""
        if self.console:
            self.console.print(Panel(content, title=title))
        else:
            print(f"\n=== {title} ===")
            print(content)
            print("=" * (len(title) + 8))
    
    def analyze_log_text(self, log_text: str) -> LogSummary:
        """
        Analyze log text and return comprehensive summary
        
        Args:
            log_text: Raw log text to analyze
            
        Returns:
            LogSummary object with analysis results
        """
        # Step 1: Parse the log
        self.print_styled("🔍 Parsing log entries...", "blue")
        entries = self.parser.parse_text(log_text)
        
        if not entries:
            self.print_styled("❌ No valid log entries found", "red")
            return LogSummary(
                timeline_summary="No log entries could be parsed",
                key_events=[],
                risk_assessment="No data to assess",
                recommendations=[],
                technical_details="No technical information available",
                overall_status="Unknown"
            )
        
        parsing_stats = self.parser.get_parsing_stats(entries)
        self.print_styled(f"✅ Parsed {parsing_stats['parsed_entries']} entries", "green")
        
        # Step 2: Categorize entries
        self.print_styled("📊 Categorizing log entries...", "blue")
        categorized = self.categorizer.categorize_log_sequence(entries)
        
        # Step 3: Generate AI summary
        self.print_styled("🤖 Generating AI analysis...", "blue")
        summary = self.summarizer.generate_summary(categorized)
        
        return summary
    
    def analyze_log_file(self, file_path: str) -> LogSummary:
        """
        Analyze a log file and return comprehensive summary
        
        Args:
            file_path: Path to the log file
            
        Returns:
            LogSummary object with analysis results
        """
        self.print_styled(f"📁 Loading log file: {file_path}", "blue")
        
        # Use the parser's file reading method which handles encoding properly
        entries = self.parser.parse_file(file_path)
        
        if not entries:
            self.print_styled("❌ No valid log entries found", "red")
            return LogSummary(
                timeline_summary="No log entries could be parsed",
                key_events=[],
                risk_assessment="No data to assess",
                recommendations=[],
                technical_details="No technical information available",
                overall_status="Unknown"
            )
        
        parsing_stats = self.parser.get_parsing_stats(entries)
        self.print_styled(f"✅ Parsed {parsing_stats['parsed_entries']} entries", "green")
        
        # Step 2: Categorize entries
        self.print_styled("📊 Categorizing log entries...", "blue")
        categorized = self.categorizer.categorize_log_sequence(entries)
        
        # Step 3: Generate AI summary
        self.print_styled("🤖 Generating AI analysis...", "blue")
        summary = self.summarizer.generate_summary(categorized)
        
        return summary
    
    def display_detailed_timeline(self, log_text_or_file: str):
        """Display detailed timeline with categorization"""
        # Check if it's a file path or text content
        if len(log_text_or_file) < 500 and '\n' not in log_text_or_file[:100]:
            # Likely a file path
            entries = self.parser.parse_file(log_text_or_file)
        else:
            # Text content
            entries = self.parser.parse_text(log_text_or_file)
        categorized = self.categorizer.categorize_log_sequence(entries)
        
        if not categorized:
            self.print_styled("❌ No entries to display", "red")
            return
        
        if self.console:
            # Create a rich table
            table = Table(title="Detailed Log Timeline")
            table.add_column("Time", style="cyan", no_wrap=True)
            table.add_column("System", style="magenta")
            table.add_column("Phase", style="yellow")
            table.add_column("Risk", justify="center")
            table.add_column("Event", style="white")
            table.add_column("Explanation", style="dim")
            
            for entry in categorized:
                table.add_row(
                    entry.log_entry.timestamp,
                    entry.log_entry.subsystem,
                    entry.phase.value,
                    entry.risk_level.value,
                    entry.log_entry.event,
                    entry.explanation
                )
            
            self.console.print(table)
        else:
            # Fallback to plain text
            print("\n" + "="*100)
            print("DETAILED LOG TIMELINE")
            print("="*100)
            print(f"{'Time':<15} {'System':<15} {'Phase':<10} {'Risk':<6} {'Event':<30} {'Explanation'}")
            print("-"*100)
            
            for entry in categorized:
                print(f"{entry.log_entry.timestamp:<15} "
                      f"{entry.log_entry.subsystem:<15} "
                      f"{entry.phase.value:<10} "
                      f"{entry.risk_level.value:<6} "
                      f"{entry.log_entry.event:<30} "
                      f"{entry.explanation}")
    
    def display_summary(self, summary: LogSummary):
        """Display the analysis summary"""
        if self.console:
            # Rich formatted output
            self.console.print(f"\n[bold green]Overall Status: {summary.overall_status}[/bold green]")
            
            self.console.print(Panel(
                summary.timeline_summary,
                title="📖 Timeline Summary",
                border_style="blue"
            ))
            
            if summary.key_events:
                key_events_text = "\n".join(summary.key_events)
                self.console.print(Panel(
                    key_events_text,
                    title="🔑 Key Events",
                    border_style="yellow"
                ))
            
            self.console.print(Panel(
                summary.risk_assessment,
                title="⚠️ Risk Assessment",
                border_style="red"
            ))
            
            if summary.recommendations:
                recommendations_text = "\n".join([f"• {rec}" for rec in summary.recommendations])
                self.console.print(Panel(
                    recommendations_text,
                    title="💡 Recommendations",
                    border_style="green"
                ))
            
            self.console.print(Panel(
                summary.technical_details,
                title="🔧 Technical Details",
                border_style="cyan"
            ))
            
        else:
            # Plain text output
            print(f"\n=== LOG ANALYSIS SUMMARY ===")
            print(f"Overall Status: {summary.overall_status}")
            
            print(f"\n📖 Timeline Summary:")
            print(summary.timeline_summary)
            
            if summary.key_events:
                print(f"\n🔑 Key Events:")
                for event in summary.key_events:
                    print(f"  {event}")
            
            print(f"\n⚠️ Risk Assessment:")
            print(summary.risk_assessment)
            
            if summary.recommendations:
                print(f"\n💡 Recommendations:")
                for rec in summary.recommendations:
                    print(f"  • {rec}")
            
            print(f"\n🔧 Technical Details:")
            print(summary.technical_details)
    
    def interactive_mode(self):
        """Run in interactive mode for pasting logs"""
        self.print_styled("🎯 LogNarrator AI - Interactive Mode", "bold green")
        self.print_styled("Paste your log text below. Press Ctrl+D (Linux/Mac) or Ctrl+Z (Windows) when done.", "dim")
        
        try:
            log_text = sys.stdin.read()
            if log_text.strip():
                summary = self.analyze_log_text(log_text)
                self.display_summary(summary)
            else:
                self.print_styled("❌ No log text provided", "red")
        except KeyboardInterrupt:
            self.print_styled("\n👋 Goodbye!", "yellow")
            sys.exit(0)
    
    def run_cli(self):
        """Run the main CLI interface"""
        parser = argparse.ArgumentParser(
            description="LogNarrator AI - Transform raw logs into readable summaries",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python main.py -f logfile.txt           # Analyze a log file
  python main.py -i                       # Interactive mode (paste logs)
  python main.py -f logfile.txt -d        # Show detailed timeline
  python main.py -f logfile.txt -o summary.txt  # Save summary to file
  
  cat logfile.txt | python main.py        # Pipe log data
            """
        )
        
        parser.add_argument("-f", "--file", type=str, help="Log file to analyze")
        parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
        parser.add_argument("-d", "--detailed", action="store_true", help="Show detailed timeline")
        parser.add_argument("-o", "--output", type=str, help="Output file for summary")
        parser.add_argument("--api-key", type=str, help="Anthropic API key for enhanced AI analysis")
        
        args = parser.parse_args()
        
        # Set API key if provided
        if args.api_key:
            os.environ["ANTHROPIC_API_KEY"] = args.api_key
            self.summarizer = LogSummarizer(args.api_key)
        
        # Check if data is being piped
        if not sys.stdin.isatty():
            log_text = sys.stdin.read()
            if log_text.strip():
                summary = self.analyze_log_text(log_text)
                if args.detailed:
                    self.display_detailed_timeline(log_text)
                self.display_summary(summary)
                
                if args.output:
                    self.save_summary_to_file(summary, args.output)
                return
        
        # Handle file input
        if args.file:
            if not os.path.exists(args.file):
                self.print_styled(f"❌ File not found: {args.file}", "red")
                sys.exit(1)
            
            if args.detailed:
                self.display_detailed_timeline(args.file)
            
            summary = self.analyze_log_file(args.file)
            self.display_summary(summary)
            
            if args.output:
                self.save_summary_to_file(summary, args.output)
            return
        
        # Handle interactive mode
        if args.interactive:
            self.interactive_mode()
            return
        
        # No arguments provided, show help
        parser.print_help()
    
    def save_summary_to_file(self, summary: LogSummary, output_file: str):
        """Save summary to a text file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=== LOG ANALYSIS SUMMARY ===\n")
                f.write(f"Overall Status: {summary.overall_status}\n\n")
                
                f.write("Timeline Summary:\n")
                f.write(summary.timeline_summary + "\n\n")
                
                if summary.key_events:
                    f.write("Key Events:\n")
                    for event in summary.key_events:
                        f.write(f"  {event}\n")
                    f.write("\n")
                
                f.write("Risk Assessment:\n")
                f.write(summary.risk_assessment + "\n\n")
                
                if summary.recommendations:
                    f.write("Recommendations:\n")
                    for rec in summary.recommendations:
                        f.write(f"  • {rec}\n")
                    f.write("\n")
                
                f.write("Technical Details:\n")
                f.write(summary.technical_details + "\n")
            
            self.print_styled(f"📁 Summary saved to: {output_file}", "green")
        except Exception as e:
            self.print_styled(f"❌ Error saving file: {e}", "red")


def main():
    """Main entry point"""
    cli = LogNarratorCLI()
    cli.run_cli()


if __name__ == "__main__":
    main() 