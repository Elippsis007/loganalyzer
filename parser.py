"""
Log Parser for LogNarrator AI

This module handles parsing raw log entries into structured data.
"""

import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LogEntry:
    """Represents a parsed log entry"""
    timestamp: str
    subsystem: str
    event: str
    raw_line: str
    line_number: int


class LogParser:
    """Parses raw log files into structured LogEntry objects"""
    
    def __init__(self):
        # Main regex pattern for log parsing
        # Matches: timestamp (HH:MM:SS.mmm) + subsystem + event description
        self.log_pattern = re.compile(
            r'(\d{2}:\d{2}:\d{2}\.\d{3})\s+([A-Za-z][A-Za-z\s\w]*?)\s+(.+)'
        )
        
        # Alternative patterns for different log formats
        self.alt_patterns = [
            # Pattern without milliseconds: HH:MM:SS
            re.compile(r'(\d{2}:\d{2}:\d{2})\s+([A-Za-z][A-Za-z\s\w]*?)\s+(.+)'),
            # Pattern with date: YYYY-MM-DD HH:MM:SS.mmm
            re.compile(r'\d{4}-\d{2}-\d{2}\s+(\d{2}:\d{2}:\d{2}\.\d{3})\s+([A-Za-z][A-Za-z\s\w]*?)\s+(.+)'),
            # Pattern with brackets: [HH:MM:SS.mmm]
            re.compile(r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\]\s+([A-Za-z][A-Za-z\s\w]*?)\s+(.+)'),
            # SEM log format: Jul 04,00:39:24.243 INFO [ADR Logger] Event
            re.compile(r'\w{3}\s+\d{2},(\d{2}:\d{2}:\d{2}\.\d{3})\s+\w+\s+\[([^\]]+)\]\s+(.+)'),
            # SEM log format variation: Jul 04,00:39:24.243 INFO System Event
            re.compile(r'\w{3}\s+\d{2},(\d{2}:\d{2}:\d{2}\.\d{3})\s+\w+\s+([A-Za-z][A-Za-z\s\w]*?)\s+(.+)')
        ]
    
    def parse_line(self, line: str, line_number: int) -> Optional[LogEntry]:
        """
        Parse a single log line into a LogEntry object
        
        Args:
            line: Raw log line
            line_number: Line number in the original file
            
        Returns:
            LogEntry object or None if parsing fails
        """
        line = line.strip()
        if not line:
            return None
            
        # Try main pattern first
        match = self.log_pattern.match(line)
        if match:
            timestamp, subsystem, event = match.groups()
            return LogEntry(
                timestamp=timestamp.strip(),
                subsystem=subsystem.strip(),
                event=event.strip(),
                raw_line=line,
                line_number=line_number
            )
        
        # Try alternative patterns
        for pattern in self.alt_patterns:
            match = pattern.match(line)
            if match:
                timestamp, subsystem, event = match.groups()
                return LogEntry(
                    timestamp=timestamp.strip(),
                    subsystem=subsystem.strip(),
                    event=event.strip(),
                    raw_line=line,
                    line_number=line_number
                )
        
        # If no pattern matches, create a generic entry
        return LogEntry(
            timestamp="Unknown",
            subsystem="Unknown",
            event=line,
            raw_line=line,
            line_number=line_number
        )
    
    def parse_text(self, text: str) -> List[LogEntry]:
        """
        Parse a string containing multiple log lines
        
        Args:
            text: Multi-line log text
            
        Returns:
            List of LogEntry objects
        """
        lines = text.split('\n')
        entries = []
        
        for i, line in enumerate(lines, 1):
            entry = self.parse_line(line, i)
            if entry:
                entries.append(entry)
        
        return entries
    
    def parse_file(self, file_path: str) -> List[LogEntry]:
        """
        Parse a log file
        
        Args:
            file_path: Path to the log file
            
        Returns:
            List of LogEntry objects
        """
        try:
            # Try different encodings to handle various file formats
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                # If all encodings fail, read as binary and decode with error handling
                with open(file_path, 'rb') as f:
                    raw_content = f.read()
                content = raw_content.decode('utf-8', errors='replace')
            
            return self.parse_text(content)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
    
    def get_parsing_stats(self, entries: List[LogEntry]) -> Dict[str, any]:
        """
        Get statistics about the parsing results
        
        Args:
            entries: List of parsed log entries
            
        Returns:
            Dictionary with parsing statistics
        """
        if not entries:
            return {"total_entries": 0, "parsed_entries": 0, "unknown_entries": 0}
        
        unknown_count = sum(1 for entry in entries if entry.timestamp == "Unknown")
        parsed_count = len(entries) - unknown_count
        
        subsystems = set(entry.subsystem for entry in entries if entry.subsystem != "Unknown")
        
        return {
            "total_entries": len(entries),
            "parsed_entries": parsed_count,
            "unknown_entries": unknown_count,
            "unique_subsystems": len(subsystems),
            "subsystems": sorted(list(subsystems))
        }


def demo_parser():
    """Demo function to show parser in action"""
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
    
    print("Parsed Log Entries:")
    print("-" * 60)
    for entry in entries:
        print(f"Time: {entry.timestamp:15} | System: {entry.subsystem:12} | Event: {entry.event}")
    
    print("\nParsing Statistics:")
    stats = parser.get_parsing_stats(entries)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    demo_parser() 