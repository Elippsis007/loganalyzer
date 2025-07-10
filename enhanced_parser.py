"""
Enhanced Parser for LogNarrator AI

This module provides enhanced parsing capabilities including:
- Multi-line log entry support
- Better format detection
- Performance optimizations
- Export capabilities
"""

import re
import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from parser import LogEntry


@dataclass
class MultiLineLogEntry:
    """Extended log entry that can span multiple lines"""
    timestamp: str
    subsystem: str
    event: str
    raw_lines: List[str]  # All raw lines that make up this entry
    line_numbers: List[int]  # Line numbers in original file
    entry_type: str  # 'single', 'multiline', 'stack_trace', 'json', 'xml'
    metadata: Dict  # Additional metadata


@dataclass
class ParsedLogFile:
    """Container for parsed log file with metadata"""
    entries: List[Union[LogEntry, MultiLineLogEntry]]
    file_info: Dict
    parsing_stats: Dict
    detected_format: str
    encoding_used: str


class EnhancedLogParser:
    """Enhanced parser with multi-line support and format detection"""
    
    def __init__(self):
        # Enhanced regex patterns for various log formats
        self.log_patterns = {
            'standard': re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s+([A-Za-z][A-Za-z\s\w]*?)\s+(.+)'),
            'iso_timestamp': re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\s+([A-Za-z][A-Za-z\s\w]*?)\s+(.+)'),
            'bracketed': re.compile(r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\]\s+([A-Za-z][A-Za-z\s\w]*?)\s+(.+)'),
            'sem_format': re.compile(r'\w{3}\s+\d{2},(\d{2}:\d{2}:\d{2}\.\d{3})\s+\w+\s+\[([^\]]+)\]\s+(.+)'),
            'sem_alt': re.compile(r'\w{3}\s+\d{2},(\d{2}:\d{2}:\d{2}\.\d{3})\s+\w+\s+([A-Za-z][A-Za-z\s\w]*?)\s+(.+)'),
            'syslog': re.compile(r'(\w{3}\s+\d{2}\s+\d{2}:\d{2}:\d{2})\s+(\w+)\s+(.+)'),
            'windows_event': re.compile(r'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})\s+([A-Za-z][A-Za-z\s\w]*?)\s+(.+)'),
        }
        
        # Multi-line patterns
        self.multiline_indicators = {
            'stack_trace': [
                r'^\s+at\s+[\w\.]+\(',
                r'^\s+\.\.\.\s+\d+\s+more',
                r'^\s*Caused by:',
                r'^\s*Exception in thread',
            ],
            'json': [
                r'^\s*[\{\[]',
                r'^\s*["\w]+\s*:',
                r'^\s*[\}\]]',
            ],
            'xml': [
                r'^\s*<[\w/]',
                r'^\s*</',
                r'^\s*/>',
            ],
            'continuation': [
                r'^\s+\w+',  # Indented continuation
                r'^\s*\.\.\.',  # Ellipsis continuation
                r'^$',  # Empty line as part of multiline
            ],
        }
        
        # Format detection patterns
        self.format_signatures = {
            'apache_access': r'\d+\.\d+\.\d+\.\d+.*\[.*\].*"(GET|POST|PUT|DELETE)',
            'apache_error': r'\[.*\]\s+\[.*\]\s+\[.*\]',
            'nginx': r'\d+\.\d+\.\d+\.\d+.*\[.*\].*"(GET|POST|PUT|DELETE).*"\s+\d+',
            'iis': r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}.*\d+\.\d+\.\d+\.\d+',
            'java_log': r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}.*\[(INFO|WARN|ERROR|DEBUG)\]',
            'python_log': r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}.*\[(INFO|WARNING|ERROR|DEBUG)\]',
            'windows_event': r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}.*Event ID',
            'sem_log': r'\w{3}\s+\d{2},\d{2}:\d{2}:\d{2}\.\d{3}.*\[(.*)\]',
        }
        
        # Performance tracking
        self.parsing_stats = {
            'total_lines': 0,
            'parsed_entries': 0,
            'multiline_entries': 0,
            'parsing_time': 0,
            'format_detection_time': 0,
        }
    
    def parse_file_enhanced(self, file_path: str) -> ParsedLogFile:
        """
        Enhanced file parsing with format detection and multi-line support
        
        Args:
            file_path: Path to the log file
            
        Returns:
            ParsedLogFile object with comprehensive parsing results
        """
        start_time = datetime.now()
        
        # Read file with multiple encoding attempts
        content, encoding_used = self._read_file_robust(file_path)
        if not content:
            return self._create_empty_result(file_path, "Failed to read file")
        
        # Detect log format
        format_start = datetime.now()
        detected_format = self._detect_log_format(content)
        format_time = (datetime.now() - format_start).total_seconds()
        
        # Parse with format-specific logic
        entries = self._parse_content_enhanced(content, detected_format)
        
        # Calculate file info
        file_info = self._get_file_info(file_path, content, encoding_used)
        
        # Calculate parsing stats
        total_time = (datetime.now() - start_time).total_seconds()
        parsing_stats = {
            **self.parsing_stats,
            'parsing_time': total_time,
            'format_detection_time': format_time,
            'entries_per_second': len(entries) / total_time if total_time > 0 else 0,
        }
        
        return ParsedLogFile(
            entries=entries,
            file_info=file_info,
            parsing_stats=parsing_stats,
            detected_format=detected_format,
            encoding_used=encoding_used
        )
    
    def _read_file_robust(self, file_path: str) -> Tuple[Optional[str], str]:
        """Read file with multiple encoding attempts"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii', 'utf-16']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                return content, encoding
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading file with {encoding}: {e}")
                continue
        
        # Last resort: binary read with error replacement
        try:
            with open(file_path, 'rb') as f:
                raw_content = f.read()
            content = raw_content.decode('utf-8', errors='replace')
            return content, 'utf-8-with-replacement'
        except Exception as e:
            print(f"Failed to read file: {e}")
            return None, 'failed'
    
    def _detect_log_format(self, content: str) -> str:
        """Detect the most likely log format"""
        lines = content.split('\n')[:100]  # Check first 100 lines
        sample = '\n'.join(lines)
        
        format_scores = {}
        
        for format_name, pattern in self.format_signatures.items():
            matches = len(re.findall(pattern, sample, re.MULTILINE))
            if matches > 0:
                format_scores[format_name] = matches
        
        if format_scores:
            best_format = max(format_scores, key=format_scores.get)
            return best_format
        
        # Fallback to pattern matching
        for pattern_name, pattern in self.log_patterns.items():
            matches = len([line for line in lines if pattern.match(line)])
            if matches > len(lines) * 0.1:  # At least 10% match
                return f"custom_{pattern_name}"
        
        return "unknown"
    
    def _parse_content_enhanced(self, content: str, detected_format: str) -> List[Union[LogEntry, MultiLineLogEntry]]:
        """Parse content with enhanced multi-line support"""
        lines = content.split('\n')
        self.parsing_stats['total_lines'] = len(lines)
        
        entries = []
        current_multiline = None
        
        for i, line in enumerate(lines):
            line = line.rstrip()
            if not line:
                continue
            
            # Check if this line starts a new log entry
            is_new_entry = self._is_new_log_entry(line)
            
            if is_new_entry:
                # Finish previous multiline entry if exists
                if current_multiline:
                    entries.append(self._finalize_multiline_entry(current_multiline))
                    current_multiline = None
                
                # Try to parse as single line entry
                entry = self._parse_single_line(line, i + 1)
                if entry:
                    # Check if this might be start of multiline
                    if self._might_be_multiline_start(line):
                        current_multiline = {
                            'base_entry': entry,
                            'additional_lines': [],
                            'line_numbers': [i + 1],
                        }
                    else:
                        entries.append(entry)
                        self.parsing_stats['parsed_entries'] += 1
            else:
                # This is a continuation line
                if current_multiline:
                    current_multiline['additional_lines'].append(line)
                    current_multiline['line_numbers'].append(i + 1)
                else:
                    # Orphaned continuation line - create generic entry
                    entry = LogEntry(
                        timestamp="Unknown",
                        subsystem="Unknown",
                        event=line,
                        raw_line=line,
                        line_number=i + 1
                    )
                    entries.append(entry)
        
        # Finish final multiline entry if exists
        if current_multiline:
            entries.append(self._finalize_multiline_entry(current_multiline))
        
        return entries
    
    def _is_new_log_entry(self, line: str) -> bool:
        """Check if line starts a new log entry"""
        for pattern in self.log_patterns.values():
            if pattern.match(line):
                return True
        return False
    
    def _might_be_multiline_start(self, line: str) -> bool:
        """Check if line might start a multiline entry"""
        indicators = [
            'exception', 'error', 'stack', 'trace', 'caused by',
            'json', 'xml', 'sql', 'query', '{', '<', 'select'
        ]
        line_lower = line.lower()
        return any(indicator in line_lower for indicator in indicators)
    
    def _parse_single_line(self, line: str, line_number: int) -> Optional[LogEntry]:
        """Parse a single line using appropriate pattern"""
        for pattern in self.log_patterns.values():
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
        
        # Fallback to generic parsing
        return LogEntry(
            timestamp="Unknown",
            subsystem="Unknown",
            event=line,
            raw_line=line,
            line_number=line_number
        )
    
    def _finalize_multiline_entry(self, multiline_data: Dict) -> MultiLineLogEntry:
        """Convert multiline data to MultiLineLogEntry"""
        base_entry = multiline_data['base_entry']
        additional_lines = multiline_data['additional_lines']
        line_numbers = multiline_data['line_numbers']
        
        # Combine all lines into event
        all_lines = [base_entry.event] + additional_lines
        combined_event = '\n'.join(all_lines)
        
        # Detect entry type
        entry_type = self._detect_multiline_type(all_lines)
        
        # Extract metadata
        metadata = self._extract_metadata(all_lines, entry_type)
        
        self.parsing_stats['multiline_entries'] += 1
        
        return MultiLineLogEntry(
            timestamp=base_entry.timestamp,
            subsystem=base_entry.subsystem,
            event=combined_event,
            raw_lines=[base_entry.raw_line] + additional_lines,
            line_numbers=line_numbers,
            entry_type=entry_type,
            metadata=metadata
        )
    
    def _detect_multiline_type(self, lines: List[str]) -> str:
        """Detect the type of multiline entry"""
        combined_text = '\n'.join(lines).lower()
        
        if any(indicator in combined_text for indicator in ['exception', 'stack trace', 'at ', 'caused by']):
            return 'stack_trace'
        elif combined_text.strip().startswith('{') or '"' in combined_text:
            return 'json'
        elif combined_text.strip().startswith('<'):
            return 'xml'
        elif 'select' in combined_text or 'insert' in combined_text:
            return 'sql'
        else:
            return 'multiline'
    
    def _extract_metadata(self, lines: List[str], entry_type: str) -> Dict:
        """Extract metadata from multiline entry"""
        metadata = {
            'line_count': len(lines),
            'total_chars': sum(len(line) for line in lines),
            'entry_type': entry_type,
        }
        
        if entry_type == 'stack_trace':
            metadata['exception_type'] = self._extract_exception_type(lines)
            metadata['stack_depth'] = len([line for line in lines if 'at ' in line])
        
        elif entry_type == 'json':
            metadata['json_valid'] = self._validate_json('\n'.join(lines))
        
        elif entry_type == 'xml':
            metadata['xml_valid'] = self._validate_xml('\n'.join(lines))
        
        return metadata
    
    def _extract_exception_type(self, lines: List[str]) -> Optional[str]:
        """Extract exception type from stack trace"""
        for line in lines:
            if 'Exception' in line or 'Error' in line:
                # Simple extraction - could be more sophisticated
                parts = line.split()
                for part in parts:
                    if 'Exception' in part or 'Error' in part:
                        return part.split(':')[0]
        return None
    
    def _validate_json(self, text: str) -> bool:
        """Check if text is valid JSON"""
        try:
            json.loads(text)
            return True
        except:
            return False
    
    def _validate_xml(self, text: str) -> bool:
        """Check if text is valid XML"""
        try:
            ET.fromstring(text)
            return True
        except:
            return False
    
    def _get_file_info(self, file_path: str, content: str, encoding: str) -> Dict:
        """Get comprehensive file information"""
        path = Path(file_path)
        lines = content.split('\n')
        
        return {
            'file_name': path.name,
            'file_size': path.stat().st_size if path.exists() else 0,
            'file_path': str(path.absolute()),
            'line_count': len(lines),
            'encoding': encoding,
            'first_line': lines[0] if lines else '',
            'last_line': lines[-1] if lines else '',
            'empty_lines': sum(1 for line in lines if not line.strip()),
            'average_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
        }
    
    def _create_empty_result(self, file_path: str, error_msg: str) -> ParsedLogFile:
        """Create empty result for failed parsing"""
        return ParsedLogFile(
            entries=[],
            file_info={'file_path': file_path, 'error': error_msg},
            parsing_stats={'error': error_msg},
            detected_format='unknown',
            encoding_used='failed'
        )
    
    def export_to_json(self, parsed_file: ParsedLogFile, output_path: str):
        """Export parsed results to JSON"""
        export_data = {
            'file_info': parsed_file.file_info,
            'parsing_stats': parsed_file.parsing_stats,
            'detected_format': parsed_file.detected_format,
            'encoding_used': parsed_file.encoding_used,
            'entries': []
        }
        
        for entry in parsed_file.entries:
            if isinstance(entry, MultiLineLogEntry):
                export_data['entries'].append({
                    'type': 'multiline',
                    **asdict(entry)
                })
            else:
                export_data['entries'].append({
                    'type': 'single',
                    **asdict(entry)
                })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def export_to_csv(self, parsed_file: ParsedLogFile, output_path: str):
        """Export parsed results to CSV"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'timestamp', 'subsystem', 'event', 'line_number', 
                'entry_type', 'multiline', 'metadata'
            ])
            
            # Data rows
            for entry in parsed_file.entries:
                if isinstance(entry, MultiLineLogEntry):
                    writer.writerow([
                        entry.timestamp,
                        entry.subsystem,
                        entry.event.replace('\n', '\\n'),  # Escape newlines
                        ','.join(map(str, entry.line_numbers)),
                        entry.entry_type,
                        'Yes',
                        json.dumps(entry.metadata)
                    ])
                else:
                    writer.writerow([
                        entry.timestamp,
                        entry.subsystem,
                        entry.event,
                        entry.line_number,
                        'single',
                        'No',
                        '{}'
                    ])
    
    def export_to_xml(self, parsed_file: ParsedLogFile, output_path: str):
        """Export parsed results to XML"""
        root = ET.Element('log_analysis')
        
        # File info
        file_info_elem = ET.SubElement(root, 'file_info')
        for key, value in parsed_file.file_info.items():
            elem = ET.SubElement(file_info_elem, key)
            elem.text = str(value)
        
        # Parsing stats
        stats_elem = ET.SubElement(root, 'parsing_stats')
        for key, value in parsed_file.parsing_stats.items():
            elem = ET.SubElement(stats_elem, key)
            elem.text = str(value)
        
        # Entries
        entries_elem = ET.SubElement(root, 'entries')
        for entry in parsed_file.entries:
            entry_elem = ET.SubElement(entries_elem, 'entry')
            
            if isinstance(entry, MultiLineLogEntry):
                entry_elem.set('type', 'multiline')
                entry_elem.set('entry_type', entry.entry_type)
                
                timestamp_elem = ET.SubElement(entry_elem, 'timestamp')
                timestamp_elem.text = entry.timestamp
                
                subsystem_elem = ET.SubElement(entry_elem, 'subsystem')
                subsystem_elem.text = entry.subsystem
                
                event_elem = ET.SubElement(entry_elem, 'event')
                event_elem.text = entry.event
                
                lines_elem = ET.SubElement(entry_elem, 'line_numbers')
                lines_elem.text = ','.join(map(str, entry.line_numbers))
                
            else:
                entry_elem.set('type', 'single')
                
                timestamp_elem = ET.SubElement(entry_elem, 'timestamp')
                timestamp_elem.text = entry.timestamp
                
                subsystem_elem = ET.SubElement(entry_elem, 'subsystem')
                subsystem_elem.text = entry.subsystem
                
                event_elem = ET.SubElement(entry_elem, 'event')
                event_elem.text = entry.event
                
                line_elem = ET.SubElement(entry_elem, 'line_number')
                line_elem.text = str(entry.line_number)
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True) 