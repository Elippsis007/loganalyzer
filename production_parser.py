"""
Production Multi-line Parser for LogNarrator AI

Real multi-line log parsing with:
- Stack trace reconstruction
- JSON/XML extraction
- Log continuation detection
- Advanced format detection
- Performance optimization

No demo data - production functionality only.
"""

import re
import json
import xml.etree.ElementTree as ET
import csv
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import hashlib
import traceback


class LogEntryType(Enum):
    """Types of log entries"""
    SINGLE_LINE = "single_line"
    MULTI_LINE = "multi_line"
    STACK_TRACE = "stack_trace"
    JSON_DATA = "json_data"
    XML_DATA = "xml_data"
    SQL_QUERY = "sql_query"
    CONFIGURATION = "configuration"
    BINARY_DATA = "binary_data"


@dataclass
class ParsedContent:
    """Structured content extracted from logs"""
    content_type: LogEntryType
    raw_content: str
    structured_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    parsing_errors: Optional[List[str]] = None


@dataclass
class ProductionLogEntry:
    """Production log entry with enhanced capabilities"""
    timestamp: str
    subsystem: str
    event: str
    raw_lines: List[str]
    line_numbers: List[int]
    entry_type: LogEntryType
    parsed_content: Optional[ParsedContent]
    file_position: int
    parsing_confidence: float
    entry_hash: str


@dataclass
class ParsingMetrics:
    """Real parsing performance metrics"""
    total_lines_processed: int
    successful_parses: int
    multi_line_entries: int
    stack_traces_found: int
    json_blocks_found: int
    xml_blocks_found: int
    parsing_time: float
    average_confidence: float
    encoding_used: str
    format_detected: str
    errors_encountered: List[str]


class ProductionLogParser:
    """Production-ready multi-line log parser"""
    
    def __init__(self, db_path: str = "lognarrator.db"):
        self.db_path = db_path
        self._init_parser_database()
        
        # Enhanced timestamp patterns for various log formats
        self.timestamp_patterns = [
            # Standard: HH:MM:SS.mmm
            r'(\d{2}:\d{2}:\d{2}\.\d{3})',
            # ISO with date: YYYY-MM-DDTHH:MM:SS.mmm
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})',
            # SEM format: Jul 04,HH:MM:SS.mmm
            r'\w{3}\s+\d{2},(\d{2}:\d{2}:\d{2}\.\d{3})',
            # Unix timestamp: 1234567890.123
            r'(\d{10}\.\d{3})',
            # Windows: MM/DD/YYYY HH:MM:SS
            r'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})',
            # Syslog: MMM DD HH:MM:SS
            r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})',
        ]
        
        # Log format patterns with extraction groups
        self.log_format_patterns = {
            'standard': re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s+([^\s]+(?:\s+[^\s]+)*?)\s+(.+)'),
            'sem_bracketed': re.compile(r'\w{3}\s+\d{2},(\d{2}:\d{2}:\d{2}\.\d{3})\s+\w+\s+\[([^\]]+)\]\s+(.+)'),
            'sem_standard': re.compile(r'\w{3}\s+\d{2},(\d{2}:\d{2}:\d{2}\.\d{3})\s+\w+\s+([^\s]+(?:\s+[^\s]+)*?)\s+(.+)'),
            'iso_timestamp': re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\s+([^\s]+)\s+(.+)'),
            'syslog': re.compile(r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+([^\s]+)\s+(.+)'),
            'windows': re.compile(r'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})\s+([^\s]+)\s+(.+)'),
        }
        
        # Multi-line detection patterns
        self.multiline_start_patterns = {
            'stack_trace': [
                r'.*[Ee]xception.*:',
                r'.*[Ee]rror.*:',
                r'.*[Ss]tack\s+[Tt]race.*:',
                r'.*\s+at\s+[\w\.]+\(',
                r'.*[Cc]aused\s+by:',
            ],
            'json_start': [
                r'.*\{\s*$',
                r'.*\[\s*$',
                r'.*"[\w\s]+"\s*:\s*\{',
            ],
            'xml_start': [
                r'.*<[\w:]+[^>]*>\s*$',
                r'.*<\?xml.*\?>\s*$',
            ],
            'sql_start': [
                r'.*\b(SELECT|INSERT|UPDATE|DELETE)\b.*',
                r'.*\bFROM\s+\w+.*',
            ],
            'config_start': [
                r'.*\[[\w\s\.]+\]\s*$',
                r'.*[\w\.]+\s*=\s*.*',
            ],
        }
        
        # Multi-line continuation patterns
        self.continuation_patterns = {
            'stack_trace': [
                r'^\s+at\s+[\w\$\.]+\(',
                r'^\s+\.\.\.\s+\d+\s+more',
                r'^\s*Caused\s+by:',
                r'^\s+\[[\w\.]+\]',
            ],
            'json_continuation': [
                r'^\s*"[\w\s]+"\s*:',
                r'^\s*[\}\]],?\s*$',
                r'^\s*\{',
                r'^\s*\[',
            ],
            'xml_continuation': [
                r'^\s*<[\w:/]+[^>]*>',
                r'^\s*</[\w:]+>\s*$',
                r'^\s*[\w\s]+$',  # Text content
            ],
            'sql_continuation': [
                r'^\s+(FROM|WHERE|ORDER\s+BY|GROUP\s+BY|HAVING)\b',
                r'^\s+(AND|OR)\b',
                r'^\s*\w+\s*[,\)]\s*$',
            ],
            'indented_continuation': [
                r'^\s{2,}\S',  # Indented lines
                r'^\t+\S',     # Tab-indented lines
            ],
        }
        
        # End patterns for multi-line blocks
        self.multiline_end_patterns = {
            'stack_trace': [
                r'^\s*$',  # Empty line
                r'^\d{2}:\d{2}:\d{2}',  # New timestamp
            ],
            'json_end': [
                r'^\s*[\}\]]\s*$',
                r'.*[\}\]]\s*[,;]?\s*$',
            ],
            'xml_end': [
                r'.*</[\w:]+>\s*$',
                r'.*/>\s*$',
            ],
        }
        
        # Performance tracking
        self.parsing_stats = {
            'files_processed': 0,
            'total_parsing_time': 0.0,
            'entries_parsed': 0,
            'multiline_entries_found': 0,
        }
    
    def _init_parser_database(self):
        """Initialize database for parser metrics and caching"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS parsing_metrics (
                    id INTEGER PRIMARY KEY,
                    file_path TEXT,
                    file_hash TEXT,
                    lines_processed INTEGER,
                    entries_found INTEGER,
                    multiline_entries INTEGER,
                    parsing_time REAL,
                    format_detected TEXT,
                    encoding_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS format_patterns (
                    id INTEGER PRIMARY KEY,
                    pattern_name TEXT UNIQUE,
                    pattern_regex TEXT,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.5,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def parse_file_production(self, file_path: str) -> Tuple[List[ProductionLogEntry], ParsingMetrics]:
        """
        Production file parsing with comprehensive multi-line support
        """
        start_time = datetime.now()
        
        # Read file with encoding detection
        content, encoding_used = self._read_file_robust(file_path)
        if not content:
            return [], self._create_error_metrics(file_path, "Failed to read file")
        
        # Detect log format
        detected_format = self._detect_log_format_production(content)
        
        # Parse content with multi-line support
        entries = self._parse_content_multiline(content, detected_format, file_path)
        
        # Calculate metrics
        parsing_time = (datetime.now() - start_time).total_seconds()
        
        # Create comprehensive metrics
        metrics = self._create_parsing_metrics(
            file_path, content, entries, parsing_time, detected_format, encoding_used
        )
        
        # Store metrics in database
        self._store_parsing_metrics(file_path, metrics)
        
        return entries, metrics
    
    def _read_file_robust(self, file_path: str) -> Tuple[Optional[str], str]:
        """Robust file reading with multiple encoding attempts"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii', 'utf-16', 'iso-8859-1']
        
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
    
    def _detect_log_format_production(self, content: str) -> str:
        """Production log format detection"""
        lines = content.split('\n')[:100]  # Sample first 100 lines
        
        format_scores = {}
        
        for format_name, pattern in self.log_format_patterns.items():
            matches = 0
            for line in lines:
                if line.strip() and pattern.match(line.strip()):
                    matches += 1
            
            if matches > 0:
                score = matches / len([l for l in lines if l.strip()])
                format_scores[format_name] = score
        
        if format_scores:
            best_format = max(format_scores, key=format_scores.get)
            confidence = format_scores[best_format]
            
            # Store learning data
            self._update_format_confidence(best_format, confidence > 0.5)
            
            return best_format
        
        return 'unknown'
    
    def _parse_content_multiline(self, content: str, detected_format: str, file_path: str) -> List[ProductionLogEntry]:
        """Parse content with comprehensive multi-line support"""
        lines = content.split('\n')
        entries = []
        
        i = 0
        file_position = 0
        
        while i < len(lines):
            line = lines[i].rstrip()
            
            if not line:
                i += 1
                file_position += len(lines[i-1]) + 1 if i > 0 else 0
                continue
            
            # Check if this line starts a new log entry
            parsed_entry = self._try_parse_single_line(line, i + 1, detected_format, file_position)
            
            if parsed_entry:
                # Check if this might be a multi-line entry
                multiline_type = self._detect_multiline_start(line)
                
                if multiline_type:
                    # Parse as multi-line entry
                    multiline_entry, lines_consumed = self._parse_multiline_entry(
                        lines, i, parsed_entry, multiline_type, detected_format, file_position
                    )
                    entries.append(multiline_entry)
                    i += lines_consumed
                else:
                    # Single line entry
                    entries.append(parsed_entry)
                    i += 1
            else:
                # Could be a continuation line - try to attach to previous entry
                if entries and self._is_continuation_line(line, entries[-1].entry_type):
                    last_entry = entries[-1]
                    self._append_to_multiline_entry(last_entry, line, i + 1)
                else:
                    # Orphaned line - create minimal entry
                    orphan_entry = self._create_orphaned_entry(line, i + 1, file_position)
                    entries.append(orphan_entry)
                
                i += 1
            
            file_position += len(line) + 1
        
        # Post-process entries for better structure
        self._post_process_entries(entries)
        
        return entries
    
    def _try_parse_single_line(self, line: str, line_number: int, format_name: str, file_position: int) -> Optional[ProductionLogEntry]:
        """Try to parse a line as the start of a log entry"""
        
        if format_name in self.log_format_patterns:
            pattern = self.log_format_patterns[format_name]
            match = pattern.match(line.strip())
            
            if match:
                groups = match.groups()
                if len(groups) >= 3:
                    timestamp, subsystem, event = groups[0], groups[1], groups[2]
                    
                    return ProductionLogEntry(
                        timestamp=timestamp.strip(),
                        subsystem=subsystem.strip(),
                        event=event.strip(),
                        raw_lines=[line],
                        line_numbers=[line_number],
                        entry_type=LogEntryType.SINGLE_LINE,
                        parsed_content=None,
                        file_position=file_position,
                        parsing_confidence=0.9,
                        entry_hash=self._generate_entry_hash(line, line_number)
                    )
        
        # Try other patterns if specific format failed
        for pattern in self.log_format_patterns.values():
            match = pattern.match(line.strip())
            if match:
                groups = match.groups()
                if len(groups) >= 3:
                    timestamp, subsystem, event = groups[0], groups[1], groups[2]
                    
                    return ProductionLogEntry(
                        timestamp=timestamp.strip(),
                        subsystem=subsystem.strip(),
                        event=event.strip(),
                        raw_lines=[line],
                        line_numbers=[line_number],
                        entry_type=LogEntryType.SINGLE_LINE,
                        parsed_content=None,
                        file_position=file_position,
                        parsing_confidence=0.7,  # Lower confidence for fallback
                        entry_hash=self._generate_entry_hash(line, line_number)
                    )
        
        return None
    
    def _detect_multiline_start(self, line: str) -> Optional[str]:
        """Detect if line starts a multi-line entry"""
        line_lower = line.lower()
        
        for entry_type, patterns in self.multiline_start_patterns.items():
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    return entry_type
        
        return None
    
    def _parse_multiline_entry(self, lines: List[str], start_index: int, 
                             base_entry: ProductionLogEntry, multiline_type: str,
                             format_name: str, file_position: int) -> Tuple[ProductionLogEntry, int]:
        """Parse a multi-line log entry"""
        
        all_lines = [lines[start_index]]
        line_numbers = [start_index + 1]
        lines_consumed = 1
        
        # Map multiline_type to LogEntryType
        type_mapping = {
            'stack_trace': LogEntryType.STACK_TRACE,
            'json_start': LogEntryType.JSON_DATA,
            'xml_start': LogEntryType.XML_DATA,
            'sql_start': LogEntryType.SQL_QUERY,
            'config_start': LogEntryType.CONFIGURATION,
        }
        mapped_entry_type = type_mapping.get(multiline_type, LogEntryType.MULTI_LINE)
        
        # Look ahead for continuation lines
        i = start_index + 1
        while i < len(lines):
            line = lines[i]
            
            # Check if this is a continuation
            if self._is_continuation_line(line, mapped_entry_type):
                all_lines.append(line)
                line_numbers.append(i + 1)
                lines_consumed += 1
                i += 1
            elif self._is_multiline_end(line, multiline_type):
                # End of multi-line block
                break
            elif not line.strip():
                # Empty line - might be part of multi-line or end
                all_lines.append(line)
                line_numbers.append(i + 1)
                lines_consumed += 1
                i += 1
            else:
                # Check if next line starts a new log entry
                if self._try_parse_single_line(line, i + 1, format_name, file_position):
                    break
                else:
                    # Assume it's part of the multi-line entry
                    all_lines.append(line)
                    line_numbers.append(i + 1)
                    lines_consumed += 1
                    i += 1
        
        # Parse the collected content
        combined_content = '\n'.join(all_lines)
        parsed_content = self._parse_structured_content(combined_content, multiline_type)
        
        # Determine final entry type
        final_entry_type = self._determine_final_entry_type(multiline_type, parsed_content)
        
        # Calculate confidence based on parsing success
        confidence = self._calculate_parsing_confidence(parsed_content, len(all_lines))
        
        return ProductionLogEntry(
            timestamp=base_entry.timestamp,
            subsystem=base_entry.subsystem,
            event=combined_content,
            raw_lines=all_lines,
            line_numbers=line_numbers,
            entry_type=final_entry_type,
            parsed_content=parsed_content,
            file_position=file_position,
            parsing_confidence=confidence,
            entry_hash=self._generate_entry_hash(combined_content, start_index + 1)
        ), lines_consumed
    
    def _is_continuation_line(self, line: str, entry_type: LogEntryType) -> bool:
        """Check if line is a continuation of a multi-line entry"""
        
        if entry_type == LogEntryType.SINGLE_LINE:
            return False
        
        type_map = {
            LogEntryType.STACK_TRACE: 'stack_trace',
            LogEntryType.JSON_DATA: 'json_continuation',
            LogEntryType.XML_DATA: 'xml_continuation',
            LogEntryType.SQL_QUERY: 'sql_continuation',
        }
        
        pattern_key = type_map.get(entry_type, 'indented_continuation')
        
        if pattern_key in self.continuation_patterns:
            patterns = self.continuation_patterns[pattern_key]
            for pattern in patterns:
                if re.match(pattern, line):
                    return True
        
        # Generic indentation check
        if line.startswith(('  ', '\t')) and line.strip():
            return True
        
        return False
    
    def _is_multiline_end(self, line: str, multiline_type: str) -> bool:
        """Check if line marks the end of a multi-line entry"""
        
        if multiline_type in self.multiline_end_patterns:
            patterns = self.multiline_end_patterns[multiline_type]
            for pattern in patterns:
                if re.match(pattern, line):
                    return True
        
        # Check for new log entry start
        for pattern in self.log_format_patterns.values():
            if pattern.match(line.strip()):
                return True
        
        return False
    
    def _parse_structured_content(self, content: str, content_type: str) -> ParsedContent:
        """Parse structured content from multi-line entries"""
        
        errors = []
        structured_data = None
        metadata = {}
        final_content_type = LogEntryType.MULTI_LINE  # Default type
        
        try:
            if content_type == 'json_start':
                structured_data = self._extract_json_data(content)
                final_content_type = LogEntryType.JSON_DATA
                
            elif content_type == 'xml_start':
                structured_data = self._extract_xml_data(content)
                final_content_type = LogEntryType.XML_DATA
                
            elif content_type == 'stack_trace':
                structured_data = self._extract_stack_trace_data(content)
                final_content_type = LogEntryType.STACK_TRACE
                
            elif content_type == 'sql_start':
                structured_data = self._extract_sql_data(content)
                final_content_type = LogEntryType.SQL_QUERY
                
            elif content_type == 'config_start':
                # Configuration data - could add structured parsing later
                final_content_type = LogEntryType.CONFIGURATION
                
            else:
                final_content_type = LogEntryType.MULTI_LINE
                
        except Exception as e:
            errors.append(f"Parsing error: {str(e)}")
        
        # Extract metadata
        metadata = {
            'line_count': len(content.split('\n')),
            'character_count': len(content),
            'contains_special_chars': bool(re.search(r'[^\w\s]', content)),
            'indentation_detected': bool(re.search(r'^\s{2,}', content, re.MULTILINE)),
        }
        
        return ParsedContent(
            content_type=final_content_type,
            raw_content=content,
            structured_data=structured_data,
            metadata=metadata,
            parsing_errors=errors if errors else None
        )
    
    def _extract_json_data(self, content: str) -> Optional[Dict]:
        """Extract and parse JSON data"""
        try:
            # Try to find JSON block
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            
            # Try array format
            array_match = re.search(r'\[.*\]', content, re.DOTALL)
            if array_match:
                json_str = array_match.group()
                return json.loads(json_str)
                
        except json.JSONDecodeError:
            # Try to clean up and parse again
            try:
                # Remove log prefixes and clean content
                cleaned = re.sub(r'^\d{2}:\d{2}:\d{2}\.\d{3}.*?(?=\{|\[)', '', content, flags=re.MULTILINE)
                cleaned = re.sub(r'^\s*[A-Za-z\s]+\s*', '', cleaned, flags=re.MULTILINE)
                
                json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
        
        return None
    
    def _extract_xml_data(self, content: str) -> Optional[Dict]:
        """Extract and parse XML data"""
        try:
            # Find XML block
            xml_match = re.search(r'<[^>]+>.*</[^>]+>', content, re.DOTALL)
            if xml_match:
                xml_str = xml_match.group()
                root = ET.fromstring(xml_str)
                return self._xml_to_dict(root)
                
        except ET.ParseError:
            # Try to clean and parse
            try:
                # Remove log prefixes
                cleaned = re.sub(r'^\d{2}:\d{2}:\d{2}\.\d{3}.*?(?=<)', '', content, flags=re.MULTILINE)
                xml_match = re.search(r'<[^>]+>.*</[^>]+>', cleaned, re.DOTALL)
                if xml_match:
                    root = ET.fromstring(xml_match.group())
                    return self._xml_to_dict(root)
            except:
                pass
        
        return None
    
    def _xml_to_dict(self, element) -> Dict:
        """Convert XML element to dictionary"""
        result = {}
        
        # Add attributes
        if element.attrib:
            result['@attributes'] = element.attrib
        
        # Add text content
        if element.text and element.text.strip():
            result['text'] = element.text.strip()
        
        # Add child elements
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result
    
    def _extract_stack_trace_data(self, content: str) -> Optional[Dict]:
        """Extract structured data from stack traces"""
        try:
            lines = content.split('\n')
            
            # Find exception type and message
            exception_line = lines[0] if lines else ""
            exception_match = re.match(r'.*?([A-Za-z\.]+Exception)(?::\s*(.*))?', exception_line)
            
            exception_type = None
            exception_message = None
            
            if exception_match:
                exception_type = exception_match.group(1)
                exception_message = exception_match.group(2) if exception_match.group(2) else ""
            
            # Extract stack frames
            stack_frames = []
            for line in lines[1:]:
                frame_match = re.match(r'\s*at\s+([\w\.\$]+)\((.*?)\)', line)
                if frame_match:
                    method = frame_match.group(1)
                    location = frame_match.group(2)
                    stack_frames.append({
                        'method': method,
                        'location': location,
                        'raw_line': line.strip()
                    })
            
            # Find caused by
            caused_by = []
            for line in lines:
                if 'caused by' in line.lower():
                    caused_by.append(line.strip())
            
            return {
                'exception_type': exception_type,
                'exception_message': exception_message,
                'stack_frames': stack_frames,
                'caused_by': caused_by,
                'total_frames': len(stack_frames)
            }
            
        except Exception:
            return None
    
    def _extract_sql_data(self, content: str) -> Optional[Dict]:
        """Extract structured data from SQL queries"""
        try:
            # Clean up the SQL
            sql_content = re.sub(r'^\d{2}:\d{2}:\d{2}\.\d{3}.*?(?=SELECT|INSERT|UPDATE|DELETE)', '', content, flags=re.IGNORECASE | re.MULTILINE)
            sql_content = sql_content.strip()
            
            # Extract SQL components
            query_type = None
            tables = []
            
            # Detect query type
            if re.search(r'\bSELECT\b', sql_content, re.IGNORECASE):
                query_type = 'SELECT'
            elif re.search(r'\bINSERT\b', sql_content, re.IGNORECASE):
                query_type = 'INSERT'
            elif re.search(r'\bUPDATE\b', sql_content, re.IGNORECASE):
                query_type = 'UPDATE'
            elif re.search(r'\bDELETE\b', sql_content, re.IGNORECASE):
                query_type = 'DELETE'
            
            # Extract table names
            table_matches = re.findall(r'\bFROM\s+(\w+)', sql_content, re.IGNORECASE)
            tables.extend(table_matches)
            
            table_matches = re.findall(r'\bINTO\s+(\w+)', sql_content, re.IGNORECASE)
            tables.extend(table_matches)
            
            table_matches = re.findall(r'\bUPDATE\s+(\w+)', sql_content, re.IGNORECASE)
            tables.extend(table_matches)
            
            return {
                'query_type': query_type,
                'tables': list(set(tables)),
                'query_length': len(sql_content),
                'has_where_clause': bool(re.search(r'\bWHERE\b', sql_content, re.IGNORECASE)),
                'has_join': bool(re.search(r'\bJOIN\b', sql_content, re.IGNORECASE)),
            }
            
        except Exception:
            return None
    
    def _determine_final_entry_type(self, initial_type: str, parsed_content: ParsedContent) -> LogEntryType:
        """Determine final entry type based on parsing results"""
        
        if parsed_content and parsed_content.structured_data:
            return parsed_content.content_type
        
        type_mapping = {
            'stack_trace': LogEntryType.STACK_TRACE,
            'json_start': LogEntryType.JSON_DATA,
            'xml_start': LogEntryType.XML_DATA,
            'sql_start': LogEntryType.SQL_QUERY,
            'config_start': LogEntryType.CONFIGURATION,
        }
        
        return type_mapping.get(initial_type, LogEntryType.MULTI_LINE)
    
    def _calculate_parsing_confidence(self, parsed_content: Optional[ParsedContent], line_count: int) -> float:
        """Calculate confidence in parsing accuracy"""
        
        base_confidence = 0.7
        
        if parsed_content:
            # Boost confidence if structured data was extracted
            if parsed_content.structured_data:
                base_confidence += 0.2
            
            # Reduce confidence if there were parsing errors
            if parsed_content.parsing_errors:
                base_confidence -= 0.1 * len(parsed_content.parsing_errors)
        
        # Adjust based on line count (more lines = more complex = potentially lower confidence)
        if line_count > 10:
            base_confidence -= 0.1
        elif line_count > 50:
            base_confidence -= 0.2
        
        return max(0.1, min(1.0, base_confidence))
    
    def _append_to_multiline_entry(self, entry: ProductionLogEntry, line: str, line_number: int):
        """Append a continuation line to an existing multi-line entry"""
        entry.raw_lines.append(line)
        entry.line_numbers.append(line_number)
        entry.event += '\n' + line
        
        # Re-parse content if it was structured
        if entry.parsed_content:
            combined_content = '\n'.join(entry.raw_lines)
            entry.parsed_content = self._parse_structured_content(
                combined_content, entry.entry_type.value
            )
    
    def _create_orphaned_entry(self, line: str, line_number: int, file_position: int) -> ProductionLogEntry:
        """Create entry for orphaned lines"""
        return ProductionLogEntry(
            timestamp="Unknown",
            subsystem="Unknown",
            event=line,
            raw_lines=[line],
            line_numbers=[line_number],
            entry_type=LogEntryType.SINGLE_LINE,
            parsed_content=None,
            file_position=file_position,
            parsing_confidence=0.3,
            entry_hash=self._generate_entry_hash(line, line_number)
        )
    
    def _post_process_entries(self, entries: List[ProductionLogEntry]):
        """Post-process entries for improved accuracy"""
        
        for i, entry in enumerate(entries):
            # Improve confidence based on context
            if i > 0:
                prev_entry = entries[i-1]
                
                # If previous entry was also unknown, this might be part of it
                if (prev_entry.subsystem == "Unknown" and entry.subsystem == "Unknown" and
                    entry.parsing_confidence < 0.5):
                    
                    # Check if this should be merged
                    if (not self._looks_like_new_entry(entry.event) and 
                        len(prev_entry.raw_lines) < 20):  # Don't merge into huge entries
                        
                        # Merge into previous entry
                        prev_entry.raw_lines.extend(entry.raw_lines)
                        prev_entry.line_numbers.extend(entry.line_numbers)
                        prev_entry.event += '\n' + entry.event
                        prev_entry.entry_type = LogEntryType.MULTI_LINE
                        
                        # Mark this entry for removal
                        entry.parsing_confidence = 0.0
            
            # Final confidence adjustment
            if entry.entry_type != LogEntryType.SINGLE_LINE and len(entry.raw_lines) > 1:
                entry.parsing_confidence = min(entry.parsing_confidence + 0.1, 1.0)
        
        # Remove entries marked for removal (confidence = 0)
        entries[:] = [e for e in entries if e.parsing_confidence > 0]
    
    def _looks_like_new_entry(self, text: str) -> bool:
        """Check if text looks like the start of a new log entry"""
        # Check for timestamp patterns
        for pattern in self.timestamp_patterns:
            if re.search(pattern, text):
                return True
        
        # Check for common log entry starters
        starters = ['INFO', 'ERROR', 'WARN', 'DEBUG', 'TRACE', '[', 'Starting', 'Stopping']
        for starter in starters:
            if text.strip().startswith(starter):
                return True
        
        return False
    
    def _generate_entry_hash(self, content: str, line_number: int) -> str:
        """Generate unique hash for entry tracking"""
        hash_content = f"{line_number}:{content[:100]}"  # Use first 100 chars to avoid huge hashes
        return hashlib.md5(hash_content.encode()).hexdigest()[:16]
    
    def _create_parsing_metrics(self, file_path: str, content: str, entries: List[ProductionLogEntry],
                              parsing_time: float, detected_format: str, encoding_used: str) -> ParsingMetrics:
        """Create comprehensive parsing metrics"""
        
        lines = content.split('\n')
        
        # Count different entry types
        multiline_count = len([e for e in entries if e.entry_type != LogEntryType.SINGLE_LINE])
        stack_trace_count = len([e for e in entries if e.entry_type == LogEntryType.STACK_TRACE])
        json_count = len([e for e in entries if e.entry_type == LogEntryType.JSON_DATA])
        xml_count = len([e for e in entries if e.entry_type == LogEntryType.XML_DATA])
        
        # Calculate average confidence
        confidences = [e.parsing_confidence for e in entries]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Collect errors
        errors = []
        for entry in entries:
            if entry.parsed_content and entry.parsed_content.parsing_errors:
                errors.extend(entry.parsed_content.parsing_errors)
        
        return ParsingMetrics(
            total_lines_processed=len(lines),
            successful_parses=len(entries),
            multi_line_entries=multiline_count,
            stack_traces_found=stack_trace_count,
            json_blocks_found=json_count,
            xml_blocks_found=xml_count,
            parsing_time=parsing_time,
            average_confidence=avg_confidence,
            encoding_used=encoding_used,
            format_detected=detected_format,
            errors_encountered=errors
        )
    
    def _create_error_metrics(self, file_path: str, error_msg: str) -> ParsingMetrics:
        """Create error metrics for failed parsing"""
        return ParsingMetrics(
            total_lines_processed=0,
            successful_parses=0,
            multi_line_entries=0,
            stack_traces_found=0,
            json_blocks_found=0,
            xml_blocks_found=0,
            parsing_time=0.0,
            average_confidence=0.0,
            encoding_used='failed',
            format_detected='unknown',
            errors_encountered=[error_msg]
        )
    
    def _store_parsing_metrics(self, file_path: str, metrics: ParsingMetrics):
        """Store parsing metrics in database"""
        
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO parsing_metrics 
                (file_path, file_hash, lines_processed, entries_found, multiline_entries, 
                 parsing_time, format_detected, encoding_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_path, file_hash, metrics.total_lines_processed,
                metrics.successful_parses, metrics.multi_line_entries,
                metrics.parsing_time, metrics.format_detected, metrics.encoding_used
            ))
            conn.commit()
    
    def _update_format_confidence(self, format_name: str, success: bool):
        """Update format detection confidence based on results"""
        with sqlite3.connect(self.db_path) as conn:
            # Get current stats
            cursor = conn.execute('''
                SELECT success_count, failure_count FROM format_patterns 
                WHERE pattern_name = ?
            ''', (format_name,))
            
            result = cursor.fetchone()
            
            if result:
                success_count, failure_count = result
                if success:
                    success_count += 1
                else:
                    failure_count += 1
            else:
                success_count = 1 if success else 0
                failure_count = 0 if success else 1
            
            # Calculate new confidence
            total = success_count + failure_count
            confidence = success_count / total if total > 0 else 0.5
            
            # Update database
            conn.execute('''
                INSERT OR REPLACE INTO format_patterns 
                (pattern_name, pattern_regex, success_count, failure_count, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (format_name, "", success_count, failure_count, confidence))
            
            conn.commit()
    
    def get_parsing_statistics(self) -> Dict:
        """Get comprehensive parsing statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as files_processed,
                    SUM(lines_processed) as total_lines,
                    SUM(entries_found) as total_entries,
                    SUM(multiline_entries) as total_multiline,
                    AVG(parsing_time) as avg_parsing_time,
                    AVG(entries_found * 1.0 / lines_processed) as avg_parse_rate
                FROM parsing_metrics
            ''')
            
            result = cursor.fetchone()
            
            # Get format distribution
            cursor2 = conn.execute('''
                SELECT format_detected, COUNT(*) as count
                FROM parsing_metrics
                GROUP BY format_detected
            ''')
            
            format_distribution = dict(cursor2.fetchall())
            
            return {
                'files_processed': result[0] if result else 0,
                'total_lines_processed': result[1] if result else 0,
                'total_entries_found': result[2] if result else 0,
                'total_multiline_entries': result[3] if result else 0,
                'average_parsing_time': result[4] if result else 0.0,
                'average_parse_rate': result[5] if result else 0.0,
                'format_distribution': format_distribution,
                'learning_active': True
            }
    
    def export_structured_data(self, entries: List[ProductionLogEntry], output_path: str, format_type: str = 'json'):
        """Export structured data from parsed entries"""
        
        structured_entries = []
        
        for entry in entries:
            entry_data = {
                'timestamp': entry.timestamp,
                'subsystem': entry.subsystem,
                'event': entry.event,
                'entry_type': entry.entry_type.value,
                'line_numbers': entry.line_numbers,
                'parsing_confidence': entry.parsing_confidence,
                'entry_hash': entry.entry_hash
            }
            
            if entry.parsed_content:
                entry_data['parsed_content'] = {
                    'content_type': entry.parsed_content.content_type.value,
                    'structured_data': entry.parsed_content.structured_data,
                    'metadata': entry.parsed_content.metadata,
                    'parsing_errors': entry.parsed_content.parsing_errors
                }
            
            structured_entries.append(entry_data)
        
        if format_type.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_entries, f, indent=2, default=str)
        
        elif format_type.lower() == 'csv':
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'timestamp', 'subsystem', 'event_preview', 'entry_type',
                    'line_count', 'parsing_confidence', 'has_structured_data'
                ])
                
                # Data
                for entry_data in structured_entries:
                    writer.writerow([
                        entry_data['timestamp'],
                        entry_data['subsystem'],
                        entry_data['event'][:100] + '...' if len(entry_data['event']) > 100 else entry_data['event'],
                        entry_data['entry_type'],
                        len(entry_data['line_numbers']),
                        entry_data['parsing_confidence'],
                        'Yes' if 'parsed_content' in entry_data else 'No'
                    ])
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


# Compatibility function for existing code
def convert_to_legacy_format(production_entries: List[ProductionLogEntry]) -> List:
    """Convert production entries to legacy format for compatibility"""
    from parser import LogEntry
    
    legacy_entries = []
    
    for entry in production_entries:
        # Create legacy LogEntry
        legacy_entry = LogEntry(
            timestamp=entry.timestamp,
            subsystem=entry.subsystem,
            event=entry.event,
            raw_line=entry.raw_lines[0] if entry.raw_lines else entry.event,
            line_number=entry.line_numbers[0] if entry.line_numbers else 0
        )
        
        legacy_entries.append(legacy_entry)
    
    return legacy_entries 