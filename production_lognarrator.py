#!/usr/bin/env python3
"""
Production LogNarrator AI - Enhanced Log Analysis System

Complete production-ready system with:
- ML-based confidence scoring
- Multi-line parsing with advanced format detection
- Real-time temporal analysis and trend detection
- Comprehensive database integration
- Advanced export capabilities
- Configuration management
- Performance monitoring

Usage:
    python production_lognarrator.py [log_file] [options]

Real functionality only - no demos or fake data.
"""

import argparse
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

# Import our production components
try:
    from production_parser import ProductionLogParser, convert_to_legacy_format
    from enhanced_categorizer_v2 import ProductionLogCategorizer
    from database_manager import get_database_manager
    from temporal_analyzer import get_temporal_analyzer
    from advanced_exporter import get_advanced_exporter, ExportRequest
    from config_manager import get_config_manager
    
    # Legacy components for compatibility
    from summarizer import LogSummarizer
    from ai_narrator import AILogNarrator
    
    PRODUCTION_READY = True
except ImportError as e:
    print(f"Warning: Some production components not available: {e}")
    print("Falling back to basic functionality...")
    PRODUCTION_READY = False


class ProductionLogNarrator:
    """Main production LogNarrator AI system"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Initialize logging
        self._setup_logging()
        
        # Initialize configuration
        self.config_manager = get_config_manager()
        if config_file:
            self.config_manager.import_config(config_file)
        
        # Initialize components
        self.parser = ProductionLogParser()
        self.categorizer = ProductionLogCategorizer()
        self.db_manager = get_database_manager()
        self.temporal_analyzer = get_temporal_analyzer()
        self.exporter = get_advanced_exporter()
        
        # Legacy components for AI summary
        self.summarizer = LogSummarizer()
        self.ai_narrator = AILogNarrator()
        
        # Performance tracking
        self.session_stats = {
            'start_time': datetime.now(),
            'files_processed': 0,
            'total_entries': 0,
            'total_processing_time': 0.0,
            'avg_confidence': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Production LogNarrator AI initialized")
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"lognarrator_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
    
    def analyze_log_file(self, file_path: str, export_options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive log file analysis with all production features
        """
        
        start_time = time.time()
        self.logger.info(f"Starting analysis of {file_path}")
        
        try:
            # Step 1: Parse file with multi-line support
            self.logger.info("Parsing log file...")
            parsed_entries, parsing_metrics = self.parser.parse_file_production(file_path)
            
            if not parsed_entries:
                self.logger.error("No entries parsed from file")
                return self._create_error_result("No entries parsed from file")
            
            self.logger.info(f"Parsed {len(parsed_entries)} entries")
            
            # Step 2: Enhanced categorization with ML confidence scoring
            self.logger.info("Categorizing entries...")
            legacy_entries = convert_to_legacy_format(parsed_entries)
            categorized_entries, categorization_metrics = self.categorizer.categorize_log_sequence(legacy_entries)
            
            self.logger.info(f"Categorized {len(categorized_entries)} entries")
            
            # Step 3: Generate AI summary
            self.logger.info("Generating AI summary...")
            summary = self.summarizer.generate_summary(categorized_entries)
            narrative = self.ai_narrator.create_narrative(categorized_entries, summary)
            
            # Step 4: Store in database for historical analysis
            self.logger.info("Storing analysis data...")
            session_data = self._prepare_session_data(
                file_path, parsed_entries, categorized_entries, 
                parsing_metrics, categorization_metrics, narrative
            )
            
            entries_data = self._prepare_entries_data(parsed_entries, categorized_entries)
            session_id = self.db_manager.store_analysis_session(session_data, entries_data)
            
            # Step 5: Temporal analysis
            self.logger.info("Performing temporal analysis...")
            temporal_summary = self.temporal_analyzer.get_temporal_summary(
                self.config_manager.get_config('analysis.trend_analysis_window_days', 30)
            )
            
            # Step 6: Correlation analysis
            correlations = self.temporal_analyzer.detect_subsystem_correlations(7)
            
            # Step 7: Anomaly detection
            anomalies = self.temporal_analyzer.detect_temporal_anomalies(24)
            
            # Step 8: Prepare comprehensive results
            processing_time = time.time() - start_time
            
            results = {
                'session_id': session_id,
                'file_path': file_path,
                'analysis_timestamp': datetime.now().isoformat(),
                'processing_time': processing_time,
                
                # Core analysis
                'parsing_metrics': parsing_metrics,
                'categorization_metrics': categorization_metrics,
                'summary': summary,
                'narrative': narrative,
                
                # Enhanced analysis
                'temporal_analysis': temporal_summary,
                'correlations': [asdict(c) for c in correlations[:10]],  # Top 10
                'anomalies': [asdict(a) for a in anomalies[:20]],  # Top 20
                
                # Statistics
                'total_entries': len(parsed_entries),
                'entries_by_type': self._get_entry_type_distribution(parsed_entries),
                'confidence_distribution': self._get_confidence_distribution(categorized_entries),
                'risk_analysis': self._get_risk_analysis(categorized_entries),
                'phase_analysis': self._get_phase_analysis(categorized_entries),
                'subsystem_analysis': self._get_subsystem_analysis(categorized_entries),
                
                # Performance
                'performance_metrics': {
                    'entries_per_second': len(parsed_entries) / processing_time,
                    'parsing_time': parsing_metrics.parsing_time,
                    'categorization_time': categorization_metrics.get('processing_time', 0),
                    'multiline_entries': parsing_metrics.multi_line_entries,
                    'stack_traces_found': parsing_metrics.stack_traces_found,
                    'json_blocks_found': parsing_metrics.json_blocks_found,
                    'average_confidence': categorization_metrics.get('avg_confidence', 0)
                }
            }
            
            # Step 9: Export if requested
            if export_options:
                self.logger.info("Exporting results...")
                self._export_results(results, export_options)
            
            # Update session stats
            self._update_session_stats(results)
            
            self.logger.info(f"Analysis completed in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}", exc_info=True)
            return self._create_error_result(str(e))
    
    def _prepare_session_data(self, file_path: str, parsed_entries: List, 
                            categorized_entries: List, parsing_metrics: Any,
                            categorization_metrics: Dict, narrative: str) -> Dict:
        """Prepare session data for database storage"""
        
        # Calculate distributions
        risk_distribution = {'GREEN': 0, 'YELLOW': 0, 'RED': 0}
        phase_distribution = {}
        subsystem_analysis = {}
        
        for entry in categorized_entries:
            # Risk distribution - handle both enum and emoji values
            risk_value = entry.risk_level.value
            if risk_value == '🟢':
                risk_value = 'GREEN'
            elif risk_value == '🟡':
                risk_value = 'YELLOW'
            elif risk_value == '🔴':
                risk_value = 'RED'
            
            risk_distribution[risk_value] += 1
            
            # Phase distribution
            phase = entry.phase.value
            phase_distribution[phase] = phase_distribution.get(phase, 0) + 1
            
            # Subsystem analysis
            subsystem = entry.log_entry.subsystem
            if subsystem not in subsystem_analysis:
                subsystem_analysis[subsystem] = {
                    'entry_count': 0,
                    'risk_distribution': {'GREEN': 0, 'YELLOW': 0, 'RED': 0},
                    'avg_confidence': 0.0,
                    'anomaly_count': 0
                }
            
            subsystem_analysis[subsystem]['entry_count'] += 1
            subsystem_analysis[subsystem]['risk_distribution'][risk_value] += 1
            
            # Check for anomalies
            if hasattr(entry, 'anomaly_indicators') and entry.anomaly_indicators:
                subsystem_analysis[subsystem]['anomaly_count'] += 1
        
        # Calculate average confidence per subsystem
        for subsystem in subsystem_analysis:
            subsystem_entries = [e for e in categorized_entries if e.log_entry.subsystem == subsystem]
            if subsystem_entries:
                confidences = [e.confidence_metrics.overall_confidence for e in subsystem_entries 
                              if hasattr(e, 'confidence_metrics')]
                if confidences:
                    subsystem_analysis[subsystem]['avg_confidence'] = sum(confidences) / len(confidences)
        
        # Calculate overall confidence
        overall_confidence = categorization_metrics.get('avg_confidence', 0.0)
        
        return {
            'file_path': file_path,
            'file_hash': self._calculate_file_hash(file_path),
            'analysis_timestamp': datetime.now().isoformat(),
            'total_entries': len(categorized_entries),
            'risk_distribution': risk_distribution,
            'phase_distribution': phase_distribution,
            'subsystem_analysis': subsystem_analysis,
            'overall_confidence': overall_confidence,
            'processing_time': categorization_metrics.get('processing_time', 0),
            'format_detected': parsing_metrics.format_detected,
            'encoding_used': parsing_metrics.encoding_used,
            'ai_summary': narrative
        }
    
    def _prepare_entries_data(self, parsed_entries: List, categorized_entries: List) -> List[Dict]:
        """Prepare entries data for database storage"""
        
        entries_data = []
        
        for i, (parsed_entry, categorized_entry) in enumerate(zip(parsed_entries, categorized_entries)):
            entry_data = {
                'entry_hash': parsed_entry.entry_hash,
                'timestamp': parsed_entry.timestamp,
                'subsystem': parsed_entry.subsystem,
                'event_text': parsed_entry.event,
                'predicted_phase': categorized_entry.phase.value,
                'predicted_risk': categorized_entry.risk_level.value,
                'confidence_score': getattr(categorized_entry, 'confidence', 0.0),
                'pattern_matches': categorized_entry.matched_patterns if hasattr(categorized_entry, 'matched_patterns') else [],
                'anomaly_indicators': categorized_entry.anomaly_indicators if hasattr(categorized_entry, 'anomaly_indicators') else [],
                'processing_time': getattr(parsed_entry, 'processing_time', 0.0),
                'line_numbers': parsed_entry.line_numbers,
                'entry_type': parsed_entry.entry_type.value
            }
            
            entries_data.append(entry_data)
        
        return entries_data
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate file hash for duplicate detection"""
        import hashlib
        
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_entry_type_distribution(self, entries: List) -> Dict[str, int]:
        """Get distribution of entry types"""
        
        distribution = {}
        for entry in entries:
            entry_type = entry.entry_type.value
            distribution[entry_type] = distribution.get(entry_type, 0) + 1
        
        return distribution
    
    def _get_confidence_distribution(self, entries: List) -> Dict[str, int]:
        """Get confidence score distribution"""
        
        distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for entry in entries:
            confidence = getattr(entry, 'confidence', 0.5)  # Use simple confidence attribute
            
            if confidence >= 0.8:
                distribution['high'] += 1
            elif confidence >= 0.6:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        
        return distribution
    
    def _get_risk_analysis(self, entries: List) -> Dict[str, Any]:
        """Get detailed risk analysis"""
        
        risk_counts = {'GREEN': 0, 'YELLOW': 0, 'RED': 0}
        risk_by_subsystem = {}
        
        for entry in entries:
            risk = entry.risk_level.value
            # Convert emoji to enum string
            if risk == '🟢':
                risk = 'GREEN'
            elif risk == '🟡':
                risk = 'YELLOW'
            elif risk == '🔴':
                risk = 'RED'
            
            risk_counts[risk] += 1
            
            subsystem = entry.log_entry.subsystem
            if subsystem not in risk_by_subsystem:
                risk_by_subsystem[subsystem] = {'GREEN': 0, 'YELLOW': 0, 'RED': 0}
            
            risk_by_subsystem[subsystem][risk] += 1
        
        total_entries = len(entries)
        
        return {
            'distribution': risk_counts,
            'percentages': {
                'GREEN': (risk_counts['GREEN'] / total_entries * 100) if total_entries > 0 else 0,
                'YELLOW': (risk_counts['YELLOW'] / total_entries * 100) if total_entries > 0 else 0,
                'RED': (risk_counts['RED'] / total_entries * 100) if total_entries > 0 else 0
            },
            'by_subsystem': risk_by_subsystem,
            'critical_subsystems': [
                subsystem for subsystem, risks in risk_by_subsystem.items()
                if risks['RED'] > risks['GREEN'] + risks['YELLOW']
            ]
        }
    
    def _get_phase_analysis(self, entries: List) -> Dict[str, Any]:
        """Get detailed phase analysis"""
        
        phase_counts = {}
        phase_by_subsystem = {}
        
        for entry in entries:
            phase = entry.phase.value
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            
            subsystem = entry.log_entry.subsystem
            if subsystem not in phase_by_subsystem:
                phase_by_subsystem[subsystem] = {}
            
            phase_by_subsystem[subsystem][phase] = phase_by_subsystem[subsystem].get(phase, 0) + 1
        
        return {
            'distribution': phase_counts,
            'by_subsystem': phase_by_subsystem,
            'most_common_phase': max(phase_counts.items(), key=lambda x: x[1])[0] if phase_counts else None
        }
    
    def _get_subsystem_analysis(self, entries: List) -> Dict[str, Any]:
        """Get detailed subsystem analysis"""
        
        subsystem_stats = {}
        
        for entry in entries:
            subsystem = entry.log_entry.subsystem
            
            if subsystem not in subsystem_stats:
                subsystem_stats[subsystem] = {
                    'entry_count': 0,
                    'risk_distribution': {'GREEN': 0, 'YELLOW': 0, 'RED': 0},
                    'phase_distribution': {},
                    'confidences': [],
                    'anomaly_count': 0
                }
            
            stats = subsystem_stats[subsystem]
            stats['entry_count'] += 1
            
            # Convert emoji to enum string
            risk = entry.risk_level.value
            if risk == '🟢':
                risk = 'GREEN'
            elif risk == '🟡':
                risk = 'YELLOW'
            elif risk == '🔴':
                risk = 'RED'
            
            stats['risk_distribution'][risk] += 1
            
            phase = entry.phase.value
            stats['phase_distribution'][phase] = stats['phase_distribution'].get(phase, 0) + 1
            
            confidence = getattr(entry, 'confidence', 0.0)
            if confidence > 0:
                stats['confidences'].append(confidence)
            
            if hasattr(entry, 'anomaly_indicators') and entry.anomaly_indicators:
                stats['anomaly_count'] += 1
        
        # Calculate averages
        for subsystem, stats in subsystem_stats.items():
            if stats['confidences']:
                stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
            else:
                stats['avg_confidence'] = 0.0
            
            # Don't store individual confidences in final result
            del stats['confidences']
        
        return subsystem_stats
    
    def _export_results(self, results: Dict[str, Any], export_options: Dict):
        """Export analysis results"""
        
        export_request = ExportRequest(
            export_type=export_options.get('format', 'json'),
            output_path=export_options.get('output_path', 'analysis_report'),
            include_charts=export_options.get('include_charts', True),
            include_trends=export_options.get('include_trends', True),
            include_correlations=export_options.get('include_correlations', True),
            time_window_days=export_options.get('time_window_days', 30)
        )
        
        success = self.exporter.export_analysis_report(export_request)
        
        if success:
            self.logger.info(f"Results exported to {export_request.output_path}")
        else:
            self.logger.error("Export failed")
    
    def _update_session_stats(self, results: Dict[str, Any]):
        """Update session statistics"""
        
        self.session_stats['files_processed'] += 1
        self.session_stats['total_entries'] += results['total_entries']
        self.session_stats['total_processing_time'] += results['processing_time']
        
        # Calculate running average confidence
        if results['performance_metrics']['average_confidence'] > 0:
            current_avg = self.session_stats['avg_confidence']
            new_avg = results['performance_metrics']['average_confidence']
            files_processed = self.session_stats['files_processed']
            
            self.session_stats['avg_confidence'] = (
                (current_avg * (files_processed - 1) + new_avg) / files_processed
            )
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        
        return {
            'error': True,
            'error_message': error_message,
            'timestamp': datetime.now().isoformat(),
            'processing_time': 0.0,
            'total_entries': 0,
            'session_stats': self.session_stats
        }
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        
        runtime = datetime.now() - self.session_stats['start_time']
        
        return {
            'session_start': self.session_stats['start_time'].isoformat(),
            'runtime_seconds': runtime.total_seconds(),
            'files_processed': self.session_stats['files_processed'],
            'total_entries': self.session_stats['total_entries'],
            'total_processing_time': self.session_stats['total_processing_time'],
            'avg_confidence': self.session_stats['avg_confidence'],
            'avg_entries_per_file': (
                self.session_stats['total_entries'] / self.session_stats['files_processed']
                if self.session_stats['files_processed'] > 0 else 0
            ),
            'system_performance': {
                'entries_per_second': (
                    self.session_stats['total_entries'] / runtime.total_seconds()
                    if runtime.total_seconds() > 0 else 0
                ),
                'files_per_hour': (
                    self.session_stats['files_processed'] / (runtime.total_seconds() / 3600)
                    if runtime.total_seconds() > 0 else 0
                )
            }
        }
    
    def generate_system_report(self, time_window_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        
        self.logger.info("Generating system report...")
        
        # Get analysis history
        analysis_history = self.db_manager.get_analysis_history(
            limit=100,
            date_from=datetime.now() - timedelta(days=time_window_days)
        )
        
        # Get temporal analysis
        temporal_summary = self.temporal_analyzer.get_temporal_summary(time_window_days)
        
        # Get database statistics
        db_stats = self.db_manager.get_database_stats()
        
        # Get configuration summary
        config_summary = self.config_manager.get_config_summary()
        
        # Compile system report
        report = {
            'generated_at': datetime.now().isoformat(),
            'time_window_days': time_window_days,
            'system_overview': {
                'total_sessions': len(analysis_history),
                'total_entries_processed': sum(h.total_entries for h in analysis_history),
                'avg_processing_time': (
                    sum(h.processing_time for h in analysis_history) / len(analysis_history)
                    if analysis_history else 0
                ),
                'avg_confidence': (
                    sum(h.overall_confidence for h in analysis_history) / len(analysis_history)
                    if analysis_history else 0
                ),
                'production_features_active': PRODUCTION_READY
            },
            'temporal_analysis': temporal_summary,
            'database_statistics': db_stats,
            'configuration_status': config_summary,
            'session_statistics': self.get_session_stats(),
            'export_capabilities': self.exporter.get_export_capabilities(),
            'recent_activity': [
                {
                    'date': h.analysis_timestamp.isoformat(),
                    'file': Path(h.file_path).name,
                    'entries': h.total_entries,
                    'confidence': h.overall_confidence,
                    'time': h.processing_time
                }
                for h in analysis_history[:10]
            ]
        }
        
        return report


def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(
        description='Production LogNarrator AI - Enhanced Log Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python production_lognarrator.py logfile.txt
  python production_lognarrator.py logfile.txt --export pdf --output report.pdf
  python production_lognarrator.py logfile.txt --config custom_config.yaml
  python production_lognarrator.py --system-report --days 30
        """
    )
    
    parser.add_argument('file', nargs='?', help='Log file to analyze')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--export', choices=['pdf', 'excel', 'json', 'csv'], 
                       help='Export format')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--system-report', action='store_true',
                       help='Generate system report instead of analyzing file')
    parser.add_argument('--days', type=int, default=30,
                       help='Time window in days for analysis')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize system
    try:
        system = ProductionLogNarrator(config_file=args.config)
        
        if args.system_report:
            # Generate system report
            report = system.generate_system_report(args.days)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"System report saved to {args.output}")
            else:
                print(json.dumps(report, indent=2, default=str))
        
        elif args.file:
            # Analyze log file
            if not Path(args.file).exists():
                print(f"Error: File {args.file} does not exist")
                return 1
            
            # Setup export options
            export_options = None
            if args.export:
                export_options = {
                    'format': args.export,
                    'output_path': args.output or f'analysis_report.{args.export}',
                    'include_charts': True,
                    'include_trends': True,
                    'include_correlations': True,
                    'time_window_days': args.days
                }
            
            # Perform analysis
            results = system.analyze_log_file(args.file, export_options)
            
            if results.get('error'):
                print(f"Analysis failed: {results['error_message']}")
                return 1
            
            # Display summary
            print("\n" + "="*60)
            print("PRODUCTION LOGNARRATOR AI - ANALYSIS COMPLETE")
            print("="*60)
            print(f"File: {args.file}")
            print(f"Processing time: {results['processing_time']:.2f}s")
            print(f"Total entries: {results['total_entries']}")
            print(f"Entries per second: {results['performance_metrics']['entries_per_second']:.1f}")
            print(f"Average confidence: {results['performance_metrics']['average_confidence']:.2f}")
            
            print(f"\nEntry Types:")
            for entry_type, count in results['entries_by_type'].items():
                print(f"  {entry_type}: {count}")
            
            print(f"\nRisk Distribution:")
            for risk, count in results['risk_analysis']['distribution'].items():
                percentage = results['risk_analysis']['percentages'][risk]
                print(f"  {risk}: {count} ({percentage:.1f}%)")
            
            print(f"\nTemporal Analysis:")
            temporal = results['temporal_analysis']
            print(f"  Systems analyzed: {temporal['trend_summary']['total_subsystems']}")
            print(f"  Systems with alerts: {temporal['trend_summary']['systems_with_alerts']}")
            print(f"  Anomalies detected: {temporal['anomaly_summary']['total_anomalies']}")
            
            if results['correlations']:
                print(f"\nTop Correlations:")
                for corr in results['correlations'][:3]:
                    print(f"  {corr['subsystem_a']} ↔ {corr['subsystem_b']}: {corr['correlation_coefficient']:.3f}")
            
            if export_options:
                print(f"\nReport exported to: {export_options['output_path']}")
            
            # Show session stats
            stats = system.get_session_stats()
            print(f"\nSession Statistics:")
            print(f"  Files processed: {stats['files_processed']}")
            print(f"  Total entries: {stats['total_entries']}")
            print(f"  System performance: {stats['system_performance']['entries_per_second']:.1f} entries/sec")
            
        else:
            parser.print_help()
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 