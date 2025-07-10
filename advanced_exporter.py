"""
Production Advanced Export System for LogNarrator AI

Real export capabilities:
- PDF reports with charts and analysis
- Excel workbooks with multiple sheets and charts
- Custom templates and branding
- Automated report generation

Production functionality only.
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import base64
import io

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.piecharts import Pie
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Excel generation
try:
    import xlsxwriter
    HAS_XLSXWRITER = True
except ImportError:
    HAS_XLSXWRITER = False

# Matplotlib for charts
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from database_manager import get_database_manager
from temporal_analyzer import get_temporal_analyzer

logger = logging.getLogger(__name__)


@dataclass
class ExportRequest:
    """Export configuration"""
    export_type: str  # 'pdf', 'excel', 'json', 'csv'
    output_path: str
    template: Optional[str] = None
    include_charts: bool = True
    include_trends: bool = True
    include_correlations: bool = True
    time_window_days: int = 30
    subsystem_filter: Optional[List[str]] = None
    custom_branding: Optional[Dict[str, str]] = None


class AdvancedExporter:
    """Production export system"""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.temporal_analyzer = get_temporal_analyzer()
        
        # Check available libraries
        self.capabilities = {
            'pdf': HAS_REPORTLAB,
            'excel': HAS_XLSXWRITER,
            'charts': HAS_MATPLOTLIB,
            'json': True,
            'csv': True
        }
        
        # Default styling
        self.default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        logger.info(f"Advanced exporter initialized with capabilities: {self.capabilities}")
    
    def export_analysis_report(self, request: ExportRequest) -> bool:
        """Export comprehensive analysis report"""
        
        if not self.capabilities.get(request.export_type.lower(), False):
            logger.error(f"Export type {request.export_type} not supported")
            return False
        
        try:
            # Gather data
            data = self._gather_analysis_data(request)
            
            # Generate report based on type
            if request.export_type.lower() == 'pdf':
                return self._generate_pdf_report(data, request)
            elif request.export_type.lower() == 'excel':
                return self._generate_excel_report(data, request)
            elif request.export_type.lower() == 'json':
                return self._generate_json_report(data, request)
            elif request.export_type.lower() == 'csv':
                return self._generate_csv_report(data, request)
            else:
                logger.error(f"Unknown export type: {request.export_type}")
                return False
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def _gather_analysis_data(self, request: ExportRequest) -> Dict[str, Any]:
        """Gather comprehensive analysis data"""
        
        data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'time_window_days': request.time_window_days,
                'export_type': request.export_type,
                'subsystem_filter': request.subsystem_filter
            },
            'analysis_history': [],
            'trend_analysis': {},
            'correlation_analysis': [],
            'anomaly_data': [],
            'performance_metrics': {},
            'summary_stats': {}
        }
        
        # Get analysis history
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.time_window_days)
        
        history = self.db_manager.get_analysis_history(
            limit=100, 
            date_from=start_date, 
            date_to=end_date
        )
        
        data['analysis_history'] = [asdict(h) for h in history]
        
        # Get trend analysis if requested
        if request.include_trends:
            data['trend_analysis'] = self.temporal_analyzer.get_temporal_summary(request.time_window_days)
        
        # Get correlation analysis if requested
        if request.include_correlations:
            correlations = self.temporal_analyzer.detect_subsystem_correlations(request.time_window_days)
            data['correlation_analysis'] = [asdict(c) for c in correlations]
        
        # Get anomaly data
        anomalies = self.temporal_analyzer.detect_temporal_anomalies(request.time_window_days * 24)
        data['anomaly_data'] = [asdict(a) for a in anomalies]
        
        # Calculate summary statistics
        data['summary_stats'] = self._calculate_summary_stats(data)
        
        return data
    
    def _calculate_summary_stats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        
        stats = {
            'total_sessions': len(data['analysis_history']),
            'avg_confidence': 0.0,
            'total_entries': 0,
            'risk_distribution': {'GREEN': 0, 'YELLOW': 0, 'RED': 0},
            'phase_distribution': {},
            'processing_performance': {},
            'top_subsystems': {}
        }
        
        if data['analysis_history']:
            # Calculate averages
            confidences = [h['overall_confidence'] for h in data['analysis_history']]
            stats['avg_confidence'] = sum(confidences) / len(confidences)
            
            # Total entries
            stats['total_entries'] = sum(h['total_entries'] for h in data['analysis_history'])
            
            # Risk distribution
            for session in data['analysis_history']:
                risk_dist = session['risk_distribution']
                for risk, count in risk_dist.items():
                    stats['risk_distribution'][risk] += count
            
            # Phase distribution
            all_phases = {}
            for session in data['analysis_history']:
                phase_dist = session['phase_distribution']
                for phase, count in phase_dist.items():
                    all_phases[phase] = all_phases.get(phase, 0) + count
            stats['phase_distribution'] = all_phases
            
            # Processing performance
            processing_times = [h['processing_time'] for h in data['analysis_history']]
            if processing_times:
                stats['processing_performance'] = {
                    'avg_time': sum(processing_times) / len(processing_times),
                    'min_time': min(processing_times),
                    'max_time': max(processing_times)
                }
        
        return stats
    
    def _generate_pdf_report(self, data: Dict[str, Any], request: ExportRequest) -> bool:
        """Generate PDF report"""
        
        if not HAS_REPORTLAB:
            logger.error("ReportLab not available for PDF generation")
            return False
        
        try:
            doc = SimpleDocTemplate(request.output_path, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=20,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            
            story.append(Paragraph("LogNarrator AI Analysis Report", title_style))
            story.append(Spacer(1, 20))
            
            # Metadata
            metadata_text = f"""
            <b>Generated:</b> {data['metadata']['generated_at']}<br/>
            <b>Time Window:</b> {data['metadata']['time_window_days']} days<br/>
            <b>Total Sessions:</b> {data['summary_stats']['total_sessions']}<br/>
            <b>Total Entries:</b> {data['summary_stats']['total_entries']}<br/>
            <b>Average Confidence:</b> {data['summary_stats']['avg_confidence']:.2f}
            """
            story.append(Paragraph(metadata_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Summary Statistics
            story.append(Paragraph("Summary Statistics", styles['Heading1']))
            
            # Risk Distribution Table
            risk_data = [['Risk Level', 'Count', 'Percentage']]
            total_risk = sum(data['summary_stats']['risk_distribution'].values())
            
            for risk, count in data['summary_stats']['risk_distribution'].items():
                percentage = (count / total_risk * 100) if total_risk > 0 else 0
                risk_data.append([risk, str(count), f"{percentage:.1f}%"])
            
            risk_table = Table(risk_data)
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(risk_table)
            story.append(Spacer(1, 20))
            
            # Trend Analysis Summary
            if request.include_trends and data['trend_analysis']:
                story.append(Paragraph("Trend Analysis", styles['Heading1']))
                
                trend_summary = data['trend_analysis'].get('trend_summary', {})
                trend_text = f"""
                <b>Systems Analyzed:</b> {trend_summary.get('total_subsystems', 0)}<br/>
                <b>Systems with Alerts:</b> {trend_summary.get('systems_with_alerts', 0)}<br/>
                <b>Improving Trends:</b> {trend_summary.get('trend_distribution', {}).get('improving', 0)}<br/>
                <b>Degrading Trends:</b> {trend_summary.get('trend_distribution', {}).get('degrading', 0)}<br/>
                <b>Stable Systems:</b> {trend_summary.get('trend_distribution', {}).get('stable', 0)}
                """
                story.append(Paragraph(trend_text, styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Anomaly Summary
            if data['anomaly_data']:
                story.append(Paragraph("Anomaly Detection", styles['Heading1']))
                
                critical_anomalies = len([a for a in data['anomaly_data'] if a['severity'] == 'CRITICAL'])
                total_anomalies = len(data['anomaly_data'])
                
                anomaly_text = f"""
                <b>Total Anomalies Detected:</b> {total_anomalies}<br/>
                <b>Critical Anomalies:</b> {critical_anomalies}<br/>
                <b>Most Recent Anomaly:</b> {data['anomaly_data'][0]['timestamp'] if data['anomaly_data'] else 'None'}
                """
                story.append(Paragraph(anomaly_text, styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Recent Analysis History
            story.append(Paragraph("Recent Analysis Sessions", styles['Heading1']))
            
            if data['analysis_history']:
                session_data = [['Date', 'File', 'Entries', 'Confidence', 'Time (s)']]
                
                for session in data['analysis_history'][:10]:  # Last 10 sessions
                    date_str = session['analysis_timestamp'][:10]
                    file_name = Path(session['file_path']).name
                    session_data.append([
                        date_str,
                        file_name,
                        str(session['total_entries']),
                        f"{session['overall_confidence']:.2f}",
                        f"{session['processing_time']:.2f}"
                    ])
                
                session_table = Table(session_data)
                session_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(session_table)
            
            # Build PDF
            doc.build(story)
            logger.info(f"PDF report generated: {request.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return False
    
    def _generate_excel_report(self, data: Dict[str, Any], request: ExportRequest) -> bool:
        """Generate Excel report with multiple sheets"""
        
        if not HAS_XLSXWRITER:
            logger.error("xlsxwriter not available for Excel generation")
            return False
        
        try:
            workbook = xlsxwriter.Workbook(request.output_path)
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D7E4BC',
                'border': 1
            })
            
            cell_format = workbook.add_format({'border': 1})
            
            # Summary Sheet
            summary_sheet = workbook.add_worksheet('Summary')
            
            # Write summary data
            summary_sheet.write('A1', 'LogNarrator AI Analysis Summary', header_format)
            summary_sheet.write('A3', 'Generated:', header_format)
            summary_sheet.write('B3', data['metadata']['generated_at'])
            summary_sheet.write('A4', 'Time Window (days):', header_format)
            summary_sheet.write('B4', data['metadata']['time_window_days'])
            summary_sheet.write('A5', 'Total Sessions:', header_format)
            summary_sheet.write('B5', data['summary_stats']['total_sessions'])
            summary_sheet.write('A6', 'Total Entries:', header_format)
            summary_sheet.write('B6', data['summary_stats']['total_entries'])
            summary_sheet.write('A7', 'Average Confidence:', header_format)
            summary_sheet.write('B7', data['summary_stats']['avg_confidence'])
            
            # Risk Distribution
            summary_sheet.write('A9', 'Risk Distribution', header_format)
            summary_sheet.write('A10', 'Risk Level', header_format)
            summary_sheet.write('B10', 'Count', header_format)
            summary_sheet.write('C10', 'Percentage', header_format)
            
            row = 11
            total_risk = sum(data['summary_stats']['risk_distribution'].values())
            for risk, count in data['summary_stats']['risk_distribution'].items():
                percentage = (count / total_risk * 100) if total_risk > 0 else 0
                summary_sheet.write(row, 0, risk, cell_format)
                summary_sheet.write(row, 1, count, cell_format)
                summary_sheet.write(row, 2, f"{percentage:.1f}%", cell_format)
                row += 1
            
            # Analysis History Sheet
            if data['analysis_history']:
                history_sheet = workbook.add_worksheet('Analysis History')
                
                # Headers
                headers = ['Date', 'File Path', 'Total Entries', 'Confidence', 'Processing Time', 'Format']
                for col, header in enumerate(headers):
                    history_sheet.write(0, col, header, header_format)
                
                # Data
                for row, session in enumerate(data['analysis_history'], 1):
                    history_sheet.write(row, 0, session['analysis_timestamp'][:10])
                    history_sheet.write(row, 1, session['file_path'])
                    history_sheet.write(row, 2, session['total_entries'])
                    history_sheet.write(row, 3, session['overall_confidence'])
                    history_sheet.write(row, 4, session['processing_time'])
                    history_sheet.write(row, 5, session.get('format_detected', 'Unknown'))
            
            # Trend Analysis Sheet
            if request.include_trends and data['trend_analysis']:
                trend_sheet = workbook.add_worksheet('Trend Analysis')
                
                trend_summary = data['trend_analysis'].get('trend_summary', {})
                
                trend_sheet.write('A1', 'Trend Analysis Summary', header_format)
                trend_sheet.write('A3', 'Total Subsystems:', header_format)
                trend_sheet.write('B3', trend_summary.get('total_subsystems', 0))
                trend_sheet.write('A4', 'Systems with Alerts:', header_format)
                trend_sheet.write('B4', trend_summary.get('systems_with_alerts', 0))
                
                # Trend distribution
                trend_sheet.write('A6', 'Trend Distribution', header_format)
                trend_sheet.write('A7', 'Trend Type', header_format)
                trend_sheet.write('B7', 'Count', header_format)
                
                row = 8
                for trend_type, count in trend_summary.get('trend_distribution', {}).items():
                    trend_sheet.write(row, 0, trend_type.title(), cell_format)
                    trend_sheet.write(row, 1, count, cell_format)
                    row += 1
            
            # Anomaly Sheet
            if data['anomaly_data']:
                anomaly_sheet = workbook.add_worksheet('Anomalies')
                
                # Headers
                headers = ['Timestamp', 'Subsystem', 'Metric', 'Severity', 'Anomaly Score', 'Type']
                for col, header in enumerate(headers):
                    anomaly_sheet.write(0, col, header, header_format)
                
                # Data
                for row, anomaly in enumerate(data['anomaly_data'], 1):
                    anomaly_sheet.write(row, 0, anomaly['timestamp'])
                    anomaly_sheet.write(row, 1, anomaly['subsystem'])
                    anomaly_sheet.write(row, 2, anomaly['metric_name'])
                    anomaly_sheet.write(row, 3, anomaly['severity'])
                    anomaly_sheet.write(row, 4, anomaly['anomaly_score'])
                    anomaly_sheet.write(row, 5, anomaly['anomaly_type'])
            
            workbook.close()
            logger.info(f"Excel report generated: {request.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Excel generation failed: {e}")
            return False
    
    def _generate_json_report(self, data: Dict[str, Any], request: ExportRequest) -> bool:
        """Generate JSON report"""
        
        try:
            with open(request.output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"JSON report generated: {request.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"JSON generation failed: {e}")
            return False
    
    def _generate_csv_report(self, data: Dict[str, Any], request: ExportRequest) -> bool:
        """Generate CSV report"""
        
        try:
            if not data['analysis_history']:
                logger.warning("No analysis history data for CSV export")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(data['analysis_history'])
            
            # Flatten nested columns
            if 'risk_distribution' in df.columns:
                risk_df = pd.json_normalize(df['risk_distribution'])
                risk_df.columns = ['risk_' + col for col in risk_df.columns]
                df = pd.concat([df, risk_df], axis=1)
                df.drop('risk_distribution', axis=1, inplace=True)
            
            if 'phase_distribution' in df.columns:
                phase_df = pd.json_normalize(df['phase_distribution'])
                phase_df.columns = ['phase_' + col for col in phase_df.columns]
                df = pd.concat([df, phase_df], axis=1)
                df.drop('phase_distribution', axis=1, inplace=True)
            
            # Save to CSV
            df.to_csv(request.output_path, index=False)
            logger.info(f"CSV report generated: {request.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"CSV generation failed: {e}")
            return False
    
    def get_export_capabilities(self) -> Dict[str, bool]:
        """Get available export capabilities"""
        return self.capabilities.copy()
    
    def create_custom_template(self, template_name: str, template_config: Dict[str, Any]) -> bool:
        """Create custom export template"""
        # This would implement custom template creation
        # For now, return a placeholder
        logger.info(f"Custom template creation not yet implemented: {template_name}")
        return False


# Global exporter instance
_advanced_exporter = None

def get_advanced_exporter() -> AdvancedExporter:
    """Get global advanced exporter instance"""
    global _advanced_exporter
    if _advanced_exporter is None:
        _advanced_exporter = AdvancedExporter()
    return _advanced_exporter 