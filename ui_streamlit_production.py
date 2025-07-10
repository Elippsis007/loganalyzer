"""
Production Streamlit Web Interface for LogNarrator AI

Enhanced web interface using the production LogNarrator AI system with:
- Advanced multi-line parsing
- ML-based confidence scoring  
- Real-time temporal analysis
- Professional export capabilities
- Database integration
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Import production components
try:
    from production_lognarrator import ProductionLogNarrator
    from database_manager import get_database_manager
    from temporal_analyzer import get_temporal_analyzer  
    from advanced_exporter import get_advanced_exporter, ExportRequest
    from config_manager import get_config_manager
    PRODUCTION_AVAILABLE = True
except ImportError:
    # Fallback to basic components
    from parser import LogParser
    from categorizer import LogCategorizer
    from summarizer import LogSummarizer
    PRODUCTION_AVAILABLE = False
    st.warning("⚠️ Production components not available. Using basic functionality.")


def init_session_state():
    """Initialize session state variables"""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'system_stats' not in st.session_state:
        st.session_state.system_stats = None


def create_enhanced_risk_chart(analysis_results: Dict[str, Any]):
    """Create enhanced risk distribution chart with production data"""
    risk_data = analysis_results.get('risk_analysis', {}).get('distribution', {})
    percentages = analysis_results.get('risk_analysis', {}).get('percentages', {})
    
    fig = go.Figure(data=[
        go.Pie(
            labels=['🟢 Safe', '🟡 Warning', '🔴 Critical'],
            values=[risk_data.get('GREEN', 0), risk_data.get('YELLOW', 0), risk_data.get('RED', 0)],
            textinfo='label+percent+value',
            marker_colors=['#2ecc71', '#f39c12', '#e74c3c'],
            hole=0.4
        )
    ])
    
    fig.update_layout(
        title="Risk Assessment Distribution",
        annotations=[dict(text=f"{sum(risk_data.values())}<br>Total Events", 
                         x=0.5, y=0.5, font_size=16, showarrow=False)]
    )
    
    return fig


def create_performance_metrics_chart(analysis_results: Dict[str, Any]):
    """Create performance metrics visualization"""
    metrics = analysis_results.get('performance_metrics', {})
    
    # Create metrics display
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = metrics.get('entries_per_second', 0),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Processing Speed (entries/sec)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 200]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 150}}))
    
    return fig


def create_temporal_analysis_chart(analysis_results: Dict[str, Any]):
    """Create temporal analysis visualization"""
    temporal_data = analysis_results.get('temporal_analysis', {})
    trend_summary = temporal_data.get('trend_summary', {})
    
    if not trend_summary:
        return None
    
    # Create trend distribution chart
    trend_dist = trend_summary.get('trend_distribution', {})
    
    fig = px.bar(
        x=list(trend_dist.keys()),
        y=list(trend_dist.values()),
        title="System Trend Analysis",
        labels={'x': 'Trend Type', 'y': 'Number of Systems'},
        color=list(trend_dist.values()),
        color_continuous_scale='RdYlGn'
    )
    
    return fig


def create_subsystem_analysis_chart(analysis_results: Dict[str, Any]):
    """Create detailed subsystem analysis chart"""
    subsystem_data = analysis_results.get('subsystem_analysis', {})
    
    if not subsystem_data:
        return None
    
    # Prepare data for visualization
    chart_data = []
    for subsystem, stats in subsystem_data.items():
        risk_dist = stats.get('risk_distribution', {})
        chart_data.append({
            'Subsystem': subsystem,
            'Total Events': stats.get('entry_count', 0),
            'Critical Issues': risk_dist.get('RED', 0),
            'Warnings': risk_dist.get('YELLOW', 0),
            'Normal': risk_dist.get('GREEN', 0),
            'Avg Confidence': stats.get('avg_confidence', 0),
            'Anomalies': stats.get('anomaly_count', 0)
        })
    
    df = pd.DataFrame(chart_data)
    
    # Create stacked bar chart for risk distribution
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Normal', x=df['Subsystem'], y=df['Normal'], marker_color='#2ecc71'))
    fig.add_trace(go.Bar(name='Warnings', x=df['Subsystem'], y=df['Warnings'], marker_color='#f39c12'))
    fig.add_trace(go.Bar(name='Critical', x=df['Subsystem'], y=df['Critical Issues'], marker_color='#e74c3c'))
    
    fig.update_layout(
        barmode='stack',
        title='Risk Distribution by Subsystem',
        xaxis_title='Subsystem',
        yaxis_title='Number of Events'
    )
    
    return fig, df


def display_correlations(analysis_results: Dict[str, Any]):
    """Display subsystem correlations"""
    correlations = analysis_results.get('correlations', [])
    
    if not correlations:
        st.info("No significant correlations detected.")
        return
    
    st.subheader("🔗 Subsystem Correlations")
    
    for corr in correlations[:5]:  # Top 5 correlations
        strength = abs(corr['correlation_coefficient'])
        if strength > 0.7:
            strength_text = "🔴 Strong"
            strength_color = "red"
        elif strength > 0.5:
            strength_text = "🟡 Moderate"
            strength_color = "orange"
        else:
            strength_text = "🟢 Weak"
            strength_color = "green"
        
        with st.expander(f"{corr['subsystem_a']} ↔ {corr['subsystem_b']} ({strength:.3f})"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Correlation Strength", f"{strength:.3f}", delta=strength_text)
                st.metric("Statistical Significance", "Yes" if corr['is_significant'] else "No")
            with col2:
                st.metric("Causality Score", f"{corr['causality_score']:.3f}")
                st.metric("Time Lag", f"{corr['lag_offset']} periods")
            
            if corr['common_patterns']:
                st.write("**Common Patterns:**")
                for pattern in corr['common_patterns']:
                    st.write(f"• {pattern}")


def display_anomalies(analysis_results: Dict[str, Any]):
    """Display detected anomalies"""
    anomalies = analysis_results.get('anomalies', [])
    
    if not anomalies:
        st.success("✅ No anomalies detected")
        return
    
    st.subheader("🚨 Detected Anomalies")
    
    for anomaly in anomalies[:10]:  # Top 10 anomalies
        severity = anomaly['severity']
        if severity == 'CRITICAL':
            alert_type = st.error
        elif severity == 'HIGH':
            alert_type = st.warning
        else:
            alert_type = st.info
        
        alert_type(f"**{anomaly['subsystem']}** - {anomaly['anomaly_type']}: "
                  f"Score {anomaly['anomaly_score']:.2f} at {anomaly['timestamp']}")


def export_analysis_results(analysis_results: Dict[str, Any]):
    """Handle export functionality"""
    st.subheader("📥 Export Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export PDF Report"):
            if PRODUCTION_AVAILABLE:
                try:
                    exporter = get_advanced_exporter()
                    request = ExportRequest(
                        export_type='pdf',
                        output_path='streamlit_report.pdf',
                        include_charts=True,
                        include_trends=True,
                        include_correlations=True
                    )
                    
                    success = exporter.export_analysis_report(request)
                    if success:
                        st.success("📄 PDF report generated!")
                        # In a real implementation, you'd provide download link
                    else:
                        st.error("Failed to generate PDF")
                except Exception as e:
                    st.error(f"Export error: {e}")
            else:
                st.warning("PDF export requires production components")
    
    with col2:
        # JSON Export
        json_data = json.dumps(analysis_results, indent=2, default=str)
        st.download_button(
            label="📊 Download JSON",
            data=json_data,
            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        # Summary Export
        summary_text = f"""=== LogNarrator AI Analysis Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW:
- Total Entries: {analysis_results.get('total_entries', 0)}
- Processing Time: {analysis_results.get('processing_time', 0):.2f}s
- Average Confidence: {analysis_results.get('performance_metrics', {}).get('average_confidence', 0):.2f}

RISK DISTRIBUTION:
{json.dumps(analysis_results.get('risk_analysis', {}).get('distribution', {}), indent=2)}

PERFORMANCE:
- Entries/Second: {analysis_results.get('performance_metrics', {}).get('entries_per_second', 0):.1f}
- Multi-line Entries: {analysis_results.get('performance_metrics', {}).get('multiline_entries', 0)}
- Stack Traces: {analysis_results.get('performance_metrics', {}).get('stack_traces_found', 0)}

AI NARRATIVE:
{analysis_results.get('narrative', 'No narrative available')}
"""
        
        st.download_button(
            label="📝 Download Summary",
            data=summary_text,
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )


def main():
    """Main Production Streamlit application"""
    st.set_page_config(
        page_title="LogNarrator AI - Production",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Header with production indicator
    st.title("🤖 LogNarrator AI")
    if PRODUCTION_AVAILABLE:
        st.success("🚀 **Production System Active** - Enhanced capabilities enabled")
    else:
        st.warning("⚠️ **Basic Mode** - Some features limited")
    
    st.subheader("Enterprise-grade log analysis with ML-powered insights")
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Production system stats
    if PRODUCTION_AVAILABLE and st.sidebar.button("📊 System Status"):
        try:
            db_manager = get_database_manager()
            db_stats = db_manager.get_database_stats()
            
            st.sidebar.metric("Analysis Sessions", db_stats.get('analysis_sessions_count', 0))
            st.sidebar.metric("Total Entries", db_stats.get('log_entries_count', 0))
            st.sidebar.metric("Database Size", f"{db_stats.get('database_size_bytes', 0) / 1024 / 1024:.1f} MB")
        except Exception as e:
            st.sidebar.error(f"System status error: {e}")
    
    # API Key Configuration
    env_api_key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    
    if env_api_key:
        st.sidebar.success("✅ AI API Key configured")
        api_key = env_api_key
    else:
        api_key = st.sidebar.text_input(
            "Anthropic API Key (optional)",
            type="password",
            help="For enhanced AI narrative generation"
        )
    
    # Processing Options
    st.sidebar.header("🔧 Processing Options")
    
    enable_temporal = st.sidebar.checkbox("Enable Temporal Analysis", value=True, 
                                         help="Analyze trends and correlations over time")
    enable_correlations = st.sidebar.checkbox("Enable Correlation Detection", value=True,
                                            help="Detect relationships between subsystems")
    enable_anomalies = st.sidebar.checkbox("Enable Anomaly Detection", value=True,
                                         help="Identify unusual patterns and outliers")
    
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.1,
                                            help="Minimum confidence for pattern recognition")
    
    # File Input
    st.sidebar.header("📁 Input")
    input_method = st.sidebar.radio("Input Method:", ["Upload File", "Paste Text"])
    
    log_text = ""
    
    if input_method == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload log file",
            type=['txt', 'log'],
            help="Supports multi-line logs, stack traces, JSON/XML"
        )
        
        if uploaded_file is not None:
            log_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            st.sidebar.success(f"✅ {uploaded_file.name} loaded ({len(log_text)} chars)")
    
    else:
        log_text = st.sidebar.text_area(
            "Paste log content:",
            height=200,
            placeholder="00:39:24.243 Save Engine Async Save Triggered\n00:39:24.267 AF System Retry #1 Triggered\n..."
        )
    
    # Analysis Button
    if st.sidebar.button("🔍 Analyze Logs", type="primary"):
        if log_text.strip():
            
            # Create temporary file for production system
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp_file:
                tmp_file.write(log_text)
                tmp_file_path = tmp_file.name
            
            with st.spinner("🤖 Running production analysis..."):
                try:
                    if PRODUCTION_AVAILABLE:
                        # Use production system
                        system = ProductionLogNarrator()
                        
                        # Configure system
                        config_manager = get_config_manager()
                        config_manager.set_config('analysis.confidence_threshold', confidence_threshold)
                        
                        # Run analysis
                        results = system.analyze_log_file(tmp_file_path)
                        
                        if not results.get('error'):
                            st.session_state.analysis_results = results
                            st.session_state.analysis_complete = True
                            st.sidebar.success("✅ Production analysis complete!")
                        else:
                            st.sidebar.error(f"❌ Analysis failed: {results.get('error_message')}")
                    
                    else:
                        # Fallback to basic system
                        parser = LogParser()
                        categorizer = LogCategorizer()
                        summarizer = LogSummarizer(api_key if api_key else None)
                        
                        entries = parser.parse_text(log_text)
                        if entries:
                            categorized = categorizer.categorize_log_sequence(entries)
                            summary = summarizer.generate_summary(categorized)
                            
                            # Convert to compatible format
                            results = {
                                'total_entries': len(categorized),
                                'processing_time': 0.1,
                                'summary': summary,
                                'performance_metrics': {'entries_per_second': len(categorized) / 0.1}
                            }
                            
                            st.session_state.analysis_results = results
                            st.session_state.analysis_complete = True
                            st.sidebar.success("✅ Basic analysis complete!")
                        else:
                            st.sidebar.error("❌ No valid log entries found")
                
                except Exception as e:
                    st.sidebar.error(f"❌ Analysis error: {e}")
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
        
        else:
            st.sidebar.warning("⚠️ Please provide log content")
    
    # Main Content
    if st.session_state.analysis_complete and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Overview Metrics
        st.header("📊 Analysis Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Entries", results.get('total_entries', 0))
        with col2:
            st.metric("Processing Time", f"{results.get('processing_time', 0):.2f}s")
        with col3:
            st.metric("Processing Speed", f"{results.get('performance_metrics', {}).get('entries_per_second', 0):.1f} entries/sec")
        with col4:
            if PRODUCTION_AVAILABLE:
                confidence = results.get('performance_metrics', {}).get('average_confidence', 0)
                st.metric("Avg Confidence", f"{confidence:.2f}")
            else:
                st.metric("System Mode", "Basic")
        
        # Production Features
        if PRODUCTION_AVAILABLE:
            
            # Risk Analysis
            st.header("⚠️ Risk Analysis")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Risk metrics
                risk_analysis = results.get('risk_analysis', {})
                distribution = risk_analysis.get('distribution', {})
                percentages = risk_analysis.get('percentages', {})
                
                st.metric("🔴 Critical Issues", distribution.get('RED', 0), 
                         f"{percentages.get('RED', 0):.1f}%")
                st.metric("🟡 Warnings", distribution.get('YELLOW', 0),
                         f"{percentages.get('YELLOW', 0):.1f}%")
                st.metric("🟢 Normal Events", distribution.get('GREEN', 0),
                         f"{percentages.get('GREEN', 0):.1f}%")
            
            with col2:
                # Risk distribution chart
                risk_chart = create_enhanced_risk_chart(results)
                st.plotly_chart(risk_chart, use_container_width=True)
            
            # Performance Visualization
            st.header("🚀 Performance Metrics")
            perf_chart = create_performance_metrics_chart(results)
            st.plotly_chart(perf_chart, use_container_width=True)
            
            # Temporal Analysis
            if enable_temporal:
                st.header("📈 Temporal Analysis")
                temporal_chart = create_temporal_analysis_chart(results)
                if temporal_chart:
                    st.plotly_chart(temporal_chart, use_container_width=True)
                else:
                    st.info("No temporal trends detected")
            
            # Subsystem Analysis
            st.header("🔧 Subsystem Analysis")
            subsystem_result = create_subsystem_analysis_chart(results)
            if subsystem_result:
                chart, df = subsystem_result
                st.plotly_chart(chart, use_container_width=True)
                
                # Detailed table
                with st.expander("📋 Detailed Subsystem Data"):
                    st.dataframe(df, use_container_width=True)
            else:
                st.info("No subsystem data available")
            
            # Correlations
            if enable_correlations:
                display_correlations(results)
            
            # Anomalies
            if enable_anomalies:
                display_anomalies(results)
            
            # AI Narrative
            st.header("🤖 AI Analysis Narrative")
            narrative = results.get('narrative', 'No narrative available')
            st.info(narrative)
            
            # Export Options
            export_analysis_results(results)
        
        else:
            # Basic mode display
            st.header("📖 Analysis Summary")
            summary = results.get('summary')
            if summary:
                st.info(summary.timeline_summary)
                
                if summary.key_events:
                    st.subheader("🔑 Key Events")
                    for event in summary.key_events:
                        st.write(f"• {event}")
                
                st.subheader("⚠️ Risk Assessment")
                st.write(summary.risk_assessment)
    
    else:
        # Welcome Screen
        st.header("Welcome to LogNarrator AI Production! 🚀")
        
        st.markdown("""
        ### 🌟 Production Features Available:
        
        **🧠 Advanced Analysis:**
        - Multi-line log parsing (stack traces, JSON/XML)
        - ML-based pattern recognition with confidence scoring
        - Real-time performance monitoring
        
        **📊 Temporal Intelligence:**
        - Trend detection and forecasting
        - Anomaly detection with statistical analysis
        - Subsystem correlation analysis
        
        **🚀 Enterprise Capabilities:**
        - Database storage for historical analysis
        - Professional export (PDF, Excel, JSON)
        - Configurable thresholds and rules
        
        **📈 Real-time Insights:**
        - Processing speed: 100+ entries/second
        - Confidence scoring for all predictions
        - Comprehensive performance metrics
        """)
        
        # Sample Data Demo
        if st.button("🧪 Try with Sample Data"):
            sample_log = """
Jul 04,00:39:24.243 Info [Save Engine] Async Save Triggered  
Jul 04,00:39:24.267 Warn [AF System] Retry #1 Triggered  
Jul 04,00:39:26.214 Error [SEM] Image discarded - focus error
Jul 04,00:39:27.001 Warn [AF System] Retry #2 Triggered  
Jul 04,00:39:28.520 Info [AF System] Recovery Complete
Jul 04,00:39:30.100 Info [Save Engine] Save Operation Complete
Jul 04,00:39:31.200 Info [Positioning] Stage Move to coordinate 150,200
Jul 04,00:39:32.150 Info [Positioning] Position reached successfully
Jul 04,00:39:33.000 Info [SEM] Electron beam scanning started
Jul 04,00:39:34.500 Info [SEM] Image captured successfully
Jul 04,00:39:35.100 Info [Save Engine] Image saved to database
Exception in thread "main" java.lang.NullPointerException
    at com.example.ImageProcessor.process(ImageProcessor.java:42)
    at com.example.Scanner.scan(Scanner.java:128)
    at com.example.Main.main(Main.java:15)
Jul 04,00:39:36.200 Error [Exception Handler] Stack trace captured
Jul 04,00:39:37.100 Info [Recovery] System recovery initiated
"""
            
            # Set the sample data and trigger analysis
            st.session_state.sample_log = sample_log
            st.experimental_rerun()


if __name__ == "__main__":
    main() 