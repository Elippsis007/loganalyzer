"""
Streamlit Web Interface for LogNarrator AI

This module provides a web-based interface for log analysis using Streamlit.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import os
from datetime import datetime

# Import our modules - try production first
try:
    from production_lognarrator import ProductionLogNarrator
    from database_manager import get_database_manager
    from config_manager import get_config_manager
    from temporal_analyzer import get_temporal_analyzer
    from advanced_exporter import get_advanced_exporter
    PRODUCTION_MODE = True
except ImportError:
    from parser import LogParser
    from categorizer import LogCategorizer, Phase, RiskLevel
    from summarizer import LogSummarizer
    PRODUCTION_MODE = False


def init_session_state():
    """Initialize session state variables"""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'log_data' not in st.session_state:
        st.session_state.log_data = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None


def create_risk_pie_chart(categorized_entries):
    """Create a pie chart showing risk distribution"""
    risk_counts = {"🟢 Green": 0, "🟡 Yellow": 0, "🔴 Red": 0}
    
    for entry in categorized_entries:
        if entry.risk_level == RiskLevel.GREEN:
            risk_counts["🟢 Green"] += 1
        elif entry.risk_level == RiskLevel.YELLOW:
            risk_counts["🟡 Yellow"] += 1
        elif entry.risk_level == RiskLevel.RED:
            risk_counts["🔴 Red"] += 1
    
    fig = px.pie(
        values=list(risk_counts.values()),
        names=list(risk_counts.keys()),
        title="Risk Level Distribution",
        color_discrete_map={
            "🟢 Green": "#2ecc71",
            "🟡 Yellow": "#f39c12",
            "🔴 Red": "#e74c3c"
        }
    )
    return fig


def create_phase_timeline(categorized_entries):
    """Create a timeline showing phases over time"""
    timeline_data = []
    
    for i, entry in enumerate(categorized_entries):
        timeline_data.append({
            "Index": i,
            "Time": entry.log_entry.timestamp,
            "Phase": entry.phase.value,
            "Risk": entry.risk_level.value,
            "System": entry.log_entry.subsystem,
            "Event": entry.log_entry.event
        })
    
    df = pd.DataFrame(timeline_data)
    
    # Create color mapping for phases
    phase_colors = {
        "Init": "#3498db",
        "Position": "#9b59b6",
        "Scan": "#2ecc71",
        "Save": "#1abc9c",
        "Error": "#e74c3c",
        "Recovery": "#f39c12",
        "Abort": "#95a5a6",
        "Unknown": "#bdc3c7"
    }
    
    fig = px.scatter(
        df,
        x="Index",
        y="Phase",
        color="Phase",
        hover_data=["Time", "System", "Event"],
        title="Phase Timeline",
        color_discrete_map=phase_colors
    )
    
    fig.update_layout(
        xaxis_title="Event Sequence",
        yaxis_title="Operational Phase"
    )
    
    return fig


def create_system_activity_chart(categorized_entries):
    """Create a chart showing activity by subsystem"""
    system_counts = {}
    
    for entry in categorized_entries:
        system = entry.log_entry.subsystem
        if system not in system_counts:
            system_counts[system] = {"Green": 0, "Yellow": 0, "Red": 0}
        
        if entry.risk_level == RiskLevel.GREEN:
            system_counts[system]["Green"] += 1
        elif entry.risk_level == RiskLevel.YELLOW:
            system_counts[system]["Yellow"] += 1
        elif entry.risk_level == RiskLevel.RED:
            system_counts[system]["Red"] += 1
    
    # Convert to DataFrame for plotting
    chart_data = []
    for system, counts in system_counts.items():
        for risk_level, count in counts.items():
            chart_data.append({
                "System": system,
                "Risk Level": risk_level,
                "Count": count
            })
    
    df = pd.DataFrame(chart_data)
    
    fig = px.bar(
        df,
        x="System",
        y="Count",
        color="Risk Level",
        title="Activity by Subsystem",
        color_discrete_map={
            "Green": "#2ecc71",
            "Yellow": "#f39c12",
            "Red": "#e74c3c"
        }
    )
    
    return fig


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="LogNarrator AI",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Header
    st.title("🤖 LogNarrator AI")
    if PRODUCTION_MODE:
        st.success("🚀 **Production System Active** - Enhanced capabilities enabled")
        st.subheader("Enterprise-grade log analysis with ML-powered insights")
    else:
        st.warning("⚠️ **Basic Mode** - Some features limited")
        st.subheader("Transform raw machine logs into readable summaries")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # API Key input - check environment first, then allow override
    env_api_key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    
    if env_api_key:
        st.sidebar.success("✅ API Key loaded from environment")
        api_key = env_api_key
    else:
        api_key = st.sidebar.text_input(
            "Anthropic API Key (optional)",
            type="password",
            help="Enter your Anthropic API key for enhanced AI analysis"
        )
        
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            st.sidebar.success("✅ API Key configured")
    
    # File upload or text input
    st.sidebar.header("Input Method")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload File", "Paste Text"]
    )
    
    log_text = ""
    
    if input_method == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload log file",
            type=['txt', 'log'],
            help="Upload a .txt or .log file"
        )
        
        if uploaded_file is not None:
            log_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            st.sidebar.success(f"✅ File loaded: {uploaded_file.name}")
    
    else:  # Paste Text
        log_text = st.sidebar.text_area(
            "Paste log text here:",
            height=200,
            placeholder="00:39:24.243 Save Engine Async Save Triggered\n00:39:24.267 AF System Retry #1 Triggered\n..."
        )
    
    # Analysis button
    if st.sidebar.button("🔍 Analyze Logs", type="primary"):
        if log_text.strip():
            with st.spinner("🤖 Running analysis..."):
                try:
                    if PRODUCTION_MODE:
                        # Use production system
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tmp_file:
                            tmp_file.write(log_text)
                            tmp_file_path = tmp_file.name
                        
                        try:
                            # Initialize production system with error handling
                            system = ProductionLogNarrator()
                            results = system.analyze_log_file(tmp_file_path)
                            
                            if not results.get('error'):
                                # Convert for compatibility with existing UI
                                st.session_state.log_data = results  # Store full results
                                st.session_state.summary = type('Summary', (), {
                                    'overall_status': '✅ Production Analysis Complete',
                                    'timeline_summary': results.get('narrative', 'Analysis completed successfully'),
                                    'key_events': [f"Processed {results.get('total_entries', 0)} entries in {results.get('processing_time', 0):.2f}s"],
                                    'risk_assessment': f"Risk Distribution: {results.get('risk_analysis', {}).get('distribution', {})}",
                                    'recommendations': [f"Processing speed: {results.get('performance_metrics', {}).get('entries_per_second', 0):.1f} entries/sec"],
                                    'technical_details': f"Database session: {results.get('session_id', 'N/A')}\nFormat detected: {results.get('parsing_metrics', {}).get('format_detected', 'Unknown')}"
                                })()
                                st.session_state.analysis_complete = True
                                st.sidebar.success("✅ Production analysis complete!")
                            else:
                                st.sidebar.error(f"❌ Analysis failed: {results.get('error_message')}")
                        except Exception as prod_error:
                            st.sidebar.error(f"❌ Production system error: {prod_error}")
                            # Fall back to basic mode
                            st.sidebar.info("🔄 Falling back to basic analysis...")
                            parser = LogParser()
                            categorizer = LogCategorizer()
                            summarizer = LogSummarizer(api_key if api_key else None)
                            
                            entries = parser.parse_text(log_text)
                            if entries:
                                categorized = categorizer.categorize_log_sequence(entries)
                                summary = summarizer.generate_summary(categorized)
                                
                                st.session_state.log_data = categorized
                                st.session_state.summary = summary
                                st.session_state.analysis_complete = True
                                st.sidebar.success("✅ Basic analysis complete!")
                        finally:
                            try:
                                os.unlink(tmp_file_path)
                            except:
                                pass
                    
                    else:
                        # Use basic system
                        parser = LogParser()
                        categorizer = LogCategorizer()
                        summarizer = LogSummarizer(api_key if api_key else None)
                        
                        entries = parser.parse_text(log_text)
                        
                        if entries:
                            categorized = categorizer.categorize_log_sequence(entries)
                            summary = summarizer.generate_summary(categorized)
                            
                            st.session_state.log_data = categorized
                            st.session_state.summary = summary
                            st.session_state.analysis_complete = True
                            st.sidebar.success("✅ Basic analysis complete!")
                        else:
                            st.sidebar.error("❌ No valid log entries found")
                
                except Exception as e:
                    st.sidebar.error(f"❌ Analysis error: {e}")
        else:
            st.sidebar.warning("⚠️ Please provide log text")
    
    # Main content area
    if st.session_state.analysis_complete:
        summary = st.session_state.summary
        categorized_entries = st.session_state.log_data
        
        # Overall status
        st.header("📊 Analysis Results")
        
        # Status indicator
        status_col1, status_col2 = st.columns([1, 3])
        with status_col1:
            st.metric("Overall Status", summary.overall_status)
        
        with status_col2:
            # Quick stats
            if PRODUCTION_MODE and isinstance(categorized_entries, dict):
                # Production mode with full results
                total_entries = categorized_entries.get('total_entries', 0)
                processing_time = categorized_entries.get('processing_time', 0)
                entries_per_sec = categorized_entries.get('performance_metrics', {}).get('entries_per_second', 0)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Events", total_entries)
                col2.metric("Processing Time", f"{processing_time:.2f}s")
                col3.metric("Speed", f"{entries_per_sec:.0f} entries/sec")
            else:
                # Basic mode
                total_entries = len(categorized_entries) if hasattr(categorized_entries, '__len__') else 0
                error_count = 0
                warning_count = 0
                
                if hasattr(categorized_entries, '__iter__'):
                    try:
                        error_count = sum(1 for e in categorized_entries if hasattr(e, 'risk_level') and e.risk_level == RiskLevel.RED)
                        warning_count = sum(1 for e in categorized_entries if hasattr(e, 'risk_level') and e.risk_level == RiskLevel.YELLOW)
                    except:
                        pass
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Events", total_entries)
                col2.metric("Errors", error_count)
                col3.metric("Warnings", warning_count)
        
        # Timeline Summary
        st.header("📖 Timeline Summary")
        st.info(summary.timeline_summary)
        
        # Key Events
        if summary.key_events:
            st.header("🔑 Key Events")
            for event in summary.key_events:
                st.write(f"• {event}")
        
        # Risk Assessment
        st.header("⚠️ Risk Assessment")
        if "HIGH RISK" in summary.risk_assessment:
            st.error(summary.risk_assessment)
        elif "MEDIUM RISK" in summary.risk_assessment:
            st.warning(summary.risk_assessment)
        else:
            st.success(summary.risk_assessment)
        
        # Recommendations
        if summary.recommendations:
            st.header("💡 Recommendations")
            for rec in summary.recommendations:
                st.write(f"• {rec}")
        
        # Technical Details
        with st.expander("🔧 Technical Details"):
            st.text(summary.technical_details)
        
        # Visualizations
        st.header("📈 Visualizations")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Risk Distribution", "Phase Timeline", "System Activity", "Detailed Table"])
        
        with tab1:
            st.plotly_chart(create_risk_pie_chart(categorized_entries), use_container_width=True)
        
        with tab2:
            st.plotly_chart(create_phase_timeline(categorized_entries), use_container_width=True)
        
        with tab3:
            st.plotly_chart(create_system_activity_chart(categorized_entries), use_container_width=True)
        
        with tab4:
            # Detailed table
            table_data = []
            for entry in categorized_entries:
                table_data.append({
                    "Time": entry.log_entry.timestamp,
                    "System": entry.log_entry.subsystem,
                    "Phase": entry.phase.value,
                    "Risk": entry.risk_level.value,
                    "Event": entry.log_entry.event,
                    "Explanation": entry.explanation
                })
            
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
        
        # Export options
        st.header("📥 Export")
        
        # Create downloadable content
        export_content = f"""=== LOG ANALYSIS SUMMARY ===
Overall Status: {summary.overall_status}

Timeline Summary:
{summary.timeline_summary}

Key Events:
{chr(10).join(summary.key_events)}

Risk Assessment:
{summary.risk_assessment}

Recommendations:
{chr(10).join(['• ' + rec for rec in summary.recommendations])}

Technical Details:
{summary.technical_details}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        st.download_button(
            label="📄 Download Summary",
            data=export_content,
            file_name=f"log_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    else:
        # Welcome screen
        st.header("Welcome to LogNarrator AI! 🚀")
        
        st.markdown("""
        **LogNarrator AI** transforms raw machine/tool logs into human-readable summaries using advanced AI analysis.
        
        ### Features:
        - 🔍 **Automatic Log Parsing** - Extracts timestamps, subsystems, and events
        - 📊 **Smart Categorization** - Classifies events into operational phases
        - ⚠️ **Risk Assessment** - Identifies critical issues and warnings
        - 🤖 **AI-Powered Summaries** - Generates plain-English explanations
        - 📈 **Interactive Visualizations** - Charts and graphs for better understanding
        
        ### How to Use:
        1. **Configure** your Anthropic API key in the sidebar (optional but recommended)
        2. **Upload** a log file or paste log text
        3. **Click** "Analyze Logs" to process your data
        4. **Explore** the results with interactive visualizations
        5. **Download** the summary report
        
        ### Example Log Format:
        ```
        00:39:24.243 Save Engine Async Save Triggered
        00:39:24.267 AF System Retry #1 Triggered
        00:39:26.214 SEM Image discarded
        ```
        """)
        
        # Sample data demo
        if st.button("🧪 Try with Sample Data"):
            sample_log = """
00:39:24.243 Save Engine Async Save Triggered  
00:39:24.267 AF System Retry #1 Triggered  
00:39:26.214 SEM Image discarded
00:39:27.001 AF System Retry #2 Triggered  
00:39:28.520 AF System Recovery Complete
00:39:30.100 Save Engine Save Operation Complete
00:39:31.200 Positioning Stage Move to coordinate 150,200
00:39:32.150 Positioning Stage Position reached successfully
00:39:33.000 SEM Electron beam scanning started
00:39:34.500 SEM Image captured successfully
00:39:35.100 Save Engine Image saved to database
"""
            
            with st.spinner("Processing sample data..."):
                # Initialize components
                parser = LogParser()
                categorizer = LogCategorizer()
                summarizer = LogSummarizer()
                
                # Parse logs
                entries = parser.parse_text(sample_log)
                categorized = categorizer.categorize_log_sequence(entries)
                summary = summarizer.generate_summary(categorized)
                
                # Store in session state
                st.session_state.log_data = categorized
                st.session_state.summary = summary
                st.session_state.analysis_complete = True
                
                st.rerun()


if __name__ == "__main__":
    main() 