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
    PRODUCTION_MODE = False

# Import educational system
try:
    from educational_system import (
        StoryNarrativeEngine, 
        ContextualEducationSystem, 
        PatternRecognitionTrainer,
        LearningLevel,
        StoryType
    )
    from investigation_methodology import (
        InvestigationMethodologyGuide,
        InvestigationFramework,
        InvestigationPhase
    )
    EDUCATIONAL_MODE = True
except ImportError:
    EDUCATIONAL_MODE = False

# Always import basic components for RiskLevel enum used in visualizations
try:
    from categorizer import Phase, RiskLevel
except ImportError:
    # Define fallback enums if basic imports fail
    from enum import Enum
    class Phase(Enum):
        UNKNOWN = "Unknown"
    class RiskLevel(Enum):
        GREEN = "🟢"
        YELLOW = "🟡"
        RED = "🔴"


def init_session_state():
    """Initialize session state variables"""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'log_data' not in st.session_state:
        st.session_state.log_data = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'educational_mode' not in st.session_state:
        st.session_state.educational_mode = False
    if 'learning_level' not in st.session_state:
        st.session_state.learning_level = 'beginner'
    if 'log_story' not in st.session_state:
        st.session_state.log_story = None
    if 'training_scenario' not in st.session_state:
        st.session_state.training_scenario = None


def create_risk_pie_chart(categorized_entries):
    """Create a pie chart showing risk distribution"""
    if not categorized_entries:
        # Return empty chart if no data
        fig = px.pie(values=[1], names=["No Data"], title="Risk Level Distribution")
        return fig
    
    risk_counts = {"🟢 Green": 0, "🟡 Yellow": 0, "🔴 Red": 0}
    
    try:
        for entry in categorized_entries:
            if hasattr(entry, 'risk_level'):
                if entry.risk_level == RiskLevel.GREEN:
                    risk_counts["🟢 Green"] += 1
                elif entry.risk_level == RiskLevel.YELLOW:
                    risk_counts["🟡 Yellow"] += 1
                elif entry.risk_level == RiskLevel.RED:
                    risk_counts["🔴 Red"] += 1
    except Exception as e:
        st.error(f"Error creating risk chart: {e}")
        return px.pie(values=[1], names=["Error"], title="Risk Level Distribution")
    
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
    if not categorized_entries:
        # Return empty chart if no data
        return px.scatter(title="Phase Timeline - No Data")
    
    timeline_data = []
    
    try:
        for i, entry in enumerate(categorized_entries):
            if hasattr(entry, 'log_entry') and hasattr(entry, 'phase') and hasattr(entry, 'risk_level'):
                timeline_data.append({
                    "Index": i,
                    "Time": entry.log_entry.timestamp,
                    "Phase": entry.phase.value,
                    "Risk": entry.risk_level.value,
                    "System": entry.log_entry.subsystem,
                    "Event": entry.log_entry.event
                })
    except Exception as e:
        st.error(f"Error creating timeline: {e}")
        return px.scatter(title="Phase Timeline - Error")
    
    if not timeline_data:
        return px.scatter(title="Phase Timeline - No Valid Data")
    
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
    if not categorized_entries:
        # Return empty chart if no data
        return px.bar(title="Activity by Subsystem - No Data")
    
    system_counts = {}
    
    try:
        for entry in categorized_entries:
            if hasattr(entry, 'log_entry') and hasattr(entry, 'risk_level'):
                system = entry.log_entry.subsystem
                if system not in system_counts:
                    system_counts[system] = {"Green": 0, "Yellow": 0, "Red": 0}
                
                if entry.risk_level == RiskLevel.GREEN:
                    system_counts[system]["Green"] += 1
                elif entry.risk_level == RiskLevel.YELLOW:
                    system_counts[system]["Yellow"] += 1
                elif entry.risk_level == RiskLevel.RED:
                    system_counts[system]["Red"] += 1
    except Exception as e:
        st.error(f"Error creating system activity chart: {e}")
        return px.bar(title="Activity by Subsystem - Error")
    
    if not system_counts:
        return px.bar(title="Activity by Subsystem - No Valid Data")
    
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
    
    # Status indicators
    header_col1, header_col2 = st.columns([2, 1])
    with header_col1:
        if PRODUCTION_MODE:
            st.success("🚀 **Production System Active** - Enhanced capabilities enabled")
            st.subheader("Enterprise-grade log analysis with ML-powered insights")
        else:
            st.warning("⚠️ **Basic Mode** - Some features limited")
            st.subheader("Transform raw machine logs into readable summaries")
    
    with header_col2:
        if EDUCATIONAL_MODE and st.session_state.educational_mode:
            st.success("🎓 **Educational Mode Active**")
            st.write(f"Learning Level: {st.session_state.learning_level.title()}")
        elif EDUCATIONAL_MODE:
            st.info("🎓 Educational Features Available")
        else:
            st.warning("🎓 Educational Features Unavailable")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Educational mode toggle
    if EDUCATIONAL_MODE:
        st.sidebar.header("🎓 Educational Mode")
        st.session_state.educational_mode = st.sidebar.checkbox(
            "Enable Educational Features", 
            value=st.session_state.educational_mode,
            help="Enable story-driven explanations and learning features"
        )
        
        if st.session_state.educational_mode:
            st.session_state.learning_level = st.sidebar.selectbox(
                "Learning Level",
                ["beginner", "intermediate", "advanced", "expert"],
                index=["beginner", "intermediate", "advanced", "expert"].index(st.session_state.learning_level),
                help="Select your experience level for tailored explanations"
            )
    
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
                                # Extract categorized entries for UI compatibility
                                categorized_entries = results.get('categorized_entries', [])
                                st.session_state.log_data = categorized_entries  # Store categorized entries
                                st.session_state.production_results = results  # Store full results for reference
                                
                                # Create summary object for UI compatibility
                                st.session_state.summary = type('Summary', (), {
                                    'overall_status': '✅ Production Analysis Complete',
                                    'timeline_summary': results.get('narrative', 'Analysis completed successfully'),
                                    'key_events': [f"Processed {results.get('total_entries', 0)} entries in {results.get('processing_time', 0):.2f}s"],
                                    'risk_assessment': f"Risk Distribution: {results.get('risk_analysis', {}).get('distribution', {})}",
                                    'recommendations': [f"Processing speed: {results.get('performance_metrics', {}).get('entries_per_second', 0):.1f} entries/sec"],
                                    'technical_details': f"Database session: {results.get('session_id', 'N/A')}\nFormat detected: {getattr(results.get('parsing_metrics'), 'format_detected', 'Unknown')}"
                                })()
                                st.session_state.analysis_complete = True
                                st.sidebar.success("✅ Production analysis complete!")
                                
                                # Generate educational story if enabled
                                if st.session_state.educational_mode and EDUCATIONAL_MODE:
                                    try:
                                        story_engine = StoryNarrativeEngine()
                                        learning_level = LearningLevel(st.session_state.learning_level.upper())
                                        story = story_engine.generate_story(categorized_entries, learning_level)
                                        st.session_state.log_story = story
                                        st.sidebar.success("✅ Educational story generated!")
                                    except Exception as story_error:
                                        st.sidebar.warning(f"⚠️ Educational story generation failed: {story_error}")
                            else:
                                st.sidebar.error(f"❌ Analysis failed: {results.get('error_message')}")
                        except Exception as prod_error:
                            st.sidebar.error(f"❌ Production system error: {prod_error}")
                            # Fall back to basic mode
                            st.sidebar.info("🔄 Falling back to basic analysis...")
                            try:
                                from parser import LogParser
                                from categorizer import LogCategorizer
                                from summarizer import LogSummarizer
                                
                                parser = LogParser()
                                categorizer = LogCategorizer()
                                summarizer = LogSummarizer(api_key if api_key else None)
                            except ImportError as import_error:
                                st.sidebar.error(f"❌ Failed to import basic components: {import_error}")
                                return
                            
                            entries = parser.parse_text(log_text)
                            if entries:
                                categorized = categorizer.categorize_log_sequence(entries)
                                summary = summarizer.generate_summary(categorized)
                                
                                st.session_state.log_data = categorized
                                st.session_state.summary = summary
                                st.session_state.analysis_complete = True
                                st.sidebar.success("✅ Basic analysis complete!")
                                
                                # Generate educational story if enabled
                                if st.session_state.educational_mode and EDUCATIONAL_MODE:
                                    try:
                                        story_engine = StoryNarrativeEngine()
                                        learning_level = LearningLevel(st.session_state.learning_level.upper())
                                        story = story_engine.generate_story(categorized, learning_level)
                                        st.session_state.log_story = story
                                        st.sidebar.success("✅ Educational story generated!")
                                    except Exception as story_error:
                                        st.sidebar.warning(f"⚠️ Educational story generation failed: {story_error}")
                        finally:
                            try:
                                os.unlink(tmp_file_path)
                            except:
                                pass
                    
                    else:
                        # Use basic system
                        try:
                            from parser import LogParser
                            from categorizer import LogCategorizer
                            from summarizer import LogSummarizer
                            
                            parser = LogParser()
                            categorizer = LogCategorizer()
                            summarizer = LogSummarizer(api_key if api_key else None)
                        except ImportError as import_error:
                            st.sidebar.error(f"❌ Failed to import basic components: {import_error}")
                            return
                        
                        entries = parser.parse_text(log_text)
                        
                        if entries:
                            categorized = categorizer.categorize_log_sequence(entries)
                            summary = summarizer.generate_summary(categorized)
                            
                            st.session_state.log_data = categorized
                            st.session_state.summary = summary
                            st.session_state.analysis_complete = True
                            st.sidebar.success("✅ Basic analysis complete!")
                            
                            # Generate educational story if enabled
                            if st.session_state.educational_mode and EDUCATIONAL_MODE:
                                try:
                                    story_engine = StoryNarrativeEngine()
                                    learning_level = LearningLevel(st.session_state.learning_level.upper())
                                    story = story_engine.generate_story(categorized, learning_level)
                                    st.session_state.log_story = story
                                    st.sidebar.success("✅ Educational story generated!")
                                except Exception as story_error:
                                    st.sidebar.warning(f"⚠️ Educational story generation failed: {story_error}")
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
            if PRODUCTION_MODE and hasattr(st.session_state, 'production_results') and st.session_state.production_results:
                # Production mode with enhanced metrics
                prod_results = st.session_state.production_results
                total_entries = prod_results.get('total_entries', 0)
                processing_time = prod_results.get('processing_time', 0)
                entries_per_sec = prod_results.get('performance_metrics', {}).get('entries_per_second', 0)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Events", total_entries)
                col2.metric("Processing Time", f"{processing_time:.2f}s")
                col3.metric("Speed", f"{entries_per_sec:.0f} entries/sec")
            else:
                # Basic mode or fallback
                total_entries = len(categorized_entries) if hasattr(categorized_entries, '__len__') else 0
                error_count = 0
                warning_count = 0
                
                if hasattr(categorized_entries, '__iter__') and categorized_entries:
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
        if st.session_state.educational_mode and EDUCATIONAL_MODE:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Risk Distribution", "Phase Timeline", "System Activity", 
                "Detailed Table", "📖 Log Story", "🎓 Learning Center"
            ])
        else:
            tab1, tab2, tab3, tab4 = st.tabs(["Risk Distribution", "Phase Timeline", "System Activity", "Detailed Table"])
        
        with tab1:
            st.plotly_chart(create_risk_pie_chart(categorized_entries), use_container_width=True)
        
        with tab2:
            st.plotly_chart(create_phase_timeline(categorized_entries), use_container_width=True)
        
        with tab3:
            st.plotly_chart(create_system_activity_chart(categorized_entries), use_container_width=True)
        
        with tab4:
            # Detailed table
            if not categorized_entries:
                st.info("No data available for detailed table.")
            else:
                table_data = []
                try:
                    for entry in categorized_entries:
                        if hasattr(entry, 'log_entry') and hasattr(entry, 'phase') and hasattr(entry, 'risk_level'):
                            table_data.append({
                                "Time": entry.log_entry.timestamp,
                                "System": entry.log_entry.subsystem,
                                "Phase": entry.phase.value,
                                "Risk": entry.risk_level.value,
                                "Event": entry.log_entry.event,
                                "Explanation": getattr(entry, 'explanation', 'No explanation available')
                            })
                    
                    if table_data:
                        df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Add educational context if enabled
                        if st.session_state.educational_mode and EDUCATIONAL_MODE:
                            st.subheader("🎓 Educational Context")
                            st.write("Select a log entry to see detailed educational explanations:")
                            
                            # Entry selector
                            selected_entry_idx = st.selectbox(
                                "Choose log entry for detailed explanation:",
                                range(len(categorized_entries)),
                                format_func=lambda x: f"Entry {x+1}: {categorized_entries[x].log_entry.timestamp} - {categorized_entries[x].log_entry.event[:50]}..."
                            )
                            
                            if selected_entry_idx is not None:
                                try:
                                    selected_entry = categorized_entries[selected_entry_idx]
                                    education_system = ContextualEducationSystem()
                                    learning_level = LearningLevel(st.session_state.learning_level.upper())
                                    
                                    context = education_system.get_contextual_explanation(
                                        selected_entry.log_entry, 
                                        learning_level
                                    )
                                    
                                    # Display contextual explanation
                                    st.write(f"**Pattern: {context.pattern_name}**")
                                    
                                    explanation_col1, explanation_col2 = st.columns(2)
                                    
                                    with explanation_col1:
                                        st.write("**What it means:**")
                                        st.write(context.what_it_means)
                                        
                                        st.write("**Why it happens:**")
                                        st.write(context.why_it_happens)
                                        
                                        st.write("**Business impact:**")
                                        st.write(context.business_impact)
                                    
                                    with explanation_col2:
                                        st.write("**Investigation steps:**")
                                        for step in context.investigation_steps:
                                            st.write(f"• {step}")
                                        
                                        if context.red_flags:
                                            st.write("**🚩 Red flags:**")
                                            for flag in context.red_flags:
                                                st.write(f"• {flag}")
                                        
                                        if context.common_causes:
                                            st.write("**Common causes:**")
                                            for cause in context.common_causes:
                                                st.write(f"• {cause}")
                                    
                                    if context.remediation_steps:
                                        st.write("**Remediation steps:**")
                                        for step in context.remediation_steps:
                                            st.write(f"• {step}")
                                
                                except Exception as e:
                                    st.error(f"Error generating educational context: {e}")
                    else:
                        st.info("No valid entries found for detailed table.")
                
                except Exception as e:
                    st.error(f"Error creating detailed table: {e}")
        
        # Educational tabs (only if educational mode is enabled)
        if st.session_state.educational_mode and EDUCATIONAL_MODE:
            with tab5:
                # Log Story tab
                if st.session_state.log_story:
                    story = st.session_state.log_story
                    
                    st.header(f"📖 {story.title}")
                    
                    # Executive Summary
                    st.subheader("Executive Summary")
                    st.info(story.executive_summary)
                    
                    # Story Type Badge
                    st.subheader("Story Type")
                    story_type_colors = {
                        "error_cascade": "🔴",
                        "performance_degradation": "🟡", 
                        "resource_exhaustion": "🟠",
                        "security_incident": "🔴",
                        "normal_operation": "🟢"
                    }
                    story_icon = story_type_colors.get(story.story_type.value, "🔵")
                    st.write(f"{story_icon} **{story.story_type.value.replace('_', ' ').title()}**")
                    
                    # Detailed Narrative
                    st.subheader("What Happened?")
                    st.write(story.detailed_narrative)
                    
                    # Story Segments
                    st.subheader("Timeline Story")
                    for i, segment in enumerate(story.segments):
                        with st.expander(f"⏰ {segment.timestamp.strftime('%H:%M:%S')} - {segment.story_narrative[:50]}..."):
                            st.write("**What this means:**")
                            st.write(segment.story_narrative)
                            
                            st.write("**Business Impact:**")
                            st.write(segment.business_impact)
                            
                            st.write("**Learning Points:**")
                            for point in segment.learning_points:
                                st.write(f"• {point}")
                            
                            st.write("**Severity Explanation:**")
                            st.write(segment.severity_explanation)
                            
                            if segment.red_flags:
                                st.write("**🚩 Red Flags:**")
                                for flag in segment.red_flags:
                                    st.write(f"• {flag}")
                            
                            if segment.related_patterns:
                                st.write("**🔗 Related Patterns:**")
                                for pattern in segment.related_patterns:
                                    st.write(f"• {pattern}")
                    
                    # Lessons Learned
                    if story.lessons_learned:
                        st.subheader("📚 Lessons Learned")
                        for lesson in story.lessons_learned:
                            st.write(f"• {lesson}")
                    
                    # Investigation Methodology
                    st.subheader("🔍 How to Investigate This Type of Issue")
                    st.text(story.investigation_methodology)
                    
                    # Prevention Tips
                    if story.prevention_tips:
                        st.subheader("🛡️ Prevention Tips")
                        for tip in story.prevention_tips:
                            st.write(f"• {tip}")
                    
                    # Related Case Studies
                    if story.related_case_studies:
                        st.subheader("📊 Related Case Studies")
                        for case_study in story.related_case_studies:
                            st.write(f"• {case_study}")
                
                else:
                    st.info("No educational story available. Make sure to enable educational mode before analysis.")
            
            with tab6:
                # Learning Center tab
                st.header("🎓 Learning Center")
                
                # Pattern Recognition Training
                st.subheader("🎯 Pattern Recognition Training")
                
                training_col1, training_col2 = st.columns(2)
                
                with training_col1:
                    scenario_type = st.selectbox(
                        "Choose Training Scenario",
                        ["connection_timeout", "memory_leak", "performance_degradation"],
                        format_func=lambda x: x.replace('_', ' ').title()
                    )
                
                with training_col2:
                    difficulty = st.selectbox(
                        "Difficulty Level",
                        ["beginner", "intermediate", "advanced"]
                    )
                
                if st.button("🚀 Start Training Scenario"):
                    try:
                        trainer = PatternRecognitionTrainer()
                        scenario = trainer.create_training_scenario(scenario_type, difficulty)
                        st.session_state.training_scenario = scenario
                        st.success("Training scenario loaded!")
                    except Exception as e:
                        st.error(f"Error creating training scenario: {e}")
                
                # Display training scenario
                if st.session_state.training_scenario:
                    scenario = st.session_state.training_scenario
                    
                    st.subheader(f"📋 Scenario: {scenario['description']}")
                    
                    # Learning Objectives
                    st.write("**Learning Objectives:**")
                    for obj in scenario['learning_objectives']:
                        st.write(f"• {obj}")
                    
                    # Sample Logs
                    st.subheader("📝 Sample Logs")
                    log_df = pd.DataFrame(scenario['log_entries'])
                    st.dataframe(log_df, use_container_width=True)
                    
                    # Interactive Questions
                    st.subheader("❓ Questions")
                    for i, question in enumerate(scenario['questions']):
                        st.write(f"**Question {i+1}:** {question['question']}")
                        
                        if question['type'] == 'multiple_choice':
                            user_answer = st.radio(
                                f"Select your answer for Question {i+1}:",
                                question['options'],
                                key=f"q_{i}"
                            )
                            
                            if st.button(f"Check Answer {i+1}", key=f"check_{i}"):
                                if question['options'].index(user_answer) == question['correct_answer']:
                                    st.success(f"✅ Correct! {question['explanation']}")
                                else:
                                    st.error(f"❌ Incorrect. {question['explanation']}")
                        
                        elif question['type'] == 'short_answer':
                            user_answer = st.text_input(f"Your answer for Question {i+1}:", key=f"short_{i}")
                            
                            if st.button(f"Check Answer {i+1}", key=f"check_short_{i}"):
                                if user_answer.lower().strip() == question['correct_answer'].lower().strip():
                                    st.success(f"✅ Correct! {question['explanation']}")
                                else:
                                    st.error(f"❌ Incorrect. The correct answer is: {question['correct_answer']}. {question['explanation']}")
                    
                    # Hints
                    if scenario.get('hints'):
                        with st.expander("💡 Hints"):
                            for hint in scenario['hints']:
                                st.write(f"• {hint}")
                    
                    # Expected Patterns
                    st.subheader("🎯 Patterns to Look For")
                    for pattern in scenario['expected_patterns']:
                        st.write(f"• {pattern}")
                
                # Investigation Methodology Guide
                st.subheader("🔍 Investigation Methodology")
                
                methodology_guide = InvestigationMethodologyGuide()
                
                # Framework Selection
                method_col1, method_col2 = st.columns(2)
                
                with method_col1:
                    selected_framework = st.selectbox(
                        "Choose Investigation Framework",
                        [framework.value for framework in InvestigationFramework],
                        format_func=lambda x: x.replace('_', ' ').title()
                    )
                
                with method_col2:
                    investigation_scenario = st.selectbox(
                        "Investigation Scenario",
                        ["error_cascade", "performance_degradation", "security_incident", "resource_exhaustion"],
                        format_func=lambda x: x.replace('_', ' ').title()
                    )
                
                # Display framework guide
                framework_enum = InvestigationFramework(selected_framework)
                framework_guide = methodology_guide.get_framework_guide(framework_enum)
                
                if framework_guide:
                    st.write(f"**{framework_guide.get('name', 'Framework')}**")
                    st.write(framework_guide.get('description', ''))
                    
                    if 'questions' in framework_guide:
                        st.write("**Key Questions to Ask:**")
                        for category, questions in framework_guide['questions'].items():
                            st.write(f"*{category}:*")
                            for question in questions:
                                st.write(f"  • {question}")
                    
                    if 'benefits' in framework_guide:
                        st.write("**Benefits:**")
                        for benefit in framework_guide['benefits']:
                            st.write(f"• {benefit}")
                    
                    if 'best_for' in framework_guide:
                        st.write("**Best For:**")
                        for use_case in framework_guide['best_for']:
                            st.write(f"• {use_case}")
                
                # Investigation Plan
                if st.button("📋 Get Investigation Plan"):
                    try:
                        plan = methodology_guide.get_investigation_plan(investigation_scenario, framework_enum)
                        st.session_state.investigation_plan = plan
                        st.success("Investigation plan generated!")
                    except Exception as e:
                        st.error(f"Error generating investigation plan: {e}")
                
                # Display investigation plan
                if hasattr(st.session_state, 'investigation_plan') and st.session_state.investigation_plan:
                    plan = st.session_state.investigation_plan
                    
                    st.subheader(f"📋 Investigation Plan: {plan.scenario_type.replace('_', ' ').title()}")
                    
                    # Plan overview
                    plan_col1, plan_col2 = st.columns(2)
                    
                    with plan_col1:
                        st.write(f"**Estimated Time:** {plan.estimated_time}")
                        st.write("**Required Skills:**")
                        for skill in plan.required_skills:
                            st.write(f"• {skill}")
                    
                    with plan_col2:
                        st.write("**Success Criteria:**")
                        for criteria in plan.success_criteria:
                            st.write(f"• {criteria}")
                    
                    # Investigation steps
                    st.write("**Investigation Steps:**")
                    for i, step in enumerate(plan.steps):
                        with st.expander(f"Step {i+1}: {step.title}"):
                            st.write(f"**Phase:** {step.phase.value.replace('_', ' ').title()}")
                            st.write(f"**Description:** {step.description}")
                            
                            st.write("**Questions to Ask:**")
                            for question in step.questions_to_ask:
                                st.write(f"• {question}")
                            
                            st.write("**What to Look For:**")
                            for item in step.what_to_look_for:
                                st.write(f"• {item}")
                            
                            st.write("**Tools to Use:**")
                            for tool in step.tools_to_use:
                                st.write(f"• {tool}")
                            
                            st.write("**Expected Outcomes:**")
                            for outcome in step.expected_outcomes:
                                st.write(f"• {outcome}")
                            
                            if step.red_flags:
                                st.write("**🚩 Red Flags:**")
                                for flag in step.red_flags:
                                    st.write(f"• {flag}")
                            
                            if step.best_practices:
                                st.write("**📝 Best Practices:**")
                                for practice in step.best_practices:
                                    st.write(f"• {practice}")
                
                # Investigation Checklist
                st.subheader("✅ Investigation Checklist")
                checklist = methodology_guide.get_investigation_checklist(investigation_scenario)
                
                st.write(f"**Checklist for {investigation_scenario.replace('_', ' ').title()}:**")
                for item in checklist:
                    st.write(item)
                
                # Red Flags Guide
                st.subheader("🚩 Red Flags Guide")
                red_flags_guide = methodology_guide.get_red_flags_guide()
                
                for category, flags in red_flags_guide.items():
                    st.write(f"**{category.replace('_', ' ').title()}:**")
                    for flag in flags:
                        st.write(flag)
                
                # Best Practices
                st.subheader("📝 Best Practices")
                best_practices_categories = ["general", "data_collection", "analysis", "communication", "follow_up"]
                
                selected_category = st.selectbox(
                    "Choose Best Practices Category",
                    best_practices_categories,
                    format_func=lambda x: x.replace('_', ' ').title()
                )
                
                best_practices = methodology_guide.get_best_practices(selected_category)
                for practice in best_practices:
                    st.write(f"• {practice}")
                
                # Contextual Help
                st.subheader("🔍 Contextual Help")
                st.write("Click on any log entry in the detailed table to get contextual explanations!")
                
                # Learning Resources
                st.subheader("📚 Learning Resources")
                resources = [
                    "**Log Levels Guide**: Understanding ERROR, WARN, INFO, DEBUG",
                    "**Pattern Recognition**: Common patterns in system logs",
                    "**Investigation Methodology**: The 5 W's framework for log analysis",
                    "**Business Impact**: Translating technical issues to business terms"
                ]
                
                for resource in resources:
                    st.write(f"• {resource}")
        
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
                try:
                    from parser import LogParser
                    from categorizer import LogCategorizer
                    from summarizer import LogSummarizer
                    
                    parser = LogParser()
                    categorizer = LogCategorizer()
                    summarizer = LogSummarizer()
                except ImportError as import_error:
                    st.error(f"❌ Failed to load demo components: {import_error}")
                    return
                
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