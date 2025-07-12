"""
Investigation Methodology Guide for Log Analysis
Provides systematic approaches and frameworks for investigating log data
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

class InvestigationPhase(Enum):
    """Phases of log investigation"""
    INITIAL_ASSESSMENT = "initial_assessment"
    PATTERN_IDENTIFICATION = "pattern_identification"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    IMPACT_ASSESSMENT = "impact_assessment"
    RESOLUTION_PLANNING = "resolution_planning"
    VERIFICATION = "verification"

class InvestigationFramework(Enum):
    """Investigation frameworks available"""
    FIVE_WS = "five_ws"  # Who, What, When, Where, Why
    TIMELINE_ANALYSIS = "timeline_analysis"
    PATTERN_MATCHING = "pattern_matching"
    CORRELATION_ANALYSIS = "correlation_analysis"
    RISK_ASSESSMENT = "risk_assessment"

@dataclass
class InvestigationStep:
    """A single step in the investigation process"""
    phase: InvestigationPhase
    title: str
    description: str
    questions_to_ask: List[str]
    what_to_look_for: List[str]
    tools_to_use: List[str]
    expected_outcomes: List[str]
    red_flags: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)

@dataclass
class InvestigationPlan:
    """Complete investigation plan for a specific scenario"""
    scenario_type: str
    framework: InvestigationFramework
    steps: List[InvestigationStep]
    estimated_time: str
    required_skills: List[str]
    success_criteria: List[str]

class InvestigationMethodologyGuide:
    """Guide for systematic log investigation"""
    
    def __init__(self):
        self.frameworks = self._load_frameworks()
        self.scenario_plans = self._load_scenario_plans()
        self.best_practices = self._load_best_practices()
    
    def get_investigation_plan(self, scenario_type: str, framework: InvestigationFramework = InvestigationFramework.FIVE_WS) -> InvestigationPlan:
        """Get investigation plan for a specific scenario"""
        plan_key = f"{scenario_type}_{framework.value}"
        
        if plan_key in self.scenario_plans:
            return self.scenario_plans[plan_key]
        
        # Generate plan on the fly
        return self._generate_investigation_plan(scenario_type, framework)
    
    def get_framework_guide(self, framework: InvestigationFramework) -> Dict[str, Any]:
        """Get detailed guide for a specific framework"""
        return self.frameworks.get(framework, {})
    
    def get_best_practices(self, category: str = "general") -> List[str]:
        """Get best practices for log investigation"""
        return self.best_practices.get(category, [])
    
    def _load_frameworks(self) -> Dict[InvestigationFramework, Dict[str, Any]]:
        """Load investigation frameworks"""
        return {
            InvestigationFramework.FIVE_WS: {
                "name": "5 W's Framework",
                "description": "Systematic approach using Who, What, When, Where, Why",
                "questions": {
                    "Who": [
                        "Who was affected by this issue?",
                        "Who needs to be notified?",
                        "Who has the authority to fix this?",
                        "Who can provide more context?"
                    ],
                    "What": [
                        "What exactly happened?",
                        "What systems were involved?",
                        "What was the business impact?",
                        "What are the symptoms vs. root cause?"
                    ],
                    "When": [
                        "When did this issue start?",
                        "When was it first detected?",
                        "When did it end (if resolved)?",
                        "When have similar issues occurred?"
                    ],
                    "Where": [
                        "Where in the system did this occur?",
                        "Where else might it have spread?",
                        "Where are the logs stored?",
                        "Where should we look for more information?"
                    ],
                    "Why": [
                        "Why did this happen?",
                        "Why wasn't it prevented?",
                        "Why did it take so long to detect?",
                        "Why is this a priority to fix?"
                    ]
                },
                "benefits": [
                    "Comprehensive coverage of all aspects",
                    "Prevents missing important details",
                    "Great for incident reports",
                    "Easy to follow and teach"
                ],
                "best_for": ["incident response", "root cause analysis", "comprehensive investigations"]
            },
            
            InvestigationFramework.TIMELINE_ANALYSIS: {
                "name": "Timeline Analysis",
                "description": "Chronological reconstruction of events",
                "steps": [
                    "Collect all relevant timestamps",
                    "Create chronological timeline",
                    "Identify patterns and correlations",
                    "Look for cause-and-effect relationships",
                    "Identify decision points and opportunities"
                ],
                "benefits": [
                    "Clear sequence of events",
                    "Easy to spot correlations",
                    "Good for performance issues",
                    "Helps identify cascading failures"
                ],
                "best_for": ["performance analysis", "cascade failures", "system startup/shutdown"]
            },
            
            InvestigationFramework.PATTERN_MATCHING: {
                "name": "Pattern Matching",
                "description": "Identifying recurring patterns and anomalies",
                "techniques": [
                    "Frequency analysis",
                    "Statistical analysis",
                    "Baseline comparison",
                    "Anomaly detection",
                    "Clustering analysis"
                ],
                "benefits": [
                    "Identifies recurring issues",
                    "Spots anomalies quickly",
                    "Good for preventive analysis",
                    "Helps with trend analysis"
                ],
                "best_for": ["recurring issues", "trend analysis", "anomaly detection"]
            },
            
            InvestigationFramework.CORRELATION_ANALYSIS: {
                "name": "Correlation Analysis",
                "description": "Finding relationships between different events",
                "approach": [
                    "Identify all relevant data sources",
                    "Look for temporal correlations",
                    "Check for causal relationships",
                    "Analyze cross-system dependencies",
                    "Validate correlations with additional data"
                ],
                "benefits": [
                    "Uncovers hidden relationships",
                    "Good for complex systems",
                    "Helps with dependency mapping",
                    "Identifies systemic issues"
                ],
                "best_for": ["complex systems", "dependency issues", "multi-system problems"]
            },
            
            InvestigationFramework.RISK_ASSESSMENT: {
                "name": "Risk Assessment",
                "description": "Evaluating potential impacts and priorities",
                "factors": [
                    "Business impact severity",
                    "Likelihood of recurrence",
                    "Time to resolution",
                    "Resource requirements",
                    "Stakeholder impact"
                ],
                "benefits": [
                    "Prioritizes issues effectively",
                    "Helps resource allocation",
                    "Good for triage decisions",
                    "Aligns with business objectives"
                ],
                "best_for": ["incident triage", "resource planning", "business impact assessment"]
            }
        }
    
    def _load_scenario_plans(self) -> Dict[str, InvestigationPlan]:
        """Load pre-defined investigation plans for common scenarios"""
        return {
            "error_cascade_five_ws": InvestigationPlan(
                scenario_type="error_cascade",
                framework=InvestigationFramework.FIVE_WS,
                steps=[
                    InvestigationStep(
                        phase=InvestigationPhase.INITIAL_ASSESSMENT,
                        title="Initial Assessment",
                        description="Understand the scope and impact of the error cascade",
                        questions_to_ask=[
                            "How many systems are affected?",
                            "What is the business impact?",
                            "Is the cascade still ongoing?"
                        ],
                        what_to_look_for=[
                            "Error frequency and distribution",
                            "Time of first error",
                            "Systems showing errors"
                        ],
                        tools_to_use=[
                            "Log aggregation tools",
                            "Monitoring dashboards",
                            "Alert systems"
                        ],
                        expected_outcomes=[
                            "Scope of impact defined",
                            "Priority level established",
                            "Initial timeline created"
                        ],
                        red_flags=[
                            "Rapidly increasing error rates",
                            "Critical systems affected",
                            "Customer-facing services down"
                        ],
                        best_practices=[
                            "Document everything",
                            "Notify stakeholders early",
                            "Establish communication channels"
                        ]
                    ),
                    
                    InvestigationStep(
                        phase=InvestigationPhase.PATTERN_IDENTIFICATION,
                        title="Pattern Identification",
                        description="Identify the cascade pattern and initial trigger",
                        questions_to_ask=[
                            "What was the first error?",
                            "How did errors propagate?",
                            "Are there any common factors?"
                        ],
                        what_to_look_for=[
                            "Chronological order of errors",
                            "Common error messages",
                            "System dependencies"
                        ],
                        tools_to_use=[
                            "Timeline visualization",
                            "Dependency mapping",
                            "Pattern analysis tools"
                        ],
                        expected_outcomes=[
                            "Cascade pattern mapped",
                            "Initial trigger identified",
                            "Propagation path understood"
                        ],
                        red_flags=[
                            "Circular dependencies",
                            "Insufficient error handling",
                            "Tight coupling between systems"
                        ],
                        best_practices=[
                            "Create visual timeline",
                            "Verify dependencies",
                            "Document assumptions"
                        ]
                    ),
                    
                    InvestigationStep(
                        phase=InvestigationPhase.ROOT_CAUSE_ANALYSIS,
                        title="Root Cause Analysis",
                        description="Identify the fundamental cause of the initial failure",
                        questions_to_ask=[
                            "Why did the initial failure occur?",
                            "What conditions enabled the cascade?",
                            "Could this have been prevented?"
                        ],
                        what_to_look_for=[
                            "Resource constraints",
                            "Configuration changes",
                            "External factors"
                        ],
                        tools_to_use=[
                            "Resource monitoring",
                            "Configuration management",
                            "External service status"
                        ],
                        expected_outcomes=[
                            "Root cause identified",
                            "Contributing factors mapped",
                            "Prevention opportunities identified"
                        ],
                        red_flags=[
                            "Multiple root causes",
                            "Systemic issues",
                            "Recurring patterns"
                        ],
                        best_practices=[
                            "Use 5 Whys technique",
                            "Validate with data",
                            "Consider human factors"
                        ]
                    ),
                    
                    InvestigationStep(
                        phase=InvestigationPhase.IMPACT_ASSESSMENT,
                        title="Impact Assessment",
                        description="Quantify the business and technical impact",
                        questions_to_ask=[
                            "How many users were affected?",
                            "What was the financial impact?",
                            "How long did the outage last?"
                        ],
                        what_to_look_for=[
                            "User activity logs",
                            "Revenue impact data",
                            "Service availability metrics"
                        ],
                        tools_to_use=[
                            "Analytics platforms",
                            "Business intelligence tools",
                            "SLA monitoring"
                        ],
                        expected_outcomes=[
                            "Impact metrics calculated",
                            "SLA breach assessment",
                            "Business impact quantified"
                        ],
                        red_flags=[
                            "SLA violations",
                            "Customer complaints",
                            "Revenue loss"
                        ],
                        best_practices=[
                            "Use multiple data sources",
                            "Include indirect impacts",
                            "Document methodology"
                        ]
                    ),
                    
                    InvestigationStep(
                        phase=InvestigationPhase.RESOLUTION_PLANNING,
                        title="Resolution Planning",
                        description="Develop comprehensive resolution and prevention plan",
                        questions_to_ask=[
                            "How can we prevent this cascade?",
                            "What monitoring is needed?",
                            "Who needs to be involved in fixes?"
                        ],
                        what_to_look_for=[
                            "Circuit breaker opportunities",
                            "Monitoring gaps",
                            "Architectural improvements"
                        ],
                        tools_to_use=[
                            "Architecture review tools",
                            "Monitoring platforms",
                            "Project management tools"
                        ],
                        expected_outcomes=[
                            "Prevention plan created",
                            "Monitoring improvements identified",
                            "Responsibilities assigned"
                        ],
                        red_flags=[
                            "Complex fixes required",
                            "Multiple team dependencies",
                            "Long implementation timeline"
                        ],
                        best_practices=[
                            "Prioritize quick wins",
                            "Plan for monitoring",
                            "Include testing strategy"
                        ]
                    ),
                    
                    InvestigationStep(
                        phase=InvestigationPhase.VERIFICATION,
                        title="Verification",
                        description="Verify resolution and document learnings",
                        questions_to_ask=[
                            "Have the fixes been effective?",
                            "What did we learn?",
                            "How can we share this knowledge?"
                        ],
                        what_to_look_for=[
                            "Absence of similar errors",
                            "Improved monitoring data",
                            "Validation metrics"
                        ],
                        tools_to_use=[
                            "Monitoring dashboards",
                            "Testing frameworks",
                            "Documentation tools"
                        ],
                        expected_outcomes=[
                            "Resolution verified",
                            "Documentation completed",
                            "Knowledge shared"
                        ],
                        red_flags=[
                            "Similar errors recurring",
                            "Incomplete fixes",
                            "Poor documentation"
                        ],
                        best_practices=[
                            "Create runbooks",
                            "Share lessons learned",
                            "Update monitoring"
                        ]
                    )
                ],
                estimated_time="2-4 hours",
                required_skills=[
                    "Log analysis experience",
                    "System architecture understanding",
                    "Debugging skills",
                    "Communication skills"
                ],
                success_criteria=[
                    "Root cause identified and verified",
                    "Prevention plan implemented",
                    "Documentation completed",
                    "Knowledge shared with team"
                ]
            ),
            
            "performance_degradation_timeline": InvestigationPlan(
                scenario_type="performance_degradation",
                framework=InvestigationFramework.TIMELINE_ANALYSIS,
                steps=[
                    InvestigationStep(
                        phase=InvestigationPhase.INITIAL_ASSESSMENT,
                        title="Performance Baseline",
                        description="Establish performance baseline and identify degradation",
                        questions_to_ask=[
                            "What is normal performance?",
                            "When did degradation start?",
                            "How severe is the degradation?"
                        ],
                        what_to_look_for=[
                            "Response time metrics",
                            "Throughput measurements",
                            "Error rate changes"
                        ],
                        tools_to_use=[
                            "Performance monitoring tools",
                            "APM solutions",
                            "Log analysis tools"
                        ],
                        expected_outcomes=[
                            "Baseline established",
                            "Degradation quantified",
                            "Timeline started"
                        ],
                        red_flags=[
                            "Severe degradation",
                            "Rapid deterioration",
                            "Multiple metrics affected"
                        ],
                        best_practices=[
                            "Use multiple metrics",
                            "Compare to historical data",
                            "Include user experience metrics"
                        ]
                    ),
                    
                    InvestigationStep(
                        phase=InvestigationPhase.PATTERN_IDENTIFICATION,
                        title="Timeline Construction",
                        description="Build detailed timeline of performance changes",
                        questions_to_ask=[
                            "What events correlate with performance changes?",
                            "Are there any patterns in the degradation?",
                            "What changed in the system recently?"
                        ],
                        what_to_look_for=[
                            "Deployment timestamps",
                            "Configuration changes",
                            "Resource utilization spikes"
                        ],
                        tools_to_use=[
                            "Change management systems",
                            "Resource monitoring",
                            "Timeline visualization tools"
                        ],
                        expected_outcomes=[
                            "Detailed timeline created",
                            "Correlations identified",
                            "Change events mapped"
                        ],
                        red_flags=[
                            "Multiple concurrent changes",
                            "Resource exhaustion",
                            "Cascading effects"
                        ],
                        best_practices=[
                            "Include all system changes",
                            "Correlate with external events",
                            "Validate timestamps"
                        ]
                    )
                ],
                estimated_time="1-2 hours",
                required_skills=[
                    "Performance analysis",
                    "Timeline analysis",
                    "System monitoring"
                ],
                success_criteria=[
                    "Performance degradation explained",
                    "Timeline fully documented",
                    "Improvement plan created"
                ]
            )
        }
    
    def _load_best_practices(self) -> Dict[str, List[str]]:
        """Load best practices for different aspects of investigation"""
        return {
            "general": [
                "Document everything you do and find",
                "Keep detailed timestamps for all actions",
                "Validate assumptions with data",
                "Communicate findings regularly to stakeholders",
                "Don't jump to conclusions without evidence",
                "Consider multiple hypotheses simultaneously",
                "Use systematic approaches rather than random exploration",
                "Save all relevant data before it's rotated/deleted",
                "Create visual representations when possible",
                "Verify your findings with independent sources"
            ],
            
            "data_collection": [
                "Collect data from multiple sources",
                "Preserve original log files",
                "Note the time range of data collection",
                "Include system metadata and configuration",
                "Document data collection methodology",
                "Validate data completeness",
                "Check for data quality issues",
                "Consider time zone differences",
                "Include baseline/normal data for comparison",
                "Save intermediate analysis results"
            ],
            
            "analysis": [
                "Start with high-level patterns before diving deep",
                "Use statistical analysis to identify trends",
                "Look for correlations across different data sources",
                "Consider both technical and business context",
                "Validate patterns with additional data",
                "Question your assumptions regularly",
                "Consider alternative explanations",
                "Use visualization to reveal hidden patterns",
                "Test hypotheses systematically",
                "Document your reasoning process"
            ],
            
            "communication": [
                "Tailor communication to your audience",
                "Use clear, non-technical language for business stakeholders",
                "Provide regular status updates",
                "Create visual summaries for complex findings",
                "Distinguish between facts and hypotheses",
                "Be transparent about uncertainty",
                "Provide actionable recommendations",
                "Include confidence levels in your assessments",
                "Prepare for questions and challenges",
                "Document decisions and rationale"
            ],
            
            "follow_up": [
                "Create actionable remediation plans",
                "Assign clear ownership for follow-up actions",
                "Set realistic timelines for implementation",
                "Plan for verification of fixes",
                "Document lessons learned",
                "Update monitoring and alerting",
                "Share knowledge with the broader team",
                "Create runbooks for similar issues",
                "Schedule periodic reviews",
                "Measure effectiveness of improvements"
            ]
        }
    
    def _generate_investigation_plan(self, scenario_type: str, framework: InvestigationFramework) -> InvestigationPlan:
        """Generate investigation plan for scenarios not in the library"""
        # This would generate a plan based on the scenario type and framework
        # For now, return a basic plan
        return InvestigationPlan(
            scenario_type=scenario_type,
            framework=framework,
            steps=[
                InvestigationStep(
                    phase=InvestigationPhase.INITIAL_ASSESSMENT,
                    title="Initial Assessment",
                    description=f"Assess the {scenario_type} situation",
                    questions_to_ask=[
                        "What is the scope of the issue?",
                        "What is the business impact?",
                        "What data is available?"
                    ],
                    what_to_look_for=[
                        "Error messages and patterns",
                        "System resource usage",
                        "Recent changes"
                    ],
                    tools_to_use=[
                        "Log analysis tools",
                        "Monitoring dashboards",
                        "System metrics"
                    ],
                    expected_outcomes=[
                        "Scope defined",
                        "Impact assessed",
                        "Data sources identified"
                    ]
                )
            ],
            estimated_time="1-2 hours",
            required_skills=["Log analysis", "System troubleshooting"],
            success_criteria=["Issue understood", "Plan created", "Actions identified"]
        )
    
    def get_investigation_checklist(self, scenario_type: str) -> List[str]:
        """Get investigation checklist for a scenario"""
        checklists = {
            "error_cascade": [
                "☐ Identify the first error in the cascade",
                "☐ Map the propagation path",
                "☐ Determine business impact",
                "☐ Check for circuit breakers",
                "☐ Verify error handling mechanisms",
                "☐ Document timeline of events",
                "☐ Identify root cause",
                "☐ Plan prevention measures",
                "☐ Test recovery procedures",
                "☐ Update monitoring and alerting"
            ],
            
            "performance_degradation": [
                "☐ Establish performance baseline",
                "☐ Quantify degradation severity",
                "☐ Identify affected components",
                "☐ Check resource utilization",
                "☐ Review recent changes",
                "☐ Analyze user impact",
                "☐ Check database performance",
                "☐ Review network metrics",
                "☐ Identify optimization opportunities",
                "☐ Plan performance improvements"
            ],
            
            "security_incident": [
                "☐ Assess security impact",
                "☐ Identify attack vectors",
                "☐ Check for data exposure",
                "☐ Review access logs",
                "☐ Verify security controls",
                "☐ Document incident timeline",
                "☐ Notify security team",
                "☐ Preserve evidence",
                "☐ Plan remediation",
                "☐ Update security measures"
            ],
            
            "resource_exhaustion": [
                "☐ Identify exhausted resource",
                "☐ Determine usage patterns",
                "☐ Check resource limits",
                "☐ Review allocation policies",
                "☐ Analyze growth trends",
                "☐ Identify resource leaks",
                "☐ Plan capacity increases",
                "☐ Implement monitoring",
                "☐ Test scaling procedures",
                "☐ Update capacity planning"
            ]
        }
        
        return checklists.get(scenario_type, [
            "☐ Assess the situation",
            "☐ Gather relevant data",
            "☐ Analyze patterns",
            "☐ Identify root cause",
            "☐ Plan resolution",
            "☐ Implement fixes",
            "☐ Verify resolution",
            "☐ Document findings",
            "☐ Share learnings",
            "☐ Update processes"
        ])
    
    def get_red_flags_guide(self) -> Dict[str, List[str]]:
        """Get guide for identifying red flags in different scenarios"""
        return {
            "immediate_escalation": [
                "🚨 Customer-facing services completely down",
                "🚨 Data loss or corruption detected",
                "🚨 Security breach confirmed",
                "🚨 Cascading failures across multiple systems",
                "🚨 SLA violations exceeding thresholds",
                "🚨 Critical business processes stopped"
            ],
            
            "high_priority": [
                "🔴 Error rates above 10%",
                "🔴 Performance degradation > 50%",
                "🔴 Resource utilization > 90%",
                "🔴 Multiple system components failing",
                "🔴 Increasing error trends",
                "🔴 Failed health checks"
            ],
            
            "medium_priority": [
                "🟡 Error rates 5-10%",
                "🟡 Performance degradation 20-50%",
                "🟡 Resource utilization 70-90%",
                "🟡 Intermittent failures",
                "🟡 Warning threshold breaches",
                "🟡 Degraded service quality"
            ],
            
            "monitoring_required": [
                "🟢 Error rates 1-5%",
                "🟢 Performance degradation < 20%",
                "🟢 Resource utilization 50-70%",
                "🟢 Occasional anomalies",
                "🟢 Trending towards thresholds",
                "🟢 Minor service disruptions"
            ]
        }

# Export main class
__all__ = ['InvestigationMethodologyGuide', 'InvestigationPlan', 'InvestigationStep', 'InvestigationPhase', 'InvestigationFramework'] 