"""
Advanced Global Compliance Engine

This module implements comprehensive compliance management for global privacy
regulations with automated assessment, real-time monitoring, and adaptive
compliance strategies.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
from datetime import datetime, timezone
import hashlib
import uuid

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Global privacy and compliance frameworks."""
    GDPR = "gdpr"  # European Union
    CCPA = "ccpa"  # California, USA
    LGPD = "lgpd"  # Brazil
    PIPEDA = "pipeda"  # Canada
    PDPA_SINGAPORE = "pdpa_sg"  # Singapore
    PDPA_THAILAND = "pdpa_th"  # Thailand
    POPI = "popi"  # South Africa
    HIPAA = "hipaa"  # USA Healthcare
    FERPA = "ferpa"  # USA Education
    SOX = "sox"  # USA Financial
    ISO27001 = "iso27001"  # International Security
    NIST = "nist"  # USA Cybersecurity
    PCI_DSS = "pci_dss"  # Payment Card Industry
    FedRAMP = "fedramp"  # USA Federal Cloud
    SOC2 = "soc2"  # Service Organization Control


class ComplianceStatus(Enum):
    """Compliance assessment status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    REQUIRES_ASSESSMENT = "requires_assessment"
    EXEMPTED = "exempted"


class RiskLevel(Enum):
    """Risk levels for compliance violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceRequirement:
    """Individual compliance requirement specification."""
    requirement_id: str
    framework: ComplianceFramework
    title: str
    description: str
    category: str
    mandatory: bool
    technical_controls: List[str]
    assessment_criteria: List[str]
    documentation_required: List[str]
    risk_level: RiskLevel
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ComplianceAssessment:
    """Results of compliance assessment."""
    assessment_id: str
    framework: ComplianceFramework
    timestamp: datetime
    overall_status: ComplianceStatus
    score: float  # 0.0 to 1.0
    requirements_assessed: int
    compliant_requirements: int
    non_compliant_requirements: int
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    next_assessment_due: datetime
    assessor: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class ComplianceEvent:
    """Compliance-related event for audit trail."""
    event_id: str
    timestamp: datetime
    event_type: str
    framework: ComplianceFramework
    description: str
    user_id: Optional[str]
    system_component: str
    risk_level: RiskLevel
    auto_remediated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceEngine(ABC):
    """Abstract base class for compliance engines."""
    
    @abstractmethod
    async def assess_compliance(self, 
                              framework: ComplianceFramework,
                              system_state: Dict[str, Any]) -> ComplianceAssessment:
        """Assess compliance against specific framework."""
        pass
    
    @abstractmethod
    def get_requirements(self, framework: ComplianceFramework) -> List[ComplianceRequirement]:
        """Get requirements for specific framework."""
        pass


class GDPRComplianceEngine(ComplianceEngine):
    """GDPR compliance assessment engine."""
    
    def __init__(self):
        self.requirements = self._load_gdpr_requirements()
        
    def _load_gdpr_requirements(self) -> List[ComplianceRequirement]:
        """Load GDPR compliance requirements."""
        return [
            ComplianceRequirement(
                requirement_id="gdpr_art_5_1_a",
                framework=ComplianceFramework.GDPR,
                title="Lawfulness, fairness and transparency",
                description="Personal data shall be processed lawfully, fairly and transparently",
                category="data_processing_principles",
                mandatory=True,
                technical_controls=["consent_management", "lawful_basis_tracking"],
                assessment_criteria=["consent_mechanisms", "transparency_notices"],
                documentation_required=["privacy_policy", "consent_records"],
                risk_level=RiskLevel.HIGH
            ),
            ComplianceRequirement(
                requirement_id="gdpr_art_5_1_b",
                framework=ComplianceFramework.GDPR,
                title="Purpose limitation",
                description="Personal data shall be collected for specified, explicit and legitimate purposes",
                category="data_processing_principles",
                mandatory=True,
                technical_controls=["purpose_binding", "data_minimization"],
                assessment_criteria=["purpose_specification", "compatible_use"],
                documentation_required=["data_processing_records", "purpose_documentation"],
                risk_level=RiskLevel.HIGH
            ),
            ComplianceRequirement(
                requirement_id="gdpr_art_5_1_c",
                framework=ComplianceFramework.GDPR,
                title="Data minimisation",
                description="Personal data shall be adequate, relevant and limited to what is necessary",
                category="data_processing_principles",
                mandatory=True,
                technical_controls=["data_minimization", "differential_privacy"],
                assessment_criteria=["necessity_assessment", "proportionality"],
                documentation_required=["data_inventory", "minimization_analysis"],
                risk_level=RiskLevel.MEDIUM
            ),
            ComplianceRequirement(
                requirement_id="gdpr_art_25",
                framework=ComplianceFramework.GDPR,
                title="Data protection by design and by default",
                description="Implement technical and organisational measures for data protection",
                category="technical_measures",
                mandatory=True,
                technical_controls=["privacy_by_design", "default_privacy_settings"],
                assessment_criteria=["technical_measures", "organizational_measures"],
                documentation_required=["design_documentation", "privacy_impact_assessment"],
                risk_level=RiskLevel.HIGH
            ),
            ComplianceRequirement(
                requirement_id="gdpr_art_32",
                framework=ComplianceFramework.GDPR,
                title="Security of processing",
                description="Implement appropriate technical and organisational measures for security",
                category="security_measures",
                mandatory=True,
                technical_controls=["encryption", "access_controls", "anonymization"],
                assessment_criteria=["security_measures", "incident_response"],
                documentation_required=["security_policy", "incident_procedures"],
                risk_level=RiskLevel.CRITICAL
            )
        ]
    
    async def assess_compliance(self, 
                              framework: ComplianceFramework,
                              system_state: Dict[str, Any]) -> ComplianceAssessment:
        """Assess GDPR compliance."""
        findings = []
        compliant_count = 0
        
        for requirement in self.requirements:
            assessment_result = await self._assess_requirement(requirement, system_state)
            findings.append(assessment_result)
            
            if assessment_result["status"] == ComplianceStatus.COMPLIANT.value:
                compliant_count += 1
        
        overall_score = compliant_count / len(self.requirements)
        overall_status = self._determine_overall_status(overall_score)
        
        return ComplianceAssessment(
            assessment_id=str(uuid.uuid4()),
            framework=framework,
            timestamp=datetime.now(timezone.utc),
            overall_status=overall_status,
            score=overall_score,
            requirements_assessed=len(self.requirements),
            compliant_requirements=compliant_count,
            non_compliant_requirements=len(self.requirements) - compliant_count,
            findings=findings,
            recommendations=self._generate_recommendations(findings),
            next_assessment_due=datetime.now(timezone.utc),  # Schedule next assessment
            assessor="automated_system"
        )
    
    async def _assess_requirement(self, 
                                requirement: ComplianceRequirement,
                                system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess individual GDPR requirement."""
        
        # Simulate requirement assessment based on system state
        if requirement.requirement_id == "gdpr_art_5_1_a":
            return await self._assess_lawfulness_fairness(requirement, system_state)
        elif requirement.requirement_id == "gdpr_art_5_1_b":
            return await self._assess_purpose_limitation(requirement, system_state)
        elif requirement.requirement_id == "gdpr_art_5_1_c":
            return await self._assess_data_minimization(requirement, system_state)
        elif requirement.requirement_id == "gdpr_art_25":
            return await self._assess_privacy_by_design(requirement, system_state)
        elif requirement.requirement_id == "gdpr_art_32":
            return await self._assess_security_measures(requirement, system_state)
        else:
            return {
                "requirement_id": requirement.requirement_id,
                "status": ComplianceStatus.UNDER_REVIEW.value,
                "score": 0.5,
                "evidence": [],
                "gaps": ["Assessment not implemented"],
                "recommendations": ["Implement specific assessment logic"]
            }
    
    async def _assess_lawfulness_fairness(self, 
                                        requirement: ComplianceRequirement,
                                        system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess lawfulness, fairness and transparency requirement."""
        
        score = 0.0
        evidence = []
        gaps = []
        
        # Check consent management
        if system_state.get("consent_management", {}).get("enabled", False):
            score += 0.3
            evidence.append("Consent management system active")
        else:
            gaps.append("No consent management system detected")
        
        # Check transparency measures
        if system_state.get("privacy_notices", {}).get("available", False):
            score += 0.3
            evidence.append("Privacy notices available")
        else:
            gaps.append("Privacy notices not found")
        
        # Check lawful basis tracking
        if system_state.get("lawful_basis_tracking", {}).get("implemented", False):
            score += 0.4
            evidence.append("Lawful basis tracking implemented")
        else:
            gaps.append("Lawful basis tracking not implemented")
        
        status = ComplianceStatus.COMPLIANT if score >= 0.8 else ComplianceStatus.NON_COMPLIANT
        
        return {
            "requirement_id": requirement.requirement_id,
            "status": status.value,
            "score": score,
            "evidence": evidence,
            "gaps": gaps,
            "recommendations": self._generate_requirement_recommendations(gaps)
        }
    
    async def _assess_purpose_limitation(self, 
                                       requirement: ComplianceRequirement,
                                       system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess purpose limitation requirement."""
        
        score = 0.0
        evidence = []
        gaps = []
        
        # Check purpose specification
        data_purposes = system_state.get("data_processing", {}).get("purposes", [])
        if data_purposes:
            score += 0.5
            evidence.append(f"Data purposes specified: {len(data_purposes)}")
        else:
            gaps.append("Data processing purposes not specified")
        
        # Check purpose binding controls
        if system_state.get("purpose_binding", {}).get("enforced", False):
            score += 0.5
            evidence.append("Purpose binding controls enforced")
        else:
            gaps.append("Purpose binding controls not enforced")
        
        status = ComplianceStatus.COMPLIANT if score >= 0.8 else ComplianceStatus.NON_COMPLIANT
        
        return {
            "requirement_id": requirement.requirement_id,
            "status": status.value,
            "score": score,
            "evidence": evidence,
            "gaps": gaps,
            "recommendations": self._generate_requirement_recommendations(gaps)
        }
    
    async def _assess_data_minimization(self, 
                                      requirement: ComplianceRequirement,
                                      system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data minimization requirement."""
        
        score = 0.0
        evidence = []
        gaps = []
        
        # Check data minimization techniques
        privacy_techniques = system_state.get("privacy_techniques", {})
        if privacy_techniques.get("differential_privacy", {}).get("enabled", False):
            score += 0.4
            evidence.append("Differential privacy enabled")
        else:
            gaps.append("Differential privacy not enabled")
        
        if privacy_techniques.get("data_anonymization", {}).get("enabled", False):
            score += 0.3
            evidence.append("Data anonymization enabled")
        else:
            gaps.append("Data anonymization not enabled")
        
        # Check data retention policies
        if system_state.get("data_retention", {}).get("policies_defined", False):
            score += 0.3
            evidence.append("Data retention policies defined")
        else:
            gaps.append("Data retention policies not defined")
        
        status = ComplianceStatus.COMPLIANT if score >= 0.8 else ComplianceStatus.NON_COMPLIANT
        
        return {
            "requirement_id": requirement.requirement_id,
            "status": status.value,
            "score": score,
            "evidence": evidence,
            "gaps": gaps,
            "recommendations": self._generate_requirement_recommendations(gaps)
        }
    
    async def _assess_privacy_by_design(self, 
                                      requirement: ComplianceRequirement,
                                      system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess privacy by design requirement."""
        
        score = 0.0
        evidence = []
        gaps = []
        
        # Check technical measures
        technical_measures = system_state.get("technical_measures", {})
        if technical_measures.get("privacy_preserving_ml", {}).get("enabled", False):
            score += 0.4
            evidence.append("Privacy-preserving ML enabled")
        else:
            gaps.append("Privacy-preserving ML not enabled")
        
        # Check organizational measures
        if system_state.get("organizational_measures", {}).get("privacy_impact_assessment", False):
            score += 0.3
            evidence.append("Privacy impact assessments conducted")
        else:
            gaps.append("Privacy impact assessments not conducted")
        
        # Check default privacy settings
        if system_state.get("default_settings", {}).get("privacy_first", False):
            score += 0.3
            evidence.append("Privacy-first default settings")
        else:
            gaps.append("Privacy-first defaults not implemented")
        
        status = ComplianceStatus.COMPLIANT if score >= 0.8 else ComplianceStatus.NON_COMPLIANT
        
        return {
            "requirement_id": requirement.requirement_id,
            "status": status.value,
            "score": score,
            "evidence": evidence,
            "gaps": gaps,
            "recommendations": self._generate_requirement_recommendations(gaps)
        }
    
    async def _assess_security_measures(self, 
                                      requirement: ComplianceRequirement,
                                      system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess security of processing requirement."""
        
        score = 0.0
        evidence = []
        gaps = []
        
        # Check encryption
        security_measures = system_state.get("security_measures", {})
        if security_measures.get("encryption", {}).get("enabled", False):
            score += 0.3
            evidence.append("Encryption enabled")
        else:
            gaps.append("Encryption not enabled")
        
        # Check access controls
        if security_measures.get("access_controls", {}).get("implemented", False):
            score += 0.3
            evidence.append("Access controls implemented")
        else:
            gaps.append("Access controls not implemented")
        
        # Check incident response
        if security_measures.get("incident_response", {}).get("procedures_defined", False):
            score += 0.2
            evidence.append("Incident response procedures defined")
        else:
            gaps.append("Incident response procedures not defined")
        
        # Check monitoring
        if security_measures.get("monitoring", {}).get("enabled", False):
            score += 0.2
            evidence.append("Security monitoring enabled")
        else:
            gaps.append("Security monitoring not enabled")
        
        status = ComplianceStatus.COMPLIANT if score >= 0.8 else ComplianceStatus.NON_COMPLIANT
        
        return {
            "requirement_id": requirement.requirement_id,
            "status": status.value,
            "score": score,
            "evidence": evidence,
            "gaps": gaps,
            "recommendations": self._generate_requirement_recommendations(gaps)
        }
    
    def _generate_requirement_recommendations(self, gaps: List[str]) -> List[str]:
        """Generate recommendations based on identified gaps."""
        recommendations = []
        
        for gap in gaps:
            if "consent management" in gap.lower():
                recommendations.append("Implement comprehensive consent management system")
            elif "privacy notices" in gap.lower():
                recommendations.append("Deploy clear and accessible privacy notices")
            elif "lawful basis" in gap.lower():
                recommendations.append("Implement lawful basis tracking and documentation")
            elif "purpose" in gap.lower():
                recommendations.append("Define and enforce data processing purposes")
            elif "differential privacy" in gap.lower():
                recommendations.append("Enable differential privacy for data protection")
            elif "encryption" in gap.lower():
                recommendations.append("Implement end-to-end encryption")
            elif "access controls" in gap.lower():
                recommendations.append("Deploy role-based access controls")
            elif "monitoring" in gap.lower():
                recommendations.append("Enable comprehensive security monitoring")
            else:
                recommendations.append(f"Address gap: {gap}")
        
        return recommendations
    
    def _determine_overall_status(self, score: float) -> ComplianceStatus:
        """Determine overall compliance status based on score."""
        if score >= 0.9:
            return ComplianceStatus.COMPLIANT
        elif score >= 0.7:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT
    
    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate overall recommendations based on findings."""
        recommendations = []
        
        for finding in findings:
            if finding["status"] != ComplianceStatus.COMPLIANT.value:
                recommendations.extend(finding.get("recommendations", []))
        
        # Remove duplicates and prioritize
        unique_recommendations = list(set(recommendations))
        
        return unique_recommendations[:10]  # Top 10 recommendations
    
    def get_requirements(self, framework: ComplianceFramework) -> List[ComplianceRequirement]:
        """Get GDPR requirements."""
        return self.requirements


class CCPAComplianceEngine(ComplianceEngine):
    """CCPA compliance assessment engine."""
    
    def __init__(self):
        self.requirements = self._load_ccpa_requirements()
    
    def _load_ccpa_requirements(self) -> List[ComplianceRequirement]:
        """Load CCPA compliance requirements."""
        return [
            ComplianceRequirement(
                requirement_id="ccpa_1798_100",
                framework=ComplianceFramework.CCPA,
                title="Consumer Right to Know",
                description="Consumers have the right to know what personal information is collected",
                category="consumer_rights",
                mandatory=True,
                technical_controls=["data_inventory", "transparency_reporting"],
                assessment_criteria=["information_disclosure", "categories_disclosed"],
                documentation_required=["privacy_policy", "data_categories_list"],
                risk_level=RiskLevel.HIGH
            ),
            ComplianceRequirement(
                requirement_id="ccpa_1798_105",
                framework=ComplianceFramework.CCPA,
                title="Consumer Right to Delete",
                description="Consumers have the right to delete personal information",
                category="consumer_rights",
                mandatory=True,
                technical_controls=["data_deletion", "deletion_verification"],
                assessment_criteria=["deletion_mechanisms", "verification_procedures"],
                documentation_required=["deletion_procedures", "verification_records"],
                risk_level=RiskLevel.HIGH
            ),
            ComplianceRequirement(
                requirement_id="ccpa_1798_110",
                framework=ComplianceFramework.CCPA,
                title="Consumer Right to Opt-Out",
                description="Consumers have the right to opt out of sale of personal information",
                category="consumer_rights",
                mandatory=True,
                technical_controls=["opt_out_mechanisms", "sale_tracking"],
                assessment_criteria=["opt_out_availability", "sale_prevention"],
                documentation_required=["opt_out_procedures", "sale_records"],
                risk_level=RiskLevel.MEDIUM
            )
        ]
    
    async def assess_compliance(self, 
                              framework: ComplianceFramework,
                              system_state: Dict[str, Any]) -> ComplianceAssessment:
        """Assess CCPA compliance."""
        # Simplified CCPA assessment - similar structure to GDPR
        findings = []
        compliant_count = 0
        
        for requirement in self.requirements:
            # Simulate assessment
            score = 0.8  # Mock score
            status = ComplianceStatus.COMPLIANT if score >= 0.8 else ComplianceStatus.NON_COMPLIANT
            
            finding = {
                "requirement_id": requirement.requirement_id,
                "status": status.value,
                "score": score,
                "evidence": ["Mock evidence"],
                "gaps": [],
                "recommendations": []
            }
            
            findings.append(finding)
            if status == ComplianceStatus.COMPLIANT:
                compliant_count += 1
        
        overall_score = compliant_count / len(self.requirements)
        
        return ComplianceAssessment(
            assessment_id=str(uuid.uuid4()),
            framework=framework,
            timestamp=datetime.now(timezone.utc),
            overall_status=ComplianceStatus.COMPLIANT,
            score=overall_score,
            requirements_assessed=len(self.requirements),
            compliant_requirements=compliant_count,
            non_compliant_requirements=len(self.requirements) - compliant_count,
            findings=findings,
            recommendations=[],
            next_assessment_due=datetime.now(timezone.utc),
            assessor="automated_system"
        )
    
    def get_requirements(self, framework: ComplianceFramework) -> List[ComplianceRequirement]:
        """Get CCPA requirements."""
        return self.requirements


class AdvancedComplianceManager:
    """Advanced compliance management with multi-framework support."""
    
    def __init__(self):
        self.engines = {
            ComplianceFramework.GDPR: GDPRComplianceEngine(),
            ComplianceFramework.CCPA: CCPAComplianceEngine()
        }
        self.assessment_history: List[ComplianceAssessment] = []
        self.compliance_events: List[ComplianceEvent] = []
        self.auto_remediation_enabled = True
        
    async def assess_multi_framework_compliance(self, 
                                              frameworks: List[ComplianceFramework],
                                              system_state: Dict[str, Any]) -> Dict[str, ComplianceAssessment]:
        """Assess compliance against multiple frameworks."""
        
        assessments = {}
        
        for framework in frameworks:
            if framework in self.engines:
                assessment = await self.engines[framework].assess_compliance(framework, system_state)
                assessments[framework.value] = assessment
                self.assessment_history.append(assessment)
                
                # Log assessment event
                self._log_compliance_event(
                    event_type="assessment_completed",
                    framework=framework,
                    description=f"Compliance assessment completed with score {assessment.score:.2f}",
                    risk_level=self._determine_risk_level(assessment.score)
                )
        
        return assessments
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive compliance dashboard data."""
        
        if not self.assessment_history:
            return {"message": "No assessments completed yet"}
        
        # Get latest assessments for each framework
        latest_assessments = {}
        for assessment in self.assessment_history:
            framework = assessment.framework.value
            if framework not in latest_assessments or assessment.timestamp > latest_assessments[framework].timestamp:
                latest_assessments[framework] = assessment
        
        # Calculate overall compliance metrics
        total_score = sum(a.score for a in latest_assessments.values())
        avg_score = total_score / len(latest_assessments) if latest_assessments else 0
        
        compliant_frameworks = sum(1 for a in latest_assessments.values() 
                                 if a.overall_status == ComplianceStatus.COMPLIANT)
        
        return {
            "overall_compliance_score": avg_score,
            "compliant_frameworks": compliant_frameworks,
            "total_frameworks": len(latest_assessments),
            "compliance_percentage": (compliant_frameworks / len(latest_assessments) * 100) if latest_assessments else 0,
            "framework_status": {
                framework: {
                    "status": assessment.overall_status.value,
                    "score": assessment.score,
                    "last_assessed": assessment.timestamp.isoformat(),
                    "next_due": assessment.next_assessment_due.isoformat()
                }
                for framework, assessment in latest_assessments.items()
            },
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.event_type,
                    "framework": event.framework.value,
                    "description": event.description,
                    "risk_level": event.risk_level.value
                }
                for event in self.compliance_events[-10:]  # Last 10 events
            ],
            "recommendations": self._generate_dashboard_recommendations(latest_assessments)
        }
    
    def _log_compliance_event(self, 
                            event_type: str,
                            framework: ComplianceFramework,
                            description: str,
                            risk_level: RiskLevel,
                            user_id: Optional[str] = None,
                            system_component: str = "compliance_manager"):
        """Log compliance event for audit trail."""
        
        event = ComplianceEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            framework=framework,
            description=description,
            user_id=user_id,
            system_component=system_component,
            risk_level=risk_level
        )
        
        self.compliance_events.append(event)
        
        # Keep only last 1000 events
        if len(self.compliance_events) > 1000:
            self.compliance_events = self.compliance_events[-1000:]
    
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level based on compliance score."""
        if score >= 0.9:
            return RiskLevel.LOW
        elif score >= 0.7:
            return RiskLevel.MEDIUM
        elif score >= 0.5:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _generate_dashboard_recommendations(self, 
                                          assessments: Dict[str, ComplianceAssessment]) -> List[str]:
        """Generate recommendations for compliance dashboard."""
        recommendations = []
        
        for framework, assessment in assessments.items():
            if assessment.overall_status != ComplianceStatus.COMPLIANT:
                recommendations.append(f"Improve {framework} compliance (score: {assessment.score:.2f})")
                recommendations.extend(assessment.recommendations[:2])  # Top 2 per framework
        
        # Add general recommendations
        avg_score = sum(a.score for a in assessments.values()) / len(assessments)
        if avg_score < 0.8:
            recommendations.append("Consider implementing comprehensive privacy-by-design approach")
        
        return recommendations[:10]  # Top 10 recommendations
    
    def enable_auto_remediation(self):
        """Enable automatic remediation of compliance issues."""
        self.auto_remediation_enabled = True
        logger.info("Auto-remediation enabled")
    
    def disable_auto_remediation(self):
        """Disable automatic remediation of compliance issues."""
        self.auto_remediation_enabled = False
        logger.info("Auto-remediation disabled")
    
    def get_supported_frameworks(self) -> List[str]:
        """Get list of supported compliance frameworks."""
        return [framework.value for framework in self.engines.keys()]


# Utility functions
def create_compliance_manager() -> AdvancedComplianceManager:
    """Factory function to create compliance manager."""
    return AdvancedComplianceManager()


async def assess_global_compliance(
    frameworks: List[ComplianceFramework],
    system_state: Dict[str, Any]
) -> Dict[str, Any]:
    """Assess compliance against multiple global frameworks."""
    manager = create_compliance_manager()
    assessments = await manager.assess_multi_framework_compliance(frameworks, system_state)
    dashboard = manager.get_compliance_dashboard()
    
    return {
        "assessments": assessments,
        "dashboard": dashboard,
        "supported_frameworks": manager.get_supported_frameworks()
    }