"""Global-First implementation for international deployment and compliance.

This module provides comprehensive enterprise-grade global deployment features including:
- Multi-region compliance (GDPR, CCPA, HIPAA, PIPEDA, LGPD, PDPA, POPI)
- Comprehensive internationalization (i18n) and localization (l10n) for 40+ languages
- Cross-cultural privacy frameworks and cultural adaptation
- Cross-platform deployment and container orchestration
- Regional data residency and sovereignty
- Advanced multi-region deployment with disaster recovery
- Cultural sensitivity and cross-cultural UX adaptation
- Automated compliance reporting and regulatory change monitoring
"""

# Import available components
try:
    from .compliance_manager import ComplianceManager, RegionalCompliance, ComplianceFramework
    from .internationalization import I18nManager, LocaleConfiguration, CultureSettings, SupportedLocale
    from .deployment_orchestrator import DeploymentOrchestrator, RegionConfiguration, PlatformTarget
    from .cultural_privacy import (
        CulturalPrivacyFramework, 
        CulturalContext, 
        CulturalPrivacyProfile,
        CulturalAdaptation,
        CrossCulturalValidation
    )
    
    __all__ = [
        # Compliance Management
        "ComplianceManager", 
        "RegionalCompliance",
        "ComplianceFramework",
        
        # Internationalization
        "I18nManager",
        "LocaleConfiguration", 
        "CultureSettings",
        "SupportedLocale",
        
        # Deployment Orchestration
        "DeploymentOrchestrator",
        "RegionConfiguration",
        "PlatformTarget",
        
        # Cultural Privacy
        "CulturalPrivacyFramework",
        "CulturalContext",
        "CulturalPrivacyProfile",
        "CulturalAdaptation",
        "CrossCulturalValidation"
    ]
    
except ImportError as e:
    import logging
    logging.warning(f"Some global-first components not available: {e}")
    __all__ = []