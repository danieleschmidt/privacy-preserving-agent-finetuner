"""Global-First implementation for international deployment and compliance.

This module provides enterprise-grade global deployment features including:
- Multi-region compliance (GDPR, CCPA, HIPAA, PIPEDA)
- Internationalization (i18n) and localization (l10n)
- Cross-platform deployment and container orchestration
- Regional data residency and sovereignty
"""

# Import available components
try:
    from .compliance_manager import ComplianceManager, RegionalCompliance, ComplianceFramework
    from .internationalization import I18nManager, LocaleConfiguration, CultureSettings
    from .deployment_orchestrator import DeploymentOrchestrator, RegionConfiguration, PlatformTarget
    
    __all__ = [
        "ComplianceManager", 
        "RegionalCompliance",
        "ComplianceFramework",
        "I18nManager",
        "LocaleConfiguration", 
        "CultureSettings",
        "DeploymentOrchestrator",
        "RegionConfiguration",
        "PlatformTarget"
    ]
    
except ImportError as e:
    import logging
    logging.warning(f"Some global-first components not available: {e}")
    __all__ = []