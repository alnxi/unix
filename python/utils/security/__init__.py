"""
Security and Privacy Utilities

This module provides comprehensive security utilities including data encryption,
privacy-preserving techniques, and security best practices for machine learning workflows.

Features:
- Data encryption at rest and in transit
- Privacy-preserving machine learning utilities
- Audit logging for security events
- Secure credential management
- Data anonymization and masking
"""

from .encryption_utils import (
    DataEncryption,
    SecureDataManager,
    encrypt_data,
    decrypt_data,
    generate_encryption_key,
    hash_data
)

from .privacy_preserving import (
    DifferentialPrivacy,
    DataAnonymizer,
    PrivacyBudgetManager,
    add_noise,
    anonymize_dataset,
    k_anonymity
)

from .audit_logging import (
    SecurityAuditLogger,
    SecurityEvent,
    log_security_event,
    get_audit_logger
)

__all__ = [
    # Encryption
    "DataEncryption",
    "SecureDataManager", 
    "encrypt_data",
    "decrypt_data",
    "generate_encryption_key",
    "hash_data",
    
    # Privacy
    "DifferentialPrivacy",
    "DataAnonymizer",
    "PrivacyBudgetManager",
    "add_noise", 
    "anonymize_dataset",
    "k_anonymity",
    
    # Audit
    "SecurityAuditLogger",
    "SecurityEvent",
    "log_security_event",
    "get_audit_logger"
]