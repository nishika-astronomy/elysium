
"""
ELYSIUM — Scientific Data Processing Package

Responsible for validating, reading and processing
orbital and topographical datasets used in ELYSIUM.

Modules:
    validation — Scientific data validation
    mola       — MOLA elevation data processing
    ctx        — CTX image processing
    hirise     — HiRISE image processing
    themis     — THEMIS thermal and visual data processing
"""

__version__ = "0.1.0"

__all__ = [
    "validation",
    "mola",
    "ctx",
    "hirise",
    "themis",
]
