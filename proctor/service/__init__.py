from __future__ import annotations
"""
Proctor Service

Main proctoring service that integrates all components.
"""

from proctor.service.proctoring import ProctoringService, main

__all__ = [
    "ProctoringService",
    "main",
]
