"""
EigenScript Geometric Encryption Module

Implements encryption/decryption using XOR observer semantics,
temporal tracking, and LRVM geometric principles.

EXPERIMENTAL/RESEARCH ONLY - Not for production use.
"""

from eigenscript.crypto.core import (
    EigenKey,
    encrypt,
    decrypt,
    generate_key,
)

from eigenscript.crypto.xor_observer import (
    XORObserver,
    apply_xor_observation,
)

__all__ = [
    "EigenKey",
    "encrypt",
    "decrypt",
    "generate_key",
    "XORObserver",
    "apply_xor_observation",
]
