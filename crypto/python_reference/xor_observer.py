"""
XOR Observer Implementation

The XOR operation in EigenScript is the "observer" - it sets conditions
for interference without choosing outcomes. This module implements
integer-based XOR with temporal tracking.
"""

from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
class TemporalState:
    """
    Tracks the temporal evolution of XOR observations.
    
    Bit flips are tracked like time - each XOR operation
    advances the iteration counter and records history.
    """
    iteration: int = 0
    history: List[int] = field(default_factory=list)
    max_history: int = 1024
    
    def advance(self, value: int) -> int:
        """Record an observation and advance time."""
        self.iteration += 1
        self.history.append(value)
        
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        return self.iteration
    
    def get_signature(self) -> str:
        """
        Compute Minkowski-style signature of current state.
        
        Returns:
            "timelike", "spacelike", or "lightlike"
        """
        if len(self.history) < 2:
            return "lightlike"
        
        diff = self.history[-1] - self.history[-2]
        
        if diff < 0:
            return "timelike"
        elif diff > 0:
            return "spacelike"
        else:
            return "lightlike"


@dataclass
class XORObserver:
    """
    The XOR Observer - sets conditions for wave interference.
    
    In EigenScript's geometric model, XOR doesn't "flip bits" -
    it establishes the observation conditions under which
    interference patterns emerge.
    """
    config: List[int]
    temporal: TemporalState = field(default_factory=TemporalState)
    
    def observe(self, value: int, keystream_byte: int) -> Tuple[int, str]:
        """
        Apply XOR observation.
        
        Args:
            value: Input value (plaintext byte or ciphertext byte)
            keystream_byte: Key-derived observation condition
        
        Returns:
            Tuple of (observed_value, signature_type)
        """
        observed = value ^ keystream_byte
        
        self.temporal.advance(observed)
        
        signature = self.temporal.get_signature()
        
        return observed, signature


def apply_xor_observation(
    data: bytes,
    keystream: bytes,
    observer: XORObserver
) -> Tuple[bytes, List[str]]:
    """
    Apply XOR observation to a sequence of bytes.
    
    Args:
        data: Input data
        keystream: Key-derived observation conditions
        observer: XOR observer with temporal state
    
    Returns:
        Tuple of (output_bytes, signature_sequence)
    """
    if len(data) != len(keystream):
        raise ValueError("Data and keystream must be same length")
    
    output = bytearray(len(data))
    signatures = []
    
    for i, (d, k) in enumerate(zip(data, keystream)):
        observed, sig = observer.observe(d, k)
        output[i] = observed
        signatures.append(sig)
    
    return bytes(output), signatures


def compute_avalanche(input1: bytes, input2: bytes) -> float:
    """
    Compute avalanche effect (percentage of differing bits).
    
    Args:
        input1: First byte sequence
        input2: Second byte sequence
    
    Returns:
        Percentage of bits that differ (0.0 to 1.0)
    """
    if len(input1) != len(input2):
        raise ValueError("Inputs must be same length")
    
    total_bits = len(input1) * 8
    differing_bits = 0
    
    for b1, b2 in zip(input1, input2):
        xor = b1 ^ b2
        differing_bits += bin(xor).count('1')
    
    return differing_bits / total_bits
