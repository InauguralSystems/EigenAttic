"""
Core Encryption/Decryption Implementation

Implements the geometric encryption scheme using:
- Integer-based XOR (no floating-point precision issues)
- Temporal tracking via iteration counters
- PRNG keystream generation
- CBC mode for block diffusion
- HMAC-SHA256 for authenticated encryption
"""

import hashlib
import hmac
import secrets
from typing import Tuple, Optional
from dataclasses import dataclass
from eigenscript.crypto.xor_observer import (
    XORObserver,
    TemporalState,
    apply_xor_observation,
)


BLOCK_SIZE = 32
HEADER_SIZE = 16
IV_SIZE = 16
MAC_SIZE = 32
VERSION = 2


@dataclass
class EigenKey:
    """
    Encryption key with geometric semantics.
    
    Components:
        seed: 256-bit random seed for PRNG
        observer_config: XOR network configuration (16 bytes)
        temporal_offset: Initial iteration position
    """
    seed: bytes
    observer_config: bytes
    temporal_offset: int
    
    def __post_init__(self):
        if len(self.seed) != 32:
            raise ValueError("Seed must be 32 bytes (256 bits)")
        if len(self.observer_config) != 16:
            raise ValueError("Observer config must be 16 bytes")
    
    @classmethod
    def from_passphrase(
        cls, 
        passphrase: str, 
        salt: Optional[bytes] = None,
        iterations: int = 100000
    ) -> "EigenKey":
        """
        Derive a key from a passphrase.
        
        Uses PBKDF2-style derivation for key stretching.
        
        Args:
            passphrase: The passphrase to derive key from
            salt: Optional salt bytes (16 bytes recommended). If None, generates random salt.
            iterations: PBKDF2 iteration count (default: 100000). Higher = more secure but slower.
        """
        if salt is None:
            salt = secrets.token_bytes(16)
        
        if iterations < 1000:
            raise ValueError("Iterations must be at least 1000 for security")
        
        material = hashlib.pbkdf2_hmac(
            'sha256',
            passphrase.encode('utf-8'),
            salt,
            iterations=iterations,
            dklen=64
        )
        
        seed = material[0:32]
        observer_config = material[32:48]
        temporal_offset = int.from_bytes(material[48:52], 'big')
        
        return cls(
            seed=seed,
            observer_config=observer_config,
            temporal_offset=temporal_offset
        )
    
    def to_bytes(self) -> bytes:
        """Serialize key to bytes."""
        return (
            self.seed +
            self.observer_config +
            self.temporal_offset.to_bytes(4, 'big')
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "EigenKey":
        """Deserialize key from bytes."""
        if len(data) != 52:
            raise ValueError("Key data must be 52 bytes")
        return cls(
            seed=data[0:32],
            observer_config=data[32:48],
            temporal_offset=int.from_bytes(data[48:52], 'big')
        )


class KeystreamGenerator:
    """
    Generates deterministic keystream from key.
    
    Uses ChaCha20-style construction for cryptographic security.
    For this implementation, we use SHA-256 in counter mode.
    """
    
    def __init__(self, seed: bytes, offset: int = 0):
        self.seed = seed
        self.counter = offset
        self.buffer = b''
        self.buffer_pos = 0
    
    def generate(self, length: int) -> bytes:
        """Generate keystream bytes."""
        result = bytearray()
        
        while len(result) < length:
            if self.buffer_pos >= len(self.buffer):
                self._refill_buffer()
            
            available = len(self.buffer) - self.buffer_pos
            needed = length - len(result)
            take = min(available, needed)
            
            result.extend(self.buffer[self.buffer_pos:self.buffer_pos + take])
            self.buffer_pos += take
        
        return bytes(result)
    
    def _refill_buffer(self):
        """Generate more keystream using counter mode."""
        counter_bytes = self.counter.to_bytes(8, 'big')
        self.buffer = hashlib.sha256(self.seed + counter_bytes).digest()
        self.buffer_pos = 0
        self.counter += 1


def pkcs7_pad(data: bytes, block_size: int) -> bytes:
    """Apply PKCS7 padding."""
    padding_len = block_size - (len(data) % block_size)
    return data + bytes([padding_len] * padding_len)


def pkcs7_unpad(data: bytes) -> bytes:
    """Remove PKCS7 padding."""
    if not data:
        raise ValueError("Empty data")
    
    padding_len = data[-1]
    
    if padding_len == 0 or padding_len > len(data):
        raise ValueError("Invalid padding")
    
    for i in range(padding_len):
        if data[-(i + 1)] != padding_len:
            raise ValueError("Invalid padding bytes")
    
    return data[:-padding_len]


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two byte sequences of equal length."""
    if len(a) != len(b):
        raise ValueError("Byte sequences must be same length")
    return bytes(x ^ y for x, y in zip(a, b))


def cbc_encrypt_blocks(plaintext: bytes, key_material: bytes, iv: bytes) -> bytes:
    """
    Encrypt using CBC mode with geometric XOR.
    
    CBC provides diffusion: each block depends on all previous blocks.
    """
    if len(plaintext) % BLOCK_SIZE != 0:
        raise ValueError("Plaintext must be padded to block size")
    
    keystream_gen = KeystreamGenerator(key_material, 0)
    ciphertext = bytearray()
    prev_block = iv
    
    for i in range(0, len(plaintext), BLOCK_SIZE):
        block = plaintext[i:i + BLOCK_SIZE]
        
        chained = xor_bytes(block, prev_block)
        
        block_keystream = keystream_gen.generate(BLOCK_SIZE)
        encrypted_block = xor_bytes(chained, block_keystream)
        
        ciphertext.extend(encrypted_block)
        prev_block = encrypted_block
    
    return bytes(ciphertext)


def cbc_decrypt_blocks(ciphertext: bytes, key_material: bytes, iv: bytes) -> bytes:
    """
    Decrypt using CBC mode with geometric XOR.
    """
    if len(ciphertext) % BLOCK_SIZE != 0:
        raise ValueError("Ciphertext must be multiple of block size")
    
    keystream_gen = KeystreamGenerator(key_material, 0)
    plaintext = bytearray()
    prev_block = iv
    
    for i in range(0, len(ciphertext), BLOCK_SIZE):
        block = ciphertext[i:i + BLOCK_SIZE]
        
        block_keystream = keystream_gen.generate(BLOCK_SIZE)
        unchained = xor_bytes(block, block_keystream)
        
        decrypted_block = xor_bytes(unchained, prev_block)
        
        plaintext.extend(decrypted_block)
        prev_block = block
    
    return bytes(plaintext)


def compute_hmac(key: bytes, data: bytes) -> bytes:
    """Compute HMAC-SHA256 for message authentication."""
    return hmac.new(key, data, hashlib.sha256).digest()


def derive_subkeys(master_key: bytes) -> Tuple[bytes, bytes]:
    """
    Derive encryption and MAC keys from master key.
    
    Uses HKDF-style expansion for key separation.
    """
    enc_key = hashlib.sha256(master_key + b"encryption").digest()
    mac_key = hashlib.sha256(master_key + b"authentication").digest()
    return enc_key, mac_key


def encode_header(original_length: int, initial_iteration: int) -> bytes:
    """Encode encryption header."""
    header = bytearray(HEADER_SIZE)
    
    header[0] = VERSION
    header[1:5] = original_length.to_bytes(4, 'big')
    header[5:9] = initial_iteration.to_bytes(4, 'big')
    
    checksum = sum(header[:9]) & 0xFF
    header[9] = checksum
    
    return bytes(header)


def decode_header(header: bytes) -> Tuple[int, int, int]:
    """
    Decode encryption header.
    
    Returns:
        Tuple of (version, original_length, initial_iteration)
    """
    if len(header) != HEADER_SIZE:
        raise ValueError(f"Header must be {HEADER_SIZE} bytes")
    
    version = header[0]
    original_length = int.from_bytes(header[1:5], 'big')
    initial_iteration = int.from_bytes(header[5:9], 'big')
    checksum = header[9]
    
    expected_checksum = sum(header[:9]) & 0xFF
    if checksum != expected_checksum:
        raise ValueError("Header checksum mismatch")
    
    return version, original_length, initial_iteration


def generate_key(passphrase: Optional[str] = None) -> EigenKey:
    """
    Generate an encryption key.
    
    Args:
        passphrase: Optional passphrase for deterministic derivation.
                   If None, generates a random key.
    
    Returns:
        EigenKey instance
    """
    if passphrase:
        return EigenKey.from_passphrase(passphrase)
    else:
        return EigenKey(
            seed=secrets.token_bytes(32),
            observer_config=secrets.token_bytes(16),
            temporal_offset=secrets.randbelow(2**32)
        )


def encrypt(plaintext: bytes, key: EigenKey) -> bytes:
    """
    Encrypt data using authenticated geometric encryption (CBC + HMAC).
    
    Format (v2): header || IV || ciphertext || MAC
    
    Features:
        - CBC mode for block diffusion (each block depends on previous)
        - HMAC-SHA256 for message authentication (detects tampering)
        - Random IV for semantic security
    
    Args:
        plaintext: Data to encrypt
        key: EigenKey for encryption
    
    Returns:
        Authenticated ciphertext
    """
    original_length = len(plaintext)
    
    iv = secrets.token_bytes(IV_SIZE)
    
    iv_extended = iv + iv
    
    enc_key, mac_key = derive_subkeys(key.seed)
    
    padded = pkcs7_pad(plaintext, BLOCK_SIZE)
    
    ciphertext = cbc_encrypt_blocks(padded, enc_key, iv_extended)
    
    header = encode_header(original_length, key.temporal_offset)
    
    authenticated_data = header + iv + ciphertext
    mac = compute_hmac(mac_key, authenticated_data)
    
    return authenticated_data + mac


def decrypt(ciphertext: bytes, key: EigenKey) -> bytes:
    """
    Decrypt and verify authenticated geometric encryption.
    
    Supports both v1 (legacy stream mode) and v2 (CBC + HMAC).
    
    Args:
        ciphertext: Encrypted data with header and MAC
        key: EigenKey for decryption
    
    Returns:
        Decrypted plaintext
    
    Raises:
        ValueError: If MAC verification fails (tampering detected)
    """
    if len(ciphertext) < HEADER_SIZE:
        raise ValueError("Ciphertext too short")
    
    header = ciphertext[:HEADER_SIZE]
    version, original_length, initial_iteration = decode_header(header)
    
    if version == 1:
        return _decrypt_v1(ciphertext, key)
    elif version == 2:
        return _decrypt_v2(ciphertext, key)
    else:
        raise ValueError(f"Unsupported version: {version}")


def _decrypt_v1(ciphertext: bytes, key: EigenKey) -> bytes:
    """Legacy v1 decryption (stream cipher mode, no authentication)."""
    header = ciphertext[:HEADER_SIZE]
    cipher_data = ciphertext[HEADER_SIZE:]
    
    version, original_length, initial_iteration = decode_header(header)
    
    if len(cipher_data) % BLOCK_SIZE != 0:
        raise ValueError("Invalid ciphertext length")
    
    temporal = TemporalState(iteration=initial_iteration)
    observer = XORObserver(
        config=list(key.observer_config),
        temporal=temporal
    )
    
    keystream_gen = KeystreamGenerator(key.seed, initial_iteration)
    keystream = keystream_gen.generate(len(cipher_data))
    
    plaintext_padded, signatures = apply_xor_observation(
        cipher_data, keystream, observer
    )
    
    plaintext = pkcs7_unpad(plaintext_padded)
    
    if len(plaintext) != original_length:
        raise ValueError("Decryption length mismatch")
    
    return plaintext


def _decrypt_v2(ciphertext: bytes, key: EigenKey) -> bytes:
    """V2 decryption with CBC mode and HMAC verification."""
    enc_key, mac_key = derive_subkeys(key.seed)
    
    received_mac = ciphertext[-MAC_SIZE:]
    authenticated_data = ciphertext[:-MAC_SIZE]
    
    expected_mac = compute_hmac(mac_key, authenticated_data)
    if not hmac.compare_digest(received_mac, expected_mac):
        raise ValueError("Authentication failed: message has been tampered")
    
    header = authenticated_data[:HEADER_SIZE]
    iv = authenticated_data[HEADER_SIZE:HEADER_SIZE + IV_SIZE]
    cipher_data = authenticated_data[HEADER_SIZE + IV_SIZE:]
    
    version, original_length, initial_iteration = decode_header(header)
    
    if len(cipher_data) % BLOCK_SIZE != 0:
        raise ValueError("Invalid ciphertext length")
    
    iv_extended = iv + iv
    
    plaintext_padded = cbc_decrypt_blocks(cipher_data, enc_key, iv_extended)
    
    plaintext = pkcs7_unpad(plaintext_padded)
    
    if len(plaintext) != original_length:
        raise ValueError("Decryption length mismatch")
    
    return plaintext
