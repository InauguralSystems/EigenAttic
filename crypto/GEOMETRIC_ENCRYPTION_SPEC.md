# EigenScript Geometric Encryption Specification

## Version 2.0 (Research/Experimental)

---

## 1. Overview

This document specifies a novel encryption scheme built on EigenScript's geometric semantics. The scheme leverages:

- **XOR as Observer**: The XOR operation acts as a measurement condition, not an outcome selector
- **Temporal Tracking**: Bit flips are tracked as time evolution in LRVM space
- **Minkowski Signatures**: Encryption/decryption states are classified geometrically

### 1.1 Design Philosophy

Traditional encryption: Bit manipulation with computational hardness assumptions.

EigenScript encryption: Wave interference through an observer network, where the key configures the observation conditions and ciphertext emerges from geometric state evolution.

### 1.2 Security Status

**EXPERIMENTAL/RESEARCH ONLY**

This scheme has not undergone formal cryptographic analysis. It should not be used for protecting sensitive data in production until:
1. Formal security proofs are established
2. Independent cryptanalysis is performed
3. Implementation undergoes security audit

---

## 2. Threat Model

### 2.1 Attacker Capabilities (Assumed)

| Capability | Included |
|------------|----------|
| Ciphertext-only access | Yes |
| Known-plaintext pairs | Yes |
| Chosen-plaintext attacks | Yes |
| Chosen-ciphertext attacks | Future consideration |
| Side-channel access | Out of scope (v1) |
| Quantum computing | Out of scope (v1) |

### 2.2 Security Goals

1. **Confidentiality**: Ciphertext reveals no information about plaintext without the key
2. **Integrity**: (Optional) Tampering detection via geometric signature validation
3. **Key Sensitivity**: Single-bit key change produces completely different ciphertext

### 2.3 Trust Assumptions

- Key is generated with sufficient entropy (256+ bits recommended)
- Key is transmitted/stored securely (out of scope)
- Implementation is correct (verified via self-bootstrapping)

---

## 3. Mathematical Foundation

### 3.1 XOR in LRVM Space

The XOR operation is defined geometrically:

```
XOR(x, y) = (x OR y) AND NOT(x AND y)
```

With norm computation:
```
‖x XOR y‖² = |‖x‖² - ‖y‖²|
```

**Interpretation**: XOR measures the absolute difference in "semantic intensity" between two values.

### 3.2 Temporal State Evolution

Each value in EigenScript carries temporal metadata:

```c
struct EigenValue {
    double value;           // Current value
    int iteration;          // Time counter (increments on update)
    double history[N];      // Past values (trajectory)
    int history_index;      // Circular buffer pointer
};
```

Encryption leverages this: the `iteration` counter becomes a nonce, and `history` captures the encryption trajectory.

### 3.3 Minkowski Signature Classification

Values are classified by their geometric signature:

| Signature | Condition | Encryption Meaning |
|-----------|-----------|-------------------|
| Timelike | ‖v‖² < 0 | Key-influenced state |
| Spacelike | ‖v‖² > 0 | Data-influenced state |
| Lightlike | ‖v‖² = 0 | Observer boundary (XOR) |

The ciphertext trajectory should exhibit balanced signature transitions.

---

## 4. Key Specification

### 4.1 Key Structure

```
EigenKey = {
    seed: bytes[32],           // 256-bit random seed
    observer_config: int[16],  // XOR network configuration
    temporal_offset: int       // Initial iteration offset
}
```

### 4.2 Key Space

- Seed: 2^256 possibilities
- Observer config: 2^64 possibilities (16 × 4-bit values)
- Temporal offset: 2^32 possibilities

**Total key space**: ~2^352 (exceeds 256-bit security target)

### 4.3 Key Derivation

From a passphrase or master key:

```
1. hash = SHA-256(passphrase || salt)
2. seed = hash[0:32]
3. observer_config = derive_config(hash)
4. temporal_offset = hash[28:32] as uint32
```

---

## 5. Encryption Algorithm

### 5.1 Initialization

```python
def init_encryption(key: EigenKey) -> EncryptionState:
    state = EncryptionState()
    state.iteration = key.temporal_offset
    state.observer = configure_xor_network(key.observer_config)
    state.prng = init_prng(key.seed)
    return state
```

### 5.2 Block Encryption

For each block of plaintext (block size: 256 bits = 32 bytes):

```python
def encrypt_block(state: EncryptionState, plaintext: bytes) -> bytes:
    # Convert to integer array for precision
    p = bytes_to_ints(plaintext)
    
    # Generate keystream from temporal state
    keystream = state.prng.generate(len(p))
    
    # Apply XOR through observer network
    ciphertext = []
    for i, (plain_int, key_int) in enumerate(zip(p, keystream)):
        # XOR as observer: sets conditions, doesn't choose outcome
        xor_result = plain_int ^ key_int
        
        # Track in temporal state
        state.history.append(xor_result)
        state.iteration += 1
        
        # Compute geometric signature for verification
        signature = compute_signature(xor_result, state.iteration)
        
        ciphertext.append(xor_result)
    
    return ints_to_bytes(ciphertext)
```

### 5.3 CBC Mode (V2)

Version 2 uses Cipher Block Chaining for diffusion:

```python
def encrypt(key: EigenKey, plaintext: bytes) -> bytes:
    # Generate random IV for semantic security
    iv = random_bytes(16)
    
    # Derive separate encryption and MAC keys
    enc_key, mac_key = derive_subkeys(key.seed)
    
    # Pad to block boundary
    padded = pkcs7_pad(plaintext, BLOCK_SIZE)
    
    # CBC encryption: each block XORed with previous ciphertext
    ciphertext = b''
    prev_block = iv
    for i in range(0, len(padded), BLOCK_SIZE):
        block = padded[i:i+BLOCK_SIZE]
        chained = xor_bytes(block, prev_block)
        encrypted = encrypt_block(enc_key, chained)
        ciphertext += encrypted
        prev_block = encrypted
    
    # Build authenticated message
    header = encode_header(len(plaintext), key.temporal_offset)
    authenticated_data = header + iv + ciphertext
    
    # Compute HMAC for integrity
    mac = hmac_sha256(mac_key, authenticated_data)
    
    return authenticated_data + mac
```

### 5.4 Message Format (V2)

```
+----------+------+------------+--------+
|  Header  |  IV  | Ciphertext |  MAC   |
| 16 bytes | 16B  |  variable  | 32 bytes|
+----------+------+------------+--------+
```

---

## 6. Decryption Algorithm

### 6.1 V2 Decryption Process (CBC + HMAC)

```python
def decrypt(key: EigenKey, ciphertext: bytes) -> bytes:
    # Check version and dispatch
    version = ciphertext[0]
    if version == 1:
        return decrypt_v1(key, ciphertext)  # Legacy stream mode
    
    # Derive keys
    enc_key, mac_key = derive_subkeys(key.seed)
    
    # Verify MAC first (Encrypt-then-MAC pattern)
    received_mac = ciphertext[-32:]
    authenticated_data = ciphertext[:-32]
    expected_mac = hmac_sha256(mac_key, authenticated_data)
    
    if not constant_time_compare(received_mac, expected_mac):
        raise AuthenticationError("Message has been tampered")
    
    # Parse authenticated data
    header = authenticated_data[0:16]
    iv = authenticated_data[16:32]
    cipher_data = authenticated_data[32:]
    
    # CBC decryption
    plaintext = b''
    prev_block = iv
    for i in range(0, len(cipher_data), BLOCK_SIZE):
        block = cipher_data[i:i+BLOCK_SIZE]
        decrypted = decrypt_block(enc_key, block)
        plaintext_block = xor_bytes(decrypted, prev_block)
        plaintext += plaintext_block
        prev_block = block
    
    # Remove padding and verify length
    return pkcs7_unpad(plaintext, header.original_length)
```

### 6.2 Legacy V1 Decryption (Stream Mode)

For backward compatibility with v1 ciphertext:

```python
def decrypt_v1(key: EigenKey, ciphertext: bytes) -> bytes:
    header, cipher_data = parse_header(ciphertext)
    
    state = init_encryption(key)
    state.iteration = header.initial_iteration
    
    keystream = state.prng.generate(len(cipher_data))
    plaintext_padded = xor_bytes(cipher_data, keystream)
    
    return pkcs7_unpad(plaintext_padded, header.original_length)
```

---

## 7. Security Properties

### 7.1 Expected Properties

| Property | Status | Notes |
|----------|--------|-------|
| IND-CPA | Unproven | Requires formal analysis |
| Key sensitivity | Expected | Single-bit key change → complete output change |
| Avalanche effect | To be tested | Target: 50% bit flip on input change |
| Randomness | To be tested | Must pass NIST SP 800-22 tests |

### 7.2 V2 Security Improvements

Version 2.0 adds:

1. **CBC Mode**: Block chaining provides diffusion - changing one plaintext byte affects all subsequent ciphertext blocks
2. **HMAC-SHA256**: Message authentication code detects any tampering of ciphertext
3. **Random IV**: Each encryption uses a random 16-byte IV for semantic security (same plaintext → different ciphertext)
4. **Key Separation**: Separate encryption and MAC keys derived from master key

### 7.3 Known Limitations

1. **Floating-point precision**: All operations use integers internally to avoid precision attacks
2. **Side channels**: Current implementation does not address timing attacks (constant-time HMAC comparison used)

### 7.4 Geometric Security Hypothesis

The security of this scheme may derive from:

1. **Trajectory unpredictability**: Without the key, the attacker cannot reconstruct the temporal trajectory
2. **Observer configuration**: The XOR network topology is key-dependent
3. **Minkowski metric**: Geometric properties may provide additional structure

**Note**: These are hypotheses requiring formal proof.

---

## 8. Implementation Requirements

### 8.1 Precision

- All XOR operations on integers (no floating-point)
- Intermediate values stored as 64-bit integers
- PRNG must be cryptographically secure (e.g., ChaCha20)

### 8.2 Memory Safety

- Zeroize key material after use
- Constant-time comparison for authentication (future)
- No dynamic memory allocation in hot path

### 8.3 API Design

```python
# EigenScript builtins - format uses pipe separator
# Basic usage:
eigen_encrypt("plaintext|passphrase") -> hex_ciphertext
eigen_decrypt("hex_ciphertext|passphrase") -> plaintext
eigen_keygen("passphrase") -> json_key_info

# With configurable salt and iterations:
eigen_encrypt("plaintext|passphrase|salt|iterations") -> hex_ciphertext
eigen_decrypt("hex_ciphertext|passphrase|salt|iterations") -> plaintext
eigen_keygen("passphrase|salt|iterations") -> json_key_info

# Parameters:
# - salt: Custom salt string (default: "eigenscript_salt")
# - iterations: PBKDF2 iterations (default: 100000, min: 1000)
```

---

## 9. Test Vectors

### 9.1 Basic Test

```
Key (hex): 0x0000...0001 (256 bits, value 1)
Plaintext: "Hello, EigenScript!"
Expected: [To be generated during implementation]
```

### 9.2 Avalanche Test

```
Key1: 0x00...00
Key2: 0x00...01 (single bit difference)
Plaintext: "Test"
Expected: >45% bit difference in ciphertext
```

---

## 10. Future Work

1. **Formal proofs**: IND-CPA/IND-CCA2 security proofs using geometric properties
2. ~~**Authentication**: Add AEAD mode with geometric integrity checking~~ ✓ DONE (HMAC-SHA256)
3. **Side-channel resistance**: Full constant-time implementation
4. **Quantum analysis**: Post-quantum security evaluation
5. **Hardware acceleration**: CUDA kernel for bulk encryption
6. **Key management**: HSM integration and secure key storage

---

## Appendix A: Relation to Traditional Cryptography

| Concept | Traditional | EigenScript Geometric |
|---------|-------------|----------------------|
| XOR | Bitwise operation | Observer setting conditions |
| Key | Binary string | Observer configuration + temporal offset |
| Nonce | Counter/random | Iteration counter (temporal position) |
| Security basis | Computational hardness | Geometric trajectory unpredictability |
| State | Cipher state machine | LRVM vector evolution |

---

## Appendix B: References

1. EigenScript Language Specification
2. LRVM (Lightlike-Relational Vector Model) Documentation
3. EigenScript Logic Calculus (XOR definition)
4. NIST SP 800-22: Statistical Test Suite for Random Number Generators

---

*Document Status: Draft v2.1*
*Last Updated: January 2026*

---

## Changelog

### v2.1 (January 2026)
- Added configurable salt and iterations for PBKDF2 key derivation
- Minimum 1000 iterations enforced for security
- API extended: `passphrase|salt|iterations` format supported
- 55 tests total (35 core + 20 builtin integration)

### v2.0 (January 2026)
- Added CBC mode for block-level diffusion
- Added HMAC-SHA256 authentication (Encrypt-then-MAC)
- Added random IV for semantic security
- Added key separation (encryption key + MAC key)
- Backward compatible: v1 ciphertext still decryptable

### v1.0 (January 2026)
- Initial release with stream cipher mode
- XOR observer semantics
- Temporal tracking and Minkowski signatures
