"""
Two-time-pad break against crypto/python_reference/core.py (v2 path).

Reproduces the v2 construction faithfully, then recovers the XOR of two
plaintexts encrypted under the SAME key using only the ciphertexts and the
public IVs — no key. With a crib (one guessed plaintext) the other message
falls out verbatim.

Root cause: the keystream KS_i = sha256(enc_key || block_index) has no IV
and no nonce in it (cbc_encrypt_blocks calls KeystreamGenerator(enc_key, 0)),
so every message under a given key reuses the identical keystream.

    $ python3 two_time_pad_break.py
    recovered P_a XOR P_b == true P_a XOR P_b : True
    decrypted p_b via crib on p_a           : b'weather is nice today, ...'
"""
import hashlib, secrets

BLOCK = 32
def xorb(a, b): return bytes(x ^ y for x, y in zip(a, b))
def sub(seed): return hashlib.sha256(seed + b"encryption").digest()

def ks(enc_key, i):                       # KS_i = sha256(enc_key || i) — no IV, no nonce
    return hashlib.sha256(enc_key + i.to_bytes(8, "big")).digest()

def enc(pt, seed):                        # faithful to cbc_encrypt_blocks(...) in core.py
    enc_key = sub(seed)
    iv = secrets.token_bytes(16); iv_ext = iv + iv
    pad = BLOCK - (len(pt) % BLOCK); pt = pt + bytes([pad]) * pad
    out = bytearray(); prev = iv_ext
    for i in range(0, len(pt), BLOCK):
        chained = xorb(pt[i:i+BLOCK], prev)
        c = xorb(chained, ks(enc_key, i // BLOCK))
        out += c; prev = c
    return iv, bytes(out)

if __name__ == "__main__":
    seed = secrets.token_bytes(32)        # ONE key, reused for two messages (the normal case)
    p_a = b"TRANSFER $10000 TO ACCOUNT 4471 -- routing done at noon sharp!!"
    p_b = b"weather is nice today, going for a walk in the park later on!!!"
    iv_a, c_a = enc(p_a, seed)
    iv_b, c_b = enc(p_b, seed)

    # Attacker knows only c_a, c_b, iv_a, iv_b (IVs travel in cleartext).
    def blocks(b): return [b[i:i+BLOCK] for i in range(0, len(b), BLOCK)]
    ca, cb = blocks(c_a), blocks(c_b)
    recovered = bytearray()
    prev_a, prev_b = iv_a + iv_a, iv_b + iv_b
    for i in range(len(ca)):
        # P_i ^ P_i' = (C_i ^ C_{i-1}) ^ (C_i' ^ C_{i-1}') — KS_i and the chain both cancel
        recovered += xorb(xorb(ca[i], prev_a), xorb(cb[i], prev_b))
        prev_a, prev_b = ca[i], cb[i]

    truth = xorb(p_a.ljust(len(recovered), b"\0"), p_b.ljust(len(recovered), b"\0"))
    print("recovered P_a XOR P_b == true P_a XOR P_b :",
          bytes(recovered[:len(p_a)]) == truth[:len(p_a)])
    print("decrypted p_b via crib on p_a           :",
          xorb(bytes(recovered[:len(p_a)]), p_a))
