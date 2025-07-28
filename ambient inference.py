import zlib
import numpy as np
from hashlib import sha256

# --- Step 1: Complex evolving environment ---
def coupled_logistic_map(x, r=3.9, coupling=0.1):
    """
    x: current state vector (length d)
    returns next state vector
    """
    d = len(x)
    next_x = np.zeros(d)
    for i in range(d):
        left = x[i-1] if i > 0 else x[-1]
        right = x[i+1] if i < d-1 else x[0]
        next_x[i] = (1 - coupling) * r * x[i] * (1 - x[i]) + \
                    (coupling / 2) * (r * left * (1 - left) + r * right * (1 - right))
    return next_x

def generate_environment_series(seed_state, steps, r=3.9, coupling=0.1):
    env_states = [seed_state]
    x = seed_state
    for _ in range(steps - 1):
        x = coupled_logistic_map(x, r, coupling)
        env_states.append(x)
    return np.array(env_states)

# --- Step 2: Deterministic unbiased binary mapping from entire environment ---
def environment_to_bit(env_vector):
    # Quantize the float vector for hashing (4 decimal places)
    quantized = ''.join(f"{int(e*10000):04d}" for e in env_vector)
    h = sha256(quantized.encode()).digest()
    # XOR all bytes to get parity bit (0 or 1)
    parity = 0
    for b in h:
        parity ^= b
    return parity % 2

# --- Step 3: Generate coin flip sequence ---
def generate_coin_flips(env_series):
    return [environment_to_bit(state) for state in env_series]

# --- Step 4: Compressibility metric ---
def compressibility_ratio(bits):
    bitstring = ''.join(map(str, bits)).encode()
    compressed = zlib.compress(bitstring)
    return len(compressed) / len(bitstring)

# --- Main ---
if __name__ == "__main__":
    # Environment dimension (e.g. 20 complex microvariables)
    dim = 20

    # Initial random seed state (values in (0,1))
    seed = np.random.rand(dim)

    # Generate environment evolving over 1000 steps
    env = generate_environment_series(seed, 1000, r=3.9, coupling=0.1)

    # Generate coin flip sequence from environment
    coin_flips = generate_coin_flips(env)

    # Print first 50 coin flips
    coin_flip_str = ''.join(map(str, coin_flips))
    print("First 50 coin flips:")
    print(coin_flip_str[:50])

    # Compute compressibility
    ratio = compressibility_ratio(coin_flips)
    print(f"Compression ratio of coin flip sequence: {ratio:.3f} (lower = more compressible)")
