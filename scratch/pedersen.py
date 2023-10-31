from Crypto.PublicKey import ECC
from Crypto.Random import random
import hashlib
import json


def generate_pedersen_commitment(chunk: bytes):
    """
    Generates a Pedersen commitment for a given data chunk and provides associated cryptographic
    data including the hash of the data chunk, point of commitment, and the randomness used.

    Returns:
        A dictionary containing:
        - index (simply the order of creation, starting from 1)
        - hash (SHA256 hash of the data chunk)
        - data_chunk (the input data chunk)
        - point (ECC point representing the commitment)
        - randomness (random scalar used in commitment generation)
        - merkle_proof (a sample merkle proof list containing dictionaries with left/right nodes)
    """
    # Constants for the curve and base points
    G = ECC._curves["NIST P-256"].G
    H = ECC._curves["NIST P-256"].point_at_coordinates(
        3, 7
    )  # Assuming H is a fixed point on curve

    # Randomness
    r = random.getrandbits(256)

    # Commitment
    C = r * G + hashlib.sha256(chunk).digest() * H

    # Constructing the result
    result = {
        "index": generate_pedersen_commitment.counter,
        "hash": int.from_bytes(hashlib.sha256(chunk).digest(), "big"),
        "data_chunk": chunk,
        "point": C,
        "randomness": r,
        "merkle_proof": [
            {"left": "313737343235...837363333"},
            {"right": "50187c137c60...15489b53"},
            {"left": "d6de056c72ce...6432ae0"},
            {"right": "5a0cbd07b862...3f113f"},
            {"left": "e9990f110037...7d20e4c6"},
        ],  # This is just a sample. In real-world applications, this will be computed differently.
    }

    # Increasing the counter
    generate_pedersen_commitment.counter += 1

    return result


# Initializing the counter attribute
generate_pedersen_commitment.counter = 1

# Test
data_chunk = b"Test data for commitment"
commitment = generate_pedersen_commitment(data_chunk)
print(json.dumps(commitment, indent=2, default=str))
