from Crypto import Random
from Crypto.Random import random
from Crypto.PublicKey import ECC
import hashlib


def hash_data(data):
    hasher = hashlib.sha3_256()
    hasher.update(str(data).encode())
    return hasher.hexdigest()


class ECCommitment:
    def __init__(self, g, h):
        self.g = g  # Base point of the curve
        self.h = h  # Another random point on the curve

    def commit(self, m):
        m_val = int(hash_data(m), 16)  # Compute hash of the data
        r = random.randint(1, 2**256)
        c1 = self.g.__mul__(m_val)
        c2 = self.h.__mul__(r)
        c = c1.__add__(c2)
        return c, r


def issue_challenge():
    return random.randint(1, 2**256)


class StorageProver:
    def __init__(self, g, h):
        self.committer = ECCommitment(g, h)
        self.data = None
        self.original_commitment = None
        self.r = None

    def store_data(self, data):
        self.data = data
        self.original_commitment, self.r = self.committer.commit(data)

    def get_commitment(self):
        return self.original_commitment

    def respond_to_challenge(self, challenge):
        combined_hash = hash_data(f"{self.data}{challenge}")
        response, _ = self.committer.commit(combined_hash)
        return response


class StorageVerifier:
    def __init__(self, g, h):
        self.g = g
        self.h = h
        self.stored_commitment = None
        self.committer = ECCommitment(g, h)

    def receive_commitment(self, commitment):
        self.stored_commitment = commitment

    def validate_response(self, challenge, response):
        combined_hash = hash_data(f"{self.stored_commitment}{challenge}")
        expected_response, _ = self.committer.commit(combined_hash)
        print(f"Expected response: {expected_response}")
        print(f"Actual response: {response}")
        return response.__eq__(expected_response)
        # return expected_response == response


# CRS Setup
curve = ECC.generate(curve="P-256")
g = curve.pointQ  # Base point
h = ECC.generate(curve="P-256").pointQ  # Another random point

# Initialize prover and verifier with the CRS
prover = StorageProver(g, h)
verifier = StorageVerifier(g, h)

# Prover stores their data and sends a commitment to the verifier
data = "Hello, world!"
prover.store_data(data)
commitment = prover.get_commitment()
verifier.receive_commitment(commitment)

# Verifier issues a random challenge to the prover
challenge = issue_challenge()

# Prover sends back a response
response = prover.respond_to_challenge(challenge)

# Verifier validates the response against their expectations
result = verifier.validate_response(challenge, response)
print("Verification passed?", result)
