from Crypto import Random
from Crypto.Random import random
from Crypto.PublicKey import ECC
import hashlib
import binascii
from pprint import pformat


def hash_data(data):
    if not isinstance(data, (bytes, bytearray)):
        data_str = str(data)
        data = data_str.encode()
    h = hashlib.sha3_256(data).hexdigest()
    return int(h, 16)


def validate_merkle_proof(proof, target_hash, merkle_root):
    merkle_root = bytearray.fromhex(merkle_root)
    target_hash = bytearray.fromhex(target_hash)
    if len(proof) == 0:
        return target_hash == merkle_root
    else:
        proof_hash = target_hash
        for p in proof:
            try:
                # the sibling is a left node
                sibling = bytearray.fromhex(p["left"])
                proof_hash = hashlib.sha3_256(sibling + proof_hash).digest()
            except:
                # the sibling is a right node
                sibling = bytearray.fromhex(p["right"])
                proof_hash = hashlib.sha3_256(proof_hash + sibling).digest()
        return proof_hash == merkle_root


def ecc_point_to_hex(point):
    point_str = "{},{}".format(point.x, point.y)
    return binascii.hexlify(point_str.encode()).decode()


def hex_to_ecc_point(hex_str, curve):
    point_str = binascii.unhexlify(hex_str).decode()
    x, y = map(int, point_str.split(","))
    return ECC.EccPoint(curve, x, y)


class MerkleTree(object):
    def __init__(self, hash_type="sha3_256"):
        hash_type = hash_type.lower()
        if hash_type in [
            "sha256",
            "sha224",
            "sha384",
            "sha512",
            "sha3_256",
            "sha3_224",
            "sha3_384",
            "sha3_512",
        ]:
            self.hash_function = getattr(hashlib, hash_type)
        else:
            raise Exception("`hash_type` {} nor supported".format(hash_type))

        self.reset_tree()

    def _to_hex(self, x):
        try:  # python3
            return x.hex()
        except:  # python2
            return binascii.hexlify(x)

    def reset_tree(self):
        self.leaves = list()
        self.levels = None
        self.is_ready = False

    def add_leaf(self, values, do_hash=False):
        self.is_ready = False
        # check if single leaf
        if not isinstance(values, tuple) and not isinstance(values, list):
            values = [values]
        for v in values:
            if do_hash:
                v = v.encode("utf-8")
                v = self.hash_function(v).hexdigest()
            v = bytearray.fromhex(v)
            self.leaves.append(v)

    def get_leaf(self, index):
        return self._to_hex(self.leaves[index])

    def get_leaf_count(self):
        return len(self.leaves)

    def get_tree_ready_state(self):
        return self.is_ready

    def _calculate_next_level(self):
        solo_leave = None
        N = len(self.levels[0])  # number of leaves on the level
        if N % 2 == 1:  # if odd number of leaves on the level
            solo_leave = self.levels[0][-1]
            N -= 1

        new_level = []
        for l, r in zip(self.levels[0][0:N:2], self.levels[0][1:N:2]):
            new_level.append(self.hash_function(l + r).digest())
        if solo_leave is not None:
            new_level.append(solo_leave)
        self.levels = [
            new_level,
        ] + self.levels  # prepend new level

    def make_tree(self):
        self.is_ready = False
        if self.get_leaf_count() > 0:
            self.levels = [
                self.leaves,
            ]
            while len(self.levels[0]) > 1:
                self._calculate_next_level()
        self.is_ready = True

    def get_merkle_root(self):
        if self.is_ready:
            if self.levels is not None:
                return self._to_hex(self.levels[0][0])
            else:
                return None
        else:
            return None

    def get_proof(self, index):
        if self.levels is None:
            return None
        elif not self.is_ready or index > len(self.leaves) - 1 or index < 0:
            return None
        else:
            proof = []
            for x in range(len(self.levels) - 1, 0, -1):
                level_len = len(self.levels[x])
                if (index == level_len - 1) and (
                    level_len % 2 == 1
                ):  # skip if this is an odd end node
                    index = int(index / 2.0)
                    continue
                is_right_node = index % 2
                sibling_index = index - 1 if is_right_node else index + 1
                sibling_pos = "left" if is_right_node else "right"
                sibling_value = self._to_hex(self.levels[x][sibling_index])
                proof.append({sibling_pos: sibling_value})
                index = int(index / 2.0)
            return proof


class ECCommitment:
    def __init__(self, g, h):
        self.g = g  # Base point of the curve
        self.h = h  # Another random point on the curve

    def commit(self, m):  # AKA Seal.
        m_val = hash_data(m)  # Compute hash of the data
        r = random.randint(1, 2**256)
        c1 = self.g.__mul__(m_val)
        c2 = self.h.__mul__(r)
        c = c1.__add__(c2)
        print(
            f"Committing: Data = {m}\nHashed Value = {m_val}\nRandom Value = {r}\nComputed Commitment = {c}\n"
        )
        return c, m_val, r

    def open(self, c, m_val, r):
        c1 = self.g.__mul__(m_val)
        c2 = self.h.__mul__(r)
        computed_c = c1.__add__(c2)
        print(
            f"\nOpening: Hashed Value = {m_val}\nRandom Value = {r}\nRecomputed Commitment = {computed_c}\nOriginal Commitment = {c}"
        )
        return computed_c == c


class StorageProver:
    def __init__(self, g, h):
        self.committer = ECCommitment(g, h)
        self.storage = {}
        self.data = {}
        self.commitments = {}
        self.random_vals = {}
        self.merkle_tree = MerkleTree()

    def store_data(self, data_blocks):
        for index, data in enumerate(data_blocks):
            c, m_val, r = self.committer.commit(data)
            self.data[index] = m_val  # Store hash value instead of data
            self.commitments[index] = c
            self.random_vals[index] = r
            self.merkle_tree.add_leaf(ecc_point_to_hex(c))
        self.merkle_tree.make_tree()
        return self.merkle_tree.get_merkle_root()

    def get_commitments(self):
        return self.commitments


class StorageVerifier:
    def __init__(self, g, h):
        self.stored_merkle_root = None
        self.committer = ECCommitment(g, h)
        self.stored_commitments = {}

    def receive_commitments(self, merkle_root, commitments):
        self.stored_merkle_root = merkle_root
        self.stored_commitments = commitments

    def challenge(self, num_blocks):
        num_blocks = min(num_blocks, len(self.stored_commitments.keys()))
        return random.sample(list(self.stored_commitments.keys()), num_blocks)


def prover_response(prover, challenge_indices):
    return [
        (prover.data[i], prover.random_vals[i], prover.merkle_tree.get_proof(i))
        for i in challenge_indices
    ]


def verifier_verify(verifier, challenge_indices, responses, merkle_root):
    for index, (data, r, merkle_proof) in zip(challenge_indices, responses):
        commitment = verifier.stored_commitments[index]
        if not verifier.committer.open(commitment, data, r):
            print("Opening failed")
            return False
        if not validate_merkle_proof(
            merkle_proof, ecc_point_to_hex(commitment), merkle_root
        ):
            print("Merkle proof validation failed")
            return False
    return True


# CRS Setup
curve = ECC.generate(curve="P-256")
g = curve.pointQ  # Base point
h = ECC.generate(curve="P-256").pointQ  # Another random point

# Initialize prover and verifier with the CRS
p = StorageProver(g, h)
v = StorageVerifier(g, h)

# Example usage:
data_blocks = [
    "Hello, world!",
    "This is a test.",
    bytearray(b"Some bytes"),
    {"key": "value"},
]

# miner stores data and commit to it
merkle_root = p.store_data(data_blocks)
print("Merkle root:", merkle_root)

# Miner sends back commitments with merkle root to validator
v.receive_commitments(merkle_root, p.get_commitments())

# validator issues challenge based on random subsample of commitments
challenge_indices = v.challenge(2)
print("indices to challenge:", challenge_indices)

# miner uses challenge indices to respond with data, random values, and merkle proofs
# responses are: [(data_i, r_i, merkle_proof_i) for i in challenge_indices]
responses = prover_response(p, challenge_indices)
print("\nresponses:", pformat(responses))

result = verifier_verify(v, challenge_indices, responses, merkle_root)
print("\nVerification passed?", result)
