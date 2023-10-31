import bittensor as bt
# import random
import os
import sys
import numpy as np
import pandas as pd
import hashlib
from collections import defaultdict
from Crypto import Random
from Crypto.Random import random
from Crypto.PublicKey import ECC
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from pprint import pprint, pformat
import binascii

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

    def update_leaf(self, index, new_value):
        """Update a specific leaf in the tree and propagate changes upwards."""
        if not self.is_ready:
            return None
        new_value = bytearray.fromhex(new_value)
        self.levels[-1][index] = new_value
        for x in range(len(self.levels) - 1, 0, -1):
            parent_index = index // 2
            left_child = self.levels[x][parent_index * 2]
            try:
                right_child = self.levels[x][parent_index * 2 + 1]
            except IndexError:
                right_child = bytearray()
            self.levels[x-1][parent_index] = self.hash_function(left_child + right_child).digest()
            index = parent_index


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

# Hash encrypted data E with SHA3-256 (args: encrypted data E) to get hash H
def hash_data(data):
    if not isinstance(data, (bytes, bytearray)):
        data_str = str(data)
        data = data_str.encode()
    h = hashlib.sha3_256(data).hexdigest()
    return int(h, 16)

def setup_CRS():
    curve = ECC.generate(curve="P-256")
    g = curve.pointQ  # Base point
    h = ECC.generate(curve="P-256").pointQ  # Another random point
    return g, h

g, h = setup_CRS()
print("g:", g)
print("h:", h)

def make_random_file(name=None, maxsize=1024):
    size = random.randint(32, maxsize)
    data = os.urandom(size)
    if isinstance(name, str):
        with open(name, "wb") as fout:
            fout.write(data)
    else:
        return data

f = random_file = make_random_file()
print("len(f):", len(f))


def encrypt_data(filename, key):
    """
    Encrypt the data in the given filename using AES-GCM.

    Parameters:
    - filename: str or bytes. If str, it's considered as a file name. If bytes, as the data itself.
    - key: bytes. 16-byte (128-bit), 24-byte (192-bit), or 32-byte (256-bit) secret key.

    Returns:
    - cipher_text: bytes. The encrypted data.
    - nonce: bytes. The nonce used for the GCM mode.
    - tag: bytes. The tag for authentication.
    """

    # If filename is a string, treat it as a file name and read the data
    if isinstance(filename, str):
        with open(filename, "rb") as file:
            data = file.read()
    else:
        data = filename

    # Initialize AES-GCM cipher
    cipher = AES.new(key, AES.MODE_GCM)
    
    # Encrypt the data
    cipher_text, tag = cipher.encrypt_and_digest(data)

    return cipher_text, cipher.nonce, tag

# Example usage
key = get_random_bytes(32)  # 256-bit key
encrypted_data, nonce, tag = encrypt_data(random_file, key)
pprint(f"encrypted_data: {encrypted_data}")

# Determine a random chunksize between 2kb-128kb (random sample from this range) store as chunksize_E
def get_random_chunksize(maxsize=128):
    return random.randint(2, maxsize)

bytesize = sys.getsizeof(encrypted_data)
print("bytesize:", bytesize)
chunksize = get_random_chunksize(bytesize // 4)
print("chunksize:", chunksize)

def chunk_data(data, chunksize: int):
    for i in range(0, len(data), chunksize):
        yield data[i : i + chunksize]

chunks = list(chunk_data(encrypted_data, chunksize))
print("n chunks:", len(chunks))
pprint(f"chunks: {chunks}")

committer = ECCommitment(g, h)
print("committer:", committer)

def commit_data(committer, data_chunks):
    merkle_tree = MerkleTree()
    commitments = defaultdict(lambda: [None] * len(data_chunks))

    for index, chunk in enumerate(data_chunks):
        c, m_val, r = committer.commit(chunk)
        commitments[index] = {
            "index": index,
            "hash": m_val,
            "data_chunk": chunk,
            "point": c,
            "randomness": r,
            "merkle_proof": None,
        }
        merkle_tree.add_leaf(ecc_point_to_hex(c))

    merkle_tree.make_tree()
    return {merkle_tree.get_merkle_root(): {'commitments': commitments, 'merkle_tree': merkle_tree}}

def ecc_point_to_hex(point):
    point_str = "{},{}".format(point.x, point.y)
    return binascii.hexlify(point_str.encode()).decode()

def hex_to_ecc_point(hex_str, curve):
    point_str = binascii.unhexlify(hex_str).decode()
    x, y = map(int, point_str.split(","))
    return ECC.EccPoint(curve, x, y)

store_data = commit_data
commitments_E = store_data(committer, chunks)
print("commitments_E:", commitments_E)

keys = list(commitments_E.keys())
print("keys:", keys)

inner_dict = commitments_E[keys[0]]
print("inner_dict:", inner_dict)

hashes = inner_dict["commitments"]["hashes"]
assert len(hashes) == len(chunks)

def get_challenge_indices(num_chunks, factor=0.1):
    return random.sample(list(range(num_chunks)), int(1 + num_chunks * factor))

ci = challenge_indices = get_challenge_indices(len(chunks))
print("challenge indices:", ci)

def get_merkle_root_to_challenge(commitments):
    merkle_roots = list(commitments.keys())
    return random.choice(merkle_roots)

mrc = merkle_root_challenge = get_merkle_root_to_challenge(commitments_E)
print("merkle root to challenge:", mrc)

responses = []
challenge_data = commitments_E[merkle_root_challenge]
merkle_tree = challenge_data['merkle_tree']
for i in challenge_indices:
    challenge_data['commitments'][i]['merkle_proof'] = merkle_tree.get_proof(i)
    responses.append(challenge_data['commitments'][i])

challenge_response = {merkle_root_challenge: responses}
challenge_response

merkle_root = list(challenge_response.keys())[0]
commitments = challenge_response[merkle_root]
print("merkle_root:", merkle_root)
print("commitments:", commitments)

for commitment_i in commitments:
    index = commitment_i['index']
    commitment = commitment_i['point']
    data = commitment_i['data_chunk']
    r = commitment_i['randomness']
    merkle_proof_i = commitment_i['merkle_proof']

    if not committer.open(commitment, hash_data(data), r):
        print(f"Opening commitment {index} failed")
        raise ValueError

    if not validate_merkle_proof(
        merkle_proof_i, ecc_point_to_hex(commitment_i['point']), merkle_root
    ):
        print(f"Merkle proof {index} validation failed")
        raise ValueError

print("All chunks validated successfuly!")


# Recommit data and send back to validator (miner side)
def recommit_data(committer, challenge_indices, merkle_tree, data):
    new_commitments = {}
    for i in challenge_indices:
        c, m_val, r = committer.commit(data[i])
        new_commitments[i] = {
            "commitment": c,
            "random_vals": r
        }
        merkle_tree.update_leaf(i, ecc_point_to_hex(c))
    new_merkle_root = merkle_tree.get_merkle_root()
    return new_merkle_root, new_commitments

new_merkle_root, new_commitments = recommit_data(committer, challenge_indices, merkle_tree, chunks)
print("new_merkle_root:", new_merkle_root)
print("new_commitments:", new_commitments)
