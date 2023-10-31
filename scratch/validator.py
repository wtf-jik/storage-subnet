import bittensor as bt
import random
import os
import numpy as np
import pandas as pd
import hashlib
from collections import defaultdict
from typing import Union


# VALIDATOR (Verifier):
# Setup ECC group by sharing g, h with miners/provers
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


def setup_CRS():
    curve = ECC.generate(curve="P-256")
    g = curve.pointQ  # Base point
    h = ECC.generate(curve="P-256").pointQ  # Another random point
    return g, h


# Generate random data D (of random length within boundary 32b-1024b)
def make_random_file(name=None, maxsize=1024):
    size = random.randint(32, maxsize)
    data = os.urandom(size)
    if isinstance(name, str):
        with open(name, "wb") as fout:
            fout.write(data)
    else:
        return data


# Encrypt data D with AES-256 (args: data D) to get encrypted data E
def encrypt_file(name: Union[str, bytes], key: bytes):
    """This does (insecure) XOR encryption"""
    with open(name, "rb") as fin:
        data = fin.read()
    data = bytearray(data)
    for i in range(len(data)):
        data[i] ^= key
    with open(name, "wb") as fout:
        fout.write(data)


# Hash encrypted data E with SHA-256 (args: encrypted data E) to get hash H
def hash_data(data):
    if not isinstance(data, (bytes, bytearray)):
        data_str = str(data)
        data = data_str.encode()
    h = hashlib.sha3_256(data).hexdigest()
    return int(h, 16)


# Determine a random chunksize between 2kb-128kb (random sample from this range) store as chunksize_E
def get_random_chunksize(maxsize=128):
    return random.randint(2, maxsize) * 1024


# Store hash H as key in a dictionary for later lookup (will store miner response later)
# Query miner M to store encrypted data E and generate commitments C_e (c, r, m_val) and return only (c, m_val). Sends chunksize_E with encrypted data E to miner.
# Store commitment responses C in a dictionary for later lookup (store miner response with merkle root as key)


# MINER (Prover):
# Chunk E according to chunksize_E into contiguous blocks of data E_i
def chunk_data(data, chunksize: int):
    for i in range(0, len(data), chunksize):
        yield data[i : i + chunksize]


# Create commitment params C_i: (c, r, m_val) for each chunk E_i of encrypted data E received from validator.
# Generate merkle tree and proofs for each chunk of data E_i (produces merkle root MR_e and list of merkle proofs MP_e_i)
def commit_data(committer, data_chunks):
    merkle_tree = MerkleTree()
    commitments = defaultdict(lambda: [None] * chunk_length)

    for index, data in enumerate(data_chunks):
        c, m_val, r = committer.commit(data)

        commitments["hashes"][index] = m_val
        commitments["point"][index] = c
        commitments["random"][index] = r

        merkle_tree.add_leaf(ecc_point_to_hex(c))

    merkle_tree.make_tree()
    return {merkle_tree.get_merkle_root(): commitments}


# Store commitments C and merkle proofs MP in a dictionary for later lookup with merkle root as key (store validator response)
