import os
import json
import base64
import hashlib
import binascii
from collections import defaultdict
from typing import Dict, List, Any

import Crypto
from Crypto.Random import random
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

from .ecc import hex_to_ecc_point, ecc_point_to_hex, hash_data, ECCommitment
from .merkle import MerkleTree


def make_random_file(name=None, maxsize=1024):
    size = random.randint(128, maxsize)
    data = os.urandom(size)
    if isinstance(name, str):
        with open(name, "wb") as fout:
            fout.write(data)
    else:
        return data


# Determine a random chunksize between 2kb-128kb (random sample from this range) store as chunksize_E
def get_random_chunksize(maxsize=128):
    return random.randint(2, maxsize)


def chunk_data(data, chunksize: int):
    for i in range(0, len(data), chunksize):
        yield data[i : i + chunksize]


def is_hex_str(s):
    """
    Check if the input string is a valid hexadecimal string.

    :param s: The string to check
    :return: True if s is a valid hexadecimal string, False otherwise
    """
    # A valid hex string must have an even number of characters
    if len(s) % 2 != 0:
        return False

    # Check if each character is a valid hex character
    try:
        int(s, 16)
        return True
    except ValueError:
        return False


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


def decode_storage(encoded_storage):
    decoded_storage = base64.b64decode(encoded_storage).decode("utf-8")
    dict_storage = json.loads(decoded_storage)
    dict_storage["commitments"] = decode_commitments(dict_storage["commitments"])
    dict_storage["params"] = json.loads(
        base64.b64decode(dict_storage["params"]).decode("utf-8")
    )
    return dict_storage


def b64_encode(data):
    if isinstance(data, list) and isinstance(data[0], bytes):
        data = [d.hex() for d in data]
    if isinstance(data, dict) and isinstance(data[list(data.keys())[0]], bytes):
        data = {k: v.hex() for k, v in data.items()}
    return base64.b64encode(json.dumps(data).encode()).decode("utf-8")


def b64_decode(data, decode_hex=False):
    data = data.decode("utf-8") if isinstance(data, bytes) else data
    decoded_data = json.loads(base64.b64decode(data).decode("utf-8"))
    if decode_hex:
        try:
            decoded_data = (
                [bytes.fromhex(d) for d in decoded_data]
                if isinstance(decoded_data, list)
                else {k: bytes.fromhex(v) for k, v in decoded_data.items()}
            )
        except:
            pass
    return decoded_data


def encode_miner_storage(**kwargs):
    randomness = kwargs.get("randomness")
    chunks = kwargs.get("data_chunks")
    points = kwargs.get("commitments")
    points = [
        ecc_point_to_hex(p)
        for p in points
        if isinstance(p, Crypto.PublicKey.ECC.EccPoint)
    ]
    merkle_tree = kwargs.get("merkle_tree")

    # store (randomness values, merkle tree, commitments, data chunks)
    miner_store = {
        "randomness": b64_encode(randomness),
        "data_chunks": b64_encode(chunks),
        "commitments": b64_encode(points),
        "merkle_tree": b64_encode(merkle_tree.serialize()),
    }
    return json.dumps(miner_store).encode()


def decode_miner_storage(encoded_storage, curve):
    xy = json.loads(encoded_storage.decode("utf-8"))
    xz = {
        k: b64_decode(v, decode_hex=True if k != "commitments" else False)
        for k, v in xy.items()
    }
    xz["commitments"] = [hex_to_ecc_point(c, curve) for c in xz["commitments"]]
    xz["merkle_tree"] = MerkleTree().deserialize(xz["merkle_tree"])
    return xz


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


def verify_challenge(synapse):
    # TODO: Add checks and defensive programming here to handle all types
    # (bytes, str, hex, ecc point, etc)
    committer = ECCommitment(
        hex_to_ecc_point(synapse.g, synapse.curve),
        hex_to_ecc_point(synapse.h, synapse.curve),
    )
    commitment = hex_to_ecc_point(synapse.commitment, synapse.curve)

    if not committer.open(
        commitment, hash_data(synapse.data_chunk), synapse.random_value
    ):
        print(f"Opening commitment failed")
        return False

    if not validate_merkle_proof(
        synapse.merkle_proof, ecc_point_to_hex(commitment), synapse.merkle_root
    ):
        print(f"Merkle proof validation failed")
        return False

    return True
