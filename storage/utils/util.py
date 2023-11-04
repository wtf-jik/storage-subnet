import os
import json
import base64
import binascii
from collections import defaultdict
from typing import Dict, List, Any

from Crypto.Random import random
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


def make_random_file(name=None, maxsize=1024):
    size = random.randint(32, maxsize)
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


def serialize_dict_with_bytes(commitments: Dict[int, Dict[str, Any]]) -> str:
    # Convert our custom objects to serializable objects
    for commitment in commitments:
        # Check if 'point' is a bytes-like object, if not, it's already a string (hex)
        if isinstance(commitment.get("point"), bytes):
            commitment["point"] = commitment["point"].hex()

        if commitment.get("data_chunk"):
            commitment["data_chunk"] = commitment["data_chunk"].hex()

        # Similarly, check for 'merkle_proof' and convert if necessary
        if commitment.get("merkle_proof"):
            serialized_merkle_proof = []
            for proof in commitment["merkle_proof"]:
                serialized_proof = {}
                for side, value in proof.items():
                    # Check if value is a bytes-like object, if not, it's already a string (hex)
                    if isinstance(value, bytes):
                        serialized_proof[side] = value.hex()
                    else:
                        serialized_proof[side] = value
                serialized_merkle_proof.append(serialized_proof)
            commitment["merkle_proof"] = serialized_merkle_proof

        # Randomness is an integer and should be safely converted to string without checking type
        if commitment.get("randomness"):
            commitment["randomness"] = str(commitment["randomness"])

    # Convert the entire structure to JSON
    return json.dumps(commitments)


# Deserializer function
def deserialize_dict_with_bytes(serialized: str) -> Dict[int, Dict[str, Any]]:
    def hex_to_bytes(hex_str: str) -> bytes:
        return bytes.fromhex(hex_str)

    def deserialize_helper(d: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in d.items():
            if key == "data_chunk":
                d[key] = hex_to_bytes(value)
            elif key == "randomness":
                d[key] = int(value)
            elif key == "merkle_proof" and value is not None:
                d[key] = [{k: v for k, v in item.items()} for item in value]
        return d

    # Parse the JSON string back to a dictionary
    return json.loads(serialized, object_hook=deserialize_helper)


def decode_commitments(encoded_commitments):
    decoded_commitments = base64.b64decode(encoded_commitments)
    commitments = deserialize_dict_with_bytes(decoded_commitments)
    return commitments


def decode_storage(encoded_storage):
    decoded_storage = base64.b64decode(encoded_storage).decode("utf-8")
    dict_storage = json.loads(decoded_storage)
    dict_storage["commitments"] = decode_commitments(dict_storage["commitments"])
    dict_storage["params"] = json.loads(
        base64.b64decode(dict_storage["params"]).decode("utf-8")
    )
    return dict_storage


def GetSynapse(config):
    # Setup CRS for this round of validation
    g, h = setup_CRS(curve=config.curve)

    # Make a random bytes file to test the miner
    random_data = make_random_file(maxsize=config.maxsize)

    # Random encryption key for now (never will decrypt)
    key = get_random_bytes(32)  # 256-bit key

    # Encrypt the data
    encrypted_data, nonce, tag = encrypt_data(
        random_data,
        key,  # TODO: Use validator key as the encryption key?
    )

    # Convert to base64 for compactness
    b64_encrypted_data = base64.b64encode(encrypted_data).decode("utf-8")

    # Hash the encrypted data
    data_hash = hash_data(encrypted_data)

    # Chunk the data
    chunk_size = get_random_chunksize()
    # chunks = list(chunk_data(encrypted_data, chunksize))

    syn = synapse = protocol.Store(
        chunk_size=chunk_size,
        encrypted_data=b64_encrypted_data,
        data_hash=data_hash,
        curve=config.curve,
        g=ecc_point_to_hex(g),
        h=ecc_point_to_hex(h),
        size=sys.getsizeof(encrypted_data),
    )
    return synapse
