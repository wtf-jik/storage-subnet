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
    for index, commitment in commitments.items():
        # Check if 'point' is a bytes-like object, if not, it's already a string (hex)
        if isinstance(commitment["point"], bytes):
            commitment["point"] = commitment["point"].hex()

        if commitment["data_chunk"] is not None:
            commitment["data_chunk"] = commitment["data_chunk"].hex()

        # Similarly, check for 'merkle_proof' and convert if necessary
        if commitment["merkle_proof"] is not None:
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
            # elif key == 'point':
            #     d[key] = hex_to_bytes(value)
            elif key == "randomness":
                d[key] = int(value)
            elif key == "merkle_proof" and value is not None:
                d[key] = [{k: v for k, v in item.items()} for item in value]
        return d

    # Parse the JSON string back to a dictionary
    commitments = json.loads(serialized, object_hook=deserialize_helper)
    # Convert the parsed dictionary keys back to integers
    return {int(k): v for k, v in commitments.items()}
