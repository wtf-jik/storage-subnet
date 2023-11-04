import os
import json
import base64
import binascii

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


# Function to serialize a dictionary with bytes
def serialize_dict_with_bytes(data):
    def encode(item):
        if isinstance(item, bytes):
            return base64.b64encode(item).decode("utf-8")
        elif isinstance(item, dict):
            return {k: encode(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [encode(element) for element in item]
        else:
            return item

    return json.dumps({k: encode(v) for k, v in data.items()})


# Function to deserialize a dictionary with bytes
def deserialize_dict_with_bytes(data):
    def decode(item):
        if isinstance(item, str):
            try:
                return base64.b64decode(item)
            except (TypeError, ValueError):
                return item
        elif isinstance(item, dict):
            return {k: decode(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [decode(element) for element in item]
        else:
            return item

    return {k: decode(v) for k, v in json.loads(data).items()}
