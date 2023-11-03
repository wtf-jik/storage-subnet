import base64
import json


def encode_dict(d):
    """
    Recursively encode bytes in a dictionary to base64 strings
    """
    for k, v in d.items():
        # if the value is bytes, encode it in base64
        if isinstance(v, bytes):
            d[k] = base64.b64encode(v).decode("utf-8")
        # if it's a dictionary, recurse
        elif isinstance(v, dict):
            d[k] = encode_dict(v)
        # if it's a list, iterate through items
        elif isinstance(v, list):
            d[k] = [encode_dict(item) if isinstance(item, dict) else item for item in v]
    return d


def serialize_dict(d):
    """
    Serialize the dictionary to a JSON string after encoding bytes
    """
    encoded_dict = encode_dict(d.copy())  # make a copy to preserve original dictionary
    return json.dumps(encoded_dict)


def is_base64(sb):
    """
    Check if a string is a valid base64 encoded string.
    """
    try:
        if isinstance(sb, str):
            # If there's any unicode here, an exception will be thrown and the function will return false
            sb_bytes = bytes(sb, "ascii")
        elif isinstance(sb, bytes):
            sb_bytes = sb
        else:
            raise ValueError("Argument must be a string or bytes")

        return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
    except Exception:
        return False


def decode_dict(d):
    """
    Recursively decode base64 strings in a dictionary to bytes
    """
    for k, v in d.items():
        # if the value is a string and valid base64, decode it
        if isinstance(v, str) and is_base64(v):
            d[k] = base64.b64decode(v)
        # if it's a dictionary, recurse
        elif isinstance(v, dict):
            d[k] = decode_dict(v)
        # if it's a list, iterate through items
        elif isinstance(v, list):
            d[k] = [
                decode_dict(item)
                if isinstance(item, dict)
                else base64.b64decode(item)
                if isinstance(item, str) and is_base64(item)
                else item
                for item in v
            ]
    return d


def deserialize_dict(json_str):
    """
    Deserialize the JSON string to a dictionary after decoding base64 strings
    """
    decoded_dict = decode_dict(json.loads(json_str))
    return decoded_dict


# Example usage:
serialized_dict = """
{
    "index": 21,
    "hash": 99165210027860523417610286477615221696832788962937574144033165071675932234664,
    "data_chunk": "7768617473207570",
    "other_data": {
        "nested_bytes": "AAEC"
    },
    "list_of_bytes": ["b25l", "dHdv", "dGhyZWU="]
}
"""

deserialized_dict = deserialize_dict(serialized_dict)
print(deserialized_dict)

# Example usage:
my_dict = {
    "index": 21,
    "hash": 99165210027860523417610286477615221696832788962937574144033165071675932234664,
    "data_chunk": b"\xef\xbb\xbfThe quick brown fox jumps over the lazy dog",
    "other_data": {
        "nested_bytes": b"\x00\x01\x02",
    },
    "list_of_bytes": [b"one", b"two", b"three"],
}

serialized_dict = serialize_dict(my_dict)
print(serialized_dict)
