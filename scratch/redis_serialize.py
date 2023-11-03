import redis
import json
import base64

# Establish a connection to Redis
r = redis.Redis(host="localhost", port=6379, db=0)


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


# Dictionary with byte values
my_dict = {
    "key1": b"some bytes",
    "key2": {"nestedKey": b"nested bytes"},
    "key3": [b"list", b"of", b"bytes"],
}

print("original dict:", my_dict)

# Serialize the dictionary
serialized_dict = serialize_dict_with_bytes(my_dict)
print("serialized_dict:", serialized_dict)
print(type(serialized_dict))

b64_encoded_dict = base64.b64encode(serialized_dict.encode()).decode("utf-8")
print("b64 enc:", base64.b64encode(serialized_dict.encode()).decode("utf-8"))
print("unencoded:", base64.b64decode(b64_encoded_dict).decode("utf-8"))

# Store in Redis
r.set("my_dict", serialized_dict)

# Retrieve from Redis
retrieved_serialized_dict = r.get("my_dict")
print("retrieved_seialized:", retrieved_serialized_dict)

# Deserialize the dictionary
retrieved_dict = deserialize_dict_with_bytes(retrieved_serialized_dict.decode("utf-8"))

print("retrieved_unseralized:", retrieved_dict)
