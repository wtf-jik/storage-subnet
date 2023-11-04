import redis

r = redis.StrictRedis(host="localhost", port=6379, db=0)


def insert_data(hash_key, data):
    # 'data' should be in bytes
    r.set(hash_key, data)
    print(f"Data inserted for hash: {hash_key}")


def query_data(hash_key):
    data = r.get(hash_key)
    if data:
        # 'data' is already in bytes, no need to decode
        return data
    else:
        print("Data not found for given hash")
        return None


# Example usage:
# insert_data('somehashkey', b'encrypted_bytes_here')
# data = query_data('somehashkey')
# print(data)

# Fetch all keys
all_keys = r.keys("*")

# Print all keys
for key in all_keys:
    print(key)

# All keys is expensive, try batched scan instead
# Initialize the scan cursor and match pattern
cursor = "0"
match = "*"

while cursor != 0:
    cursor, keys = r.scan(cursor=cursor, match=match)
    for key in keys:
        print(key)
