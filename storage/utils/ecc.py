import hashlib
from Crypto.Random import random
from Crypto.PublicKey import ECC


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


def ecc_point_to_hex(point):
    point_str = "{},{}".format(point.x, point.y)
    return binascii.hexlify(point_str.encode()).decode()


def hex_to_ecc_point(hex_str, curve):
    point_str = binascii.unhexlify(hex_str).decode()
    x, y = map(int, point_str.split(","))
    return ECC.EccPoint(x, y, curve=curve)


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
