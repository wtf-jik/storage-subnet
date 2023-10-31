import hashlib
from pprint import pformat
from random import sample


class MerkleTree:
    def __init__(self):
        self.leaves = list()
        self.levels = None

    @staticmethod
    def hash(data):
        return hashlib.sha256(data).digest()

    def add_leaf(self, value):
        value = value.encode("utf-8") if isinstance(value, str) else value
        self.leaves.append(value)

    def _calculate_next_level(self):
        solo_leaf = None
        N = len(self.levels[0])
        if N % 2 == 1:
            solo_leaf = self.levels[0][-1]
            N -= 1

        new_level = []
        for l, r in zip(self.levels[0][0:N:2], self.levels[0][1:N:2]):
            new_level.append(self.hash(l + r))
        if solo_leaf:
            new_level.append(solo_leaf)
        self.levels = [new_level] + self.levels

    def build_tree(self):
        self.levels = [self.leaves]
        while len(self.levels[0]) > 1:
            self._calculate_next_level()

    def get_merkle_root(self):
        return self.levels[0][0] if self.levels else None

    def get_proof(self, index):
        if not self.levels:
            return None

        proof = []
        for x in range(len(self.levels) - 1, 0, -1):
            level_len = len(self.levels[x])
            if index % 2 == 0 and index < level_len - 1:
                proof.append(self.levels[x][index + 1])
            elif index % 2 == 1:
                proof.append(self.levels[x][index - 1])
            index //= 2
        return proof


class Prover:
    def __init__(self, data_blocks):
        self.data_blocks = data_blocks
        self.tree = MerkleTree()
        for block in data_blocks:
            self.tree.add_leaf(block)
        self.tree.build_tree()

    def get_merkle_root(self):
        return self.tree.get_merkle_root()

    def prove(self, indices):
        proofs = []
        for i in indices:
            proofs.append((self.data_blocks[i], self.tree.get_proof(i)))
        return proofs


class Verifier:
    def __init__(self, merkle_root):
        self.merkle_root = merkle_root

    def challenge(self, num_blocks):
        return sample(range(num_blocks), num_blocks)

    def verify(self, block, proof):
        current = block
        for sibling in proof:
            current = MerkleTree.hash(min(current, sibling) + max(current, sibling))
        return current == self.merkle_root


data_blocks = ["Hello, world!", "This is a test.", "Some bytes", "key-value"]

p = Prover(data_blocks)
merkle_root = p.get_merkle_root()
print(f"Merkle Root: {merkle_root.hex()}")

v = Verifier(merkle_root)
challenge_indices = v.challenge(len(data_blocks))
print(f"Challenged indices: {challenge_indices}")

responses = p.prove(challenge_indices)
results = [v.verify(*response) for response in responses]
print(f"Verification results: {results}")

assert all(results), "Verification failed for some blocks!"
