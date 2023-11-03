# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 philanthrope

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Bittensor Miner Template:

# Step 1: Import necessary libraries and modules
import os
import time
import redis
import typing
import base64
import argparse
import traceback
import bittensor as bt
from collections import defaultdict

# import this repo
import storage
from storage.utils import (
    hash_data,
    chunk_data,
    MerkleTree,
    ECCommitment,
    ecc_point_to_hex,
    hex_to_ecc_point,
    serialize_dict_with_bytes,
    deserialize_dict_with_bytes,
)


def get_config():
    # Step 2: Set up the configuration parser
    # This function initializes the necessary command-line arguments.
    # Using command-line arguments allows users to customize various miner settings.
    parser = argparse.ArgumentParser()
    # TODO(developer): Adds your custom miner arguments to the parser.
    parser.add_argument(
        "--custom", default="my_custom_value", help="Adds a custom value to the parser."
    )
    # Adds override arguments for network and netuid.
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Adds axon specific arguments i.e. --axon.port ...
    bt.axon.add_args(parser)
    # Activating the parser to read any command-line inputs.
    # To print help message, run python3 template/miner.py --help
    config = bt.config(parser)

    # Step 3: Set up logging directory
    # Logging captures events for diagnosis or understanding miner's behavior.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "miner",
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config


def commit_data(committer, data_chunks):
    merkle_tree = MerkleTree()
    commitments = defaultdict(lambda: [None] * len(data_chunks))

    # Commit each chunk of data
    for index, chunk in enumerate(data_chunks):
        c, m_val, r = committer.commit(chunk)
        commitments[index] = {
            "index": index,
            "hash": m_val,
            "data_chunk": chunk,
            "point": c,
            "randomness": r,
            "merkle_proof": None,
        }
        merkle_tree.add_leaf(ecc_point_to_hex(c))

    # Create the tree from the leaves
    merkle_tree.make_tree()

    # Get the merkle proof for each commitment
    for index, commitment in commitments.items():
        merkle_proof = merkle_tree.get_proof(index)
        commitments[index]["merkle_proof"] = merkle_proof

    return commitments, merkle_tree.get_merkle_root()


# Recommit data and send back to validator (miner side)
def recommit_data(committer, challenge_indices, merkle_tree, data):
    # TODO: Store the g,h values in the database so we can retrieve them later
    # new_commitments = {}
    new_commitments = []
    for i in challenge_indices:
        c, m_val, r = committer.commit(data[i])
        commitment_hash = hash_data(
            ecc_point_to_hex(c)
        )  # Assuming a hash_function is available.
        new_commitments.append(
            {
                "index": i,
                "hash": m_val,  # commitment_hash,
                "data_chunk": data[i],
                "point": c,
                "randomness": r,
                "merkle_proof": None,
            }
        )
        merkle_tree.update_leaf(i, ecc_point_to_hex(c))
    new_merkle_root = merkle_tree.get_merkle_root()
    return new_merkle_root, new_commitments


# Main takes the config and starts the miner.
def main(config):
    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

    # Step 4: Initialize Bittensor miner objects
    # These classes are vital to interact and function within the Bittensor network.
    bt.logging.info("Setting up bittensor objects.")

    # Wallet holds cryptographic information, ensuring secure transactions and communication.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    # metagraph provides the network's current state, holding state about other participants in a subnet.
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour miner: {wallet} is not registered to chain connection: {subtensor} \nRun btcli register and try again. "
        )
        exit()

    bt.logging.info(
        f"Running miner for subnet: {config.netuid} on network: {subtensor.chain_endpoint} with config:"
    )

    # Each miner gets a unique identity (UID) in the network for differentiation.
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"Running miner on uid: {my_subnet_uid}")

    # Set up the miner's data storage.
    # This is where the miner will store the data it receives.
    database = redis.StrictRedis(host="localhost", port=6379, db=0)

    # Step 5: Set up miner functionalities
    # The following functions control the miner's response to incoming requests.
    # The blacklist function decides if a request should be ignored.
    def blacklist_fn(
        synapse: typing.Union[storage.protocol.Store, storage.protocol.Challenge]
    ) -> typing.Tuple[bool, str]:
        # TODO(developer): Define how miners should blacklist requests. This Function
        # Runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        # The synapse is instead contructed via the headers of the request. It is important to blacklist
        # requests before they are deserialized to avoid wasting resources on requests that will be ignored.
        # Below: Check that the hotkey is a registered entity in the metagraph.
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"
        # TODO(developer): In practice it would be wise to blacklist requests from entities that
        # are not validators, or do not have enough stake. This can be checked via metagraph.S
        # and metagraph.validator_permit. You can always attain the uid of the sender via a
        # metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.
        # Otherwise, allow the request to be processed further.
        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    # The priority function determines the order in which requests are handled.
    # More valuable or higher-priority requests are processed before others.
    def priority_fn(
        synapse: typing.Union[storage.protocol.Store, storage.protocol.Challenge]
    ) -> float:
        # TODO(developer): Define how miners should prioritize requests.
        # Miners may recieve messages from multiple entities at once. This function
        # determines which request should be processed first. Higher values indicate
        # that the request should be processed first. Lower values indicate that the
        # request should be processed later.
        # Below: simple logic, prioritize requests from entities with more stake.
        caller_uid = metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(metagraph.S[caller_uid])  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def store(synapse: storage.protocol.Store) -> storage.protocol.Store:
        # Chunk the data according to the specified (random) chunk size
        encrypted_data_bytes = base64.b64decode(synapse.encrypted_data)
        data_chunks = chunk_data(encrypted_data_bytes, synapse.chunk_size)

        # Extract setup params
        g = hex_to_ecc_point(synapse.g, synapse.curve)
        h = hex_to_ecc_point(synapse.h, synapse.curve)

        # Commit the data chunks based on the provided curve points
        committer = ECCommitment(g, h)
        commitments, merkle_root = commit_data(committer, data_chunks)

        # Store commitments in local storage indexed by the data hash
        commitments["setup_params"] = {
            "g": synapse.g,
            "h": synapse.h,
            "curve": synapse.curve,
        }
        serialized_commitments = serialize_dict_with_bytes(commitments)
        database.set(synapse.data_hash, serialized_commitments)

        # Do not send randomness values to the validator until challenged
        for commitment in commitments:
            del commitment["randomness"]

        # Encode base64 so we can send less data over the wire
        serialized_commitments_return = base64.b64encode(
            serialize_dict_with_bytes(commitments).encode()
        ).decode("utf-8")

        # Return the commitments and merkle root
        synapse.commitments = serialized_commitments_return
        synapse.merkle_root = merkle_root

        return synapse

    def challenge(synapse: storage.protocol.Challenge) -> storage.protocol.Challenge:
        # Retrieve commitments from local storage
        commitments = database.get(synapse.data_hash)
        if commitments is None:
            bt.logging.error(
                f"Commitments not found for data hash: {synapse.data_hash}"
            )
            return synapse

        # Recommit data and send back to validator (miner side)
        committer = ECCommitment(synapse.g, synapse.h)

        return synapse

    # Step 6: Build and link miner functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    axon = bt.axon(wallet=wallet, config=config)
    bt.logging.info(f"Axon {axon}")

    # Attach determiners which functions are called when servicing a request.
    bt.logging.info(f"Attaching forward function to axon.")
    axon.attach(
        forward_fn=dummy,
        blacklist_fn=blacklist_fn,
        priority_fn=priority_fn,
    )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(
        f"Serving axon {dummy} on network: {config.subtensor.chain_endpoint} with netuid: {config.netuid}"
    )
    axon.serve(netuid=config.netuid, subtensor=subtensor)

    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    # Step 7: Keep the miner alive
    # This loop maintains the miner's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    step = 0
    while True:
        try:
            # TODO(developer): Define any additional operations to be performed by the miner.
            # Below: Periodically update our knowledge of the network graph.
            if step % 5 == 0:
                metagraph = subtensor.metagraph(config.netuid)
                log = (
                    f"Step:{step} | "
                    f"Block:{metagraph.block.item()} | "
                    f"Stake:{metagraph.S[my_subnet_uid]} | "
                    f"Rank:{metagraph.R[my_subnet_uid]} | "
                    f"Trust:{metagraph.T[my_subnet_uid]} | "
                    f"Consensus:{metagraph.C[my_subnet_uid] } | "
                    f"Incentive:{metagraph.I[my_subnet_uid]} | "
                    f"Emission:{metagraph.E[my_subnet_uid]}"
                )
                bt.logging.info(log)
            step += 1
            time.sleep(1)

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            continue


# This is the main function, which runs the miner.
if __name__ == "__main__":
    main(get_config())
