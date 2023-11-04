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

# Bittensor Validator Template:

# Step 1: Import necessary libraries and modules
import os
import sys
import time
import redis
import torch
import base64
import argparse
import traceback
import bittensor as bt
from random import choice
from Crypto.Random import get_random_bytes

# import this repo
from storage import protocol
from storage.utils import (
    hash_data,
    setup_CRS,
    chunk_data,
    MerkleTree,
    encrypt_data,
    ECCommitment,
    make_random_file,
    get_random_chunksize,
    ecc_point_to_hex,
    hex_to_ecc_point,
    serialize_dict_with_bytes,
    deserialize_dict_with_bytes,
    decode_storage,
)


# TODO:
def Challenge(database):
    keys = database.keys("*")
    key_ = choice(keys)
    data = database.get(key_)
    data = decode_storage(data)
    commitments = data["commitments"]
    merkle_root = data["merkle_root"]
    params = data["params"]
    index = choice(range(len(commitments)))
    commitment = commitments[index]

    # We want to get back from the miner:
    # - the data chunk itself (to prove they still have it)
    # - the random value to open the commitment
    # - the merkle proof for the data chunk

    # TODO: maybe we additionally want a random challenge here on the original data?
    # Perhaps verify they signed the data chunk with their wallet? (e.g. dendrite signature and axon verify?)
    # Or a (data_chunk + <random_value|miner_key>) signature?
    # This way we have 3 layers of security:
    # - (1) the commitment + random value + original data to prove they have the data now
    # - (2) the merkle proof to prove they stored it originally
    # - (3) the data chunk signature to verify they also have the data (redundant with 1?)


def StoreRandomData():
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

    # TODO: select subset of miners to query (e.g. redunancy factor of N)
    # Broadcast a query to all miners on the network.
    responses = dendrite.query(
        metagraph.axons,
        synapse,
        deserialize=True,
    )

    # Log the results for monitoring purposes.
    bt.logging.info(f"Received responses: {responses}")

    # TODO: Store data params in GUNdb instead of Redis
    setup_params = {"g": g, "h": h, "curve": config.curve}
    setup_params_base64 = base64.b64encode(
        serialize_dict_with_bytes([setup_params]).encode()
    ).decode("utf-8")

    for response in responses:
        # TODO: come up with better key mapping (merkle root based?)
        key = f"{response.data_hash}.{response.axon.hotkey}"  # or UUID?
        # Package up the required data into a dict
        response_storage = {
            "commitments": response.commitments,
            "merkle_root": response.merkle_root,
            "params": setup_params_base64,
        }
        # encode the response_storage dict as base64
        response_storage_encoded = base64.b64encode(
            json.dumps(response_storage).encode()
        ).decode("utf-8")
        # Store in the database according to the data hash and the miner hotkey
        database.set(key, response_storage)


# Step 2: Set up the configuration parser
# This function is responsible for setting up and parsing command-line arguments.
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha", default=0.9, type=float, help="The weight moving average scoring."
    )
    # TODO(developer): Adds your custom validator arguments to the parser.
    parser.add_argument(
        "--curve",
        default="P-256",
        help="Curve for elliptic curve cryptography.",
        choices=["P-256"],  # TODO: expand this list
    )
    parser.add_argument(
        "--maxsize",
        default=128,
        type=int,
        help="Maximum size of random data to store.",
    )
    parser.add_argument(
        "--redundancy",
        type=int,
        default=3,
        help="Number of miners to store each piece of data on.",
    )
    # Adds override arguments for network and netuid.
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Parse the config (will take command-line arguments if provided)
    # To print help message, run python3 template/validator.py --help
    config = bt.config(parser)

    # Step 3: Set up logging directory
    # Logging is crucial for monitoring and debugging purposes.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "validator",
        )
    )
    # Ensure the logging directory exists.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)

    # Return the parsed config.
    return config


def main(config):
    # Set up logging with the provided configuration and directory.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:"
    )
    # Log the configuration for reference.
    bt.logging.info(config)

    # Step 4: Build Bittensor validator objects
    # These are core Bittensor classes to interact with the network.
    bt.logging.info("Setting up bittensor objects.")

    # The wallet holds the cryptographic key pairs for the validator.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # The subtensor is our connection to the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    # Dendrite is the RPC client; it lets us send messages to other nodes (axons) in the network.
    dendrite = bt.dendrite(wallet=wallet)
    bt.logging.info(f"Dendrite: {dendrite}")

    # The metagraph holds the state of the network, letting us know about other validators and miners.
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    # Step 5: Connect the validator to the network
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour validator: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again."
        )
        exit()

    # Each validator gets a unique identity (UID) in the network for differentiation.
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

    # Step 6: Set up initial scoring weights for validation
    bt.logging.info("Building validation weights.")
    scores = torch.ones_like(metagraph.S, dtype=torch.float32)
    bt.logging.info(f"Weights: {scores}")
    # Step 7: The Main Validation Loop
    bt.logging.info("Starting validator loop.")
    step = 0
    while True:
        try:
            # Setup CRS for this round of validation
            g, h = setup_CRS(curve=config.curve)

            # Make a random bytes file to test the miner
            random_data = make_random_file(maxsize=config.maxsize)

            # Encrypt the data
            encrypted_data, nonce, tag = encrypt_data(
                random_data,
                wallet.hotkey.secret,  # TODO: Use validator key as the encryption key?
            )

            # Convert to base64 for compactness
            b64_encrypted_data = base64.b64encode(encrypted_data).decode("utf-8")

            # Hash the encrypted data
            data_hash = hash_data(encrypted_data)

            # Chunk the data
            chunksize = get_random_chunksize()
            # chunks = list(chunk_data(encrypted_data, chunksize))

            # Broadcast a query to all miners on the network.
            responses = dendrite.query(
                metagraph.axons,
                protocol.Store(
                    chunksize=chunksize,
                    encrypted_data=b64_encrypted_data,
                    data_hash=data_hash,
                    curve=config.curve,
                    g=ecc_point_to_hex(g),
                    h=ecc_point_to_hex(h),
                    size=sys.getsizeof(encrypted_data),
                ),
                deserialize=True,
            )

            # TODO: Store data params in Redis or GUNdb
            setup_params = {
                "g": g,
                "h": h,
                "curve": config.curve,
            }

            # Log the results for monitoring purposes.
            bt.logging.info(f"Received responses: {responses}")

            # TODO(developer): Define how the validator scores responses.
            # Adjust the scores based on responses from miners.
            for i, resp_i in enumerate(responses):
                # Check if the miner has provided the correct response by doubling the dummy input.
                # If correct, set their score for this round to 1. Otherwise, set it to 0.
                score = template.reward.dummy(step, resp_i)

                # Update the global score of the miner.
                # This score contributes to the miner's weight in the network.
                # A higher weight means that the miner has been consistently responding correctly.
                scores[i] = config.alpha * scores[i] + (1 - config.alpha) * score

            bt.logging.info(f"Scores: {scores}")
            # Periodically update the weights on the Bittensor blockchain.
            if (step + 1) % 10 == 0:
                # TODO(developer): Define how the validator normalizes scores before setting weights.
                weights = torch.nn.functional.normalize(scores, p=1.0, dim=0)
                bt.logging.info(f"Setting weights: {weights}")
                # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
                # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
                result = subtensor.set_weights(
                    netuid=config.netuid,  # Subnet to set weights on.
                    wallet=wallet,  # Wallet to sign set weights using hotkey.
                    uids=metagraph.uids,  # Uids of the miners to set weights for.
                    weights=weights,  # Weights to set for the miners.
                    wait_for_inclusion=True,
                )
                if result:
                    bt.logging.success("Successfully set weights.")
                else:
                    bt.logging.error("Failed to set weights.")

            # End the current step and prepare for the next iteration.
            step += 1
            # Resync our local state with the latest state from the blockchain.
            metagraph = subtensor.metagraph(config.netuid)
            # Sleep for a duration equivalent to the block time (i.e., time between successive blocks).
            time.sleep(bt.__blocktime__)

        # If we encounter an unexpected error, log it for debugging.
        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()

        # If the user interrupts the program, gracefully exit.
        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting validator.")
            exit()


# # The main function parses the configuration and runs the validator.
# if __name__ == "__main__":
#     # Parse the configuration.
#     config = get_config()
#     # Run the main function.
#     main(config)
