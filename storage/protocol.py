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

import Crypto
import typing
import bittensor as bt

from Crypto.PublicKey import ECC


# Basically setup for a given piece of data
class Store(bt.Synapse):
    # TODO: write deserialize

    # Receieves
    encrypted_data: str  # base64 encoded string of encrypted data (bytes)
    data_hash: str  # the hash of the encrypted data
    chunk_size: int  # bytes (e.g. 1024) for how big the chunks should be
    n_chunks: int  # expected number of chunks
    size: typing.Optional[int]  # bytes (e.g. 9234) size of full data block

    # Setup parameters
    curve: str  # e.g. P-256
    g: str  # base point   (hex string representation)
    h: str  # random point (hex string representation)

    # Returns serialized commitments
    commitments: typing.Optional[
        str
    ] = None  # base64 encoded string of serialized commitments dict
    merkle_root: typing.Optional[str] = None  # the merkle root of the data


class Challenge(bt.Synapse):
    # Receives
    challenge_hash: str  # hash of the data to challenge
    challenge_index: int  # block indices to challenge

    # Returns
    # - commitment (point represented as hex string)
    # - data chunk (base64 encoded string of bytes)
    # - random value (int)
    # - merkle proof (List[Dict[<left|right>, hex strings])
    # - merkle root (hex string)
    commitment: typing.Optional[str] = None
    data_chunk: typing.Optional[str] = None
    random_value: typing.Optional[int] = None
    merkle_proof: typing.Optional[typing.List[typing.Dict[str, str]]] = None
    merkle_root: typing.Optional[str] = None

    # responses: typing.Optional[
    #     typing.List[
    #         typing.Dict[
    #             str,  # index, commitment, data_chunk, random_value, merkle_proof
    #             typing.Union[
    #                 int,  # index, random_value
    #                 str,  # hex point representation
    #                 bytes,  # data chunk
    #                 typing.List[typing.Tuple[str, str]],  # merkle proof
    #             ],
    #         ]
    #     ]
    # ] = None
