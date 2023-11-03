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


class Setup(bt.Synapse):
    curve: str  # e.g. P-256
    g: typing.Union[
        str, Crypto.PublicKey.ECC.EccPoint
    ]  # base point   (or hex string representation)
    h: typing.Union[
        str, Crypto.PublicKey.ECC.EccPoint
    ]  # random point (or hex string representation)


class Store(bt.Synapse):
    # Receieves
    encrypted_data: bytes  # raw bytes of encrypted data
    data_hash: str  # the hash of the encrypted data
    chunk_size: int  # bytes (e.g. 1024) for how big the chunks should be
    size: typing.Optional[int]  # bytes (e.g. 9234) size of full data block

    # Returns
    commitments: typing.Opional[
        typing.Dict[str, typing.List[typing.Union[str, Crypto.PublicKey.ECC.EccPoint]]]
    ] = None  # the commitment to the data
    merkle_root: typing.Optional[str] = None  # the merkle root of the data


class Challenge(bt.Synapse):
    # Receives
    challenge_indices: typing.List[int]  # list of block indices to challenge

    # Returns
    responses: typing.Optional[
        typing.List[
            typing.Dict[
                str,  # index, commitment, data_chunk, random_value, merkle_proof
                typing.Union[
                    int,  # index, random_value
                    str,  # hex point representation
                    Crypto.PublicKey.ECC.EccPoint,  # point
                    bytes,  # data chunk
                    typing.List[typing.Tuple[str, str]],  # merkle proof
                ],
            ]
        ]
    ] = None
