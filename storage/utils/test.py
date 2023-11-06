def GetSynapse(config):
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
    return synapse
