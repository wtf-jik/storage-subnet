import bittensor as bt

sub = bt.subtensor()

wallet = bt.wallet(name="subnet", hotkey="miner")
wallet.coldkey

netuid = 21
parameter = "max_weight_limit"
value = 455

sub.set_hyperparameter(
    netuid=netuid,
    parameter=parameter,
    value=value,
    wallet=wallet,
)
