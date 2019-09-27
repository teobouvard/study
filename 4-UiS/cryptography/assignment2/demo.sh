#!/bin/bash

# this is a fake communcation channel between Alice and Bob
mkfifo internet_communication

# Alice generates her public key
python3 keygen.py --secret 42 --write --output keys/alice_pubkey.txt

# Bob generates his public key
python3 keygen.py --secret 231 --write --output keys/bob_pubkey.txt

# Alice sends her public key to bob
cat keys/alice_pubkey.txt > internet_communication &

# Bob receives it and generates the shared private key between 
# them by using it as a seed to the CSPRNG
python3 bbs.py --seed "$(cat internet_communication)"

# the communication channel is closed
rm internet_communication