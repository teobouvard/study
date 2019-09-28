#!/bin/bash

ALICE_SECRET=42
BOB_SECRET=231

# we open a communcation channel between Alice and Bob
mkfifo internet_communication

### ALICE SIDE ###

printf "Alice generates her public key\n"
python3 keygen.py --mode generate --secret $ALICE_SECRET --output keys/alice_pubkey.txt
printf "\n"

printf "Alice sends her public key to Bob on the internet\n"
cat keys/alice_pubkey.txt > internet_communication &
printf "\n"

printf "Bob receives it and generates the shared private key between them\n"
python3 keygen.py --mode merge --secret $BOB_SECRET --public "$(cat internet_communication)" --output keys/bob_shared_key.txt
printf "\n"

printf "Bob generates a more secure shared key by using the previously computed key as as seed to a CSPRNG\n"
python3 bbs.py --seed "$(cat keys/bob_shared_key.txt)" --output keys/bob_encrypted_shared_key.key
printf "\n"

### BOB SIDE ###

printf "Bob generates his public key\n"
python3 keygen.py --mode generate --secret $BOB_SECRET --output keys/bob_pubkey.txt
printf "\n"

printf "Bob sends his public key to Alice on the internet\n"
cat keys/bob_pubkey.txt > internet_communication &
printf "\n"

printf "Alice receives it and generates the shared private key between them\n"
python3 keygen.py --mode merge --secret $ALICE_SECRET --public "$(cat internet_communication)" --output keys/alice_shared_key.txt
printf "\n"

printf "Alice generates a more secure shared key by using the previously computed key as as seed to a CSPRNG\n"
python3 bbs.py --seed "$(cat keys/alice_shared_key.txt)" --output keys/alice_encrypted_shared_key.key
printf "\n"

printf "We verify that both private shared keys are identical\n"
printf "Number of differences between both keys: "
diff keys/alice_encrypted_shared_key.key keys/bob_encrypted_shared_key.key | wc -l 
printf "\n"


python3 cipher.py --mode encrypt --key "$(cat keys/alice_encrypted_shared_key.key)" --input files/really_secret_file.txt --output files/encrypted_file.bin

python3 cipher.py --mode decrypt --key "$(cat keys/alice_encrypted_shared_key.key)" --input files/encrypted_file.bin --output files/decrypted_file.txt

# the communication channel is closed
rm internet_communication