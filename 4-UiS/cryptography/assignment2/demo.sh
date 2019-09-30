#!/bin/bash

ALICE_SECRET=0xF42F42F42F42F
BOB_SECRET=0xFF9876543210FF
VERBOSE=""

printf "First, we can verify that the program works as expected by running the tests\n"
python3 keygen.py --mode "test"
printf "\n"


### USECASE ###

printf "A communcation channel is opened between Alice and Bob\n"
mkfifo internet_communication
printf "\n"

### ALICE SIDE ###

printf "Alice generates her public key\n"
python3 keygen.py --mode generate --secret $ALICE_SECRET --output keys/alice_pubkey.txt $VERBOSE
printf "\n"

printf "Alice sends her public key to Bob on the Internet\n"
cat keys/alice_pubkey.txt > internet_communication &
printf "\n"

printf "Bob receives it and generates the shared private key between them\n"
python3 keygen.py --mode merge --secret $BOB_SECRET --public "$(cat internet_communication)" --output keys/bob_shared_key.txt $VERBOSE
printf "\n"

printf "Bob generates a more secure shared key by using the previously computed key as as seed to a CSPRNG\n"
python3 bbs.py --seed "$(cat keys/bob_shared_key.txt)" --output keys/bob_encrypted_shared_key.key $VERBOSE
printf "\n"

### BOB SIDE ###

printf "Bob generates his public key\n"
python3 keygen.py --mode generate --secret $BOB_SECRET --output keys/bob_pubkey.txt $VERBOSE
printf "\n"

printf "Bob sends his public key to Alice on the Internet\n"
cat keys/bob_pubkey.txt > internet_communication &
printf "\n"

printf "Alice receives it and generates the shared private key between them\n"
python3 keygen.py --mode merge --secret $ALICE_SECRET --public "$(cat internet_communication)" --output keys/alice_shared_key.txt $VERBOSE
printf "\n"

printf "Alice generates a more secure shared key by using the previously computed key as as seed to a CSPRNG\n"
python3 bbs.py --seed "$(cat keys/alice_shared_key.txt)" --output keys/alice_encrypted_shared_key.key $VERBOSE
printf "\n"

### TEST ###

printf "We verify that both party's shared private keys are identical\n"
printf "Number of differences between both keys: "
diff keys/alice_encrypted_shared_key.key keys/bob_encrypted_shared_key.key | wc -l 
printf "\n"

### COMMUNICATION USING A SYMMETRIC CIPHER ###

printf "Alice encrypts her secret file with AES and her private key\n"
python3 cipher.py --mode encrypt --key "$(cat keys/alice_encrypted_shared_key.key)" --input files/really_secret_file.txt --output files/encrypted_file.bin $VERBOSE
printf "\n"

printf "She sends her encrypted file to Bob on the Internet\n"
cat files/encrypted_file.bin > internet_communication &
printf "\n"

printf "Bob recieves it and decrypts it using the same cipher with his private key\n"
python3 cipher.py --mode decrypt --key "$(cat keys/bob_encrypted_shared_key.key)" --input internet_communication --output files/decrypted_file.txt $VERBOSE
printf "\n"

### TEST ###
printf "We verify that the original message and the decrypted one are identical\n"
printf "Number of differences between both files: "
diff files/really_secret_file.txt files/decrypted_file.txt | wc -l 
printf "Which means that Dec(Enc(input, key), key) == input\n\n"

printf "The communication channel is closed\n"
rm internet_communication