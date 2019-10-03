#!/bin/sh

clean_exit(){
    rm internet_communication
    printf "An error happened, exiting script \n"
    exit
}

ALICE_SECRET=0xF42F42F42F42F
BOB_SECRET=0xFF9876543210FF
KEY_SIZE=128

VERBOSE="--verbose"

### USECASE ###

printf "A communcation channel is opened between Alice and Bob\n"
mkfifo internet_communication
printf "\n"

### ALICE SIDE ###

printf "Alice generates her public key\n"
if ! python3 keygen.py --mode generate --secret $ALICE_SECRET --output keys/alice_pubkey.txt $VERBOSE
    then clean_exit
fi
printf "\n"

printf "Alice sends her public key to Bob on the Internet\n"
cat keys/alice_pubkey.txt > internet_communication &
printf "\n"

printf "Bob receives it and generates the shared private key between them\n"
if ! python3 keygen.py --mode merge --secret $BOB_SECRET --public "$(cat internet_communication)" --output keys/bob_shared_key.txt $VERBOSE
    then clean_exit
fi
printf "\n"

printf "Bob generates a more secure shared key by using the previously computed key as as seed to a CSPRNG\n"
if ! python3 bbs.py --seed "$(cat keys/bob_shared_key.txt)" --size $KEY_SIZE --output keys/bob_encrypted_shared_key.key $VERBOSE
    then clean_exit
fi
printf "\n"

### BOB SIDE ###

printf "Bob generates his public key\n"
if ! python3 keygen.py --mode generate --secret $BOB_SECRET --output keys/bob_pubkey.txt $VERBOSE
    then clean_exit
fi
printf "\n"

printf "Bob sends his public key to Alice on the Internet\n"
cat keys/bob_pubkey.txt > internet_communication &
printf "\n"

printf "Alice receives it and generates the shared private key between them\n"
if ! python3 keygen.py --mode merge --secret $ALICE_SECRET --public "$(cat internet_communication)" --output keys/alice_shared_key.txt $VERBOSE
    then clean_exit
fi
printf "\n"

printf "Alice generates a more secure shared key by using the previously computed key as as seed to a CSPRNG\n"
if ! python3 bbs.py --seed "$(cat keys/alice_shared_key.txt)" --size $KEY_SIZE --output keys/alice_encrypted_shared_key.key $VERBOSE
    then clean_exit
fi
printf "\n"

### TEST ###

printf "We verify that both party's shared private keys are identical\n"
printf "Number of differences between both keys: "
if ! diff keys/alice_encrypted_shared_key.key keys/bob_encrypted_shared_key.key | wc -l 
    then clean_exit
fi
printf "\n"

### COMMUNICATION USING A SYMMETRIC CIPHER ###

printf "Alice encrypts her secret file with AES and her private key\n"
if ! python3 cipher.py --mode encrypt --key "$(cat keys/alice_encrypted_shared_key.key)" --input files/really_secret_file.txt --output files/encrypted_file.bin $VERBOSE
    then clean_exit
fi
printf "\n"

printf "She sends her encrypted file to Bob on the Internet\n"
cat files/encrypted_file.bin > internet_communication &
printf "\n"

printf "Bob recieves it and decrypts it using the same cipher with his private key\n"
if ! python3 cipher.py --mode decrypt --key "$(cat keys/bob_encrypted_shared_key.key)" --input internet_communication --output files/decrypted_file.txt $VERBOSE
    then clean_exit
fi
printf "\n"

### TEST ###
printf "We verify that the original message and the decrypted one are identical\n"
printf "Number of differences between both files: "
if ! diff files/really_secret_file.txt files/decrypted_file.txt | wc -l 
    then clean_exit
fi
printf "Which means that Dec(Enc(input, key), key) == input\n\n"

printf "The communication channel is closed\n"
rm internet_communication