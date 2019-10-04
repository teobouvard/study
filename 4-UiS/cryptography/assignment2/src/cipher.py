import argparse
import os 

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# get the script directory to access test data 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

### AUTO BASE INTEGERS ###

def auto_int(x):
    return int(x, 0)

### CIPHER METHODS ###

def encrypt(plaintext, key):
    cipher = AES.new(key, AES.MODE_ECB)
    if len(plaintext) % 16 != 0:
        plaintext = pad(plaintext, 16)
    ciphertext = cipher.encrypt(plaintext)
    return ciphertext

def decrypt(ciphertext, key):
    cipher = AES.new(key, AES.MODE_ECB)
    plaintext = cipher.decrypt(ciphertext)
    if len(plaintext) % 16 != 0:
        plaintext = unpad(plaintext, 16)
    return plaintext

### DISPLAY METHODS ###

def display(key, plaintext, ciphertext):
    print('Key length : {} bits'.format(len(key)*8))
    print('Key : 0x{}\n\n'.format(key.hex()))
    print('Plaintext : {}\n\n'.format(plaintext))
    print('Ciphertext : {}'.format(ciphertext))


### MAIN PROGRAM ###

def argument_parser():
    parser = argparse.ArgumentParser(description='Encrypt and decrypt data using AES')
    parser.add_argument('--mode', choices=['encrypt', 'decrypt', 'test'], required=True, help='Encrypt data, decrypt data, or run the tests')
    parser.add_argument('--key', type=auto_int, help='The key used for encryption or decryption')
    parser.add_argument('--input', type=str, help='Path to the file to encrypt or decrypt')
    parser.add_argument('--output', type=str, help='Path to wich the encrypted or decrypted data is written. If not specified, output is redirected to stdout')
    parser.add_argument('--verbose', action='store_true', help='Run in verbose mode')


    return parser

if __name__ == '__main__':

    # parse program arguments
    parser = argument_parser()
    args = parser.parse_args()

    mode = args.mode
    key = args.key.to_bytes(int(args.key.bit_length()/8), 'big') if args.key is not None else None
    input_file = args.input
    output_file = args.output
    verbose = args.verbose

    # arguments check
    if key is None and mode != 'test':
        print('--key argument is required when not in --mode test')
        exit()
    if input is None and mode != 'test':
        print('--input argument is required when not in --mode test')
        exit()
    if mode != 'test' and 8*len(key) not in [128, 192, 256]:
        print('Key length must be 128, 192 or 256 bits but is {} bits'.format(len(key)))
        exit(1)
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if mode == 'encrypt':
        with open(input_file, 'rb') as f:
            plaintext = f.read()
            ciphertext = encrypt(plaintext, key)

        # display parameters if program is run in verbose mode or if ciphertext is not written to disk
        if output_file is None or verbose:
            display(key, plaintext, ciphertext)

        if output_file is not None:
            with open(output_file, 'wb') as f:
                f.write(ciphertext)
                print('Encrypted file written to {}'.format(output_file))
    
    elif mode == 'decrypt':
        with open(input_file, 'rb') as f:
            ciphertext = f.read()
            plaintext = decrypt(ciphertext, key)

        # display parameters if program is run in verbose mode or if ciphertext is not written to disk
        if output_file is None or verbose:
            display(key, plaintext, ciphertext)

        if output_file is not None:
            with open(output_file, 'wb') as f:
                f.write(plaintext)
                print('Decrypted file written to {}'.format(output_file))
            

    elif mode == 'test':

        # different key length tests
        with open(SCRIPT_DIR + '/files/AES test data/128_key.txt', 'r') as f:
            key = ''.join(f.read().split())
            key_128 = bytes.fromhex(key)
        with open(SCRIPT_DIR + '/files/AES test data/192_key.txt', 'r') as f:
            key = ''.join(f.read().split())
            key_192 = bytes.fromhex(key)
        with open(SCRIPT_DIR + '/files/AES test data/256_key.txt', 'r') as f:
            key = ''.join(f.read().split())
            key_256 = bytes.fromhex(key)

        # same plaintext for all keys
        with open(SCRIPT_DIR + '/files/AES test data/plaintext.txt', 'r') as f:
            plaintext = ''.join(f.read().split())
            plaintext = bytes.fromhex(plaintext)

        # ciphertexts for each key
        with open(SCRIPT_DIR + '/files/AES test data/128_ciphertext.txt', 'r') as f:
            ciphertext = ''.join(f.read().split())
            ciphertext_128 = bytes.fromhex(ciphertext)
        with open(SCRIPT_DIR + '/files/AES test data/192_ciphertext.txt', 'r') as f:
            ciphertext = ''.join(f.read().split())
            ciphertext_192 = bytes.fromhex(ciphertext)
        with open(SCRIPT_DIR + '/files/AES test data/256_ciphertext.txt', 'r') as f:
            ciphertext = ''.join(f.read().split())
            ciphertext_256 = bytes.fromhex(ciphertext)
        
        for key, ciphertext in zip([key_128, key_192, key_256], [ciphertext_128, ciphertext_192, ciphertext_256]):
            test_ciphertext = encrypt(plaintext, key)
            test_plaintext = decrypt(ciphertext, key)

            if test_ciphertext == ciphertext:
                print('{} bit key AES encryption OK'.format(8*len(key)))
            else:
                print('{} bit key AES encryption FAILED'.format(8*len(key)))

            if test_plaintext == plaintext:
                print('{} bit key AES decryption OK'.format(8*len(key)))
            else:
                print('{} bit key AES decryption FAILED'.format(8*len(key)))