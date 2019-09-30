import argparse
import os 

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

### AUTO BASE INTEGERS ###

def auto_int(x):
    return int(x, 0)

### CIPHER METHODS ###

def encrypt(plaintext, key):
    cipher = AES.new(key, AES.MODE_ECB)
    plaintext = pad(plaintext, len(key))
    ciphertext = cipher.encrypt(plaintext)
    return ciphertext

def decrypt(ciphertext, key):
    cipher = AES.new(key, AES.MODE_ECB)
    plaintext = cipher.decrypt(ciphertext)
    return unpad(plaintext, len(key))

### DISPLAY METHODS ###

def display(key, plaintext, ciphertext):
    print('Key length : {} bits'.format(len(key)*8))
    print('Key : {}\n\n'.format(key))
    print('Plaintext : {}\n\n'.format(plaintext))
    print('Ciphertext : {}'.format(ciphertext))


### MAIN PROGRAM ###

def argument_parser():
    parser = argparse.ArgumentParser(description='Encrypt and decrypt data using AES')
    parser.add_argument('--mode', choices=['encrypt', 'decrypt'], required=True, help='Encrypt or decrypt data')
    parser.add_argument('--key', type=auto_int, required=True, help='The key used for encryption or decryption')
    parser.add_argument('--input', type=str, required=True, help='Path to the file to encrypt or decrypt')
    parser.add_argument('--output', type=str, help='Path to wich the encrypted or decrypted data is written. If not specified, output is redirected to stdout')
    parser.add_argument('--verbose', action='store_true', help='Run in verbose mode')


    return parser

if __name__ == '__main__':

    # parse program arguments
    parser = argument_parser()
    args = parser.parse_args()

    mode = args.mode
    key = args.key.to_bytes(int(args.key.bit_length()/8), 'big')
    input_file = args.input
    output_file = args.output
    verbose = args.verbose

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
