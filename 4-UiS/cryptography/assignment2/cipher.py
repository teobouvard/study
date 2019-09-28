import argparse

from Crypto.Cipher import AES


def encrypt(plaintext, key):
    cipher = AES.new(key, AES.MODE_EAX)
    plaintext = bytes(plaintext, encoding='utf8')
    ciphertext = cipher.encrypt(plaintext)
    return ciphertext

def decrypt(ciphertext, key):
    cipher = AES.new(key, AES.MODE_EAX)
    plaintext = cipher.decrypt(ciphertext)
    print(plaintext)
    return plaintext

### MAIN PROGRAM ###

def argument_parser():
    parser = argparse.ArgumentParser(description='Encrypt and decrypt data using AES')
    parser.add_argument('--mode', choices=['encrypt', 'decrypt'], required=True, help='Encrypt or decrypt data')
    parser.add_argument('--key', type=int, required=True, help='The key used for encryption or decryption')
    parser.add_argument('--input', type=str, required=True, help='Path to the file to encrypt or decrypt')
    parser.add_argument('--output', type=str, help='Path to wich the encrypted or decrypted data is written. If not specified, output is redirected to stdout')

    return parser

if __name__ == '__main__':

    # parse program arguments
    parser = argument_parser()
    args = parser.parse_args()

    mode = args.mode
    key = args.key.to_bytes(16, 'big')
    input_file = args.input
    output_file = args.output

    if mode == 'encrypt':
        with open(input_file, 'r') as f:
            plaintext = f.read()
            ciphertext = encrypt(plaintext, key)
        
        if output_file is not None:
            with open(output_file, 'wb') as f:
                f.write(ciphertext)
        else:
            print(ciphertext)
    
    elif mode == 'decrypt':
        with open(input_file, 'rb') as f:
            ciphertext = f.read()
            plaintext = decrypt(ciphertext, key)

        if output_file is not None:
            with open(output_file, 'wb') as f:
                f.write(plaintext)
        else:
            print(plaintext)

            #ECB -> pad + unpad, or send iv & nonces
