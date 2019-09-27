import argparse
import math
import os

### DEFAULT VALUES ###

P = 1009
R = 263
SECRET = 42

### ARTIHMETIC HELPER FUNCTIONS ###

def is_prime(n):
    return not (n < 2 or any(n % x == 0 for x in range(2, int(n ** 0.5) + 1)))

def are_coprime(a, b):
    return math.gcd(a, b) == 1

def is_primitive_root(r, p):
    powers = [r ** i % p for i in range(p-1)]
    return len(set(powers)) == p-1


### DIFFIE HELLMANN KEY GENERATION ###

def pubkeygen(prime, root, secret):
    assert(is_prime(prime))
    assert(is_primitive_root(root, prime))

    return (root ** secret) % prime


### MAIN PROGRAM ###

def argument_parser():
    parser = argparse.ArgumentParser(description='Generate public keys with Diffie-Hellmann algorithm')
    parser.add_argument('--prime', type=int, default=P, help='Prime used for key generation')
    parser.add_argument('--root', type=int, default=R, help='Primitive root used for key generation')
    parser.add_argument('--secret', type=int, default=SECRET, help='Private key (integer) used for key generation')
    parser.add_argument('--write', '-w', action='store_true', help='Write public key to a file')
    parser.add_argument('--output', type=str, default='keys/pubkey.txt', help='File to write public key, must be used with \'-w\'')

    return parser

def display(prime, root, secret, pubkey):
    print('Prime used for key generation : {}'.format(prime))
    print('Primitive root used for key generation : {}'.format(root))
    print('Private key used for key generation : {}'.format(secret))
    print('Generated public key : {}'.format(pubkey))

if __name__ == '__main__':

    parser = argument_parser()
    args = parser.parse_args()

    prime = args.prime
    root = args.root
    secret = args.secret
    output = args.output
    os.makedirs(os.path.dirname(output), exist_ok=True)

    if not is_prime(prime):
        print('Number specified with --prime is not prime')
        exit()
    if not is_primitive_root(root, prime):
        print('Number specified with --root is not a generator of G({})'.format(prime))
        exit()
    if not 1 <= secret < prime:
        print('Private key {} must be between 1 and prime {}'.format(secret, prime))

    public_key = pubkeygen(prime, root, secret)
    display(prime, root, secret, public_key)

    if args.write:
        with open(output, 'w') as f:
            f.write(str(public_key))
        print('Public key written to', output)