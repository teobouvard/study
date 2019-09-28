import argparse
import math
import os

### DEFAULT VALUES ###

P = 1009
R = 263

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

def shared_secret_key(secret, other_public_key):
    return other_public_key ** secret % prime

### MAIN PROGRAM ###

def argument_parser():
    parser = argparse.ArgumentParser(description='Generate public and private keys with the Diffie-Hellmann algorithm')
    parser.add_argument('--mode', choices=['generate', 'merge'], required=True, help='Generate a public key or compute a private shared key')    
    parser.add_argument('--prime', type=int, default=P, help='Prime used for key generation')
    parser.add_argument('--root', type=int, default=R, help='Primitive root used for key generation')
    parser.add_argument('--secret', type=int, required=True, help='Private key (integer) used for key generation')
    parser.add_argument('--verbose', action='store_true', help='Display parameters used for key generation')
    parser.add_argument('--output', type=str, help='File to which the public key is written (standard ouput if not specified)')
    parser.add_argument('--public', type=int, help='Public key to be merged with the private key')

    return parser

def display_public(prime, root, secret, pubkey):
    print('Prime used for key generation : {}'.format(prime))
    print('Primitive root used for key generation : {}'.format(root))
    print('Private key used for key generation : {}'.format(secret))
    print('Generated public key : {}'.format(pubkey))

def display_private(secret, pubkey, shared):
    print('Private key used for shared key generation : {}'.format(secret))
    print('Public key used for shared key generation : {}'.format(pubkey))
    print('Computed private shared key : {}'.format(shared))

if __name__ == '__main__':

    # parse program arguments
    parser = argument_parser()
    args = parser.parse_args()

    prime = args.prime
    root = args.root
    secret = args.secret
    output = args.output
    public = args.public

    # check parameters correctness
    if not is_prime(prime):
        print('Number specified with --prime is not prime')
        exit()
    if not is_primitive_root(root, prime):
        print('Number specified with --root is not a generator of G({})'.format(prime))
        exit()
    if not 1 <= secret < prime:
        print('Private key {} must be between 1 and prime {}'.format(secret, prime))
        exit()
    if output is not None:
        os.makedirs(os.path.dirname(output), exist_ok=True)

    if args.mode == 'generate':
        public_key = pubkeygen(prime, root, secret)

        # display parameters if program is run in verbose mode or if key is not written to disk
        if args.output is None or args.verbose:
            display_public(prime, root, secret, public_key)

        # write pubkey to file if write argument passed
        if args.output:
            with open(output, 'w') as f:
                f.write(str(public_key))
            print('Public key written to', output)
    
    elif args.mode == 'merge':

        if public is None:
            print('A public key is necessary to compute the shared private key')
            exit()
        
        shared = shared_secret_key(secret, public)

        if args.output is None or args.verbose:
            display_private(secret, public, shared)
        
        if args.output:
            with open(output, 'w') as f:
                f.write(str(shared))
            print('Shared private key written to', output)
