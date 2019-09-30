import argparse
import math
import os

# if the prime used is really large, we assume the user know what he is doing 
# and do not check for primality nor for root generation of the group (too expensive)
SMALL_INT_SIZE = 10000

## TEST VALUES ##
#P = 1009
#R = 263

## 2048-bit MODP Group ##
with open('files/2048-bit MODP Group/prime.txt', 'r') as f:
    P = int(''.join(f.read().split()), 16)

with open('files/2048-bit MODP Group/generator.txt', 'r') as f:
    R = int(''.join(f.read().split()), 16)

### ARTIHMETIC HELPER FUNCTIONS ###

def is_prime(n):
    if n < SMALL_INT_SIZE:
        return not (n < 2 or any(n % x == 0 for x in range(2, int(math.sqrt(n)+ 1))))
    else:
        return True

def are_coprime(a, b):
    return math.gcd(a, b) == 1

def is_primitive_root(r, p):
    if p < SMALL_INT_SIZE:
        powers = [r ** i % p for i in range(p-1)]
        return len(set(powers)) == p-1
    else:
        return True

def auto_int(x):
    return int(x, 0)


### DIFFIE HELLMANN KEY GENERATION ###

def pubkeygen(prime, root, secret):
    assert(is_prime(prime))
    assert(is_primitive_root(root, prime))

    return pow(root, secret, prime)

def shared_secret_key(secret, other_public_key):
    return pow(other_public_key, secret, prime)

### MAIN PROGRAM ###

def argument_parser():
    parser = argparse.ArgumentParser(description='Generate public and private keys with the Diffie-Hellmann algorithm')
    parser.add_argument('--mode', choices=['generate', 'merge', 'test'], required=True, help='Generate a public key, compute a shared private key, or test program')    
    parser.add_argument('--prime', type=auto_int, default=P, help='Prime used for key generation')
    parser.add_argument('--root', type=auto_int, default=R, help='Primitive root used for key generation')
    parser.add_argument('--secret', type=auto_int, help='Private key (integer) used for key generation')
    parser.add_argument('--verbose', action='store_true', help='Display parameters used for key generation')
    parser.add_argument('--output', type=str, help='File to which the public key is written (standard ouput if not specified)')
    parser.add_argument('--public', type=auto_int, help='Public key to be merged with the private key')

    return parser

def display_public(prime, root, secret, pubkey):
    print('Prime used for key generation : {}'.format(hex(prime)))
    print('Primitive root used for key generation : {}'.format(hex(root)))
    print('Private key used for key generation : {}'.format(hex(secret)))
    print('Generated public key : {}'.format(hex(pubkey)))

def display_private(secret, pubkey, shared):
    print('Private key used for shared key generation : {}'.format(hex(secret)))
    print('Public key used for shared key generation : {}'.format(hex(pubkey)))
    print('Computed private shared key : {}'.format(hex(shared)))

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
        print('Number specified with --prime is not prime.')
        exit(1)
    if not is_primitive_root(root, prime):
        print('Number specified with --root is not a generator of G({}).'.format(prime))
        exit(1)
    if secret is None and args.mode != 'test':
        print('No private key and not in test mode. Use the --secret argument.')
        exit(1)
    if args.mode != 'test' and not 1 <= secret < prime:
        print('Private key {} must be between 1 and prime {}.'.format(secret, prime))
        exit(1)
    if output is not None:
        os.makedirs(os.path.dirname(output), exist_ok=True)

    if args.mode == 'generate':
        public_key = pubkeygen(prime, root, secret)

        # display parameters if program is run in verbose mode or if key is not written to disk
        if args.output is None or args.verbose:
            display_public(prime, root, secret, public_key)

        # write pubkey to file if output argument passed
        if args.output:
            with open(output, 'w') as f:
                f.write(hex(public_key))
            print('Public key written to', output)

    elif args.mode == 'merge':

        if public is None:
            print('A public key is necessary to compute the shared private key. Use the --public argument.')
            exit(1)
        
        shared = shared_secret_key(secret, public)

        # display parameters if program is run in verbose mode or if key is not written to disk
        if args.output is None or args.verbose:
            display_private(secret, public, shared)
        
        # write private key to file if output argument passed
        if args.output:
            with open(output, 'w') as f:
                f.write(hex(shared))
            print('Shared private key written to', output)

    elif args.mode == 'test':
        with open('files/2048-bit MODP Group/test_xA.txt', 'r') as f:
            test_xA = int(''.join(f.read().split()), 16)
        with open('files/2048-bit MODP Group/test_yA.txt', 'r') as f:
            test_yA = int(''.join(f.read().split()), 16)
        with open('files/2048-bit MODP Group/test_xB.txt', 'r') as f:
            test_xB = int(''.join(f.read().split()), 16)
        with open('files/2048-bit MODP Group/test_yB.txt', 'r') as f:
            test_yB = int(''.join(f.read().split()), 16)
        with open('files/2048-bit MODP Group/test_Z.txt', 'r') as f:
            test_Z = int(''.join(f.read().split()), 16)

        yA = pubkeygen(P, R, test_xA)
        yB = pubkeygen(P, R, test_xB)
        ZA = shared_secret_key(test_xA, yB)
        ZB = shared_secret_key(test_xB, yA)

        if yA != test_yA:
            print('ERROR : A public key is not equal to the test one !')
            exit(1)
        else:
            print('A public key is OK')

        if yB != test_yB:
            print('ERROR : B public key is not equal to the test one !')
            exit(1)
        else:
            print('B public key is OK')

        if ZA != test_Z:
            print('ERROR : A shared private key is not equal to the test one !')
            exit(1)
        else:
            print('A shared private key is OK')

        if ZB != test_Z:
            print('ERROR : B shared private key is not equal to the test one !')
            exit(1)
        else:
            print('B shared private key is OK')