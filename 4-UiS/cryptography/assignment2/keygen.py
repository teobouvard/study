import argparse
import math

### DEFAULT VALUES ###

P = 23
R = 5
SECRET = 4

### ARTIHMETIC HELPER FUNCTIONS ###

def is_prime(n):
    return not (n < 2 or any(n % x == 0 for x in range(2, int(n ** 0.5) + 1)))

def are_coprime(a, b):
    return math.gcd(a, b) == 1

def is_primitive_root(r, p):
    powers = [r ** i % p for i in range(p-1)]
    return len(set(powers)) == len(powers)


### DIFFIE HELLMANN KEY GENERATION ###

def pubkeygen(prime, root, secret):
    assert(is_prime(prime))
    assert(is_primitive_root(root, prime))

    return root ** secret % prime



### MAIN PROGRAM ###

def argument_parser():
    parser = argparse.ArgumentParser(description='Generate public keys with Diffie-Hellmann algorithm')
    parser.add_argument('--prime', type=int, default=P, help='Prime used for key generation')
    parser.add_argument('--root', type=int, default=R, help='Primitive root used for key generation')
    parser.add_argument('--secret', type=int, default=SECRET, help='Private key (integer) used for key generation')

    return parser



if __name__ == '__main__':

    parser = argument_parser()
    args = parser.parse_args()

    prime = args.prime
    root = args.root
    secret = args.secret

    if not is_prime(prime):
        print('Number specified with --prime is not prime')
        exit()
    if not is_primitive_root(root, prime):
        print('Number specified with --root is not a generator of G({})'.format(prime))
        exit()

    public_key = pubkeygen(prime, root, secret)
    print(public_key)