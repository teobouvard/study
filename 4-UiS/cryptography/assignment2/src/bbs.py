import argparse
import os

DEFAULT_SIZE = 128
P = 30000000091
Q = 40000000003
M = P*Q

### AUTO BASE INTEGERS ###

def auto_int(x):
    return int(x, 0)

### PSEUDO RANDOM NUMBER GENERATION ###

def generate_random(seed, size):
    bits = []

    for _ in range(size):
        seed = pow(seed, 2, M)
        bits.append(bin(seed)[-1])
    
    return int(''.join(bits), 2)

### MAIN PROGRAM ###

def argument_parser():
    parser = argparse.ArgumentParser(description='Generate a random number using Blum Blum Shub algorithm')
    parser.add_argument('--seed', type=auto_int, required=True, help='Seed used for random number generation')
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE, help='Size in bits of the generated number, 128 if not specified')
    parser.add_argument('--output', type=str, help='File to which the random number is written')
    parser.add_argument('--verbose', '-v', action='store_true', help='Display parameters used for key generation')

    return parser

def display(seed, size, number):
    print('Seed used : {}'.format(hex(seed)))
    print('Number of bits generated : {}'.format(size))
    print('Random number generated: {}'.format(hex(number)))

if __name__ == '__main__':

    # parse program arguments
    parser = argument_parser()
    args = parser.parse_args()

    # check arguments correctness
    if args.size < 1:
        print('--size must be a positive integer')
        exit()
    if args.output is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    prn = generate_random(args.seed, args.size)

    # display parameters if program is run in verbose mode or if key is not written to disk
    if args.output is None or args.verbose:
        display(args.seed, args.size, prn)

    if args.output is not None:
        with open(args.output, 'w') as f:
            f.write(hex(prn))
            print('Random number written to', args.output)
