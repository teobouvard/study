import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Generate a random number using Blum Blum Shub algorithm')
    parser.add_argument('--seed', type=int, help='Seed used for random number generation')
    parser.add_argument('--size', type=int, help='Size in bytes of the generated number')

    return parser


if __name__ == '__main__':

     # parse program arguments
    parser = argument_parser()
    args = parser.parse_args()

    seed = args.seed

    if seed is None:
        print('Please provide a seed using the --seed argument')
        exit()
    