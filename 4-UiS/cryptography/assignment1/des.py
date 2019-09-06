S0 = [[1, 0, 3, 2], [3, 2, 1, 0], [0, 2, 1, 3], [3, 1, 3, 2]]
S1 = [[0, 1, 2, 3], [2, 0, 1, 3], [3, 0, 1, 0], [2, 1, 0, 3]]


# BITFIELD UTILITY METHODS #

def create_bitfield(string):
    return [int(x) for x in string]

def left_shift(bitfield, n=1):
    n = n % len(bitfield)
    return bitfield[n:] + bitfield[:n]

def xor(a, b):
    return [int(x)^int(y) for x,y in zip(a, b)]

def switch(bitfield):
    assert(len(bitfield) == 8)
    return bitfield[4:] + bitfield[:4]


# DES ALGORITHM METHODS #

def subkeygen(key):
    assert(len(key) == 10)

    permut_10 = [key[2], key[4], key[1], key[6], key[3], key[9], key[0], key[8], key[7], key[5]]

    subshifted_0 = left_shift(permut_10[:5])
    subshifted_1 = left_shift(permut_10[5:])

    shifted = subshifted_0 + subshifted_1

    subkey_0 = [shifted[5], shifted[2], shifted[6], shifted[3], shifted[7], shifted[4], shifted[9], shifted[8]]

    subshifted_0 = left_shift(subshifted_0, 2)
    subshifted_1 = left_shift(subshifted_1, 2)

    shifted = subshifted_0 + subshifted_1

    subkey_1 = [shifted[5], shifted[2], shifted[6], shifted[3], shifted[7], shifted[4], shifted[9], shifted[8]]

    return subkey_0, subkey_1

def initial_permutation(bitfield):
    return [bitfield[1], bitfield[5], bitfield[2], bitfield[0], bitfield[3], bitfield[7], bitfield[4], bitfield[6]]

def inverse_initial_permutation(bitfield):
    return [bitfield[3], bitfield[0], bitfield[2], bitfield[4], bitfield[6], bitfield[1], bitfield[7], bitfield[5]]

def fk(bitfield, subkey):
    assert(len(bitfield) == 8)
    
    left, right = bitfield[:4], bitfield[4:]
    expansion = [right[3], right[0], right[1], right[2], right[1], right[2], right[3], right[0]]
    mapping = xor(expansion, subkey)

    row0 = int(''.join(str(x) for x in [mapping[0], mapping[3]]), 2)
    row1 = int(''.join(str(x) for x in [mapping[4], mapping[7]]), 2)
    column0 = int(''.join(str(x) for x in [mapping[1], mapping[2]]), 2)
    column1 = int(''.join(str(x) for x in [mapping[5], mapping[6]]), 2)

    s0 = create_bitfield(format(S0[row0][column0], '02b'))
    s1 = create_bitfield(format(S1[row1][column1], '02b'))

    union = s0 + s1

    output = [union[1], union[3], union[2], union[0]]

    return xor(left, output) + right

def encrypt(message, key):
    subkey_0, subkey_1 = subkeygen(example_key)

    message = initial_permutation(message)
    message = fk(message, subkey_0)
    message = switch(message)
    message = fk(message, subkey_1)
    message = inverse_initial_permutation(message)

    return message

def decrypt(cipher, key):
    subkey_0, subkey_1 = subkeygen(example_key)

    message = initial_permutation(message)
    message = fk(message, subkey_1)
    message = switch(message)
    message = fk(message, subkey_0)
    message = inverse_initial_permutation(message)

    return message

def triple_encrypt(message, key0, key1):
    message = encrypt(message, key0)
    message = decrypt(message, key1)
    message = encrypt(message, key0)

    return message

def triple_decrypt(cipher, key0, key1):
    cipher = encrypt(cipher, key0)
    cipher = decrypt(cipher, key1)
    cipher = encrypt(cipher, key0)

    return cipher

if __name__ == "__main__":

    example_key = create_bitfield('1111111111')
    example_message = create_bitfield('00000100')

    cipher = encrypt(example_message, example_key)

    print(cipher)
    #print(inverse_initial_permutation(create_bitfield('10111101')))