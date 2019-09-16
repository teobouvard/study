from timeit import default_timer as timer
from tqdm import tqdm

# S-BOXES #

S0 = [[1, 0, 3, 2], [3, 2, 1, 0], [0, 2, 1, 3], [3, 1, 3, 2]]
S1 = [[0, 1, 2, 3], [2, 0, 1, 3], [3, 0, 1, 0], [2, 1, 0, 3]]


# ASCII UTILITY METHODS #

def ascii2bin(text):
    return ''.join(str(format(ord(character), '08b')) for character in text)

def bin2ascii(binary_str):
    assert(len(binary_str) % 8 == 0)
    return ''.join(chr(int(binary_str[index:index+8], 2)) for index in range(0, len(binary_str), 8))


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

def bitfield_to_string(bitfield):
    return ''.join(str(x) for x in bitfield)


# DES SUBROUTINES #

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


# DES ALGORITHM METHODS #

def encrypt_byte(message, key):
    subkey_0, subkey_1 = subkeygen(key)

    cipher = initial_permutation(message)
    cipher = fk(cipher, subkey_0)
    cipher = switch(cipher)
    cipher = fk(cipher, subkey_1)
    cipher = inverse_initial_permutation(cipher)

    return cipher

def decrypt_byte(cipher, key):
    subkey_0, subkey_1 = subkeygen(key)

    message = initial_permutation(cipher)
    message = fk(message, subkey_1)
    message = switch(message)
    message = fk(message, subkey_0)
    message = inverse_initial_permutation(message)

    return message

def decrypt_message(cipher, key):
    assert(len(cipher)%8 == 0)

    message = []

    for index in range(0, len(cipher), 8):
        decrypted_byte = decrypt_byte(cipher[index:index+8], key)
        message.extend(decrypted_byte)
    
    return bitfield_to_string(message)


# TRIPLE DES ALGORITHM METHODS #

def triple_encrypt_byte(message, key0, key1):
    message = encrypt_byte(message, key0)
    message = decrypt_byte(message, key1)
    message = encrypt_byte(message, key0)

    return message

def triple_decrypt_byte(cipher, key0, key1):
    cipher = decrypt_byte(cipher, key0)
    cipher = encrypt_byte(cipher, key1)
    cipher = decrypt_byte(cipher, key0)

    return cipher

def triple_decrypt_message(cipher, key0, key1):
    assert(len(cipher)%8 == 0)

    message = []

    for index in range(0, len(cipher), 8):
        decrypted_byte = triple_decrypt_byte(cipher[index:index+8], key0, key1)
        message.extend(decrypted_byte)
    
    return bitfield_to_string(message)


# CIPHER CRACKING #

def des_bruteforce(cipher, probable_word):

    probable_keys = []

    for key in range(1024):
        key = format(key, '010b')
        key = create_bitfield(key)
        message = decrypt_message(cipher, key)
        message = bitfield_to_string(message)
        if message.find(probable_word) != -1:
            probable_keys.append(key)
            decrypted = decrypt_message(cipher, key)
            print('key : {} -> message : {}'.format(key, bin2ascii(decrypted)))
    
    return probable_keys

def triple_des_bruteforce(cipher, probable_word):
    probable_keys = []

    for key0 in tqdm(range(1024)):
        key0 = format(key0, '010b')
        key0 = create_bitfield(key0)
        for key1 in tqdm(range(1024)):
            key1 = format(key1, '010b')
            key1 = create_bitfield(key1)

            message = triple_decrypt_message(cipher, key0, key1)
            message = bitfield_to_string(message)
            if message.find(probable_word) != -1:
                probable_keys.append((key0, key1))
                decrypted = triple_decrypt_message(cipher, keys[0], keys[1])
                print('keys : {} -> message : {}'.format(keys, bin2ascii(decrypted)))
    
    return probable_keys


# ASSIGNMENT #

def decrypt_sdes_cipher():
    with open('ctx1.txt', 'r') as f:
        cipher = create_bitfield(f.read())

        start = timer()
        probable_keys = des_bruteforce(cipher, ascii2bin('des'))
        stop = timer()

        print('Elapsed time : {}'.format(stop-start))

def decrypt_triple_sdes_cipher():
    with open('ctx2.txt', 'r') as f:
        cipher = create_bitfield(f.read())

        start = timer()
        probable_keys = triple_des_bruteforce(cipher, ascii2bin('des'))
        stop = timer()

        print('Elapsed time : {}'.format(stop-start))

if __name__ == "__main__":

    example_key = create_bitfield('1110001110')
    example_byte = create_bitfield('10101010')

    #decrypt_sdes_cipher()
    decrypt_triple_sdes_cipher()

    #print(bitfield_to_string(cipher))

