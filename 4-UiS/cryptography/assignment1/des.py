from timeit import default_timer as timer
import multiprocessing

# CONSTANT CIPHERTETXTS
with open('ctx1.txt', 'r') as f:
    CTX1 = f.read()

with open('ctx2.txt', 'r') as f:
    CTX2 = f.read()


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
    message = create_bitfield(message)
    key = create_bitfield(key)

    subkey_0, subkey_1 = subkeygen(key)

    cipher = initial_permutation(message)
    cipher = fk(cipher, subkey_0)
    cipher = switch(cipher)
    cipher = fk(cipher, subkey_1)
    cipher = inverse_initial_permutation(cipher)

    return bitfield_to_string(cipher)

def decrypt_byte(cipher, key):
    cipher = create_bitfield(cipher)
    key = create_bitfield(key)

    subkey_0, subkey_1 = subkeygen(key)

    message = initial_permutation(cipher)
    message = fk(message, subkey_1)
    message = switch(message)
    message = fk(message, subkey_0)
    message = inverse_initial_permutation(message)

    return bitfield_to_string(message)

def decrypt_message(cipher, key):
    assert(len(cipher)%8 == 0)

    message = ''

    for index in range(0, len(cipher), 8):
        decrypted_byte = decrypt_byte(cipher[index:index+8], key)
        message += decrypted_byte
    
    return message


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
            print('key : {} -> message : {}'.format(bitfield_to_string(key), bin2ascii(message)))
    
    return probable_keys

def triple_des_bruteforce(cipher, probable_word):
    probable_keys = []

    for key0 in range(1024):
        key0 = format(key0, '010b')
        key0 = create_bitfield(key0)
        for key1 in range(1024):
            key1 = format(key1, '010b')
            key1 = create_bitfield(key1)

            message = triple_decrypt_message(cipher, key0, key1)
            message = bitfield_to_string(message)
            if message.find(probable_word) != -1:
                probable_keys.append((key0, key1))
                decrypted = triple_decrypt_message(cipher, key0, key1)
                print('keys : {} -> message : {}'.format((key0, key1), bin2ascii(decrypted)))
    
    return probable_keys

def parallel_des_bruteforce(key):
    return decrypt_message(CTX1, key), key

def parallel_triple_des_bruteforce(keys):
    return triple_decrypt_message(CTX2, keys[0], keys[1]), keys

# ASSIGNMENT #

def test_cases():

    # Encryption test 1
    message = '10101010'
    key = '0000000000'
    expected_cipher = '00010001'
    assert(encrypt_byte(message, key) == expected_cipher)
    assert(decrypt_byte(expected_cipher, key) == message)

    # Encryption test 2
    message = '10101010'
    key = '1110001110'
    expected_cipher = '11001010'
    assert(encrypt_byte(message, key) == expected_cipher)
    assert(decrypt_byte(expected_cipher, key) == message)
    
    # Encryption test 3
    message = '01010101'
    key = '1110001110'
    expected_cipher = '01110000'
    assert(encrypt_byte(message, key) == expected_cipher)
    assert(decrypt_byte(expected_cipher, key) == message)

    # Encryption test 3
    message = '10101010'
    key = '1111111111'
    expected_cipher = '00000100'
    assert(encrypt_byte(message, key) == expected_cipher)
    assert(decrypt_byte(expected_cipher, key) == message)

    print("All tests passed")

def task1():

    # DES encryption 1
    key = '0000000000'
    message = '00000000'
    ciphertext = encrypt_byte(message, key)
    print('key : {} | plaintext : {} | ciphertext : {}'.format(key, message, ciphertext))

    # DES encryption 2
    key = '0000011111'
    message = '11111111'
    ciphertext = encrypt_byte(message, key)
    print('key : {} | plaintext : {} | ciphertext : {}'.format(key, message, ciphertext))

    # DES encryption 3
    key = '0010011111'
    message = '11111100'
    ciphertext = encrypt_byte(message, key)
    print('key : {} | plaintext : {} | ciphertext : {}'.format(key, message, ciphertext))

    # DES encryption 4
    key = '0010011111'
    message = '10100101'
    ciphertext = encrypt_byte(message, key)
    print('key : {} | plaintext : {} | ciphertext : {}'.format(key, message, ciphertext))

    # DES decryption 1
    key = '1111111111'
    ciphertext = '00001111'
    message = decrypt_byte(ciphertext, key)
    print('key : {} | plaintext : {} | ciphertext : {}'.format(key, message, ciphertext))

    # DES decryption 2
    key = '0000011111'
    ciphertext = '01000011'
    message = decrypt_byte(ciphertext, key)
    print('key : {} | plaintext : {} | ciphertext : {}'.format(key, message, ciphertext))

    # DES decryption 3
    key = '1000101110'
    ciphertext = '00011100'
    message = decrypt_byte(ciphertext, key)
    print('key : {} | plaintext : {} | ciphertext : {}'.format(key, message, ciphertext))

    # DES decryption 4
    key = '0000011111'
    ciphertext = '01000011'
    message = decrypt_byte(ciphertext, key)
    print('key : {} | plaintext : {} | ciphertext : {}'.format(key, message, ciphertext))

def task2():

    # Triple DES encryption 1
    keys = ('1000101110', '0110101110')
    message = '11010111'
    ciphertext = triple_encrypt_byte(message, keys[0], keys[1])
    print('keys : {} | plaintext : {} | ciphertext : {}'.format(keys, message, ciphertext))

    # Triple DES encryption 2
    keys = ('1000101110', '0110101110')
    message = '10101010'
    ciphertext = triple_encrypt_byte(message, keys[0], keys[1])
    print('keys : {} | plaintext : {} | ciphertext : {}'.format(keys, message, ciphertext))

    # Triple DES encryption 3
    keys = ('1111111111', '1111111111')
    message = '00000000'
    ciphertext = triple_encrypt_byte(message, keys[0], keys[1])
    print('keys : {} | plaintext : {} | ciphertext : {}'.format(keys, message, ciphertext))

    # Triple DES encryption 4
    keys = ('0000000000', '0000000000')
    message = '01010010'
    ciphertext = triple_encrypt_byte(message, keys[0], keys[1])
    print('keys : {} | plaintext : {} | ciphertext : {}'.format(keys, message, ciphertext))

    # Triple DES decryption 1
    keys = ('1000101110', '0110101110')
    ciphertext = '11100110'
    message = triple_decrypt_byte(ciphertext, keys[0], keys[1])
    print('keys : {} | plaintext : {} | ciphertext : {}'.format(keys, message, ciphertext))

    # Triple DES decryption 2
    keys = ('1011101111', '0110101110')
    ciphertext = '01010000'
    message = triple_decrypt_byte(ciphertext, keys[0], keys[1])
    print('keys : {} | plaintext : {} | ciphertext : {}'.format(keys, message, ciphertext))

    # Triple DES decryption 3
    keys = ('1111111111', '1111111111')
    ciphertext = '00000100'
    message = triple_decrypt_byte(ciphertext, keys[0], keys[1])
    print('keys : {} | plaintext : {} | ciphertext : {}'.format(keys, message, ciphertext))

    # Triple DES decryption 4
    keys = ('0000000000', '0000000000')
    ciphertext = '11110000'
    message = triple_decrypt_byte(ciphertext, keys[0], keys[1])
    print('keys : {} | plaintext : {} | ciphertext : {}'.format(keys, message, ciphertext))

def decrypt_sdes_cipher_parallel():
    
    numkeys = 1024
    chunksize = int(numkeys / multiprocessing.cpu_count())
    keys = [format(x, '010b') for x in range(1024)]

    probable_word = ascii2bin('security')

    with multiprocessing.Pool() as p:
        for message, key in p.imap_unordered(func=parallel_des_bruteforce, iterable=keys, chunksize=chunksize):
            if message.find(probable_word) != -1:
                print('key : {} -> message : {}'.format(bitfield_to_string(key), bin2ascii(message)))


def decrypt_triple_sdes_cipher_parallel():

    numkeys = 1024 ** 2
    chunksize = int(numkeys / multiprocessing.cpu_count())
    keys = [(format(x, '010b'), format(y, '010b')) for x in range(1024) for y in range(1024)]

    probable_word = ascii2bin('security')

    with multiprocessing.Pool() as p:
        for message, keys in p.imap_unordered(func=parallel_triple_des_bruteforce, iterable=keys, chunksize=chunksize):
            if message.find(probable_word) != -1:
                print('keys : ({}, {}) -> message : {}'.format(bitfield_to_string(keys[0]), bitfield_to_string(keys[1]), bin2ascii(message)))


if __name__ == "__main__":

    print('Testing implementation correctness ...', end=' ')
    test_cases()

    print('\n', 'Simple DES encryptions and decryptions', end='\n\n')
    task1()

    print('\n', 'Triple DES encryptions and decryptions', end='\n\n')
    task2()

    print('\n', 'Cracking SDES ciphertext with only one CPU', end='\n\n')
    start = timer()
    des_bruteforce(CTX1, ascii2bin('des'))
    stop = timer()
    print('Elapsed time : {0:.3f}s'.format(stop-start))

    print('\n', 'Cracking SDES ciphertext with parallelized bruteforce', end='\n\n')
    start = timer()
    decrypt_sdes_cipher_parallel()
    stop = timer()
    print('Elapsed time : {0:.3f}s'.format(stop-start))

    print('\n', 'Cracking triple SDES ciphertext with parallelized bruteforce', end='\n\n')
    start = timer()
    print('Uncomment next line in source code to execute it. Might take some time depending on your hardware.')
    #decrypt_triple_sdes_cipher_parallel()
    stop = timer()
    print('Elapsed time : {0:.3f}s'.format(stop-start))
