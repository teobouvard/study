MAX_KEY_LENGTH = 10
MIN_PATTERN_LENGTH = 3
MAX_PATTERN_LENGTH = 10
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def char_frequency(s):
    ''' Returns the frequency of each letter in a string '''

    chars = {}

    for letter in s:
        if letter not in chars:
            chars[letter] = 1
        else:
            chars[letter] += 1

    chars = {char: occurence/len(s) for char, occurence in chars.items()}

    return sorted(chars.items(), key=lambda x:x[1], reverse=True)

def poly_decrypt(cipher, key):
    ''' Decrypts a ciphertext given its key '''

    # make sure key and cipher are in uppercase and without whitespace
    cipher = cipher.upper().replace(" ", "")
    key = key.upper().strip().replace(" ", "")

    # expand the key so that it matches the length of the cipher
    expanded_key = ''.join(key[i % len(key)] for i in range(len(cipher)))

    decrypted_message = ''

    for cipher_letter, key_letter in zip(cipher, expanded_key):
        decrypted_index = (LETTERS.find(cipher_letter) - LETTERS.find(key_letter)) % 26
        decrypted_message += LETTERS[decrypted_index]

    return decrypted_message
    
def read_cipher():
    ''' Returns the ciphertext as a formatted string '''
    with open('polyalphabetic_cipher.txt') as f:
        cipher = f.read()
        cipher = cipher.replace(" ", "")
        cipher = cipher.replace("\n", "")
    return cipher

def kasiski_examination(cipher):
    ''' Tries to guess the key length by identifying repeated substrings '''

    possible_key_lengths = []

    for pattern_length in range(MIN_PATTERN_LENGTH, MAX_PATTERN_LENGTH):
        for index in range(len(cipher)):
            substring = cipher[index:index+pattern_length]
            distance = cipher[index+pattern_length:].find(substring)

            if distance != -1:
                possible_key_lengths.append((distance + pattern_length) % MAX_KEY_LENGTH)

    probable_key_lengths = {length:possible_key_lengths.count(length) for length in possible_key_lengths}

    return sorted(probable_key_lengths.items(), key=lambda x:x[1], reverse=True)

def attack(cipher, key_length):
    candidates = []

    for index in range(key_length):
        frequencies = char_frequency(message[index::key_length])

        for most_common in frequencies[:3]:
            E_offset = (LETTERS.find(most_common[0]) - LETTERS.find('E')) % 26
            candidates.append(LETTERS[E_offset])

        print("key[{}] candidates : {}".format(index, candidates))

if __name__ == "__main__":

    message = read_cipher()

    examination = kasiski_examination(message)
    print(examination)
    # key length is probably 8

    attack(message, 8)

    #decrypted = poly_decrypt("CSASTPKVSIQUTGQUCSAS  TPIUAQJB","ABCD")
    #print(decrypted)