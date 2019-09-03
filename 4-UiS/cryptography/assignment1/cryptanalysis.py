MAX_KEY_LENGTH = 10

def char_frequency(s):
    ''' Returns the frequency of each letter in a string '''

    chars = {}

    for letter in s:
        if letter not in chars:
            chars[letter] = 1
        else:
            chars[letter] += 1

    chars = {char: occurence/len(s) for char, occurence in chars.items()}

    return sorted(chars.items(), key=lambda x:1/x[1])

def poly_decrypt(cipher, key):
    ''' Decrypts a ciphertext given its key '''
    # make sure key and cipher are in uppercase to use ascii indexes easily
    cipher = cipher.upper()
    key = key.upper()

    # expand the key so that it matches the length of the cipher
    expanded_key = ''.join(key[i % len(key)] for i in range(len(cipher)))

    decrypted_message = ""

    for cipher_letter, key_letter in zip(cipher, expanded_key):

        # retreive the alphabetical index of the letters and decrypt the cipher letter 
        cipher_index = ord(cipher_letter)
        key_index = ord(key_letter)

        decrypted_index = (cipher_index - key_index) % 26

        # match the decrypted index to the ascii letter
        decrypted_message += chr(ord('A') + decrypted_index)

    return decrypted_message
    
def read_cipher():
    ''' Returns the polyalphabetic ciphertext as a formatted string '''
    with open('polyalphabetic_cipher.txt') as f:
        cipher = f.read()
        cipher = cipher.replace(" ", "")
        cipher = cipher.replace("\n", "")
    return cipher

def kasiski_examination(cipher):

    repeated_patterns = {}

    for pattern_length in range(4, 10):
        for index in range(len(cipher)):
            substring = cipher[index:index+pattern_length]
            distance = cipher[index+pattern_length:].find(substring)

            if distance != -1:
                if substring not in repeated_patterns:
                    repeated_patterns[substring] = [2, (distance + pattern_length) % MAX_KEY_LENGTH]
                else:
                    repeated_patterns[substring][0] += 1
                    repeated_patterns[substring][1] = min((distance + pattern_length) % MAX_KEY_LENGTH, repeated_patterns[substring][1])

    return repeated_patterns

if __name__ == "__main__":

    message = read_cipher()


    examination = kasiski_examination(message)

    # key length is probably eight
    KEY_LENGTH = 8

    for index in range(KEY_LENGTH):
        frequencies = char_frequency(message[index::KEY_LENGTH])
        candidates = []
        for most_common in frequencies[:3]:
            offset = (ord(most_common[0]) - ord('E')) % 26
            candidates.append(chr(ord('A') + offset))
        print("key[{}] candidates : {}".format(index, candidates))


    decrypted = poly_decrypt(message,"BDAAETC")
    print(decrypted)