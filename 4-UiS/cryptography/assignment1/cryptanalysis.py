MAX_KEY_LENGTH = 11
MIN_PATTERN_LENGTH = 3
MAX_PATTERN_LENGTH = 15
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# KEY = BDLAEKCY

def char_frequency(s):
	''' Compute the frequency of each letter in a string '''

	chars = {}

	for letter in s:
		if letter not in chars:
			chars[letter] = 1
		else:
			chars[letter] += 1

	chars = {char: occurence/len(s) for char, occurence in chars.items()}

	return sorted(chars.items(), key=lambda x:x[1], reverse=True)

def poly_decrypt(cipher, key):
	''' Decrypt a ciphertext given its key '''

	# make sure key and cipher are in uppercase and without whitespace
	cipher = cipher.upper().replace(' ', '')
	key = key.upper().replace(' ', '')

	# expand the key so that it matches the length of the cipher
	expanded_key = ''.join(key[i % len(key)] for i in range(len(cipher)))

	decrypted_message = ''

	for cipher_letter, key_letter in zip(cipher, expanded_key):
		decrypted_index = (LETTERS.find(cipher_letter) - LETTERS.find(key_letter)) % 26
		decrypted_message += LETTERS[decrypted_index]

	return decrypted_message

def poly_encrypt(message, key):
	''' Encrypt a message given its key '''

	# make sure key and message are in uppercase and without whitespace
	message = message.upper().replace(' ', '')
	key = key.upper().replace(' ', '')

	# expand the key so that it matches the length of the cipher
	expanded_key = ''.join(key[i % len(key)] for i in range(len(message)))

	encrypted_message = ''

	for message_letter, key_letter in zip(message, expanded_key):
		encrypted_index = (LETTERS.find(message_letter) + LETTERS.find(key_letter)) % 26
		encrypted_message += LETTERS[encrypted_index]

	return encrypted_message

def read_cipher():
	''' Returns the ciphertext as a formatted string '''

	with open('polyalphabetic_cipher.txt') as f:
		cipher = f.read()
		cipher = cipher.replace(' ', '')
		cipher = cipher.replace('\n', '')
	return cipher

def get_divisors(n):
	''' Returns the divisors of n up to the max key length '''

	return [x for x in range(2, MAX_KEY_LENGTH) if n%x == 0]

def kasiski_examination(cipher):
	''' Sort the key length probabilities by identifying repeated substrings '''

	possible_key_lengths = []

	for pattern_length in range(MIN_PATTERN_LENGTH, MAX_PATTERN_LENGTH):
		for index in range(len(cipher)):
			substring = cipher[index:index+pattern_length]
			distance = cipher[index+pattern_length:].find(substring)

			if distance != -1:
				possible_key_lengths.extend(get_divisors(distance + pattern_length))

	probable_key_lengths = {length:possible_key_lengths.count(length) for length in possible_key_lengths}

	return sorted(probable_key_lengths.items(), key=lambda x:x[1], reverse=True)

def frequency_analysis(cipher, key_length):
	candidates = []

	for column in range(key_length):
		frequencies = char_frequency(message[column::key_length])
		column_candidates = []

		for most_common in frequencies[:1]:
			E_offset = (LETTERS.find(most_common[0]) - LETTERS.find('E')) % 26
			column_candidates.append(LETTERS[E_offset])
		print("key[{}] candidate : {}".format(column, column_candidates))
		candidates.append(column_candidates)

	return candidates

def print_columns(message, key_length):
	splitted_message = [message[i:i+key_length] for i in range(0, len(message), key_length)]

	for split in splitted_message:
		print(split)

if __name__ == "__main__":

	message = read_cipher()

	print('Guessing key length by kasiski examination', end='\n\n')
	repeats = kasiski_examination(message)
	for distance, occurences in repeats:
		print('key length : {} -> {} occurences'.format(distance, occurences))
	print()

	# key length is either 2, 4 or 8 -> guessing it is 8
	key_length = 8

	print('Probable key letters by frequency analysis', end='\n\n')
	cand = frequency_analysis(message, key_length)
	print()

	print('Pre decrypted message by trying most probable key "BDAAETCY"', end='\n\n')
	decrypted = poly_decrypt(message, 'BDAAETCY')
	print(decrypted)
	print()

	print('We can identify possible words, let\'s print the columns to see which part of the key are wrong')
	print()
	print_columns(decrypted, 8)
	print()

	print('"CRYPEOLFGY" word spotted on last line of column-printed message')
	print('Changing key[2] and key[5] to transform "CRYPEOLFGY" into "CRYPTOLOGY"')

	print('Decrypted message with correct key "BDLAEKCY"', end='\n\n')
	print(poly_decrypt(message, 'BDLAEKCY'))
