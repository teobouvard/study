MAX_KEY_LENGTH = 10
MIN_PATTERN_LENGTH = 5
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

def get_factors(n):
	factors = []
	while n%2 == 0:
		if n > MAX_KEY_LENGTH:
			pass
		else:
			factors.append(int(n))
		n = n/2
	return factors

def kasiski_examination(cipher):
	''' Sort the key length probabilities by identifying repeated substrings '''

	possible_key_lengths = []

	for pattern_length in range(MIN_PATTERN_LENGTH, MAX_PATTERN_LENGTH):
		for index in range(len(cipher)):
			substring = cipher[index:index+pattern_length]
			distance = cipher[index+pattern_length:].find(substring)

			if distance != -1:
				possible_key_lengths.extend(get_factors(distance + pattern_length))

	probable_key_lengths = {length:possible_key_lengths.count(length) for length in possible_key_lengths}

	return sorted(probable_key_lengths.items(), key=lambda x:x[1], reverse=True)

def attack(cipher, key_length):
	candidates = []

	for column in range(key_length):
		frequencies = char_frequency(message[column::key_length])
		column_candidates = []

		for most_common in frequencies[:5]:
			E_offset = (LETTERS.find(most_common[0]) - LETTERS.find('E')) % 26
			column_candidates.append(LETTERS[E_offset])
		print("key[{}] candidates : {}".format(column, column_candidates))
		candidates.append(column_candidates)

	return candidates


def key_elimination(cipher, key_length, probable_word):

	self_encrypted_word = poly_encrypt(probable_word, probable_word)

	offset_cipher = ''.join(LETTERS[0] for i in range(len(cipher)))
	
	self_encrypted_cipher = ''.join(LETTERS[LETTERS.find(a)-LETTERS.find(b)] for a,b in zip(cipher, offset_cipher))

	if self_encrypted_cipher.find(self_encrypted_word) != -1:
		print('YES')

if __name__ == "__main__":

	message = read_cipher()

	examination = kasiski_examination(message)
	print(examination)

	#key length is probably 8
	#key_length = 8

	#key_elimination(message, key_length, "ATTACK")

	#cand = attack(message, key_length)

	#decrypted = poly_decrypt(message,"BDAAETCY")

	#splitted_message = [decrypted[i:i+key_length] for i in range(0, len(decrypted), key_length)]

	#for split in splitted_message:
		#print(split)

	#print(decrypted)
