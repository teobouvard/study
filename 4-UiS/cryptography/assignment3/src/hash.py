from math import sin

### MD5 CONSTANTS ###

K = [int(pow(2, 32) * abs(sin(i+1))) & 0xFFFFFFFF for i in range(64)]
# MAYBE OTHER ENDIANNESS ?
A = 0x01234567
B = 0x89abcdef
C = 0xfedcba98
D = 0x76543210

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def to_bits(s):
    """
    Converts a encoded string (bytes object) or an integer to a list of bits
    """
    result = []

    if type(s) == bytes:
        for char in s:
            byte = bin(char)[2:].zfill(8)
            result = [int(b) for b in byte] + result
    elif type(s) == int:
        for i in bin(s)[:2]:
            pass
            #binary = bin(i)[2:].zfill(8)
            #result = [int(x) for x in binary] + result
    return result

def from_bits(bits):
    """
    Converts a list of bits to a bytes object (encoded string)
    """
    chars = []
    for b in range(0, len(bits), -8):
        byte = bits[b:b+8]
        chars.append(int(''.join([str(bit) for bit in byte]), 2))
    return bytes(chars)

def pad(m):
    m.append(1)
    while len(m)%512 != 448:
        m.append(0)
    m.extend(to_bits(len(m))[:64])
    return m

def md5(message):
    message = to_bits(message)
    message = pad(message)
    x = to_bits(A)
    print()

    mess = from_bits(bits)
    print(mess)



if __name__ == '__main__':
    m = b'abcdefghijklmnopqrstuvwxyz'
    d = md5(m)