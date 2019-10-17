from math import sin

### MD5 CONSTANTS ###

K = [int(pow(2, 32) * abs(sin(i+1))) & 0xFFFFFFFF for i in range(64)]
SHIFT = [
    7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
    5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
    4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
    6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21
]

# MAYBE OTHER ENDIANNESS ?
a0 = 0x67452301
b0 = 0xefcdab89
c0 = 0x98badcfe
d0 = 0x10325476

### SUBROUTINES ###

def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]

def to_bits(s):
    """
    Converts a encoded string (bytes object) or an integer to a list of bits
    """
    if type(s) == bytes:
        result = []
        for char in s:
            byte = bin(char)[2:].zfill(8)
            result = [int(b) for b in byte] + result

    elif type(s) == int:
        result = [int(x) for x in bin(s)[2:].zfill(64)]

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
    m_len = to_bits(len(m) % pow(2, 64))
    m.extend(m_len[32:64])
    m.extend(m_len[0:32])
    return m

def rotate_left(x, n):
    x &= 0xFFFFFFFF
    return ((x << n) | (x >> (32-n))) & 0xFFFFFFFF

def md5(message):
    # Constants
    global K, SHIFT, a0, b0, c0, d0
    # Padding step
    message = to_bits(message)
    message = pad(message)

    # Message processing
    for chunk in chunks(message, 512):
        words = chunks(chunk, 32)
        A = a0
        B = b0
        C = c0
        D = d0
        for i in range(64):

            if 0 <= i <= 15:
                F = (B & C) | ((~B) & D)
                g = i
            elif 16 <= i <= 31:
                F = (D & B) | ((~D) & C)
                g = (5*i + 1) % 16
            elif 32 <= i <= 47:
                F = B ^ C ^ D
                g = (3*i + 5) % 16
            elif 48 <= i <= 63:
                F = C ^ (B | (~D))
                g = (7*i) % 16

            F = F + A + K[i] + words[g]
            A = D
            D = C
            C = B
            B = B + rotate_left(F, SHIFT[i])

        a0 += A
        b0 += B
        c0 += C
        d0 += D

    digest = int(''.join(str(x) for x in [a0, b0, c0, d0]))
    return hex(digest)


if __name__ == '__main__':
    m = b'abcdefghijklmnopqrstuvwxyz'
    d = md5(m)
    stop = True