from math import sin

### MD5 CONSTANTS AND INITIAL VALUES ###

K = [int(pow(2, 32) * abs(sin(i+1))) & 0xFFFFFFFF for i in range(64)]
SHIFT = 4*[7, 12, 17, 22]
SHIFT.extend(4*[5,  9, 14, 20])
SHIFT.extend(4*[4, 11, 16, 23])
SHIFT.extend(4*[6, 10, 15, 21])

INIT_BUFFER = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476]


### SUBROUTINES ###

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def pad(m):
    m = bytearray(m)
    m_bit_length = (8 * len(m)) & 0xFFFFFFFFFFFFFFFF
    m.append(0x80)
    while len(m) % 64 != 56:
        m.append(0)
    m += m_bit_length.to_bytes(8, byteorder='little')
    return m

def rotate_left(x, n):
    x &= 0xFFFFFFFF
    return ((x << n) | (x >> (32-n))) & 0xFFFFFFFF

def md5(message):
    # Constants
    global K, SHIFT, INIT_BUFFER

    # Padding step
    message = pad(message)

    # Message processing
    for chunk in chunks(message, 64):
        A, B, C, D =  INIT_BUFFER[:]

        for i in range(64):
            if 0 <= i <= 15:
                F = (B & C) | ((~B) & D)
                G = i
            elif 16 <= i <= 31:
                F = (D & B) | ((~D) & C)
                G = (5*i + 1) % 16
            elif 32 <= i <= 47:
                F = B ^ C ^ D
                G = (3*i + 5) % 16
            elif 48 <= i <= 63:
                F = C ^ (B | (~D))
                G = (7*i) % 16

            F += A + K[i] + int.from_bytes(chunk[4*G:4*G+4], byteorder='little')
            A = D
            D = C
            C = B
            B = (B + rotate_left(F, SHIFT[i])) & 0xFFFFFFFF

        for i, buffer in enumerate([A, B, C, D]):
            INIT_BUFFER[i] += buffer
            INIT_BUFFER[i] &= 0xFFFFFFFF
 
    digest_little = sum(buf << (32*i) for i, buf in enumerate(INIT_BUFFER))
    digest_bytes = digest_little.to_bytes(16, byteorder='little')
    digest_big = int.from_bytes(digest_bytes, byteorder='big')
    return hex(digest_big)
    

if __name__ == '__main__':
    m = b'abcdefghijklmnopqrstuvwxyz'
    d = md5(m)
    stop = True