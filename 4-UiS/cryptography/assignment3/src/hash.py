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
    """
    Generates chunks of size n from iterable l
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def pad(m):
    """
    Returns the padded message according to MD5 specification
    """
    # convert bytes message to array of bytes
    m = bytearray(m)
    m_bit_length = (8 * len(m)) & 0xFFFFFFFFFFFFFFFF

    # append '1' bit once
    m.append(0x80)

    # append '0' bits until 8 bytes left
    while len(m) % 64 != 56:
        m.append(0)

    # append 8 LSB of message length
    m += m_bit_length.to_bytes(8, byteorder='little')
    return m

def rotate_left(x, n):
    """
    Returns x value rotated left n times with wrapping of 2^32
    """
    x &= 0xFFFFFFFF
    return ((x << n) | (x >> (32-n))) & 0xFFFFFFFF

def md5(message):
    # Constants
    global K, SHIFT, INIT_BUFFER
    buffer = INIT_BUFFER.copy()

    # Padding step
    message = pad(message)

    # Message processing
    for chunk in chunks(message, 64):
        A, B, C, D = buffer[:]

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

        for i, val in enumerate([A, B, C, D]):
            buffer[i] += val
            buffer[i] &= 0xFFFFFFFF
 
    digest_little = sum(buf << (32*i) for i, buf in enumerate(buffer))
    digest_bytes = digest_little.to_bytes(16, byteorder='little')
    digest_big = int.from_bytes(digest_bytes, byteorder='big')
    return f'0x{hex(digest_big)[2:].zfill(32)}'
    

if __name__ == '__main__':
    test_cases = {
        b'' : '0xd41d8cd98f00b204e9800998ecf8427e',
        b'a' : '0x0cc175b9c0f1b6a831c399e269772661',
        b'abc' : '0x900150983cd24fb0d6963f7d28e17f72',
        b'message digest' : '0xf96b697d7cb7938d525a2f31aaf161d0',
        b'abcdefghijklmnopqrstuvwxyz' : '0xc3fcd3d76192e4007dfb496cca67e13b',
        b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789' : '0xd174ab98d277d9f5a5611c2c9f419d9f',
        b'12345678901234567890123456789012345678901234567890123456789012345678901234567890' : '0x57edf4a22be3c955ac49da2e2107b67a',
    }

    passed_tests = 0
    for message, test_hash in test_cases.items():
        message_hash = md5(message)
        if int(message_hash, 16) == int(test_hash, 16):
            result = 'OK'
            passed_tests += 1
        else:
            result = 'ERROR'
        print('{} -> {} : {}'.format(message, message_hash, result))
    print('PASSED TESTS : {}/{}'.format(passed_tests, len(test_cases.keys())))