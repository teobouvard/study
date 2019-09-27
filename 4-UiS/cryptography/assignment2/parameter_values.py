import keygen

START = 1000
LIMIT = 1010

if __name__ == '__main__':
    
    for i in range(START, LIMIT):
        if keygen.is_prime(i):
            print('prime : {}'.format(i), end=' roots : ')
            for j in range(i):
                if keygen.is_primitive_root(j, i):
                    print(j, end=' ')
            print()