import random

from blockchain import Block, Blockchain, Transaction

USERS = ['Alice', 'Bob', 'John', 'David', 'Thomas', 'Isaac', 'Bill']


if __name__ == '__main__':

    chain = Blockchain()

    for _ in range(random.randint(1, 5)):
        block = Block()

        for _ in range(random.randint(1, 3)):
            users = random.sample(USERS, k=2)
            transaction = Transaction(sender=users[0], receiver=users[1], value=random.randint(1, 100))
            block.add_transaction(transaction) 
        chain.add_block(block)
        print(chain)

