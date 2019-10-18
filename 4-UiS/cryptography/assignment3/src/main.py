import random

from blockchain import Block, Blockchain, Transaction

USERS = ['Alice', 'Bob', 'John', 'David', 'Thomas', 'Isaac', 'Bill']


if __name__ == '__main__':

    chain = Blockchain()

    for _ in range(random.randint(1, 5)):
        block = Block()

        for _ in range(random.randint(1, 3)):
            sender, receiver = random.sample(USERS, k=2)
            transaction = Transaction(sender=sender, receiver=receiver, value=random.randint(1, 100))
            block.add_transaction(transaction)
            #print(block)
        chain.add_block(block)
        print(chain)

