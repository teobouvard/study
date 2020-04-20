import random
import time

from blockchain import Block, Blockchain, Transaction

USERS = ['Alice', 'James', 'Smith', 'David', 'Adams', 'Isaac', 'Lewis']

SLEEP_TIME = 0


if __name__ == '__main__':

    chain = Blockchain()

    for _ in range(random.randint(1, 5)):
        block = Block()

        for _ in range(random.randint(1, 3)):
            sender, receiver = random.sample(USERS, k=2)
            transaction = Transaction(sender=sender, receiver=receiver, value=random.randint(10, 100))
            block.add_transaction(transaction)
            time.sleep(SLEEP_TIME)
        chain.add_block(block)
        print(chain)
        time.sleep(SLEEP_TIME)
