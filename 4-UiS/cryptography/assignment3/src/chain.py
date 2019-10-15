import random
from datetime import datetime
from hashlib import sha3_256

def hash_function(obj):
    sha = sha3_256()
    sha.update(str(obj).encode())
    return sha.hexdigest()


USERS = ['Alice', 'Bob', 'John', 'David', 'Thomas', 'Isaac', 'Bill']


class Blockchain():

    def __init__(self):
        self.blocks = [Block()]

    def add_block(self, block):
        block.previous_hash = hash_function(self.blocks[-1])
        self.blocks.append(block)


class Block():
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.previous_hash = 'Block not been added to the chain yet !'
        self.transactions = [None]
        self.tx_root = [None]
        self.n_transactions = 0

    def add_transaction(self, transaction):
        self.transactions.append(transaction)
        self.n_transactions += 1
        self.compute_transaction_hashes()

    def compute_transaction_hashes(self, index=0):
        # balance tree
        if self.n_transactions % 2 != 0:
            self.transactions.append(Transaction())
        
        # if node has children nodes, concatenate them and hash
        if 2*index + 2 < self.n_transactions:
            concat = self.compute_transaction_hashes(2*index+1) + self.compute_transaction_hashes(2*index+2)
            self.tx_root[index] = hash_function(concat)

        else:
            index_hash = hash_function(self.transactions[index])
            if index < self.n_transactions:
                self.tx_root.append(index_hash)
            else:
                self.tx_root[index] = index_hash
            return index_hash

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = 'Block created at {}\n'.format(self.timestamp)
        s += 'Previous block hash : {}\n'.format(self.previous_hash)
        s += 'Transactions : {}\n'.format(self.transactions[1:self.n_transactions+1])
        s += 'Tx_root : {}\n'.format(self.tx_root)
        return s

class Transaction():
    counter = 0
    def __init__(self, sender=None, receiver=None, value=None):
        self.timestamp = datetime.now()
        self.index = Transaction.counter
        Transaction.counter += 1
        self.sender = sender
        self.receiver = receiver
        self.value = value

    def __str__(self):
        return str(self.timestamp)

if __name__ == '__main__':

    chain = Blockchain()
    block0 = Block()
    for i in range(1):
        users = random.sample(USERS, k=2)
        transaction = Transaction(sender=users[0], receiver=users[1], value=random.randint(1, 100))
        block0.add_transaction(transaction)

    print(block0)
    chain.add_block(block0)
    print(block0)
