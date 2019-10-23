from datetime import datetime
from hash import md5

HSEP = '■'
RVSEP = '▐'
LVSEP =  '▌'
VERBOSE = True

def hash_function(message):
    if type(message) == bytes:
        return md5(message)
    elif type(message) == str:
        return md5(message.encode())
    else:
        raise ValueError(f'MD5 implementation can only hash strings and bytes, not {type(message)}')

class Blockchain():

    def __init__(self):
        self.blocks = []
        if VERBOSE:
            print('Blockchain created\n')

    def add_block(self, block):
        block.previous_hash = hash_function(self.blocks[-1].to_hash()) if len(self.blocks) != 0 else '0x0'
        self.blocks.append(block)
        if VERBOSE:
            print(f'Block #{block.block_index} added to the chain\n')

    def __repr__(self):
        return str(self)

    def __str__(self):
        if len(self.blocks) == 0:
            return 'Empty chain ! \n\n'
        elif len(self.blocks) == 1:
            s = 'Blockchain contains 1 block\n\n'
        else:
            s = f'Blockchain contains {len(self.blocks)} blocks\n\n'
        for block in self.blocks[:0:-1]:
            s += 100*HSEP + '\n'
            s += str(block)
            s += f'{LVSEP}{RVSEP:>99}\n{100*HSEP}\n'
            s += 2*f'{50*" "}|\n'
            s += f'{50*" "}V\n'
        s += f'{100*HSEP}\n'
        s += str(self.blocks[0])
        s += f'{LVSEP}{RVSEP:>99}\n{100*HSEP}\n'
        return s


class Block():
    block_index = 0
    def __init__(self):
        Block.block_index +=1
        self.block_index = Block.block_index
        self.timestamp = datetime.now()
        self.previous_hash = 'Block not been added to the chain yet !'
        self.tx_root = MerkleTree()
        if VERBOSE:
            print(f'New block (#{self.block_index}) created at {self.timestamp.strftime("%H:%M:%S.%f on %D")}', end='\n\n')

    def add_transaction(self, transaction):
        self.tx_root.add(transaction)
        print(f'Transaction {transaction.hash} added to block {self.block_index}\n')

    def to_hash(self):
        return f'{self.timestamp} {self.previous_hash} {self.tx_root.to_hash()}'

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = f'{LVSEP} {" ":^18} Block {self.block_index} - {hash_function(self.to_hash())} - {self.tx_root.n_transactions:2} transactions {RVSEP:>16}\n{LVSEP}{RVSEP:>99}\n'
        s += f'{LVSEP} Timestamp                  : {self.timestamp.strftime("%H:%M:%S.%f %D")} {RVSEP:>44}\n'
        s += f'{LVSEP} Previous block hash        : {self.previous_hash:34} {RVSEP:>34}\n'
        s += f'{LVSEP} Transaction root hash      : {self.tx_root.root} {RVSEP:>34}\n'
        for transaction in self.tx_root.transactions:
            s+= f'{LVSEP} {transaction} {RVSEP:>34}\n'
        return s


class Transaction():
    def __init__(self, sender=None, receiver=None, value=None):
        self.timestamp = datetime.now()
        self.sender = sender
        self.receiver = receiver
        self.value = value
        self.hash = hash_function(self.to_hash())
        if VERBOSE:
            print(f'New transaction')
            print(f'Timestamp : {self.timestamp.strftime("%H:%M:%S.%f - %D")}')
            print(f'Sender    : {self.sender}')
            print(f'Receiver  : {self.receiver}')
            print(f'Value     : {self.value}')
            print(f'Hash      : {self.hash}\n')
    
    def to_hash(self):
        return f'{self.timestamp} {self.value} {self.sender} {self.receiver}'

    def __str__(self):
        return f'{self.timestamp} : {self.value} transferred from {self.sender} to {self.receiver}'

    
class MerkleTree():
    def __init__(self):
        self.transactions = []
        self.hashes = []
        self.n_transactions = 0
        self.root = 0

    def add(self, transaction):
        self.transactions.append(transaction)
        self.hashes.append(hash_function(transaction.to_hash()))
        self.n_transactions += 1
        self.compute_root(self.hashes)
    
    def to_hash(self):
        return f'{[x.to_hash() for x in self.transactions]} {self.hashes} {self.n_transactions} {self.root}'

    def compute_root(self, hashlist):
        pair_hashes = []

        for i, current_hash in enumerate(hashlist[::2]):
            if i < self.n_transactions - 1:
                pair_hashes.append(hash_function(current_hash + self.hashes[i+1]))
            else:
                pair_hashes.append(hash_function(current_hash))

        if len(pair_hashes) > 1:
            self.compute_root(pair_hashes)
        else:
            self.root = pair_hashes[0]
