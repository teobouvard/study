# Get started

```{shell}
cd src
pip3 install pycryptodome
./test.sh
./demo.sh
```

# Directory structure
.
├── README.md
├── report.pdf
└── src
    ├── bbs.py
    ├── cipher.py
    ├── demo.sh
    ├── files
    │   ├── 2048-bit MODP Group
    │   │   ├── generator.txt
    │   │   ├── prime.txt
    │   │   ├── test_xA.txt
    │   │   ├── test_xB.txt
    │   │   ├── test_yA.txt
    │   │   ├── test_yB.txt
    │   │   └── test_Z.txt
    │   └── really_secret_file.txt
    ├── keygen.py
    └── test.sh

# Files

- README.md : this file
- report.pdf : report of the assignment

- src/ : source code directory
    - bbs.py : Python implementation of the Blum Blum Shub PRNG
    - cipher.py : Python implementation of AES encryption/decryption
    - keygen.py : Python implementation of the Diffie-Hellman key exchange scheme

- files/ : a directory containing various files used by the different tools
    - really_secret_file.txt : the file Alice wishes to send to Bob without disclosing its contents
    - 2048-bit MODP Group/ : 