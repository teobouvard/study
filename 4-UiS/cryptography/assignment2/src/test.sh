#!/bin/bash

### TESTS ### 

printf "We verify that the Diffie-Hellman implementation is correct and works as expected by running the tests\n"
python3 keygen.py --mode "test"
printf "\n"