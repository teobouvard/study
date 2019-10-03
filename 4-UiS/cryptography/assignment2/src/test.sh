#!/bin/sh

### TESTS ### 

printf "We verify that the Diffie-Hellman implementation is correct and works as expected by running the tests\n"
python3 keygen.py --mode "test"
printf "\n"

printf "We visually check for pseudo-randomness\n"
python3 bbs.py --mode "test" --seed 0xF42F42F42
printf "\n"