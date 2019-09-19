# Cryptography - Assignment 1

## Polyalphabetic cipher cryptoanalysis

The first part of the assignment is implemented in `cryptanalysis.py`. 
Use 
```
python3 cryptanalysis.py > output.txt && vi output.txt
```
to run it and examine the output.


## Simplified DES implementation

The second part of the assignment is implemented in `des.py`.
Use
```
python3 des.py
```
to run it. In order for the script to run in a reasonable time, the triple DES bruteforce part has been commented out. If you want to run it, uncomment line 399 of `des.py`. The script will now take some time to run, depending on your hardware, so you might want to run it in background and examine output once finished. To do so, run 
```
python3 des.py > output.txt &
```
and wait for process to finish before opening `output.txt` file.