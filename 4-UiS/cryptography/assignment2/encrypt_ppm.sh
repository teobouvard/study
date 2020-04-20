# First convert the Tux to PPM with Gimp
# Then take the header apart
head -n 4 img/enigma.ppm > img/header.txt
tail -n +5 img/enigma.ppm > img/body.bin
# Then encrypt with ECB (experiment with some different keys)
openssl enc -aes128 -in img/body.bin -out img/body.ecb.bin
# And finally put the result together and convert to some better format with Gimp
cat img/header.txt img/body.ecb.bin > img/encrypted_enigma_cbc.ppm