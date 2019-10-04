# First convert the Tux to PPM with Gimp
# Then take the header apart
head -n 4 Tux.ppm > header.txt
tail -n +5 Tux.ppm > body.bin
# Then encrypt with ECB (experiment with some different keys)
openssl enc -aes-128-ecb -nosalt -pass pass:"ANNA" -in body.bin -out body.ecb.bin
# And finally put the result together and convert to some better format with Gimp
cat header.txt body.ecb.bin > Tux.ecb.ppm