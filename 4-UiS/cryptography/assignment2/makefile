report: purge clean
	mkdir teo_bouvard_a2
	cp -r src README.pdf report.pdf teo_bouvard_a2
	cd teo_bouvard_a2 && tar -czf ../teo_bouvard_a2.tar.gz *
	rm -rf teo_bouvard_a2

clean:
	rm -rf teo_bouvard_a2 teo_bouvard_a2.tar.gz

purge:
	rm -rf src/keys
	rm -rf src/img
	rm src/files/encrypted_file.bin src/files/decrypted_file.txt

demo:
	cd src && ./demo.sh

test:
	cd src && ./test.sh