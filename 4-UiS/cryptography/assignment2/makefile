report: clean
	mkdir teo_bouvard_a2
	cp -r src files README.md report.pdf teo_bouvard_a2
	cd teo_bouvard_a2 && tar -czf ../teo_bouvard_a2.tar.gz *
	rm -rf teo_bouvard_a2
clean:
	rm -rf teo_bouvard_a2 teo_bouvard_a2.tar.gz

purge:
	rm -rf keys
	rm files/encrypted_file.bin files/decrypted_file.txt