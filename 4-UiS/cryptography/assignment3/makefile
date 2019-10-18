report: clean
	mkdir teo_bouvard_a3
	cp -r src README.pdf report.pdf teo_bouvard_a3
	cd teo_bouvard_a3 && tar -czf ../teo_bouvard_a3.tar.gz *
	rm -rf teo_bouvard_a3

clean:
	rm -rf teo_bouvard_a3 teo_bouvard_a3.tar.gz

demo:
	@python3 src/main.py

test:
	@python3 src/hash.py
