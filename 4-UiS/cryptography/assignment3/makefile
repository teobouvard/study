report: clean
	mkdir teo_bouvard_a3
	cp -r src makefile README.pdf report.pdf teo_bouvard_a3
	rm -rf teo_bouvard_a3/src/__pycache__
	cd teo_bouvard_a3 && tar -czf ../teo_bouvard_a3.tar.gz *
	rm -rf teo_bouvard_a3

clean:
	rm -rf teo_bouvard_a3 teo_bouvard_a3.tar.gz

demo:
	@python3 src/demo.py

tests:
	@python3 src/hash.py
