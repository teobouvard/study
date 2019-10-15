report: clean
	mkdir teo_bouvard_a3
	cp -r src README.pdf report.pdf teo_bouvard_a3
	cd teo_bouvard_a3 && tar -czf ../teo_bouvard_a3.tar.gz *
	rm -rf teo_bouvard_a3

clean:
	rm -rf teo_bouvard_a3 teo_bouvard_a3.tar.gz


demo:
	cd src && ./demo.sh

test:
	cd src && ./test.sh

dev:
	python3 src/chain.py