.PHONY: network

run:
	docker-compose up -d

build:
	docker-compose build

network:
	docker network create --subnet=192.168.0.0/16 --internal lab3

clean:
	docker system prune
