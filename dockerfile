FROM golang:latest

RUN apt-get update && apt-get -y install vim
RUN go get gopkg.in/yaml.v2


WORKDIR /home
COPY lab3 .
RUN go build

EXPOSE 8081:8085