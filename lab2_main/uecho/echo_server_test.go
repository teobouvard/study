package main

import (
	"io/ioutil"
	"log"
	"strings"
	"testing"
)

func init() {
	log.SetOutput(ioutil.Discard)
}

var testRequests = []struct {
	cmd, in, out string
	err          error
}{
	{"UPPER", "i want to be upper case", "I WANT TO BE UPPER CASE", nil},
	{"UPPER", "I WANT TO BE UPPER CASE", "I WANT TO BE UPPER CASE", nil},
	{"LOWER", "I WANT TO BE LOWER CASE", "i want to be lower case", nil},
	{"LOWER", "i want to be lower case", "i want to be lower case", nil},
	{"CAMEL", "i want to be camel case", "I Want To Be Camel Case", nil},
	{"CAMEL", "I WANT TO BE CAMEL CASE", "I Want To Be Camel Case", nil},
	{"CAMEL", "I Want To Be Camel Case", "I Want To Be Camel Case", nil},
	{"SWAP", "I WANT TO BE ANOTHER CASE", "i want to be another case", nil},
	{"SWAP", "i want to be another case", "I WANT TO BE ANOTHER CASE", nil},
	{"SWAP", "I Want To Be Another Case", "i wANT tO bE aNOTHER cASE", nil},
	{"UPPER", "i want to be: upper case", "I WANT TO BE: UPPER CASE", nil},
	{"ROT13", "i want to be rot13", "v jnag gb or ebg13", nil},
}

func TestEchoServerProtocol(t *testing.T) {
	const addr = "localhost:12111"
	server, err := NewUDPServer(addr)
	if err != nil {
		t.Fatalf("TestEchoServer: got error when setting up server: %v", err)
	}
	go server.ServeUDP()
	for i, tr := range testRequests {
		res, err := SendCommand(addr, tr.cmd, tr.in)
		if err != tr.err {
			t.Errorf("TestEchoServer %d: got error: %v, want %v", i, err, tr.err)
		}
		if res != tr.out {
			t.Errorf("TestEchoServer %d: got %q, want %q", i, res, tr.out)
		}
	}
}

func TestPreAndPostSetup(t *testing.T) {
	const addr = "localhost:12112"
	_, err := SendCommand(addr, "UPPER", "FooBar")
	if err == nil {
		t.Fatalf("TestPreAndPostSetup: want error for SendCommand pre server setup, got none")
	}
	if !(strings.Contains(err.Error(), "connection refused") || strings.Contains(err.Error(), "WSARecv")) {
		t.Fatalf("TestPrePostSetup: got error pre server setup, but want connection refused or WSARecv error")
	}
	server, err := NewUDPServer(addr)
	if err != nil {
		t.Fatalf("TestPrePostSetup: got error when setting up server: %v", err)
	}
	go server.ServeUDP()
	_, err = SendCommand(addr, "UPPER", "FooBar")
	if err != nil {
		t.Fatalf("TestPrePostSetup: want no error for SendCommand post setup, got %v", err)
	}
}

var testMalformedRequests = []struct {
	cmd, in string
	err     error
}{
	{"UPPERZ", "Some payload", nil},
	{"Hello", "World", nil},
	{"UPPER LOWER ROT13", "i want to be upper case", nil},
	{"UPPERLOWERROT13", "i want to be upper case", nil},
	{"SWAP|:|", "abcdefghiklm", nil},
}

func TestMalformedRequest(t *testing.T) {
	const (
		addr        = "localhost:12113"
		badReqReply = "Unknown command"
	)
	server, err := NewUDPServer(addr)
	if err != nil {
		t.Fatalf("TestEchoServer: got error when setting up server: %v", err)
	}
	go server.ServeUDP()
	for i, tr := range testMalformedRequests {
		res, err := SendCommand(addr, tr.cmd, tr.in)
		if err != tr.err {
			t.Errorf("TestMalformedRequest %d: got error: %v, want %v", i, err, tr.err)
		}
		if res != badReqReply {
			t.Errorf("TestMalformedRequest %d: got %q, want %q", i, res, badReqReply)
		}
	}
}
