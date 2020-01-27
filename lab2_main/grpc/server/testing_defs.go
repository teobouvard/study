package main

// DO NOT EDIT

type Command int

const (
	INSERT Command = iota
	LOOKUP
	KEYS
)

type gRPCTest struct {
	server  *keyValueServicesServer
	desc    string // Test description
	setup   []gRPCAction
	actions []gRPCAction
}

type gRPCAction struct {
	command Command // INSERT, LOOKUP, or KEYS
	input   string  // "key,value" for INSERT, "key" for LOOKUP, empty string for KEYS
	want    string  // "true" or "false" for INSERT, "value" for LOOKUP, "key1,key2,..." for KEYS
}
