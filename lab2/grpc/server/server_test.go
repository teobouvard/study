package main

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/net/context"

	pb "github.com/dat520-2020/assignments/lab2/grpc/proto"
)

func TestGRPCServer(t *testing.T) {
	testGRPCCalls(lookupTests, "Lookup Tests", t)
	testGRPCCalls(keysTests, "Keys Tests", t)
}

//**************************************************************************************************************
// This function performs all the tests listed in the slice of gRPCTests
// Input: tests      A slice of gRPCTests to perform
//        testGroupName The name of the group of tests
//        t          A pointer to the testing struct
//**************************************************************************************************************
func testGRPCCalls(tests []gRPCTest, testGroupName string, t *testing.T) {
	for i, test := range tests {
		server := test.server
		server.kv = make(map[string]string)

		setupSuccessful := true
		for j, action := range test.setup {
			desc := fmt.Sprintf("\n%s\ntest nr: %d\ndescription: %s\naction nr: %d", testGroupName, i+1, test.desc, j+1)
			switch {
			case action.command == INSERT:
				setupSuccessful = performInsert(&action, desc, server, t)
			case action.command == LOOKUP:
				setupSuccessful = performLookup(&action, desc, server, t)
			case action.command == KEYS:
				setupSuccessful = performKeys(&action, desc, server, t)
			}

			if setupSuccessful == false {
				break // break out of setup
			}
		}

		if setupSuccessful == false {
			break // break out of this test
		}

		for j, action := range test.actions {
			desc := fmt.Sprintf("\n%s\ntest nr: %d\ndescription: %s\naction nr: %d", testGroupName, i+1, test.desc, j+1)
			switch {
			case action.command == INSERT:
				performInsert(&action, desc, server, t)
			case action.command == LOOKUP:
				performLookup(&action, desc, server, t)
			case action.command == KEYS:
				performKeys(&action, desc, server, t)
			}
		}
	}
}

//**************************************************************************************************************
// This function calls the Insert() function on the server. It updates the score if this is not a setup command.
// Input: action     A pointer to the gRPCAction to perform
//        desc       A description to print out in case of error
//        server     A pointer to the keyValueServicesServer
//        t          A pointer to the testing struct
// Output: A boolean value representing success (true) or failure (false)
//**************************************************************************************************************
func performInsert(action *gRPCAction, desc string, server *keyValueServicesServer, t *testing.T) bool {

	input := strings.Split(action.input, ",")
	insertReq := pb.InsertRequest{Key: input[0], Value: input[1]}
	insertRsp, err := server.Insert(context.Background(), &insertReq)
	if err != nil {
		t.Errorf("%s\nError: %v", desc, err)
		return false
	}

	got := strconv.FormatBool(insertRsp.Success)
	if got != action.want {
		t.Errorf("%s\nWant: %v\nGot: %v", desc, action.want, got)
		return false
	}

	return true
}

//**************************************************************************************************************
// This function calls the Lookup() function on the server. It updates the score if this is not a setup command.
// Input: action     A pointer to the gRPCAction to perform
//        desc       A description to print out in case of error
//        server     A pointer to the keyValueServicesServer
//        t          A pointer to the testing struct
// Output: A boolean value representing success (true) or failure (false)
//**************************************************************************************************************
func performLookup(action *gRPCAction, desc string, server *keyValueServicesServer, t *testing.T) bool {

	lookupReq := pb.LookupRequest{Key: action.input}
	lookupRsp, err := server.Lookup(context.Background(), &lookupReq)
	if err != nil {
		t.Errorf("%s\nError: %v", desc, err)
		return false
	}

	got := lookupRsp.Value
	if got != action.want {
		t.Errorf("%s\nWant: %v\nGot: %v", desc, action.want, got)
		return false
	}

	return true
}

//**************************************************************************************************************
// This function calls the Keys() function on the server. It updates the score if this is not a setup command.
// Input: action     A pointer to the gRPCAction to perform
//        desc       A description to print out in case of error
//        server     A pointer to the keyValueServicesServer
//        t          A pointer to the testing struct
// Output: A boolean value representing success (true) or failure (false)
//**************************************************************************************************************
func performKeys(action *gRPCAction, desc string, server *keyValueServicesServer, t *testing.T) bool {

	keysReq := pb.KeysRequest{}
	keysRsp, err := server.Keys(context.Background(), &keysReq)
	if err != nil {
		t.Errorf("%s\nError: %v", desc, err)
		return false
	}

	got := keysRsp.Keys
	sort.Strings(got)
	var want []string
	if len(action.want) == 0 {
		want = nil
	} else {
		want = strings.Split(action.want, ",")
		sort.Strings(want)
	}

	// Check that the number of keys match what is expected
	if len(got) != len(want) {
		t.Errorf("%s\nWant: %v\nGot: %v", desc, want, got)
		return false
	}

	// Check that the keys match what is expected
	for index := range got {
		if got[index] != want[index] {
			t.Errorf("%s\nWant: %v\nGot: %v", desc, want, got)
			return false
		}
	}

	return true
}

var lookupTests = []gRPCTest{
	{
		new(keyValueServicesServer),
		"Lookup After Zero Key/Values Inserted",
		nil,
		[]gRPCAction{
			{
				LOOKUP,
				"1",
				"",
			},
		},
	},
	{
		new(keyValueServicesServer),
		"Lookup After One Key/Value Inserted",
		[]gRPCAction{
			{
				INSERT,
				"1,one",
				"true",
			},
		},
		[]gRPCAction{
			{
				LOOKUP,
				"1",
				"one",
			},
		},
	},
	{
		new(keyValueServicesServer),
		"Lookup After Two Key/Values Inserted",
		[]gRPCAction{
			{
				INSERT,
				"1,one",
				"true",
			},
			{
				INSERT,
				"2,two",
				"true",
			},
		},
		[]gRPCAction{
			{
				LOOKUP,
				"2",
				"two",
			},
		},
	},
	{
		new(keyValueServicesServer),
		"Lookup After Two Key/Values Inserted: Same key, different values",
		[]gRPCAction{
			{
				INSERT,
				"1,one",
				"true",
			},
			{
				INSERT,
				"1,one again",
				"true",
			},
		},
		[]gRPCAction{
			{
				LOOKUP,
				"1",
				"one again",
			},
		},
	},
	{
		new(keyValueServicesServer),
		"Lookup After Three Key/Values Inserted",
		[]gRPCAction{
			{
				INSERT,
				"1,one",
				"true",
			},
			{
				INSERT,
				"2,two",
				"true",
			},
			{
				INSERT,
				"3,three",
				"true",
			},
		},
		[]gRPCAction{
			{
				LOOKUP,
				"3",
				"three",
			},
		},
	},
}

var keysTests = []gRPCTest{
	{
		new(keyValueServicesServer),
		"Keys After No Key/Values Inserted",
		nil,
		[]gRPCAction{
			{
				KEYS,
				"",
				"",
			},
		},
	},
	{
		new(keyValueServicesServer),
		"Keys After One Key/Value Inserted",
		[]gRPCAction{
			{
				INSERT,
				"1,one",
				"true",
			},
		},
		[]gRPCAction{
			{
				KEYS,
				"",
				"1",
			},
		},
	},
	{
		new(keyValueServicesServer),
		"Keys After Two Key/Values Inserted",
		[]gRPCAction{
			{
				INSERT,
				"1,one",
				"true",
			},
			{
				INSERT,
				"2,two",
				"true",
			},
		},
		[]gRPCAction{
			{
				KEYS,
				"",
				"1,2",
			},
		},
	},
	{
		new(keyValueServicesServer),
		"Keys After Two Key/Values Inserted: Same Key Twice",
		[]gRPCAction{
			{
				INSERT,
				"1,one",
				"true",
			},
			{
				INSERT,
				"1,one again",
				"true",
			},
		},
		[]gRPCAction{
			{
				KEYS,
				"",
				"1",
			},
		},
	},
	{
		new(keyValueServicesServer),
		"Keys After Three Key/Values Inserted",
		[]gRPCAction{
			{
				INSERT,
				"1,one",
				"true",
			},
			{
				INSERT,
				"2,two",
				"true",
			},
			{
				INSERT,
				"3,three",
				"true",
			},
		},
		[]gRPCAction{
			{
				KEYS,
				"",
				"1,2,3",
			},
		},
	},
}
