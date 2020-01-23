// +build !solution

// Leave an empty line above this comment.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	//pb "../proto"
	pb "github.com/dat520-2020/assignments/lab2/grpc/proto"
	"google.golang.org/grpc"
)

var (
	help = flag.Bool(
		"help",
		false,
		"Show usage help",
	)
	endpoint = flag.String(
		"endpoint",
		"localhost:12111",
		"Endpoint on which server runs or to which client connects",
	)
)

func Usage() {
	fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS]\n", os.Args[0])
	fmt.Fprintf(os.Stderr, "\nOptions:\n")
	flag.PrintDefaults()
}

func main() {
	flag.Usage = Usage
	flag.Parse()
	if *help {
		flag.Usage()
		return
	}

	conn, err := grpc.Dial(*endpoint, grpc.WithInsecure())
	if err != nil {
		panic(err)
	}
	defer conn.Close()
	client := pb.NewKeyValueServiceClient(conn)

	r1, err := client.Insert(context.Background(), &pb.InsertRequest{Key: "first", Value: "42"})
	fmt.Println(r1, err)
	r2, err := client.Insert(context.Background(), &pb.InsertRequest{Key: "second", Value: "0x2A"})
	fmt.Println(r2, err)
	r3, err := client.Lookup(context.Background(), &pb.LookupRequest{Key: "first"})
	fmt.Println(r3, err)
	r4, err := client.Insert(context.Background(), &pb.InsertRequest{Key: "first", Value: "zap"})
	fmt.Println(r4, err)
	r5, err := client.Insert(context.Background(), &pb.InsertRequest{Key: "third", Value: "eaxeax"})
	fmt.Println(r5, err)
	r6, err := client.Keys(context.Background(), &pb.KeysRequest{})
	fmt.Println(r6, err)
}
