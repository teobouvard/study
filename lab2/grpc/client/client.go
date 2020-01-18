// +build !solution

// Leave an empty line above this comment.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	pb "../proto"
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
	req := &pb.InsertRequest{"test", "test"}
	a, err := client.Insert(context.Background(), req)
	fmt.Println(a)
}
