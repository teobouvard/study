package main

import (
	"flag"
	"fmt"
	"os"

	"../detector"
)

var (
	help = flag.Bool(
		"help",
		false,
		"Show usage help",
	)
	id = flag.Int(
		"id",
		detector.UnknownID,
		"ID of running node",
	)
	config = flag.String(
		"config",
		"config.yaml",
		"Configuration file of nodes",
	)
)

func usage() {
	fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS]\n", os.Args[0])
	fmt.Fprintf(os.Stderr, "\nOptions:\n")
	flag.PrintDefaults()
}

func main() {
	flag.Parse()

	if *help {
		usage()
		os.Exit(0)
	} else if *id == detector.UnknownID {
		fmt.Fprintf(os.Stderr, "--id parameter is required and should be a positive integer\n")
	} else {
		fmt.Println("Starting up app...")

	}
}
