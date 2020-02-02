package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"

	"./app"
	"./detector"
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

func fileExists(path string) bool {
	_, err := ioutil.ReadFile(path)
	if err != nil {
		return false
	}
	return true
}

func main() {
	flag.Parse()

	if *help {
		usage()
		os.Exit(0)
	} else if *id == detector.UnknownID {
		fmt.Fprintf(os.Stderr, "[--id] parameter is required and should be a positive integer\n")
	} else if !fileExists(*config) {
		fmt.Fprintf(os.Stderr, "[--config] file does not exist or is not readable\n")
	} else {
		fmt.Fprintf(os.Stderr, "Starting up app for node %d\n", *id)
		app := app.NewApp(*id, *config)
		app.Run()
	}
}
