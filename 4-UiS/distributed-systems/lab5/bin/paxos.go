package main

import (
	"flag"
	"log"

	"dat520/lab3/detector"
	"dat520/lab5/client"
	"dat520/lab5/netlayer"
	"dat520/lab5/server"
	"dat520/lab5/util"
)

var (
	config = flag.String(
		"config",
		"lab5/config.yaml", //"dat520/lab5/config.yaml",
		"YAML configuration file of nodes",
	)
	id = flag.Int(
		"id",
		detector.UnknownID,
		"ID of running node",
	)
	serve = flag.Bool(
		"server",
		false,
		"Starts a server if true, else a client",
	)
	mode = flag.Int(
		"mode",
		0,
		"Starts in manual mode if 0, else in benchmark mode with X randomly generated requests",
	)
)

func main() {
	flag.Parse()
	// use log package for printing time as prefix to output messages
	log.SetFlags(0)
	// SetPrefix sets the output prefix for the standard logger.
	log.SetPrefix("[\033[93;1m LOG \033[0m] ")

	// Updates Config, starts listening
	network := netlayer.NewNetwork(*config, *id)
	// Start the servers event loop
	network.Start()

	// *serve reads the argument given to the function by -server true/false
	if *serve {
		if !network.IsServer(*id) { // *id checks the argument given to the function by -id ...
			util.Raise("Trying to start a server from a client node")
		}
		server := server.NewServer(network)
		server.Run()
	} else {
		if network.IsServer(*id) {
			util.Raise("Trying to start a client from a server node")
		}
		client := client.NewClient(network, *id, *mode)
		client.Run()
	}
}
