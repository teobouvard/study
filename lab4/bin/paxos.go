package main

import (
	"flag"
	"log"

	"github.com/dat520-2020/TeamPilots/lab3/detector"
	"github.com/dat520-2020/TeamPilots/lab4/client"
	"github.com/dat520-2020/TeamPilots/lab4/netlayer"
	"github.com/dat520-2020/TeamPilots/lab4/server"
	"github.com/dat520-2020/TeamPilots/lab4/util"
)

var (
	config = flag.String(
		"config",
		"config.yaml",
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
)

func main() {
	flag.Parse()
	log.SetFlags(0)
	log.SetPrefix("[\033[93;1m LOG \033[0m] ")

	network := netlayer.NewNetwork(*config, *id)
	network.Start()

	if *serve {
		if !network.IsServer(*id) {
			util.Raise("Trying to start a server from a client node")
		}
		server := server.NewServer(network)
		server.Run()
	} else {
		if network.IsServer(*id) {
			util.Raise("Trying to start a client from a server node")
		}
		client := client.NewClient(network)
		client.Run()
	}

}
