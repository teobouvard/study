package main

import (
	"flag"
	"log"

	"github.com/dat520-2020/TeamPilots/lab3/detector"
	"github.com/dat520-2020/TeamPilots/lab4/netlayer"
	"github.com/dat520-2020/TeamPilots/lab4/server"
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
)

func main() {
	flag.Parse()
	log.SetFlags(0)
	log.SetPrefix("[\033[93;1m LOG \033[0m] ")

	network := netlayer.NewNetwork(*config, *id)
	network.Start()

	server := server.NewServer(network)
	server.Run()
}
