package app

import (
	"fmt"
	"io/ioutil"
	"os"
	"time"

	"../detector"
	"gopkg.in/yaml.v2"
)

type Node struct {
	Hostname string
	Port     int
	ID       int
}

type Config struct {
	Nodes []Node
}

type App struct {
	id int

	fd *detector.EvtFailureDetector
	ld *detector.MonLeaderDetector
}

func NewApp(id int, config string) *App {
	nodes := parse(config)
	nodeIDs := []int{}

	for _, node := range nodes {
		fmt.Printf("[CONFIG] Node [%d] at %v:%v\n", node.ID, node.Hostname, node.Port)
		nodeIDs = append(nodeIDs, node.ID)
	}

	hbSend := make(chan<- detector.Heartbeat)
	ld := detector.NewMonLeaderDetector(nodeIDs)
	fd := detector.NewEvtFailureDetector(id, nodeIDs, ld, time.Second, hbSend)

	return &App{
		id: id,
		ld: ld,
		fd: fd,
	}
}

func (app *App) Run() {
	fmt.Fprintf(os.Stderr, "Starting up app for node %d\n", app.id)
	app.fd.Start()
}

func parse(config string) []Node {
	var c Config
	file, _ := ioutil.ReadFile(config)
	err := yaml.Unmarshal(file, &c)
	check(err)
	return c.Nodes
}

func check(err error) {
	if err != nil {
		fmt.Print(err)
	}
}
