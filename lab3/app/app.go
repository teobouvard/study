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

	subscriptions []chan int
}

func NewApp(id int, config string) *App {
	nodes := parse(config)
	nodeIDs := []int{}

	for _, node := range nodes {
		fmt.Printf("[\u001b[32;1mCONFIG\u001b[0m] Node [%d] @ %v:%v\n", node.ID, node.Hostname, node.Port)
		nodeIDs = append(nodeIDs, node.ID)
		// connect to node (UDP?, TCP)
	}

	hbSend := make(chan<- detector.Heartbeat)
	ld := detector.NewMonLeaderDetector(nodeIDs)
	fd := detector.NewEvtFailureDetector(id, nodeIDs, ld, time.Second, hbSend)

	return &App{
		id:            id,
		ld:            ld,
		fd:            fd,
		subscriptions: make([]chan int, len(nodeIDs)),
	}
}

func (app *App) Run() {
	fmt.Fprintf(os.Stderr, "Starting up app for node %d\n", app.id)
	app.fd.Start()

	//for _, ch := range app.subscriptions {
	//	go func() {
	//		for {
	//			select {
	//				// THINGS TO DO
	//				// app.fd.DeliverHeartbeat(hb)
	//				// app.ld.Subscribe()
	//			}
	//		}
	//	}()
	//}
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
