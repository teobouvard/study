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
	//subscriptions []chan int
}

func NewApp(id int, config string) *App {
	nodes := parseConfig(config)
	nodeIDs := []int{}

	for _, node := range nodes {
		fmt.Printf("[\u001b[32;1mCONFIG\u001b[0m] Node [%d] @ %v:%v\n", node.ID, node.Hostname, node.Port)
		nodeIDs = append(nodeIDs, node.ID)
		// connect to node (UDP?, TCP)
	}

	if !contains(nodeIDs, id) {
		fmt.Fprintf(os.Stderr, "[--id] is not present in config file. Exiting.\n")
		os.Exit(1)
	}

	hbSend := make(chan<- detector.Heartbeat, 100) // to not block
	ld := detector.NewMonLeaderDetector(nodeIDs)
	fd := detector.NewEvtFailureDetector(id, nodeIDs, ld, time.Second, hbSend)

	return &App{
		id: id,
		fd: fd,
		ld: ld,
		//subscriptions: make([]chan int, len(nodeIDs)),
	}
}

func (app *App) Run() {
	fmt.Fprintf(os.Stderr, "[LOG] Starting up app for node %d\n", app.id)
	app.fd.Start()
	sub := app.ld.Subscribe()
	leader := app.ld.Leader()
	fmt.Printf("[LOG] Initial leader : Node [%d]\n", leader)

	for {
		select {
		case leader := <-sub:
			fmt.Printf("[LOG] Change of leader : Node [%d] elected.\n", leader)
		}
	}

	//for _, ch := range app.subscriptions {
	//	go func() {
	//		for {
	//			select {
	//				// THINGS TO DO
	//				// app.fd.DeliverHeartbeat(hb)
	//				// app.ld.Leader()
	//				// app.ld.Subscribe()
	// 				// app.fd.Stop()
	//			}
	//		}
	//	}()
	//}

}

func parseConfig(config string) []Node {
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

func contains(arr []int, x int) bool {
	for _, e := range arr {
		if e == x {
			return true
		}
	}
	return false
}
