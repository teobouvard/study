package app

import (
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"time"

	"github.com/dat520-2020/assignments/lab3/detector"
	"github.com/dat520-2020/TeamPilots/lab3/network"
)

type App struct {
	id       int
	fd       *detector.EvtFailureDetector
	ld       *detector.MonLeaderDetector
	registry map[int]*net.UDPAddr
	server   *network.Server
	hbSend   chan detector.Heartbeat
}

func NewApp(id int, configFile string) *App {
	config := Parse(configFile)
	nodeIDs := []int{}
	registry := make(map[int]*net.UDPAddr)

	for _, node := range config {
		fmt.Fprintf(os.Stderr, "[\033[32;1m CONFIG \033[0m] Node [%d] @ %v:%v\n", node.ID, node.Hostname, node.Port)
		addr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("%s:%d", node.Hostname, node.Port))
		Check(err)
		registry[node.ID] = addr
		nodeIDs = append(nodeIDs, node.ID)
	}

	if !Contains(nodeIDs, id) {
		Raise(fmt.Sprintf("Node [%d] is not present in config file. Change --id parameter\n", id))
	}

	hbSend := make(chan detector.Heartbeat, 100) // to not block
	ld := detector.NewMonLeaderDetector(nodeIDs)
	fd := detector.NewEvtFailureDetector(id, nodeIDs, ld, time.Second, hbSend)
	server, err := network.NewServer(registry[id])
	Check(err)

	return &App{
		id:       id,
		fd:       fd,
		ld:       ld,
		registry: registry,
		server:   server,
		hbSend:   hbSend,
	}
}

// Run runs app main loop
func (app *App) Run() {
	log.Printf("Starting up app for node %d\n", app.id)
	app.fd.Start()
	sub := app.ld.Subscribe()
	notify := app.server.Listen()

	sig := make(chan os.Signal, 10)
	signal.Notify(sig, os.Interrupt)

	go func() {
		for {
			select {
			case leader := <-sub:
				log.Printf("Change of leader : Node [%d] elected.\n", leader)
			case hb := <-notify:
				app.fd.DeliverHeartbeat(hb)
			case hb := <-app.hbSend:
				addr := app.registry[hb.To]
				n, err := app.server.Send(addr, hb)
				if err != nil || n <= 0 {
					panic(err)
				}
			}
		}
	}()

	<-sig
	log.Printf("Received interrupt, shutting down app for node %d\n", app.id)
	app.ld.Unsubscribe(sub)
	app.server.Stop()
	app.fd.Stop()
}
