package netlayer

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"log"
	"net"
	"os"

	"github.com/dat520-2020/TeamPilots/lab4/util"
)

const (
	heartbeatMessage byte = iota
	prepareMessage
	promiseMessage
	acceptMessage
	learnMessage
)

// Network TODO
type Network struct {
	conn *net.UDPConn

	registry map[int]*net.UDPAddr
	nodeIDs  []int
	notify   chan string
}

// NewNetwork creates a network from the config file passed
func NewNetwork(configFile string, id int) *Network {
	config := Parse(configFile)
	nodeIDs := []int{}
	registry := make(map[int]*net.UDPAddr)

	for _, node := range config {
		fmt.Fprintf(os.Stderr, "[\033[32;1m CONFIG \033[0m] Node [%d] @ %v:%v\n", node.ID, node.Hostname, node.Port)
		addr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("%s:%d", node.Hostname, node.Port))
		util.Check(err)
		registry[node.ID] = addr
		nodeIDs = append(nodeIDs, node.ID)
	}

	if !util.Contains(nodeIDs, id) {
		util.Raise(fmt.Sprintf("Node [%d] is not present in config file. Make sure to correctly set --id parameter.\n", id))
	}

	conn, err := net.ListenUDP("udp", registry[id])
	util.Check(err)

	return &Network{
		registry: registry,
		nodeIDs:  nodeIDs,
		conn:     conn,
	}
}

// Broadcast sends a value to all nodes on the network
func (n *Network) Broadcast(value string) {

}

// Start starts the server event loop
func (n *Network) Start() {
	go n.eventLoop()
}

// Listen returns a channel
func (n *Network) Listen() <-chan string {
	return n.notify
}

func (n *Network) eventLoop() {
	var buf [512]byte
	for {
		nbytes, _, err := n.conn.ReadFromUDP(buf[:])
		util.Check(err)
		decoder := gob.NewDecoder(bytes.NewReader(buf[:nbytes]))
		_ = decoder

		switch buf[0] {
		case heartbeatMessage:
			log.Println("Heartbeat")
		case prepareMessage:
			log.Println("PrepareMessage")
		default:
			log.Println("Unknown Message")
		}

		/*
			var hb detector.Heartbeat
			decoder := gob.NewDecoder(bytes.NewReader(buf[:nbytes]))
			err = decoder.Decode(&hb)
			util.Check(err)
			n.notify <- hb
		*/
	}
}
