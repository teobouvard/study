package client

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"os/signal"

	"github.com/dat520-2020/TeamPilots/lab4/netlayer"
	"github.com/dat520-2020/TeamPilots/lab4/singlepaxos"
	"github.com/dat520-2020/TeamPilots/lab4/util"
)

// Client is a paxos client
type Client struct {
	valueIn chan singlepaxos.Value
	network *netlayer.Network
}

// NewClient assigns the client to the network
func NewClient(network *netlayer.Network) *Client {
	return &Client{
		valueIn: make(chan singlepaxos.Value),
		network: network,
	}
}

// Run runs the client until interruption
func (c *Client) Run() {
	log.Printf("Starting client [%v]\n", c.network.NodeID())

	sig := make(chan os.Signal)
	signal.Notify(sig, os.Interrupt)

	go c.scanInputLoop()
	go c.eventLoop()

	<-sig
	fmt.Fprintf(os.Stderr, "\n")
	log.Printf("Received interrupt, shutting down client [%v]\n", c.network.NodeID())
}

// Internal : eventLoop broadcasts input value to the network and displays values voted by the network
func (c *Client) eventLoop() {
	if c.network == nil {
		util.Raise("Client is not connected to a network.")
	}

	voted := c.network.ListenValue()

	for {
		fmt.Fprintf(os.Stderr, "[\033[34;1m INPUT \033[0m] Enter value : ")
		select {
		case val := <-c.valueIn:
			c.network.BroadcastRequestedValue(val)
			log.Printf("Sent value : %v\n", val)
		case val := <-voted:
			fmt.Println()
			log.Printf("Voted value : %v\n", val)
		}
	}
}

// Internal : scanInputLoop reads the standard input and forwards newline
// delimited strings to the client main loop
func (c *Client) scanInputLoop() {
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		c.valueIn <- singlepaxos.Value(scanner.Text())
	}
	util.Check(scanner.Err())
}
