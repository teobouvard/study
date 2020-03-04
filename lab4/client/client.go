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

// Client TODO
type Client struct {
	valueIn chan singlepaxos.Value
	network *netlayer.Network
}

// NewClient TODO
func NewClient(network *netlayer.Network) *Client {
	return &Client{
		valueIn: make(chan singlepaxos.Value),
		network: network,
	}
}

// Run starts client main loop
func (c *Client) Run() {
	log.Printf("Starting client\n")

	sig := make(chan os.Signal)
	signal.Notify(sig, os.Interrupt)

	go c.scanInputLoop()
	go c.eventLoop()

	<-sig
	fmt.Fprintf(os.Stderr, "\n")
	log.Printf("Received interrupt, shutting down client\n")
}

func (c *Client) eventLoop() {
	if c.network == nil {
		util.Raise("Client is not connected to a network.")
	}

	notify := c.network.ListenLearn()

	for {
		fmt.Fprintf(os.Stderr, "[\033[34;1m INPUT \033[0m] Enter value : ")
		select {
		case val := <-c.valueIn:
			c.network.BroadcastValue(val)
			log.Printf("Sent value : %v\n", val)
		case val := <-notify:
			log.Printf("Received value : %v\n", val)
		}
	}
}

func (c *Client) scanInputLoop() {
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		c.valueIn <- singlepaxos.Value(scanner.Text())
	}
	util.Check(scanner.Err())
}
