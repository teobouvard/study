package client

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"os/signal"

	"github.com/dat520-2020/TeamPilots/lab4/netlayer"
	"github.com/dat520-2020/TeamPilots/lab4/util"
)

// Client TODO
type Client struct {
	valueIn chan string
	network *netlayer.Network
}

// NewClient TODO
func NewClient() *Client {
	return &Client{
		valueIn: make(chan string),
		network: nil,
	}
}

// Connect
func (c *Client) Connect(network *netlayer.Network) {
	c.network = network
}

// Run starts client main loop
func (c *Client) Run() {
	log.Printf("Starting client\n")

	sig := make(chan os.Signal)
	signal.Notify(sig, os.Interrupt)

	go c.scanInput()
	go c.eventLoop()

	<-sig
	log.Printf("Received interrupt, shutting down client\n")
}

func (c *Client) sendValue(value string) {
	// maybe create Value here ?
	c.network.Broadcast(value)
}

func (c *Client) eventLoop() {
	if c.network == nil {
		util.Raise("Client is not connected to a network.")
	}
	notify := c.network.Listen()
	for {
		fmt.Fprintf(os.Stderr, "[\033[34;1m INPUT \033[0m] Enter value : ")
		select {
		case val := <-c.valueIn:
			c.sendValue(val)
			log.Printf("Sent value : %v\n", val)
		case val := <-notify:
			log.Printf("Received value : %v\n", val)
		}
	}
}

func (c *Client) scanInput() {
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		c.valueIn <- scanner.Text()
	}
	util.Check(scanner.Err())
}
