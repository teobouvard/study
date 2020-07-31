package client

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"time"

	"dat520/lab5/bank"
	"dat520/lab5/multipaxos"
	"dat520/lab5/netlayer"
	"dat520/lab5/util"
)

// Client is a paxos client
type Client struct {
	valueIn     chan multipaxos.Value
	network     *netlayer.Network
	curSequence int

	ClientID  string
	mode      int
	sendTimes []time.Time
	rtt       []int
	finished  bool // finished transaction
}

// NewClient assigns the client to the network
func NewClient(network *netlayer.Network, id int, mode int) *Client {
	return &Client{
		valueIn:     make(chan multipaxos.Value),
		network:     network,
		curSequence: 1,
		ClientID:    strconv.Itoa(id),
		mode:        mode,
		finished:    true,
	}
}

// Run runs the client until interruption
func (c *Client) Run() {
	log.Printf("Starting client [%v]\n", c.network.NodeID())

	sig := make(chan os.Signal)
	signal.Notify(sig, os.Interrupt)

	//go c.requestTransaction()
	if c.mode == 0 {
		go c.eventLoopManual()
	} else {
		go c.eventLoopBenchmark()
	}

	<-sig
	fmt.Fprintf(os.Stderr, "\n")
	log.Printf("Received interrupt, shutting down client [%v]\n", c.network.NodeID())
}

// Internal : eventLoop broadcasts input value to the network and displays values voted by the network
func (c *Client) eventLoopManual() {
	if c.network == nil {
		util.Raise("Client is not connected to a network.")
	}

	go c.broadcastTransactions()

	reader := bufio.NewReader(os.Stdin)
	for {
		if c.finished {
			c.finished = false
			fmt.Fprintf(os.Stderr, "[\033[34;1m INPUT \033[0m] Enter Account Number : ")
			val, _ := reader.ReadString('\n')
			accountNr, _ := strconv.Atoi(strings.TrimSuffix(val, "\n"))

			fmt.Fprintf(os.Stderr, "[\033[34;1m INPUT \033[0m] Enter Operation Number : ")
			val, _ = reader.ReadString('\n')
			operationIn, _ := strconv.Atoi(strings.TrimSuffix(val, "\n"))

			fmt.Fprintf(os.Stderr, "[\033[34;1m INPUT \033[0m] Enter Amount : ")
			val, _ = reader.ReadString('\n')
			amountIn, _ := strconv.Atoi(strings.TrimSuffix(val, "\n"))

			c.requestTransaction(accountNr, operationIn, amountIn)
		}
	}
}

func (c *Client) eventLoopBenchmark() {
	if c.network == nil {
		util.Raise("Client is not connected to a network.")
	}

	go c.broadcastTransactions()

	for {
		if c.finished && len(c.rtt) < c.mode {
			c.finished = false
			accountNr := rand.Intn(1001)
			operationIn := rand.Intn(3)
			amountIn := rand.Intn(10001)

			c.requestTransaction(accountNr, operationIn, amountIn)
		}
	}
}

func (c *Client) broadcastTransactions() {
	// channel that notifies only this client
	resp := c.network.ListenValueClient()

	for {
		select {
		case val := <-c.valueIn:
			c.network.BroadcastRequestedValue(val)
			log.Printf("Requested Transaction: %v\n", val)
		case decidedVal := <-resp:
			now := time.Now()
			roundTripTime := now.Sub(c.sendTimes[decidedVal.ClientSeq-1])
			log.Printf("Response time %v", int(roundTripTime.Milliseconds()))
			c.rtt = append(c.rtt, int(roundTripTime.Milliseconds()))
			log.Printf("Successful Transaction: %v\n", decidedVal)
			if len(c.rtt) == c.mode {
				statistics(c.rtt)
			}
			c.finished = true
		}
	}
}

// Internal : requestTransaction reads the standard input and forwards newline
// delimited strings to the client main loop
func (c *Client) requestTransaction(accountNr int, operationIn int, amountIn int) {
	transactionIn := bank.Transaction{
		Op:     bank.Operation(operationIn),
		Amount: amountIn,
	}
	val := multipaxos.Value{
		ClientID:  c.ClientID,
		ClientSeq: c.curSequence,
		//Noop:       false,
		AccountNum: accountNr,
		Txn:        transactionIn,
	}
	// increase sequence number
	c.curSequence++
	c.sendTimes = append(c.sendTimes, time.Now())
	// broadcast value to servers
	c.valueIn <- val
}

func statistics(responsetimes []int) {
	// wait for a second, so that statistics are shown last
	//u, _ := time.ParseDuration("1s")
	//log.Printf("Waited for %v seconds", u)

	sort.Ints(responsetimes)
	sum := 0
	for _, time := range responsetimes {
		sum = sum + time
	}
	mean := sum / len(responsetimes)
	minimum := responsetimes[0]
	maximum := responsetimes[len(responsetimes)-1]
	var median int
	if len(responsetimes)%2 != 0 {
		median = responsetimes[(len(responsetimes)+1)/2]
	} else {
		median = (responsetimes[len(responsetimes)/2] + responsetimes[(len(responsetimes)+2)/2]) / 2
	}

	var sumSquaredDifference float64
	for _, time := range responsetimes {
		sumSquaredDifference += math.Pow(float64(time-mean), 2)
	}
	sampleVariance := sumSquaredDifference / float64(len(responsetimes)-1)
	sampleStdev := int(math.Sqrt(sampleVariance))
	fmt.Printf("\n#RTT statistics from %v transactions:\nMean: %vµs\nMin: %vµs\nMax: %vµs\nMedian: %vµs\nSDeviation: %vµs\n", len(responsetimes), mean, minimum, maximum, median, sampleStdev)
}
