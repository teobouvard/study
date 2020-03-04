package netlayer

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"net"
	"os"

	"github.com/dat520-2020/TeamPilots/lab3/detector"
	"github.com/dat520-2020/TeamPilots/lab4/singlepaxos"
	"github.com/dat520-2020/TeamPilots/lab4/util"
)

// byte constants for different message types
const (
	heartbeatMessage byte = iota
	valueMessage
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

	notifyHb      chan detector.Heartbeat
	notifyValue   chan singlepaxos.Value
	notifyPrepare chan singlepaxos.Prepare
	notifyPromise chan singlepaxos.Promise
	notifyAccept  chan singlepaxos.Accept
	notifyLearn   chan singlepaxos.Learn
}

// NewNetwork creates a network from the config file passed
func NewNetwork(configFile string, id int) *Network {
	nodeIDs := []int{}
	registry := make(map[int]*net.UDPAddr)

	for _, node := range Parse(configFile) {
		fmt.Fprintf(os.Stderr, "[\033[32;1m CONFIG \033[0m] Node [%d] @ %v:%v\n", node.ID, node.Hostname, node.Port)
		addr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("%s:%d", node.Hostname, node.Port))
		util.Check(err)
		registry[node.ID] = addr
		nodeIDs = append(nodeIDs, node.ID)
	}

	if !util.Contains(nodeIDs, id) {
		util.Raise(fmt.Sprintf("Node [%d] is not present in config file. Make sure to correctly set the --id flag.\n", id))
	}

	conn, err := net.ListenUDP("udp", registry[id])
	util.Check(err)

	return &Network{
		registry: registry,
		nodeIDs:  nodeIDs,
		conn:     conn,

		notifyHb:      make(chan detector.Heartbeat),
		notifyValue:   make(chan singlepaxos.Value),
		notifyPrepare: make(chan singlepaxos.Prepare),
		notifyPromise: make(chan singlepaxos.Promise),
		notifyAccept:  make(chan singlepaxos.Accept),
		notifyLearn:   make(chan singlepaxos.Learn),
	}
}

// BroadcastValue sends a value to all nodes on the network
func (net *Network) BroadcastValue(value singlepaxos.Value) {
	for _, id := range net.nodeIDs {
		net.sendValue(id, value)
	}
}

// SendValue TODO
func (net *Network) sendValue(id int, value singlepaxos.Value) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err := encoder.Encode(value)
	util.Check(err)
	encoded := append([]byte{valueMessage}, buf.Bytes()...)
	_, err = net.conn.WriteTo(encoded, net.registry[id])
	util.Check(err)
}

// Start starts the server event loop
func (net *Network) Start() {
	go net.eventLoop()
}

// ListenLearn returns a notification channel for learn messages
func (net *Network) ListenLearn() <-chan singlepaxos.Learn {
	return net.notifyLearn
}

// ListenValue returns a notification channel for Value messages
func (net *Network) ListenValue() <-chan singlepaxos.Value {
	return net.notifyValue
}

func (net *Network) eventLoop() {
	var buf [512]byte
	for {
		n, _, err := net.conn.ReadFromUDP(buf[:])
		util.Check(err)
		decoder := gob.NewDecoder(bytes.NewReader(buf[1:n]))
		switch buf[0] {
		case heartbeatMessage:
			var hb detector.Heartbeat
			err = decoder.Decode(&hb)
			util.Check(err)
			net.notifyHb <- hb
		case valueMessage:
			var val singlepaxos.Value
			err = decoder.Decode(&val)
			util.Check(err)
			net.notifyValue <- val
		case prepareMessage:
			var prp singlepaxos.Prepare
			err = decoder.Decode(&prp)
			util.Check(err)
			net.notifyPrepare <- prp
		case promiseMessage:
			var prm singlepaxos.Promise
			err = decoder.Decode(&prm)
			util.Check(err)
			net.notifyPromise <- prm
		case acceptMessage:
			var acc singlepaxos.Accept
			err = decoder.Decode(&acc)
			util.Check(err)
			net.notifyAccept <- acc
		case learnMessage:
			var lrn singlepaxos.Learn
			err = decoder.Decode(&lrn)
			util.Check(err)
			net.notifyLearn <- lrn
		default:
			util.Raise("Unknown message")
		}
	}
}
