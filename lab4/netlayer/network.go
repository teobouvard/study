package netlayer

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"net"

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
	nodeID int
	config *Config

	conn *net.UDPConn

	notifyHb      chan detector.Heartbeat
	notifyValue   chan singlepaxos.Value
	notifyPrepare chan singlepaxos.Prepare
	notifyPromise chan singlepaxos.Promise
	notifyAccept  chan singlepaxos.Accept
	notifyLearn   chan singlepaxos.Learn
}

// NewNetwork creates a network from the config file passed
func NewNetwork(configFile string, id int) *Network {
	config := NewConfig(configFile)

	if !config.Contains(id) {
		util.Raise(fmt.Sprintf("Node [%d] is not present in config file. Make sure to correctly set the --id flag.\n", id))
	}

	conn, err := net.ListenUDP("udp", config.AddrOf(id))
	util.Check(err)

	return &Network{
		nodeID: id,
		config: config,

		conn: conn,

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
	for _, id := range net.config.ServerIDs() {
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
	_, err = net.conn.WriteTo(encoded, net.config.AddrOf(id))
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
	// pretty sure this is the dumbest way of doing this, is there another ?
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

// ServerIDs is an accessor to ids of all server nodes
func (net *Network) ServerIDs() []int {
	return net.config.ServerIDs()
}

// NodeID is an accessor to the id of the current node
func (net *Network) NodeID() int {
	return net.nodeID
}
