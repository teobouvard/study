package netlayer

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"net"
	"strconv"

	"dat520/lab3/detector"
	"dat520/lab5/multipaxos"
	"dat520/lab5/util"
	//"dat520/lab5/bank"
)

// Byte constants for different message types
// The corresponding value is prepended to each message being sent
const (
	heartbeatMessage byte = iota
	valueMessageServer
	valueMessageClient
	prepareMessage
	promiseMessage
	acceptMessage
	learnMessage
)

// Network implements the network layer, by providing read-only
// notification channels and sending methods
type Network struct {
	nodeID int
	config *Config

	conn *net.UDPConn

	notifyHb          chan detector.Heartbeat
	notifyValueClient chan multipaxos.Response // notify the Client
	notifyValueServer chan multipaxos.Value    // notify the Server
	notifyPrepare     chan multipaxos.Prepare
	notifyPromise     chan multipaxos.Promise
	notifyAccept      chan multipaxos.Accept
	notifyLearn       chan multipaxos.Learn
}

// NewNetwork creates a network from the config file passed
func NewNetwork(configFile string, id int) *Network {
	// Updates the Config where all participants in the network are stored
	config := NewConfig(configFile)

	if !config.Contains(id) {
		util.Raise(fmt.Sprintf("Node [%d] is not present in config file. Make sure to correctly set the --id flag.\n", id))
	}

	// start listening on the address in the config file
	conn, err := net.ListenUDP("udp", config.AddrOf(id))
	util.Check(err)

	return &Network{
		nodeID: id,
		config: config,

		conn: conn,

		notifyHb:          make(chan detector.Heartbeat),
		notifyValueClient: make(chan multipaxos.Response), // notify the Client
		notifyValueServer: make(chan multipaxos.Value),    // notify the Server
		notifyPrepare:     make(chan multipaxos.Prepare),
		notifyPromise:     make(chan multipaxos.Promise),
		notifyAccept:      make(chan multipaxos.Accept),
		notifyLearn:       make(chan multipaxos.Learn),
	}
}

// BroadcastRequestedValue sends a value to all *servers* on the network
func (net *Network) BroadcastRequestedValue(value multipaxos.Value) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err := encoder.Encode(value)
	util.Check(err)
	encoded := append([]byte{valueMessageServer}, buf.Bytes()...)
	for _, addr := range net.config.servers {
		_, err = net.conn.WriteTo(encoded, addr)
		util.Check(err)
	}
}

// BroadcastVotedValue sends a value to one client on the network
func (net *Network) BroadcastVotedValue(response multipaxos.Response) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err := encoder.Encode(response)
	util.Check(err)
	encoded := append([]byte{valueMessageClient}, buf.Bytes()...)

	clientID, _ := strconv.Atoi(response.ClientID)
	addr := net.config.clients[clientID]
	_, _ = net.conn.WriteTo(encoded, addr)
	//util.Check(err)
}

// BroadcastPrepare sends a prepare message to all servers on the network
func (net *Network) BroadcastPrepare(prp multipaxos.Prepare) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err := encoder.Encode(prp)
	util.Check(err)
	encoded := buf.Bytes()
	encoded = append([]byte{prepareMessage}, encoded...)
	for _, addr := range net.config.servers {
		_, err = net.conn.WriteTo(encoded, addr)
		util.Check(err)
	}
}

// BroadcastAccept sends an accept message to all servers on the network
func (net *Network) BroadcastAccept(acc multipaxos.Accept) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err := encoder.Encode(acc)
	util.Check(err)
	encoded := append([]byte{acceptMessage}, buf.Bytes()...)
	for _, addr := range net.config.servers {
		_, err = net.conn.WriteTo(encoded, addr)
		util.Check(err)
	}
}

// BroadcastLearn sends a learn message to all servers on the network
func (net *Network) BroadcastLearn(lrn multipaxos.Learn) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err := encoder.Encode(lrn)
	util.Check(err)
	encoded := append([]byte{learnMessage}, buf.Bytes()...)
	for _, addr := range net.config.servers {
		_, err = net.conn.WriteTo(encoded, addr)
		util.Check(err)
	}
}

// SendHeartbeat sends a hearbeat message to its recipient
func (net *Network) SendHeartbeat(hb detector.Heartbeat) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err := encoder.Encode(hb)
	util.Check(err)
	encoded := append([]byte{heartbeatMessage}, buf.Bytes()...)
	_, err = net.conn.WriteTo(encoded, net.config.AddrOf(hb.To))
	util.Check(err)

}

// SendPromise sends a promise message to its recipient
func (net *Network) SendPromise(prm multipaxos.Promise) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err := encoder.Encode(prm)
	util.Check(err)
	encoded := append([]byte{promiseMessage}, buf.Bytes()...)
	_, err = net.conn.WriteTo(encoded, net.config.AddrOf(prm.To))
	util.Check(err)
}

// Start starts the server event loop
func (net *Network) Start() {
	go net.eventLoop()
}

// ListenHeartbeat returns a notification channel for Heartbeat messages
func (net *Network) ListenHeartbeat() <-chan detector.Heartbeat {
	return net.notifyHb
}

// ListenValue returns a notification channel for Value messages from Clients
func (net *Network) ListenValueClient() <-chan multipaxos.Response {
	return net.notifyValueClient
}

// ListenValue returns a notification channel for Value messages from Clients
func (net *Network) ListenValueServer() <-chan multipaxos.Value {
	return net.notifyValueServer
}

// ListenPrepare returns a notification channel for Prepare messages
func (net *Network) ListenPrepare() <-chan multipaxos.Prepare {
	return net.notifyPrepare
}

// ListenPromise returns a notification channel for Promise messages
func (net *Network) ListenPromise() <-chan multipaxos.Promise {
	return net.notifyPromise
}

// ListenAccept returns a notification channel for Accept messages
func (net *Network) ListenAccept() <-chan multipaxos.Accept {
	return net.notifyAccept
}

// ListenLearn returns a notification channel for Learn messages
func (net *Network) ListenLearn() <-chan multipaxos.Learn {
	return net.notifyLearn
}

// Internal : net's event loop reads all incoming UDP packets and forwards the decoded objects to the notification channels.
// The first byte of each packet discriminates between the different messages types.
func (net *Network) eventLoop() {
	var buf [512]byte
	// pretty sure this is the dumbest way of doing this, is there another ?
	for {
		n, _, err := net.conn.ReadFromUDP(buf[:])
		util.Check(err)
		decoder := gob.NewDecoder(bytes.NewReader(buf[1:n]))

		// the first element of the buffer identifies the message type
		switch buf[0] {
		case heartbeatMessage:
			var hb detector.Heartbeat
			err = decoder.Decode(&hb)
			util.Check(err)
			//log.Printf("NETWORK : %v\n", hb)
			net.notifyHb <- hb
		case valueMessageServer:
			var val multipaxos.Value
			err = decoder.Decode(&val)
			util.Check(err)
			//log.Printf("NETWORK : Value '%s'\n", val)
			net.notifyValueServer <- val
		case valueMessageClient:
			var val multipaxos.Response
			err = decoder.Decode(&val)
			util.Check(err)
			//log.Printf("NETWORK : Value '%s'\n", val)
			net.notifyValueClient <- val
		case prepareMessage:
			var prp multipaxos.Prepare
			err = decoder.Decode(&prp)
			util.Check(err)
			//log.Printf("NETWORK : '%s'\n", prp)
			net.notifyPrepare <- prp
		case promiseMessage:
			var prm multipaxos.Promise
			_ = decoder.Decode(&prm)
			//util.Check(err)
			//log.Printf("NETWORK : '%s'\n", prm)
			net.notifyPromise <- prm
		case acceptMessage:
			var acc multipaxos.Accept
			err = decoder.Decode(&acc)
			util.Check(err)
			//log.Printf("NETWORK : '%s'\n", acc)
			net.notifyAccept <- acc
		case learnMessage:
			var lrn multipaxos.Learn
			err = decoder.Decode(&lrn)
			util.Check(err)
			//log.Printf("NETWORK : '%s'\n", lrn)
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

// IsServer checks if id is a server
func (net *Network) IsServer(id int) bool {
	return net.config.IsServer(id)
}
