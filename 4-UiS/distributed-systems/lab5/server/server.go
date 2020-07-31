package server

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"sort"
	"time"

	"dat520/lab3/detector"
	"dat520/lab5/bank"
	"dat520/lab5/multipaxos"
	"dat520/lab5/netlayer"
)

const noSlotID multipaxos.SlotID = -1

var valueBuffer map[multipaxos.SlotID]multipaxos.Response

// Server is a Paxos server
type Server struct {
	nodeID int

	fd *detector.EvtFailureDetector
	ld *detector.MonLeaderDetector

	proposer *multipaxos.Proposer
	acceptor *multipaxos.Acceptor
	learner  *multipaxos.Learner

	network *netlayer.Network

	hbOut      <-chan detector.Heartbeat
	prepareOut <-chan multipaxos.Prepare
	promiseOut <-chan multipaxos.Promise
	acceptOut  <-chan multipaxos.Accept
	learnOut   <-chan multipaxos.Learn
	valueOut   <-chan multipaxos.DecidedValue

	lastValue     multipaxos.Value
	lastDecValue  multipaxos.DecidedValue
	bankAccounts  map[int]bank.Account // map to store all bank accounts
	highestSlotID multipaxos.SlotID    // Keep track of the id of highest decided slot.
	leader        bool
}

// NewServer constructs a server assigned to the network
func NewServer(network *netlayer.Network) *Server {
	const bufferSize int = 100

	serverIDs := network.ServerIDs()
	nodeID := network.NodeID()

	hbOut := make(chan detector.Heartbeat, bufferSize)
	ld := detector.NewMonLeaderDetector(serverIDs)
	fd := detector.NewEvtFailureDetector(nodeID, serverIDs, ld, time.Second, hbOut)

	prepareOut := make(chan multipaxos.Prepare, bufferSize)
	acceptOut := make(chan multipaxos.Accept, bufferSize)
	proposer := multipaxos.NewProposer(nodeID, len(serverIDs), -1, ld, prepareOut, acceptOut)

	promiseOut := make(chan multipaxos.Promise, bufferSize)
	learnOut := make(chan multipaxos.Learn, bufferSize)
	acceptor := multipaxos.NewAcceptor(nodeID, promiseOut, learnOut)

	valueOut := make(chan multipaxos.DecidedValue, bufferSize)
	learner := multipaxos.NewLearner(nodeID, len(serverIDs), valueOut)

	leader := false
	if ld.Leader() == nodeID {
		leader = true
	}

	// initial bank accounts
	bankAccounts := make(map[int]bank.Account)

	for accountNr := 0; accountNr <= 1000; accountNr++ {
		bankAccounts[accountNr] = bank.Account{Number: accountNr, Balance: rand.Intn(10001)}
	}

	return &Server{
		nodeID:  nodeID,
		network: network,

		proposer: proposer,
		acceptor: acceptor,
		learner:  learner,

		ld: ld,
		fd: fd,

		hbOut:      hbOut,
		prepareOut: prepareOut,
		promiseOut: promiseOut,
		acceptOut:  acceptOut,
		learnOut:   learnOut,
		valueOut:   valueOut,

		bankAccounts:  bankAccounts,
		highestSlotID: noSlotID,
		leader:        leader,
	}
}

// Run runs the server until interruption. It handles the start and stop of the paxos components of s.
func (s *Server) Run() {
	log.Printf("Starting paxos server [%v]\n", s.network.NodeID())

	sig := make(chan os.Signal)
	signal.Notify(sig, os.Interrupt)

	s.fd.Start()
	s.proposer.Start()
	s.acceptor.Start()
	s.learner.Start()

	go s.eventLoop()
	<-sig
	log.Printf("Received interrupt, shutting down server [%v]\n", s.network.NodeID())

	s.learner.Stop()
	s.acceptor.Stop()
	s.proposer.Stop()
	s.fd.Stop()
}

// Internal : s subscribes to the notification channels of its network, and forwards the objects to its paxos components.
// The event loop also forwards the objects returned by its paxos components to the server for broadcasting
func (s *Server) eventLoop() {
	hbIn := s.network.ListenHeartbeat()
	valueIn := s.network.ListenValueServer()
	prepareIn := s.network.ListenPrepare()
	promiseIn := s.network.ListenPromise()
	acceptIn := s.network.ListenAccept()
	learnIn := s.network.ListenLearn()
	newLeader := s.ld.Subscribe()

	valueBuffer = make(map[multipaxos.SlotID]multipaxos.Response)

	for {
		select {
		case hb := <-hbIn:
			s.fd.DeliverHeartbeat(hb)
		case hb := <-s.hbOut:
			s.network.SendHeartbeat(hb)
		case val := <-valueIn:
			s.lastValue = val
			s.proposer.DeliverClientValue(val)
		case val := <-s.valueOut:
			s.lastDecValue = val
			//if s.leader {
			if s.ld.Leader() == s.nodeID {
				s.handleOutgoing(val)
			}
		case prp := <-s.prepareOut:
			s.network.BroadcastPrepare(prp)
		case prp := <-prepareIn:
			s.acceptor.DeliverPrepare(prp)
		case prm := <-promiseIn:
			s.proposer.DeliverPromise(prm)
		case prm := <-s.promiseOut:
			s.network.SendPromise(prm)
		case acc := <-acceptIn:
			s.acceptor.DeliverAccept(acc)
		case acc := <-s.acceptOut:
			s.network.BroadcastAccept(acc)
		case lrn := <-learnIn:
			s.learner.DeliverLearn(lrn)
		case lrn := <-s.learnOut:
			s.network.BroadcastLearn(lrn)
		case curLeader := <-newLeader:
			//fmt.Fprintf(os.Stderr, "[\033[36;1m LEADERCHANGE \033[0m] New leader : %v \n", curLeader)
			if curLeader == s.nodeID {
				s.leader = true
				// resend last decided value
				s.handleOutgoing(s.lastDecValue)
				// resend latest request to proposer
				s.proposer.DeliverClientValue(s.lastValue)
			} else {
				s.leader = false
			}
		}
	}
}

func (s *Server) handleOutgoing(val multipaxos.DecidedValue) {

	// Check noop
	if val.Value.Noop {
		// Notify the Proposer when a new slot has been decided.
		s.proposer.IncrementAllDecidedUpTo()
	} else {

		// Process decidedValue before it is sent to the Client
		account := s.bankAccounts[val.Value.AccountNum]
		transactionResult := account.Process(val.Value.Txn)

		respToClient := multipaxos.Response{
			ClientID:  val.Value.ClientID,
			ClientSeq: val.Value.ClientSeq,
			TxnRes:    transactionResult,
		}

		if val.SlotID == s.highestSlotID+1 { // correct order

			// Send TransactionResult to the Client
			s.network.BroadcastVotedValue(respToClient)
			// Notify the Proposer when a new slot has been decided.
			s.proposer.IncrementAllDecidedUpTo()
			// keep track of the id of highest decided slot that was sent.
			s.highestSlotID = val.SlotID

			if len(valueBuffer) != 0 { // send buffered values
				keys := make([]multipaxos.SlotID, 0, len(valueBuffer))
				for k := range valueBuffer {
					keys = append(keys, k)
				}
				sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
				for _, k := range keys {
					if k == s.highestSlotID+1 {
						s.network.BroadcastVotedValue(valueBuffer[k])
						// Notify the Proposer when a new slot has been decided.
						s.proposer.IncrementAllDecidedUpTo()
						// keep track of the id of highest decided slot that was sent.
						s.highestSlotID = val.SlotID
						// remove entry of map
						delete(valueBuffer, k)
					}
				}
			}

		} else if val.SlotID > s.highestSlotID+1 { // wrong order, delivered too soon
			valueBuffer[val.SlotID] = respToClient
		} else {
			//util.Raise("Server handleOutgoing: unknown SlotID")
			fmt.Fprintf(os.Stderr, "[\033[40;1m ERROR \033[0m] SlotID too high : %v \n", val.SlotID)
		}

	}

}
