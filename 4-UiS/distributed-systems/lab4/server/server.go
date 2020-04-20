package server

import (
	"log"
	"os"
	"os/signal"
	"time"

	"github.com/dat520-2020/TeamPilots/lab3/detector"
	"github.com/dat520-2020/TeamPilots/lab4/netlayer"
	"github.com/dat520-2020/TeamPilots/lab4/singlepaxos"
)

// Server is a Paxos server
type Server struct {
	fd *detector.EvtFailureDetector
	ld *detector.MonLeaderDetector

	proposer *singlepaxos.Proposer
	acceptor *singlepaxos.Acceptor
	learner  *singlepaxos.Learner

	network *netlayer.Network

	hbOut      <-chan detector.Heartbeat
	prepareOut <-chan singlepaxos.Prepare
	promiseOut <-chan singlepaxos.Promise
	acceptOut  <-chan singlepaxos.Accept
	learnOut   <-chan singlepaxos.Learn
	valueOut   <-chan singlepaxos.Value
}

// NewServer constructs a server assigned to the network
func NewServer(network *netlayer.Network) *Server {
	const bufferSize int = 100

	serverIDs := network.ServerIDs()
	nodeID := network.NodeID()

	hbOut := make(chan detector.Heartbeat, bufferSize)
	ld := detector.NewMonLeaderDetector(serverIDs)
	fd := detector.NewEvtFailureDetector(nodeID, serverIDs, ld, time.Second, hbOut)

	prepareOut := make(chan singlepaxos.Prepare, bufferSize)
	acceptOut := make(chan singlepaxos.Accept, bufferSize)
	proposer := singlepaxos.NewProposer(nodeID, len(serverIDs), ld, prepareOut, acceptOut)

	promiseOut := make(chan singlepaxos.Promise, bufferSize)
	learnOut := make(chan singlepaxos.Learn, bufferSize)
	acceptor := singlepaxos.NewAcceptor(nodeID, promiseOut, learnOut)

	valueOut := make(chan singlepaxos.Value, bufferSize)
	learner := singlepaxos.NewLearner(nodeID, len(serverIDs), valueOut)

	return &Server{
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
	valueIn := s.network.ListenValue()
	prepareIn := s.network.ListenPrepare()
	promiseIn := s.network.ListenPromise()
	acceptIn := s.network.ListenAccept()
	learnIn := s.network.ListenLearn()

	for {
		select {
		case hb := <-hbIn:
			s.fd.DeliverHeartbeat(hb)
		case hb := <-s.hbOut:
			s.network.SendHeartbeat(hb)
		case val := <-valueIn:
			s.proposer.DeliverClientValue(val)
		case val := <-s.valueOut:
			s.network.BroadcastVotedValue(val)
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
		}
	}
}
