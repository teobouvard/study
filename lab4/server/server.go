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

// Server TODO
type Server struct {
	fd *detector.EvtFailureDetector
	ld *detector.MonLeaderDetector

	proposer *singlepaxos.Proposer
	acceptor *singlepaxos.Acceptor
	learner  *singlepaxos.Learner

	network *netlayer.Network

	hbOut      <-chan detector.Heartbeat
	prepareOut <-chan singlepaxos.Prepare
	acceptOut  <-chan singlepaxos.Accept
	valueOut   <-chan singlepaxos.Value
}

//NewServer constructs a server
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
		acceptOut:  acceptOut,
		valueOut:   valueOut,
	}
}

// Run TODO
func (s *Server) Run() {
	log.Printf("Starting paxos server\n")

	sig := make(chan os.Signal)
	signal.Notify(sig, os.Interrupt)

	s.fd.Start()
	s.proposer.Start()
	s.acceptor.Start()
	s.learner.Start()

	go s.eventLoop()
	<-sig
	log.Printf("Received interrupt, shutting down server\n")

	s.learner.Stop()
	s.acceptor.Stop()
	s.proposer.Stop()
	s.fd.Stop()
}

func (s *Server) eventLoop() {
	valueIn := s.network.ListenValue()

	for {
		select {
		case val := <-valueIn:
			s.proposer.DeliverClientValue(val)
		}
	}
}
