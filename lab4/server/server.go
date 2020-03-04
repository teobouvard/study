package server

import (
	"log"
	"os"
	"os/signal"

	"github.com/dat520-2020/TeamPilots/lab4/netlayer"
	"github.com/dat520-2020/TeamPilots/lab4/singlepaxos"
)

// Server TODO
type Server struct {
	proposer singlepaxos.Proposer
	acceptor singlepaxos.Acceptor
	learner  singlepaxos.Learner
}

//NewServer constructs a server
func NewServer() *Server {
	return &Server{}
}

// Connect TODO
func (s *Server) Connect(network *netlayer.Network) {

}

// Run TODO
func (s *Server) Run() {
	log.Printf("Starting paxos server\n")

	sig := make(chan os.Signal)
	signal.Notify(sig, os.Interrupt)

	go s.eventLoop()

	<-sig
	log.Printf("Received interrupt, shutting down server\n")
}

func (s *Server) eventLoop() {
	for {
		select {}
	}
}
