package server

import "github.com/dat520-2020/TeamPilots/lab4/singlepaxos"

// Server TODO
type Server struct {
	proposer singlepaxos.Proposer
	acceptor singlepaxos.Acceptor
	learner  singlepaxos.Learner
}
