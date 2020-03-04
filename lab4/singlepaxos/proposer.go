// +build !solution

package singlepaxos

import (
	"time"

	"github.com/dat520-2020/TeamPilots/lab3/detector"
)

// Proposer represents a proposer as defined by the single-decree Paxos
// algorithm.
type Proposer struct {
	id        int
	nrOfNodes int

	quorumReached bool
	quorumTimeout *time.Timer

	crnd        Round
	clientValue Value

	mv map[int]Promise

	ld detector.LeaderDetector

	valueIn   chan Value
	promiseIn chan Promise

	prepareOut chan<- Prepare
	acceptOut  chan<- Accept

	stop chan struct{}

	//TODO(student): Task 2 and 3 - algorithm and distributed implementation
	// Add other needed fields
}

// NewProposer returns a new single-decree Paxos proposer.
// It takes the following arguments:
//
// id: The id of the node running this instance of a Paxos proposer.
//
// nrOfNodes: The total number of Paxos nodes.
//
// ld: A leader detector implementing the detector.LeaderDetector interface.
//
// prepareOut: A send only channel used to send prepares to other nodes.
//
// The proposer's internal crnd field should initially be set to the value of
// its id.
func NewProposer(id int, nrOfNodes int, ld detector.LeaderDetector, prepareOut chan<- Prepare, acceptOut chan<- Accept) *Proposer {
	//TODO(student): Task 2 and 3 - algorithm and distributed implementation
	return &Proposer{
		id:            id,
		nrOfNodes:     nrOfNodes,
		quorumReached: false,
		//quorumTimeout: nil,

		crnd:        Round(id),
		clientValue: ZeroValue,
		mv:          make(map[int]Promise),

		ld: ld,

		valueIn:   make(chan Value, 10),
		promiseIn: make(chan Promise, 10),

		prepareOut: prepareOut,
		acceptOut:  acceptOut,

		stop: make(chan struct{}),
	}
}

// Start starts p's main run loop as a separate goroutine. The main run loop
// handles incoming promise messages and leader detector trust messages.
func (p *Proposer) Start() {
	go func() {
		for {
			//TODO(student): Task 3 - distributed implementation
			select {
			case prm := <-p.promiseIn:
				acc, output := p.handlePromise(prm)
				if output {
					p.acceptOut <- acc
					p.quorumReached = true
				}
			case val := <-p.valueIn:
				p.prepareOut <- p.craftPrepare(val)
				p.quorumTimeout = time.NewTimer(time.Second) // second is arbitrary
			case <-p.quorumTimeout.C:
				if !p.quorumReached {
					p.DeliverClientValue(p.clientValue)
				}
			case <-p.stop:
				return
			}
		}
	}()
}

// Stop stops p's main run loop.
func (p *Proposer) Stop() {
	//TODO(student): Task 3 - distributed implementation
	p.stop <- struct{}{}
}

// DeliverPromise delivers promise prm to proposer p.
func (p *Proposer) DeliverPromise(prm Promise) {
	//TODO(student): Task 3 - distributed implementation
	p.promiseIn <- prm
}

// DeliverClientValue delivers client value val from to proposer p.
func (p *Proposer) DeliverClientValue(val Value) {
	//TODO(student): Task 3 - distributed implementation
	if p.ld.Leader() == p.id {
		p.valueIn <- val
	}
}

func (p *Proposer) craftPrepare(val Value) Prepare {
	p.clientValue = val
	p.increaseCrnd()
	p.quorumReached = false
	return Prepare{From: p.id, Crnd: p.crnd}
}

// Internal: handlePromise processes promise prm according to the single-decree
// Paxos algorithm. If handling the promise results in proposer p emitting a
// corresponding accept, then output will be true and acc contain the promise.
// If handlePromise returns false as output, then acc will be a zero-valued
// struct.
func (p *Proposer) handlePromise(prm Promise) (acc Accept, output bool) {
	//TODO(student): Task 2 - algorithm implementation
	if prm.Rnd == p.crnd {
		p.mv[prm.From] = prm
		if p.consensus() {
			p.pickValue()
			return Accept{From: p.id, Rnd: p.crnd, Val: p.clientValue}, true
		}
	}
	return Accept{}, false
}

// Internal: increaseCrnd increases proposer p's crnd field by the total number
// of Paxos nodes.
func (p *Proposer) increaseCrnd() {
	p.crnd += Round(p.nrOfNodes)
	//TODO(student): Task 2 - algorithm implementation
}

//TODO(student): Add any other unexported methods needed.

func (p *Proposer) consensus() bool {
	return len(p.mv) > p.nrOfNodes/2
}

func (p *Proposer) pickValue() {
	var picked Promise

	for _, promise := range p.mv {
		if promise.Vrnd > picked.Vrnd {
			picked = promise
		}
	}

	if picked.Vval != ZeroValue {
		p.clientValue = picked.Vval
	}
}
