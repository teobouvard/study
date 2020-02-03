// +build !solution

package detector

import (
	"log"
	"time"
)

// EvtFailureDetector represents a Eventually Perfect Failure Detector as
// described at page 53 in:
// Christian Cachin, Rachid Guerraoui, and Luís Rodrigues: "Introduction to
// Reliable and Secure Distributed Programming" Springer, 2nd edition, 2011.
type EvtFailureDetector struct {
	id        int          // this node's id
	nodeIDs   []int        // node ids for every node in cluster
	alive     map[int]bool // map of node ids considered alive
	suspected map[int]bool // map of node ids  considered suspected

	sr SuspectRestorer // Provided SuspectRestorer implementation

	delay         time.Duration // the current delay for the timeout procedure
	delta         time.Duration // the delta value to be used when increasing delay
	timeoutSignal *time.Ticker  // the timeout procedure ticker

	hbSend chan<- Heartbeat // channel for sending outgoing heartbeat messages
	hbIn   chan Heartbeat   // channel for receiving incoming heartbeat messages
	stop   chan struct{}    // channel for signaling a stop request to the main run loop

	testingHook func() // DO NOT REMOVE THIS LINE. A no-op when not testing.
}

// NewEvtFailureDetector returns a new Eventual Failure Detector. It takes the
// following arguments:
//
// id: The id of the node running this instance of the failure detector.
//
// nodeIDs: A list of ids for every node in the cluster (including the node
// running this instance of the failure detector).
//
// ld: A leader detector implementing the SuspectRestorer interface.
//
// delta: The initial value for the timeout interval. Also the value to be used
// when increasing delay.
//
// hbSend: A send only channel used to send heartbeats to other nodes.
func NewEvtFailureDetector(id int, nodeIDs []int, sr SuspectRestorer, delta time.Duration, hbSend chan<- Heartbeat) *EvtFailureDetector {
	suspected := make(map[int]bool)
	alive := make(map[int]bool)

	for i := range nodeIDs {
		alive[i] = true
	}

	return &EvtFailureDetector{
		id:        id,
		nodeIDs:   nodeIDs,
		alive:     alive,
		suspected: suspected,

		sr: sr,

		delay: delta,
		delta: delta,

		hbSend: hbSend,
		hbIn:   make(chan Heartbeat, 8),
		stop:   make(chan struct{}),

		testingHook: func() {}, // DO NOT REMOVE THIS LINE. A no-op when not testing.
	}
}

// Start starts e's main run loop as a separate goroutine. The main run loop
// handles incoming heartbeat requests and responses. The loop also trigger e's
// timeout procedure at an interval corresponding to e's internal delay
// duration variable.
func (e *EvtFailureDetector) Start() {
	e.setTimer()
	go func() {
		for {
			e.testingHook() // DO NOT REMOVE THIS LINE. A no-op when not testing.
			select {
			case hb := <-e.hbIn:
				// TODO(student): Handle incoming heartbeat
				e.handleHeartbeat(hb)
				log.Printf("Received Heartbeat\n")
			case <-e.timeoutSignal.C:
				log.Printf("Failure detector timeout\n")
				e.timeout()
			case <-e.stop:
				return
			}
		}
	}()
}

// DeliverHeartbeat delivers heartbeat hb to failure detector e.
func (e *EvtFailureDetector) DeliverHeartbeat(hb Heartbeat) {
	e.hbIn <- hb
}

// Stop stops e's main run loop.
func (e *EvtFailureDetector) Stop() {
	e.stop <- struct{}{}
}

// Internal: timeout runs e's timeout procedure.
func (e *EvtFailureDetector) timeout() {
	// TODO(student): Implement timeout procedure
	for i := range e.nodeIDs {
		if e.alive[i] && (e.alive[i] == e.suspected[i]) {
			e.delay += e.delta
			log.Printf("Increasing delay\n")
			break
		}
	}

	for _, i := range e.nodeIDs {
		if !e.alive[i] && !e.suspected[i] {
			e.suspected[i] = true
			e.sr.Suspect(i)
		} else if e.alive[i] && e.suspected[i] {
			delete(e.suspected, i)
			e.sr.Restore(i)
		}
		e.sendHeartbeatRequest(i)
	}

	e.clearAlive()
}

// TODO(student): Add other unexported functions or methods if needed.

// sendHeartbeatRequest sends a heartbeat request to node toNode.
func (e *EvtFailureDetector) sendHeartbeatRequest(toNode int) {
	hb := Heartbeat{
		From:    e.id,
		To:      toNode,
		Request: true,
	}
	e.hbSend <- hb
}

// clearAlive empties the alive set of node (l17)
func (e *EvtFailureDetector) clearAlive() {
	e.alive = make(map[int]bool)
}

func (e *EvtFailureDetector) setTimer() {
	e.timeoutSignal = time.NewTicker(e.delay)
}

func (e *EvtFailureDetector) handleHeartbeat(hb Heartbeat) {
	if hb.Request {
		hb := Heartbeat{
			From:    e.id,
			To:      hb.From,
			Request: false,
		}
		e.hbSend <- hb
	} else {
		e.alive[hb.From] = true
	}
}
