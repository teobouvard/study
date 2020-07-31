// +build !solution

package multipaxos

import "sort"

// Acceptor represents an acceptor as defined by the Multi-Paxos algorithm.
type Acceptor struct {
	//TODO(student): Task 2 and 3 - algorithm and distributed implementation
	acceptedRnd Round  // last accepted round so far
	lastVoteVal Value  // Previous voted value
	lastVoteRnd Round  // Previous voted round
	slotID      SlotID // Slot id
	lastSlotID  SlotID // previous Slot id
	slots       map[SlotID]PromiseSlot

	id            int
	stop          chan bool      // Channel for signalling a stop request to the start run loop
	sendLrnMsg    chan<- Learn   // Channel to send Learn message
	receiveAccMsg chan Accept    // Channel to recv Accept message
	sendPrmMsg    chan<- Promise // Channel to send Promise message
	receivePrpMsg chan Prepare   // Channel to recv Prepare message
}

// NewAcceptor returns a new single-decree Paxos acceptor.
// It takes the following arguments:
//
// id: The id of the node running this instance of a Paxos acceptor.
//
// sendPrmMsg: A send only channel used to send promises to other nodes.
//
// sendLrnMsg: A send only channel used to send learns to other nodes.
func NewAcceptor(id int, sendPrmMsg chan<- Promise, sendLrnMsg chan<- Learn) *Acceptor {
	//TODO(student): Task 2 and 3 - algorithm and distributed implementation
	return &Acceptor{

		acceptedRnd:   NoRound,
		lastVoteRnd:   NoRound,
		slots:         make(map[SlotID]PromiseSlot),
		id:            id,
		stop:          make(chan bool),
		sendLrnMsg:    sendLrnMsg,
		receiveAccMsg: make(chan Accept),
		sendPrmMsg:    sendPrmMsg,
		receivePrpMsg: make(chan Prepare),
	}
}

// Start starts a's main run loop as a separate goroutine. The main run loop
// handles incoming prepare and accept messages.
func (a *Acceptor) Start() {
	go func() {
		for {
			select {
			case <-a.stop:
				return

			case acceptMsg := <-a.receiveAccMsg:
				learnMsg, out := a.handleAccept(acceptMsg)
				if out {
					a.sendLrnMsg <- learnMsg
				}
			case prepareMessage := <-a.receivePrpMsg:
				promiseMsg, out := a.handlePrepare(prepareMessage)
				if out {
					a.sendPrmMsg <- promiseMsg
				}

			}

		}
	}()
}

// Stop stops a's main run loop.
func (a *Acceptor) Stop() {
	//TODO(student): Task 3 - distributed implementation
	a.stop <- true
}

// DeliverPrepare delivers prepare prepareMessage to acceptor a.
func (a *Acceptor) DeliverPrepare(prepareMessage Prepare) {
	//TODO(student): Task 3 - distributed implementation
	a.receivePrpMsg <- prepareMessage
}

// DeliverAccept delivers accept acc to acceptor a.
func (a *Acceptor) DeliverAccept(acc Accept) {
	//TODO(student): Task 3 - distributed implementation
	a.receiveAccMsg <- acc
}

// Internal: handlePrepare processes prepare prp according to the Multi-Paxos
// algorithm. If handling the prepare results in acceptor a emitting a
// corresponding promise, then output will be true and prm contain the promise.
// If handlePrepare returns false as output, then prm will be a zero-valued
// struct.
func (a *Acceptor) handlePrepare(prepareMessage Prepare) (prm Promise, output bool) {

	if prepareMessage.Crnd > a.acceptedRnd {
		a.acceptedRnd = prepareMessage.Crnd
		a.slotID = prepareMessage.Slot

		acceptedSlots := []PromiseSlot{}
		for slotID, promisedSlot := range a.slots {
			if slotID >= prepareMessage.Slot {
				//find all previous promises
				acceptedSlots = append(acceptedSlots, promisedSlot)
			}
		}

		sort.Sort(slotData(acceptedSlots))

		//Sort the based on SlotID to maintain sequence in state
		promise := Promise{To: prepareMessage.From, From: a.id, Rnd: a.acceptedRnd}

		if len(acceptedSlots) > 0 {
			promise.Slots = acceptedSlots
		}
		return promise, true
	}
	return Promise{}, false
}

// Internal: handleAccept processes accept acc according to the Multi-Paxos
// algorithm. If handling the accept results in acceptor a emitting a
// corresponding learn, then output will be true and lrn contain the learn.  If
// handleAccept returns false as output, then lrn will be a zero-valued struct.
func (a *Acceptor) handleAccept(acc Accept) (lrn Learn, output bool) {
	//TODO(student): Task 2 - algorithm implementation
	//return Learn{From: -1, Rnd: -2, Val: "FooBar"}, true
	if acc.Rnd >= a.acceptedRnd { // New round, or same round (incase of duplicate accept message)
		a.acceptedRnd = acc.Rnd
		a.lastVoteVal = acc.Val
		a.lastVoteRnd = acc.Rnd
		a.slotID = acc.Slot
		a.lastSlotID = acc.Slot

		acceptedSlot := PromiseSlot{ID: acc.Slot, Vrnd: a.lastVoteRnd, Vval: a.lastVoteVal}
		prevSlot, ok := a.slots[acc.Slot]

		if (ok && prevSlot.Vrnd < acc.Rnd) || !ok {
			// Add new accept,
			//or update old accept with higher round for a SlotID
			a.slots[acc.Slot] = acceptedSlot
		}

		learn := Learn{
			a.id,
			a.slotID,
			a.lastVoteRnd,
			a.lastVoteVal,
		}
		return learn, true
	}
	return Learn{}, false
}

// Sort the Promise Slot by ID
type slotData []PromiseSlot

// This function is used by sort interface to find the length of slice
func (slice slotData) Len() int {
	return len(slice)
}

// This function is used by sort interface to find minimum ID
func (slice slotData) Less(i, j int) bool {
	return slice[i].ID < slice[j].ID
}

// This function is used by sort interface to swap entries
func (slice slotData) Swap(i, j int) {
	slice[i], slice[j] = slice[j], slice[i]
}
