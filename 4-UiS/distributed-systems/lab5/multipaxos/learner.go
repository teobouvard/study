// +build !solution

package multipaxos

// Learner represents a learner as defined by the Multi-Paxos algorithm.
type Learner struct {
	id     int // This node ID
	n      int // Number of nodes
	quorum int // Quorum value

	recvLearn chan Learn          // Channel to recv Learn message
	sendValue chan<- DecidedValue // Channel to send decided value
	stop      chan struct{}       // Channel for signalling a stop request to the start run loop
	learns    map[string]Learn    // receiving learn holder
}

// NewLearner returns a new Multi-Paxos learner. It takes the following
// arguments:
//
// id: The id of the node running this instance of a Paxos learner.
//
// nrOfNodes: The total number of Paxos nodes.
//
// decidedOut: A send only channel used to send values that has been learned,
// i.e. decided by the Paxos nodes.
func NewLearner(id int, nrOfNodes int, decidedOut chan<- DecidedValue) *Learner {
	return &Learner{
		id:     id,
		n:      nrOfNodes,
		quorum: (nrOfNodes / 2) + 1,

		recvLearn: make(chan Learn),
		sendValue: decidedOut,
		stop:      make(chan struct{}),
		learns:    make(map[string]Learn),
	}
}

// Start starts l's main run loop as a separate goroutine. The main run loop
// handles incoming learn messages.
func (l *Learner) Start() {
	go func() {
		for {
			select {
			case learn := <-l.recvLearn:
				val, SlotID, output := l.handleLearn(learn)
				if output {
					decidedVal := DecidedValue{SlotID: SlotID, Value: val}
					l.sendValue <- decidedVal
				}
			case <-l.stop:
				return
			}
		}
	}()
}

// Stop stops l's main run loop.
func (l *Learner) Stop() {
	// TODO(student)
	l.stop <- struct{}{}
}

// DeliverLearn delivers learn lrn to learner l.
func (l *Learner) DeliverLearn(lrn Learn) {
	// TODO(student)
	l.recvLearn <- lrn
}

// Internal: handleLearn processes learn lrn according to the Multi-Paxos
// algorithm. If handling the learn results in learner l emitting a
// corresponding decided value, then output will be true, sid the id for the
// slot that was decided and val contain the decided value. If handleLearn
// returns false as output, then val and sid will have their zero value.
func (l *Learner) handleLearn(learn Learn) (val Value, sid SlotID, output bool) {
	// TODO(student)
	countLearns := make(map[int]int)
	slots := make(map[int][]SlotID)
	countSlots := make(map[int]int)
	slotValues := make(map[string]Value)

	key := string(learn.From) + "_" + string(learn.Slot)

	prev_learn, ok := l.learns[key]
	if (ok && learn.Rnd > prev_learn.Rnd) || !ok {
		l.learns[key] = learn
	}

	//return the chosen value
	for _, learn := range l.learns {
		countLearns[int(learn.Rnd)]++
		countSlots[int(learn.Slot)]++
		key := string(learn.Rnd) + "_" + string(learn.Slot)
		slotValues[key] = learn.Val

		answer, ok := slots[int(learn.Rnd)]
		if ok {
			slots[int(learn.Rnd)] = append(answer, learn.Slot)
		} else {
			slots[int(learn.Rnd)] = []SlotID{learn.Slot}
		}
	}

	for round, count := range countLearns {
		if count >= l.quorum {
			for slt, count := range countSlots {
				if count >= l.quorum {
					ok := l.checkRound(slots[round], slt)
					if ok {
						//remove already learnt values
						l.removeLearn(Round(round), SlotID(slt))

						key := string(round) + "_" + string(slt)
						return slotValues[key], SlotID(slt), true
					}
				}
			}
		}
	}
	return Value{}, 0, false
}

func (l *Learner) removeLearn(round Round, slt SlotID) {
	for k, learn := range l.learns {
		if learn.Rnd == round && learn.Slot == slt {
			delete(l.learns, k)
		}
	}
}

func (l *Learner) checkRound(slots []SlotID, slt int) bool {
	count := 0
	for _, sl := range slots {
		if int(sl) == slt {
			count++
		}
		if count >= l.quorum {
			return true
		}
	}
	return false
}
