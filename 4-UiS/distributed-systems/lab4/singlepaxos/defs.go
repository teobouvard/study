package singlepaxos

import "fmt"

// Type definitions - DO NOT EDIT

// Round represents a Paxos round number.
type Round int

// NoRound is a constant that represents no specific round. It should be used
// as the value for the Vrnd field in Promise messages to indicate that an
// acceptor has not voted in any previous round.
const NoRound Round = -1

// Value represents a value that can be chosen using the Paxos algorithm.
type Value string

// ZeroValue is a constant that represents the zero value for the Value type.
const ZeroValue Value = ""

// Message definitions - DO NOT EDIT

// Prepare represents a single-decree Paxos prepare message.
type Prepare struct {
	From int
	Crnd Round
}

// Promise represents a single-decree Paxos promise message.
type Promise struct {
	To, From int
	Rnd      Round
	Vrnd     Round
	Vval     Value
}

// Accept represents a single-decree Paxos accept message.
type Accept struct {
	From int
	Rnd  Round
	Val  Value
}

// Learn represents a single-decree Paxos learn message.
type Learn struct {
	From int
	Rnd  Round
	Val  Value
}

// String returns a string representation of prepare p.
func (p Prepare) String() string {
	return fmt.Sprintf("Prepare{From: %d, Crnd: %d}", p.From, p.Crnd)
}

// String returns a string representation of promise p.
func (p Promise) String() string {
	if p.Vrnd == NoRound {
		return fmt.Sprintf("Promise{To: %d, From: %d, Rnd: %d, No value reported}",
			p.To, p.From, p.Rnd)
	}
	return fmt.Sprintf("Promise{To: %d, From: %d, Rnd: %d, Vrnd: %d, Vval: %v}",
		p.To, p.From, p.Rnd, p.Vrnd, p.Vval)
}

// String returns a string representation of accept a.
func (a Accept) String() string {
	return fmt.Sprintf("Accept{From: %d, Rnd: %d, Val: %v}", a.From, a.Rnd, a.Val)
}

// String returns a string representation of learn l.
func (l Learn) String() string {
	return fmt.Sprintf("Learn{From: %d, Rnd: %d, Val: %v}", l.From, l.Rnd, l.Val)
}
