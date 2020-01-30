package detector

import "testing"

var clsaiTests = []struct {
	nodes      []int
	wantLeader int
}{
	{[]int{0}, 0},
	{[]int{0, 1}, 1},
	{[]int{0, 1, 2}, 2},
	{[]int{0, 1, 2, 3, 4}, 4},
	{[]int{0, 1, 2, 3, 42}, 42},
	{[]int{-1}, UnknownID},
	{[]int{-2, -3}, UnknownID},
	{[]int{0, -1, 2}, 2},
}

func TestCorrectLeaderSetAfterInit(t *testing.T) {
	for i, test := range clsaiTests {
		ld := NewMonLeaderDetector(test.nodes)
		gotLeader := ld.Leader()
		if gotLeader != test.wantLeader {
			t.Errorf("TestCorrectLeaderSetAfterInit %d: got leader %d, want leader %d",
				i+1, gotLeader, test.wantLeader)
		}
	}
}

type event struct {
	suspect bool // true -> suspect, false -> restore
	id      int  // node id to suspect or restore
}

var clasarqTests = []struct {
	nodes      []int
	events     []event
	wantLeader int
}{
	{
		[]int{0, 1, 2},
		[]event{
			event{true, 2},
		},
		1,
	},
	{
		[]int{0, 1, 2},
		[]event{
			{true, 2},
			{true, 1},
		},
		0,
	},
	{
		[]int{0, 1, 2},
		[]event{
			{true, 2},
			{true, 1},
			{true, 0},
		},
		UnknownID,
	},
	{
		[]int{0, 1, 2},
		[]event{
			{true, 2},
			{true, 2},
			{true, 0},
			{true, 0},
			{true, 0},
		},
		1,
	},
	{
		[]int{0, 1, 2},
		[]event{
			{true, 2},
			{false, 2},
			{true, 0},
		},
		2,
	},
	{
		[]int{0, 1, 2},
		[]event{
			{false, 2},
			{false, 1},
			{false, 0},
		},
		2,
	},
	{
		[]int{0, 1, 2},
		[]event{
			{false, 2},
			{true, 2},
			{false, 1},
			{true, 0},
		},
		1,
	},
	{
		[]int{0, 1, 2},
		[]event{
			{false, 122},
			{true, 42},
			{false, 14},
			{true, 0},
		},
		2,
	},
	{
		[]int{0, 1, 2, 3, 4},
		[]event{
			{false, 3},
			{true, 3},
			{false, 0},
			{true, 4},
		},
		2,
	},
}

func TestCorrectLeaderAfterSuspectAndRestoreSeq(t *testing.T) {
	for i, test := range clasarqTests {
		ld := NewMonLeaderDetector(test.nodes)
		for _, event := range test.events {
			if event.suspect {
				ld.Suspect(event.id)
			} else {
				ld.Restore(event.id)
			}
		}
		gotLeader := ld.Leader()
		if gotLeader != test.wantLeader {
			t.Errorf("TestCorrectLeaderSetAfterSuspectRestoreSeq %d: got leader %d, want leader %d",
				i+1, gotLeader, test.wantLeader)
		}
	}
}

type eventPubSub struct {
	suspect       bool // true -> suspect, false -> restore
	id            int  // node id to suspect or restore
	produceOutput bool // should the event produce output to subscribers
	wantLeader    int  // what leader id should publication contain
}

// Basic set of events.
var pubSubTestSeq = []eventPubSub{
	{true, 2, true, 1},         // suspect 2, want pub for 1
	{false, 2, true, 2},        // restore 2, want pub for 2
	{false, 2, false, -1},      // restore 2, no change -> no output
	{true, 1, false, -1},       // suspect 1, no change -> no output
	{true, 0, false, -1},       // suspect 0, no change -> no output
	{true, 2, true, UnknownID}, // suspect 2, all suspected, want pub for UnknownID
	{false, 0, true, 0},        // restore 0, want pub for 0
	{false, 1, true, 1},        // restore 1, want pub for 1
	{false, 2, true, 2},        // restore 2, want pub for 2
}

var pubSubTests = []struct {
	nodes    []int
	events   []eventPubSub
	nrOfSubs int
}{
	{
		[]int{0, 1, 2},
		pubSubTestSeq,
		1, // One subscriber.
	},
	{
		[]int{0, 1, 2},
		pubSubTestSeq,
		5, // Five subscribers.
	},
}

func TestPublishSubscribe(t *testing.T) {
	for i, test := range pubSubTests {
		ld := NewMonLeaderDetector(test.nodes)
		subscribers := make([]<-chan int, test.nrOfSubs)
		for n := 0; n < len(subscribers); n++ {
			subscribers[n] = ld.Subscribe()
			if subscribers[n] == nil {
				t.Fatalf("TestPubSub %d: subscriber %d got nil channel, want non-nil", i+1, n+1)
			}
		}
		for j, event := range test.events {
			if event.suspect {
				ld.Suspect(event.id)
			} else {
				ld.Restore(event.id)
			}
			for k, subscriber := range subscribers {
				if event.produceOutput {
					select {
					case gotLeader := <-subscriber:
						// Got publication, check if leader is correct.
						if gotLeader != event.wantLeader {
							t.Errorf("TestPubSub %d: Event %d: Subscriber %d: got publication for leader %d, want leader %d",
								i+1, j+1, k+1, gotLeader, event.wantLeader)
						}
					default:
						// We want publication, but got none.
						t.Errorf("TestPubSub %d: Event %d: Subscriber %d: got no publication, want one for leader %d",
							i+1, j+1, k+1, event.wantLeader)
					}

				} else {
					select {
					case gotLeader := <-subscriber:
						// Got publication, want none.
						t.Errorf("TestPubSub %d: Event %d: Subscriber %d: got publication for leader %d, but want no publication",
							i+1, j+1, k+1, gotLeader)
					default:
						// Got no publication, and want none.
					}
				}
			}
		}
	}
}
