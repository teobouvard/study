package multipaxos

import (
	"reflect"
	"testing"
)

func TestHandlePrepareAndAccept(t *testing.T) {
	for i, test := range acceptorTests {
		for j, action := range test.actions {
			switch action.msgtype {
			case prepare:
				gotPrm, gotOutput := test.acceptor.handlePrepare(action.prepare)
				switch {
				case !action.wantOutput && gotOutput:
					t.Errorf("\nHandlePrepare\ntest nr:%d\naction nr: %d\naction description: %s\nwant no output\ngot %v",
						i+1, j+1, action.desc, gotPrm)
				case action.wantOutput && !gotOutput:
					t.Errorf("\nHandlePrepare\ntest nr:%d\naction nr: %d\naction description: %s\nwant %v\ngot no output",
						i+1, j+1, action.desc, action.wantPrm)
				case action.wantOutput && gotOutput:
					if !reflect.DeepEqual(gotPrm, action.wantPrm) {
						t.Errorf("\nHandlePrepare\ntest nr:%d\naction nr: %d\naction description: %s\nwant:\t%v\ngot:\t%v",
							i+1, j+1, action.desc, action.wantPrm, gotPrm)
					}
				}
			case accept:
				gotLrn, gotOutput := test.acceptor.handleAccept(action.accept)
				switch {
				case !action.wantOutput && gotOutput:
					t.Errorf("\nHandleAccept\ntest nr:%dnaction nr: %d\naction description: %s\nwant no output\ngot %v",
						i+1, j+1, action.desc, gotLrn)
				case action.wantOutput && !gotOutput:
					t.Errorf("\nHandleAccept\ntest nr:%d\naction nr: %d\naction description: %s\nwant %v\ngot no output",
						i+1, j+1, action.desc, action.wantLrn)
				case action.wantOutput && gotOutput:
					if gotLrn != action.wantLrn {
						t.Errorf("\nHandleAccept\ntest nr:%d\naction nr: %d\naction description: %s\nwant:\t%v\ngot:\t%v",
							i+1, j+1, action.desc, action.wantLrn, gotLrn)
					}
				}
			default:
				t.Fatal("assertion failed: unkown message type for acceptor")
			}
		}
	}
}

type msgtype int

const (
	prepare msgtype = iota
	accept
)

type acceptorAction struct {
	desc       string
	msgtype    msgtype
	prepare    Prepare
	accept     Accept
	wantOutput bool
	wantPrm    Promise
	wantLrn    Learn
}

var acceptorTests = []struct {
	acceptor *Acceptor
	actions  []acceptorAction
}{
	{
		NewAcceptor(0, nil, nil),
		[]acceptorAction{
			{
				desc:    "prepare slot 1, crnd 2 -> output corresponding promise",
				msgtype: prepare,
				prepare: Prepare{
					From: 2,
					Slot: 1,
					Crnd: 2,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   2,
					From: 0,
					Rnd:  2,
				},
			},
			{
				desc:    "prepare slot 1, crnd 1 -> no output, ignore due to lower crnd",
				msgtype: prepare,
				prepare: Prepare{
					From: 1,
					Slot: 1,
					Crnd: 1,
				},
				wantOutput: false,
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		[]acceptorAction{
			{
				desc:    "prepare with crnd 0 and slot 1, inital acceptor round is NoRound (-1) -> output and promise",
				msgtype: prepare,
				prepare: Prepare{
					From: 2,
					Slot: 1,
					Crnd: 0,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   2,
					From: 0,
					Rnd:  0,
				},
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		[]acceptorAction{
			{
				desc:    "accept for slot 1 with rnd 2, current acceptor rnd should be NoRound (-1)  -> output learn with correct slot, rnd and value",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueThree,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueThree,
				},
			},
			{
				desc:    "prepare, crnd 1 and slot 1, previous seen accept (rnd 2) and repsonded with learn -> no output, ignore promise",
				msgtype: prepare,
				prepare: Prepare{
					From: 1,
					Slot: 1,
					Crnd: 1,
				},
				wantOutput: false,
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		[]acceptorAction{
			{
				desc:    "prepare with crnd 2, no previous prepare or accepts -> promise with correct rnd and no slots",
				msgtype: prepare,
				prepare: Prepare{
					From: 2,
					Slot: 0,
					Crnd: 2,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   2,
					From: 0,
					Rnd:  2,
				},
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		[]acceptorAction{
			{
				desc:    "prepare for slot 1 with round 1, no previous history (slots) -> output prepare wih correct rnd and not slots",
				msgtype: prepare,
				prepare: Prepare{
					From: 1,
					Crnd: 1,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   1,
					From: 0,
					Rnd:  1,
				},
			},
			{
				desc:    "another prepare for slot 1 with round 2  -> output another prepare wih correct rnd and not slots",
				msgtype: prepare,
				prepare: Prepare{
					From: 2,
					Crnd: 2,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   2,
					From: 0,
					Rnd:  2,
				},
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		[]acceptorAction{
			{
				desc:    "prepare for slot 1, crnd 2 with no previous history -> output correct promise",
				msgtype: prepare,
				prepare: Prepare{
					From: 2,
					Slot: 1,
					Crnd: 2,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   2,
					From: 0,
					Rnd:  2,
				},
			},
			{
				desc:    "accept for slot 1 with current round -> output learn with correct slot, rnd and value",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
			},
			{
				desc:    "new prepare for slot 1 with higher crnd -> output promise with correct rnd and history (slot 1)",
				msgtype: prepare,
				prepare: Prepare{
					From: 1,
					Slot: 1,
					Crnd: 3,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   1,
					From: 0,
					Rnd:  3,
					Slots: []PromiseSlot{
						{
							ID:   1,
							Vrnd: 2,
							Vval: testingValueOne,
						},
					},
				},
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		[]acceptorAction{
			{
				desc:    "prepare for slot 1, crnd 2, no previous history -> output correct promise",
				msgtype: prepare,
				prepare: Prepare{
					From: 2,
					Slot: 1,
					Crnd: 2,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   2,
					From: 0,
					Rnd:  2,
				},
			},
			{
				desc:    "accept for slot 1 with current rnd -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
			},
			{
				desc:    "accept for slot 1 with _higher_ rnd -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 1,
					Rnd:  5,
					Val:  testingValueTwo,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 1,
					Rnd:  5,
					Val:  testingValueTwo,
				},
			},
			{
				desc:    "accept for slot 2 with _higher_ rnd -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 2,
					Rnd:  8,
					Val:  testingValueThree,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 2,
					Rnd:  8,
					Val:  testingValueThree,
				},
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		[]acceptorAction{
			{
				desc:    "prepare for slot 1, crnd 4, no previous history -> out correct promise",
				msgtype: prepare,
				prepare: Prepare{
					From: 1,
					Slot: 1,
					Crnd: 4,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   1,
					From: 0,
					Rnd:  4,
				},
			},
			{
				desc:    "accept with _lower_ rnd (2) than out current rnd (4) -> no output, ignore accept ",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
				wantOutput: false,
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		[]acceptorAction{
			{
				desc:    "prepare for slot 1, crnd 2, no previous history -> output correct promise",
				msgtype: prepare,
				prepare: Prepare{
					From: 2,
					Slot: 1,
					Crnd: 2,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   2,
					From: 0,
					Rnd:  2,
				},
			},
			{
				desc:    "accept for slot 1, rnd 2 -> output learn with correct slot, rnd and value",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
			},
			{
				desc:    "accept for slot 2, rnd 2 -> output learn with correct slot, rnd and value",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 3,
					Rnd:  2,
					Val:  testingValueTwo,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 3,
					Rnd:  2,
					Val:  testingValueTwo,
				},
			},
			{
				desc:    "accept for slot 4, rnd 2 -> output learn with correct slot, rnd and value",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 4,
					Rnd:  2,
					Val:  testingValueThree,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 4,
					Rnd:  2,
					Val:  testingValueThree,
				},
			},
			{
				desc:    "new prepare for slot 2, crnd 3 -> output promise with correct rnd and history (slot 3 and 4)",
				msgtype: prepare,
				prepare: Prepare{
					From: 1,
					Slot: 2,
					Crnd: 3,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   1,
					From: 0,
					Rnd:  3,
					Slots: []PromiseSlot{
						{
							ID:   3,
							Vrnd: 2,
							Vval: testingValueTwo,
						},
						{
							ID:   4,
							Vrnd: 2,
							Vval: testingValueThree,
						},
					},
				},
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		[]acceptorAction{
			{
				desc:    "prepare for slot 1, crnd 2, no previous history -> output correct promise",
				msgtype: prepare,
				prepare: Prepare{
					From: 2,
					Slot: 1,
					Crnd: 2,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   2,
					From: 0,
					Rnd:  2,
				},
			},
			{
				desc:    "accept for slot 1, rnd 2 -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
			},
			{
				desc:    "accept for slot 3, with different sender and lower round -> no output, ignore accept",
				msgtype: accept,
				accept: Accept{
					From: 1,
					Slot: 3,
					Rnd:  1,
					Val:  testingValueTwo,
				},
				wantOutput: false,
			},
			{
				desc:    "accept for slot 4, rnd 2 -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 4,
					Rnd:  2,
					Val:  testingValueThree,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 4,
					Rnd:  2,
					Val:  testingValueThree,
				},
			},
			{
				desc:    "accept for slot 5 with _higher_ rnd (5) -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 5,
					Rnd:  5,
					Val:  testingValueOne,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 5,
					Rnd:  5,
					Val:  testingValueOne,
				},
			},
			{
				desc:    "accept for slot 7 with old lower rnd (1) -> no output, ignore accept",
				msgtype: accept,
				accept: Accept{
					From: 1,
					Slot: 7,
					Rnd:  1,
					Val:  testingValueTwo,
				},
				wantOutput: false,
			},
			{
				desc:    "new prepare for slot 2 and higher round (7) -> output promise with correct rnd (7) and history (slot 4 and 5)",
				msgtype: prepare,
				prepare: Prepare{
					From: 1,
					Slot: 2,
					Crnd: 7,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   1,
					From: 0,
					Rnd:  7,
					Slots: []PromiseSlot{
						{
							ID:   4,
							Vrnd: 2,
							Vval: testingValueThree,
						},
						{
							ID:   5,
							Vrnd: 5,
							Vval: testingValueOne,
						},
					},
				},
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		[]acceptorAction{
			{
				desc:    "prepare for slot 1, crnd 2, no previous history -> output correct promise",
				msgtype: prepare,
				prepare: Prepare{
					From: 2,
					Slot: 1,
					Crnd: 2,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   2,
					From: 0,
					Rnd:  2,
				},
			},
			{
				desc:    "accept for slot 1, rnd 2 -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
			},
			{
				desc:    "accept for slot 3, with different sender and lower round -> no output, ignore accept",
				msgtype: accept,
				accept: Accept{
					From: 1,
					Slot: 3,
					Rnd:  1,
					Val:  testingValueTwo,
				},
				wantOutput: false,
			},
			{
				desc:    "accept for slot 4, rnd 2 -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 4,
					Rnd:  2,
					Val:  testingValueThree,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 4,
					Rnd:  2,
					Val:  testingValueThree,
				},
			},
			{
				desc:    "accept for slot 6 with _higher_ rnd (5) -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 6,
					Rnd:  5,
					Val:  testingValueOne,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 6,
					Rnd:  5,
					Val:  testingValueOne,
				},
			},
			{
				desc:    "accept for slot 7 with old lower rnd (1) -> no output, ignore accept",
				msgtype: accept,
				accept: Accept{
					From: 1,
					Slot: 7,
					Rnd:  1,
					Val:  testingValueTwo,
				},
				wantOutput: false,
			},
			{
				desc:    "new prepare for slot 2 and higher round (7) -> output promise with correct rnd (7) and history (slot 4 and 6)",
				msgtype: prepare,
				prepare: Prepare{
					From: 1,
					Slot: 2,
					Crnd: 7,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   1,
					From: 0,
					Rnd:  7,
					Slots: []PromiseSlot{
						{
							ID:   4,
							Vrnd: 2,
							Vval: testingValueThree,
						},
						{
							ID:   6,
							Vrnd: 5,
							Vval: testingValueOne,
						},
					},
				},
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		[]acceptorAction{
			{
				desc:    "prepare for slot 1, crnd 2, no previous history -> output correct promise",
				msgtype: prepare,
				prepare: Prepare{
					From: 2,
					Slot: 1,
					Crnd: 2,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   2,
					From: 0,
					Rnd:  2,
				},
			},
			{
				desc:    "accept for slot 1, rnd 2 -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
			},
			{
				desc:    "accept for slot 1 again but with rnd 5 -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 1,
					Rnd:  5,
					Val:  testingValueOne,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 1,
					Rnd:  5,
					Val:  testingValueOne,
				},
			},
			{
				desc:    "accept for slot 3, with different sender and lower round -> no output, ignore accept",
				msgtype: accept,
				accept: Accept{
					From: 1,
					Slot: 3,
					Rnd:  1,
					Val:  testingValueTwo,
				},
				wantOutput: false,
			},
			{
				desc:    "accept for slot 4, rnd 5 -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 4,
					Rnd:  5,
					Val:  testingValueThree,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 4,
					Rnd:  5,
					Val:  testingValueThree,
				},
			},
			{
				desc:    "accept again for slot 4, with _higher_ round, rnd 8 -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 4,
					Rnd:  8,
					Val:  testingValueThree,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 4,
					Rnd:  8,
					Val:  testingValueThree,
				},
			},
			{
				desc:    "accept for slot 6 with _higher_ rnd (11) -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 6,
					Rnd:  11,
					Val:  testingValueOne,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 6,
					Rnd:  11,
					Val:  testingValueOne,
				},
			},
			{
				desc:    "accept for slot 7 with old lower rnd (1) -> no output, ignore accept",
				msgtype: accept,
				accept: Accept{
					From: 1,
					Slot: 7,
					Rnd:  1,
					Val:  testingValueTwo,
				},
				wantOutput: false,
			},
			{
				desc:    "new prepare for slot 2 and higher round (13) from 1 -> output promise with correct rnd (13) and history (slot 4 and 6)",
				msgtype: prepare,
				prepare: Prepare{
					From: 1,
					Slot: 2,
					Crnd: 13,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   1,
					From: 0,
					Rnd:  13,
					Slots: []PromiseSlot{
						{
							ID:   4,
							Vrnd: 8,
							Vval: testingValueThree,
						},
						{
							ID:   6,
							Vrnd: 11,
							Vval: testingValueOne,
						},
					},
				},
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		[]acceptorAction{
			{
				desc:    "prepare for slot 1, crnd 2, no previous history -> output correct promise",
				msgtype: prepare,
				prepare: Prepare{
					From: 2,
					Slot: 1,
					Crnd: 2,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   2,
					From: 0,
					Rnd:  2,
				},
			},
			{
				desc:    "accept for slot 1 with current rnd -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
			},
			{
				desc:    "accept for slot 2 with _higher_ rnd -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 2,
					Rnd:  5,
					Val:  testingValueThree,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 2,
					Rnd:  5,
					Val:  testingValueThree,
				},
			},
			{
				desc:    "accept for slot 1 again with _higher_ rnd after we previously have sent accept for slot 2 -> output correct learn",
				msgtype: accept,
				accept: Accept{
					From: 2,
					Slot: 1,
					Rnd:  8,
					Val:  testingValueTwo,
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Slot: 1,
					Rnd:  8,
					Val:  testingValueTwo,
				},
			},
		},
	},
}
