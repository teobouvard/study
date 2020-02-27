package singlepaxos

import "testing"

func TestHandlePrepareAndAccept(t *testing.T) {
	for i, test := range acceptorTests {
		for j, action := range test.actions {
			switch action.msgtype {
			case prepare:
				gotPrm, gotOutput := test.acceptor.handlePrepare(action.prepare)
				switch {
				case !action.wantOutput && gotOutput:
					t.Errorf("\nHandlePrepare\ntest nr:%d\ndescription: %s\naction nr: %d\nwant no output\ngot %v",
						i+1, test.desc, j+1, gotPrm)
				case action.wantOutput && !gotOutput:
					t.Errorf("\nHandlePrepare\ntest nr:%d\ndescription: %s\naction nr: %d\nwant %v\ngot no output",
						i+1, test.desc, j+1, action.wantPrm)
				case action.wantOutput && gotOutput:
					if gotPrm != action.wantPrm {
						t.Errorf("\nHandlePrepare\ntest nr:%d\ndescription: %s\naction nr: %d\nwant: %v\ngot: %v",
							i+1, test.desc, j+1, action.wantPrm, gotPrm)
					}
				}
			case accept:
				gotLrn, gotOutput := test.acceptor.handleAccept(action.accept)
				switch {
				case !action.wantOutput && gotOutput:
					t.Errorf("\nHandleAccept\ntest nr:%d\ndescription: %s\naction nr: %d\nwant no output\ngot %v",
						i+1, test.desc, j+1, gotLrn)
				case action.wantOutput && !gotOutput:
					t.Errorf("\nHandleAccept\ntest nr:%d\ndescription: %s\naction nr: %d\nwant %v\ngot no output",
						i+1, test.desc, j+1, action.wantLrn)
				case action.wantOutput && gotOutput:
					if gotLrn != action.wantLrn {
						t.Errorf("\nHandleAccept\ntest nr:%d\ndescription: %s\naction nr: %d\nwant: %v\ngot: %v",
							i+1, test.desc, j+1, action.wantLrn, gotLrn)
					}
				}
			default:
				t.Fatal("assertion failed: unkown messages type for acceptor")
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
	msgtype    msgtype
	prepare    Prepare
	accept     Accept
	wantOutput bool
	wantPrm    Promise
	wantLrn    Learn
}

var acceptorTests = []struct {
	acceptor *Acceptor
	desc     string
	actions  []acceptorAction
}{
	{
		NewAcceptor(0, nil, nil),
		"no previous received prepare -> reply with correct rnd and no vrnd/vval",
		[]acceptorAction{
			{
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
					Vrnd: NoRound,
					Vval: ZeroValue,
				},
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		"two prepares, the second with higher round -> reply correctly to both",
		[]acceptorAction{
			{
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
					Vrnd: NoRound,
					Vval: ZeroValue,
				},
			},
			{
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
					Vrnd: NoRound,
					Vval: ZeroValue,
				},
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		"single prepare followed by corresponding accept -> emitt learn. then new prepare with higher round -> report correct vrnd, vval",
		[]acceptorAction{
			{
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
					Vrnd: NoRound,
					Vval: ZeroValue,
				},
			},
			{
				msgtype: accept,
				accept: Accept{
					From: 1,
					Rnd:  1,
					Val:  "Lamport",
				},
				wantOutput: true,
				wantLrn: Learn{
					From: 0,
					Rnd:  1,
					Val:  "Lamport",
				},
			},
			{
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
					Vrnd: 1,
					Vval: "Lamport",
				},
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		"prepare with crnd lower than seen rnd -> ignore prepare",
		[]acceptorAction{
			{
				msgtype: prepare,
				prepare: Prepare{
					From: 1,
					Crnd: 2,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   1,
					From: 0,
					Rnd:  2,
					Vrnd: NoRound,
					Vval: ZeroValue,
				},
			},
			{
				msgtype: prepare,
				prepare: Prepare{
					From: 1,
					Crnd: 1,
				},
				wantOutput: false,
			},
		},
	},
	{
		NewAcceptor(0, nil, nil),
		"accept with lower rnd than what we have sent in promise -> ignore accept, i.e. no learn",
		[]acceptorAction{
			{
				msgtype: prepare,
				prepare: Prepare{
					From: 1,
					Crnd: 2,
				},
				wantOutput: true,
				wantPrm: Promise{
					To:   1,
					From: 0,
					Rnd:  2,
					Vrnd: NoRound,
					Vval: ZeroValue,
				},
			},
			{
				msgtype: accept,
				accept: Accept{
					From: 2,
					Rnd:  1,
					Val:  "Lamport",
				},
				wantOutput: false,
			},
		},
	},
}
