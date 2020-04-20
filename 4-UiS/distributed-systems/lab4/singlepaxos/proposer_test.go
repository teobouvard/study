package singlepaxos

import "testing"

func TestHandlePromise(t *testing.T) {
	for i, test := range proposerTests {
		test.proposer.clientValue = test.clientValue
		for j, action := range test.actions {
			gotAcc, gotOutput := test.proposer.handlePromise(action.promise)
			switch {
			case !action.wantOutput && gotOutput:
				t.Errorf("\nHandlePromise\ntest nr:%d\ndescription: %s\naction nr: %d\nwant no output\ngot %v",
					i+1, test.desc, j+1, gotAcc)
			case action.wantOutput && !gotOutput:
				t.Errorf("\nHandlePromise\ntest nr:%d\ndescription: %s\naction nr: %d\nwant %v\ngot no output",
					i+1, test.desc, j+1, action.wantAcc)
			case action.wantOutput && gotOutput:
				if gotAcc != action.wantAcc {
					t.Errorf("\nHandlePromise\ntest nr:%d\ndescription: %s\naction nr: %d\nwant: %v\ngot: %v",
						i+1, test.desc, j+1, action.wantAcc, gotAcc)
				}
			}
		}
	}
}

const (
	valueFromClientOne = "A client command"
	valueFromClientTwo = "Another client command"
)

type paction struct {
	promise    Promise
	wantOutput bool
	wantAcc    Accept
}

var proposerTests = []struct {
	proposer    *Proposer
	desc        string
	clientValue Value
	actions     []paction
}{
	{
		NewProposer(2, 3, &mockLD{}, nil, nil),
		"no quorum -> no output",
		ZeroValue,
		[]paction{
			{
				Promise{
					To:   2,
					From: 1,
					Rnd:  0,
					Vrnd: NoRound,
					Vval: ZeroValue,
				},
				false,
				Accept{},
			},
		},
	},
	{
		NewProposer(2, 3, &mockLD{}, nil, nil),
		"valid quorum and no value reported -> propose (send accept) client value (free value) from proposer.clientValue field (I)",
		valueFromClientOne,
		[]paction{
			{
				Promise{
					To:   2,
					From: 1,
					Rnd:  2,
					Vrnd: NoRound,
					Vval: ZeroValue,
				},
				false,
				Accept{},
			},
			{
				Promise{
					To:   2,
					From: 0,
					Rnd:  2,
					Vrnd: NoRound,
					Vval: ZeroValue,
				},
				true,
				Accept{
					From: 2,
					Rnd:  2,
					Val:  valueFromClientOne,
				},
			},
		},
	},
	{
		NewProposer(2, 3, &mockLD{}, nil, nil),
		"valid quorum and no value reported -> propose (send accept) client value (free value) from proposer.clientValue field (II)",
		valueFromClientTwo,
		[]paction{
			{
				Promise{
					To:   2,
					From: 1,
					Rnd:  2,
					Vrnd: NoRound,
					Vval: ZeroValue,
				},
				false,
				Accept{},
			},
			{
				Promise{
					To:   2,
					From: 0,
					Rnd:  2,
					Vrnd: NoRound,
					Vval: ZeroValue,
				},
				true,
				Accept{
					From: 2,
					Rnd:  2,
					Val:  valueFromClientTwo,
				},
			},
		},
	},
	{
		NewProposer(2, 3, &mockLD{}, nil, nil),
		"valid quorum and a value reported -> propose correct value in accept",
		ZeroValue,
		[]paction{
			{
				Promise{
					To:   2,
					From: 1,
					Rnd:  2,
					Vrnd: NoRound,
					Vval: "",
				},
				false,
				Accept{},
			},
			{
				Promise{
					To:   2,
					From: 0,
					Rnd:  2,
					Vrnd: 1,
					Vval: "Leslie",
				},
				true,
				Accept{
					From: 2,
					Rnd:  2,
					Val:  "Leslie",
				},
			},
		},
	},
	{
		NewProposer(2, 3, &mockLD{}, nil, nil),
		"promise for different round than our current one -> ignore promise",
		ZeroValue,
		[]paction{
			{
				Promise{
					To:   2,
					From: 1,
					Rnd:  1,
					Vrnd: NoRound,
					Vval: "",
				},
				false,
				Accept{},
			},
		},
	},
	{
		NewProposer(2, 3, &mockLD{}, nil, nil),
		"three promises, all for different rounds than our current one -> ignore all promises",
		ZeroValue,
		[]paction{
			{
				Promise{
					To:   2,
					From: 1,
					Rnd:  1,
					Vrnd: NoRound,
					Vval: "",
				},
				false,
				Accept{},
			},
			{
				Promise{
					To:   2,
					From: 0,
					Rnd:  6,
					Vrnd: NoRound,
					Vval: "",
				},
				false,
				Accept{},
			},
			{
				Promise{
					To:   2,
					From: 1,
					Rnd:  4,
					Vrnd: NoRound,
					Vval: "",
				},
				false,
				Accept{},
			},
		},
	},
	{
		NewProposer(2, 3, &mockLD{}, nil, nil),
		"three identical promises from the same sender -> no quourm, no output",
		ZeroValue,
		[]paction{
			{
				Promise{
					To:   2,
					From: 1,
					Rnd:  2,
					Vrnd: NoRound,
					Vval: "",
				},
				false,
				Accept{},
			},
			{
				Promise{
					To:   2,
					From: 1,
					Rnd:  2,
					Vrnd: NoRound,
					Vval: "",
				},
				false,
				Accept{},
			},
			{
				Promise{
					To:   2,
					From: 1,
					Rnd:  2,
					Vrnd: NoRound,
					Vval: "",
				},
				false,
				Accept{},
			},
		},
	},
	{
		NewProposer(2, 3, &mockLD{}, nil, nil),
		"three identical promises from the same sender for our round  -> no quourm, no output\n" +
			"then single promise for different round then ours -> ignore, no quorum, no output\n" +
			"then single promise for our round from last node -> quorum, report output and accept",
		valueFromClientTwo,
		[]paction{
			{
				Promise{
					To:   2,
					From: 1,
					Rnd:  2,
					Vrnd: NoRound,
					Vval: "",
				},
				false,
				Accept{},
			},
			{
				Promise{
					To:   2,
					From: 1,
					Rnd:  2,
					Vrnd: NoRound,
					Vval: "",
				},
				false,
				Accept{},
			},
			{
				Promise{
					To:   2,
					From: 1,
					Rnd:  2,
					Vrnd: NoRound,
					Vval: "",
				},
				false,
				Accept{},
			},
			{
				Promise{
					To:   2,
					From: 0,
					Rnd:  0,
					Vrnd: NoRound,
					Vval: "",
				},
				false,
				Accept{},
			},
			{
				Promise{
					To:   2,
					From: 0,
					Rnd:  2,
					Vrnd: NoRound,
					Vval: "",
				},
				true,
				Accept{
					From: 2,
					Rnd:  2,
					Val:  valueFromClientTwo,
				},
			},
		},
	},
	{
		NewProposer(2, 3, &mockLD{}, nil, nil),
		"valid quorum and two different values reported -> propose correct value (highest vrnd) in accept",
		ZeroValue,
		[]paction{
			{
				Promise{
					To:   2,
					From: 0,
					Rnd:  2,
					Vrnd: 1,
					Vval: "Lamport",
				},
				false,
				Accept{},
			},
			{
				Promise{
					To:   2,
					From: 1,
					Rnd:  2,
					Vrnd: 0,
					Vval: "Leslie",
				},
				true,
				Accept{
					From: 2,
					Rnd:  2,
					Val:  "Lamport",
				},
			},
		},
	},
	{
		NewProposer(2, 3, &mockLD{}, nil, nil),
		"two promises (majority) for different round than our current one -> ignore all promises and no output",
		ZeroValue,
		[]paction{
			{
				Promise{
					To:   2,
					From: 0,
					Rnd:  6,
					Vrnd: NoRound,
					Vval: "",
				},
				false,
				Accept{},
			},
			{
				Promise{
					To:   2,
					From: 1,
					Rnd:  6,
					Vrnd: NoRound,
					Vval: "",
				},
				false,
				Accept{},
			},
		},
	},
}

var incCrndTests = []struct {
	id    int
	n     int
	crnds []Round
}{
	{
		0,
		3,
		[]Round{0, 3, 6, 9, 12, 15, 18, 21},
	},
	{
		1,
		5,
		[]Round{1, 6, 11, 16, 21, 26, 31, 36},
	},
	{
		4,
		7,
		[]Round{4, 11, 18, 25, 32, 39, 46},
	},
}

func TestIncreaseCrnd(t *testing.T) {
	for i, test := range incCrndTests {
		proposer := NewProposer(test.id, test.n, &mockLD{}, nil, nil)
		for j, wantCrnd := range test.crnds {
			if proposer.crnd != wantCrnd {
				t.Errorf("TestIncreaseCrnd %d, inc nr %d: proposer has current crnd %d, should have %d",
					i, j, proposer.crnd, wantCrnd)
			}
			proposer.increaseCrnd()
		}
	}
}

type mockLD struct{}

func (l *mockLD) Leader() int {
	return -1
}

func (l *mockLD) Subscribe() <-chan int {
	return make(chan int)
}
