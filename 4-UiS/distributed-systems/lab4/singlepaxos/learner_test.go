package singlepaxos

import "testing"

func TestHandleLearn(t *testing.T) {
	for i, test := range learnerTests {
		for j, action := range test.actions {
			gotVal, gotOutput := test.learner.handleLearn(action.learn)
			switch {
			case !action.wantOutput && gotOutput:
				t.Errorf("\nHandleLearn\ntest nr:%d\ndescription: %s\naction nr: %d\nwant no output\ngot: %v",
					i+1, test.desc, j+1, gotVal)
			case action.wantOutput && !gotOutput:
				t.Errorf("\nHandleLearn\ntest nr:%d\ndescription: %s\naction nr: %d\nwant: %v\ngot no output",
					i+1, test.desc, j+1, action.wantVal)
			case action.wantOutput && gotOutput:
				if gotVal != action.wantVal {
					t.Errorf("\nHandleLearn\ntest nr:%d\ndescription: %s\naction nr: %d\nwant: %v\ngot: %v",
						i+1, test.desc, j+1, action.wantVal, gotVal)
				}
			}

		}
	}
}

type learnerAction struct {
	learn      Learn
	wantOutput bool
	wantVal    Value
}

var learnerTests = []struct {
	learner *Learner
	desc    string
	actions []learnerAction
}{
	{
		NewLearner(0, 3, nil),
		"single learn, 3 nodes, no quorum -> no output",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Rnd:  1,
					Val:  "Lamport",
				},
				false,
				ZeroValue,
			},
		},
	},
	{
		NewLearner(0, 3, nil),
		"two learns, 3 nodes, same round and value, unique senders = quorum -> report output and value",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Rnd:  1,
					Val:  "Lamport",
				},
				false,
				ZeroValue,
			},
			{
				Learn{
					From: 2,
					Rnd:  1,
					Val:  "Lamport",
				},
				true,
				"Lamport",
			},
		},
	},
	{
		NewLearner(0, 3, nil),
		"two learns, 3 nodes, same round and value, same sender = no quorum -> no output",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Rnd:  1,
					Val:  "Lamport",
				},
				false,
				ZeroValue,
			},
			{
				Learn{
					From: 1,
					Rnd:  1,
					Val:  "Lamport",
				},
				false,
				ZeroValue,
			},
		},
	},
	{
		NewLearner(0, 3, nil),
		"two learns, 3 nodes, different rounds, unique senders = no quorum -> no output",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Rnd:  1,
					Val:  "Lamport",
				},
				false,
				ZeroValue,
			},
			{
				Learn{
					From: 2,
					Rnd:  2,
					Val:  "Lamport",
				},
				false,
				ZeroValue,
			},
		},
	},
	{
		NewLearner(0, 3, nil),
		"two learns, 3 nodes, second learn should be ignored due to lower round -> no output",
		[]learnerAction{
			{
				Learn{
					From: 2,
					Rnd:  2,
					Val:  "Lamport",
				},
				false,
				ZeroValue,
			},
			{
				Learn{
					From: 1,
					Rnd:  1,
					Val:  "Lamport",
				},
				false,
				ZeroValue,
			},
		},
	},
	{
		NewLearner(0, 3, nil),
		"3 nodes, single learn with rnd 2, then two learns with rnd 4 (quorum) -> report output and value of quorum",
		[]learnerAction{
			{
				Learn{
					From: 2,
					Rnd:  2,
					Val:  "Lamport",
				},
				false,
				ZeroValue,
			},
			{
				Learn{
					From: 1,
					Rnd:  4,
					Val:  "Leslie",
				},
				false,
				ZeroValue,
			},
			{
				Learn{
					From: 0,
					Rnd:  4,
					Val:  "Leslie",
				},
				true,
				"Leslie",
			},
		},
	},
}
