package multipaxos

import "testing"

func TestHandleLearn(t *testing.T) {
	testHandleLearn(learnerTestsThreeNodes, t)
	testHandleLearn(learnerTestsFiveNodes, t)
}

func testHandleLearn(tests []learnerTest, t *testing.T) {
	for i, test := range tests {
		for j, action := range test.actions {
			gotVal, gotSid, gotOutput := test.learner.handleLearn(action.learn)
			switch {
			case !action.wantOutput && gotOutput:
				t.Errorf("\nHandleLearn\ntest nr:%d\ndescription: %s\naction nr: %d\nwant no output\ngot: %v",
					i+1, test.desc, j+1, gotVal)
			case action.wantOutput && !gotOutput:
				t.Errorf("\nHandleLearn\ntest nr:%d\ndescription: %s\naction nr: %d\nwant: %v\ngot no output",
					i+1, test.desc, j+1, action.wantVal)
			case action.wantOutput && gotOutput:
				if gotVal != action.wantVal {
					t.Errorf("\nHandleLearn\ntest nr:%d\ndescription: %s\naction nr: %d\nwant value: %v\ngot value: %v",
						i+1, test.desc, j+1, action.wantVal, gotVal)
				}
				if gotSid != action.wantSid {
					t.Errorf("\nHandleLearn\ntest nr:%d\ndescription: %s\naction nr: %d\nwant slot id: %v\ngot slot id: %v",
						i+1, test.desc, j+1, action.wantSid, gotSid)
				}
			}
		}
	}
}

type learnerTest struct {
	learner *Learner
	desc    string
	actions []learnerAction
}

type learnerAction struct {
	learn      Learn
	wantOutput bool
	wantVal    Value
	wantSid    SlotID
}

var learnerTestsThreeNodes = []learnerTest{
	{
		NewLearner(0, 3, nil),
		"single learn for slot 1, 3 nodes, no quorum -> report zero value",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  1,
					Val:  testingValueOne,
				},
				false,
				Value{},
				0,
			},
		},
	},
	{

		NewLearner(0, 3, nil),
		"single learn for slot 1&2, 3 nodes, no quorum for both slots -> report zero value",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  1,
					Val:  testingValueOne,
				},
				false,
				Value{},
				0,
			},
		},
	},

	{
		NewLearner(0, 3, nil),
		"two learns for slot 1, 3 nodes, equal round and value, unique senders = quorum -> report output and value",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  1,
					Val:  testingValueOne,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 1,
					Rnd:  1,
					Val:  testingValueOne,
				},
				true,
				testingValueOne,
				1,
			},
		},
	},
	{
		NewLearner(0, 3, nil),
		"two learns for slot 1, 3 nodes, equal round and value, same sender = no quorum -> no output",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  1,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  1,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
		},
	},
	{
		NewLearner(0, 3, nil),
		"two learns for slot 1, 3 nodes, same sender but different round = no quorum -> no output",
		[]learnerAction{
			{
				Learn{
					From: 2,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 1,
					Rnd:  5,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
		},
	},
	{
		NewLearner(0, 3, nil),
		"two learns for slot 1, 3 nodes, different rounds, unique senders = no quorum -> no output",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Rnd:  1,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Rnd:  2,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
		},
	},
	{
		NewLearner(0, 3, nil),
		"two learns for slot 1, 3 nodes, second learn should be ignored due to lower round -> no output",
		[]learnerAction{
			{
				Learn{
					From: 2,
					Rnd:  2,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Rnd:  1,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
		},
	},
	{
		NewLearner(0, 3, nil),
		"single learn for slot 1-6, 3 nodes, no quorum each slot -> no output for every learn received",
		[]learnerAction{
			{
				Learn{
					From: 2,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueOne,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 2,
					Rnd:  1,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 3,
					Rnd:  5,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 4,
					Rnd:  2,
					Val:  testingValueOne,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 5,
					Rnd:  1,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 6,
					Rnd:  5,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
		},
	},
	{
		NewLearner(2, 3, nil),
		"quorum of learns for slot 1-3, 3 nodes -> report output and value for every slot",
		[]learnerAction{
			{
				Learn{
					From: 0,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				true,
				testingValueTwo,
				1,
			},
			{
				Learn{
					From: 1,
					Slot: 2,
					Rnd:  3,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 0,
					Slot: 2,
					Rnd:  3,
					Val:  testingValueThree,
				},
				true,
				testingValueThree,
				2,
			},
			{
				Learn{
					From: 0,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				true,
				testingValueTwo,
				3,
			},
		},
	},
	{
		NewLearner(2, 3, nil),
		"quorum of learns (two) for slot 1, the third learn -> report output and value but ignore last learn",
		[]learnerAction{
			{
				Learn{
					From: 0,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				true,
				testingValueTwo,
				1,
			},
			{
				Learn{
					From: 2,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
		},
	},
	{
		NewLearner(2, 3, nil),
		"single learn for slot 42, rnd 3, then quorum of learns (2) for slot 42 with higher rnd (4), -> report output and value after quorum",
		[]learnerAction{
			{
				Learn{
					From: 0,
					Slot: 42,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 42,
					Rnd:  4,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 42,
					Rnd:  4,
					Val:  testingValueThree,
				},
				true,
				testingValueThree,
				42,
			},
		},
	},
	{
		NewLearner(2, 3, nil),
		"quorum of learns for slot 1-3, 3 nodes, learns are not received in slot order, received in replica order," +
			"i.e. you need to store learns for different slots -> report output and value for every slot",
		[]learnerAction{
			{
				Learn{
					From: 0,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 0,
					Slot: 2,
					Rnd:  3,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 0,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},

			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				true,
				testingValueTwo,
				1,
			},
			{
				Learn{
					From: 1,
					Slot: 2,
					Rnd:  3,
					Val:  testingValueThree,
				},
				true,
				testingValueThree,
				2,
			},
			{
				Learn{
					From: 1,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				true,
				testingValueTwo,
				3,
			},
		},
	},
	{
		NewLearner(2, 3, nil),
		"quorum of learns for slot 3-1, 3 nodes, learns are received in mixed slot order (3,1,2)," +
			"-> report output and value for every slot",
		[]learnerAction{
			{
				Learn{
					From: 0,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				true,
				testingValueTwo,
				3,
			},

			{
				Learn{
					From: 0,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				true,
				testingValueTwo,
				1,
			},
			{
				Learn{
					From: 1,
					Slot: 2,
					Rnd:  3,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 0,
					Slot: 2,
					Rnd:  3,
					Val:  testingValueThree,
				},
				true,
				testingValueThree,
				2,
			},
		},
	},
}

var learnerTestsFiveNodes = []learnerTest{
	{

		NewLearner(0, 5, nil),
		"single learn for slot 1, 5 nodes, no quorum -> report zero value",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  1,
					Val:  testingValueOne,
				},
				false,
				Value{},
				0,
			},
		},
	},
	{

		NewLearner(0, 5, nil),
		"single learn for slot 1&2, 5 nodes, no quorum for both slots -> report zero value",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  1,
					Val:  testingValueOne,
				},
				false,
				Value{},
				0,
			},
		},
	},

	{
		NewLearner(0, 5, nil),
		"three learns for slot 1, 5 nodes, equal round and value, unique senders = quorum -> report output and value",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  1,
					Val:  testingValueOne,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 1,
					Rnd:  1,
					Val:  testingValueOne,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 3,
					Slot: 1,
					Rnd:  1,
					Val:  testingValueOne,
				},
				true,
				testingValueOne,
				1,
			},
		},
	},
	{
		NewLearner(0, 5, nil),
		"three learns for slot 1, 5 nodes, equal round and value, same sender = no quorum -> no output",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  1,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  1,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  1,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
		},
	},
	{
		NewLearner(0, 5, nil),
		"three learns for slot 1, 5 nodes, same sender but different round = no quorum -> no output",
		[]learnerAction{
			{
				Learn{
					From: 2,
					Slot: 1,
					Rnd:  2,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 1,
					Rnd:  5,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 1,
					Rnd:  8,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
		},
	},
	{
		NewLearner(0, 5, nil),
		"three learns for slot 1, 5 nodes, different rounds, unique senders = no quorum -> no output",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Rnd:  1,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Rnd:  2,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 3,
					Rnd:  3,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
		},
	},
	{
		NewLearner(0, 5, nil),
		"three learns for slot 1, 5 nodes, 2. and 3. learn should be ignored due to lower round -> no output",
		[]learnerAction{
			{
				Learn{
					From: 2,
					Rnd:  2,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Rnd:  1,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 3,
					Rnd:  0,
					Val:  testingValueOne,
				},
				false,
				Value{},
				0,
			},
		},
	},
	{
		NewLearner(0, 5, nil),
		"single learn for slot 1-6, 5 nodes, no quorum each slot -> no output for every learn received",
		[]learnerAction{
			{
				Learn{
					From: 0,
					Slot: 1,
					Rnd:  1,
					Val:  testingValueOne,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 2,
					Rnd:  2,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 3,
					Slot: 4,
					Rnd:  4,
					Val:  testingValueOne,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 4,
					Slot: 5,
					Rnd:  5,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 6,
					Rnd:  6,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
		},
	},
	{
		NewLearner(2, 5, nil),
		"quorum of learns for slot 1-3, 5 nodes -> report output and value for every slot",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 3,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				true,
				testingValueTwo,
				1,
			},
			{
				Learn{
					From: 1,
					Slot: 2,
					Rnd:  3,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 2,
					Rnd:  3,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},

			{
				Learn{
					From: 3,
					Slot: 2,
					Rnd:  3,
					Val:  testingValueThree,
				},
				true,
				testingValueThree,
				2,
			},
			{
				Learn{
					From: 1,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},

			{
				Learn{
					From: 3,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				true,
				testingValueTwo,
				3,
			},
		},
	},
	{
		NewLearner(2, 5, nil),
		"quorum of learns (three) for slot 1, the fourth learn -> report output and value but ignore last learn",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 3,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				true,
				testingValueTwo,
				1,
			},
			{
				Learn{
					From: 4,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
		},
	},
	{
		NewLearner(2, 5, nil),
		"single learn for slot 42, rnd 3, then quorum of learns (3) for slot 42 with higher rnd (4), -> report output and value after quorum",
		[]learnerAction{
			{
				Learn{
					From: 4,
					Slot: 42,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 42,
					Rnd:  4,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 42,
					Rnd:  4,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 3,
					Slot: 42,
					Rnd:  4,
					Val:  testingValueThree,
				},
				true,
				testingValueThree,
				42,
			},
		},
	},
	{
		NewLearner(2, 5, nil),
		"quorum of learns for slot 1-3, 3 nodes, learns are not received in slot order, received in replica order," +
			"i.e. you need to store learns for different slots -> report output and value for every slot",
		[]learnerAction{
			{
				Learn{
					From: 0,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 0,
					Slot: 2,
					Rnd:  3,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 0,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 2,
					Rnd:  3,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 4,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				true,
				testingValueTwo,
				1,
			},
			{
				Learn{
					From: 4,
					Slot: 2,
					Rnd:  3,
					Val:  testingValueThree,
				},
				true,
				testingValueThree,
				2,
			},
			{
				Learn{
					From: 4,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				true,
				testingValueTwo,
				3,
			},
		},
	},
	{
		NewLearner(2, 5, nil),
		"quorum of learns for slot 3-1, 5 nodes, learns are received in mixed slot order (3,1,2)," +
			"-> report output and value for every slot",
		[]learnerAction{
			{
				Learn{
					From: 1,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 4,
					Slot: 3,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				true,
				testingValueTwo,
				3,
			},
			{
				Learn{
					From: 3,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 0,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 1,
					Slot: 1,
					Rnd:  3,
					Val:  testingValueTwo,
				},
				true,
				testingValueTwo,
				1,
			},
			{
				Learn{
					From: 1,
					Slot: 2,
					Rnd:  3,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 2,
					Slot: 2,
					Rnd:  3,
					Val:  testingValueThree,
				},
				false,
				Value{},
				0,
			},
			{
				Learn{
					From: 0,
					Slot: 2,
					Rnd:  3,
					Val:  testingValueThree,
				},
				true,
				testingValueThree,
				2,
			},
		},
	},
}
