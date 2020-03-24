## Answers to Paxos Questions 

You should write down your answers to the
[questions](https://dat520.github.io/r/?assignments/tree/master/lab4-singlepaxos#questions-10)
for Lab 4 in this file. 

1. Is it possible that Paxos enters an infinite loop? Explain.
Yes, it is possible when the majority of the nodes prepare at the same time and keep retring to prepare at the same time over and over again.
For example:
A sends a prepare(1) message.
Simultaneously, B sends a prepare(1) message.
A denies B's prepare(1).
B denies A's prepare(1).
This could theoretically continue infinitely.

2. Is the value to agree on included in the Prepare message?
No, the value is not included in the prepare message. It is included in the subsequent accept message if the prepare message results in a quorum of promises.

3. Does Paxos rely on an increasing proposal/round number in order to work? Explain.
Yes.
Every time a node prepares to become the leader, the round number is increased. This ensures that the previous leader will not longer be considered as leader by the other nodes as the round number is not equivalent to the current round number. Therefore Paxos relies on round numbers to identify the current leader if more than one node pretend being the leader.

4. Look at this description for Phase 1B: If the proposal number N is larger than any previous proposal, then each Acceptor promises not to accept proposals less than N, and sends the value it last accepted for this instance to the Proposer. What is meant by “the value it last accepted”? And what is an “instance” in this case?

5. Explain, with an example, what will happen if there are multiple proposers.

6. What happens if two proposers both believe themselves to be the leader and send Prepare messages simultaneously?

7. What can we say about system synchrony if there are multiple proposers (or leaders)?

8. Can an acceptor accept one value in round 1 and another value in round 2? Explain.

9. What must happen for a value to be “chosen”? What is the connection between chosen values and learned values?
