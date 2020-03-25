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
The value it last accepted means the value that last was sent from the acceptor of this node to the learners of all nodes. In other words it's the value that was last processed to the end and was sent out from the acceptor to the learners.
An instance can also be called a slot. It describes the number of executions that can be made after the proposer is accepted for this instance.

5. Explain, with an example, what will happen if there are multiple proposers.
Given a network of three nodes A, B and C.
A and B propose.
A cannot promise the lead to B, as it already promised the lead to itself and B cannot promise the lead to A due to the same reason.
The proposal of A reaches C first and C promises the lead to A.
A receives the promise of C and obtains a majority of promises. Therefore A is the new leader.

6. What happens if two proposers both believe themselves to be the leader and send Prepare messages simultaneously?
There's the following possibilities:
- One of them still reaches a majority of promises and can be the leader
- Noone receives a majority of promises. After some time another prepare message has to be sent out to retry the precedure.

7. What can we say about system synchrony if there are multiple proposers (or leaders)?
This means that there are communication issues inside the network, because some messages between proposers and acceptors were lost.
Consequently, the system is not synchron anymore.

8. Can an acceptor accept one value in round 1 and another value in round 2? Explain.
Yes. An acceptor can accept new values every new round.

9. What must happen for a value to be “chosen”? What is the connection between chosen values and learned values?
A value is chosen when the following steps have happened:
The proposer with the value proposes and receives a majority of promises. Afterwards it sends the accept message with the value to all the acceptors. The acceptors update their properties and send learn messages to the learners. When a learner receives a majority of learn messages, it updates it's property and tells the choice to the client.
Learned values are the values that will then be sent out to the client. The value has to be a learnt value before it can be sent to the client what makes it a chosen value.
