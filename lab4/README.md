![UiS](https://www.uis.no/getfile.php/13391907/Biblioteket/Logo%20og%20veiledninger/UiS_liggende_logo_liten.png)

# Lab 4: Single-decree Paxos

| Lab 4:		| Single-decree Paxos 			|
| -------------------- 	| ------------------------------------- |
| Subject: 		| DAT520 Distributed Systems 		|
| Deadline:		| Thursday Mar 26 2020 18:00		|
| Expected effort:	| 30 hours 				|
| Grading: 		| Pass/fail + Lab Exam 			|
| Submission: 		| Group					|

### Table of Contents

- [Lab 4: Single-decree Paxos](#lab-4-single-decree-paxos)
    - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Resources](#resources)
  - [Questions (10%)](#questions-10)
  - [Prerequisites](#prerequisites)
  - [Algorithm implementation (30%)](#algorithm-implementation-30)
  - [Distributed implementation (50%)](#distributed-implementation-50)
      - [Singlepaxos Package](#singlepaxos-package)
      - [Distributed Paxos Application](#distributed-paxos-application)
  - [Dockerize your application (10%)](#dockerize-your-application-10)
      - [Test Scenario](#test-scenario)
  - [Lab Approval](#lab-approval)

## Introduction

The overall objective of this lab is to implement a single-decree version of
Paxos. The assignment consist of four parts:

1. A set of theory questions that you should answer (10%).

2. Implementation of the core algorithm for each of the three Paxos roles, the
   Proposer, Acceptor and Learner. This implementation has corresponding unit
   tests and will be verified by Autograder (30%).

3. Integration of the Paxos implementation into your application for
   distributed leader detection from Lab 3. The goal is to use the failure and
   leader detection capabilities from your previous implementation to choose a
   single proposer amongst a group a Paxos processes.  This subtask will be
   verified by a member of the teaching staff during lab hours (50%).
   
4. The application should be containerized with Docker. Your container(s) will be verified by a member of the teaching staff during lab hours (10%).

The Paxos implementation for this lab will be a single-decree version. This
variant of Paxos is only expected to choose a single command. It does not need
several slots. However, in the next labs your implementation will be required
to choose multiple commands.

## Resources

Several Paxos resources are listed below. You should use these resources to
answer the [questions](#questions-10) for this lab. You are also advised to use
them as support literature when working on your implementation now and in
future lab assignments.

* [Paxos Explained from Scratch](resources/paxos-scratch-slides.pdf)  - slides.
* [Paxos Explained from Scratch ](resources/paxos-scratch-paper.pdf) - paper.
* [Paxos Made Insanely Simple](resources/paxos-insanely-simple.pdf) - slides. Also
  contains pseudo code for Proposer and Acceptor.
* [Paxos Made Simple](resources/paxos-simple.pdf)
* [Paxos Made Moderately Complex](resources/paxos-made-moderately-complex.pdf)
* [Paxos Made Moderately Complex (ACM Computing Surveys)](resources/a42-renesse.pdf)
* [Paxos for System Builders](resources/paxos-system-builders.pdf)
* [The Part-time Parliament](resources/part-time-parliment.pdf)

## Questions (10%)

Answer the questions below. You should write down and submit your answers by
using the
[answers.md](https://dat520.github.io/r?assignments/blob/master/lab4/answers.md)
file.

1. Is it possible that Paxos enters an infinite loop? Explain.

2. Is the value to agree on included in the `Prepare` message?

3. Does Paxos rely on an increasing proposal/round number in order to work?
   Explain.

4. Look at this description for Phase 1B: If the proposal number N is larger
   than any previous proposal, then each Acceptor promises not to accept
   proposals less than N, and sends the value it last accepted for this
   instance to the Proposer. What is meant by “the value it last accepted”? And
   what is an “instance” in this case?

5. Explain, with an example, what will happen if there are multiple
   proposers.
   
6. What happens if two proposers both believe themselves to be the leader and
   send `Prepare` messages simultaneously?

7. What can we say about system synchrony if there are multiple proposers (or
   leaders)?

8. Can an acceptor accept one value in round 1 and another value in round 2?
   Explain.

9. What must happen for a value to be “chosen”? What is the connection between
   chosen values and learned values?

## Prerequisites

In order for the go imports to work you need to pull the assignments repository 
to a new directory.

```go
# Pull the assignments into the correct location
go get -u dat520.github.io

# Run this command to get new or updated assignments
git pull
```

## Algorithm implementation (30%)

You will in this task implement the single-decree Paxos algorithm for each of
the three Paxos roles. This task will be verified by Autograder.

The skeleton code, unit tests and definitions for this assignment can be found
in the `singlepaxos` package. Each of the three Paxos roles has a separate
file for skeleton code (e.g. `acceptor.go`) and one for unit tests (e.g.
`acceptor_test.go`). There is additionally a single file called `defs.go`. This
file contain struct definitions for the four Paxos messages. You should not
edit this file.

Each of the three Paxos roles have a similar skeleton code structure. They all
have a constructor, e.g. `NewAcceptor(...) *Acceptor`. Each Paxos role also
have a `handle` method for each message type they should receive. For example, the
Acceptor has a method for processing accept messages with the following
signature:

```go
func (a *Acceptor) handleAccept(acc Accept) (lrn Learn, output bool)
```

A `handle` method returns a Paxos message and a boolean value `output`. The
value `output` indicate if handling the message resulted in an output message
from the Paxos module. Example: If an acceptor handles an accept message and
should according to the algorithm reply with a learn message, then the
`handleAccept` would return with `output` set to true and the corresponding
learn message as `lrn`. If handling the accept resulted in no outgoing learn
message, then `output` should be set to false. In other words, the caller
should _always_ check the value of `output` before eventually using the Paxos
message. If `output` is false, then then each field of the Paxos message struct
should have the zero value (e.g. initialized using an empty struct literal,
`Acceptor{}`).

The `handleLearn` method from `learner.go` does not output a Paxos message.
The return value `val` instead represent a value the Paxos nodes had reached
consensus on (i.e. decided). You should in the next
[task](#distributed-implementation-50) not send this value over the network as
a message. It is meant to be used internally on a node to indicate that a value
was chosen. The Value type is defined in `defs.go`:

```go
type Value string

const ZeroValue Value = ""
```

The `Value` definition represents the type of value the Paxos nodes should
agree on. The definition is in this assignment for simplicity's sake a type
alias for the `string` type. It will in later lab assignments represent
something more application specific, e.g. a client request. A constant named
`ZeroValue` is also defined to represent the empty value.

The Paxos message definitions found in `defs.go` uses the naming conventions
found in
[this](https://dat520.github.io/r?assignments/blob/master/lab4/resources/paxos-insanely-simple.pdf)
algorithm specification (slide 64 and 65):

```go
type Prepare struct {
	From int
	Crnd Round
}

type Promise struct {
	To, From int
	Rnd      Round
	Vrnd     Round
	Vval     Value
}

type Accept struct {
	From int
	Rnd  Round
	Val  Value
}

type Learn struct {
	From int
	Rnd  Round
	Val  Value
}
```

Note that _only_ the `Promise` message struct has a `To` field. This is because
the promise only should be sent to the proposer who sent the corresponding
prepare (unicast). The other three messages should all be sent to every other
Paxos node (broadcast).

The `Round` type definition is also found in `defs.go`, and is type alias for
`int`:

```go
type Round int

const NoRound Round = -1
```

There is also an important constant named `NoRound` defined along with the
type alias. The constant should be used in promise messages, and more
specifically for the `Vrnd` field, to indicate that an acceptor has not voted
in any previous round.  The Proposer has in addition to its `handlePromise`
method also one named `increaseCrnd`. Every proposer maintain, as described in
[literature](#resources), at set of unique round numbers to use when issuing
proposals.  A proposer is (for this assignment) defined to have its `crnd`
field initially set to the same value as its id. Note that you need to do a
type conversion to do this assignment in the constructor (`Round(id)`).  The
`increaseCrnd` method should increase the current `crnd` by the total number of
Paxos nodes. This is one way to ensure that every proposer uses a disjoint set
of round numbers for proposals.

You should for this task implement the following (all marked with
`TODO(student)`):

* Any **unexported** field you may need in the `Proposer`, `Acceptor` and
  `Learner` struct.

* The constructor for each of the three Paxos roles: `NewProposer`,
  `NewAcceptor` and `NewLearner`. Note that you do not need to use or assign
  the channels for outgoing messages before the [next
  subtask](#distributed-implementation-50).

* The `increaseCrnd` method in `proposer.go`.

* The `handlePromise` method in `proposer.go`. _Note:_ The Proposer has a field
  named `clientValue` of type `Value`. The Proposer should in its
  `handlePromise` method use the value of this field as the value to be chosen
  in an eventual outgoing accept message if and only if it has not be bounded
  by another value reported in a quorum of promises.

* The `handlePrepare` and `handleAccept` method in `acceptor.go`.

* The `handleLearn` method in `learner.go`.

Each of the three Paxos roles has a `_test.go` file with unit tests. You should
not edit these files. If you wish to write your own test cases, which is
something that we encourage you to do, then do so by creating separate test
files. How to run the complete test suite or an individual test cases has been
thoroughly described in previous lab assignments.

The test cases for each Paxos role is a set of actions, more specifically a
sequence of message inputs to the `handle` methods. The test cases also provide
a description of the actual invariant being tested. You should take a look at
the test code to get an understanding of what is going on. An example of a
failing acceptor test case is shown below:

```go
=== RUN TestHandlePrepareAndAccept
--- FAIL: TestHandlePrepareAndAccept (0.00s)
        acceptor_test.go:17: 
                HandlePrepare
                test nr:1
                description: no previous received prepare -> reply with correct rnd and no vrnd/vval
                action nr: 1
                want Promise{To: 1, From: 0, Rnd: 1, No value reported}
                got no output

```

*Note:* You should not (and don't need to) use any of the incoming or outgoing
channels for this task. If you use the outgoing channels the unit tests will
hang since a send on a nil channel blocks forever. This task is solely the core
message handling for each Paxos role.  You may (and need) to add fields to each
Paxos role struct to maintain Paxos state. You will use the incoming and
outgoing channels in the next task when you complete the total implementation
of each Paxos role.

## Distributed implementation (50%)

This task consist of two parts. You will in the first task complete the
implementation of the three Paxos roles. The second task is to integrate the
Paxos implementation together with your network and failure/leader detector
from Lab 3. You will additionally extend your application to handle clients.

#### Singlepaxos Package

You will need to implement the remaining method stubs to complete each Paxos
role. All Paxos roles has a main message handling loop (as the failure detector
from Lab 3). The loop is started in an separate goroutine using each role's
`Start()` method. Each loop uses a `select` statement to handle any incoming
message or value. You should use the handle methods already implemented to
process each type of message and send any resulting reply message using the
outgoing channels. An example taken from the acceptor is shown below:

```go
func (a *Acceptor) Start() {
	go func() {
		for {
			select {
			case prp := <-a.prepareIn:
				prm, output := a.handlePrepare(prp)
				if output {
					a.promiseOut <- prm
				}
				// ...
			}
		 }
	}()
```

The `Stop` and `Deliver` methods are trivial and used to feed messages into the
main handling loop.  Again, see the failure detector from Lab 3 for an example
of how this is done. The Proposer should additionally subscribe and listen to
leader change messages from the Leader Detector to keep track of the current
leader. The Proposer should handle values from clients (via the
`DeliverClientValue` method) as a separate case in its message handling loop.
It should handle a value according to the following pseudo code:

```
if not leader
	ignore client value
else
	set clientValue field to the incoming value
	increase crnd
	create and send prepare
```

*Summary:* You should in this part implement the following in the
`singlepaxos` package (all marked with `TODO(student)`):

* Any more **unexported** fields you may need in the `Proposer`, `Acceptor` and
  `Learner` struct.

* Any additions to the constructor for each of the three Paxos roles:
  `NewProposer`, `NewAcceptor` and `NewLearner`. Note that you will need to use
  and assign the channels for outgoing messages for this task (similar to what
  you did for the failure detector in Lab 3).

* The `Start` and `Stop` method for each Paxos role.

* All `Deliver` methods for each of the three Paxos roles.

*Proposer progress check:*

The Proposer must also be able to handle the case where after becoming leader,
its `crnd` variable is still lower than the previous leaders `crnd`. This may
be true even after calling its own `increaseCrnd` method on leader change. An
acceptor will silently ignore a prepare (and accept) message for a lower round
than it has already seen in a prepare or accept. The Proposer which consider
itself the leader should therefore use a timer (with an appropriate timeout
value) to check that it actually receives a quorum of promises after
broadcasting a prepare. If the timer fires after sending a prepare, and a
quorum of promises has not received, the Proposer should increase its `crnd`,
send a new prepare message and start the timer again.

#### Distributed Paxos Application

You will in this task integrate the Paxos implementation with your network and
failure/leader detector from Lab 3. The purpose of this task is to use the
leader detector to select a single node as the Proposer for the Paxos
algorithm. You will additionally need extend your application to handle
clients. Clients should be able to connect to the Paxos leader and issue
commands. Note that the single-decree version you have implemented for this lab
only decides on a single value. This means that only one client command can be
chosen (and stored) by your system. You will in the next lab extend your
application to handle multiple commands.

Your application should consist of a least the following parts:

* The failure and leader detector from Lab 3.

* The three Paxos roles (proposer, acceptor and learner) implemented in this
  lab assignment.

* The network layer you started working on in Lab 3. Your network layer must be
  able to handle the new set of Paxos messages in addition to the Heartbeat
  message from Lab 3.

* A new simple client handling module.

The client handling module should listen for and handle client connections.
Clients should connect to _all_ Paxos nodes and send any command to _all_ of
them. The commands should be of type string, thereby matching the `Value` type
used by the Paxos roles. The client handling module should forward any command
to the proposer using the `DeliverClientCommand` method. The Proposer should in
turn propose the command if it consider itself the current leader.  Your
application should listen for decided values from the learner using the
`valueOut` channel. Any decided value should be given to the client handling
module, which in turn should forward it to _every_ connected client.

A rudimentary overview of how the different modules should interact is shown
below. Note that the arrows only represent a general form of interaction.

![Lab 4 Overview](../img/lab4.png)

The current implementation and design has some simplifications. The is
mainly due to the nature of single-decree Paxos (only a single value can be
chosen) and to keep the expected effort and complexity for the lab assignment
manageable. Many of this issues will be handled in Lab 5. The main focus for
this assignment is implement and understand the core Paxos algorithm.

For example, the client handling module can not currently connect a chosen
command from the learner to the actual client who sent the command. We will
extend the `Value` type in Lab 5 to also carry information about which client
sent the command. Any command sent to a non-leader node will now be ignored.
The client module will therefore later be extended to redirect clients to
current Paxos leader.

The current proposer implementation also have some limitations. It may for
example overwrite its `clientValue` field between sending a prepare and accept
message, resulting in that the original client command being lost. You may add
additional logic to handle this case if you want. The proposer does also not
implement a progress check to verify that a prepare or accept actually results
in a quorum of promises or a value being chosen.

## Dockerize your application (10%)
In subsequent labs we will use [containers](https://www.docker.com/resources/what-container) 
to run multiple instances of Paxos nodes. As part of this lab you are expected to complete 
the installation of Docker and containerize your application.

1. [Docker installation](https://docs.docker.com/install/)
2. [Deploying Go servers with Docker](https://blog.golang.org/docker)

```Docker
# This is a template Dockerfile

# Start from a Debian image with the latest version of Go installed
# and a workspace (GOPATH) configured at /go.
FROM golang

# Copy the local package files to the container's workspace.
COPY ./ /go/src/dat520.github.io

# Build your application inside the container.
RUN go install dat520.github.io/lab4/PaxosServer

# Run your application when the container starts
ENTRYPOINT /go/bin/PaxosServer
```
```go
# Go to your group repository. This is neccessary to use the command below
# to build your container.
cd $GOPATH/src/$gpath

# Build your container
docker build -t dat520-lab4 -f lab4/PaxosServer/Dockerfile .

# Create network
docker network create --subnet=192.168.0.0/16 dat520

# Run your container
docker run -itd --name lab4_1 --net dat520 --ip 192.168.1.1 --rm dat520-lab4

# Attach standard input, output and error streams
docker attach lab4_1
``` 

#### Test Scenario

You should for this lab, in addition to explaining your code and design,
present a working demo of your application. You will at least be asked to
demonstrate the following scenario:

1. Start a cluster consisting of three Paxos nodes. Start two clients.

2. Send a command (string) to the system from a single client. The value of
   this command should be decided and all clients should receive the value as a
   response.

3. Send another different client command to the system. The system should reply
   with the value _from the previous command_.

4. Stop the leader Paxos node and let the system choose a new leader. Let the
   clients connect to the new leader and send yet another command. The system
   should again reply with the value of the command from Step 1.

5. Stop one of the two remaining Paxos nodes. Send another client command. The
   client should not receive any reply to this command.

## Lab Approval

To have your lab assignment approved, you must come to the lab during lab hours
and present your solution. This lets you present the thought process behind
your solution, and gives us more information for grading purposes and we may
also provide feedback on your solution then and there. When you are ready to
show your solution, reach out to a member of the teaching staff. It is
expected that you can explain your code and show how it works. You may show
your solution on a lab workstation or your own computer.

**At least 60% of this lab must be passed and all subtasks must be attempted for the lab to be approved.** The results from
Autograder will be taken into consideration when approving a lab.

You should for this lab present a working demo of the application described in
the previous [section](#distributed-implementation-50). The application should
run on three machines or container instances in the lab (not enough to run just on localhost). You should demonstrate that your implementation fulfills the previously listed
specification. The task will be verified by a member of the teaching staff
during lab hours.

Also see the [Grading and Collaboration
Policy](https://dat520.github.io/r?course-info/blob/master/policy.md) document for
additional information.
