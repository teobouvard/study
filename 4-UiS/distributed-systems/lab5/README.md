![UiS](https://www.uis.no/getfile.php/13391907/Biblioteket/Logo%20og%20veiledninger/UiS_liggende_logo_liten.png)

# Lab 5: Multi-Paxos

| Lab 5:               | Multi-Paxos                   |
| -------------------- | ----------------------------- |
| Subject:             | DAT520 Distributed Systems    |
| Deadline:            | Thursday April 23 2020 12:00  |
| Expected effort:     | 30-40 hours                   |
| Grading:             | Pass/fail                     |
| Submission:          | Group                         |

## Table of Contents

1. [Introduction](#introduction)
2. [Background and Resources](#background-and-resources)
3. [Prerequisites](#prerequisites)
4. [Algorithm Implementation](#algorithm-implementation-30)
5. [Distributed Implementation](#distributed-implementation-50)
6. [Dockerize your application](#dockerize-your-application-20)
7. [Lab Approval](#lab-approval)
8. [Appendix - Proposer Test Cases](#appendix---proposer-test-cases)

## Introduction

The overall objective of this lab is to implement a multi-decree version of
Paxos (Multi-Paxos). The assignment consist of two parts:

1. Implementation of the core algorithm for each of the three Paxos roles, the
   Proposer, Acceptor and Learner. This implementation has corresponding unit
   tests and will be verified by Autograder (30%).

2. Extension of you distributed application from Lab 3 and 4. The goal is to
   reuse large parts of the application (e.g. failure and leader detector), but
   replace the `singlepaxos` modules from Lab 4 with the `multipaxos` modules
   implemented in this lab assignment. You will use your implementation to
   replicate a set of bank accounts. You will also be required to extend your
   application's client handling abilities. This subtask will be verified by a
   member of the teaching staff during lab hours (50%).

3. The application should be containerized with Docker. Your container(s) will
   be verified by a member of the teaching staff during lab hours (20%).

## Background and Resources

Practical systems normally use Paxos as a building block to achieve consensus
on a sequence of values. One way to achieve this would be to run a full
instance of single-decree Paxos, both Phase 1 and 2, for every value. This
would require four message delays for every value to be decided. With
Multi-Paxos it is possible to reduce this overhead. Multi-Paxos only perform
Phase 1 once on every leader change. A Proposer, thinking it is the new leader,
issue a prepare for every slot higher than the highest consecutive decided slot
it has seen. Every other Paxos acceptor respond to with a promise if the round
is higher than their current one. The promise message may contain a set of
(_vrnd, vval_) tuples for every slot higher or equal to the one from the
prepare message if the acceptor has accepted any value for these slots. On
receiving a quorum of promises, the Proposer is bounded by the highest
(_vrnd, vval_) tuple reported for any slot higher than the slot from the corresponding
prepare message. The Proposer can then perform Phase 2 (accept and learn) for
every value to be decided. Only two message delays are required to get a value
accepted.  You are _strongly_ advised to read Section 3, _Implementing a State
Machine_, from _Paxos Made Simple_ by Leslie Lamport for a more complete
description of the Multi-Paxos optimization.  You may also find the other
[resources](../lab4/resources/) listed for Lab 4 useful.

## Prerequisites

(_Note that, you should not set the `GOPATH` variable any more. With the new Go modules system, that is no longer necessary. If you have used `GOPATH` so far, I recommend you clone your repo again in a new folder, e.g. `$HOME/dat520`, and run VSCode from that folder without setting the `GOPATH`. PS: Make sure your local changes have been pushed to github first. TODO(meling): remove this comment next time, when lab1-4 is updated accordingly._)

(_Note, the following instructions, explains the new recommended way of working with repositories for Autograder. That is, instead of cloning the course's `assignments` repo, you should instead clone your own group repository, and make a remote pointing to the course's `assignments` repo. This can be done as follows, if not already done:_)

```sh
cd $HOME/dat520

# Copy the repo link from the GitHub page of your repository
git clone https://github.com/dat520-2020/<your group name>

# Set the remote 'assignments' to point to the course's assignments repository
git remote add assignments https://github.com/dat520-2020/assignments.git
```

(_TODO: The following instructions assumes that the above steps has been performed in earlier labs. The instructions above will be removed in the future, when lab1-4 has been updated accoringly._)

To get the skeleton code for this assignment, navigate to your project's
root folder, e.g. `$HOME/dat520`, and run:

```sh
# Enter the assignment directory
cd $HOME/dat520

# Run this command to get new or updated assignments from the course's assignments repository
git pull assignments master

# Run this command to get the latest updates from your shared group repository
git pull

# Run this command to share your code changes with your group
git push
```

## Algorithm implementation (30%)

You will in this task implement the Multi-Paxos algorithm for each of the three
Paxos roles. This task will be verified by Autograder. The task is similar to
what you did for single-decree Paxos in Lab 4, but is more complex since
Multi-Paxos is able to choose multiple commands. Both Phase 1 and Phase 2 of
the Paxos protocol (as described in _Paxos Made Simple_) needs to be adjusted.
Especially the prepare-promise exchange need changes.

The skeleton code, unit tests and definitions for this assignment can be found
in the `mulitpaxos` package. Each of the three Paxos roles has a separate file
for skeleton code (e.g. `acceptor.go`) and one for unit tests (e.g.
`acceptor_test.go`). There is additionally a single file called `defs.go`. This
file contain struct definitions for the four Paxos messages and other related
definitions. You should not edit this file.

Each of the three Paxos roles has a similar skeleton code structure. They all
have a constructor, e.g. `NewAcceptor(...) *Acceptor`. Each Paxos role also
have a `handle` method for each message type they should receive. For example,
the Acceptor has a method for processing accept messages with the following
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
message. If `output` is false, then each field of the Paxos message struct
should have the zero value (e.g. initialized using an empty struct literal,
`Acceptor{}`).

The `handleLearn` method from `learner.go` has the following signature:

```go
func (l *Learner) handleLearn(lrn Learn) (val Value, sid SlotID, output bool)
```

The method does not output a Paxos message. The return value `val` instead
represent a value for a specific slot (`sid`) that the Paxos nodes has reached
consensus on (i.e. decided). A difference from the single-decree Paxos learner
from Lab 4 is the `sid SlotID` return value. The return value indicates what
slot the corresponding decided value (`val`) belongs to. The `Value` type
definition (in `defs.go`) has also changed from being a type alias for `string`
in Lab 4 to the following struct definition for Lab 5:

```go
type Value struct {
	ClientID   string
	ClientSeq  int
	Noop       bool
	AccountNum int
	Txn        bank.Transaction
}
```

The `Value` type now also carries information about the client that sent the
command. `ClientID` is a unique client identifier. `ClientSeq` represents a
client sequence number. The number should be used by clients to match a
response received from a Paxos system to a corresponding request. The actual
value is now a combination of the `AccountNum` field and `Txn` field of type
`Transaction` from the `bank` package. The `bank` package will be introduced in
the next [part](#distributed-implementation-50) of this lab assignment.

The `Round` type definition found in `defs.go` has *not* changed from Lab 4:

```go
type Round int

const NoRound Round = -1
```

The Paxos messages has also changed slightly from Lab 4:

```go
type Prepare struct {
	From int
	Slot SlotID
	Crnd Round
}

type Promise struct {
	To, From int
	Rnd      Round
	Slots    []PromiseSlot
}

type Accept struct {
	From int
	Slot SlotID
	Rnd  Round
	Val  Value
}

type Learn struct {
	From int
	Slot SlotID
	Rnd  Round
	Val  Value
}
```

The `Prepare`, `Accept` and `Learn` messages have all gotten a `Slot` field of
type `SlotID`. This means that every `Accept` and `Learn` message now relates
to a specific slot. The `Slot` field in the `Prepare` message has a somewhat
different meaning.  In Multi-Paxos, as explained
[previously](#background-and-resources), a proposer only executes Phase 1 once
on every leader change if it considers itself the leader.  The `Slot` field in
the `Prepare` message represents the slot after the highest consecutive decided
slot the Proposer has seen. This slot identifier is used by an acceptor to
construct a corresponding `Promise` as a reply. An acceptor attach information
(`vrnd` and `vval`) for every slot it has sent an accept for equal or higher to
the one received in the prepare message. This information is stored in the
slice `Slots` of type `PromiseSlot`. The slice should be sorted by increasing
`SlotID`. The `PromiseSlot` struct is defined in `defs.go`:

```go
type PromiseSlot struct {
	ID   SlotID
	Vrnd Round
	Vval Value
}
```

To create and append the correct slots (if any) to the slice, an acceptor need
to keep track of the highest seen slot it has sent an accept for. This can for
example be done by maintaining a `maxSlot` variable of type `SlotID`. The
Proposer is bounded by the highest `PromiseSlot` (highest `vrnd`) reported in a
quorum for any slot higher than the slot from a corresponding prepare message.

The `handlePromise` method from `proposer.go` has the following signature:

```go
func (p *Proposer) handlePromise(prm Promise) (accs []Accept, output bool)
```

Specification:

* *Input:* A single `Promise` message.

* The Proposer should ignore a promise message if the promise has a round
  different from the Proposer's current round, i.e. it is not an answer to a
  previously sent prepare from the Proposer.

* The Proposer should ignore a promise message if it has previously received a
  promise from the _same_ node for the _same_ round.

* *Output:* If handling the input promise result in a quorum for the current
  round, then `accs` should contain a slice of accept message for the slots the
  Proposer is bound in. If the Proposer is not bounded in any slot the `accs`
  should be an empty slice. If `output` is false then `accs` should be `nil`.

* All accept messages in the `accs` slice should be in increasing consecutive
  slot order.

* If there is a gap in the set of slots the Proposer is bounded by, e.g. it is
  bounded in Slot 2 and 4 but not 3, then the Proposer should create an accept
  message with a no-op value for the accept for Slot 3 in the `accs` slice.

* If a `PromiseSlot` in a promise message is for a slot lower than the
  Proposer's current `adu` (all-decided-up-to), then the `PromiseSlot` should
  be ignored.

You can find a complete description of proposer test case number 7-11
[here](#appendix---proposer-test-cases).

A few other important aspects of the Paxos roles are listed below:

* A `Promise` message does, for the sake of simplicity, not indicate if a slot
  (`PromiseSlot`) above the slot identifier from the `Prepare` is decided or
  not (the new leader may not have previously learnt it). In this case a slot
  will be proposed and decided again. The Paxos protocol ensure that the same
  value will be decided for the same slot.

* An acceptor only need to maintain a single `rnd` variable (as for
  single-decree Paxos). The `rnd` variable spans across all slots. Only `vrnd`
  and `vval` must be stored for each specific slot. Similarly, the Proposer
  only need to maintain a single `crnd` variable.

* The Paxos roles share no slot history/storage in this implementation. Each
  role should maintain their own necessary variables and data structures for
  keeping track of promises, accepts and learns for each slot.

Summarized, you should for this task implement the following (all marked with
`TODO(student)`):

* Any **unexported** field you may need in the `Proposer`, `Acceptor` and
  `Learner` struct.

* The constructor for each of the two Paxos roles: `NewAcceptor` and
  `NewLearner`. Note that you do not need to use or assign the channels for
  outgoing messages before the [next subtask](#distributed-implementation-50).
  The `NewProposer` constructor is already implemented. Note that the Proposer
  also take its `adu` as an argument due to testing purposes.

* The `handlePrepare` and `handleAccept` method in `acceptor.go`.

* The `handleLearn` method in `learner.go`.

* The `handlePromise` method in `proposer.go`.

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
                action nr: 2
                action description: prepare slot 1, crnd 1 -> no output, ignore due to lower crnd
                want no output
                got Promise{To: -1, From: -1, Rnd: 0, No values reported}

```

*Note:* You should not (and don't need to) use any of the incoming or outgoing
channels for this task. If you use the outgoing channels the unit tests will
hang since a send on a nil channel blocks forever. This task is solely the core
message handling for each Paxos role. You may need to add fields to each Paxos
role struct to maintain Paxos state. You will use the incoming and outgoing
channels in the next task when you complete the total implementation of each
Paxos role.

## Distributed implementation (50%)

This task consist of three main parts. In the first task you should complete
the implementation of the three Multi-Paxos roles. The second task is to
replace the single-decree Paxos modules from Lab 4 with the Multi-Paxos modules
implemented in this lab assignment. You will also need to extend your main
application to store a set of replicated bank accounts and apply transactions
to them as they are decided by your Paxos nodes. The third task involves
creating a stand-alone client application and a corresponding client handling
module for the Paxos nodes.

### Multi-Paxos Package

You will need to implement the remaining method stubs to complete each Paxos
role. All Paxos roles has a main message handling loop (as in Lab 4). The loop
is started in an separate goroutine using each role's `Start()` method. Each
loop uses a `select` statement to handle any incoming message or value. You
should use the handle methods already implemented to process each type of
message and send any resulting reply message using the outgoing channels. An
example taken from the acceptor is shown below:

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

The Learner returns both a value and a corresponding slot id from its
`handleLearn` method. It needs to construct a `DecidedValue` (see `defs.go`)
using the value and slot id before sending it on its `decidedOut` channel.

The Proposer is quite a bit more complex for Multi-Paxos compared to
single-decree Paxos from Lab 4. An initial implementation for the Proposer can
be found in `proposer.go`. Only the `handlePromise` from the previous sub task
is not implemented. The just a proposal (no pun intended) for how the base
structure of the Proposer may look. The proposal also documents, using code,
its expected behavior. _You should, and most probably must, tweak and adjust
the code_. It has not been thoroughly tested.

The Proposer need to keep track of the highest decided _consecutive_ slot the
Paxos node has seen (all decided up to). The Proposer has a
`IncrementAllDecidedUpTo` method that should be used to indicate that a slot
has been reported decided by the Learner. An outside caller _must call this
method when slots are decided in consequtive order_ (and wait if any gaps).
The Proposer uses this information for two purposes:

* To attach the slot id for the highest decided slot to any prepare message
  it sends.

* To know if a previously sent accept message has been decided so that it can
  send a new one. _A Proposer is defined for this lab to only be allowed to
  send an accept for the next slot if the previous slot has been decided_. How
  many slots a proposer may send accept messages for concurrently can be
  generalized with an `alpha` parameter (pipelining). For this lab `alpha`
  consequently equals 1. Also see the Proposer's `sendAccept` method.

Proposer specification (see `select` on line 96):

* **Receive promise message, line number 97:** Handle promise message. Store
  resulting accept messages (if any) if handling the promise result in a
  quorum. Mark Phase One as done. Attempt to send accept message.

* **Receive client request, line number 108:** Ignore request if not the
  leader. If leader, then store command in queue. Continue message handling
  loop if Phase One is not done. Attempt to send accept if Phase One is done.

* **Receive increment adu notification, line number 117:** Increment internal
  `adu`. Continue message handling loop if not leader, or, if leader but Phase
  One is not done. Attempt to send accept if leader and Phase One is done.

* **Phase 1 progress check, line number 125:** The Proposer must be able to
  handle the case where after becoming leader, its `crnd` variable is still
  lower than the previous leaders `crnd`. This may be true even after calling
  its own `increaseCrnd` method on leader change. An acceptor will silently
  ignore a prepare (and accept) message for a lower round than it has already
  seen in a prepare or accept. The Proposer which consider itself the leader
  should therefore use a timer (with an appropriate timeout value) to check
  that it actually receives a quorum of promises after broadcasting a prepare.
  If the timer fires after sending a prepare, and a quorum of promises has not
  received, the Proposer should increase its `crnd`, send a new prepare message
  and start the timer again.

* **Receive trust message, line number 129:**  Store new leader and start Phase
  One if the trust message is for the Proposer's id.

* **Phase 2 progress check, line number n/a:** This is not implemented in the
  example code. A Proposer should in theory also verify that the accept
  messages it sends are eventually decided (if it still consider itself the
  leader). A proposer might get stuck waiting for a slot to be decided if an
  accept message is lost (for example when using UDP). You may _optionally_
  implement such a progress check if you want.

A proposer should send an accept if it considers itself the leader and the
previous slot has been decided. It should first check if there is any remaining
accept messages in its accept message queue. The queue contain any accept
messages generated as part of Phase One. The Proposer should send the
appropriate accept if the queue is non-empty. The Proposer should check its
client request queue if the accept queue is empty. The Proposer should generate
a Accept message using a client request if available. The Proposer must wait
for a client request before sending an accept if the queue is empty. See also
the code and comments for the `sendAccept` method.

*Summary:* You should in this part implement the following in the
`multipaxos` package (all marked with `TODO(student)`):

* Any more **unexported** fields you may need in the `Proposer`, `Acceptor` and
  `Learner` struct.

* Any additions to the constructor for each of the three Paxos roles:
  `NewProposer`, `NewAcceptor` and `NewLearner`. You should not change the
  signature of the constructor functions.  Note that you will need to use and
  assign the channels for outgoing messages for this task (similar to what you
  did for Lab 4).

* The `Start` and `Stop` method the Acceptor and Learner.

* All `Deliver` methods for the Acceptor and Leaner.

* Any internal changes to the Proposer you may want.

#### Distributed Multi-Paxos Application

Your system should in this assignment use the Paxos protocol to replicate a set
of bank accounts. Clients can issue transactions to the accounts. Transactions
sent by clients should be sequenced in the same order by all nodes using the
Multi-Paxos protocol you have implemented.

Your application should at least consist of the following parts:

* Your network layer from the previous assignments.

* The Failure and Leader detector from Lab 3.

* The three Multi-Paxos roles (proposer, acceptor and learner) implemented in
  this lab assignment.

* A central module that coordinates the different parts of your application .
  This may for example just be your application's `main()` function.  This
  module is named `Server` in the figure below, and this name will be used in
  the rest of the assignment text.

* A client handling module for interacting with clients.

A rudimentary overview of how the different modules should interact is shown
below. Note that the arrows only represent a general form of interaction.

![Lab 5 Overview](../img/lab5.png)

Your network layer from the previous assignments should be extended to handle
the overall requirements of this lab. The Failure and Leader detector should
not need any changes.

#### Server Module

A server module is needed to coordinate the different parts of your
application. It should at least perform the following tasks:

* Initialize and start all the relevant parts your application.
* Multiplex and demultiplex message to and from the network layer.
* Store bank accounts.
* Keep track of the id of highest decided slot.
* Apply bank transactions in correct order as they are decided.
* Notify the Proposer when a new slot has been decided.
* Route client requests from the client handling module to the Proposer (if the
  client handling module does not interact directly with the Proposer).

All bank related code can be found in the file `bank.go` from the `bank`
package. Please take a look at the comments in `bank.go` for the exact
semantics. Three structs are defined in the package: `Transaction`,
`TransactionResult` and `Account`. A `Transaction` is sent by clients together
with an account number in the `Value` struct described earlier. The
following method is defined for an `Account`:

```go
func (a *Account) Process(txn Transaction) TransactionResult
```

The method should be used by a Paxos node when it applies a decided transaction to
an account. The resulting `TransactionResult` should be used by a node in the
reply to the client. In Lab 4 the reply to a client was simply the value it
originally sent.  For this lab the reply is the result of applying a client's
transaction. A `Response` has therefore been defined in `defs.go` from
the `multipaxos` package:

```go
type Response struct {
	ClientID  string
	ClientSeq int
	TxnRes   bank.TransactionResult
}
```

A node should generate a response after applying a transaction to an account.
The `ClientID`, `ClientSeq` and `AccountNum` fields should be populated from
the corresponding decided value. A response should be forwarded to the client
handling part of your application. It should there be sent to the appropriate
client if it is connected. Client handling details are described in the next
subsection.

A learner may, as specified previously, deliver decided values out of order.
The server module therefore needs to ensure that only decided values
(transactions) are processed in order. The server needs to keep track of the id
for the highest decided slot to accomplish this. This assignment text will use
the name `adu` (all decided up to) when referring to this variable. It should
initially be set to `-1`. The server should buffer out-of-order decided values
and only apply them when it has consecutive sequence of decided slots. More
specifically the server module should handle decided values from the learner
equivalently to the logic in the following pseudo code:

```go
on receive decided value v from learner:
	handleDecideValue(v)
```

```go
handleDecidedValue(value):
	if slot id for value is larger than adu+1:
		buffer value
		return
	if value is not a no-op:
		if account for account number in value is not found:
			create and store new account with a balance of zero
		apply transaction from value to account
		create response with appropriate transaction result, client id and client seq
		forward response to client handling module
	increment adu by 1
	increment decided slot for proposer
	if has previously buffered value for (adu+1):
		handleDecidedValue(value from slot adu+1)
```

_Note_: The server should not apply a transaction if the decided value has its
`Noop` field set to true. It should for a no-op only increment its `adu` and
call the `IncrementAllDecidedUpTo()` method on the Propposer.

#### Client Handling Module

Each Paxos node should have a module for receiving client requests and sending
client responses. Specification:

* The client handling module should accept connections from clients. A client
  should exchange their identifier with the Paxos node so that a node can use
  the id to store and lookup the connection. See also the description of the
  client application in the next subsection.

* A client should be redirected if it connect _or_ send a request to a
  non-leader node. A node should respond with a `<REDIRECT, addr>` message that
  indicate which node it consider as the current leader.

* A client response, as described in the previous subsection, should be of type
  `Response` found in `defs.go` from the `multipaxos` package.

* Upon receiving a `Response` value from the server module, the client handling
  module should attempt to send the response to the client if it is connected.
  It can ignore any errors resulting from trying to send the response to the
  client.

#### Client Application

You should for this lab create a stand-alone client application to interact
with your Paxos system. The client application should be able to operate in two
modes: _manual_ and _benchmark_ mode. Specification:

* Every client should generate a unique identifier for itself. A client may for
  example use its IP-address or generate an identifier randomly. This
  identifier should be exchanged when connecting to a Paxos node and be used
  for the `ClientID` field in every request (`Value`) sent.

* A client should always send requests synchronously, i.e. wait for a response
  to the previous request before sending a new one.

* A client should maintain a internal sequence number, i.e. a client-local
  unique command identifier. It should be used to distinguish between different
  user commands. The sequence number is included in responses from the Paxos
  nodes and lets a client match responses to sent commands.

* A client request is defined to be an instance of the `Value` struct from
  `defs.go`. A client should populate the `ClientID` and `ClientReq` fields
  accordingly. A client should not set the `Noop` field. The `AccountNum` and
  `Txn` fields should be set based on user input if the client application is
  operating in manual mode or generated at random if the application is
  operating in benchmark mode.

* In _manual_ mode: The client application should have a
  command-line interface. It should be possible to send a bank transaction to
  the system by asking the user for the necessary input (account number,
  transaction type and amount).
  
* In _benchmark_ mode: The application should send a
  specified (e.g. using a flag) number of randomly generated requests. The
  random request generation should use every transaction type, account numbers
  in the range `[0,1000]` and amounts in the range `[0,10000]`. It should after
  completion print the following client request round-trip time (RTT)
  statistics:
  * Mean
  * Minimum
  * Maximum
  * Median
  * Standard deviation
  * 99th percentile (_optional_)

* A client should attempt to reconnect to a different node for the following
  cases:
  * When receiving a `<REDIRECT, addr>` message as a response to a
    connection attempt.
  * Timeout on `Dial`, `Read` and `Write` (e.g. see `SetDeadline`
    methods).
  * Connection close (`io.EOF` error for TCP).
  * Any other dial, read, or write error.

#### Handling TCP Connections

*Note:* Some groups have in the previous lab assignments created a new TCP
connection when sending a message over the network. You should reuse
connections if your implementation uses _TCP_ for network communication.
Implementations that establish a new TCP connection for every message will get
a reduced score. The same also applies to client connections.

## Dockerize your application (20%)

You should dockerize your application in the same way as in [lab 4](https://dat520.github.io/r/?assignments/tree/master/lab4#dockerize-your-application-10). Remember to replace _PaxosServer_ with the name of the directory where your application is located. If your application requires command-line arguments, they can be added at the end of the `docker run` command.

_TODO: The following has been updated to use Go modules instead of GOPATH, but hasn't been tested. Please report any issues._

```Docker
# This is a template Dockerfile

# Start from a Debian image with the latest version of Go installed
# and a workspace configured at /go.
FROM golang

# Copy the local package files to the container's workspace.
COPY ./ /go/dat520

# Build your application inside the container.
RUN go install dat520/lab5/PaxosServer

# Run your application when the container starts
ENTRYPOINT ["/go/bin/PaxosServer"]
```

```sh
# Navigate to your local assignments folder. This is neccessary to use the command below
# to build your container.
cd $HOME/dat520

# Build your container
docker build -t dat520-lab5 -f lab5/PaxosServer/Dockerfile .

# Create network
docker network create --subnet=192.168.0.0/16 dat520

# Run your container
docker run -itd --name lab5_1 --net dat520 --ip 192.168.1.1 --rm dat520-lab5

# Attach standard input, output and error streams
docker attach lab5_1
```

### Test Scenarios

You should for this lab, in addition to explaining your code and design,
present a working demo of your application. You will be asked to demonstrate
two different scenarios described below. A command refer below to a
`bank.Transaction`. The transactions should be randomly generated when
the client is operating in benchmark mode.

#### Manual test

Demonstrate the basic capabilities of your system by using a client in manual
mode.

1. Send a few transactions from two different clients.

2. Demonstrate that a client is redirected if it contacts a non-leader node.

3. Demonstrate that a client connects to a different node if it does not receive
   a response to a request after some duration or if the connection to the
   current leader has gone down.

#### Benchmark test

Demonstrate the robustness your system by using several clients in benchmark
mode.

1. Create a single client that sends 500 consecutive (unique) commands to the
   leader. The client should wait for a reply before sending a new command.
   What is the average request round-trip time (RTT) from the client's point
   view? Also calculate the minimum, maximum, median and standard deviation.

2. Run again a single client that sends 500 consecutive commands to the leader.
   You should crash the leader while the client is running, resulting in a
   leader change. The client should afterwards ensure that no command got
   chosen multiple times. How long is the delay due to failover from the
   client's point of view? What part of this delay is due to timeouts?

3. Run three clients concurrently on separate machines. Every client should
   behave as described in 1 above. Record the request RTT statistics at each
   client individually. How does the results compare to the single client run?
   You should check that all replicas have decided on exactly 1500
   (500x3) slots and that no command has been decided twice.

## Lab Approval

To have your lab assignment approved, you must come to the lab during lab hours
and present your solution. This lets you present the thought process behind
your solution, and gives us more information for grading purposes and we may
also provide feedback on your solution then and there. When you are ready to
show your solution, reach out to a member of the teaching staff. It is expected
that you can explain your code and show how it works. You may show your
solution on a lab workstation or your own computer.

**60% of this lab must be passed for the lab to be approved.** The results from
Autograder will be taken into consideration when approving a lab.

You should for this lab present a working demo of the application described in
the previous [section](#distributed-implementation). The application should run
on three machines in the lab (not enough to run just on localhost). You should
demonstrate that your implementation fulfills the previously listed
specification. The task will be verified by a member of the teaching staff
during lab hours.

Also see the [Grading and Collaboration
Policy](https://dat520.github.io/r/?course-info/blob/master/policy.md) document for
additional information.

## Appendix - Proposer Test Cases

![Test 7](../img/test7.png)

![Test 8](../img/test8.png)

![Test 9](../img/test9.png)

![Test 10](../img/test10.png)

![Test 11](../img/test11.png)
