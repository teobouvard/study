![UiS](http://www.ux.uis.no/~telea/uis-logo-en.png)

# Lab 3: Failure Detector and Leader Election

| Lab 3:		| Failure Detector and Leader Election	|
| -------------------- 	| ------------------------------------- |
| Subject: 		| DAT520 Distributed Systems 		|
| Deadline:		| Thursday Feb 27 2020 18:00		|
| Expected effort:	| 20 hours 				|
| Grading: 		| Pass/fail + Lab Exam 			|
| Submission: 		| Group					|

### Table of Contents

- [Lab 3: Failure Detector and Leader Election](#lab-3-failure-detector-and-leader-election)
    - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [Failure Detector (25%)](#failure-detector-25)
    - [Specification](#specification)
  - [Leader Detector (25%)](#leader-detector-25)
  - [Distributed Leader Detection (50%)](#distributed-leader-detection-50)
    - [Specification](#specification-1)
  - [Lab Approval](#lab-approval)

## Introduction

The main objective of this lab assignment is to implement a Failure Detector
and a Leader Detector module. A failure detector provides information about
which processes have crashed and which are correct. A leader detector can use
this information to identify a process that has *not* failed, which may act as
a *leader*, and coordinates certain steps of a distributed algorithm.  You will
later use the leader detector that you implement in this lab assignment for
exactly this purpose. Specifically, the leader detector module will be used to
elect a single node as the Proposer of the Paxos algorithm.  Recall that Paxos
needs this input before triggering the Phase 1 exchange.

This lab consist of three parts. Each part will be explained in more detail in
their own sections.

1. **Failure Detector module:** Implement the eventually perfect failure
   detector algorithm from the textbook. Use the provided skeleton code and
   unit tests. This implementation will be verified by Autograder. The task
   represents 25% of the total lab. 

2. **Leader Detector module:** Implement the eventual leader detector algorithm
   from the textbook. Use the provided skeleton code and unit tests. This
   implementation will also be verified by Autograder. The task represents 25%
   of the total lab. 

3. **Distributed Leader Detection:** Implement a distributed application using
   the two modules described above. The application should elect a leader among
   a group of processes (running on at least three machines), and output the
   current leader in response to crashes. There are no tests for this
   assignment. Instead it will be verified by a member of the teaching staff
   during lab hours. The task represents 50% of the total lab.


## Prerequisites

You need to register your group on [Autograder](https://ag.itest.run/)
before you begin this assignment as it constitutes a group project. This can be
done on your [course page](https://ag.itest.run/app/student/courses/3/members) in
the tab "Members". Select the students you are collaborating with and submit the
group selection. **Only one group member should do this.** Please don't create
a group unless you have agreed with the other member(s) up front.

If you don't have a group partner yet, you may take a look at the [Partner
match-up](https://github.com/dat520-2020/course-info/blob/master/group-partner-hunt.md)
page. If you see a person listed there that you wish to work with, please
connect with him/her directly and agree to submit a group composition
accordingly following the above instructions.

Otherwise, we will assign you to a group. **But you will still need to indicate
your desire to join a group. You do this by clicking the "Random Group"
button.**

**Important 1:** One group consist of two or three students. We only allow at
most three students to collaborate on the group project, but only if there is a
valid reason for this. Four members will not be allowed.

**Important 2:** Your group will only be approved when all members have passed
both lab assignment 1 and 2.

For the group project you should create a copy of the assignment repository. 
Each group will get access to a shared group repository when your group has 
been approved. This will be named `group#`, where `#` is replaced with your 
group number. You will receive an email notification when
Autograder creates a new team on GitHub. Refer to the procedure described in
[lab 1](https://github.com/dat520-2020/assignments/tree/master/lab1#go-assignments) for
instructions on how to setup this repository on your local machine.  Follow
steps 1 to 7 in this procedure, but use the new name `group#`
instead of `username-labs`.

A short list of relevant commands are provided below:

```shell
# Create environment variables used for subsequent labs
# Remember to add these to your ~/.profile or ~/.bashrc file
export dat520="github.com/dat520-2020"
export gpath=$dat520/"<groupname>" // replace <groupname> with your group name on Autograder

git clone https://$dat520/assignments.git $GOPATH/src/$gpath
cd $GOPATH/src/$gpath
git remote add labs https://$gpath
git pull labs master

# Run this command to get new or updated assignments
git pull origin master
# Run this command to get the latest updates from your shared group repository
git pull labs master
# Run this command to share your code changes with your group
git push labs
```

## Failure Detector (25%)

A Failure Detector can be implemented either using a 

1. request/reply approach or a 
2. lease-based approach. 

A Failure Detector that follows the request/reply approach is shown in
Algorithm 2.7 in the textbook. It sends a `HeartbeatRequest` to all other
nodes, and if the request is not answered with a `HeartbeatReply` within a
certain time, it suspects the silent process. A Failure Detector using the
lease-based approach is divided into a sending and receiving process. The
receiving process is essentially the same as the Failure Detector described in
Algorithm 2.7, but with the crucial difference that it does not send
`HeartbeatRequest` messages. Instead, the sending process is a simple loop,
that upon timeout sends a `HeartbeatReply` to all other processes.  Thus, this
Failure Detector uses two timeouts, one for receiving and one for sending
heartbeat messages.

### Specification

In this task you will implement an Eventually Perfect Failure Detector. The
specification for this failure detector is described on pages 53-56 in the
textbook. Your failure detector should use the Increasing Timeout algorithm.
See Algorithm 2.7 for more details.

You should use the provided skeleton to implement the failure detector. All
skeleton code and corresponding tests for this assignment can be found in the
`detector` package. The skeleton code for the failure detector is located in
the file `fd.go`, and is listed below. Large parts of the failure detector is
already implemented, but you will need to complete important remaining parts.
You should complete the implementation by extending the parts of the code
marked with the `TODO(student)` label. The failure detector specification is
documented using code comments. You should refer to these comments for a
detailed specification of the implementation.

The unit tests for the failure detector is located in the file `fd_test.go`.
You can run all the tests in the detector package using the command `go test
-v`. You can, as described in previous labs, use the `-run` flag to only run a
specific test. You are also encouraged to take a close look at the test code to
see what is actually being tested. This may help you when writing and debugging
your code.   

The initial skeleton code for the failure detector in `fd.go` is listed below:

```go
package detector

import "time"

// EvtFailureDetector represents a Eventually Perfect Failure Detector as
// described at page 53 in:
// Christian Cachin, Rachid Guerraoui, and Luís Rodrigues: "Introduction to
// Reliable and Secure Distributed Programming" Springer, 2nd edition, 2011.
type EvtFailureDetector struct {
	id        int          // this node's id
	nodeIDs   []int        // node ids for every node in cluster
	alive     map[int]bool // map of node ids considered alive
	suspected map[int]bool // map of node ids  considered suspected

	sr SuspectRestorer // Provided SuspectRestorer implementation

	delay         time.Duration // the current delay for the timeout procedure
	delta         time.Duration // the delta value to be used when increasing delay
	timeoutSignal *time.Ticker  // the timeout procedure ticker

	hbSend chan<- Heartbeat // channel for sending outgoing heartbeat messages
	hbIn   chan Heartbeat   // channel for receiving incoming heartbeat messages
	stop   chan struct{}    // channel for signaling a stop request to the main run loop

	testingHook func() // DO NOT REMOVE THIS LINE. A no-op when not testing.
}

// NewEvtFailureDetector returns a new Eventual Failure Detector. It takes the
// following arguments:
//
// id: The id of the node running this instance of the failure detector.
//
// nodeIDs: A list of ids for every node in the cluster (including the node
// running this instance of the failure detector).
//
// ld: A leader detector implementing the SuspectRestorer interface.
//
// delta: The initial value for the timeout interval. Also the value to be used
// when increasing delay.
//
// hbSend: A send only channel used to send heartbeats to other nodes.
func NewEvtFailureDetector(id int, nodeIDs []int, sr SuspectRestorer, delta time.Duration, hbSend chan<- Heartbeat) *EvtFailureDetector {
	suspected := make(map[int]bool)
	alive := make(map[int]bool)

	// TODO(student): perform any initialization necessary

	return &EvtFailureDetector{
		id:        id,
		nodeIDs:   nodeIDs,
		alive:     alive,
		suspected: suspected,

		sr: sr,

		delay: delta,
		delta: delta,

		hbSend: hbSend,
		hbIn:   make(chan Heartbeat, 8),
		stop:   make(chan struct{}),

		testingHook: func() {}, // DO NOT REMOVE THIS LINE. A no-op when not testing.
	}
}

// Start starts e's main run loop as a separate goroutine. The main run loop
// handles incoming heartbeat requests and responses. The loop also trigger e's
// timeout procedure at an interval corresponding to e's internal delay
// duration variable.
func (e *EvtFailureDetector) Start() {
	e.timeoutSignal = time.NewTicker(e.delay)
	go func() {
		for {
			e.testingHook() // DO NOT REMOVE THIS LINE. A no-op when not testing.
			select {
			case <-e.hbIn:
				// TODO(student): Handle incoming heartbeat
			case <-e.timeoutSignal.C:
				e.timeout()
			case <-e.stop:
				return
			}
		}
	}()
}

// DeliverHeartbeat delivers heartbeat hb to failure detector e.
func (e *EvtFailureDetector) DeliverHeartbeat(hb Heartbeat) {
	e.hbIn <- hb
}

// Stop stops e's main run loop.
func (e *EvtFailureDetector) Stop() {
	e.stop <- struct{}{}
}

// Internal: timeout runs e's timeout procedure.
func (e *EvtFailureDetector) timeout() {
	// TODO(student): Implement timeout procedure
}

// TODO(student): Add other unexported functions or methods if needed.
```

The failure detector uses a `Heartbeat` struct to send heartbeat request and
replies. Outgoing heartbeats from the failure detector should be sent on the
`hbSend` channel. The `Heartbeat` struct is defined in the file `defs.go`:

```go
// A Heartbeat is the basic message used by failure detectors to communicate
// with other nodes. A heartbeat message can be of type request or reply.
type Heartbeat struct {
	From    int
	To      int
	Request bool // true -> request, false -> reply
}
```

You should complete the following parts of the code in `fd.go`:

* Perform any initialization necessary in the `NewEvtFailureDetector(...)`
  function.

* Implement handling of incoming heartbeat messages, more specifically `case
  <-e.hbIn:` in the `select` statement from the `Start()` method. Your code
  should check if the heartbeat is a request or response, and act accordingly.   

* Implement the failure detector's timeout procedure: `func (e
  *EvtFailureDetector) timeout()`. The failure detector should during the
  timeout procedure inform the proivded `SuspectRestorer` object if it thinks a
  node is suspected of being faulty or seem to have been restored. The failure
  detector has a reference to a `SuspectRestorer` entity via the `sr` field in
  the `EvtFailureDetector` struct. This field is of the type `SuspectRestorer`
  which is an interface. It defines two methods, `Suspect(id)` and
  `Restore(id)`, which is available to the failure detector.  If the failure
  detector determines that a node should be considered suspected or restored
  during the timeout procedure, then it should inform the `SuspectRestorer`
  object using these two available methods. You will in the next task implement
  a leader detector that satifies this interface. Take a look
  [here](https://golang.org/doc/effective_go.html#interfaces_and_types) to
  learn more about how interfaces work in Go.

Other information:

* A node identifier is defined to be an integer (type `int`).

* A node should not treat itself a special during the timeout procedure. This
  means that a node should not set itself alive without receiving a heartbeat
  reply from itself. You will as a consequence have to feed back heartbeat
  replies and requests to and from the node itself into the failure detector in
  your full implementation in the last
  [task](#distributed-leader-detection-50). This is done to simply testing,
  keep the code general and to let it mirror the algorithm in the textbook as
  closely as possible.

* You may add your own *unexported* functions or methods if needed.

## Leader Detector (25%)

In this task you will implement the Monarchical Eventual Leader Detector. The
description of this type of leader detector can be found on pages 56-60 in the
textbook. See Algorithm 2.8 for a complete description.

The initial skeleton code for the leader detector from the file `ld.go` is
listed below. You should, as you did for the failure detector, implement the
relevant parts of the code marked with `TODO(student)`. Refer to the code
comments for a complete specification of each type, function and method. The
corresponding tests can be found in `ld_test.go`.

```go
package detector

// A MonLeaderDetector represents a Monarchical Eventual Leader Detector as
// described at page 53 in:
// Christian Cachin, Rachid Guerraoui, and Luís Rodrigues: "Introduction to
// Reliable and Secure Distributed Programming" Springer, 2nd edition, 2011.
type MonLeaderDetector struct {
	// TODO(student): Add needed fields
}

// NewMonLeaderDetector returns a new Monarchical Eventual Leader Detector
// given a list of node ids.
func NewMonLeaderDetector(nodeIDs []int) *MonLeaderDetector {
	m := &MonLeaderDetector{}
	return m
}

// Leader returns the current leader. Leader will return UnknownID if all nodes
// are suspected.
func (m *MonLeaderDetector) Leader() int {
	// TODO(student): Implement
	return UnknownID
}

// Suspect instructs m to consider the node with matching id as suspected. If
// the suspect indication result in a leader change the leader detector should
// this publish this change its subscribers.
func (m *MonLeaderDetector) Suspect(id int) {
	// TODO(student): Implement
}

// Restore instructs m to consider the node with matching id as restored. If
// the restore indication result in a leader change the leader detector should
// this publish this change its subscribers.
func (m *MonLeaderDetector) Restore(id int) {
	// TODO(student): Implement
}

// Subscribe returns a buffered channel that m on leader change will use to
// publish the id of the highest ranking node. The leader detector will publish
// UnknownID if all nodes become suspected. Subscribe will drop publications to
// slow subscribers. Note: Subscribe returns a unique channel to every
// subscriber; it is not meant to be shared.
func (m *MonLeaderDetector) Subscribe() <-chan int {
	return nil
}

// TODO(student): Add other unexported functions or methods if needed.
```

Other information:

* The *Monarchiacal* Eventual Leader Detection uses a node ranking. The ranking
  for this implementation is defined using the node identifiers. The highest
  ranking node is defined to be the node with the highest id (i.e. the highest
  integer).
 
* Negative node ids should be ignored by the leader detector. An exception is
  the special identifier constant `UnknownID` in `defs.go` with value `-1`.

* The leader detector algorithm from the textbook send a `< TRUST | leader >`
  indication event when a new leader is detected. This behavior is modeled in
  the code by the `Subscribe()` method. Any part of a application (e.g. a
  module) can call `Subscribe()` to be notified about leader change events.
  Each caller will receive a unique channel where they may receive `< TRUST >`
  indications in the form of node ids (`int`).   

* You may add your own *unexported* functions or methods if needed.

## Distributed Leader Detection (50%)

In this task you will implement a complete distributed application using the
previously implemented failure detector and leader detector, running on a real
network stack. 

The distributed application will simply subscribe to leader change events
from the leader detector module, and print to screen whenever there is a
change of leadership.

However, the most challenging part of this lab may be the network layer that
needs to handle network connections to multiple servers. This network layer
will serve as a foundation for future lab assignments, in which case you may
need to extend this implementation to support future assignments. *It is
therefore important that you spend some time on the design and architecture of
your network layer.* This may prevent you from having to do major code
refactoring down the line.

This part of the assignment will, as noted [below](#lab-approval), not be
tested or verified using the Autograder.

### Specification

1. Your application should at start-up establish network connections between
   the servers.

2. Your application should create an instance of both the leader detector and
   failure detector.

3. The application should after initialization start the failure detector and
   then start listening for leader changes.

4. The application should print any leader change indication to screen. You may
   provoke such an event by stopping your application on one or more of the
   machines.

Additional tips and information: 

* You may choose the type of transport layer protocol to use for network
  communication, e.g. TCP or UDP.

* You can implement the system start-up statically. This means that you can
  choose a few nodes in the lab and use their DNS name or IP address and port
  number to statically set up your group of machines. These can either be
  hardcoded in your source code, or read from a configuration file.
  
* You will need to encode/decode the `Heartbeat` messages to a data-interchange
  format when sending and receiving them on the network. You may use a text
  encoding such as JSON, or a binary encoding such as for example
  [Gob](http://golang.org/pkg/encoding/gob/).

* While not a strict requirement, it is highly recommended that you implement
  unit tests for at least a portion of the network layer functions that you
  will implement in this task.

## Lab Approval

To have your lab assignment approved, you must come to the lab during lab hours
and present your solution. This lets you present the thought process behind
your solution, and gives us more information for grading purposes and we may
also provide feedback on your solution then and there. When you are ready to
show your solution, reach out to a member of the teaching staff. It is
expected that you can explain your code and show how it works. You may show
your solution on a lab workstation or your own computer.

**60% of this lab must be passed for the lab to be approved.** The results from
Autograder will be taken into consideration when approving a lab.

You should for this lab present a working demo of the application described in
the previous section. The application should run on at least three machines in
the lab (not enough to run just on localhost). You should demonstrate that your
implementation fulfills the previously listed specification. The task will be
verified by a member of the teaching staff during lab hours.

Also see the [Grading and Collaboration
Policy](https://github.com/uis-dat520/course-info/policy.md) document for
additional information.
