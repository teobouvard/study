![UiS](https://www.uis.no/getfile.php/13391907/Biblioteket/Logo%20og%20veiledninger/UiS_liggende_logo_liten.png)

# Lab 2: Network Programming in Go

| Lab 2:           | Network Programming in Go  |
| ---------------- | -------------------------- |
| Subject:         | DAT520 Distributed Systems |
| Deadline:        | Thursday Jan 30 2020 18:00 |
| Expected effort: | 10-15 hours                |
| Grading:         | Pass/fail                  |
| Submission:      | Individually               |

### Table of Contents

1. [Introduction](https://github.com/dat520-2020/assignments/blob/master/lab2/README.md#introduction)
2. [UDP Echo Server](https://github.com/dat520-2020/assignments/blob/master/lab2/README.md#udp-echo-server)
3. [Web Server](https://github.com/dat520-2020/assignments/blob/master/lab2/README.md#web-server)
4. [gRPC](https://github.com/dat520-2020/assignments/blob/master/lab2/README.md#remote-procedure-call)
5. [Lab Approval](https://github.com/dat520-2020/assignments/blob/master/lab2/README.md#lab-approval)

## Introduction

The goal of this lab assignment is to get you started with network programming
in Go. The overall aim of the lab project is to implement a fault tolerant
distributed application, specifically a reliable bulletin board. Knowledge of
network programming in Go is naturally a prerequisite for accomplishing this.

This lab assignment consist of two parts. In the first part you are expected to
implement a simple echo server that is able to respond to different commands
specified as text. In the second part you will be implementing a simple web
server. You will need this latter skill, together with other web-design skills,
to construct a web-based user interface/front-end for your reliable bulletin
board.

The most important packages in the Go standard library that you will use in
this assignment is the [`net`](http://golang.org/pkg/net) and
[`net/http`](http://golang.org/pkg/net/http) packages. It is recommended that
you actively use the documentation available for these packages during your
work on this lab assignment. You will also find this [web
tutorial](https://golang.org/doc/articles/wiki/) highly useful.

## UDP Echo Server

In this task we will focus on the user datagram protocol (UDP), which provides
unreliable datagram service. You will find the documentation of the
[UDPConn](https://golang.org/pkg/net/#UDPConn) type useful.

In the provided code under `uecho`, we have implemented a simple
`SendCommand()` function that acts as a client, along with a bunch of tests.
You can run these test with `go test -v`, and as described in Lab 1, you can
use the `-run` flag to run only a specific test.

You can also compile your server code into a binary using `go build`. This
will produce a file named `uecho` in the same folder as the `.go` source files.
You can run this binary in two ways:

1. `./uecho -server &` will start the server in the background. Note: _This will
   not work until you have implemented the necessary server parts._

2. `./uecho` will start the command line client, from which you may interact with
   the server by typing commands into the terminal window.

If you want to extend the capabilities of this runnable client and server,
you can edit the files `echo.go` and `echo_client.go`. But note that the
tests executed by the autograder will use original `SendCommand()` provided
in the original `echo_client.go` file. If you've done something fancy,
and want to show us that's fine, but it won't be considered by the autograder.

#### Echo server specification:

The `SendCommand()` takes the following arguments:

| Argument  | Description                                                                  |
| --------- | ---------------------------------------------------------------------------- |
| `udpAddr` | UDP address of the server (`localhost:12110`)                                |
| `cmd`     | Command (as a text string) that the server should interpret and execute      |
| `txt`     | Text string on which the server should perform the command provided in `cmd` |

The `SendCommand()` function produces a string composed of the following

```
cmd|:|txt
```

For example:

```
UPPER|:|i want to be upper case
```

From this, the server is expected to produce the following reply:

```
I WANT TO BE UPPER CASE
```

See below for more details about the specific behaviors of the server.

1. For each of the following commands, implement the corresponding functions, so that the returned value corresponds to the expected test outcome. Here you are expected to implement demultiplexer that demultiplexes the input (the command) so that different actions can be taken. A hint is to use the `switch` statement. You will probably also need the `strings.Split()` function.

   | Command | Action                                                                                                                                                                                                                           |
   | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | UPPER   | Takes the provided input string `txt` and applies the translates it to upper case using `strings.ToUpper()`.                                                                                                                     |
   | LOWER   | Same as UPPER, but lower case instead.                                                                                                                                                                                           |
   | CAMEL   | Same as UPPER, but title or camel case instead.                                                                                                                                                                                  |
   | ROT13   | Takes the provided input string `txt` and applies the rot13 translation to it; see lab1 for an example.                                                                                                                          |
   | SWAP    | Takes the provided input string `txt` and inverts the case. For this command you will find the `strings.Map()` function useful, together with the `unicode.IsUpper()` and `unicode.ToLower()` and a few other similar functions. |

2. The server should reply `Unknown command` if it receives an unknown command
   or fails to interpret a request in any way.

3. Make sure that your server continues to function even if one client's
   connection or datagram packet caused an error.

#### Echo server implementation

You should implement the specification by extending the skeleton code found in
`echo_server.go`:

```go
// +build !solution

// Leave an empty line above this comment.
package main

import (
	"net"
	"strings"
)

// UDPServer implements the UDP server specification found at
// https://github.com/dat520-2020/assignments/blob/master/lab2/README.md#udp-server
type UDPServer struct {
	conn *net.UDPConn
	// TODO(student): Add fields if needed
}

// NewUDPServer returns a new UDPServer listening on addr. It should return an
// error if there was any problem resolving or listening on the provided addr.
func NewUDPServer(addr string) (*UDPServer, error) {
	// TODO(student): Implement
	return nil, nil
}

// ServeUDP starts the UDP server's read loop. The server should read from its
// listening socket and handle incoming client requests as according to the
// the specification.
func (u *UDPServer) ServeUDP() {
	// TODO(student): Implement
}

// socketIsClosed is a helper method to check if a listening socket has been
// closed.
func socketIsClosed(err error) bool {
	if strings.Contains(err.Error(), "use of closed network connection") {
		return true
	}
	return false
}
```

## Web Server

In this task you will learn how to implement a very simple web server. This
will be useful since you in a later lab most likely will have to construct one
as a user interface to your fault tolerant application.

Using the skeleton code found in `web/server.go`, implement the web server
specification found below. All patterns listed assume `localhost:8080` as root.
Go's http package is documented [here](http://golang.org/pkg/net/http/).

There are tests for each of the cases listed below in `server_test.go`. The
command `go test -v` will run all tests. As described in Lab 1, use the `-run`
flag to only run a specific test. Note that you do not need to start the web
server manually for running the provided tests. But, you may also manually test
your implementation using a web browser or the `curl` tool. In one terminal
window, start the web server by running the command `go run server.go`. The
server should then be available at `localhost:8080`. For example, run `curl -v localhost:8080` in another terminal window to manually verify the first item
of the web server specification. An example output is shown below.

```
$ curl -v localhost:8080
* Rebuilt URL to: localhost:8080/
*   Trying ::1...
* Connected to localhost (::1) port 8080 (#0)
> GET / HTTP/1.1
> User-Agent: curl/7.40.0
> Host: localhost:8080
> Accept: */*
>
< HTTP/1.1 200 OK
< Date: Sun, 18 Jan 2015 17:33:27 GMT
< Content-Length: 12
< Content-Type: text/plain; charset=utf-8
<
* Connection #0 to host localhost left intact
Hello World!%
```

#### Web server specification:

1. The pattern `/` (root) should return status code `200` and the body `Hello World!\n`.

1. The web server should count the number of HTTP requests made to it (to any
   pattern) since it started. This counter should be accessible at the pattern
   `/counter`. A request to this pattern should return status code `200` and
   the current count (inclusive the current request) as the body, e.g.
   `counter: 42\n.`

1. A request to the pattern `/lab2` should return status code `301` to the
   client with body `<a href=\"http://www.github.com/dat520-2020/assignments/tree/master/lab2\">Moved Permanently</a>.\n\n`.

1. The pattern `/fizzbuzz` should implement the Fizz buzz game. It should
   return "fizz" if the value is divisible by 3. It should return "buzz" if the
   value is divisible by 5. It should return "fizzbuzz" if the value is both
   divisible by 3 **and** 5. It should return the number itself for any other
   case. The value should be passed as a URL query string parameter named
   `value`. For example, a request to `/fizzbuzz?value=30` should return `200`
   and body `fizzbuzz\n`. A request to `/fizzbuzz?value=31` should return `200`
   and body `31\n`. The server should return status code `200` and the body
   `not an integer\n` if the `value` parameter can't be parsed as an integer.
   If the `value` parameter is empty, the server should return status code
   `200` and the body `no value provided\n`.

1. All other patterns should return status code `404` and the body `404 page not found\n`.

## Remote procedure call

The description for gRPC can be found here: [gRPC](https://github.com/dat520-2020/assignments/tree/master/lab2/grpc)

## Lab Approval

To have your lab assignment approved, you must come to the lab during lab hours
and present your solution. This lets you present the thought process behind
your solution, and gives us more information for grading purposes. When you are
ready to show your solution, reach out to a member of the teaching staff. It
is expected that you can explain your code and show how it works. You may show
your solution on a lab workstation or your own computer. The results from
Autograder will also be taken into consideration when approving a lab. At least
60% of the Autograder tests should pass for the lab to be approved. A lab needs
to be approved before Autograder will provide feedback on the next lab
assignment.

Also see the [Grading and Collaboration
Policy](https://github.com/uis-dat520/course-info/policy.md) document for
additional information.
