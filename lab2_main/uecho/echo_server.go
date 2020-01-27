// +build !solution

// Leave an empty line above this comment.
package main

import (
	"fmt"
	"net"
	"strings"
	"unicode"
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
	s := new(UDPServer)
	resolvedAddr, err := net.ResolveUDPAddr("udp", addr)
	if err != nil {
		return nil, err
	}

	s.conn, err = net.ListenUDP("udp", resolvedAddr)
	if err != nil {
		return nil, err
	}

	fmt.Printf("Server listening on %v\n", addr)
	return s, err
}

// ServeUDP starts the UDP server's read loop. The server should read from its
// listening socket and handle incoming client requests as according to the
// the specification.
func (u *UDPServer) ServeUDP() {
	var buf [512]byte

	for {
		n, addr, err := u.conn.ReadFromUDP(buf[:])

		if err != nil {
			fmt.Println("ReadFromUDP error :", err)
		} else {
			fmt.Println("Command from", addr, ":", string(buf[:n]))
			resp := executeCommand(string(buf[:n]))
			n, err = u.conn.WriteTo([]byte(resp), addr)
			if err != nil {
				fmt.Println("WriteTo error :", err)
			} else {
				fmt.Println("Replied", resp)
			}
		}
	}
}

func executeCommand(cmd string) string {
	// handling malformed input not in spec, "SWAP|:||:|abcdefghiklm" is a correct command
	s := strings.Split(cmd, "|:|")
	action, txt := strings.Join(s[:len(s)-1], " "), s[len(s)-1]

	switch action {
	case "UPPER":
		return strings.ToUpper(txt)
	case "LOWER":
		return strings.ToLower(txt)
	case "CAMEL":
		// smart API design
		return strings.Title(strings.ToLower(txt))
	case "ROT13":
		return strings.Map(rot13, txt)
	case "SWAP":
		return strings.Map(invertCase, txt)
	default:
		return "Unknown command"
	}
}

func rot13(r rune) rune {
	if 'a' <= r && r <= 'z' {
		return ((r-'a')+13)%26 + 'a'
	} else if 'A' <= r && r <= 'Z' {
		return ((r-'A')+13)%26 + 'A'
	}
	return r
}

func invertCase(r rune) rune {
	if unicode.IsUpper(r) {
		return unicode.ToLower(r)
	} else if unicode.IsLower(r) {
		return unicode.ToUpper(r)
	}
	return r
}

// socketIsClosed is a helper method to check if a listening socket has been
// closed.
func socketIsClosed(err error) bool {
	if strings.Contains(err.Error(), "use of closed network connection") {
		return true
	}
	return false
}
