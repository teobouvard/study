package network

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"net"

	"../detector"
)

type Server struct {
	conn   *net.UDPConn
	notify chan detector.Heartbeat
}

// NewServer returns a new UDP Server listening on addr.
func NewServer(addr *net.UDPAddr) (*Server, error) {
	conn, err := net.ListenUDP("udp", addr)
	s := &Server{
		conn:   conn,
		notify: make(chan detector.Heartbeat),
	}
	return s, err
}

// Listen starts the UDP server's read loop. The server read
// from its listening socket and handle incoming client requests message
func (u *Server) Listen() chan detector.Heartbeat {
	var buf [512]byte

	go func() {
		for {
			n, _, err := u.conn.ReadFromUDP(buf[:])
			if err != nil {
				fmt.Println(err)
			} else {
				var hb detector.Heartbeat
				decoder := gob.NewDecoder(bytes.NewReader(buf[:n]))
				err := decoder.Decode(&hb)
				if err != nil {
					fmt.Println(err)
				}
				u.notify <- hb
			}
		}
	}()

	return u.notify
}

func (u *Server) Send(addr *net.UDPAddr, hb detector.Heartbeat) (n int, err error) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	err = encoder.Encode(hb)
	if err != nil {
		return 0, err
	}
	n, err = u.conn.WriteTo(buf.Bytes(), addr)
	return n, err
}
