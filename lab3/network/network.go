package network

import (
	"bytes"
	"encoding/gob"
	"net"
	"strings"

	"github.com/dat520-2020/TeamPilots/lab3/detector"
)

// Server is the interface between the app and the outside world
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
				if strings.Contains(err.Error(), "use of closed network connection") {
					return
				}
				panic(err)
			}
			var hb detector.Heartbeat
			decoder := gob.NewDecoder(bytes.NewReader(buf[:n]))
			err = decoder.Decode(&hb)
			if err != nil {
				panic(err)
			}
			u.notify <- hb
		}
	}()

	return u.notify
}

// Send sends a heartbeat to a UDP adress
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

// Stop closes the connection
func (u *Server) Stop() {
	u.conn.Close()
}
