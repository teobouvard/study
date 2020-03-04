package network

import "net"

// Network TODO
type Network struct {
	registry map[int]*net.UDPAddr
}
