package main

import (
	"bufio"
	"fmt"
	"net"
	"os"
)

// SendCommand sends the command cmd with payload txt as a UDP packet to
// address updAddr. SendCommand prints errors to output.
//
// DO NOT EDIT
func SendCommand(udpAddr, cmd, txt string) (string, error) {
	addr, err := net.ResolveUDPAddr("udp", udpAddr)
	if err != nil {
		return "", err
	}
	conn, err := net.DialUDP("udp", nil, addr)
	if err != nil {
		return "", err
	}
	defer conn.Close()
	var buf [512]byte
	cmdTxt := fmt.Sprintf("%v|:|%v", cmd, txt)
	_, err = conn.Write([]byte(cmdTxt))
	if err != nil {
		return "", err
	}
	n, err := conn.Read(buf[0:])
	if err != nil {
		return "", err
	}
	return string(buf[0:n]), nil
}

var cmdTypes = []string{
	"UPPER",
	"LOWER",
	"CAMEL",
	"ROT13",
	"SWAP",
}

func clientLoop(udpAddr string) {
	var (
		scanner *bufio.Scanner = bufio.NewScanner(os.Stdin)
		ct      int
		txt     string
		resp    string
		err     error
	)
	for {
		fmt.Println("Enter command type:")
		for i, ct := range cmdTypes {
			fmt.Printf("%v for %q\n", i+1, ct)
		}
		fmt.Scanln(&ct)
		if ct < 0 || ct > len(cmdTypes) {
			fmt.Println("Error: Unkown command")
			continue
		}

		fmt.Println("Enter text:")
		scanner.Scan()
		txt = scanner.Text()
		if err := scanner.Err(); err != nil {
			fmt.Println("Error reading text:", err)
			continue
		}

		resp, err = SendCommand(udpAddr, cmdTypes[ct-1], txt)
		if err != nil {
			fmt.Println("Error attempting to send command:", err)
		} else {
			fmt.Println("Got response:", resp)
		}
	}
}
