// +build !solution

package main

import (
	"flag"
	"log"
	"net/http"
)

var httpAddr = flag.String("http", ":8080", "Listen address")

func main() {
	flag.Parse()
	server := NewServer()
	log.Fatal(http.ListenAndServe(*httpAddr, server))
}

// Server implements the web server specification found at
// https://github.com/dat520-2020/assignments/blob/master/lab2/README.md#web-server
type Server struct {
	// TODO(student): Add needed fields
}

// NewServer returns a new Server with all required internal state initialized.
// NOTE: It should NOT start to listen on an HTTP endpoint.
func NewServer() *Server {
	s := &Server{}
	// TODO(student): Implement
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// TODO(student): Implement
}
