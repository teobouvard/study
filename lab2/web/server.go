// +build !solution

package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"strconv"
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
	counter int
}

// NewServer returns a new Server with all required internal state initialized.
// NOTE: It should NOT start to listen on an HTTP endpoint.
func NewServer() *Server {
	s := &Server{}
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.counter++
	s.handleRequest(w, r)
}

func (s *Server) handleRequest(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/":
		fmt.Fprintf(w, "Hello World!\n")
		break
	case "/counter":
		fmt.Fprintf(w, "counter: %d\n", s.counter)
		break
	case "/lab2":
		w.WriteHeader(301)
		fmt.Fprintf(w, "<a href=\"http://www.github.com/dat520-2020/assignments/tree/master/lab2\">Moved Permanently</a>.\n\n")
		break
	case "/fizzbuzz":
		param := r.URL.Query().Get("value")
		if param == "" {
			w.WriteHeader(200)
			fmt.Fprintf(w, "no value provided\n")
			break
		}

		value, err := strconv.Atoi(param)
		if err != nil {
			w.WriteHeader(200)
			fmt.Fprintf(w, "not an integer\n")
			break
		}

		if value%3 == 0 && value%5 == 0 {
			fmt.Fprintf(w, "fizzbuzz\n")
		} else if value%3 == 0 {
			fmt.Fprintf(w, "fizz\n")
		} else if value%5 == 0 {
			fmt.Fprintf(w, "buzz\n")
		} else {
			fmt.Fprintf(w, "%d\n", value)
		}
		break
	default:
		w.WriteHeader(404)
		fmt.Fprintf(w, "404 page not found\n")
	}
}
