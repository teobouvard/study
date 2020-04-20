package web

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

var testRootRequests = []struct {
	reqType  string
	reqPath  string
	wantCode int
	wantBody string
}{
	{"GET", "/", 200, "Hello World!\n"},
}

func TestRoot(t *testing.T) {
	server := NewServer()
	for i, tr := range testRootRequests {
		req, err := http.NewRequest(tr.reqType, tr.reqPath, nil)
		if err != nil {
			t.Fatalf("TestRoot %d: got error creating http req: %v\n", i, err)
		}
		w := httptest.NewRecorder()
		server.ServeHTTP(w, req)
		if w.Code != tr.wantCode {
			t.Errorf("TestRoot %d: got status code %d, want %d", i, w.Code, tr.wantCode)
		}
		if w.Body.String() != tr.wantBody {
			t.Errorf("TestRoot %d: got status body %q, want %q", i, w.Body.String(), tr.wantBody)
		}
	}
}

var testNonExsistingRequests = []struct {
	reqType  string
	reqPath  string
	wantCode int
	wantBody string
}{
	{"GET", "/foo", 404, "404 page not found\n"},
	{"GET", "/foo/bar", 404, "404 page not found\n"},
	{"GET", "/uis", 404, "404 page not found\n"},
}

func TestNonExsisting(t *testing.T) {
	server := NewServer()
	for i, tr := range testNonExsistingRequests {
		req, err := http.NewRequest(tr.reqType, tr.reqPath, nil)
		if err != nil {
			t.Fatalf("TestNonExisting %d: got error creating http req: %v\n", i, err)
		}
		w := httptest.NewRecorder()
		server.ServeHTTP(w, req)
		if w.Code != tr.wantCode {
			t.Errorf("TestNonExisting %d: got status code %d, want %d", i, w.Code, tr.wantCode)
		}
		if w.Body.String() != tr.wantBody {
			t.Errorf("TestNonExisting %d: got status body %q, want %q", i, w.Body.String(), tr.wantBody)
		}
	}
}

var testRedirectRequests = []struct {
	reqType  string
	reqPath  string
	wantCode int
	wantBody string
}{
	{"GET", "/lab2", 301, "<a href=\"http://www.github.com/dat520-2020/assignments/tree/master/lab2\">Moved Permanently</a>.\n\n"},
	{"GET", "/lab2/foo", 404, "404 page not found\n"},
}

func TestRedirect(t *testing.T) {
	server := NewServer()
	for i, tr := range testRedirectRequests {
		req, err := http.NewRequest(tr.reqType, tr.reqPath, nil)
		if err != nil {
			t.Fatalf("TestRedirect %d: got error creating http req: %v\n", i, err)
		}
		w := httptest.NewRecorder()
		server.ServeHTTP(w, req)
		if w.Code != tr.wantCode {
			t.Errorf("TestRedirect %d: got status code %d, want %d", i, w.Code, tr.wantCode)
		}
		if w.Body.String() != tr.wantBody {
			t.Errorf("TestRedirect %d: got status body %q, want %q", i, w.Body.String(), tr.wantBody)
		}
	}
}

var testCounterRequests = []struct {
	reqType  string
	reqPath  string
	wantCode int
	wantBody string
}{
	{"GET", "/counter", 200, "counter: 1\n"},
	{"GET", "/counter", 200, "counter: 2\n"},
	{"GET", "/counter", 200, "counter: 3\n"},
	{"GET", "/counter", 200, "counter: 4\n"},
	{"GET", "/counter", 200, "counter: 5\n"},
}

func TestCounter(t *testing.T) {
	server := NewServer()
	for i, tr := range testCounterRequests {
		req, err := http.NewRequest(tr.reqType, tr.reqPath, nil)
		if err != nil {
			t.Fatalf("TestCounter %d: got error creating http req: %v\n", i, err)
		}
		w := httptest.NewRecorder()
		server.ServeHTTP(w, req)
		if w.Code != tr.wantCode {
			t.Errorf("TestCounter %d: got status code %d, want %d", i, w.Code, tr.wantCode)
		}
		if w.Body.String() != tr.wantBody {
			t.Errorf("TestCounter %d: got status body %q, want %q", i, w.Body.String(), tr.wantBody)
		}
	}
}

var testFizzbuzzRequests = []struct {
	reqType  string
	reqPath  string
	wantCode int
	wantBody string
}{
	{"GET", "/fizzbuzz?value=1", 200, "1\n"},
	{"GET", "/fizzbuzz?value=2", 200, "2\n"},
	{"GET", "/fizzbuzz?value=3", 200, "fizz\n"},
	{"GET", "/fizzbuzz?value=4", 200, "4\n"},
	{"GET", "/fizzbuzz?value=5", 200, "buzz\n"},
	{"GET", "/fizzbuzz?value=30", 200, "fizzbuzz\n"},
	{"GET", "/fizzbuzz?value=lol", 200, "not an integer\n"},
	{"GET", "/fizzbuzz?value", 200, "no value provided\n"},
	{"GET", "/fizzbuzz?value=60", 200, "fizzbuzz\n"},
	{"GET", "/fizzbuzz?value=61", 200, "61\n"},
}

func TestFizzbuzz(t *testing.T) {
	server := NewServer()
	for i, tr := range testFizzbuzzRequests {
		req, err := http.NewRequest(tr.reqType, tr.reqPath, nil)
		if err != nil {
			t.Fatalf("TestFizzbuzz %d: got error creating http req: %v\n", i, err)
		}
		w := httptest.NewRecorder()
		server.ServeHTTP(w, req)
		if w.Code != tr.wantCode {
			t.Errorf("TestFizzbuzz %d: got status code %d, want %d", i, w.Code, tr.wantCode)
		}
		if w.Body.String() != tr.wantBody {
			t.Errorf("TestFizzbuzz %d: got status body %q, want %q", i, w.Body.String(), tr.wantBody)
		}
	}
}

var testServerFullRequests = []struct {
	reqType  string
	reqPath  string
	wantCode int
	wantBody string
}{
	{"GET", "/", 200, "Hello World!\n"},
	{"GET", "/foo", 404, "404 page not found\n"},
	{"GET", "/counter", 200, "counter: 3\n"},
	{"GET", "/lab2", 301, "<a href=\"http://www.github.com/dat520-2020/assignments/tree/master/lab2\">Moved Permanently</a>.\n\n"},
	{"GET", "/counter", 200, "counter: 5\n"},
	{"GET", "/fizzbuzz?value=1", 200, "1\n"},
	{"GET", "/fizzbuzz?value=3", 200, "fizz\n"},
	{"GET", "/fizzbuzz?value=5", 200, "buzz\n"},
	{"GET", "/fizzbuzz?value=30", 200, "fizzbuzz\n"},
	{"GET", "/counter", 200, "counter: 10\n"},
	{"GET", "/foobar", 404, "404 page not found\n"},
	{"GET", "/fizzbuzz?value=43", 200, "43\n"},
	{"GET", "/fizzbuzz?value=44", 200, "44\n"},
	{"GET", "/fizzbuzz?value=45", 200, "fizzbuzz\n"},
	{"GET", "/counter", 200, "counter: 15\n"},
	{"GET", "/fizzbuzz?value=hei", 200, "not an integer\n"},
	{"GET", "/fizzbuzz?value", 200, "no value provided\n"},
	{"GET", "/counter", 200, "counter: 18\n"},
	{"GET", "/foobar", 404, "404 page not found\n"},
}

func TestServerFull(t *testing.T) {
	server := NewServer()
	for i, tr := range testServerFullRequests {
		req, err := http.NewRequest(tr.reqType, tr.reqPath, nil)
		if err != nil {
			t.Fatalf("TestServerFull %d: got error creating http req: %v\n", i, err)
		}
		w := httptest.NewRecorder()
		server.ServeHTTP(w, req)
		if w.Code != tr.wantCode {
			t.Errorf("TestServerFull %d: got status code %d, want %d", i, w.Code, tr.wantCode)
		}
		if w.Body.String() != tr.wantBody {
			t.Errorf("TestServerFull %d: got status body %q, want %q", i, w.Body.String(), tr.wantBody)
		}
	}
}
