package lab1

import (
	. "io"
	"testing"
)

// String: "io: read/write on closed pipe"
var s = ErrClosedPipe.Error()

var errTests = []struct {
	in   Errors
	want string
}{
	{nil, "(0 errors)"},
	{[]error{}, "(0 errors)"},
	{[]error{ErrClosedPipe}, s},
	{[]error{ErrClosedPipe, ErrClosedPipe}, s + " (and 1 other error)"},
	{[]error{ErrClosedPipe, ErrClosedPipe, ErrClosedPipe}, s + " (and 2 other errors)"},
	{[]error{nil}, "(0 errors)"},
	{[]error{ErrClosedPipe, nil}, s},
	{[]error{nil, ErrClosedPipe}, s},
	{[]error{ErrClosedPipe, ErrClosedPipe, nil}, s + " (and 1 other error)"},
	{[]error{ErrClosedPipe, ErrClosedPipe, nil, nil}, s + " (and 1 other error)"},
	{[]error{ErrClosedPipe, ErrClosedPipe, nil, nil, nil}, s + " (and 1 other error)"},
	{[]error{nil, ErrClosedPipe, nil, ErrClosedPipe}, s + " (and 1 other error)"},
	{[]error{nil, nil, ErrClosedPipe, ErrClosedPipe}, s + " (and 1 other error)"},
	{[]error{ErrClosedPipe, nil, nil, ErrClosedPipe, ErrClosedPipe}, s + " (and 2 other errors)"},
	{[]error{ErrClosedPipe, ErrClosedPipe, nil, ErrClosedPipe, nil}, s + " (and 2 other errors)"},
	{[]error{nil, nil, nil, nil, ErrClosedPipe, ErrClosedPipe, ErrClosedPipe}, s + " (and 2 other errors)"},
}

func TestErrors(t *testing.T) {
	for i, ft := range errTests {
		out := ft.in.Error()
		if out != ft.want {
			t.Errorf("Errors test %d: got %q for input %v, want %q", i, out, ft.in, ft.want)
		}
	}
}
