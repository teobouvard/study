package lab1

import (
	"io"
	"io/ioutil"
	"reflect"
	"testing"
)

type failureWriter int

func (f failureWriter) Write(p []byte) (n int, err error) {
	return int(f), nil
}

var writeTests = []struct {
	inb   []byte
	inw   []io.Writer
	wantn []int
	wante Errors
}{
	{nil, nil, []int{}, nil},
	{nil, []io.Writer{}, []int{}, nil},
	{nil, []io.Writer{ioutil.Discard}, []int{0}, nil},
	{[]byte(""), []io.Writer{ioutil.Discard}, []int{0}, nil},
	{[]byte("\n"), []io.Writer{ioutil.Discard}, []int{1}, nil},
	{[]byte("DAT520-1\n"), []io.Writer{ioutil.Discard}, []int{9}, nil},
	{[]byte("DAT520-2\n"), []io.Writer{ioutil.Discard}, []int{9}, nil},
	{[]byte("DAT520-3\n"), []io.Writer{ioutil.Discard, ioutil.Discard}, []int{9, 9}, nil},
	{[]byte("DAT520-4\n"), []io.Writer{ioutil.Discard, ioutil.Discard, ioutil.Discard}, []int{9, 9, 9}, nil},
	{
		[]byte("DAT520-5\n"), []io.Writer{ioutil.Discard,
			failureWriter(2),
			ioutil.Discard,
		},
		[]int{9, 2, 9},
		[]error{nil, io.ErrShortWrite, nil},
	},
	{
		[]byte("DAT520-6\n"), []io.Writer{ioutil.Discard,
			failureWriter(2),
			failureWriter(6),
		},
		[]int{9, 2, 6},
		[]error{nil, io.ErrShortWrite, io.ErrShortWrite},
	},
}

func TestWriters(t *testing.T) {
	for i, ft := range writeTests {
		n, errs := WriteTo(ft.inb, ft.inw...)
		if !reflect.DeepEqual(n, ft.wantn) {
			t.Errorf("Writers test %d: got %v bytes written, want %v", i+1, n, ft.wantn)
		}
		if !errsEqual(errs, ft.wante) {
			t.Errorf("Writers test %d: error slices not equal.", i+1)
			printErrors(t, errs, ft.wante)
		}
	}
}

func errsEqual(a, b []error) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func printErrors(t *testing.T, got, want []error) {
	t.Log("got:")
	for i, e := range got {
		t.Log(i, e)
	}
	t.Log("want:")
	for i, e := range want {
		t.Log(i, e)
	}
}
