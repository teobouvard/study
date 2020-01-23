// +build !solution

package lab1

import (
	"io"
)

/*
Task 3: Rot 13

This task is taken from http://tour.golang.org.

A common pattern is an io.Reader that wraps another io.Reader, modifying the
stream in some way.

For example, the gzip.NewReader function takes an io.Reader (a stream of
compressed data) and returns a *gzip.Reader that also implements io.Reader (a
stream of the decompressed data).

Implement a rot13Reader that implements io.Reader and reads from an io.Reader,
modifying the stream by applying the rot13 substitution cipher to all
alphabetical characters.

The rot13Reader type is provided for you. Make it an io.Reader by implementing
its Read method.
*/

type rot13Reader struct {
	r io.Reader
}

func (r rot13Reader) Read(p []byte) (n int, err error) {
	n, err = r.r.Read(p)
	for i, char := range string(p) {
		if 'a' <= char && char <= 'z' {
			char = char - 'a' + 13
			char %= 26
			char += 'a'
		} else if 'A' <= char && char <= 'Z' {
			char = char - 'A' + 13
			char %= 26
			char += 'A'
		}
		p[i] = byte(char)
	}
	return n, err
}
