package lab1

import (
	"strings"
	"testing"
)

var rot13Tests = []struct {
	in, want string
}{
	{"Go programming is fun.",
		"Tb cebtenzzvat vf sha."},
	{"Tb cebtenzzvat vf sha.",
		"Go programming is fun."},
	{"There are two hard things in computer science: cache invalidation, naming things, and off-by-one errors.",
		"Gurer ner gjb uneq guvatf va pbzchgre fpvrapr: pnpur vainyvqngvba, anzvat guvatf, naq bss-ol-bar reebef."},
	{"Gurer ner gjb uneq guvatf va pbzchgre fpvrapr: pnpur vainyvqngvba, anzvat guvatf, naq bss-ol-bar reebef.",
		"There are two hard things in computer science: cache invalidation, naming things, and off-by-one errors."},
}

func TestRot13(t *testing.T) {
	b := make([]byte, 1024)
	for i, rt := range rot13Tests {
		s := strings.NewReader(rt.in)
		r := rot13Reader{s}
		n, err := r.Read(b)
		if err != nil {
			t.Errorf("rot13 test %d: got %v, expected EOF", i, err)
		}
		out := string(b[:n])
		if out != rt.want {
			t.Errorf("rot13 test %d: got %q for input %q, want %q", i, out, rt.in, rt.want)
		}

	}
}
