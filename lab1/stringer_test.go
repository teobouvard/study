package lab1

import "testing"

var stringerTests = []struct {
	in   Student
	want string
}{
	{Student{
		ID:        42,
		FirstName: "John",
		LastName:  "Doe",
		Age:       25,
	}, "Student ID: 42. Name: Doe, John. Age: 25."},
	{Student{
		ID:        1490,
		FirstName: "Tormod",
		LastName:  "Lea",
		Age:       30,
	}, "Student ID: 1490. Name: Lea, Tormod. Age: 30."},
}

func TestStringer(t *testing.T) {
	for i, st := range stringerTests {
		out := st.in.String()
		if out != st.want {
			t.Errorf("stringer test %d: got %q for input %v, want %q", i, out, st.in, st.want)
		}
	}
}
