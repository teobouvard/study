package lab1

import "testing"

var fibonacciTests = []struct {
	in, want uint
}{
	{0, 0},
	{1, 1},
	{2, 1},
	{3, 2},
	{4, 3},
	{5, 5},
	{6, 8},
	{7, 13},
	{8, 21},
	{9, 34},
	{10, 55},
	{20, 6765},
}

func TestFibonacci(t *testing.T) {
	for i, ft := range fibonacciTests {
		out := fibonacci(ft.in)
		if out != ft.want {
			t.Errorf("fib test %d: got %d for input %d, want %d", i, out, ft.in, ft.want)
		}
	}
}
