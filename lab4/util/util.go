package util

import (
	"fmt"
	"os"
)

// Check adds strict error handling, any error crashes the program
func Check(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "[\033[31;1m ERROR \033[0m] %v\n", err)
		os.Exit(0)
	}
}

// Raise implements basic exception raising
func Raise(message string) {
	fmt.Fprintf(os.Stderr, "[\033[31;1m ERROR \033[0m] %v\n", message)
	os.Exit(0)
}

// Contains checks if an element is present in an array
func Contains(arr []int, x int) bool {
	for _, e := range arr {
		if e == x {
			return true
		}
	}
	return false
}
