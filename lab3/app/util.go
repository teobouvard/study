package app

import (
	"fmt"
	"os"
)

// Check adds basic error display, should do better error checking
func Check(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "[\033[31;1m ERROR \033[0m] %v", err)
		os.Exit(0)
	}
}

// Raise implements basice exception raising
func Raise(message string) {
	fmt.Fprintf(os.Stderr, "[\033[31;1m ERROR \033[0m] %v", message)
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
