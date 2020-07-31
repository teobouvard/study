package util

import (
	"fmt"
	"os"
)

// Check adds strict error handling, any error crashes the program
func Check(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "[\033[31;1m ERROR \033[0m] %v\n", err)
		panic(err)
	}
}

// Raise implements basic exception raising
func Raise(message string) {
	fmt.Fprintf(os.Stderr, "[\033[31;1m ERROR \033[0m] %v\n", message)
	os.Exit(0)
}
