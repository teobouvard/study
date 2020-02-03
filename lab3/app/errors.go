package app

import (
	"fmt"
	"os"
)

func Check(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "[\033[31;1m ERROR \033[0m] %v", err)
		os.Exit(0)
	}
}
