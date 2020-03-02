package app

import (
	"io/ioutil"

	"gopkg.in/yaml.v2"
)

// Node represents a node in the network
type Node struct {
	Hostname string
	Port     int
	ID       int
}

// Config is a list of nodes
type Config struct {
	Nodes []Node
}

// Parse reads a config file and returns the list of nodes it contains
func Parse(config string) []Node {
	var c Config
	file, _ := ioutil.ReadFile(config)
	err := yaml.Unmarshal(file, &c)
	Check(err)
	return c.Nodes
}
