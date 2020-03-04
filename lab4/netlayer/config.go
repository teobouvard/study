package netlayer

import (
	"io/ioutil"

	"github.com/dat520-2020/TeamPilots/lab4/util"
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
	file, err := ioutil.ReadFile(config)
	util.Check(err)
	err = yaml.Unmarshal(file, &c)
	util.Check(err)
	return c.Nodes
}
