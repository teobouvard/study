package app

import (
	"io/ioutil"

	"gopkg.in/yaml.v2"
)

type Node struct {
	Hostname string
	Port     int
	ID       int
}

type Config struct {
	Nodes []Node
}

func Parse(config string) []Node {
	var c Config
	file, _ := ioutil.ReadFile(config)
	err := yaml.Unmarshal(file, &c)
	Check(err)
	return c.Nodes
}
