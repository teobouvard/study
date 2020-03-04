package netlayer

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"

	"github.com/dat520-2020/TeamPilots/lab4/util"
	"gopkg.in/yaml.v2"
)

// YAMLNode : helper struct to unmarshal YAML node
type YAMLNode struct {
	Hostname string
	Port     int
	ID       int
}

// YAMLConfig : helper struct to unmarshal YAML config file
type YAMLConfig struct {
	Servers []YAMLNode
	Clients []YAMLNode
}

// Config is the network configuration as specified by the config file
type Config struct {
	serverIDs []int
	servers   map[int]*net.UDPAddr
	clients   map[int]*net.UDPAddr
}

// NewConfig reads a config file and contains the nodes in it
func NewConfig(configFile string) *Config {
	var yamlConf YAMLConfig
	file, err := ioutil.ReadFile(configFile)
	util.Check(err)
	err = yaml.Unmarshal(file, &yamlConf)
	util.Check(err)

	clients := make(map[int]*net.UDPAddr)
	servers := make(map[int]*net.UDPAddr)
	serverIDs := []int{}

	for _, node := range yamlConf.Servers {
		fmt.Fprintf(os.Stderr, "[\033[32;1m CONFIG \033[0m] Server [%d] @ %v:%v\n", node.ID, node.Hostname, node.Port)
		addr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("%s:%d", node.Hostname, node.Port))
		util.Check(err)
		servers[node.ID] = addr
		serverIDs = append(serverIDs, node.ID)
	}

	for _, node := range yamlConf.Clients {
		fmt.Fprintf(os.Stderr, "[\033[32;1m CONFIG \033[0m] Client [%d] @ %v:%v\n", node.ID, node.Hostname, node.Port)
		addr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("%s:%d", node.Hostname, node.Port))
		util.Check(err)
		clients[node.ID] = addr
	}

	return &Config{
		servers:   servers,
		clients:   clients,
		serverIDs: serverIDs,
	}
}

// Contains returns true if the config file has a node with the specified id
func (c *Config) Contains(id int) bool {
	if _, ok := c.servers[id]; ok {
		return true
	}
	if _, ok := c.clients[id]; ok {
		return true
	}
	return false
}

// AddrOf returns the UDP address mapped to the specified id
func (c *Config) AddrOf(id int) *net.UDPAddr {
	if !c.Contains(id) {
		util.Raise("id lookup failed because requested id is not in the config")
	}
	if addr, ok := c.servers[id]; ok {
		return addr
	}
	return c.clients[id]
}

// ServerIDs is an accessor to ids of server nodes
func (c *Config) ServerIDs() []int {
	return c.serverIDs
}
