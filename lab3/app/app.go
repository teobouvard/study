package app

import (
	"fmt"
	"io/ioutil"

	"../detector"
	"gopkg.in/yaml.v2"
)

type Node struct {
	Hostname string
	Port int
	ID int
}

type Config struct {
	Nodes []Node
}


type App struct {
	id int

	fd *detector.EvtFailureDetector
	ld *detector.MonLeaderDetector
}

func NewApp(id int, config string) *App{
	var c Config
	file, _ := ioutil.ReadFile(config)
	err := yaml.Unmarshal(file, &c)
	check(err)
	fmt.Println(c)
	return &App{
		id: id,
	}
}

func (app *App) Run(){
	fmt.Println("Running!")
}

func check(err error){
	if err != nil {
		fmt.Print(err)
	}
}