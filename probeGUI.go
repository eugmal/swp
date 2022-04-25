package main

import (
	"fmt"

	"github.com/eugmal/swp/ai"

	"github.com/eugmal/swp/gui"
)

func main() {
	w1, w2 := ai.GewichteLaden("/home/pi/Desktop/gewichte/01")
	for i := 0; i < 10; i++ {
		eingabeBild := gui.ZahlMalen()
		antwort := ai.Vorhersage(*eingabeBild, *w1, *w2)
		fmt.Println("Die gezeichnete Zahl ist eine: ", antwort)
	}
}
