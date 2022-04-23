package main

import (
	"fmt"

	"github.com/eugmal/swp/ai"
	"github.com/eugmal/swp/mnistLoad"
)

func main() {
	var path = "/Users/eugen/coding/lwb/mnistzip"
	_, testSet := mnistLoad.LadenSlice(path)
	w1, w2 := ai.GewichteLaden("/Users/eugen/coding/lwb/swpnew/swp/gespeichertegewichte/01")
	richtige, acc := ai.Testing(testSet, *w1, *w2)
	fmt.Println("richtige: ", richtige)
	fmt.Println("acc: ", acc)
}
