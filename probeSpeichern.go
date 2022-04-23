package main

import (
	"github.com/eugmal/swp/ai"
	"github.com/eugmal/swp/mnistLoad"
)

func main() {
	var path = "/Users/eugen/coding/lwb/mnistzip"
	trainSet, _ := mnistLoad.LadenSlice(path)
	w1neu, w2neu := ai.GradientDescendFull(trainSet, 5, 0.1)
	ai.GewichteSpeichern(*w1neu, *w2neu, "/Users/eugen/coding/lwb/swpnew/swp/gespeichertegewichte/01")

}
