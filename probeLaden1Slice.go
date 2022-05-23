package main

import (
	"fmt"

	"github.com/eugmal/swp/mnistLoad"
)

func main() {
	var path = "/Users/eugen/coding/lwb/mnistzip"
	trainset, testset := mnistLoad.Laden1Slice(path)

	fmt.Println("train bilder: ", trainset.Bilder)
	fmt.Println("train Labels: ", trainset.Labels)
	fmt.Println("test Bilder: ", testset.Bilder)
	fmt.Println("test Labels: ", testset.Labels)
	fmt.Println("len train bilder: ", len(trainset.Bilder))
	fmt.Println("len train Labels: ", len(trainset.Labels))
	fmt.Println("len test Bilder: ", len(testset.Bilder))
	fmt.Println("len test Labels: ", len(testset.Labels))

}
