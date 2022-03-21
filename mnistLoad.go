package main

import (
	"fmt"

	"github.com/moverest/mnist"
)

func main() {
	train, test, _ := mnist.Load("/Users/eugen/coding/lwb/mnistzip")
	fmt.Println("Es folgt ein trainingsbild: ")
	fmt.Println(train.Images[1])
	fmt.Println("Es folgt das entsprechende Label: ")
	fmt.Println(train.Labels[1])
	fmt.Println("Es folgt ein testbild: ")
	fmt.Println(test.Images[1])
	fmt.Println("Es folgt das entsprechende Label: ")
	fmt.Println(test.Labels[1])
	fmt.Println("Es gibt insgesamt trainingsbilder:")
	fmt.Println(len(train.Images))
	fmt.Println("Es gibt insgesamt entsprechende Labels:")
	fmt.Println(len(train.Labels))

}
