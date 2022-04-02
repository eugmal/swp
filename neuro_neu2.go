package main

import (
	"fmt"

	"github.com/eugmal/swp/mnistLoad"

	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func zufallszahl() int {
	rand.Seed(time.Now().UnixNano())
	min := 0
	max := 5999
	return (rand.Intn(max-min+1) + min)
}
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

type hidden_layer struct {
	weights_matrix mat.Dense
	biases         mat.Dense
	neuronen       int
}

func weights_init(anzahl_input_neuronen, anzahl_neuronen_akt_layer int) mat.Dense {
	data := make([]float64, (anzahl_input_neuronen * anzahl_neuronen_akt_layer))
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	a := mat.NewDense(anzahl_input_neuronen, anzahl_neuronen_akt_layer, data)
	return *a
}

//zufallswerte der biases initialisieren
func biases_init(anzahl_neuronen_akt_layer int) mat.Dense {
	data := make([]float64, (anzahl_neuronen_akt_layer))
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	a := mat.NewDense(anzahl_neuronen_akt_layer, 1, data)
	return *a

}

/*
func mul (passenderen Namen waehlen!)
berechnet den output des ersten hiddenLayers (hier nur beispielhaft. Es fehlt bias)
da x (die matrix mit dem inputbild) die dimensionen 28*28,1 hat muss Transponiert werden. bei den Parameter anzahl_nodes_des_hiddenlayer muss die Anzahl der Nodes des layers in den die daten reingehen angegeben werden. Der Output ist eine Matrix mit den Dimensionen 1, Anzahl der Nodes.(Nodes sind also gewissermassen die Spalten). Damit sind die Output-Daten als input fuer den naechsten Layer schon in der richtigen Form und es muss dann nichtmehr transponiert werden.
*/
func mul(x, y *mat.Dense, anzahl_nodes_des_hiddenLayer int) *mat.Dense {
	output := mat.NewDense(1, anzahl_nodes_des_hiddenLayer, nil)
	output.Mul(x.T(), y)
	//hilfsfunktion fuer die Apply-Methode
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	output_sigmoid := mat.NewDense(1, anzahl_nodes_des_hiddenLayer, nil)
	//wendet die sigmoid-Funktion auf jedes Ergebnis der Multiplikation an.
	output_sigmoid.Apply(applySigmoid, output)
	return output_sigmoid
}

func main() {

	var hidden1 hidden_layer
	hidden1.neuronen = 16
	hidden1.weights_matrix = weights_init(28*28, 16)
	trainingData, _ := mnistLoad.Laden("/Users/eugen/coding/lwb/mnistzip")
	fmt.Println(trainingData.Bilder[1])
	//fmt.Println(trainingData.Labels[1])
	einBild_Bild := trainingData.Bilder[1]
	dataBild := make([]float64, len(einBild_Bild))
	for i := range einBild_Bild {
		dataBild[i] = einBild_Bild[i]
	}

	einBild := mat.NewDense(28*28, 1, dataBild)
	fmt.Println(einBild)
	//hiddenMatrix := mat.NewDense(28*28, 16, nil)

	fmt.Println(mul(einBild, &hidden1.weights_matrix, 16))
}
