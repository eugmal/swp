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
func biases_init(anzahl_neuronen_akt_layer int) mat.Dense {
	data := make([]float64, (anzahl_neuronen_akt_layer))
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	a := mat.NewDense(1, anzahl_neuronen_akt_layer, data)
	return *a

}

/*
>> func createLayerOutput <<
berechnet den output des ersten hiddenLayers
da x (die matrix mit dem inputbild) die dimensionen 28*28,1 hat muss Transponiert werden. bei den Parameter anzahl_nodes_des_hiddenlayer muss die Anzahl der Nodes des layers in den die daten reingehen angegeben werden. Der Output ist eine Matrix mit den Dimensionen 1, Anzahl der Nodes.(Nodes sind also gewissermassen die Spalten). Damit sind die Output-Daten als input fuer den naechsten Layer schon in der richtigen Form und es muss dann nichtmehr transponiert werden.
*/
func create1HiddenLayerOutput(input, weights_matrix, bias_matrix *mat.Dense, anzahl_nodes_des_hiddenLayer int) *mat.Dense {
	// Matritzen-Multiplikation der Input-Matrix mit der Weights_Matrix des aktuellen Layers.
	output := mat.NewDense(1, anzahl_nodes_des_hiddenLayer, nil)
	output.Mul(input.T(), weights_matrix)
	//der entsprechende Bias wird zu den Ergebnissen addiert.
	output_bias := mat.NewDense(1, anzahl_nodes_des_hiddenLayer, nil)
	output_bias.Add(bias_matrix, output)

	//hilfsfunktion fuer die Apply-Methode
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	//sigmoid wird auf jedes Ergebnis angewandt.
	output_sigmoid := mat.NewDense(1, anzahl_nodes_des_hiddenLayer, nil)
	//wendet die sigmoid-Funktion auf jedes Ergebnis der Multiplikation an.
	output_sigmoid.Apply(applySigmoid, output_bias)
	return output_sigmoid
}

// im Prinzip das gleiche wie create1HiddenLayerOutput aber ohne den Transpose der Inputdaten
func createGeneralLayerOutput(input, weights_matrix, bias_matrix *mat.Dense, anzahl_nodes_des_hiddenLayer int) *mat.Dense {
	// Matritzen-Multiplikation der Input-Matrix mit der Weights_Matrix des aktuellen Layers.
	output := mat.NewDense(1, anzahl_nodes_des_hiddenLayer, nil)
	output.Mul(input, weights_matrix)
	//der entsprechende Bias wird zu den Ergebnissen addiert.
	output_bias := mat.NewDense(1, anzahl_nodes_des_hiddenLayer, nil)
	output_bias.Add(bias_matrix, output)

	//hilfsfunktion fuer die Apply-Methode
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	//sigmoid wird auf jedes Ergebnis angewandt.
	output_sigmoid := mat.NewDense(1, anzahl_nodes_des_hiddenLayer, nil)
	//wendet die sigmoid-Funktion auf jedes Ergebnis der Multiplikation an.
	output_sigmoid.Apply(applySigmoid, output_bias)
	return output_sigmoid
}
func durchlaufFuerEinBild(input, w1, w2, w3, b1, b2, b3 *mat.Dense) mat.Dense {
	outputLayerH1 := create1HiddenLayerOutput(input, w1, b1, 16)
	outputLayerH2 := createGeneralLayerOutput(outputLayerH1, w2, b2, 16)
	finalOutput := createGeneralLayerOutput(outputLayerH2, w3, b3, 10)
	return *finalOutput

}

func durchlaufFuerDatensatz(datensatz mnistLoad.Datensatz, w1, w2, w3, b1, b2, b3 *mat.Dense) {
	for i := 0; i < len(datensatz.Bilder); i++ {
		aktuellesBild := datensatz.Bilder[i]
		dataBild := make([]float64, len(aktuellesBild))
		for i := range aktuellesBild {
			dataBild[i] = aktuellesBild[i]
		}
		bildMatrix := mat.NewDense(28*28, 1, dataBild)
		output := durchlaufFuerEinBild(bildMatrix, w1, w2, w3, b1, b2, b3)
		fmt.Println(output)
		fmt.Println("Anzahl der durchlaeufe:")
		fmt.Println(i)
	}
}

/*
to do:

funktion, die den ganzen durchlauf von bild-input bis final-Output macht.
funktion zum berechnen des Fehlers
backpropagation zum updaten der weights und biases

funktion, die diese Schritte vereint und nacheinander fuer jedes Bild durchlaeuft.
output berechnen --> fehler berechnen --> weights und biases anpassen --> fuer naechstes Bild mit den angepassten weights und biases den Vorgang wiederholen

---- eventuell nicht fuer einzelne Bilder, sondern Minibatches......
---- loss funktion muss gewaehlt werden. (wahrscheinlich mean spuare)
---- ableitungen und Steigungen berechnen fuer backpropagation

am Ende muss das trainierte Netz gespeichert werden.
Dann anhand dieses gespeicherten Netzes und den Testdaten die genauigkeit des Netzes messen.
*/

//funktion zum generieren der LabelMatrix. der wert des Labels im entsprechenden Index des datensatzes.Label ist der Index der Outputmatrix dessen Wert auf 1 gesetzt wird. Alle anderen //Werte sind 0.
//func generiereLabelMatrix(labelDaten mnistLoad.Datensatz, index int) mat.Dense{
//	}

//loss funktion "mean squared error"
//func meanSquaredError(outputMatrix, labelMatrix mat.Dense ) float64{
//}

func main() {
	//--------------------------------------------------------------
	// bestimmen der Layer-Variablen
	var hidden1 hidden_layer
	var hidden2 hidden_layer
	var output_final hidden_layer

	//--------------------------------------------------------------

	// initialisieren der Layer weights und biases
	hidden1.neuronen = 16
	hidden1.weights_matrix = weights_init(28*28, 16)
	hidden1.biases = biases_init(16)

	hidden2.neuronen = 16
	hidden2.weights_matrix = weights_init(16, 16)
	hidden2.biases = biases_init(16)

	output_final.neuronen = 10
	output_final.weights_matrix = weights_init(16, 10)
	output_final.biases = biases_init(10)

	// -------------------------------------------------------------

	// laden der datensets
	trainingData, _ := mnistLoad.Laden("/Users/eugen/coding/lwb/mnistzip")

	durchlaufFuerDatensatz(trainingData, &hidden1.weights_matrix, &hidden2.weights_matrix, &output_final.weights_matrix, &hidden1.biases, &hidden2.biases, &output_final.biases)

	// --------------------------------------------------------------

}
