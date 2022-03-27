package main

import (
	"fmt"
	"math/rand"
	"neuronetz/mnistLoad"

	"math"

	"time"

	"github.com/moverest/mnist"
	"gonum.org/v1/gonum/mat"
)

type input_layer struct {
	input_bild mat.Dense
	//data           []float64
	//weights_matrix mat.Dense
	//biases         mat.Dense
	neuronen int
}

type hidden_layer struct {
	weights_matrix mat.Dense
	biases         mat.Dense
	neuronen       int
}

type output_layer struct {
	//ergebnisse     mat.Dense
	weights_matrix mat.Dense
	biases         mat.Dense
	neuronen       int
}

func bild_in_inputLayer(input_daten mnistLoad.Datensatz, index int) mat.Dense {
	data := make([]float64, (len(input_daten.Bilder[index])))
	for i := range data {
		data[i] = input_daten.Bilder[index][i]
	}
	a := mat.NewDense((len(input_daten.Bilder[index])), 1, data)
	return *a
}

//zufallswerte der gewichtungen initialisieren
func weights_init(anzahl_neuronen_akt_layer, anzahl_input_neuronen int) mat.Dense {
	data := make([]float64, (anzahl_input_neuronen * anzahl_neuronen_akt_layer))
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	a := mat.NewDense(anzahl_neuronen_akt_layer, anzahl_input_neuronen, data)
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
func output_init(anzahl_neuronen_akt_layer int) mat.Dense {
	data := make([]float64, (anzahl_neuronen_akt_layer))
	a := mat.NewDense(anzahl_neuronen_akt_layer, 1, data)
	return *a
}
func layer_output_berechnen(anzahl_neuronen int, weight_matrix mat.Dense, input_vektor mat.Dense, bias_vektor mat.Dense) mat.Dense {
	actual := make([]float64, anzahl_neuronen)
	out := mat.NewDense(anzahl_neuronen, 1, actual)
	out.Mul(&weight_matrix, &input_vektor)
	out.Add(out, &bias_vektor)
	actual2 := make([]float64, anzahl_neuronen)
	for i := range actual2 {
		actual2[i] = sigmoid_funktion(out.At(i, 0))
	}
	sigmoidOut := mat.NewDense(anzahl_neuronen, 1, actual2)
	return *sigmoidOut
}

//sigmoid funktion
func sigmoid_funktion(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// um bei backpropagation die sigmoid aktivierung rauszurechnen
func sigmoid_prime(x float64) float64 {
	return sigmoid_funktion(x) * (1.0 - sigmoid_funktion(x))
}

//prediction fuer ein bild
func durchlauf_fuer_ein_bild(input_daten mnistLoad.Datensatz, index, inputNeuronen, h1Neuronen, h2Neuronen, outNeuronen int, w1, w2, w3 mat.Dense, b1, b2, b3 mat.Dense) (mat.Dense, mat.Dense, mat.Dense) {
	input_vektor_einzelbild := bild_in_inputLayer(input_daten, index)
	out_HiddenLayer1 := layer_output_berechnen(h1Neuronen, w1, input_vektor_einzelbild, b1)
	out_HiddenLayer2 := layer_output_berechnen(h2Neuronen, w2, out_HiddenLayer1, b2)
	final_out := layer_output_berechnen(outNeuronen, w3, out_HiddenLayer2, b3)
	return final_out, out_HiddenLayer1, out_HiddenLayer2
}

func fehler_berechnen(label mnistLoad.Datensatz, OutputVektor mat.VecDense, index int) mat.Dense {
	akt_label := label.Labels[index]
	var richtige_antwort [10]float64
	richtige_antwort[akt_label] = 1
	richtige_antwort_vektor := mat.NewDense(10, 1, richtige_antwort[:])
	var fehler_vektor mat.Dense
	fehler_vektor.Sub(&OutputVektor, richtige_antwort_vektor)
	return fehler_vektor

}

func mini_batch(datensatz mnistLoad.Datensatz) mnistLoad.Datensatz {
	var miniBatch mnistLoad.Datensatz
	miniBatch.Bilder = make([]mnistLoad.Bild, 100)
	miniBatch.Labels = make([]mnist.Label, 100)
	for i := 0; i < 100; i++ {
		zufalls_index := zufallszahl()
		miniBatch.Bilder[i] = datensatz.Bilder[zufalls_index]
		miniBatch.Labels[i] = datensatz.Labels[zufalls_index]
	}
	return miniBatch

}

func zufallszahl() int {
	rand.Seed(time.Now().UnixNano())
	min := 0
	max := 5999
	return (rand.Intn(max-min+1) + min)
}

/*
func trainieren_backpropagation(input_daten mnistLoad.Datensatz, index, inputNeuronen, h1Neuronen, h2Neuronen, outNeuronen int, w1, w2, w3 mat.Dense, b1, b2, b3 mat.Dense, learningrate float64) {
	output_ein_durchlauf,hidden1output, hidden2output := durchlauf_fuer_ein_bild(input_daten, index, inputNeuronen, h1Neuronen, h2Neuronen, outNeuronen, w1, w2, w3, b1, b2, b3)
	fehler_vektor_aktueller_durchlauf := fehler_berechnen(input_daten, output_ein_durchlauf, index)
	var c mat.Dense
	dw2 := 1/ learningrate * c.Mul(&fehler_vektor_aktueller_durchlauf,hidden2output)
	db2 := 1/ learning_rate
}

func update_parameter(w1, w2, w3, dw1, dw2, dw3 mat.Dense, b1, b2, b3, db1, db2, db3 mat.VecDense, learningrate float64) (mat.Dense, mat.Dense, mat.Dense, mat.VecDense, mat.VecDense, mat.VecDense) {

}
*/
func main() {
	var path = "/Users/eugen/coding/lwb/mnistzip"
	var trainingSet, _ mnistLoad.Datensatz = mnistLoad.Laden(path)
	//var learning_rate float64
	var input input_layer
	var hidden1 hidden_layer
	var hidden2 hidden_layer
	var output output_layer
	//var index int = 1
	// Layer initialisieren
	input.neuronen = 28 * 28
	hidden1.neuronen = 16
	hidden2.neuronen = 16
	output.neuronen = 10

	hidden1.weights_matrix = weights_init(hidden1.neuronen, input.neuronen)
	hidden1.biases = biases_init(hidden1.neuronen)

	hidden2.weights_matrix = weights_init(hidden2.neuronen, hidden1.neuronen)
	hidden2.biases = biases_init(hidden2.neuronen)

	//output.ergebnisse = output_init(output.neuronen)
	output.weights_matrix = weights_init(output.neuronen, hidden2.neuronen)
	output.biases = biases_init(output.neuronen)

	//input.input_bild = bild_in_inputLayer(trainingSet, 1)
	//input_fuer_fehler := durchlauf_fuer_ein_bild(trainingSet, index, input.neuronen, hidden1.neuronen, hidden2.neuronen, output.neuronen, hidden1.weights_matrix, hidden2.
	//weights_matrix, output.weights_matrix, hidden1.biases, hidden2.biases, output.biases)
	//fmt.Println(fehler_berechnen(trainingSet, input_fuer_fehler, index))
	miniBatch := mini_batch(trainingSet)
	fmt.Println(miniBatch)

}
