package main

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/eugmal/swp/mnistLoad"

	"math"

	"time"

	"github.com/moverest/mnist"
	"gonum.org/v1/gonum/floats"
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
	//bild muss wahrscheinlich zur multiplikation in einer Zeile abgebildet sein. Aber nicht ganz sicher. Ist ein Versuch.
	a.T()
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

func sigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}

//prediction fuer ein bild
func durchlauf_fuer_ein_bild(input_daten mnistLoad.Datensatz, index, inputNeuronen, h1Neuronen, h2Neuronen, outNeuronen int, w1, w2, w3 mat.Dense, b1, b2, b3 mat.Dense) (mat.Dense, mat.Dense, mat.Dense) {
	input_vektor_einzelbild := bild_in_inputLayer(input_daten, index)
	out_HiddenLayer1 := layer_output_berechnen(h1Neuronen, w1, input_vektor_einzelbild, b1)
	out_HiddenLayer2 := layer_output_berechnen(h2Neuronen, w2, out_HiddenLayer1, b2)
	final_out := layer_output_berechnen(outNeuronen, w3, out_HiddenLayer2, b3)
	return final_out, out_HiddenLayer1, out_HiddenLayer2
}

func fehler_berechnen(label mnistLoad.Datensatz, OutputVektor mat.Dense, index int) mat.Dense {
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

//  backpropagation. Rueckgabe der neuen weights und biases nach einem durchlauf
func backpropagation(input_daten mnistLoad.Datensatz, index, inputNeuronen, h1Neuronen, h2Neuronen, outNeuronen int, w1, w2, w3 mat.Dense, b1, b2, b3 mat.Dense, learningrate float64) {
	output_ein_durchlauf, hidden1output, hidden2output := durchlauf_fuer_ein_bild(input_daten, index, inputNeuronen, h1Neuronen, h2Neuronen, outNeuronen, w1, w2, w3, b1, b2, b3)
	fehler_vektor_aktueller_durchlauf := fehler_berechnen(input_daten, output_ein_durchlauf, index)
	applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
	slopeOutputLayer := mat.NewDense(0, 0, nil)
	slopeOutputLayer.Apply(applySigmoidPrime, &output_ein_durchlauf)
	slopeHidden2Layer := mat.NewDense(0, 0, nil)
	slopeHidden2Layer.Apply(applySigmoidPrime, &hidden2output)
	slopeHidden1Layer := mat.NewDense(0, 0, nil)
	slopeHidden1Layer.Apply(applySigmoidPrime, &hidden1output)

	dOutput := mat.NewDense(0, 0, nil)
	dOutput.MulElem(&fehler_vektor_aktueller_durchlauf, slopeOutputLayer)

	errorAtH2Layer := mat.NewDense(0, 0, nil)
	errorAtH2Layer.Mul(dOutput, w3.T())

	dHidden2Layer := mat.NewDense(0, 0, nil)
	dHidden2Layer.MulElem(errorAtH2Layer, slopeHidden2Layer)

	errorAtH1Layer := mat.NewDense(0, 0, nil)
	errorAtH1Layer.Mul(dHidden2Layer, w2.T())

	dHidden1Layer := mat.NewDense(0, 0, nil)
	dHidden1Layer.MulElem(errorAtH1Layer, slopeHidden1Layer)

	// anpassen der parameter w und b
	wOutAdj := mat.NewDense(0, 0, nil)
	wOutAdj.Mul(hidden2output.T(), dOutput)
	wOutAdj.Scale(learningrate, wOutAdj)
	w3.Add(&w3, wOutAdj)

	bOutAdj, err := sumAlongAxis(0, dOutput)
	if err != nil {
		return
	}
	bOutAdj.Scale(learningrate, bOutAdj)
	b3.Add(&b3, bOutAdj)

	wHidden2Adj := mat.NewDense(0, 0, nil)
	wHidden2Adj.Mul(hidden1output.T(), dHidden2Layer)
	wHidden2Adj.Scale(learningrate, wHidden2Adj)
	w2.Add(&w2, wHidden2Adj)

	bHidden2Adj, err := sumAlongAxis(0, dHidden2Layer)
	if err != nil {
		return
	}
	bHidden2Adj.Scale(learningrate, bHidden2Adj)
	b2.Add(&b2, bHidden2Adj)

	output_vom_input_layer := bild_in_inputLayer(input_daten, index)
	wHidden1Adj := mat.NewDense(0, 0, nil)
	wHidden1Adj.Mul(output_vom_input_layer.T(), dHidden1Layer)
	wHidden1Adj.Scale(learningrate, wHidden1Adj)
	w1.Add(&w1, wHidden1Adj)

	bHidden1Adj, err := sumAlongAxis(0, dHidden1Layer)
	if err != nil {
		return
	}
	bHidden1Adj.Scale(learningrate, bHidden1Adj)
	b1.Add(&b1, bHidden1Adj)

	//return w1,w2,w3,b1,b2,b3

}

func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()

	var output *mat.Dense
	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}
	return output, nil
}

// hier sollen die schritte zusammengefuehrt werden um das Netz zu trainieren

func trainieren(input_daten mnistLoad.Datensatz, inputNeuronen, h1Neuronen, h2Neuronen, outNeuronen int, w1, w2, w3 mat.Dense, b1, b2, b3 mat.Dense, learningrate float64) {
	//erstmal eine Zahl(20000) fuer die Menge der Durchlaeufe. Spaeter aendern.
	for i := 0; i < 20000; i++ {
		backpropagation(input_daten, i, inputNeuronen, h1Neuronen, h2Neuronen, outNeuronen, w1, w2, w3, b1, b2, b3, learningrate)

	}

}

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
	fmt.Println("Weights hidden1 vor Training: ")
	fmt.Println(hidden1.weights_matrix)
	trainieren(trainingSet, input.neuronen, hidden1.neuronen, hidden2.neuronen, output.neuronen, hidden1.weights_matrix, hidden2.weights_matrix, output.weights_matrix, hidden1.biases, hidden2.biases, output.biases, 0.2)
	fmt.Println("weights hidden1 nach training:")
	fmt.Println(hidden1.weights_matrix)

}
