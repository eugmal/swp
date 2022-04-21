package main

import (
	"fmt"
	"math"

	"github.com/eugmal/swp/mnistLoad"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

//hilfsfunktionen matrix-manipulation

func dot(m, n mat.Matrix) *mat.Dense {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func apply(fn func(i, j int, v float64) float64, m mat.Dense) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, &m)
	return o
}

func scale(s float64, m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

func multiply(m, n *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func add(m, n *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func substract(m, n mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}
	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return
}

func matrixPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

//////////////////////////////////////////
// Aktivierungsfunktionen und derivate
func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func sigmoidPrime(m *mat.Dense) *mat.Dense {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(m, substract(ones, m))
}

/////////////////////////////////////////

func gibBildundLabel(datensatz mnistLoad.DatensatzSlices, index int) (*mat.Dense, *mat.Dense) {
	bild := datensatz.Bilder[index]
	bildMatrix := mat.NewDense(784, 1, bild)
	labelSlice := make([]float64, 10)
	labelSlice[int(datensatz.Labels[index])] = 1
	labelMatrix := mat.NewDense(10, 1, labelSlice)

	return bildMatrix, labelMatrix

}

//erstmal nur Gewichte. Das Netzwerk arbeitet noch ohne Biases
func weigthsUndBiasesInitialisieren() (mat.Dense, mat.Dense) {
	w1Slice := randomArray(20*784, 1)
	w2Slice := randomArray(10*20, 1)

	w1 := mat.NewDense(20, 784, w1Slice)
	w2 := mat.NewDense(10, 20, w2Slice)

	return *w1, *w2
}

//////////////////////////////

func forward(x, w1, w2 mat.Dense) (mat.Dense, mat.Dense, mat.Dense, mat.Dense) {
	z1 := dot(&w1, &x)
	a1 := apply(sigmoid, *z1)
	z2 := dot(&w2, a1)
	a2 := apply(sigmoid, *z2)

	return *z1, *a1, *z2, *a2

}

func backwards(z1, a1, z2, a2, w2, x, y mat.Dense) (mat.Dense, mat.Dense) {

	errorFinal := substract(&y, &a2)
	errorHidden := dot(w2.T(), errorFinal)

	primea2 := sigmoidPrime(&a2)
	primeMulFinal := multiply(&a2, primea2)
	slopeOutput := multiply(errorFinal, primeMulFinal)
	dW2 := dot(slopeOutput, a1.T())

	primea1 := sigmoidPrime(&a1)
	primeMulHidden := multiply(&a1, primea1)
	slopeHidden := multiply(errorHidden, primeMulHidden)
	dW1 := dot(slopeHidden, x.T())

	return *dW1, *dW2

}

func update(w1, w2, dW1, dW2 mat.Dense, alpha float64) (mat.Dense, mat.Dense) {

	dW1.Scale(alpha, &dW1)

	dW2.Scale(alpha, &dW2)

	w1Neu := add(&w1, &dW1)

	w2Neu := add(&w2, &dW2)

	return *w1Neu, *w2Neu
}

func gradientDescendFull(dataset mnistLoad.DatensatzSlices, epochs int, alpha float64) (*mat.Dense, *mat.Dense) {
	w1, w2 := weigthsUndBiasesInitialisieren()

	for i := 0; i < epochs; i++ {
		for j := 0; j < len(dataset.Bilder); j++ {
			bild, label := gibBildundLabel(dataset, j)
			z1, a1, z2, a2 := forward(*bild, w1, w2)

			dW1, dW2 := backwards(z1, a1, z2, a2, w2, *bild, *label)
			w1, w2 = update(w1, w2, dW1, dW2, alpha)

		}
	}
	return &w1, &w2
}

func testing(testSet mnistLoad.DatensatzSlices, w1, w2 mat.Dense) int {
	richtigCounter := 0
	for i := 0; i < len(testSet.Bilder); i++ {
		bild, _ := gibBildundLabel(testSet, i)
		_, _, _, vorhersage := forward(*bild, w1, w2)
		var vorhersageIndex int
		max := 0.0
		for j := 0; j < 10; j++ {
			if vorhersage.At(j, 0) >= max {
				max = vorhersage.At(j, 0)
				vorhersageIndex = j
			}

		}
		fmt.Println("Vorhersage: ", vorhersageIndex)
		fmt.Println("Label: ", testSet.Labels[i])
		if vorhersageIndex == int(testSet.Labels[i]) {
			richtigCounter = richtigCounter + 1
		}

	}
	fmt.Println("richtige: ", richtigCounter)
	accuracy := (float64(richtigCounter) / float64(len(testSet.Bilder)) * 100.0)
	fmt.Println("Accuracy: ", accuracy)
	return richtigCounter
}

func main() {
	var path = "/Users/eugen/coding/lwb/mnistzip"
	trainingset, testset := mnistLoad.LadenSlice(path)

	trainedW1, trainedw2 := gradientDescendFull(trainingset, 5, 0.1)
	testing(testset, *trainedW1, *trainedw2)

}
