package ai

import (
	"fmt"
	"math"

	"github.com/eugmal/swp/mnistLoad"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

//hilfsfunktionen matrix-manipulation

func Dot(m, n mat.Matrix) *mat.Dense {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func Apply(fn func(i, j int, v float64) float64, m mat.Dense) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, &m)
	return o
}

func Scale(s float64, m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

func Multiply(m, n *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func Add(m, n *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func Substract(m, n mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

func RandomArray(size int, v float64) (data []float64) {
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

func MatrixPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

//////////////////////////////////////////
// Aktivierungsfunktionen und derivate
func Sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func SigmoidPrime(m *mat.Dense) *mat.Dense {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return Multiply(m, Substract(ones, m))
}

/////////////////////////////////////////

func GibBildundLabel(datensatz mnistLoad.DatensatzSlices, index int) (*mat.Dense, *mat.Dense) {
	bild := datensatz.Bilder[index]
	bildMatrix := mat.NewDense(784, 1, bild)
	labelSlice := make([]float64, 10)
	labelSlice[int(datensatz.Labels[index])] = 1
	labelMatrix := mat.NewDense(10, 1, labelSlice)

	return bildMatrix, labelMatrix

}

//erstmal nur Gewichte. Das Netzwerk arbeitet noch ohne Biases
func WeigthsUndBiasesInitialisieren() (mat.Dense, mat.Dense) {
	w1Slice := RandomArray(20*784, 1)
	w2Slice := RandomArray(10*20, 1)

	w1 := mat.NewDense(20, 784, w1Slice)
	w2 := mat.NewDense(10, 20, w2Slice)

	return *w1, *w2
}

//////////////////////////////

func Forward(x, w1, w2 mat.Dense) (mat.Dense, mat.Dense, mat.Dense, mat.Dense) {
	z1 := Dot(&w1, &x)
	a1 := Apply(Sigmoid, *z1)
	z2 := Dot(&w2, a1)
	a2 := Apply(Sigmoid, *z2)

	return *z1, *a1, *z2, *a2

}

func Backwards(z1, a1, z2, a2, w2, x, y mat.Dense) (mat.Dense, mat.Dense) {

	errorFinal := Substract(&y, &a2)
	errorHidden := Dot(w2.T(), errorFinal)

	primea2 := SigmoidPrime(&a2)
	primeMulFinal := Multiply(&a2, primea2)
	slopeOutput := Multiply(errorFinal, primeMulFinal)
	dW2 := Dot(slopeOutput, a1.T())

	primea1 := SigmoidPrime(&a1)
	primeMulHidden := Multiply(&a1, primea1)
	slopeHidden := Multiply(errorHidden, primeMulHidden)
	dW1 := Dot(slopeHidden, x.T())

	return *dW1, *dW2

}

func Update(w1, w2, dW1, dW2 mat.Dense, alpha float64) (mat.Dense, mat.Dense) {

	dW1.Scale(alpha, &dW1)

	dW2.Scale(alpha, &dW2)

	w1Neu := Add(&w1, &dW1)

	w2Neu := Add(&w2, &dW2)

	return *w1Neu, *w2Neu
}

func GradientDescendFull(dataset mnistLoad.DatensatzSlices, epochs int, alpha float64) (*mat.Dense, *mat.Dense) {
	w1, w2 := WeigthsUndBiasesInitialisieren()

	for i := 0; i < epochs; i++ {
		for j := 0; j < len(dataset.Bilder); j++ {
			bild, label := GibBildundLabel(dataset, j)
			z1, a1, z2, a2 := Forward(*bild, w1, w2)

			dW1, dW2 := Backwards(z1, a1, z2, a2, w2, *bild, *label)
			w1, w2 = Update(w1, w2, dW1, dW2, alpha)

		}
	}
	return &w1, &w2
}

func Testing(testSet mnistLoad.DatensatzSlices, w1, w2 mat.Dense) (int, float64) {
	richtigCounter := 0
	for i := 0; i < len(testSet.Bilder); i++ {
		bild, _ := GibBildundLabel(testSet, i)
		_, _, _, vorhersage := Forward(*bild, w1, w2)
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

	accuracy := (float64(richtigCounter) / float64(len(testSet.Bilder)) * 100.0)

	return richtigCounter, accuracy
}
