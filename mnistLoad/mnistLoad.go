package mnistLoad

import (
	"github.com/moverest/mnist"
)

const weite = 28
const hoehe = 28

type Bild [weite * hoehe]float64
type Datensatz struct {
	Bilder []Bild
	Labels []mnist.Label
}

func Laden(path string) (Datensatz, Datensatz) {
	var trainingsDatensatz Datensatz
	var testDatensatz Datensatz
	trainingsDatensatz.Bilder = make([]Bild, 60000)
	testDatensatz.Bilder = make([]Bild, 10000)
	train, test, _ := mnist.Load(path)

	for i := 0; i < len(train.Images); i++ {
		neuesBild := train.Images[i]
		for j := 0; j < (weite * hoehe); j++ {
			trainingsDatensatz.Bilder[i][j] = (float64(neuesBild[j]) / 255.)

		}

	}

	for i := 0; i < len(test.Images); i++ {
		neuesBild := test.Images[i]
		for j := 0; j < (weite * hoehe); j++ {
			testDatensatz.Bilder[i][j] = (float64(neuesBild[j]) / 255.)

		}

	}

	trainingsDatensatz.Labels = train.Labels
	testDatensatz.Labels = test.Labels

	return trainingsDatensatz, testDatensatz

}