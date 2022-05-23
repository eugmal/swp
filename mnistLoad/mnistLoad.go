package mnistLoad

import (
	"github.com/moverest/mnist"
)

const weite = 28 //weite der MnistBilder
const hoehe = 28 //hoehe der MnistBilder

type Datensatz struct { // struct fuer Datentyp Datensatz, der Bilder und die entsprechenden Labels hält.
	Bilder [][]float64
	Labels []mnist.Label
}
type Datensatz1Slice struct {
	Bilder []float64
	Labels []float64
}

func Laden1Slice(path string) (Datensatz1Slice, Datensatz1Slice) {
	var trainingsDatensatz Datensatz1Slice
	var testDatensatz Datensatz1Slice
	train, test, _ := mnist.Load(path)
	for i := 0; i < len(train.Images); i++ {
		neuesBild := train.Images[i]
		for j := 0; j < (weite * hoehe); j++ {
			neuerWert := float64(neuesBild[j]) / 255.
			trainingsDatensatz.Bilder = append(trainingsDatensatz.Bilder, neuerWert)
		}

	}
	for i := 0; i < len(test.Images); i++ {
		neuesBild := test.Images[i]
		for j := 0; j < (weite * hoehe); j++ {
			neuerWert := float64(neuesBild[j]) / 255.
			testDatensatz.Bilder = append(testDatensatz.Bilder, neuerWert)
		}

	}
	for i := 0; i < len(train.Labels); i++ {
		neuesLabel := train.Labels[i]
		oneHotLabel := make([]float64, 10)
		oneHotLabel[int(neuesLabel)] = 1
		for j := 0; j < 10; j++ {
			neuerWertL := float64(oneHotLabel[j])
			trainingsDatensatz.Labels = append(trainingsDatensatz.Labels, neuerWertL)

		}
	}
	for i := 0; i < len(test.Labels); i++ {
		neuesLabel := test.Labels[i]
		oneHotLabel := make([]float64, 10)
		oneHotLabel[int(neuesLabel)] = 1
		for j := 0; j < 10; j++ {
			neuerWertL := float64(oneHotLabel[j])
			testDatensatz.Labels = append(testDatensatz.Labels, neuerWertL)

		}
	}
	return trainingsDatensatz, testDatensatz
}

func Laden(path string) (Datensatz, Datensatz) {
	var trainingsDatensatz Datensatz                     //Deklaration der variablen fuer das  trainingsets
	var testDatensatz Datensatz                          //Deklaration der variablen fuer das  testset
	trainingsDatensatz.Bilder = make([][]float64, 60000) //Erstellen des Bilder-Arrays mit Null-Werten
	testDatensatz.Bilder = make([][]float64, 10000)      //Erstellen des Bilder-Arrays mit Null-Werten
	train, test, _ := mnist.Load(path)                   // die Funktion mnist.Load aus dem Paket moverest verwenden, um Daten aus verpackten 									  //Dateien zu laden
	for i := 0; i < len(train.Images); i++ {             //die einzelnen Bilder des trainsets  werden nacheinander im trainingsdatensatz  //eingefuegt als []float64 mit der Groesse 784
		neuesBild := train.Images[i]
		trainingsDatensatz.Bilder[i] = make([]float64, 784)
		for j := 0; j < (weite * hoehe); j++ {
			trainingsDatensatz.Bilder[i][j] = (float64(neuesBild[j]) / 255.) // bevor die Werte eingefuegt werden, werden sie
			//durch 255. geteilt, um nur Werte zwischen 0 und 1 zu erhalten. (als rgb Bilder, haben sie Werte zwischen 0 und ///255)

		}

	}
	//s.O. das gleiche fuer das testset.
	for i := 0; i < len(test.Images); i++ {
		neuesBild := test.Images[i]
		testDatensatz.Bilder[i] = make([]float64, 784)
		for j := 0; j < (weite * hoehe); j++ {
			testDatensatz.Bilder[i][j] = (float64(neuesBild[j]) / 255.)

		}

	}
	//labels aus train und test fuer trainingsdatensatz und testdatensatz uebernehmen
	trainingsDatensatz.Labels = train.Labels
	testDatensatz.Labels = test.Labels

	return trainingsDatensatz, testDatensatz //rueckgabe von trainingsDatensatz und testDatensatz

}
