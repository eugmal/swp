//heidi hoehn
//Zweck: Zahlen mit der Maus zeichnen und in einem 2dimensionalen Array speichern
//Datum: 30.03.2022

package gui

import (
	"fmt"
	"gfx"

	"gonum.org/v1/gonum/mat"
)

const vergroesserung uint16 = 20

// Vor.: Grafikfenster muss offen sein.
// Eff.: Benutzer zeichnet mit der Maus eine Zahl, indem er die linke Maustaste gedrückt hält und die Maus bewegt
// 		 Wird die rechte Maustaste gedrückt, signalisiert der Benutzer: "meine Zeichung ist fertig!"
// Erg.: 2D-Array, das das gezeichnete Bild mit in Schwarz(= 1)-Weiß(= 0) kodiert

func EinlesenZeichnung() [560][560]uint16 {
	var erg [560][560]uint16 //Variable für das Ergebnis erstellen
	gfx.Stiftfarbe(0, 0, 0)  // setzt Stiftfarbe auf schwarz
	gfx.Schreibe(0, 0, "Zeichne bitte eine Zahl!")
	gfx.Schreibe(0, 12, "Wenn du fertig bist, drueck die rechte Maustaste!")
	for { //Endlosschleife, in der die Benutzereingabe abgefragt wird
		taste, status, mausX, mausY := gfx.MausLesen1() //Auslesen der Mauseigenschaften
		if taste == 1 && status != -1 {                 // Wenn Maustaste 1 gedrückt oder gehalten wird, ...
			gfx.Vollrechteck(mausX, mausY, vergroesserung, vergroesserung) //zeichnen wir einen "Pixel" (= Vollrechteck) der Größe "vergroesserung"
			var i, j uint16
			for i = 0; i < vergroesserung; i++ {
				for j = 0; j < vergroesserung; j++ {
					erg[mausX+i][mausY+j] = 1
				}
			}
		}
		if taste == 3 {
			break
		}
	}
	return erg
}

func bildSkalieren(bildOriginal [560][560]uint16) [28][28]float64 {
	var bildScaled [28][28]float64
	row := 0
	for x := 0; x < 560; {

		col := 0
		for i := 0; i < 560; {

			summe := 0
			for y := x; y < x+20; y++ {
				for j := i; j < i+20; j++ {
					summe = summe + int(bildOriginal[y][j])
				}
			}
			//fmt.Println(summe)
			summeScaled := float64(summe) / (20.0 * 20.0)
			bildScaled[row][col] = summeScaled
			fmt.Println("wert: ", bildScaled[row][col])
			fmt.Println("array: ", bildScaled)
			i = i + 20
			col++
		}
		x = x + 20
		row++
	}
	return bildScaled
}

func gibBildMatrix(bildArray [28][28]float64) *mat.Dense {
	bildSlice := make([]float64, 28*28)

	index := 0
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			bildSlice[index] = bildArray[i][j]
		}
	}
	bildMatrix := mat.NewDense(784, 1, bildSlice)
	return bildMatrix

}

//Vor.: keine
//Eff.: ein Grafikfenster mit weißem Hintergrund wird erstellt, die Funktion "EinlesenZeichnung()" wird aufgerufen, damit Benutzer eine Zahl zeichnen kann
//		Wenn der Benutzer fertig ist, wird zuerst die Kodierung des Bildes im Terminal ausgegeben und dann aus der Kodierung wieder das gezeichnete Bild rekonstruiert.
//		Der letzte Schritt stellt sicher, dass die Kodierung korrekt ist.
func ZahlMalen() *mat.Dense {
	gfx.Fenster(560, 560) // öffnet das Grafikfenster mit weißem Hintergrund
	fmt.Println(gfx.GibFont())

	var bildBinaer [560][560]uint16
	bildBinaer = EinlesenZeichnung()
	//fmt.Println(bildBinaer) //gib die Kodierung im Terminal aus
	// rekonstruiere aus der Kodierung das gezeichnete Bild
	gfx.Stiftfarbe(255, 255, 255)
	gfx.Cls()
	gfx.Stiftfarbe(0, 0, 0)
	bildScale := bildSkalieren(bildBinaer)
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			if bildScale[i][j] != 0 {
				gfx.Punkt(uint16(i), uint16(j))
			}
		}
	}
	gfx.TastaturLesen1()
	zahlMatrix := gibBildMatrix(bildScale)
	fmt.Println(zahlMatrix)
	return zahlMatrix

}
