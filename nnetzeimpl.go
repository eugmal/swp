package nnetze

import (
	"fmt"
	"gfx"
	"image"
	"image/png"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"./github.com/eugmal/swp/mnistLoad"
	"gonum.org/v1/gonum/mat"
)

/////////////////////////////////////////////////////////////////////////////////////////////////////////

type data struct {

	// das wird beim Initialisieren gesetzt

	k int // Anzahl Neuronen im Hidden Layer
	l int // Anzahl Neuronen im Output Layer

	// das wird beim Einlesen der Daten gesetzt

	m int       // Anzahl der Trainings-Datensätze
	d int       // Anzahl Pixel der Bilder
	x []float64 // alle x-Vektoren hintereinander (Länge: m * d)
	y []float64 // alle y-Vektoren hintereinander (Länge: m * l)

	// das wird beim Trainieren gesetzt

	w1 *mat.Dense // Weights im Hidden Layer
	b1 *mat.Dense // Biases im Hidden Layer
	w2 *mat.Dense // Weights im Output Layer
	b2 *mat.Dense // Biases im Output Layer
}

func New(k, l int) *data {
	n := new(data)
	n.k = k
	n.l = l
	return n
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

func (n *data) TrainingsdatenEinlesenAusMnist(d string) {
	n.m = 60000
	n.d = 784
	datensatz, _ := mnistLoad.Laden1Slice(d) // Laden der Daten aus dem Mnist-Set
	n.x = datensatz.Bilder
	n.y = datensatz.Labels
}

func (n *data) TrainingsdatenEinlesenAusPngGraustufen(name string) {

	// Auslesen der x- und y-Vektoren aus den Dateien
	f, _ := os.Open(name) // Öffnen der Trainings-Datei
	b := make([]byte, 1)  // Platzhalter zum Auslesen des aktuellen Bytes
	_, err := f.Read(b)   // Lesen des ersten Bytes
	for err != io.EOF {
		imgName := make([]byte, 0) // Platzhalter für Bild-Datei-Namen
		for b[0] != byte(' ') {
			imgName = append(imgName, b[0]) // Eintragen des Bild-Datei-Names in den Platzhalter
			f.Read(b)                       // Lesen der Bild-Datei-Namen-Bytes oder des Leerzeichens
		}
		str := string(imgName)          // Umwandeln des Bild-Datei-Names in ein String
		sl, pixelNmb := pngToSlice(str) // Auslesen der Pixel-Daten und Anzahl der Pixel
		n.d = pixelNmb                  // Speichern der Pixel-Anzahl
		n.x = append(n.x, sl...)        // Anhängen der Pixeldaten an den x-Vektor
		f.Read(b)
		yValue := make([]byte, 0) // Platzhalter für y-Wert
		for b[0] != byte('\n') {
			yValue = append(yValue, b[0]) // Eintragen des y-Werts in den Platzhalter
			f.Read(b)                     // Lesen des y-Wertes oder des Leerzeichens
		}
		str = string(yValue)                          // Umwandeln des y-Werts in ein String
		yValueInt, _ := strconv.ParseInt(str, 10, 64) // Umwandeln des y-Werts in ein Int-Wert
		yVector := make([]float64, n.l)               // y-Teilvektor der Länge l
		yVector[yValueInt] = 1.                       // ein Eintrag im y-Vektor wird auf 1 gesetzt
		n.y = append(n.y, yVector...)                 // Anhängen des y-Teilvektors an den y-Vektor
		n.m++                                         // Zählen der Trainings-Datensätze
		_, err = f.Read(b)                            // Lesen des ersten Bytes der nächsten Zeile
	}

}

// Hilfsfunktion für die Methode TrainingsdatenEinlesenAusPngGraustufen
func rgbaToGray(img image.Image) *image.Gray {
	bounds := img.Bounds()
	gray := image.NewGray(bounds)
	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			rgba := img.At(x, y)
			gray.Set(x, y, rgba)
		}
	}
	return gray
}

// Hilfsfunktion für die Methode TrainingsdatenEinlesenAusPngGraustufen
func pngToSlice(name string) ([]float64, int) {

	sl := make([]float64, 0)

	file, err := os.Open(name)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	img, err := png.Decode(file)
	if err != nil {
		log.Fatal(err)
	}

	bounds := img.Bounds()
	gray := rgbaToGray(img)
	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			sl = append(sl, float64(gray.GrayAt(x, y).Y)/255.)
		}
	}

	return sl, bounds.Max.X * bounds.Max.Y

}

////////////////////////////////////////////////////////////////////////////////////////////////////////

func (n *data) GradientenverfahrenStochastisch(alpha float64, NmbIteration int) {

	rand.Seed(time.Now().UnixNano()) // Seeden der Zufallszahlen für W1, B1, W2 und B2

	// W1

	sl := make([]float64, n.k*n.d)
	for i := 0; i < n.k*n.d; i++ {
		sl[i] = (rand.Float64() * 2.) - 1. // Zufallszahlen zwischen -1 und 1
	}
	n.w1 = mat.NewDense(n.k, n.d, sl)

	// B1

	sl = make([]float64, n.k)
	for i := 0; i < n.k; i++ {
		sl[i] = (rand.Float64() * 2.) - 1. // Zufallszahlen zwischen -1 und 1
	}
	n.b1 = mat.NewDense(n.k, 1, sl)

	// W2

	sl = make([]float64, n.k*n.l)
	for i := 0; i < n.k*n.l; i++ {
		sl[i] = (rand.Float64() * 2.) - 1. // Zufallszahlen zwischen -1 und 1
	}
	n.w2 = mat.NewDense(n.l, n.k, sl)

	// B2

	sl = make([]float64, n.l)
	for i := 0; i < n.l; i++ {
		sl[i] = (rand.Float64() * 2.) - 1. // Zufallszahlen zwischen -1 und 1
	}
	n.b2 = mat.NewDense(n.l, 1, sl)

	// Trainingsalgorithmus: GradientenverfahrenStochastisch

	for i := 0; i < NmbIteration; i++ {
		for k := 0; k < n.m; k++ {

			sl := make([]float64, 0)                 // Platzhalter für aktuellen x-Vektor
			sl = append(sl, n.x[k*n.d:(k+1)*n.d]...) // Anhängen des aktuellen x-Vektors mit der Länge d
			x := mat.NewDense(n.d, 1, sl)

			sl = make([]float64, 0)                  // Platzhalter für aktuellen y-Vektor
			sl = append(sl, n.y[k*n.l:(k+1)*n.l]...) // Anhängen des aktuellen y-Vektors mit der Länge l
			y := mat.NewDense(n.l, 1, sl)

			// forward propagation

			// I
			z1 := mat.NewDense(n.k, 1, nil)
			z1.Mul(n.w1, x)
			z1.Add(z1, n.b1)

			// II
			a1 := mat.NewDense(n.k, 1, nil)
			a1.Apply(activationSigmoid, z1)

			// III
			z2 := mat.NewDense(n.l, 1, nil)
			z2.Mul(n.w2, a1)
			z2.Add(z2, n.b2)

			// IV
			a2 := mat.NewDense(n.l, 1, nil)
			a2.Apply(activationSigmoid, z2)

			// backward propagation

			// B2

			dLdb2 := mat.NewDense(n.l, 1, nil)
			dLdb2.Sub(a2, y)
			dLdb2Learning := mat.NewDense(n.l, 1, nil)
			dLdb2Learning.Scale(alpha, dLdb2)

			// W2

			dLdw2 := mat.NewDense(n.l, n.k, nil)
			dLdw2.Mul(dLdb2, a1.T())
			dLdw2Learning := mat.NewDense(n.l, n.k, nil)
			dLdw2Learning.Scale(alpha, dLdw2)

			// B1

			dLdb1 := mat.NewDense(n.k, 1, nil)
			dLdb1.MulElem(a1, a1)
			dLdb1.Sub(a1, dLdb1)
			temp := mat.NewDense(n.l, 1, nil)
			temp.Sub(a2, y)
			temp2 := mat.NewDense(n.k, 1, nil)
			temp2.Mul(n.w2.T(), temp)
			dLdb1.MulElem(temp2, dLdb1)
			dLdb1Learning := mat.NewDense(n.k, 1, nil)
			dLdb1Learning.Scale(alpha, dLdb1)

			// W1

			dLdw1 := mat.NewDense(n.k, n.d, nil)
			dLdw1.Mul(dLdb1, x.T())
			dLdw1Learning := mat.NewDense(n.k, n.d, nil)
			dLdw1Learning.Scale(alpha, dLdw1)

			// Änderung der Parameter

			n.b2.Sub(n.b2, dLdb2Learning)
			n.w2.Sub(n.w2, dLdw2Learning)
			n.b1.Sub(n.b1, dLdb1Learning)
			n.w1.Sub(n.w1, dLdw1Learning)

		}
	}

}

// Hilfsfunktion für die Methode GradientenverfahrenStochastisch
func activationSigmoid(i, j int, v float64) float64 {
	return 1. / (1. + math.Exp(-v))
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

func (n *data) TestdatenAuswertenAusMnist(d string) {

	_, datensatz := mnistLoad.Laden1Slice(d) // Laden der Test-Daten aus dem Mnist-Set
	n.x = datensatz.Bilder
	n.y = datensatz.Labels
	n.m = 10000

	counter := 0

	for k := 0; k < n.m; k++ {

		sl := make([]float64, 0)                 // Platzhalter für aktuellen x-Vektor
		sl = append(sl, n.x[k*n.d:(k+1)*n.d]...) // Anhängen des aktuellen x-Vektors mit der Länge d
		x := mat.NewDense(n.d, 1, sl)

		sl = make([]float64, 0)                  // Platzhalter für aktuellen y-Vektor
		sl = append(sl, n.y[k*n.l:(k+1)*n.l]...) // Anhängen des aktuellen y-Vektors mit der Länge l
		y := mat.NewDense(n.l, 1, sl)

		// forward propagation

		// I
		z1 := mat.NewDense(n.k, 1, nil)
		z1.Mul(n.w1, x)
		z1.Add(z1, n.b1)

		// II
		a1 := mat.NewDense(n.k, 1, nil)
		a1.Apply(activationSigmoid, z1)

		// III
		z2 := mat.NewDense(n.l, 1, nil)
		z2.Mul(n.w2, a1)
		z2.Add(z2, n.b2)

		// IV
		a2 := mat.NewDense(n.l, 1, nil)
		a2.Apply(activationSigmoid, z2)

		hoechstes := 0.                  // Feststellen der Prognose
		index := 0
		for i := 0; i < n.l; i++ {
			if a2.At(i, 0) > hoechstes {
				hoechstes = a2.At(i, 0)
				index = i
			}
		}

		if y.At(index, 0) == 1 {         // Feststellen der Trefferwahrscheinlichkeit
			counter++
		}

	}

	fmt.Println("Trefferquote beim Mnist-Test-Datenset:", 100.*float64(counter)/float64(n.m), "%")

}

func (n *data) TestdatenAuswertenAusPngGraustufen(name string) {

	counter := 0

	f, _ := os.Open(name) // Öffnen der Test-Datei
	b := make([]byte, 1)  // Platzhalter zum Auslesen des aktuellen Bytes
	_, err := f.Read(b)   // Lesen des ersten Bytes
	for err != io.EOF {
		imgName := make([]byte, 0) // Platzhalter für Bild-Datei-Namen
		for b[0] != byte(' ') {
			imgName = append(imgName, b[0]) // Eintragen des Bild-Datei-Names in den Platzhalter
			f.Read(b)                       // Lesen der Bild-Datei-Namen-Bytes oder des Leerzeichens
		}
		str := string(imgName)   // Umwandeln des Bild-Datei-Names in ein String
		sl, _ := pngToSlice(str) // Auslesen der Pixel-Daten
		c := make([]float64, 0)
		c = append(c, sl...)
		x := mat.NewDense(n.d, 1, c)

		f.Read(b)
		y := make([]byte, 0) // Platzhalter für y-Wert
		for b[0] != byte('\n') {
			y = append(y, b[0]) // Eintragen des y-Werts in den Platzhalter
			f.Read(b)           // Lesen des y-Wertes oder des Leerzeichens
		}

		// I
		z1 := mat.NewDense(n.k, 1, nil)
		z1.Mul(n.w1, x)
		z1.Add(z1, n.b1)

		// II
		a1 := mat.NewDense(n.k, 1, nil)
		a1.Apply(activationSigmoid, z1)

		// III
		z2 := mat.NewDense(n.l, 1, nil)
		z2.Mul(n.w2, a1)
		z2.Add(z2, n.b2)

		// IV
		a2 := mat.NewDense(n.l, 1, nil)
		a2.Apply(activationSigmoid, z2)

		hoechstes := 0.                 // Feststellen der Prognose
		index := 0
		for i := 0; i < n.l; i++ {
			if a2.At(i, 0) > hoechstes {
				hoechstes = a2.At(i, 0)
				index = i
			}
		}

		if index == int(y[0])-48 {     // Feststellen der Trefferwahrscheinlichkeit
			counter++
		}

		_, err = f.Read(b) // Lesen des ersten Bytes der nächsten Zeile

	}

	fmt.Println(100.*float64(counter)/float64(n.m), "%")

}

////////////////////////////////////////////////////////////////////////////////////////////////////////

func (n *data) W(i int) []float64 {
	return n.w1.RawRowView(i)
}

func (n *data) B(i int) float64 {
	return n.b1.RawRowView(i)[0]
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

func (n *data) Darstellen(q int) {
	fmt.Println("Hidden Neuron", q)
	neuronDarstellen(n.W(q), n.B(q))
}

// Hilfsfunktion für die Methode Darstellen
func neuronDarstellen(slice []float64, bWert float64) {

	const vergroesserung uint16 = 20                                                           //Vergößert die 28*28-Pixel Bilder, um sie im gfx-Fenster darstellen zu können
	var aPositiv, bPositiv, cPositiv, dPositiv, aNegativ, bNegativ, cNegativ, dNegativ float64 //Variablen für Kategorisiereung der Werte im Slice
	var erg []float64 = make([]float64, len(slice))                                            //Variable für das Ergebnis erstellen
	//Maximum und Minimum herausfinden
	var min, max float64

	for _, w := range slice {
		w = w - bWert //hier wird der bWert (bias) neutralisiert, um die Daten nicht zu verfälschen
		if w > max {
			max = w
		} else if w < min {
			min = w
		}
	}
	//Minimum und Maximum werden als Grenzwerte genutzt, um die Abstufung in fünf verschie. Farbtönen im gfx-Fenster anzeigen zu können.
	aPositiv = max * 0.8
	bPositiv = max * 0.6
	cPositiv = max * 0.4
	dPositiv = max * 0.2
	aNegativ = min * 0.2
	bNegativ = min * 0.4
	cNegativ = min * 0.6
	dNegativ = min * 0.8

	//Die Werte werden umgewandelt und in einen Ergebnis-Slice (erg) eingetragen.
	//Die ursprügnlichen Werte werden demnach in 10 verschiedene (5x positive, 5x negative) Abstufungen sortiert.
	//Diese werden im Ergebnis-Slice festgehalten indem nur die Werte 5.0, 4.0 ...-4.0, -5.0 eingetragen werden
	for i, w := range slice {
		//w=w-bWert //hier wird der bWert (bias) neutralisiert, um die Daten nicht zu verfälschen, (nicht nötig, oben schon geschehen)
		//fmt.Println ("DAs ist w: ", w)
		switch {
		case w == 0.0:
			erg[i] = 0.0
		case w > aPositiv:
			erg[i] = 5.0
		case w > bPositiv:
			erg[i] = 4.0
		case w > cPositiv:
			erg[i] = 3.0
		case w > dPositiv:
			erg[i] = 2.0
		case w > 0.0 && w < dPositiv:
			erg[i] = 1.0
		case w > aNegativ:
			erg[i] = -1.0
		case w > bNegativ:
			erg[i] = -2.0
		case w > cNegativ:
			erg[i] = -3.0
		case w > dNegativ:
			erg[i] = -4.0
		default:
			erg[i] = -5.0
		}

	}

	if !gfx.FensterOffen() {
		gfx.Fenster(vergroesserung*28, vergroesserung*28)
	}
	var x, y uint16 //Koordinaten für den linken oberen Punkt des Quadrats
	//Je nach Ausprägung der Werte im Ergebnis-Slice werden jetzt farbige Quadrate im gfx-Fenster angezeigt.
	//hohe positive Werte werden dunkelrot, hohe negative Werte dunkelblau angezeigt.

	for _, w := range erg {
		switch {
		case w == 5.0:
			gfx.Stiftfarbe(100, 0, 0)
			gfx.Vollrechteck(x, y, vergroesserung, vergroesserung)
		case w == 4.0:
			gfx.Stiftfarbe(140, 0, 0)
			gfx.Vollrechteck(x, y, vergroesserung, vergroesserung)
		case w == 3.0:
			gfx.Stiftfarbe(180, 0, 0)
			gfx.Vollrechteck(x, y, vergroesserung, vergroesserung)
		case w == 2.0:
			gfx.Stiftfarbe(255, 0, 0)
			gfx.Vollrechteck(x, y, vergroesserung, vergroesserung)
		case w == 1.0:
			gfx.Stiftfarbe(255, 100, 0)
			gfx.Vollrechteck(x, y, vergroesserung, vergroesserung)
		case w == -1.0:
			gfx.Stiftfarbe(0, 100, 255)
			gfx.Vollrechteck(x, y, vergroesserung, vergroesserung)
		case w == -2.0:
			gfx.Stiftfarbe(0, 0, 255)
			gfx.Vollrechteck(x, y, vergroesserung, vergroesserung)
		case w == -3.0:
			gfx.Stiftfarbe(0, 0, 180)
			gfx.Vollrechteck(x, y, vergroesserung, vergroesserung)
		case w == -4.0:
			gfx.Stiftfarbe(0, 0, 140)
			gfx.Vollrechteck(x, y, vergroesserung, vergroesserung)
		case w == -5.0:
			gfx.Stiftfarbe(0, 0, 100)
			gfx.Vollrechteck(x, y, vergroesserung, vergroesserung)
		default:
			gfx.Stiftfarbe(255, 255, 255)
			gfx.Vollrechteck(x, y, vergroesserung, vergroesserung)
		}
		x = (x + vergroesserung) % (28 * vergroesserung) //Neusetzung von x
		if x == 0 {                                      //Wenn Zeilenumsprung...
			y = (y + vergroesserung) //...Neusetzung von y
		}

	}

	gfx.TastaturLesen1() //gfx-Bild solange vorhanden, bis beliebige Tastaturtaste gedrückt wird
	gfx.TastaturLesen1()
	
}

////////////////////////////////////////////////////////////////////////

func (n *data) Zeichnen() {
	
	bild := zahlMalen()      // Laden des gezeichneten Bildes

	z1 := mat.NewDense(n.k, 1, nil)
	z1.Mul(n.w1, bild)
	z1.Add(z1, n.b1)

	// II
	a1 := mat.NewDense(n.k, 1, nil)
	a1.Apply(activationSigmoid, z1)

	// III
	z2 := mat.NewDense(n.l, 1, nil)
	z2.Mul(n.w2, a1)
	z2.Add(z2, n.b2)

	// IV
	a2 := mat.NewDense(n.l, 1, nil)
	a2.Apply(activationSigmoid, z2)
	
	fmt.Println("-----------------------------------------------------")
	
	for q := 0; q < n.l; q++ {
		fmt.Println(q, ":", a2.At(q,0))
	}
	
	hoechstes := 0.            // Feststellen der Prognose
	index := 0
	for i := 0; i < n.l; i++ {
		if a2.At(i, 0) > hoechstes {
			hoechstes = a2.At(i, 0)
			index = i
		}
	}
	
	fmt.Println("----------------------------")
	
	fmt.Println("Sie haben eine", index, "gezeichnet!")

}

// Konstanten für die Methode Zeichnen
const vergroesserung uint16 = 20
const SCHWARZ uint16 = 255
const WEISS uint16 = 0

// Hilfsfunktion für die Methode Zeichnen
// Grafikfenster muss offen sein.
// Benutzer zeichnet mit der Maus eine Zahl, indem er die linke Maustaste gedrückt hält und die Maus bewegt
// Wird die rechte Maustaste gedrückt, signalisiert der Benutzer: "meine Zeichung ist fertig!"
// 2D-Array, das das gezeichnete Bild mit in Schwarz(= 0)-Weiß(= 255) kodiert
func einlesenZeichnung() [560][560]uint16 {
	var erg [560][560]uint16 //Variable für das Ergebnis erstellen
	for i := 0; i < len(erg); i++ {
		for j := 0; j < len(erg); j++ {
				erg[i][j] = WEISS
		}
	}
	gfx.Stiftfarbe(0, 0, 0)  // setzt Stiftfarbe auf schwarz
	gfx.Schreibe(0, 0, "Zeichne bitte eine Zahl!")
	gfx.Schreibe(0, 12, "Wenn du fertig bist, drueck die rechte Maustaste!")
	for { //Endlosschleife, in der die Benutzereingabe abgefragt wird
		taste, status, mausX, mausY := gfx.MausLesen1() //Auslesen der Mauseigenschaften
		if taste == 1 && status != -1 {                 // Wenn linke Maustaste gedrückt oder gehalten wird, ...
			gfx.Vollrechteck(mausX, mausY, vergroesserung, vergroesserung) //zeichnen wir einen "Pixel" (= Vollrechteck) der Größe "vergroesserung"
			var i, j uint16
			for i = 0; i < vergroesserung; i++ {
				for j = 0; j < vergroesserung; j++ {
					erg[mausX+i][mausY+j] = SCHWARZ
				}
			}
		}
		if taste == 3 { // = rechte Maustaste
			time.Sleep(time.Second)
			break
		}
	}
	return erg
}

// Hilfsfunktion für die Methode Zeichnen
// gfx.Vollrechteck(mausX-5, mausY-5, vergroesserung+10, vergroesserung+10) //zeichnen wir einen "Pixel" (= Vollrechteck) der Größe "vergroesserung"
// gfx.Stiftfarbe(0, 0, 0)  // setzt Stiftfarbe auf schwarz
func bildSkalieren(bildOriginal [560][560]uint16) [28][28]float64 { // Runterskalieren des Bildes von 560x560 auf 28x28 - das werden 28x28 Zahlen zwischen 0 und 1 sein
	var bildScaled [28][28]float64
	row := 0
	for x := 0; x < 560; {

		col := 0
		for i := 0; i < 560; {

			summe := 0
			for y := x; y < x+20; y++ {
				for j := i; j < i+20; j++ {
					summe = summe + int(bildOriginal[j][y]) //Bilder sind spaltenweise angeordnet, nicht zeilenweise!
				}
			}

			summeScaled := (float64(summe) / (20.0 * 20.0* 255))
			bildScaled[row][col] = summeScaled

			i = i + 20
			col++
		}
		x = x + 20
		row++
	}
	return bildScaled
}

// Hilfsfunktion für die Methode Zeichnen: "Flachklopfen" des 2-dimensionalen Arrays
func gibBildMatrix(bildArray [28][28]float64) *mat.Dense {
	bildSlice := make([]float64, 28*28)

	index := 0
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			bildSlice[index] = bildArray[i][j]
			index++
		}
	}
	bildMatrix := mat.NewDense(784, 1, bildSlice)

	return bildMatrix

}

// Hilfsfunktion für die Methode Zeichnen
func zahlMalen() *mat.Dense {
	gfx.Fenster(560, 560) // öffnet das Grafikfenster mit weißem Hintergrund
	fmt.Println(gfx.GibFont())

	var bildBinaer [560][560]uint16
	bildBinaer = einlesenZeichnung()
	//fmt.Println(bildBinaer) //gib die Kodierung im Terminal aus
	// rekonstruiere aus der Kodierung das gezeichnete Bild
	gfx.Stiftfarbe(255, 255, 255) // weiß
	gfx.Cls()
	gfx.Stiftfarbe(0, 0, 0) // schwarz
	bildScale := bildSkalieren(bildBinaer)
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			if bildScale[i][j] > 0.5 { //Durchschnittswert kleiner 0.5 bedeutet "schwarz"; beim komprimierten Bild wird der Hintergrund schwarz und die gezeichnete Ziffer weiß dargestellt
				gfx.Punkt(uint16(i), uint16(j))
			}
		}
	}
	gfx.TastaturLesen1()
	zahlMatrix := gibBildMatrix(bildScale)

	return zahlMatrix

}

////////////////////////////////////////////////////////////////////////////////////////////////////////

//Laden und Speichern
// gewichte und biases werden an eingegebenen Ort gespeichert (diesen Pfad dann fuer die Lade-Methode verwenden)
/*func (n *data) Speichern(speicherort string) {

	weight1, err := os.Create(speicherort + "/w1")
	defer weight1.Close()
	if err == nil {
		n.w1.MarshalBinaryTo(weight1)
	}
	weight2, err := os.Create(speicherort + "/w2")
	defer weight2.Close()
	if err == nil {
		n.w2.MarshalBinaryTo(weight2)
	}
	bias1, err := os.Create(speicherort + "/b1")
	defer bias1.Close()
	if err == nil {
		n.b1.MarshalBinaryTo(bias1)
	}
	bias2, err := os.Create(speicherort + "/b2")
	defer bias2.Close()
	if err == nil {
		n.b2.MarshalBinaryTo(bias2)
	}

}

// speicherort muss beim aufrufen der methode angegeben werden.
func (n *data) Laden(speicherort string) {
	n.w1 = mat.NewDense(n.k, n.d, nil)
	n.w2 = mat.NewDense(n.l, n.k, nil)
	n.b1 = mat.NewDense(n.k, 1, nil)
	n.b2 = mat.NewDense(n.l, 1, nil)

	weight1, err := os.Open(speicherort + "/w1")
	defer weight1.Close()
	if err == nil {
		n.w1.Reset()
		n.w1.UnmarshalBinaryFrom(weight1)
	}
	weights2, err := os.Open(speicherort + "/w2")
	if err == nil {
		n.w2.Reset()
		n.w2.UnmarshalBinaryFrom(weights2)
	}
	bias1, err := os.Open(speicherort + "/b1")
	defer bias1.Close()
	if err == nil {
		n.b1.Reset()
		n.b1.UnmarshalBinaryFrom(bias1)
	}
	bias2, err := os.Open(speicherort + "/b2")
	if err == nil {
		n.b2.Reset()
		n.b2.UnmarshalBinaryFrom(bias2)
	}

}*/
