//Laudes
//26.03.22
//Package zur Darstellung von Bildumwandlungen von Neuronen 


package ldarstellungen

import (
	"fmt"
	// ".swp//lmodell"
	 "gfx"
	//"github.com/petjanzen/lmodelle"
)

const vergroesserung uint16 = 20 //Vergößert die 28*28-Pixel Bilder
var aPositiv, bPositiv, cPositiv, dPositiv, aNegativ, bNegativ, cNegativ, dNegativ float64 //Variablen für Kategorisiereung der Werte im Slice



func NeuronDarstellen (slice []float64, bWert float64) {
	var erg []float64 = make ([]float64,len(slice))  //Variable für das Ergebnis erstellen
	fmt.Println(slice)
	//Maximum und Minimum herausfinden
	var min, max float64
	
	for _,w:= range slice {
		w=w-bWert //hier wird der bWert (bias) neutralisiert, um die Daten nicht zu verfälschen
		if w > max {
			max=w
		} else if w < min {
			min=w
		}
	}
	//Minimum und Maximum werden als Grenzwerte genutzt, um die Abstufung in fünf verschie. Farbtönen im gfx-Fenster anzeigen zu können.
	fmt.Println ("das ist max und min", max, min)
	aPositiv=max*0.8
	bPositiv=max*0.6
	cPositiv=max*0.4
	dPositiv=max*0.2
	aNegativ=min*0.2
	bNegativ=min*0.4
	cNegativ=min*0.6
	dNegativ=min*0.8

	//Die Werte werden umgewandelt und in einen Ergebnis-Slice (erg) eingetragen.
	//Die ursprügnlichen Werte werden demnach in 10 verschiedene (5x positive, 5x negative) Abstufungen sortiert.
	//Diese werden im Ergebnis-Slice festgehalten indem nur die Werte 5.0, 4.0 ...-4.0, -5.0 eingetragen werden
	for i,w := range slice {
		w=w-bWert //hier wird der bWert (bias) neutralisiert, um die Daten nicht zu verfälschen
		fmt.Println ("DAs ist w: ", w)
		switch  {
			case w == 0.0: erg[i]=0.0
			case w > aPositiv: erg[i]=5.0
			case w > bPositiv: erg[i]=4.0
			case w > cPositiv: erg[i]=3.0
			case w > dPositiv: erg[i]=2.0
			case w > 0.0 && w<dPositiv: erg[i]=1.0
			case w > aNegativ: erg[i]=-1.0
			case w > bNegativ: erg[i]=-2.0
			case w > cNegativ: erg[i]=-3.0
			case w > dNegativ: erg[i]=-4.0
			default			 : erg[i]=-5.0
		}
			
	}
	fmt.Println ("das ist der neue Slice", erg)
	
	if !gfx.FensterOffen() {
	gfx.Fenster(vergroesserung*28,vergroesserung*28)
}
	var x,y uint16 //Koordinaten für den linken oberen Punkt des Quadrats
	//Je nach Ausprägung der Werte im Ergebnis-Slice werden jetzt farbige Quadrate im gfx-Fenster angezeigt.
	//hohe positive Werte werden dunkelblau, hohe negative Werte dunkelrot angezeigt.
	
	for _,w:=range erg {
			switch  {
			case w == 5.0: gfx.Stiftfarbe (70,0,0);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)	
			case w == 4.0: gfx.Stiftfarbe (100,0,0);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			case w == 3.0: gfx.Stiftfarbe (150,0,0);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			case w == 2.0: gfx.Stiftfarbe (255,0,0);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			case w == 1.0: gfx.Stiftfarbe (255,100,0);		gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			case w == -1.0: gfx.Stiftfarbe (0,100,255);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			case w == -2.0: gfx.Stiftfarbe (0,0,255);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			case w == -3.0: gfx.Stiftfarbe (0,0,150);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			case w == -4.0: gfx.Stiftfarbe (0,0,100);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			case w == -5.0: gfx.Stiftfarbe (70,0,0);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			default: 		gfx.Stiftfarbe (255,255,255);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)	
		}
		x = (x+vergroesserung) % (28*vergroesserung)  //Neusetzung von x
		if x == 0 {									//Wenn Zeilenumsprung...
		y = (y+vergroesserung) 					//...Neusetzung von y
	}

}
gfx.TastaturLesen1() //gfx-Bild solange vorhanden, bis beliebige Tastaturtaste gedrückt wird
}



	
	
	

