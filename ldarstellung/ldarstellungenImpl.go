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


//const flip float64 = 0.0167  // Konstante frei wählbar. Hier gewählt nach Auswertung des data.w-Slices
const vergroesserung uint16 = 20 //Vergößert die 28*28-Pixel Bilder
var aPositiv, bPositiv, cPositiv, dPositiv, aNegativ, bNegativ, cNegativ, dNegativ float64



func Umwandeln (slice []float64, bWert float64) ([]float64){
	var erg []float64 = make ([]float64,len(slice))  //Variable für das Ergebnis erstellen
	var min, max float64
	
	//Maximum und Minimum rausfinden
	for _,w:= range slice {
		w=w-bWert //hier wird der bWert (bias) neutralisiert, um die Daten nicht zu verfälschen
		if w > max {
			max=w
		} else if w < min {
			min=w
		}
	}
	fmt.Println ("das ist max und min", max, min)
	aPositiv=max*0.8
	bPositiv=max*0.6
	cPositiv=max*0.4
	dPositiv=max*0.2
	aNegativ=min*0.2
	bNegativ=min*0.4
	cNegativ=min*0.6
	dNegativ=min*0.8

	
	for i,w := range slice {
		w=w-bWert //hier wird der bWert (bias) neutralisiert, um die Daten nicht zu verfälschen
		//fmt.Println (w)
		switch  {
			case w == 0.0: erg[i]=0.0
			case w > aPositiv: erg[i]=4.0
			case w > bPositiv: erg[i]=3.0
			case w > cPositiv: erg[i]=2.0
			case w > dPositiv: erg[i]=1.0
			case w > aNegativ: erg[i]=-1.0
			case w > bNegativ: erg[i]=-2.0
			case w > cNegativ: erg[i]=-3.0
			case w > dNegativ: erg[i]=-4.0
		}
			
	}
	//fmt.Println ("das ist der neue Slice", erg)
	return erg //Also: Daten als Binärcode in einem Slice vorhanden
}


func Darstellen (slice []float64) {  					//für die gfx Darstellung eines Binärcode-Slices
	var x,y uint16 									//Koordinaten für den linken oberen Punkt des Quadrats

	if !gfx.FensterOffen() {
	gfx.Fenster(vergroesserung*28,vergroesserung*28)
}
	for _,w:=range slice {

			switch  {
			case w == 4.0: gfx.Stiftfarbe (255,0,0);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			case w == 3.0: gfx.Stiftfarbe (180,0,0);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			case w == 2.0: gfx.Stiftfarbe (120,0,0);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			case w == 1.0: gfx.Stiftfarbe (70,0,0);		gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			case w == -1.0: gfx.Stiftfarbe (0,0,255);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			case w == -2.0: gfx.Stiftfarbe (0,0,180);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			case w == -3.0: gfx.Stiftfarbe (0,0,120);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			case w == -4.0: gfx.Stiftfarbe (0,0,20);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)
			default: 		gfx.Stiftfarbe (255,255,255);	gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)	
		}
		x = (x+vergroesserung) % (28*vergroesserung)  //Neusetzung von x
		if x == 0 {									//Wenn Zeilenumsprung...
		y = (y+vergroesserung) 					//...Neusetzung von
	}
		
		
		/*
		if w==1 {  									//Wenn 1 ausgelesen wird, zeichne Quardat mit Seitenlänge vergroesserung in schwarz
			gfx.Stiftfarbe (0,0,0)
			gfx.Vollrechteck (x,y, vergroesserung, vergroesserung)	
		} else if w==0{  							//Wenn 0 ausgelesen wird, zeichne Quardat mit Seitenlänge vergroesserung in türkis
			gfx.Stiftfarbe (0,255,255)
			gfx.Vollrechteck (x,y, vergroesserung,vergroesserung)
		} else if w==2{  							//Wenn 0 ausgelesen wird, zeichne Quardat mit Seitenlänge vergroesserung in türkis
			gfx.Stiftfarbe (255,0,0)
			gfx.Vollrechteck (x,y, vergroesserung,vergroesserung)
		}
		x = (x+vergroesserung) % (28*vergroesserung)  //Neusetzung von x
		if x == 0 {									//Wenn Zeilenumsprung...
			y = (y+vergroesserung) 					//...Neusetzung von y
		} 
		* */
	}
	gfx.TastaturLesen1() //gfx-Bild solange vorhanden, bis beliebige Tastaturtaste gedrückt wird
	
}
