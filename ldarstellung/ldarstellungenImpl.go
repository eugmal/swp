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




func Umwandeln (slice []float64, bWert float64) ([]uint8){
	var erg []uint8 = make ([]uint8,len(slice))  //Variable für das Ergebnis erstellen
	
	for i,w := range slice {
		w=w-bWert //hier wird der bWert (bias) neutralisiert, um die Daten nicht zu verfälschen
		fmt.Println (w)
		if w > 0 {
			erg[i] = uint8(1)  //alles. was größer ist als die Konstante "flip" wird zur 1...
		} else if w < 0 {
			erg[i] = uint8(2) 
		} else {
			erg[i] = uint8(0) //alles, was kleiner ist als die Konstante "flip"  wird zur 0.  
		}
	
			
		
	}
	return erg //Also: Daten als Binärcode in einem Slice vorhanden
}


func Darstellen (slice []uint8) {  					//für die gfx Darstellung eines Binärcode-Slices
	var x,y uint16 									//Koordinaten für den linken oberen Punkt des Quadrats

	if !gfx.FensterOffen() {
	gfx.Fenster(vergroesserung*28,vergroesserung*28)
}
	for _,w:=range slice {
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
	}
	gfx.TastaturLesen1() //gfx-Bild solange vorhanden, bis beliebige Tastaturtaste gedrückt wird
	
}
