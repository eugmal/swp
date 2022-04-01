//heidi hoehn
//Zweck: Zahlen mit der Maus zeichnen und in einem 2dimensionalen Array speichern
//Datum: 30.03.2022

package main
import ("gfx"; "fmt")

const vergroesserung uint16 = 20 

func EinlesenZeichnung() [560][560]uint16{
	var erg [560][560]uint16  //Variable für das Ergebnis erstellen
	gfx.Stiftfarbe(0,0,0) // setzt Stiftfarbe auf schwarz
	for  { //Endlosschleife, in der die Benutzereingabe abgefragt wird				
		taste,status,mausX, mausY := gfx.MausLesen1()	//Auslesen der Mauseigenschaften 
		if (taste == 1 && status != -1){				// Wenn Maustaste 1 gedrückt oder gehalten wird, ...
			gfx.Vollrechteck(mausX, mausY, vergroesserung, vergroesserung)	//zeichnen wir einen "Pixel" (= Vollrechteck) der Größe "vergroesserung"
			var i, j uint16
			for i = 0; i < vergroesserung; i++ {
				for j = 0; j < vergroesserung; j++ {
					erg[mausX + i][mausY + j] = 1
				}
			}
		}
		if taste == 3 {
			break
		}
	}
	return erg	
}	

func main(){
	gfx.Fenster(560, 560) // öffnet das Grafikfenster mit weißem Hintergrund
	var bildBinaer [560][560]uint16
	bildBinaer = EinlesenZeichnung()
	fmt.Println(bildBinaer)
	gfx.Stiftfarbe(255,255,255)
	gfx.Cls()
	gfx.Stiftfarbe(0,0,0)
	for i := 0; i < 560; i++ {
		for j := 0; j < 560; j++ {
			if bildBinaer[i][j] == 1 {
				gfx.Punkt(uint16(i), uint16(j))
			}	
		}
	}
	for{}
}
