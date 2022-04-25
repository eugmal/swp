//heidi hoehn
//Zweck: Zahlen mit der Maus zeichnen und in einem 2dimensionalen Array speichern
//Datum: 30.03.2022

package main
import ("gfx"; "fmt")

const vergroesserung uint16 = 20 

// Vor.: Grafikfenster muss offen sein.
// Eff.: Benutzer zeichnet mit der Maus eine Zahl, indem er die linke Maustaste gedrückt hält und die Maus bewegt
// 		 Wird die rechte Maustaste gedrückt, signalisiert der Benutzer: "meine Zeichung ist fertig!"
// Erg.: 2D-Array, das das gezeichnete Bild mit in Schwarz(= 1)-Weiß(= 0) kodiert

func EinlesenZeichnung() [560][560]uint16{ 
	var erg [560][560]uint16  //Variable für das Ergebnis erstellen
	gfx.Stiftfarbe(0,0,0) // setzt Stiftfarbe auf schwarz
	gfx.Schreibe(0,0,  "Zeichne bitte eine Zahl!")
	gfx.Schreibe(0,12,  "Wenn du fertig bist, drueck die rechte Maustaste!")
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

//Vor.: keine
//Eff.: ein Grafikfenster mit weißem Hintergrund wird erstellt, die Funktion "EinlesenZeichnung()" wird aufgerufen, damit Benutzer eine Zahl zeichnen kann
//		Wenn der Benutzer fertig ist, wird zuerst die Kodierung des Bildes im Terminal ausgegeben und dann aus der Kodierung wieder das gezeichnete Bild rekonstruiert.
//		Der letzte Schritt stellt sicher, dass die Kodierung korrekt ist.
func main(){
	fmt.Println(gfx.GibFont())
	gfx.Fenster(560, 560) // öffnet das Grafikfenster mit weißem Hintergrund
	var bildBinaer [560][560]uint16
	bildBinaer = EinlesenZeichnung()
	fmt.Println(bildBinaer)		//gib die Kodierung im Terminal aus
	// rekonstruiere aus der Kodierung das gezeichnete Bild
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
	for{} // Endlosschleife, damit sich das Programm nicht automatisch schließt und man das rekonstruierte Bild in Ruhe ansehen kann

}
