package main

import (
	"fmt"
	"github.com/eugmal/swp/ai"
	"github.com/eugmal/swp/gui"
	"github.com/eugmal/swp/mnistLoad"
)

func main() {

	var antwort string
	var path string
	var pathData string
	fmt.Println("Ist es das erste mal, dass du dieses Programm ausfuehrst? Eingabe: j fuer ja n fuer nein: ")

	fmt.Scanln(&antwort)
	if antwort == "j" {
		fmt.Println("Gib den Pfad an, an dem du die mnist Daten hast(von der original Seite verpackt und alle 4 Dateien in einem Ordner): ")
		fmt.Scanln(&pathData)
		fmt.Println("Das Netz wird jetzt trainiert. Dies kann einige Minuten dauern. Einfach warten.")
		trainSet, _ := mnistLoad.LadenSlice(pathData)
		w1neu, w2neu := ai.GradientDescendFull(trainSet, 5, 0.1)
		fmt.Println("Das Netz ist trainiert. Gib jetzt den Pfad ein, am dem die tranierten Gewichte gespeichert werden sollen (der Ordner sollte vor der Eingabe angelegt werden): ")
		fmt.Scanln(&path)
		ai.GewichteSpeichern(*w1neu, *w2neu, path)
		fmt.Println("Die Gewichte des trainierten Netzes sind jetzt am angegebenen Pfad gespeichert.")

	} else {
		fmt.Println("Gib den Pfad ein, wo du die Gewichte gespeichert hast: ")
		fmt.Scanln(&path)
	}

	w1, w2 := ai.GewichteLaden(path)
	fmt.Println(w1)
	fmt.Println("Jetzt kannst du 5 mal eine Zahl malen und bestimmen lassen, welche Zahl es ist. Male die Zahl, druecke die rechte Maustaste, wenn du fertig bist und im Anschluss eine beliebige Taste auf der Tastatur, um die Antwort im Terminal zu erhalten.")
	for i := 0; i < 5; i++ {
		eingabeBild := gui.ZahlMalen()
		antwort := ai.Vorhersage(*eingabeBild, *w1, *w2)
		fmt.Println("Die gezeichnete Zahl ist eine: ", antwort)
	}
}
