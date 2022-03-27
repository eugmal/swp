//Laudes
//26.03.22
//Package zur Darstellung von Bildumwandlungen von Neuronen 


package ldarstellungen

type LDarstellung interface {
	//Vor.: keine
	//Erg.: der Slice a ist mithilfe der Konstanten "flip" aus der darstellenImpl-Datei in einen Binärcode-Slice umgewandelt
	Umwandeln (a []float64) (b []uint8)
	
	//Vor.: keine
	//Erg.: ein Binärcode-Slice b aus Umwandeln() ist in einem gfx.Fenster dargestellt, wobei eine 1 in ein schwarzes Quadrat und einen 0 in ein türkises Quadrat dargestellt wird. 
	Darstellen (b []uint8)
	
}
