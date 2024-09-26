package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Definición de estructuras

type Node struct {
	FeatureIndex int
	Threshold    float64
	Left         *Node
	Right        *Node
	Prediction   int
}

type DecisionTree struct {
	Root *Node
}

type RandomForest struct {
	Trees []DecisionTree
}

type Dataset struct {
	Features [][]float64
	Labels   []int
}

// Función para dividir el dataset

func splitDataset(data Dataset, featureIndex int, threshold float64) (left Dataset, right Dataset) {
	for i, feature := range data.Features {
		if feature[featureIndex] <= threshold {
			left.Features = append(left.Features, data.Features[i])
			left.Labels = append(left.Labels, data.Labels[i])
		} else {
			right.Features = append(right.Features, data.Features[i])
			right.Labels = append(right.Labels, data.Labels[i])
		}
	}
	return left, right
}

func uniqueValues(features [][]float64, featureIndex int) []float64 {
	valuesMap := make(map[float64]bool)
	for _, feature := range features {
		valuesMap[feature[featureIndex]] = true
	}

	var uniqueVals []float64
	for value := range valuesMap {
		uniqueVals = append(uniqueVals, value)
	}
	return uniqueVals
}

func calculateGini(left, right Dataset) float64 {
	totalSize := len(left.Labels) + len(right.Labels)

	giniLeft := 1.0
	giniRight := 1.0

	for i, group := range []Dataset{left, right} {
		if len(group.Labels) == 0 {
			continue
		}

		score := 0.0
		labelCount := make(map[int]int)

		for _, label := range group.Labels {
			labelCount[label]++
		}

		for _, count := range labelCount {
			proportion := float64(count) / float64(len(group.Labels))
			score += proportion * proportion
		}

		gini := 1.0 - score

		if i == 0 {
			giniLeft = gini * float64(len(group.Labels)) / float64(totalSize)
		} else {
			giniRight = gini * float64(len(group.Labels)) / float64(totalSize)
		}
	}

	return giniLeft + giniRight
}

func majorityClass(labels []int) int {
	classCount := make(map[int]int)
	for _, label := range labels {
		classCount[label]++
	}

	majority := -1
	maxCount := 0
	for class, count := range classCount {
		if count > maxCount {
			maxCount = count
			majority = class
		}
	}
	return majority
}

// Función para encontrar la mejor división del dataset

func findBestSplit(data Dataset) (bestFeature int, bestThreshold float64, bestGini float64) {
	bestGini = 1.0
	for featureIndex := 0; featureIndex < len(data.Features[0]); featureIndex++ {
		thresholds := uniqueValues(data.Features, featureIndex)
		for _, threshold := range thresholds {
			left, right := splitDataset(data, featureIndex, threshold)
			gini := calculateGini(left, right)
			if gini < bestGini {
				bestGini = gini
				bestFeature = featureIndex
				bestThreshold = threshold
			}
		}
	}
	return bestFeature, bestThreshold, bestGini
}

// Función para construir el árbol de decisión

func buildTree(data Dataset, maxDepth int, minSize int) *Node {
	if len(data.Labels) <= minSize || maxDepth == 0 {
		return &Node{Prediction: majorityClass(data.Labels)}
	}

	bestFeature, bestThreshold, bestGini := findBestSplit(data)
	if bestGini == 1.0 {
		return &Node{Prediction: majorityClass(data.Labels)}
	}

	leftData, rightData := splitDataset(data, bestFeature, bestThreshold)
	leftNode := buildTree(leftData, maxDepth-1, minSize)
	rightNode := buildTree(rightData, maxDepth-1, minSize)

	return &Node{
		FeatureIndex: bestFeature,
		Threshold:    bestThreshold,
		Left:         leftNode,
		Right:        rightNode,
	}
}

// Predicción usando el árbol de decisión

func (n *Node) Predict(features []float64) int {
	if n.Left == nil && n.Right == nil {
		return n.Prediction
	}

	if features[n.FeatureIndex] <= n.Threshold {
		return n.Left.Predict(features)
	} else {
		return n.Right.Predict(features)
	}
}

// Función para tomar una muestra aleatoria con reemplazo (bootstrap)

func bootstrapSample(data Dataset) Dataset {
	var sample Dataset
	for i := 0; i < len(data.Features); i++ {
		index := rand.Intn(len(data.Features))
		sample.Features = append(sample.Features, data.Features[index])
		sample.Labels = append(sample.Labels, data.Labels[index])
	}
	return sample
}

// Entrenamiento del Random Forest

func (forest *RandomForest) Train(data Dataset, numTrees int, maxDepth int, minSize int) {
	forest.Trees = make([]DecisionTree, numTrees)
	var wg sync.WaitGroup

	for i := 0; i < numTrees; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sampleData := bootstrapSample(data)
			tree := DecisionTree{}
			tree.Root = buildTree(sampleData, maxDepth, minSize)
			forest.Trees[i] = tree
		}(i)
	}
	wg.Wait()
}

// Predicción usando el Random Forest

func (forest *RandomForest) Predict(features []float64) int {
	predictions := make([]int, len(forest.Trees))
	var wg sync.WaitGroup

	for i, tree := range forest.Trees {
		wg.Add(1)
		go func(i int, tree DecisionTree) {
			defer wg.Done()
			predictions[i] = tree.Root.Predict(features)
		}(i, tree)
	}
	wg.Wait()
	return majorityVote(predictions)
}

// Función para la votación de mayoría

func majorityVote(predictions []int) int {
	voteCount := make(map[int]int)
	for _, prediction := range predictions {
		voteCount[prediction]++
	}

	maxVote := 0
	maxClass := -1
	for class, count := range voteCount {
		if count > maxVote {
			maxVote = count
			maxClass = class
		}
	}
	return maxClass
}

// Función para cargar el dataset desde un archivo CSV

func loadDataset(filePath string) (Dataset, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return Dataset{}, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return Dataset{}, err
	}

	var data Dataset
	for _, record := range records[1:] { // Omitir la cabecera
		var features []float64

		// Convertir las características relevantes a valores numéricos
		year, _ := strconv.ParseFloat(record[0], 64)
		month, _ := strconv.ParseFloat(record[1], 64)
		sex := 0.0
		region := regionMap[record[2]]
		if record[14] == "FEMENINO" {
			sex = 1.0
		}
		ageGroup := convertAgeGroup(record[15])

		// Añadir características seleccionadas
		features = append(features, year, month, region, sex, ageGroup)

		// Etiqueta (cantidad de atenciones)
		label, _ := strconv.Atoi(record[16])

		data.Features = append(data.Features, features)
		data.Labels = append(data.Labels, label)
	}

	return data, nil
}

// Conversión del grupo de edad a número

func convertAgeGroup(group string) float64 {
	switch group {
	case "00 - 04 AÑOS":
		return 0.0
	case "05 - 11 AÑOS":
		return 1.0
	case "12 - 17 AÑOS":
		return 2.0
	case "18 - 29 AÑOS":
		return 3.0
	case "30 - 59 AÑOS":
		return 4.0
	case "60 - más AÑOS":
		return 5.0
	default:
		return -1.0
	}
}

var regionMap = map[string]float64{
	"AMAZONAS":           1,
	"ANCASH":             2,
	"APURIMAC":           3,
	"AREQUIPA":           4,
	"AYACUCHO":           5,
	"CAJAMARCA":          6,
	"CALLAO":             7,
	"CUSCO":              8,
	"HUANCAVELICA":       9,
	"HUANUCO":            10,
	"ICA":                11,
	"JUNIN":              12,
	"LA LIBERTAD":        13,
	"LAMBAYEQUE":         14,
	"LIMA METROPOLITANA": 15,
	"LIMA REGION":        16,
	"LORETO":             17,
	"MADRE DE DIOS":      18,
	"MOQUEGUA":           19,
	"PASCO":              20,
	"PIURA":              21,
	"PUNO":               22,
	"SAN MARTIN":         23,
	"TACNA":              24,
	"TUMBES":             25,
	"UCAYALI":            26,
}

// Función principal con menú de opciones

func main() {
	var forest RandomForest
	var numTrees, maxDepth, minSize int

	fmt.Println("Bienvenido al Bosque Aleatorio en Go")
	fmt.Print("Ingrese el número de árboles: ")
	fmt.Scan(&numTrees)

	fmt.Print("Ingrese la profundidad máxima del árbol: ")
	fmt.Scan(&maxDepth)

	fmt.Print("Ingrese el tamaño mínimo del nodo hoja: ")
	fmt.Scan(&minSize)

	// Cargar dataset desde un archivo
	data, err := loadDataset("datita.csv")
	if err != nil {
		fmt.Println("Error cargando el dataset:", err)
		return
	}

	startTime := time.Now()
	forest.Train(data, numTrees, maxDepth, minSize)
	elapsedTime := time.Since(startTime)

	fmt.Printf("Modelo entrenado en %s\n", elapsedTime)
	fmt.Println("Ingrese datos para predecir:")

	features := []float64{}

	var year, month, region, sex, ageGroup float64

	fmt.Print("Ingrese el año: ")
	fmt.Scan(&year)
	features = append(features, year)

	fmt.Print("Ingrese el mes: ")
	fmt.Scan(&month)
	features = append(features, month)

	fmt.Scanln()
	// Leer región con bufio.Reader debido a problemas con Scan y Scanln
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Ingrese la región (AMAZONAS, LIMA METROPOLITANA, etc.): ")
	regionName, _ := reader.ReadString('\n')
	regionName = strings.TrimSpace(regionName)
	fmt.Printf("Región ingresada: '%s'\n", regionName)
	region, ok := regionMap[regionName]
	if !ok {
		fmt.Println("Región no válida, verifique la entrada.")
		return
	}
	features = append(features, region)

	fmt.Print("Ingrese el sexo (0 para Masculino, 1 para Femenino): ")
	fmt.Scan(&sex)
	features = append(features, sex)

	fmt.Print("05 - 11 AÑOS:1 | 12 - 17 AÑOS:2 | 18 - 29 AÑOS:3 | 30 - 59 AÑOS:4 | 60 - más AÑOS:5\n")
	fmt.Print("Ingrese el grupo de edad:")
	fmt.Scan(&ageGroup)
	features = append(features, ageGroup)

	prediction := forest.Predict(features)
	if prediction == 1 {
		fmt.Printf("Tus datos se asocian con una atención médica registrada ✅")
	} else {
		fmt.Printf("Tus datos NO asocian con una atención médica registrada ❌")
	}
}
