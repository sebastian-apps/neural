package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"net/http"
	"os"
	"runtime/pprof"
	"sync"
	"time"

	_ "net/http/pprof"
)

const (
	ParamsJsonFilePath   = "../weights_and_biases.json"
	TrainingDataFilePath = "../digit-recognizer/train.csv"
)

type Data struct {
	W1 [][]float64 `json:"w1"`
	B1 [][]float64 `json:"b1"`
	W2 [][]float64 `json:"w2"`
	B2 [][]float64 `json:"b2"`
}

func main() {
	// http://localhost:6060/debug/pprof/
	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	// runPredict()

	// predictSequential()

	predictMultiprocess()
}

func profile2() {
	f, err := os.Create("cpu.prof")
	if err != nil {
		log.Fatal("could not create CPU profile: ", err)
	}
	defer f.Close()
	if err := pprof.StartCPUProfile(f); err != nil {
		log.Fatal("could not start CPU profile: ", err)
	}
	defer pprof.StopCPUProfile()

	// Program logic here
	predictMultiprocess()

	// cli: go tool pprof cpu.prof
}

func runPredict() {
	// Load data
	jsonData, err := ioutil.ReadFile(ParamsJsonFilePath)
	if err != nil {
		fmt.Println("Error reading JSON file:", err)
		os.Exit(1)
	}

	var data Data
	err = json.Unmarshal(jsonData, &data)
	if err != nil {
		fmt.Println("Error unmarshalling JSON:", err)
		os.Exit(1)
	}

	// fmt.Println("\nW1:", data.W1)
	// fmt.Println("\nB1:", data.B1)
	// fmt.Println("\nW2:", data.W2)
	// fmt.Println("\nB2:", data.B2)

	// Load training data
	file, err := os.Open(TrainingDataFilePath)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Remove the header row
	records = records[1:]

	floatRecords, err := convertToFloat64(records)
	if err != nil {
		fmt.Println("Error converting records:", err)
		return
	}

	_, numCols := getDimensions2DFloat(floatRecords)

	// Seed the random number generator to ensure different results each run
	rand.Seed(time.Now().UnixNano())

	X_dev, Y_dev := loadDevData(floatRecords, numCols)

	devPredictions := makePredictions(X_dev, data.W1, data.B1, data.W2, data.B2)
	// fmt.Println("devPredictions:", devPredictions)
	acc := getAccuracy(devPredictions, Y_dev)
	fmt.Printf("Accuracy: %.2f\n", acc)

}

func predictSequential() {
	// Load data
	jsonData, err := ioutil.ReadFile(ParamsJsonFilePath)
	if err != nil {
		fmt.Println("Error reading JSON file:", err)
		os.Exit(1)
	}

	var data Data
	err = json.Unmarshal(jsonData, &data)
	if err != nil {
		fmt.Println("Error unmarshalling JSON:", err)
		os.Exit(1)
	}

	// Load training data
	file, err := os.Open(TrainingDataFilePath)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Remove the header row
	records = records[1:]

	floatRecords, err := convertToFloat64(records)
	if err != nil {
		fmt.Println("Error converting records:", err)
		return
	}

	_, numCols := getDimensions2DFloat(floatRecords)

	// Seed the random number generator to ensure different results each run
	rand.Seed(time.Now().UnixNano())

	X_dev, Y_dev := loadDevData(floatRecords, numCols)

	// Number of runs
	numRuns := 10

	// measure execution time
	start := time.Now()

	// runPredictions(data, floatRecords, numCols)
	for i := 0; i < numRuns; i++ {
		fmt.Println(i)
		// runPredictions(data, floatRecords, numCols)

		devPredictions := makePredictions(X_dev, data.W1, data.B1, data.W2, data.B2)
		acc := getAccuracy(devPredictions, Y_dev)
		fmt.Printf("Accuracy: %.2f\n", acc)

	}

	elapsed := time.Since(start).Microseconds()
	fmt.Printf("Execution time: %d µs\n", elapsed)
	elapsed = time.Since(start).Milliseconds()
	fmt.Printf("Execution time: %d ms\n", elapsed)

}

func predictMultiprocess() {
	// Load data
	jsonData, err := ioutil.ReadFile(ParamsJsonFilePath)
	if err != nil {
		fmt.Println("Error reading JSON file:", err)
		os.Exit(1)
	}

	var data Data
	err = json.Unmarshal(jsonData, &data)
	if err != nil {
		fmt.Println("Error unmarshalling JSON:", err)
		os.Exit(1)
	}

	// Load training data
	file, err := os.Open(TrainingDataFilePath)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Remove the header row
	records = records[1:]

	floatRecords, err := convertToFloat64(records)
	if err != nil {
		fmt.Println("Error converting records:", err)
		return
	}

	_, numCols := getDimensions2DFloat(floatRecords)

	X_dev, _ := loadDevData(floatRecords, numCols)

	start := time.Now()
	var wg sync.WaitGroup

	// Create 10 goroutines
	for i := 0; i < 1000; i++ {
		wg.Add(1)
		go worker(&wg, i, data, X_dev)
	}

	wg.Wait()
	elapsed := time.Since(start)
	fmt.Printf("Time taken: %s\n", elapsed)

}

func worker(wg *sync.WaitGroup, num int, model Data, input [][]float64) {
	defer wg.Done()
	fmt.Printf("Worker %d started\n", num)
	devPredictions := makePredictions(input, model.W1, model.B1, model.W2, model.B2)
	fmt.Printf("Worker %d finished.\n", num)
	fmt.Println("devPredictions:", devPredictions[:10])
}
