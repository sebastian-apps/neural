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
	"time"

	_ "net/http/pprof"
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

	process()
}

func process() {
	// Load data
	jsonData, err := ioutil.ReadFile("../weights_and_biases.json")
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

	fmt.Println("\nW1:", data.W1)
	fmt.Println("\nB1:", data.B1)
	fmt.Println("\nW2:", data.W2)
	fmt.Println("\nB2:", data.B2)

	// Load training data
	file, err := os.Open("../digit-recognizer/train.csv")
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

	// Determine the dimensions of records
	numRows := len(records)
	numCols := 0
	if numRows > 0 {
		numCols = len(records[0]) // Assuming uniformity in the slice
	}
	fmt.Printf("Shape of records: %d rows x %d columns\n", numRows, numCols)

	floatRecords, err := convertToFloat64(records)
	if err != nil {
		fmt.Println("Error converting records:", err)
		return
	}

	_, _ = getDimensions2DFloat(floatRecords)

	// Seed the random number generator to ensure different results each run
	rand.Seed(time.Now().UnixNano())

	runPredictions(data, floatRecords, numCols)

	// for i := 0; i < 10000; i++ {
	// 	fmt.Println(i)
	// 	runPredictions(data, floatRecords, numCols)
	// }
}
