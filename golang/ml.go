package main

// train in python, paralellize is python?, save to json, load in go, run prediction in go, parallelize in go?

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"net/http"
	_ "net/http/pprof"
)

func pipeline() { // rename to main()

	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	fmt.Println("Running neural_network_from_scratch_in_go")

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

	// Convert [][]string to [][]float64
	floatRecords, err := convertToFloat64(records)
	if err != nil {
		fmt.Println("Error converting records:", err)
		return
	}

	_, _ = getDimensions2DFloat(floatRecords)

	// Seed the random number generator to ensure different results each run
	rand.Seed(time.Now().UnixNano())

	// Randomly shuffle the rows of the 2D slice
	rand.Shuffle(len(floatRecords), func(i, j int) {
		floatRecords[i], floatRecords[j] = floatRecords[j], floatRecords[i]
	})

	// // Print the converted records
	// fmt.Println(floatRecords)

	dataSliced := floatRecords[:1000]
	dataTransposed := transpose(dataSliced)
	dataTransposedInt := transposeInt(float64ToInt(dataSliced))
	Y_dev := dataTransposedInt[0]
	X_dev := dataTransposed[1:numCols]

	// Dividing each element of X_dev by 255
	for i := range X_dev {
		for j := range X_dev[i] {
			X_dev[i][j] /= 255.0
		}
	}

	// Printing Y_dev and a portion of X_dev to verify
	fmt.Println("Y_dev:", Y_dev[:5])           // Print first 5 elements of Y_dev for brevity
	fmt.Println("X_dev sample:", X_dev[0][:5]) // Print first 5 elements of the first row of X_dev for brevity

	// Slicing data[1000:num_rows] and Transposing
	dataSliced = floatRecords[1000:numRows]
	dataTrain := transpose(dataSliced)
	dataTrainInt := transposeInt(float64ToInt((dataSliced)))
	Y_train := dataTrainInt[0]
	X_train := dataTrain[1:numCols] // Assuming num_cols is within bounds

	// Dividing each element of X_train by 255
	for i := range X_train {
		for j := range X_train[i] {
			X_train[i][j] /= 255.0
		}
	}

	// Determining the shape of X_train
	mNum := len(X_train[0]) // Number of columns in X_train

	// Printing Y_train
	fmt.Println("Y_train:", Y_train)

	// Printing the shape of X_train
	fmt.Printf("mNum: %d", mNum)

	W1, b1, W2, b2 := gradientDescent(X_train, Y_train, 0.10, 500, mNum)

	testPrediction(0, W1, b1, W2, b2, X_train, Y_train)
	testPrediction(1, W1, b1, W2, b2, X_train, Y_train)
	testPrediction(2, W1, b1, W2, b2, X_train, Y_train)
	testPrediction(3, W1, b1, W2, b2, X_train, Y_train)

	devPredictions := makePredictions(X_dev, W1, b1, W2, b2)
	print(getAccuracy(devPredictions, Y_dev))

}

func runPredictions(data Data, floatRecords [][]float64, numCols int) {
	// Randomly shuffle the rows of the 2D slice
	rand.Shuffle(len(floatRecords), func(i, j int) {
		floatRecords[i], floatRecords[j] = floatRecords[j], floatRecords[i]
	})

	// // Print the converted records
	// fmt.Println(floatRecords)

	dataSliced := floatRecords[:1000]
	dataTransposed := transpose(dataSliced)
	dataTransposedInt := transposeInt(float64ToInt(dataSliced))
	Y_dev := dataTransposedInt[0]
	X_dev := dataTransposed[1:numCols]

	// Dividing each element of X_dev by 255
	for i := range X_dev {
		for j := range X_dev[i] {
			X_dev[i][j] /= 255.0
		}
	}

	// Printing Y_dev and a portion of X_dev to verify
	fmt.Println("Y_dev:", Y_dev[:5])           // Print first 5 elements of Y_dev for brevity
	fmt.Println("X_dev sample:", X_dev[0][:5]) // Print first 5 elements of the first row of X_dev for brevity

	devPredictions := makePredictions(X_dev, data.W1, data.B1, data.W2, data.B2)
	acc := getAccuracy(devPredictions, Y_dev)
	fmt.Printf("Accuracy: %.2f\n", acc)
}

func makePredictions(X, W1, b1, W2, b2 [][]float64) []int {
	_, _, _, A2 := forwardProp(W1, b1, W2, b2, X)
	predictions := getPredictions(A2)
	return predictions
}

// getAccuracy calculates the accuracy as the proportion of correct predictions.
// It assumes predictions and Y are slices of the same length.
func getAccuracy(predictions, Y []int) float64 {
	if len(predictions) == 0 || len(predictions) != len(Y) {
		return 0.0 // Return 0 if slices are empty or not of the same length.
	}

	correctCount := 0
	for i, prediction := range predictions {
		if prediction == Y[i] {
			correctCount++
		}
	}

	accuracy := float64(correctCount) / float64(len(Y))
	return accuracy
}

// getPredictions returns the indices of the maximum values in each column of a 2D slice.
func getPredictions(A2 [][]float64) []int {
	if len(A2) == 0 || len(A2[0]) == 0 {
		return nil // Return nil if A2 is empty or has empty sub-slices.
	}

	rows := len(A2)
	cols := len(A2[0])
	maxIndices := make([]int, cols)

	for j := 0; j < cols; j++ {
		maxIndex := 0
		maxValue := A2[0][j]
		for i := 1; i < rows; i++ {
			if A2[i][j] > maxValue {
				maxValue = A2[i][j]
				maxIndex = i
			}
		}
		maxIndices[j] = maxIndex
	}

	return maxIndices
}

// forwardProp performs forward propagation through a 2-layer neural network.
func forwardProp(W1, b1, W2, b2, X [][]float64) (
	Z1, A1, Z2, A2 [][]float64,
) {
	intermed := matrixDot(W1, X)
	Z1 = broadcastAdd(intermed, b1)
	A1 = ReLU(Z1)
	Z2 = broadcastAdd(matrixDot(W2, A1), b2)
	A2 = softmax(Z2)
	return Z1, A1, Z2, A2
}

// initParams initializes and returns matrices W1, b1, W2, b2
func initParams() ([][]float64, [][]float64, [][]float64, [][]float64) {
	rand.Seed(time.Now().UnixNano())

	W1 := make([][]float64, 10)
	b1 := make([][]float64, 10)
	for i := range W1 {
		W1[i] = make([]float64, 784)
		b1[i] = make([]float64, 1)
		for j := range W1[i] {
			W1[i][j] = rand.Float64() - 0.5
		}
		b1[i][0] = rand.Float64() - 0.5
	}

	W2 := make([][]float64, 10)
	b2 := make([][]float64, 10)
	for i := range W2 {
		W2[i] = make([]float64, 10)
		b2[i] = make([]float64, 1)
		for j := range W2[i] {
			W2[i][j] = rand.Float64() - 0.5
		}
		b2[i][0] = rand.Float64() - 0.5
	}

	return W1, b1, W2, b2
}

// backwardProp performs backward propagation through a 2-layer neural network.
func backwardProp(Z1, A1, A2, W2, X [][]float64, Y []int, mNum int) (
	dW1, db1, dW2, db2 [][]float64,
) {
	oneHotY := oneHot2(Y, len(A2))
	fmt.Println("Passed 0")

	dZ2 := matrixSubtract(A2, oneHotY)
	dW2 = matrixDot(transpose(dZ2), A1)
	for i := range dW2 {
		for j := range dW2[i] {
			dW2[i][j] /= float64(mNum)
		}
	}
	fmt.Println("Passed 1")
	db2 = make([][]float64, 1)
	db2[0] = matrixMean(dZ2, mNum)

	fmt.Println("Passed 1.5")

	dZ1 := matrixDot(W2, dZ2)
	reluDerivZ1 := ReLUderiv(Z1)
	for i := range dZ1 {
		for j := range dZ1[i] {
			dZ1[i][j] *= reluDerivZ1[i][j]
		}
	}
	fmt.Println("Passed 2")

	dW1 = matrixDot(transpose(dZ1), X)
	fmt.Println("Passed 2.5")
	for i := range dW1 {
		for j := range dW1[i] {
			dW1[i][j] /= float64(mNum)
		}
	}
	fmt.Println("Passed 3")

	db1Sum := matrixMean(dZ1, mNum)
	db1 = make([][]float64, len(db1Sum))
	for i, val := range db1Sum {
		db1[i] = []float64{val}
	}
	fmt.Println("Passed 4")

	return dW1, db1, dW2, db2
}

// updateParams updates the parameters by subtracting the gradients scaled by alpha.
// Assumes W1, b1, W2, b2, db1, dW2, db2 are all slices of slices of float64.
// alpha is a float64 scalar.
func updateParams(W1, b1, W2, b2, db1, dW2, db2 [][]float64, alpha float64) ([][]float64, [][]float64, [][]float64, [][]float64) {
	// Update b1
	for i := range b1 {
		for j := range b1[i] {
			b1[i][j] = b1[i][j] - alpha*db1[i][j]
		}
	}

	// Update W2
	for i := range W2 {
		for j := range W2[i] {
			W2[i][j] = W2[i][j] - alpha*dW2[i][j]
		}
	}

	// Update b2
	for i := range b2 {
		for j := range b2[i] {
			b2[i][j] = b2[i][j] - alpha*db2[i][j]
		}
	}

	return W1, b1, W2, b2
}

func gradientDescent(
	X [][]float64,
	Y []int,
	alpha float64,
	iterations int,
	mNum int,
) (
	W1, b1, W2, b2 [][]float64,
) {
	W1, b1, W2, b2 = initParams()
	for i := 0; i < iterations; i++ {
		fmt.Println("i:", i)
		Z1, A1, _, A2 := forwardProp(W1, b1, W2, b2, X)
		fmt.Println("passed forwardProp")
		_, db1, dW2, db2 := backwardProp(Z1, A1, A2, W2, X, Y, mNum)
		fmt.Println("passed backwardProp")
		W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, db1, dW2, db2, alpha)
		fmt.Println("passed updateParams")
		if i%10 == 0 {
			fmt.Println("Iteration:", i)
			predictions := getPredictions(A2)
			fmt.Println(getAccuracy(predictions, Y))
		}
	}
	return W1, b1, W2, b2
}

func testPrediction(
	index int,
	W1, b1, W2, b2, X_train [][]float64,
	Y_train []int,
) {
	current_image := extractColumn(X_train, index)
	prediction := makePredictions(current_image, W1, b1, W2, b2)
	label := Y_train[index]
	fmt.Println("Prediction: ", prediction)
	fmt.Println("Label: ", label)
}

// Refactor redundancy
func loadDevData(floatRecords [][]float64, numCols int) ([][]float64, []int) {
	// Randomly shuffle the rows of the 2D slice
	rand.Shuffle(len(floatRecords), func(i, j int) {
		floatRecords[i], floatRecords[j] = floatRecords[j], floatRecords[i]
	})

	// // Print the converted records
	// fmt.Println(floatRecords)

	dataSliced := floatRecords[:1000]
	dataTransposed := transpose(dataSliced)
	dataTransposedInt := transposeInt(float64ToInt(dataSliced))
	Y_dev := dataTransposedInt[0]
	X_dev := dataTransposed[1:numCols]

	// Dividing each element of X_dev by 255
	for i := range X_dev {
		for j := range X_dev[i] {
			X_dev[i][j] /= 255.0
		}
	}

	// Printing Y_dev and a portion of X_dev to verify
	fmt.Println("Y_dev:", Y_dev[:5])           // Print first 5 elements of Y_dev for brevity
	fmt.Println("X_dev sample:", X_dev[0][:5]) // Print first 5 elements of the first row of X_dev for brevity

	return X_dev, Y_dev
}
