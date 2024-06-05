package main

// train in python, parallelize is python?, save to json, load in go, run prediction in go, parallelize in go?

import (
	"fmt"
	"math"
	_ "net/http/pprof"
	"strconv"
)

// convertToFloat64 takes a 2D slice of strings and converts it to a 2D slice of float64.
// It returns an error if any string cannot be converted to float64.
func convertToFloat64(records [][]string) ([][]float64, error) {
	floatRecords := make([][]float64, len(records))

	for i, row := range records {
		floatRow := make([]float64, len(row))
		for j, strVal := range row {
			floatVal, err := strconv.ParseFloat(strVal, 64)
			if err != nil {
				return nil, err
			}
			floatRow[j] = floatVal
		}
		floatRecords[i] = floatRow
	}

	return floatRecords, nil
}

// transpose function transposes a 2D slice of float64
func transpose(slice [][]float64) [][]float64 {
	if len(slice) == 0 {
		return nil
	}
	result := make([][]float64, len(slice[0]))
	for i := range result {
		result[i] = make([]float64, len(slice))
		for j := range slice {
			result[i][j] = slice[j][i]
		}
	}
	return result
}

// transposeInt function transposes a 2D slice of int
func transposeInt(slice [][]int) [][]int {
	if len(slice) == 0 {
		return nil
	}
	result := make([][]int, len(slice[0]))
	for i := range result {
		result[i] = make([]int, len(slice))
		for j := range slice {
			result[i][j] = slice[j][i]
		}
	}
	return result
}

// float64ToInt converts a 2D float64 slice to a 2D int slice.
func float64ToInt(data [][]float64) [][]int {
	var result [][]int
	for _, row := range data {
		var intRow []int
		for _, value := range row {
			intRow = append(intRow, int(value))
		}
		result = append(result, intRow)
	}
	return result
}

func getDimensions2DFloat(records [][]float64) (int, int) {
	numRows := len(records)
	numCols := 0
	if numRows > 0 {
		numCols = len(records[0]) // Assuming uniformity in the slice
	}
	fmt.Printf("Shape of floatRecords: %d rows x %d columns\n", numRows, numCols)
	return numRows, numCols
}

// matrixDot performs matrix multiplication. It assumes the dimensions are valid for multiplication.
func matrixDot(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range result {
		result[i] = make([]float64, len(b[0]))
		for j := range result[i] {
			for k := range a[i] {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result
}

// broadcastAdd adds two matrices, broadcasting the smaller one if necessary.
// It assumes b is the matrix to be broadcast and that broadcasting is compatible.
func broadcastAdd(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			// Assuming b is (N, 1) and a is (N, M), broadcast b across the columns of a
			result[i][j] = a[i][j] + b[i][0]
		}
	}
	return result
}

// matrixAdd performs element-wise addition of two matrices.
func matrixAdd(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] + b[i][j]
		}
	}
	return result
}

// extractColumn extracts a column from a 2D slice and returns it as a 2D slice with a single column.
func extractColumn(matrix [][]float64, columnIndex int) [][]float64 {
	var column [][]float64
	for _, row := range matrix {
		// Check if the columnIndex is within the bounds of the row
		if columnIndex >= 0 && columnIndex < len(row) {
			column = append(column, []float64{row[columnIndex]})
		}
	}
	return column
}

// matrixSubtract performs element-wise subtraction between two matrices.
func matrixSubtract(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] - b[i][j]
		}
	}
	return result
}

// matrixMean calculates the mean of each row in a 2D slice.
func matrixMean(a [][]float64, m int) []float64 {
	sum := make([]float64, len(a))
	for i, row := range a {
		for _, val := range row {
			sum[i] += val
		}
		sum[i] /= float64(m)
	}
	return sum
}

// ReLU applies the Rectified Linear Unit function element-wise to a slice of float64.
// change Z name to matrix?
func ReLU(Z [][]float64) [][]float64 {
	result := make([][]float64, len(Z))
	for i, row := range Z {
		result[i] = make([]float64, len(row))
		for j, val := range row {
			if val > 0 {
				result[i][j] = val
			} else {
				result[i][j] = 0
			}
		}
	}
	return result
}

// softmax applies the softmax function to each column of a matrix.
func softmax(matrix [][]float64) [][]float64 {
	result := make([][]float64, len(matrix))
	sumExp := make([]float64, len(matrix[0]))
	for i := range matrix {
		result[i] = make([]float64, len(matrix[i]))
		for j := range matrix[i] {
			sumExp[j] += math.Exp(matrix[i][j])
		}
	}
	for i := range matrix {
		for j := range matrix[i] {
			result[i][j] = math.Exp(matrix[i][j]) / sumExp[j]
		}
	}
	return result
}

// ReLUderiv calculates the derivative of the ReLU function over a 2D slice.
func ReLUderiv(Z [][]float64) [][]float64 {
	// Initialize the result 2D slice with the same dimensions as Z.
	result := make([][]float64, len(Z))
	for i := range result {
		result[i] = make([]float64, len(Z[i]))
	}

	// Iterate over the elements of Z.
	for i, row := range Z {
		for j, val := range row {
			// If the element is greater than 0, set the corresponding element in result to 1.
			// Otherwise, set it to 0.
			if val > 0 {
				result[i][j] = 1
			} else {
				result[i][j] = 0
			}
		}
	}

	return result
}

// oneHot takes a slice of integers and returns its one-hot encoding.
// The function assumes that Y contains non-negative integers.
func oneHot(Y []int) [][]int {
	// Find the maximum value in Y to determine the size of the second dimension of the one-hot encoding.
	maxY := 0
	for _, value := range Y {
		if value > maxY {
			maxY = value
		}
	}

	// Create a slice of slices for the one-hot encoding.
	// The number of rows is determined by the maximum value in Y plus 1 (for zero-based indexing),
	// and the number of columns is the length of Y.
	oneHotY := make([][]int, maxY+1)
	for i := range oneHotY {
		oneHotY[i] = make([]int, len(Y))
	}

	// Set the appropriate index to 1 for each element in Y.
	for i, value := range Y {
		oneHotY[value][i] = 1
	}

	return oneHotY
}

// oneHot encodes a 1D slice of integers into a 2D one-hot encoded slice.
func oneHot2(y []int, classes int) [][]float64 {
	result := make([][]float64, classes)
	for i := range result {
		result[i] = make([]float64, len(y))
		for j, label := range y {
			if i == label {
				result[i][j] = 1
			}
		}
	}
	return result
}
