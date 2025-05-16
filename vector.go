package main

import (
	"fmt"
	"math/rand"
)

// Vector represents a flexible-dimension vector for neuron operations
type Vector struct {
	Dimensions []float64 `json:"dimensions"` // Values for each dimension
}

// NewVector creates a new vector with the specified number of dimensions
func NewVector(dims int) (*Vector, error) {
	if dims <= 0 {
		return nil, fmt.Errorf("vector dimensions must be positive, got %d", dims)
	}

	v := &Vector{
		Dimensions: make([]float64, dims),
	}
	return v, nil
}

// NewRandomVector creates a new vector with random values using provided random source
func NewRandomVector(dims int, r *rand.Rand) (*Vector, error) {
	if r == nil {
		return nil, fmt.Errorf("random source cannot be nil")
	}

	if dims <= 0 {
		return nil, fmt.Errorf("vector dimensions must be positive, got %d", dims)
	}

	v, err := NewVector(dims)
	if err != nil {
		return nil, err
	}

	for i := range v.Dimensions {
		v.Dimensions[i] = r.Float64()*2 - 1 // Values between -1 and 1
	}
	return v, nil
}

// NewUniqueVector creates a vector with a unique dimension configuration
// The uniqueness is determined by the seed value
func NewUniqueVector(dims int, seed int64) (*Vector, error) {
	if dims <= 0 {
		return nil, fmt.Errorf("vector dimensions must be positive, got %d", dims)
	}

	r := rand.New(rand.NewSource(seed))
	v, err := NewVector(dims)
	if err != nil {
		return nil, err
	}

	for i := range v.Dimensions {
		v.Dimensions[i] = r.Float64()*2 - 1 // Values between -1 and 1
	}
	return v, nil
}

// Transform applies the vector transformation to input data
// Returns a slice with length equal to the vector's dimension count
// Optimized for performance with pre-allocated result slice
func (v *Vector) Transform(data []float64) ([]float64, error) {
	if v == nil {
		return nil, fmt.Errorf("vector is nil")
	}

	if data == nil {
		return nil, fmt.Errorf("input data is nil")
	}

	// Pre-allocate result slice
	result := make([]float64, len(v.Dimensions))

	// Determine the minimum length to avoid bounds checking in the loop
	minLen := len(data)
	if len(v.Dimensions) < minLen {
		minLen = len(v.Dimensions)
	}

	// Process the overlapping portion
	for i := 0; i < minLen; i++ {
		result[i] = v.Dimensions[i] * data[i]
	}

	// Copy remaining dimensions if vector is longer than data
	for i := minLen; i < len(v.Dimensions); i++ {
		result[i] = v.Dimensions[i]
	}

	return result, nil
}

// GetDimensions returns the number of dimensions in the vector
func (v *Vector) GetDimensions() int {
	if v == nil {
		return 0
	}
	return len(v.Dimensions)
}

// Resize changes the number of dimensions in the vector
// If expanding, new dimensions are initialized with random values
// If shrinking, excess dimensions are discarded
func (v *Vector) Resize(newDims int, r *rand.Rand) error {
	if v == nil {
		return fmt.Errorf("vector is nil")
	}

	if newDims <= 0 {
		return fmt.Errorf("new dimensions must be positive, got %d", newDims)
	}

	if r == nil {
		return fmt.Errorf("random source cannot be nil")
	}

	currentDims := len(v.Dimensions)

	// If no change in dimensions, return early
	if newDims == currentDims {
		return nil
	}

	// Create new dimensions slice
	newDimensions := make([]float64, newDims)

	// Copy existing dimensions
	copyLen := currentDims
	if newDims < currentDims {
		copyLen = newDims
	}

	copy(newDimensions[:copyLen], v.Dimensions[:copyLen])

	// Initialize new dimensions with random values
	for i := currentDims; i < newDims; i++ {
		newDimensions[i] = r.Float64()*2 - 1 // Values between -1 and 1
	}

	v.Dimensions = newDimensions
	return nil
}
