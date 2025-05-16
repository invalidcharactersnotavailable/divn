package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sync"

	"github.com/google/uuid"
)

// Layer represents a collection of neurons in a 2D grid
type Layer struct {
	UUID    string             `json:"uuid"`
	Name    string             `json:"name"`
	Width   int                `json:"width"`   // Width of the neuron grid
	Height  int                `json:"height"`  // Height of the neuron grid
	Neurons map[string]*Neuron `json:"neurons"` // Map of neuron UUIDs to neurons
	mu      sync.RWMutex       `json:"-"`       // Mutex for thread safety
}

// NewLayer creates a new layer with the specified dimensions and vector size
func NewLayer(name string, width, height, vectorDims int, r *rand.Rand) (*Layer, error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}

	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid dimensions: width=%d, height=%d", width, height)
	}

	if r == nil {
		return nil, fmt.Errorf("random source cannot be nil")
	}

	// Calculate capacity for pre-allocation
	capacity := width * height

	layer := &Layer{
		UUID:    uuid.New().String(),
		Name:    name,
		Width:   width,
		Height:  height,
		Neurons: make(map[string]*Neuron, capacity),
	}

	// Create neurons in a grid pattern
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			neuron, err := NewNeuron(vectorDims, r)
			if err != nil {
				return nil, fmt.Errorf("failed to create neuron at (%d,%d): %w", x, y, err)
			}
			layer.Neurons[neuron.UUID] = neuron
		}
	}

	return layer, nil
}

// NewLayerWithUniqueVectors creates a layer with neurons having unique vector dimensions
func NewLayerWithUniqueVectors(name string, width, height, vectorDims int, baseSeed int64) (*Layer, error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}

	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid dimensions: width=%d, height=%d", width, height)
	}

	// Calculate capacity for pre-allocation
	capacity := width * height

	layer := &Layer{
		UUID:    uuid.New().String(),
		Name:    name,
		Width:   width,
		Height:  height,
		Neurons: make(map[string]*Neuron, capacity),
	}

	// Create neurons with unique vectors
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			seed := baseSeed + int64(x*10000+y) // Create a unique seed based on position
			neuron, err := NewNeuronWithUniqueVector(vectorDims, seed)
			if err != nil {
				return nil, fmt.Errorf("failed to create neuron at (%d,%d): %w", x, y, err)
			}
			layer.Neurons[neuron.UUID] = neuron
		}
	}

	return layer, nil
}

// GetNeuron returns a neuron by UUID
func (l *Layer) GetNeuron(uuid string) (*Neuron, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	neuron, exists := l.Neurons[uuid]
	return neuron, exists
}

// ConnectInternalNeurons connects neurons within the layer
func (l *Layer) ConnectInternalNeurons(r *rand.Rand) error {
	if r == nil {
		return fmt.Errorf("random source cannot be nil")
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	// Get all neuron UUIDs
	neuronUUIDs := make([]string, 0, len(l.Neurons))
	for uuid := range l.Neurons {
		neuronUUIDs = append(neuronUUIDs, uuid)
	}

	// Connect each neuron to a random subset of other neurons
	for _, sourceUUID := range neuronUUIDs {
		source, exists := l.Neurons[sourceUUID]
		if !exists {
			return fmt.Errorf("source neuron not found: %s", sourceUUID)
		}

		// Determine number of connections (random between 1 and 5)
		connectionCount := r.Intn(5) + 1

		// Shuffle neuron UUIDs for random selection
		for i := range neuronUUIDs {
			j := r.Intn(i + 1)
			neuronUUIDs[i], neuronUUIDs[j] = neuronUUIDs[j], neuronUUIDs[i]
		}

		// Connect to the selected subset
		for i := 0; i < connectionCount && i < len(neuronUUIDs); i++ {
			targetUUID := neuronUUIDs[i]
			if targetUUID == sourceUUID {
				continue // Skip self
			}

			// Random initial connection strength
			err := source.Connect(targetUUID, r.Float64())
			if err != nil {
				return fmt.Errorf("failed to connect %s to %s: %w", sourceUUID, targetUUID, err)
			}
		}
	}

	return nil
}

// SaveToFile saves a single layer to a JSON file
func (l *Layer) SaveToFile(filePath string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	// Create a temporary file in the same directory
	tempFile := filePath + ".tmp"

	l.mu.RLock()
	defer l.mu.RUnlock()

	// Marshal layer to JSON
	data, err := json.MarshalIndent(l, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal layer to JSON: %w", err)
	}

	// Write to temporary file
	if err := os.WriteFile(tempFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write to temporary file %s: %w", tempFile, err)
	}

	// Rename temporary file to target file (atomic operation)
	if err := os.Rename(tempFile, filePath); err != nil {
		// Try to clean up the temporary file
		os.Remove(tempFile)
		return fmt.Errorf("failed to rename temporary file to %s: %w", filePath, err)
	}

	return nil
}

// LoadLayerFromFile loads a layer from a JSON file
func LoadLayerFromFile(filePath string) (*Layer, error) {
	// Read file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file %s: %w", filePath, err)
	}

	// Unmarshal JSON to layer
	var layer Layer
	if err := json.Unmarshal(data, &layer); err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON from %s: %w", filePath, err)
	}

	return &layer, nil
}
