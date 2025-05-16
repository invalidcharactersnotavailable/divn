package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"

	"github.com/google/uuid"
)

// Network represents the entire dynamic routing network
type Network struct {
	UUID        string             `json:"uuid"`
	Name        string             `json:"name"`
	Layers      map[string]*Layer  `json:"layers"`
	NeuronCache map[string]*Neuron `json:"-"` // Global neuron cache for O(1) lookup
	mu          sync.RWMutex       `json:"-"` // Mutex for thread safety
	rand        *rand.Rand         `json:"-"` // Random source for deterministic operations
}

// NewNetwork creates a new network with the given name and random seed
func NewNetwork(name string, seed int64) (*Network, error) {
	if name == "" {
		return nil, fmt.Errorf("network name cannot be empty")
	}

	return &Network{
		UUID:        uuid.New().String(),
		Name:        name,
		Layers:      make(map[string]*Layer, 8),     // Pre-allocate with expected size
		NeuronCache: make(map[string]*Neuron, 1024), // Pre-allocate with expected size
		rand:        rand.New(rand.NewSource(seed)),
	}, nil
}

// AddLayer adds a layer to the network and updates the neuron cache
func (n *Network) AddLayer(layer *Layer) error {
	if layer == nil {
		return fmt.Errorf("layer cannot be nil")
	}

	n.mu.Lock()
	defer n.mu.Unlock()

	// Check if layer UUID already exists
	if _, exists := n.Layers[layer.UUID]; exists {
		return fmt.Errorf("layer with UUID %s already exists", layer.UUID)
	}

	n.Layers[layer.UUID] = layer

	// Update neuron cache with all neurons from this layer
	for neuronUUID, neuron := range layer.Neurons {
		if _, exists := n.NeuronCache[neuronUUID]; exists {
			return fmt.Errorf("neuron UUID conflict: %s already exists in network", neuronUUID)
		}
		n.NeuronCache[neuronUUID] = neuron
	}

	return nil
}

// GetNeuron returns a neuron from the cache by UUID
func (n *Network) GetNeuron(uuid string) (*Neuron, bool) {
	n.mu.RLock()
	defer n.mu.RUnlock()

	neuron, exists := n.NeuronCache[uuid]
	return neuron, exists
}

// ConnectLayers connects neurons between two layers
func (n *Network) ConnectLayers(sourceLayerUUID, targetLayerUUID string) error {
	n.mu.Lock()
	defer n.mu.Unlock()

	sourceLayer, sourceExists := n.Layers[sourceLayerUUID]
	if !sourceExists {
		return fmt.Errorf("source layer not found: %s", sourceLayerUUID)
	}

	targetLayer, targetExists := n.Layers[targetLayerUUID]
	if !targetExists {
		return fmt.Errorf("target layer not found: %s", targetLayerUUID)
	}

	// Get all target neuron UUIDs for efficient access
	targetNeurons := make([]string, 0, len(targetLayer.Neurons))
	for uuid := range targetLayer.Neurons {
		targetNeurons = append(targetNeurons, uuid)
	}

	// Connect each neuron in source layer to some neurons in target layer
	for _, sourceNeuron := range sourceLayer.Neurons {
		// Connect to a random subset of neurons in the target layer
		// This can be modified to use different connection strategies
		connectionCount := n.rand.Intn(5) + 1 // 1-5 connections per neuron

		// Shuffle and select a subset
		for i := range targetNeurons {
			j := n.rand.Intn(i + 1)
			targetNeurons[i], targetNeurons[j] = targetNeurons[j], targetNeurons[i]
		}

		// Connect to the selected subset
		for i := 0; i < connectionCount && i < len(targetNeurons); i++ {
			err := sourceNeuron.Connect(targetNeurons[i], n.rand.Float64())
			if err != nil {
				return fmt.Errorf("failed to connect %s to %s: %w", sourceNeuron.UUID, targetNeurons[i], err)
			}
		}
	}

	return nil
}

// ConnectAllLayers connects each layer to every other layer in the network
// This creates a fully connected network where every layer is connected to all others
func (n *Network) ConnectAllLayers() error {
	n.mu.Lock()

	// Get all layer UUIDs
	layerUUIDs := make([]string, 0, len(n.Layers))
	for uuid := range n.Layers {
		layerUUIDs = append(layerUUIDs, uuid)
	}

	n.mu.Unlock()

	// For each layer, connect it to all other layers
	totalConnections := 0
	for i, sourceUUID := range layerUUIDs {
		for j, targetUUID := range layerUUIDs {
			// Skip self-connections
			if i == j {
				continue
			}

			// Connect the layers
			err := n.ConnectLayers(sourceUUID, targetUUID)
			if err != nil {
				return fmt.Errorf("failed to connect layer %s to layer %s: %w",
					n.Layers[sourceUUID].Name, n.Layers[targetUUID].Name, err)
			}

			totalConnections++

			// Print progress for large networks
			if len(layerUUIDs) > 10 && totalConnections%100 == 0 {
				fmt.Printf("Created %d of %d layer connections...\n",
					totalConnections, len(layerUUIDs)*(len(layerUUIDs)-1))
			}
		}
	}

	return nil
}

// ProcessData processes input data through the network
// It returns the final transformed data and the path of neuron UUIDs taken
func (n *Network) ProcessData(startLayerUUID, startNeuronUUID string, data []float64, maxSteps int) ([]float64, []string, error) {
	if data == nil {
		return nil, nil, fmt.Errorf("input data cannot be nil")
	}

	if maxSteps <= 0 {
		return nil, nil, fmt.Errorf("maxSteps must be positive")
	}

	// Validate starting point
	n.mu.RLock()
	startLayer, layerExists := n.Layers[startLayerUUID]
	if !layerExists {
		n.mu.RUnlock()
		return nil, nil, fmt.Errorf("starting layer not found: %s", startLayerUUID)
	}
	n.mu.RUnlock()

	startNeuron, neuronExists := startLayer.GetNeuron(startNeuronUUID)
	if !neuronExists {
		return nil, nil, fmt.Errorf("starting neuron not found: %s", startNeuronUUID)
	}

	// Initialize processing
	currentNeuron := startNeuron
	currentData := data
	path := make([]string, 0, maxSteps) // Pre-allocate with expected size
	path = append(path, startNeuronUUID)

	// Process data through the network
	for step := 0; step < maxSteps; step++ {
		// Transform data using current neuron
		var err error
		currentData, err = currentNeuron.TransformData(currentData)
		if err != nil {
			return nil, path, fmt.Errorf("error transforming data at step %d: %w", step, err)
		}

		// Find the connected neuron with lowest resistance
		var nextNeuronUUID string
		lowestResistance := math.MaxFloat64

		// Get all connections from the current neuron
		connections := currentNeuron.GetConnections()

		for connectedUUID := range connections {
			// Use the neuron cache for O(1) lookup
			n.mu.RLock()
			connectedNeuron, exists := n.NeuronCache[connectedUUID]
			n.mu.RUnlock()

			if exists {
				resistance := connectedNeuron.GetResistance()
				if resistance < lowestResistance {
					lowestResistance = resistance
					nextNeuronUUID = connectedUUID
				}
			}
		}

		// If no connected neurons or reached a dead end
		if nextNeuronUUID == "" {
			break
		}

		// Move to the next neuron using the cache
		n.mu.RLock()
		nextNeuron, exists := n.NeuronCache[nextNeuronUUID]
		n.mu.RUnlock()

		if !exists {
			return nil, path, fmt.Errorf("neuron not found in cache: %s", nextNeuronUUID)
		}

		currentNeuron = nextNeuron
		path = append(path, nextNeuronUUID)
	}

	return currentData, path, nil
}

// ProcessDataParallel processes multiple data inputs in parallel
// Returns a slice of results and paths
func (n *Network) ProcessDataParallel(startLayerUUID, startNeuronUUID string, dataInputs [][]float64, maxSteps int) ([][]float64, [][]string, []error) {
	if len(dataInputs) == 0 {
		return nil, nil, nil
	}

	results := make([][]float64, len(dataInputs))
	paths := make([][]string, len(dataInputs))
	errors := make([]error, len(dataInputs))

	// Use a wait group to synchronize goroutines
	var wg sync.WaitGroup
	wg.Add(len(dataInputs))

	// Process each input in a separate goroutine
	for i, data := range dataInputs {
		go func(idx int, inputData []float64) {
			defer wg.Done()
			result, path, err := n.ProcessData(startLayerUUID, startNeuronUUID, inputData, maxSteps)
			results[idx] = result
			paths[idx] = path
			errors[idx] = err
		}(i, data)
	}

	// Wait for all goroutines to complete
	wg.Wait()

	return results, paths, errors
}

// SaveToFile saves the network to a JSON file
func (n *Network) SaveToFile(filePath string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	// Create a temporary file in the same directory
	tempFile := filePath + ".tmp"

	n.mu.RLock()
	defer n.mu.RUnlock()

	// Marshal network to JSON
	data, err := json.MarshalIndent(n, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal network to JSON: %w", err)
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

// LoadNetworkFromFile loads a network from a JSON file
func LoadNetworkFromFile(filePath string, seed int64) (*Network, error) {
	// Read file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file %s: %w", filePath, err)
	}

	// Unmarshal JSON to network
	var network Network
	if err := json.Unmarshal(data, &network); err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON from %s: %w", filePath, err)
	}

	// Initialize non-serialized fields with appropriate capacity
	neuronCount := 0
	for _, layer := range network.Layers {
		neuronCount += len(layer.Neurons)
	}

	network.NeuronCache = make(map[string]*Neuron, neuronCount)
	network.rand = rand.New(rand.NewSource(seed))

	// Rebuild the neuron cache
	for _, layer := range network.Layers {
		for neuronUUID, neuron := range layer.Neurons {
			network.NeuronCache[neuronUUID] = neuron
		}
	}

	return &network, nil
}
