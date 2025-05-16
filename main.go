package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"

	"github.com/google/uuid"
)

// Neuron represents a processing unit in the network with UUID-based identification
type Neuron struct {
	UUID        string             `json:"uuid"`        // Unique identifier using UUID
	Value       *Vector            `json:"value"`       // Vector for data transformation with selectable dimensions
	Resistance  float64            `json:"resistance"`  // Resistance value for routing (lower = higher priority)
	Connections map[string]float64 `json:"connections"` // Map of connected neuron UUIDs to connection strengths
	mu          sync.RWMutex       `json:"-"`           // Mutex for thread safety
}

// NewNeuron creates a new neuron with the specified vector dimensions using provided random source
func NewNeuron(vectorDims int, r *rand.Rand) (*Neuron, error) {
	if r == nil {
		return nil, fmt.Errorf("random source cannot be nil")
	}

	if vectorDims <= 0 {
		return nil, fmt.Errorf("vector dimensions must be positive, got %d", vectorDims)
	}

	// Generate a UUID for the neuron
	neuronUUID := uuid.New().String()

	vector, err := NewRandomVector(vectorDims, r)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector: %w", err)
	}

	return &Neuron{
		UUID:        neuronUUID,
		Value:       vector,
		Resistance:  r.Float64(),                 // Initial random resistance
		Connections: make(map[string]float64, 8), // Pre-allocate with expected size
	}, nil
}

// NewNeuronWithUniqueVector creates a neuron with a uniquely dimensioned vector
func NewNeuronWithUniqueVector(vectorDims int, seed int64) (*Neuron, error) {
	if vectorDims <= 0 {
		return nil, fmt.Errorf("vector dimensions must be positive, got %d", vectorDims)
	}

	// Generate a UUID for the neuron
	neuronUUID := uuid.New().String()

	// Create a deterministic random source for resistance
	r := rand.New(rand.NewSource(seed))

	vector, err := NewUniqueVector(vectorDims, seed)
	if err != nil {
		return nil, fmt.Errorf("failed to create unique vector: %w", err)
	}

	return &Neuron{
		UUID:        neuronUUID,
		Value:       vector,
		Resistance:  r.Float64(),                 // Deterministic resistance based on seed
		Connections: make(map[string]float64, 8), // Pre-allocate with expected size
	}, nil
}

// Connect establishes a connection to another neuron with a given strength
// If the connection already exists, the strength is added to the existing value
func (n *Neuron) Connect(targetUUID string, strength float64) error {
	if targetUUID == "" {
		return fmt.Errorf("target UUID cannot be empty")
	}

	n.mu.Lock()
	defer n.mu.Unlock()

	// Add to existing connection strength if it exists
	if existingStrength, exists := n.Connections[targetUUID]; exists {
		n.Connections[targetUUID] = existingStrength + strength
	} else {
		n.Connections[targetUUID] = strength
	}

	return nil
}

// GetConnections returns a copy of the neuron's connections
func (n *Neuron) GetConnections() map[string]float64 {
	n.mu.RLock()
	defer n.mu.RUnlock()

	connections := make(map[string]float64, len(n.Connections))
	for uuid, strength := range n.Connections {
		connections[uuid] = strength
	}

	return connections
}

// TransformData applies the neuron's vector transformation to input data
func (n *Neuron) TransformData(data []float64) ([]float64, error) {
	if n == nil {
		return nil, fmt.Errorf("neuron is nil")
	}

	if data == nil {
		return nil, fmt.Errorf("input data is nil")
	}

	n.mu.RLock()
	defer n.mu.RUnlock()

	return n.Value.Transform(data)
}

// GetResistance returns the neuron's resistance value
func (n *Neuron) GetResistance() float64 {
	n.mu.RLock()
	defer n.mu.RUnlock()

	return n.Resistance
}

// SetResistance updates the neuron's resistance value
func (n *Neuron) SetResistance(resistance float64) {
	n.mu.Lock()
	defer n.mu.Unlock()

	n.Resistance = resistance
}

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

// BinarySerializer provides methods for binary serialization and deserialization
type BinarySerializer struct {
	// Version of the binary format, for future compatibility
	Version uint16
}

// NewBinarySerializer creates a new binary serializer
func NewBinarySerializer() *BinarySerializer {
	return &BinarySerializer{
		Version: 1, // Initial version
	}
}

// SaveNetworkToBinary saves a network to a binary file
func (bs *BinarySerializer) SaveNetworkToBinary(network *Network, filePath string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	// Create a temporary file in the same directory
	tempFile := filePath + ".tmp"
	file, err := os.Create(tempFile)
	if err != nil {
		return fmt.Errorf("failed to create temporary file %s: %w", tempFile, err)
	}
	defer file.Close()

	// Write the binary data
	if err := bs.writeNetwork(file, network); err != nil {
		// Clean up the temporary file
		os.Remove(tempFile)
		return fmt.Errorf("failed to write network to binary file: %w", err)
	}

	// Rename temporary file to target file (atomic operation)
	if err := os.Rename(tempFile, filePath); err != nil {
		// Try to clean up the temporary file
		os.Remove(tempFile)
		return fmt.Errorf("failed to rename temporary file to %s: %w", filePath, err)
	}

	return nil
}

// LoadNetworkFromBinary loads a network from a binary file
func (bs *BinarySerializer) LoadNetworkFromBinary(filePath string, seed int64) (*Network, error) {
	// Open the file
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %w", filePath, err)
	}
	defer file.Close()

	// Read the binary data
	network, err := bs.readNetwork(file, seed)
	if err != nil {
		return nil, fmt.Errorf("failed to read network from binary file: %w", err)
	}

	return network, nil
}

// writeNetwork writes a network to a binary writer
func (bs *BinarySerializer) writeNetwork(w io.Writer, network *Network) error {
	// Write format version
	if err := binary.Write(w, binary.LittleEndian, bs.Version); err != nil {
		return fmt.Errorf("failed to write format version: %w", err)
	}

	// Write network UUID
	if err := bs.writeString(w, network.UUID); err != nil {
		return fmt.Errorf("failed to write network UUID: %w", err)
	}

	// Write network name
	if err := bs.writeString(w, network.Name); err != nil {
		return fmt.Errorf("failed to write network name: %w", err)
	}

	// Write number of layers
	layerCount := uint32(len(network.Layers))
	if err := binary.Write(w, binary.LittleEndian, layerCount); err != nil {
		return fmt.Errorf("failed to write layer count: %w", err)
	}

	// Write each layer
	for _, layer := range network.Layers {
		if err := bs.writeLayer(w, layer); err != nil {
			return fmt.Errorf("failed to write layer: %w", err)
		}
	}

	return nil
}

// readNetwork reads a network from a binary reader
func (bs *BinarySerializer) readNetwork(r io.Reader, seed int64) (*Network, error) {
	// Read format version
	var version uint16
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("failed to read format version: %w", err)
	}

	// Check version compatibility
	if version > bs.Version {
		return nil, fmt.Errorf("unsupported binary format version: %d (supported: %d)", version, bs.Version)
	}

	// Read network UUID
	networkUUID, err := bs.readString(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read network UUID: %w", err)
	}

	// Read network name
	networkName, err := bs.readString(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read network name: %w", err)
	}

	// Create a new network with the read UUID and name
	network := &Network{
		UUID:        networkUUID,
		Name:        networkName,
		Layers:      make(map[string]*Layer),
		NeuronCache: make(map[string]*Neuron),
		rand:        rand.New(rand.NewSource(seed)),
	}

	// Read number of layers
	var layerCount uint32
	if err := binary.Read(r, binary.LittleEndian, &layerCount); err != nil {
		return nil, fmt.Errorf("failed to read layer count: %w", err)
	}

	// Read each layer
	for i := uint32(0); i < layerCount; i++ {
		layer, err := bs.readLayer(r)
		if err != nil {
			return nil, fmt.Errorf("failed to read layer %d: %w", i, err)
		}

		// Add layer to network
		network.Layers[layer.UUID] = layer

		// Add neurons to cache
		for neuronUUID, neuron := range layer.Neurons {
			network.NeuronCache[neuronUUID] = neuron
		}
	}

	return network, nil
}

// writeLayer writes a layer to a binary writer
func (bs *BinarySerializer) writeLayer(w io.Writer, layer *Layer) error {
	// Write layer UUID
	if err := bs.writeString(w, layer.UUID); err != nil {
		return fmt.Errorf("failed to write layer UUID: %w", err)
	}

	// Write layer name
	if err := bs.writeString(w, layer.Name); err != nil {
		return fmt.Errorf("failed to write layer name: %w", err)
	}

	// Write layer dimensions
	if err := binary.Write(w, binary.LittleEndian, uint32(layer.Width)); err != nil {
		return fmt.Errorf("failed to write layer width: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(layer.Height)); err != nil {
		return fmt.Errorf("failed to write layer height: %w", err)
	}

	// Write number of neurons
	neuronCount := uint32(len(layer.Neurons))
	if err := binary.Write(w, binary.LittleEndian, neuronCount); err != nil {
		return fmt.Errorf("failed to write neuron count: %w", err)
	}

	// Write each neuron
	for _, neuron := range layer.Neurons {
		if err := bs.writeNeuron(w, neuron); err != nil {
			return fmt.Errorf("failed to write neuron: %w", err)
		}
	}

	return nil
}

// readLayer reads a layer from a binary reader
func (bs *BinarySerializer) readLayer(r io.Reader) (*Layer, error) {
	// Read layer UUID
	layerUUID, err := bs.readString(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read layer UUID: %w", err)
	}

	// Read layer name
	layerName, err := bs.readString(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read layer name: %w", err)
	}

	// Read layer dimensions
	var width, height uint32
	if err := binary.Read(r, binary.LittleEndian, &width); err != nil {
		return nil, fmt.Errorf("failed to read layer width: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &height); err != nil {
		return nil, fmt.Errorf("failed to read layer height: %w", err)
	}

	// Create a new layer
	layer := &Layer{
		UUID:    layerUUID,
		Name:    layerName,
		Width:   int(width),
		Height:  int(height),
		Neurons: make(map[string]*Neuron),
	}

	// Read number of neurons
	var neuronCount uint32
	if err := binary.Read(r, binary.LittleEndian, &neuronCount); err != nil {
		return nil, fmt.Errorf("failed to read neuron count: %w", err)
	}

	// Read each neuron
	for i := uint32(0); i < neuronCount; i++ {
		neuron, err := bs.readNeuron(r)
		if err != nil {
			return nil, fmt.Errorf("failed to read neuron %d: %w", i, err)
		}

		// Add neuron to layer
		layer.Neurons[neuron.UUID] = neuron
	}

	return layer, nil
}

// writeNeuron writes a neuron to a binary writer
func (bs *BinarySerializer) writeNeuron(w io.Writer, neuron *Neuron) error {
	// Write neuron UUID
	if err := bs.writeString(w, neuron.UUID); err != nil {
		return fmt.Errorf("failed to write neuron UUID: %w", err)
	}

	// Write neuron resistance
	if err := binary.Write(w, binary.LittleEndian, float32(neuron.Resistance)); err != nil {
		return fmt.Errorf("failed to write neuron resistance: %w", err)
	}

	// Write vector dimensions
	dimCount := uint32(len(neuron.Value.Dimensions))
	if err := binary.Write(w, binary.LittleEndian, dimCount); err != nil {
		return fmt.Errorf("failed to write vector dimension count: %w", err)
	}

	// Write vector values (as float32 for efficiency)
	for _, dim := range neuron.Value.Dimensions {
		if err := binary.Write(w, binary.LittleEndian, float32(dim)); err != nil {
			return fmt.Errorf("failed to write vector dimension: %w", err)
		}
	}

	// Write number of connections
	connectionCount := uint32(len(neuron.Connections))
	if err := binary.Write(w, binary.LittleEndian, connectionCount); err != nil {
		return fmt.Errorf("failed to write connection count: %w", err)
	}

	// Write each connection
	for targetUUID, strength := range neuron.Connections {
		// Write target UUID
		if err := bs.writeString(w, targetUUID); err != nil {
			return fmt.Errorf("failed to write connection target UUID: %w", err)
		}

		// Write connection strength
		if err := binary.Write(w, binary.LittleEndian, float32(strength)); err != nil {
			return fmt.Errorf("failed to write connection strength: %w", err)
		}
	}

	return nil
}

// readNeuron reads a neuron from a binary reader
func (bs *BinarySerializer) readNeuron(r io.Reader) (*Neuron, error) {
	// Read neuron UUID
	neuronUUID, err := bs.readString(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read neuron UUID: %w", err)
	}

	// Read neuron resistance
	var resistance float32
	if err := binary.Read(r, binary.LittleEndian, &resistance); err != nil {
		return nil, fmt.Errorf("failed to read neuron resistance: %w", err)
	}

	// Read vector dimensions
	var dimCount uint32
	if err := binary.Read(r, binary.LittleEndian, &dimCount); err != nil {
		return nil, fmt.Errorf("failed to read vector dimension count: %w", err)
	}

	// Create vector
	vector := &Vector{
		Dimensions: make([]float64, dimCount),
	}

	// Read vector values
	for i := uint32(0); i < dimCount; i++ {
		var dim float32
		if err := binary.Read(r, binary.LittleEndian, &dim); err != nil {
			return nil, fmt.Errorf("failed to read vector dimension %d: %w", i, err)
		}
		vector.Dimensions[i] = float64(dim)
	}

	// Create neuron
	neuron := &Neuron{
		UUID:        neuronUUID,
		Value:       vector,
		Resistance:  float64(resistance),
		Connections: make(map[string]float64),
	}

	// Read number of connections
	var connectionCount uint32
	if err := binary.Read(r, binary.LittleEndian, &connectionCount); err != nil {
		return nil, fmt.Errorf("failed to read connection count: %w", err)
	}

	// Read each connection
	for i := uint32(0); i < connectionCount; i++ {
		// Read target UUID
		targetUUID, err := bs.readString(r)
		if err != nil {
			return nil, fmt.Errorf("failed to read connection target UUID: %w", err)
		}

		// Read connection strength
		var strength float32
		if err := binary.Read(r, binary.LittleEndian, &strength); err != nil {
			return nil, fmt.Errorf("failed to read connection strength: %w", err)
		}

		// Add connection
		neuron.Connections[targetUUID] = float64(strength)
	}

	return neuron, nil
}

// writeString writes a string to a binary writer
func (bs *BinarySerializer) writeString(w io.Writer, s string) error {
	// Write string length
	length := uint32(len(s))
	if err := binary.Write(w, binary.LittleEndian, length); err != nil {
		return fmt.Errorf("failed to write string length: %w", err)
	}

	// Write string data
	if _, err := w.Write([]byte(s)); err != nil {
		return fmt.Errorf("failed to write string data: %w", err)
	}

	return nil
}

// readString reads a string from a binary reader
func (bs *BinarySerializer) readString(r io.Reader) (string, error) {
	// Read string length
	var length uint32
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", fmt.Errorf("failed to read string length: %w", err)
	}

	// Read string data
	data := make([]byte, length)
	if _, err := io.ReadFull(r, data); err != nil {
		return "", fmt.Errorf("failed to read string data: %w", err)
	}

	return string(data), nil
}

// BinarySerializationBenchmark benchmarks the binary serialization and deserialization
func BinarySerializationBenchmark(network *Network, filePath string, seed int64) {
	fmt.Println("\nRunning binary serialization benchmark...")

	// Create binary serializer
	serializer := NewBinarySerializer()

	// Benchmark binary save
	binarySaveStart := time.Now()
	binaryFilePath := filePath + ".bin"

	err := serializer.SaveNetworkToBinary(network, binaryFilePath)
	if err != nil {
		fmt.Printf("Error saving network to binary: %v\n", err)
		return
	}

	binarySaveTime := time.Since(binarySaveStart)

	// Get binary file size
	binaryFileInfo, err := os.Stat(binaryFilePath)
	if err != nil {
		fmt.Printf("Error getting binary file info: %v\n", err)
		return
	}
	binaryFileSize := binaryFileInfo.Size()

	// Benchmark binary load
	binaryLoadStart := time.Now()

	_, err = serializer.LoadNetworkFromBinary(binaryFilePath, seed)
	if err != nil {
		fmt.Printf("Error loading network from binary: %v\n", err)
		return
	}

	binaryLoadTime := time.Since(binaryLoadStart)

	// Benchmark JSON save for comparison
	jsonSaveStart := time.Now()
	jsonFilePath := filePath + ".json"

	err = network.SaveToFile(jsonFilePath)
	if err != nil {
		fmt.Printf("Error saving network to JSON: %v\n", err)
		return
	}

	jsonSaveTime := time.Since(jsonSaveStart)

	// Get JSON file size
	jsonFileInfo, err := os.Stat(jsonFilePath)
	if err != nil {
		fmt.Printf("Error getting JSON file info: %v\n", err)
		return
	}
	jsonFileSize := jsonFileInfo.Size()

	// Benchmark JSON load for comparison
	jsonLoadStart := time.Now()

	_, err = LoadNetworkFromFile(jsonFilePath, seed)
	if err != nil {
		fmt.Printf("Error loading network from JSON: %v\n", err)
		return
	}

	jsonLoadTime := time.Since(jsonLoadStart)

	// Print results
	fmt.Println("\nSerialization Benchmark Results:")
	fmt.Printf("Binary save time: %v\n", binarySaveTime)
	fmt.Printf("Binary load time: %v\n", binaryLoadTime)
	fmt.Printf("Binary file size: %v bytes\n", binaryFileSize)
	fmt.Printf("JSON save time: %v\n", jsonSaveTime)
	fmt.Printf("JSON load time: %v\n", jsonLoadTime)
	fmt.Printf("JSON file size: %v bytes\n", jsonFileSize)

	// Calculate improvements
	saveSpeedup := float64(jsonSaveTime) / float64(binarySaveTime)
	loadSpeedup := float64(jsonLoadTime) / float64(binaryLoadTime)
	sizeReduction := float64(jsonFileSize) / float64(binaryFileSize)

	fmt.Printf("\nBinary vs JSON Comparison:")
	fmt.Printf("Save speedup: %.2fx faster\n", saveSpeedup)
	fmt.Printf("Load speedup: %.2fx faster\n", loadSpeedup)
	fmt.Printf("File size reduction: %.2fx smaller\n", sizeReduction)
}

// Command-line flags structure
type CommandFlags struct {
	Seed         int64
	LayerCount   int
	NeuronSize   int
	VectorDims   int
	FilePath     string
	CPUProfile   string
	MemProfile   string
	ParallelProc bool
	FullConnect  bool // Flag for full layer interconnection
	UseBinary    bool // Flag for using binary serialization
}

func main() {
	// Check if we have enough arguments
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	// Get the command
	command := os.Args[1]

	// Remove the command from arguments to simplify flag parsing
	os.Args = append(os.Args[:1], os.Args[2:]...)

	// Create flags structure with default values
	flags := CommandFlags{
		Seed:         time.Now().UnixNano(),
		LayerCount:   3,
		NeuronSize:   8,
		VectorDims:   4,
		FilePath:     "network",
		CPUProfile:   "",
		MemProfile:   "",
		ParallelProc: false,
		FullConnect:  true, // Default to full connectivity
		UseBinary:    true, // Default to binary serialization
	}

	// Set up command line flags
	flag.Int64Var(&flags.Seed, "seed", flags.Seed, "Random seed for reproducible network generation")
	flag.IntVar(&flags.LayerCount, "layers", flags.LayerCount, "Number of layers in the network")
	flag.IntVar(&flags.NeuronSize, "size", flags.NeuronSize, "Size of neuron grid (size x size)")
	flag.IntVar(&flags.VectorDims, "dims", flags.VectorDims, "Dimensions of neuron vectors")
	flag.StringVar(&flags.FilePath, "file", flags.FilePath, "Base file path for save/load operations (without extension)")
	flag.StringVar(&flags.CPUProfile, "cpuprofile", flags.CPUProfile, "Write CPU profile to file")
	flag.StringVar(&flags.MemProfile, "memprofile", flags.MemProfile, "Write memory profile to file")
	flag.BoolVar(&flags.ParallelProc, "parallel", flags.ParallelProc, "Use parallel processing for data")
	flag.BoolVar(&flags.FullConnect, "fullconnect", flags.FullConnect, "Connect each layer to all other layers")
	flag.BoolVar(&flags.UseBinary, "binary", flags.UseBinary, "Use binary format for network serialization (faster)")

	// Parse flags
	flag.Parse()

	// Start CPU profiling if requested
	if flags.CPUProfile != "" {
		f, err := os.Create(flags.CPUProfile)
		if err != nil {
			fmt.Printf("Error creating CPU profile: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()

		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Printf("Error starting CPU profile: %v\n", err)
			os.Exit(1)
		}
		defer pprof.StopCPUProfile()
	}

	// Handle commands
	switch command {
	case "create":
		createNetwork(flags)
	case "load":
		loadNetwork(flags)
	case "test":
		testNetwork(flags)
	case "benchmark":
		benchmarkNetwork(flags)
	case "benchmark-serialization":
		benchmarkSerialization(flags)
	case "help":
		printUsage()
	default:
		fmt.Printf("Unknown command: %s\n", command)
		printUsage()
		os.Exit(1)
	}

	// Write memory profile if requested
	if flags.MemProfile != "" {
		f, err := os.Create(flags.MemProfile)
		if err != nil {
			fmt.Printf("Error creating memory profile: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()

		runtime.GC() // Get up-to-date statistics
		if err := pprof.WriteHeapProfile(f); err != nil {
			fmt.Printf("Error writing memory profile: %v\n", err)
			os.Exit(1)
		}
	}
}

func printUsage() {
	fmt.Println("Dynamic Intraconnected Vector Network (DIVN)")
	fmt.Println("\nUsage:")
	fmt.Println("  divn <command> [flags]")
	fmt.Println("\nCommands:")
	fmt.Println("  create                Create a new network with specified parameters")
	fmt.Println("  load                  Load a network from file")
	fmt.Println("  test                  Run a test of network operations")
	fmt.Println("  benchmark             Run performance benchmarks")
	fmt.Println("  benchmark-serialization  Compare binary vs JSON serialization performance")
	fmt.Println("  help                  Show this help message")
	fmt.Println("\nFlags:")
	fmt.Println("  -seed int       Random seed for reproducible network generation (default: current time)")
	fmt.Println("  -layers int     Number of layers in the network (default: 3)")
	fmt.Println("  -size int       Size of neuron grid (size x size) (default: 8)")
	fmt.Println("  -dims int       Dimensions of neuron vectors (default: 4)")
	fmt.Println("  -file string    Base file path for save/load operations (default: \"network\")")
	fmt.Println("  -binary         Use binary format for network serialization (default: true)")
	fmt.Println("  -cpuprofile     Write CPU profile to specified file")
	fmt.Println("  -memprofile     Write memory profile to specified file")
	fmt.Println("  -parallel       Use parallel processing for data (default: false)")
	fmt.Println("  -fullconnect    Connect each layer to all other layers (default: true)")
	fmt.Println("\nExamples:")
	fmt.Println("  divn create -layers 5 -size 10 -dims 8")
	fmt.Println("  divn create -layers 128 -size 64 -dims 16 -fullconnect=true -binary=true")
	fmt.Println("  divn load -file network -binary=true")
	fmt.Println("  divn test -parallel")
	fmt.Println("  divn benchmark-serialization -size 16 -dims 16")
}

func createNetwork(flags CommandFlags) {
	fmt.Println("Creating a new dynamic network...")
	fmt.Printf("Seed: %d, Layers: %d, Grid Size: %dx%d, Vector Dimensions: %d, Full Connectivity: %v, Binary Format: %v\n",
		flags.Seed, flags.LayerCount, flags.NeuronSize, flags.NeuronSize, flags.VectorDims, flags.FullConnect, flags.UseBinary)

	startTime := time.Now()

	// Initialize random source with seed for reproducibility
	r := rand.New(rand.NewSource(flags.Seed))

	// Create a new network
	network, err := NewNetwork("test_network", flags.Seed)
	if err != nil {
		fmt.Printf("Error creating network: %v\n", err)
		os.Exit(1)
	}

	// Create and add layers
	layersByName := make(map[string]*Layer) // Store layers by name for easier lookup

	for i := 0; i < flags.LayerCount; i++ {
		layerName := fmt.Sprintf("layer%d", i+1)

		var layer *Layer
		var err error

		// Make some layers with unique vectors for variety
		if i%2 == 0 {
			layer, err = NewLayer(layerName, flags.NeuronSize, flags.NeuronSize, flags.VectorDims, r)
		} else {
			layer, err = NewLayerWithUniqueVectors(layerName, flags.NeuronSize, flags.NeuronSize, flags.VectorDims, flags.Seed+int64(i*1000))
		}

		if err != nil {
			fmt.Printf("Error creating layer %s: %v\n", layerName, err)
			os.Exit(1)
		}

		// Connect neurons within the layer
		err = layer.ConnectInternalNeurons(r)
		if err != nil {
			fmt.Printf("Error connecting neurons in layer %s: %v\n", layerName, err)
			os.Exit(1)
		}

		// Add layer to network
		err = network.AddLayer(layer)
		if err != nil {
			fmt.Printf("Error adding layer %s to network: %v\n", layerName, err)
			os.Exit(1)
		}

		// Store layer by name for easier lookup
		layersByName[layerName] = layer

		fmt.Printf("Added layer %s: %dx%d grid with %d neurons\n",
			layerName, layer.Width, layer.Height, len(layer.Neurons))
	}

	// Connect layers based on connectivity mode
	if flags.FullConnect {
		// Full connectivity: each layer connects to all other layers
		fmt.Println("Connecting all layers to each other (full connectivity)...")
		connectionStartTime := time.Now()

		err := network.ConnectAllLayers()
		if err != nil {
			fmt.Printf("Error connecting layers: %v\n", err)
			os.Exit(1)
		}

		connectionTime := time.Since(connectionStartTime)
		fmt.Printf("All layers fully connected (took %v)\n", connectionTime)
	} else {
		// Sequential connectivity: only connect adjacent layers and add circular reference
		fmt.Println("Connecting layers sequentially...")

		// Connect layers sequentially
		for i := 0; i < flags.LayerCount-1; i++ {
			sourceLayerName := fmt.Sprintf("layer%d", i+1)
			targetLayerName := fmt.Sprintf("layer%d", i+2)

			sourceLayer := layersByName[sourceLayerName]
			targetLayer := layersByName[targetLayerName]

			if sourceLayer == nil || targetLayer == nil {
				fmt.Printf("Error: Could not find layers %s or %s\n", sourceLayerName, targetLayerName)
				os.Exit(1)
			}

			err := network.ConnectLayers(sourceLayer.UUID, targetLayer.UUID)
			if err != nil {
				fmt.Printf("Error connecting layers %s and %s: %v\n", sourceLayer.Name, targetLayer.Name, err)
				os.Exit(1)
			}

			fmt.Printf("Connected layer %s to layer %s\n", sourceLayer.Name, targetLayer.Name)
		}

		// Create a circular reference for demonstration (last layer to first)
		if flags.LayerCount > 2 {
			lastLayerName := fmt.Sprintf("layer%d", flags.LayerCount)
			firstLayerName := "layer1"

			lastLayer := layersByName[lastLayerName]
			firstLayer := layersByName[firstLayerName]

			if lastLayer == nil || firstLayer == nil {
				fmt.Printf("Error: Could not find layers %s or %s for circular reference\n", lastLayerName, firstLayerName)
				os.Exit(1)
			}

			err := network.ConnectLayers(lastLayer.UUID, firstLayer.UUID)
			if err != nil {
				fmt.Printf("Error connecting layers %s and %s: %v\n", lastLayer.Name, firstLayer.Name, err)
				os.Exit(1)
			}

			fmt.Printf("Connected layer %s to layer %s (circular reference)\n", lastLayer.Name, firstLayer.Name)
		}
	}

	// Save the network to a file
	saveStartTime := time.Now()

	if flags.UseBinary {
		// Use binary serialization
		serializer := NewBinarySerializer()
		binaryFilePath := flags.FilePath + ".bin"

		err := serializer.SaveNetworkToBinary(network, binaryFilePath)
		if err != nil {
			fmt.Printf("Error saving network to binary file: %v\n", err)
			os.Exit(1)
		}

		saveTime := time.Since(saveStartTime)
		fmt.Printf("Network saved to binary file %s (took %v)\n", binaryFilePath, saveTime)

		// Get file size
		fileInfo, err := os.Stat(binaryFilePath)
		if err == nil {
			fmt.Printf("Binary file size: %.2f MiB\n", float64(fileInfo.Size())/1024/1024)
		}
	} else {
		// Use JSON serialization
		jsonFilePath := flags.FilePath + ".json"

		err := network.SaveToFile(jsonFilePath)
		if err != nil {
			fmt.Printf("Error saving network to JSON file: %v\n", err)
			os.Exit(1)
		}

		saveTime := time.Since(saveStartTime)
		fmt.Printf("Network saved to JSON file %s (took %v)\n", jsonFilePath, saveTime)

		// Get file size
		fileInfo, err := os.Stat(jsonFilePath)
		if err == nil {
			fmt.Printf("JSON file size: %.2f MiB\n", float64(fileInfo.Size())/1024/1024)
		}
	}

	elapsedTime := time.Since(startTime)
	fmt.Printf("Network creation and saving completed (took %v)\n", elapsedTime)

	// Print memory stats
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("Memory usage: Alloc = %v MiB, TotalAlloc = %v MiB, Sys = %v MiB\n",
		m.Alloc/1024/1024, m.TotalAlloc/1024/1024, m.Sys/1024/1024)
}

func loadNetwork(flags CommandFlags) {
	var filePath string
	var startTime time.Time
	var network *Network
	var err error

	if flags.UseBinary {
		filePath = flags.FilePath + ".bin"
		fmt.Printf("Loading network from binary file %s...\n", filePath)
		startTime = time.Now()

		// Load using binary serialization
		serializer := NewBinarySerializer()
		network, err = serializer.LoadNetworkFromBinary(filePath, flags.Seed)
	} else {
		filePath = flags.FilePath + ".json"
		fmt.Printf("Loading network from JSON file %s...\n", filePath)
		startTime = time.Now()

		// Load using JSON serialization
		network, err = LoadNetworkFromFile(filePath, flags.Seed)
	}

	if err != nil {
		fmt.Printf("Error loading network: %v\n", err)
		os.Exit(1)
	}

	// Display some information about the loaded network
	fmt.Printf("Loaded network UUID: %s, Name: %s\n", network.UUID, network.Name)
	fmt.Printf("Number of layers: %d\n", len(network.Layers))
	fmt.Printf("Total neurons: %d\n", len(network.NeuronCache))

	for _, layer := range network.Layers {
		fmt.Printf("Layer %s: %dx%d grid with %d neurons\n",
			layer.Name, layer.Width, layer.Height, len(layer.Neurons))
	}

	elapsedTime := time.Since(startTime)
	fmt.Printf("Network loaded successfully (took %v)\n", elapsedTime)
}

func testNetwork(flags CommandFlags) {
	fmt.Println("Testing network operations...")
	fmt.Printf("Using seed: %d, Parallel processing: %v\n", flags.Seed, flags.ParallelProc)
	startTime := time.Now()

	// Initialize random source with seed for reproducibility
	r := rand.New(rand.NewSource(flags.Seed))

	// Create a test network
	network, err := NewNetwork("test_network", flags.Seed)
	if err != nil {
		fmt.Printf("Error creating network: %v\n", err)
		os.Exit(1)
	}

	// Create a simple 3x3 layer with 2-dimensional vectors
	layer, err := NewLayer("test_layer", 3, 3, 2, r)
	if err != nil {
		fmt.Printf("Error creating layer: %v\n", err)
		os.Exit(1)
	}

	err = layer.ConnectInternalNeurons(r)
	if err != nil {
		fmt.Printf("Error connecting neurons: %v\n", err)
		os.Exit(1)
	}

	err = network.AddLayer(layer)
	if err != nil {
		fmt.Printf("Error adding layer to network: %v\n", err)
		os.Exit(1)
	}

	// Get a starting neuron (first one in the layer)
	var startNeuronUUID string
	for uuid := range layer.Neurons {
		startNeuronUUID = uuid
		break
	}

	if flags.ParallelProc {
		// Create multiple test data inputs
		testDataInputs := make([][]float64, 10)
		for i := range testDataInputs {
			testDataInputs[i] = []float64{r.Float64(), r.Float64()}
		}

		// Process the data through the network in parallel
		results, paths, errors := network.ProcessDataParallel(layer.UUID, startNeuronUUID, testDataInputs, 10)

		// Display results
		fmt.Println("Parallel data processing complete")
		for i, result := range results {
			if errors[i] != nil {
				fmt.Printf("Error processing input %d: %v\n", i, errors[i])
				continue
			}
			fmt.Printf("Input %d: %v -> Output: %v, Path length: %d\n",
				i, testDataInputs[i], result, len(paths[i]))
		}
	} else {
		// Create some test data
		testData := []float64{0.5, 0.7}

		// Process the data through the network
		result, path, err := network.ProcessData(layer.UUID, startNeuronUUID, testData, 10)
		if err != nil {
			fmt.Printf("Error processing data: %v\n", err)
			os.Exit(1)
		}

		// Display results
		fmt.Println("Data processing complete")
		fmt.Printf("Input data: %v\n", testData)
		fmt.Printf("Output data: %v\n", result)
		fmt.Printf("Path taken: %v\n", path)
	}

	// Save the layer to demonstrate both serialization formats
	if flags.UseBinary {
		// Save using binary serialization
		serializer := NewBinarySerializer()
		err = serializer.SaveNetworkToBinary(network, "test_network.bin")
		if err != nil {
			fmt.Printf("Error saving network to binary: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("Test network saved to test_network.bin")
	} else {
		// Save using JSON serialization
		err = network.SaveToFile("test_network.json")
		if err != nil {
			fmt.Printf("Error saving network to JSON: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("Test network saved to test_network.json")
	}

	elapsedTime := time.Since(startTime)
	fmt.Printf("Test completed successfully (took %v)\n", elapsedTime)
}

func benchmarkNetwork(flags CommandFlags) {
	fmt.Println("Running network performance benchmarks...")
	fmt.Printf("Seed: %d, Grid Size: %dx%d, Vector Dimensions: %d, Layers: %d, Full Connectivity: %v, Binary Format: %v\n",
		flags.Seed, flags.NeuronSize, flags.NeuronSize, flags.VectorDims, flags.LayerCount, flags.FullConnect, flags.UseBinary)

	// Initialize random source with seed for reproducibility
	r := rand.New(rand.NewSource(flags.Seed))

	// Benchmark network creation
	fmt.Println("\n1. Benchmarking network creation...")
	createStart := time.Now()

	network, err := NewNetwork("benchmark_network", flags.Seed)
	if err != nil {
		fmt.Printf("Error creating network: %v\n", err)
		os.Exit(1)
	}

	// Store layers by name for easier lookup
	layersByName := make(map[string]*Layer)

	for i := 0; i < flags.LayerCount; i++ {
		layerName := fmt.Sprintf("layer%d", i+1)
		layer, err := NewLayer(layerName, flags.NeuronSize, flags.NeuronSize, flags.VectorDims, r)
		if err != nil {
			fmt.Printf("Error creating layer: %v\n", err)
			os.Exit(1)
		}

		err = layer.ConnectInternalNeurons(r)
		if err != nil {
			fmt.Printf("Error connecting neurons: %v\n", err)
			os.Exit(1)
		}

		err = network.AddLayer(layer)
		if err != nil {
			fmt.Printf("Error adding layer: %v\n", err)
			os.Exit(1)
		}

		layersByName[layerName] = layer
	}

	// Connect layers based on connectivity mode
	if flags.FullConnect {
		// Full connectivity: each layer connects to all other layers
		fmt.Println("Connecting all layers to each other (full connectivity)...")
		connectionStartTime := time.Now()

		err := network.ConnectAllLayers()
		if err != nil {
			fmt.Printf("Error connecting layers: %v\n", err)
			os.Exit(1)
		}

		connectionTime := time.Since(connectionStartTime)
		fmt.Printf("All layers fully connected (took %v)\n", connectionTime)
	} else {
		// Connect layers sequentially
		for i := 0; i < flags.LayerCount-1; i++ {
			sourceLayerName := fmt.Sprintf("layer%d", i+1)
			targetLayerName := fmt.Sprintf("layer%d", i+2)

			sourceLayer := layersByName[sourceLayerName]
			targetLayer := layersByName[targetLayerName]

			if sourceLayer == nil || targetLayer == nil {
				fmt.Printf("Error: Could not find layers %s or %s\n", sourceLayerName, targetLayerName)
				os.Exit(1)
			}

			err := network.ConnectLayers(sourceLayer.UUID, targetLayer.UUID)
			if err != nil {
				fmt.Printf("Error connecting layers: %v\n", err)
				os.Exit(1)
			}
		}
	}

	createTime := time.Since(createStart)
	fmt.Printf("Network creation time: %v\n", createTime)
	fmt.Printf("Total neurons: %d\n", len(network.NeuronCache))

	// Benchmark data processing
	fmt.Println("\n2. Benchmarking data processing...")

	// Get first layer and neuron
	firstLayer := layersByName["layer1"]
	if firstLayer == nil {
		fmt.Printf("Error: Could not find layer1\n")
		os.Exit(1)
	}

	var firstNeuronUUID string
	for neuronUUID := range firstLayer.Neurons {
		firstNeuronUUID = neuronUUID
		break
	}

	// Create test data
	testData := make([]float64, flags.VectorDims)
	for i := range testData {
		testData[i] = r.Float64()
	}

	// Single processing benchmark
	singleStart := time.Now()
	iterations := 100

	for i := 0; i < iterations; i++ {
		_, _, err := network.ProcessData(firstLayer.UUID, firstNeuronUUID, testData, 20)
		if err != nil {
			fmt.Printf("Error processing data: %v\n", err)
			os.Exit(1)
		}
	}

	singleTime := time.Since(singleStart)
	fmt.Printf("Single processing time (%d iterations): %v (avg: %v per iteration)\n",
		iterations, singleTime, singleTime/time.Duration(iterations))

	// Parallel processing benchmark
	fmt.Println("\n3. Benchmarking parallel data processing...")

	// Create multiple test data inputs
	batchSize := 100
	testDataInputs := make([][]float64, batchSize)
	for i := range testDataInputs {
		testDataInputs[i] = make([]float64, flags.VectorDims)
		for j := range testDataInputs[i] {
			testDataInputs[i][j] = r.Float64()
		}
	}

	parallelStart := time.Now()
	_, _, errors := network.ProcessDataParallel(firstLayer.UUID, firstNeuronUUID, testDataInputs, 20)

	for _, err := range errors {
		if err != nil {
			fmt.Printf("Error in parallel processing: %v\n", err)
			os.Exit(1)
		}
	}

	parallelTime := time.Since(parallelStart)
	fmt.Printf("Parallel processing time (batch size %d): %v (avg: %v per item)\n",
		batchSize, parallelTime, parallelTime/time.Duration(batchSize))

	// Benchmark file operations
	fmt.Println("\n4. Benchmarking file operations...")

	// Benchmark serialization
	if flags.UseBinary {
		// Binary serialization benchmark
		serializer := NewBinarySerializer()

		// Save benchmark
		binarySaveStart := time.Now()
		err = serializer.SaveNetworkToBinary(network, "benchmark_network.bin")
		if err != nil {
			fmt.Printf("Error saving network to binary: %v\n", err)
			os.Exit(1)
		}
		binarySaveTime := time.Since(binarySaveStart)
		fmt.Printf("Binary save time: %v\n", binarySaveTime)

		// Load benchmark
		binaryLoadStart := time.Now()
		_, err = serializer.LoadNetworkFromBinary("benchmark_network.bin", flags.Seed)
		if err != nil {
			fmt.Printf("Error loading network from binary: %v\n", err)
			os.Exit(1)
		}
		binaryLoadTime := time.Since(binaryLoadStart)
		fmt.Printf("Binary load time: %v\n", binaryLoadTime)
	} else {
		// JSON serialization benchmark
		jsonSaveStart := time.Now()
		err = network.SaveToFile("benchmark_network.json")
		if err != nil {
			fmt.Printf("Error saving network to JSON: %v\n", err)
			os.Exit(1)
		}
		jsonSaveTime := time.Since(jsonSaveStart)
		fmt.Printf("JSON save time: %v\n", jsonSaveTime)

		// Load benchmark
		jsonLoadStart := time.Now()
		_, err = LoadNetworkFromFile("benchmark_network.json", flags.Seed)
		if err != nil {
			fmt.Printf("Error loading network from JSON: %v\n", err)
			os.Exit(1)
		}
		jsonLoadTime := time.Since(jsonLoadStart)
		fmt.Printf("JSON load time: %v\n", jsonLoadTime)
	}

	// Print memory stats
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("\nMemory usage: Alloc = %v MiB, TotalAlloc = %v MiB, Sys = %v MiB\n",
		m.Alloc/1024/1024, m.TotalAlloc/1024/1024, m.Sys/1024/1024)

	fmt.Println("\nBenchmark completed successfully")
}

func benchmarkSerialization(flags CommandFlags) {
	fmt.Println("Running serialization benchmark...")
	fmt.Printf("Creating test network with %d layers, %dx%d neurons, %d dimensions\n",
		flags.LayerCount, flags.NeuronSize, flags.NeuronSize, flags.VectorDims)

	// Create a test network
	r := rand.New(rand.NewSource(flags.Seed))
	network, err := NewNetwork("benchmark_network", flags.Seed)
	if err != nil {
		fmt.Printf("Error creating network: %v\n", err)
		os.Exit(1)
	}

	// Add layers
	for i := 0; i < flags.LayerCount; i++ {
		layerName := fmt.Sprintf("layer%d", i+1)
		layer, err := NewLayer(layerName, flags.NeuronSize, flags.NeuronSize, flags.VectorDims, r)
		if err != nil {
			fmt.Printf("Error creating layer: %v\n", err)
			os.Exit(1)
		}

		err = layer.ConnectInternalNeurons(r)
		if err != nil {
			fmt.Printf("Error connecting neurons: %v\n", err)
			os.Exit(1)
		}

		err = network.AddLayer(layer)
		if err != nil {
			fmt.Printf("Error adding layer: %v\n", err)
			os.Exit(1)
		}
	}

	// Run the benchmark
	BinarySerializationBenchmark(network, "serialization_benchmark", flags.Seed)
}
