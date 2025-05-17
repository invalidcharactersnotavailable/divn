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

// ==================== VECTOR IMPLEMENTATION ====================

// Vector represents a flexible-dimension vector for neuron operations
type Vector struct {
	Dimensions []float64 // Values for each dimension
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

// ==================== NEURON IMPLEMENTATION ====================

// Neuron represents a processing unit in the network with UUID-based identification
type Neuron struct {
	UUID        string             // Unique identifier using UUID
	Value       *Vector            // Vector for data transformation with selectable dimensions
	Resistance  float64            // Resistance value for routing (lower = higher priority)
	Connections map[string]float64 // Map of connected neuron UUIDs to connection strengths
	mu          sync.RWMutex       // Mutex for thread safety
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

// ==================== LAYER IMPLEMENTATION ====================

// Layer represents a collection of neurons in a 2D grid
type Layer struct {
	UUID    string             // Unique identifier
	Name    string             // Layer name
	Width   int                // Width of the neuron grid
	Height  int                // Height of the neuron grid
	Neurons map[string]*Neuron // Map of neuron UUIDs to neurons
	mu      sync.RWMutex       // Mutex for thread safety
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

	// Connect each neuron to ALL other neurons in the layer (fully connected)
	for _, sourceUUID := range neuronUUIDs {
		source, exists := l.Neurons[sourceUUID]
		if !exists {
			return fmt.Errorf("source neuron not found: %s", sourceUUID)
		}

		// Connect to all other neurons (except self)
		for _, targetUUID := range neuronUUIDs {
			// Skip self-connections
			if targetUUID == sourceUUID {
				continue
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

// SaveToFile saves a single layer to a binary file
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

	// Create a binary serializer
	bs := NewBinarySerializer()

	// Open the temporary file for writing
	file, err := os.Create(tempFile)
	if err != nil {
		return fmt.Errorf("failed to create temporary file %s: %w", tempFile, err)
	}
	defer file.Close()

	// Write the layer using binary serialization
	if err := bs.writeLayer(file, l); err != nil {
		return fmt.Errorf("failed to write layer to binary file: %w", err)
	}

	// Rename temporary file to target file (atomic operation)
	if err := os.Rename(tempFile, filePath); err != nil {
		// Try to clean up the temporary file
		os.Remove(tempFile)
		return fmt.Errorf("failed to rename temporary file to %s: %w", filePath, err)
	}

	return nil
}

// LoadLayerFromFile loads a layer from a binary file
func LoadLayerFromFile(filePath string) (*Layer, error) {
	// Create a binary serializer
	bs := NewBinarySerializer()

	// Open the file for reading
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %w", filePath, err)
	}
	defer file.Close()

	// Read the layer using binary serialization
	layer, err := bs.readLayer(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read layer from binary file: %w", err)
	}

	return layer, nil
}

// ==================== NETWORK IMPLEMENTATION ====================

// Network represents the entire dynamic routing network
type Network struct {
	UUID        string             // Unique identifier
	Name        string             // Network name
	Layers      map[string]*Layer  // Map of layer UUIDs to layers
	NeuronCache map[string]*Neuron // Global neuron cache for O(1) lookup
	mu          sync.RWMutex       // Mutex for thread safety
	rand        *rand.Rand         // Random source for deterministic operations
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

// AddLayer adds a layer to the network, updates the neuron cache, and connects internal neurons
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

	// Connect neurons within the layer (intraconnections)
	if err := layer.ConnectInternalNeurons(n.rand); err != nil {
		return fmt.Errorf("failed to connect internal neurons in layer %s: %w", layer.Name, err)
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

// SaveToFile saves the network to a binary file
func (n *Network) SaveToFile(filePath string) error {
	// Create a binary serializer
	bs := NewBinarySerializer()

	// Use the binary serializer to save the network
	return bs.SaveNetworkToBinary(n, filePath)
}

// LoadNetworkFromFile loads a network from a binary file
func LoadNetworkFromFile(filePath string, seed int64) (*Network, error) {
	// Create a binary serializer
	bs := NewBinarySerializer()

	// Use the binary serializer to load the network
	return bs.LoadNetworkFromBinary(filePath, seed)
}

// ==================== BINARY SERIALIZER IMPLEMENTATION ====================

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

// ==================== JSON SERIALIZER IMPLEMENTATION ====================

// JSONSerializer provides methods for JSON serialization and deserialization
type JSONSerializer struct {
	// Version of the JSON format, for future compatibility
	Version int `json:"version"`
}

// NetworkJSON represents the JSON structure of a network
type NetworkJSON struct {
	Version int                  `json:"version"`
	UUID    string               `json:"uuid"`
	Name    string               `json:"name"`
	Layers  map[string]LayerJSON `json:"layers"`
}

// LayerJSON represents the JSON structure of a layer
type LayerJSON struct {
	UUID    string                `json:"uuid"`
	Name    string                `json:"name"`
	Width   int                   `json:"width"`
	Height  int                   `json:"height"`
	Neurons map[string]NeuronJSON `json:"neurons"`
}

// NeuronJSON represents the JSON structure of a neuron
type NeuronJSON struct {
	UUID        string             `json:"uuid"`
	Resistance  float64            `json:"resistance"`
	Vector      []float64          `json:"vector"`
	Connections map[string]float64 `json:"connections"`
}

// NewJSONSerializer creates a new JSON serializer
func NewJSONSerializer() *JSONSerializer {
	return &JSONSerializer{
		Version: 1, // Initial version
	}
}

// SaveNetworkToJSON saves a network to a JSON file
func (js *JSONSerializer) SaveNetworkToJSON(network *Network, filePath string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	// Convert network to JSON structure
	networkJSON := js.networkToJSON(network)

	// Marshal to JSON
	data, err := json.MarshalIndent(networkJSON, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal network to JSON: %w", err)
	}

	// Create a temporary file in the same directory
	tempFile := filePath + ".tmp"
	if err := os.WriteFile(tempFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write JSON to temporary file: %w", err)
	}

	// Rename temporary file to target file (atomic operation)
	if err := os.Rename(tempFile, filePath); err != nil {
		// Try to clean up the temporary file
		os.Remove(tempFile)
		return fmt.Errorf("failed to rename temporary file to %s: %w", filePath, err)
	}

	return nil
}

// LoadNetworkFromJSON loads a network from a JSON file
func (js *JSONSerializer) LoadNetworkFromJSON(filePath string, seed int64) (*Network, error) {
	// Read the JSON file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON file: %w", err)
	}

	// Unmarshal JSON
	var networkJSON NetworkJSON
	if err := json.Unmarshal(data, &networkJSON); err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}

	// Convert JSON structure to network
	network, err := js.jsonToNetwork(networkJSON, seed)
	if err != nil {
		return nil, fmt.Errorf("failed to convert JSON to network: %w", err)
	}

	return network, nil
}

// networkToJSON converts a network to its JSON representation
func (js *JSONSerializer) networkToJSON(network *Network) NetworkJSON {
	networkJSON := NetworkJSON{
		Version: js.Version,
		UUID:    network.UUID,
		Name:    network.Name,
		Layers:  make(map[string]LayerJSON),
	}

	// Convert each layer
	for uuid, layer := range network.Layers {
		networkJSON.Layers[uuid] = js.layerToJSON(layer)
	}

	return networkJSON
}

// layerToJSON converts a layer to its JSON representation
func (js *JSONSerializer) layerToJSON(layer *Layer) LayerJSON {
	layerJSON := LayerJSON{
		UUID:    layer.UUID,
		Name:    layer.Name,
		Width:   layer.Width,
		Height:  layer.Height,
		Neurons: make(map[string]NeuronJSON),
	}

	// Convert each neuron
	for uuid, neuron := range layer.Neurons {
		layerJSON.Neurons[uuid] = js.neuronToJSON(neuron)
	}

	return layerJSON
}

// neuronToJSON converts a neuron to its JSON representation
func (js *JSONSerializer) neuronToJSON(neuron *Neuron) NeuronJSON {
	neuronJSON := NeuronJSON{
		UUID:        neuron.UUID,
		Resistance:  neuron.Resistance,
		Vector:      make([]float64, len(neuron.Value.Dimensions)),
		Connections: make(map[string]float64),
	}

	// Copy vector dimensions
	copy(neuronJSON.Vector, neuron.Value.Dimensions)

	// Copy connections
	for uuid, strength := range neuron.Connections {
		neuronJSON.Connections[uuid] = strength
	}

	return neuronJSON
}

// jsonToNetwork converts a JSON representation to a network
func (js *JSONSerializer) jsonToNetwork(networkJSON NetworkJSON, seed int64) (*Network, error) {
	// Create a new network
	network := &Network{
		UUID:        networkJSON.UUID,
		Name:        networkJSON.Name,
		Layers:      make(map[string]*Layer),
		NeuronCache: make(map[string]*Neuron),
	}

	// Convert each layer
	for uuid, layerJSON := range networkJSON.Layers {
		layer, err := js.jsonToLayer(layerJSON)
		if err != nil {
			return nil, fmt.Errorf("failed to convert layer %s: %w", uuid, err)
		}
		network.Layers[uuid] = layer

		// Add neurons to cache
		for neuronUUID, neuron := range layer.Neurons {
			network.NeuronCache[neuronUUID] = neuron
		}
	}

	return network, nil
}

// jsonToLayer converts a JSON representation to a layer
func (js *JSONSerializer) jsonToLayer(layerJSON LayerJSON) (*Layer, error) {
	// Create a new layer
	layer := &Layer{
		UUID:    layerJSON.UUID,
		Name:    layerJSON.Name,
		Width:   layerJSON.Width,
		Height:  layerJSON.Height,
		Neurons: make(map[string]*Neuron),
	}

	// Convert each neuron
	for uuid, neuronJSON := range layerJSON.Neurons {
		neuron, err := js.jsonToNeuron(neuronJSON)
		if err != nil {
			return nil, fmt.Errorf("failed to convert neuron %s: %w", uuid, err)
		}
		layer.Neurons[uuid] = neuron
	}

	return layer, nil
}

// jsonToNeuron converts a JSON representation to a neuron
func (js *JSONSerializer) jsonToNeuron(neuronJSON NeuronJSON) (*Neuron, error) {
	// Create a new vector
	vector := &Vector{
		Dimensions: make([]float64, len(neuronJSON.Vector)),
	}

	// Copy vector dimensions
	copy(vector.Dimensions, neuronJSON.Vector)

	// Create a new neuron
	neuron := &Neuron{
		UUID:        neuronJSON.UUID,
		Value:       vector,
		Resistance:  neuronJSON.Resistance,
		Connections: make(map[string]float64),
	}

	// Copy connections
	for uuid, strength := range neuronJSON.Connections {
		neuron.Connections[uuid] = strength
	}

	return neuron, nil
}

// GetNetworkJSON returns the JSON representation of a network
func (js *JSONSerializer) GetNetworkJSON(network *Network) ([]byte, error) {
	// Convert network to JSON structure
	networkJSON := js.networkToJSON(network)

	// Marshal to JSON
	data, err := json.MarshalIndent(networkJSON, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("failed to marshal network to JSON: %w", err)
	}

	return data, nil
}

// ==================== CONNECTION MAP IMPLEMENTATION ====================

// ConnectionMap represents a separate storage for network connections
type ConnectionMap struct {
	NetworkUUID string                        `json:"network_uuid"`
	NetworkName string                        `json:"network_name"`
	Layers      map[string]LayerConnectionMap `json:"layers"`
	Version     int                           `json:"version"`
}

// LayerConnectionMap represents connections for a specific layer
type LayerConnectionMap struct {
	LayerUUID string                         `json:"layer_uuid"`
	LayerName string                         `json:"layer_name"`
	Neurons   map[string]NeuronConnectionMap `json:"neurons"`
}

// NeuronConnectionMap represents connections for a specific neuron
type NeuronConnectionMap struct {
	NeuronUUID  string             `json:"neuron_uuid"`
	Connections map[string]float64 `json:"connections"`
}

// NewConnectionMap creates a new connection map for a network
func NewConnectionMap(network *Network) *ConnectionMap {
	connectionMap := &ConnectionMap{
		NetworkUUID: network.UUID,
		NetworkName: network.Name,
		Layers:      make(map[string]LayerConnectionMap),
		Version:     1, // Initial version
	}

	// Extract connection information from the network
	for layerUUID, layer := range network.Layers {
		layerMap := LayerConnectionMap{
			LayerUUID: layerUUID,
			LayerName: layer.Name,
			Neurons:   make(map[string]NeuronConnectionMap),
		}

		// Extract neuron connections
		for neuronUUID, neuron := range layer.Neurons {
			neuronMap := NeuronConnectionMap{
				NeuronUUID:  neuronUUID,
				Connections: make(map[string]float64),
			}

			// Copy connections
			for targetUUID, strength := range neuron.Connections {
				neuronMap.Connections[targetUUID] = strength
			}

			layerMap.Neurons[neuronUUID] = neuronMap
		}

		connectionMap.Layers[layerUUID] = layerMap
	}

	return connectionMap
}

// SaveConnectionMapToJSON saves a connection map to a JSON file
func SaveConnectionMapToJSON(connectionMap *ConnectionMap, filePath string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	// Marshal to JSON
	data, err := json.MarshalIndent(connectionMap, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal connection map to JSON: %w", err)
	}

	// Create a temporary file in the same directory
	tempFile := filePath + ".tmp"
	if err := os.WriteFile(tempFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write JSON to temporary file: %w", err)
	}

	// Rename temporary file to target file (atomic operation)
	if err := os.Rename(tempFile, filePath); err != nil {
		// Try to clean up the temporary file
		os.Remove(tempFile)
		return fmt.Errorf("failed to rename temporary file to %s: %w", filePath, err)
	}

	return nil
}

// LoadConnectionMapFromJSON loads a connection map from a JSON file
func LoadConnectionMapFromJSON(filePath string) (*ConnectionMap, error) {
	// Read the JSON file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON file: %w", err)
	}

	// Unmarshal JSON
	var connectionMap ConnectionMap
	if err := json.Unmarshal(data, &connectionMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}

	return &connectionMap, nil
}

// ApplyConnectionMapToNetwork applies a connection map to a network
func ApplyConnectionMapToNetwork(connectionMap *ConnectionMap, network *Network) error {
	// Verify network UUID matches
	if connectionMap.NetworkUUID != network.UUID {
		return fmt.Errorf("connection map network UUID (%s) does not match network UUID (%s)",
			connectionMap.NetworkUUID, network.UUID)
	}

	// Apply connections to each layer and neuron
	for layerUUID, layerMap := range connectionMap.Layers {
		layer, layerExists := network.Layers[layerUUID]
		if !layerExists {
			return fmt.Errorf("layer with UUID %s not found in network", layerUUID)
		}

		// Apply neuron connections
		for neuronUUID, neuronMap := range layerMap.Neurons {
			neuron, neuronExists := layer.Neurons[neuronUUID]
			if !neuronExists {
				return fmt.Errorf("neuron with UUID %s not found in layer %s", neuronUUID, layerUUID)
			}

			// Clear existing connections and apply from map
			neuron.mu.Lock()
			neuron.Connections = make(map[string]float64)
			for targetUUID, strength := range neuronMap.Connections {
				neuron.Connections[targetUUID] = strength
			}
			neuron.mu.Unlock()
		}
	}

	return nil
}

// ExtractConnectionMapFromNetwork creates a connection map from a network
func ExtractConnectionMapFromNetwork(network *Network) *ConnectionMap {
	return NewConnectionMap(network)
}

// ==================== CONNECTION MAP VALIDATION ====================

// ValidateConnectionMap provides manual validation for connection maps
func ValidateConnectionMap() {
	fmt.Println("Validating connection map implementation...")

	// Create a temporary directory for validation files
	tempDir := filepath.Join(os.TempDir(), "divn_validation")
	os.MkdirAll(tempDir, 0755)
	defer os.RemoveAll(tempDir) // Clean up after validation

	// Create a test network
	network, err := NewNetwork("Validation Network", 12345)
	if err != nil {
		fmt.Printf("Error creating network: %v\n", err)
		return
	}

	// Add layers
	for i := 0; i < 2; i++ {
		layer, err := NewLayer(fmt.Sprintf("Layer-%d", i), 3, 3, 4, network.rand)
		if err != nil {
			fmt.Printf("Error creating layer: %v\n", err)
			return
		}
		err = network.AddLayer(layer)
		if err != nil {
			fmt.Printf("Error adding layer: %v\n", err)
			return
		}
	}

	// Connect layers
	err = network.ConnectAllLayers()
	if err != nil {
		fmt.Printf("Error connecting layers: %v\n", err)
		return
	}

	// Create connection map
	connectionMap := ExtractConnectionMapFromNetwork(network)
	if connectionMap == nil {
		fmt.Println("Error: Failed to extract connection map")
		return
	}

	// Verify connection map
	if connectionMap.NetworkUUID != network.UUID {
		fmt.Printf("Error: Connection map network UUID mismatch: got %s, want %s\n",
			connectionMap.NetworkUUID, network.UUID)
		return
	}

	if len(connectionMap.Layers) != len(network.Layers) {
		fmt.Printf("Error: Connection map layer count mismatch: got %d, want %d\n",
			len(connectionMap.Layers), len(network.Layers))
		return
	}

	// Save connection map to temporary file
	tempFile := filepath.Join(tempDir, "validation_connection_map.json")
	err = SaveConnectionMapToJSON(connectionMap, tempFile)
	if err != nil {
		fmt.Printf("Error saving connection map: %v\n", err)
		return
	}

	// Load connection map
	loadedMap, err := LoadConnectionMapFromJSON(tempFile)
	if err != nil {
		fmt.Printf("Error loading connection map: %v\n", err)
		return
	}

	// Verify loaded map
	if loadedMap.NetworkUUID != network.UUID {
		fmt.Printf("Error: Loaded map network UUID mismatch: got %s, want %s\n",
			loadedMap.NetworkUUID, network.UUID)
		return
	}

	if len(loadedMap.Layers) != len(network.Layers) {
		fmt.Printf("Error: Loaded map layer count mismatch: got %d, want %d\n",
			len(loadedMap.Layers), len(network.Layers))
		return
	}

	// Count connections in original network
	totalOriginalConnections := 0
	for _, layer := range network.Layers {
		for _, neuron := range layer.Neurons {
			totalOriginalConnections += len(neuron.Connections)
		}
	}

	// Count connections in loaded map
	totalMapConnections := 0
	for _, layerMap := range loadedMap.Layers {
		for _, neuronMap := range layerMap.Neurons {
			totalMapConnections += len(neuronMap.Connections)
		}
	}

	if totalOriginalConnections != totalMapConnections {
		fmt.Printf("Error: Connection count mismatch: network has %d, map has %d\n",
			totalOriginalConnections, totalMapConnections)
		return
	}

	fmt.Println("Connection map validation completed successfully!")
	fmt.Printf("Validated %d layers with %d total connections\n",
		len(network.Layers), totalOriginalConnections)
}

// ==================== CONNECTION MAP COMMANDS ====================

// saveNetworkWithConnectionMap saves a network and its connection map to separate files
func saveNetworkWithConnectionMap(flags CommandFlags) {
	jsonFilePath := flags.FilePath + ".json"
	connectionMapFilePath := flags.FilePath + ".connections.json"

	fmt.Printf("Loading network from %s...\n", jsonFilePath)

	// Check if file exists
	if _, err := os.Stat(jsonFilePath); os.IsNotExist(err) {
		fmt.Printf("Error: Network file %s does not exist\n", jsonFilePath)
		os.Exit(1)
	}

	// Load the network
	jsonSerializer := NewJSONSerializer()
	network, err := jsonSerializer.LoadNetworkFromJSON(jsonFilePath, flags.Seed)
	if err != nil {
		fmt.Printf("Error loading network: %v\n", err)
		os.Exit(1)
	}

	// Ensure directory exists for output
	outputDir := filepath.Dir(flags.FilePath)
	if outputDir != "" && outputDir != "." {
		if err := os.MkdirAll(outputDir, 0755); err != nil {
			fmt.Printf("Error creating output directory %s: %v\n", outputDir, err)
			os.Exit(1)
		}
	}

	// Save the network
	fmt.Printf("Saving network to %s...\n", jsonFilePath)

	// Use JSON serialization
	err = jsonSerializer.SaveNetworkToJSON(network, jsonFilePath)
	if err != nil {
		fmt.Printf("Error saving network to JSON: %v\n", err)
		os.Exit(1)
	}

	// Extract and save connection map
	connectionMap := ExtractConnectionMapFromNetwork(network)
	fmt.Printf("Saving connection map to %s...\n", connectionMapFilePath)

	err = SaveConnectionMapToJSON(connectionMap, connectionMapFilePath)
	if err != nil {
		fmt.Printf("Error saving connection map: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Network saved successfully to %s\n", jsonFilePath)
	fmt.Printf("Connection map saved successfully to %s\n", connectionMapFilePath)

	if flags.Verbose {
		fmt.Printf("Network summary: %d layers, %d total neurons\n",
			len(network.Layers), len(network.NeuronCache))
	}
}

// loadNetworkWithConnectionMap loads a network and its connection map from separate files
func loadNetworkWithConnectionMap(flags CommandFlags) {
	jsonFilePath := flags.FilePath + ".json"
	connectionMapFilePath := flags.FilePath + ".connections.json"

	fmt.Printf("Loading network from %s...\n", jsonFilePath)

	// Check if files exist
	if _, err := os.Stat(jsonFilePath); os.IsNotExist(err) {
		fmt.Printf("Error: Network file %s does not exist\n", jsonFilePath)
		os.Exit(1)
	}

	if _, err := os.Stat(connectionMapFilePath); os.IsNotExist(err) {
		fmt.Printf("Warning: Connection map file %s does not exist, will use connections from network file\n", connectionMapFilePath)
	}

	// Load the network using JSON serialization
	jsonSerializer := NewJSONSerializer()
	network, err := jsonSerializer.LoadNetworkFromJSON(jsonFilePath, flags.Seed)
	if err != nil {
		fmt.Printf("Error loading network: %v\n", err)
		os.Exit(1)
	}

	// Load and apply connection map if it exists
	connectionMapExists := true
	if _, err := os.Stat(connectionMapFilePath); os.IsNotExist(err) {
		connectionMapExists = false
	}

	if connectionMapExists {
		fmt.Printf("Loading connection map from %s...\n", connectionMapFilePath)
		connectionMap, err := LoadConnectionMapFromJSON(connectionMapFilePath)
		if err != nil {
			fmt.Printf("Error loading connection map: %v\n", err)
			fmt.Println("Will use connections from network file")
		} else {
			// Apply connection map to network
			err = ApplyConnectionMapToNetwork(connectionMap, network)
			if err != nil {
				fmt.Printf("Error applying connection map: %v\n", err)
				fmt.Println("Will use connections from network file")
			} else {
				fmt.Println("Connection map applied successfully")
			}
		}
	}

	// Print network information
	fmt.Printf("\nNetwork Information:\n")
	fmt.Printf("Name: %s\n", network.Name)
	fmt.Printf("UUID: %s\n", network.UUID)
	fmt.Printf("Layers: %d\n", len(network.Layers))
	fmt.Printf("Total Neurons: %d\n", len(network.NeuronCache))

	// Print layer information
	fmt.Printf("\nLayer Details:\n")
	for uuid, layer := range network.Layers {
		fmt.Printf("Layer %s: %s\n", uuid, layer.Name)
		fmt.Printf("  Dimensions: %dx%d\n", layer.Width, layer.Height)
		fmt.Printf("  Neurons: %d\n", len(layer.Neurons))

		if flags.Verbose {
			// Count connections
			totalConnections := 0
			for _, neuron := range layer.Neurons {
				totalConnections += len(neuron.Connections)
			}
			fmt.Printf("  Total Connections: %d\n", totalConnections)
			fmt.Printf("  Avg Connections per Neuron: %.2f\n",
				float64(totalConnections)/float64(len(layer.Neurons)))
		}
	}

	fmt.Printf("\nNetwork loaded successfully from %s\n", jsonFilePath)
	if connectionMapExists {
		fmt.Printf("Connection map loaded from %s\n", connectionMapFilePath)
	}
}

// ==================== MAIN IMPLEMENTATION ====================

// CommandFlags structure for command-line flags
type CommandFlags struct {
	Seed         int64
	LayerCount   int
	NeuronSize   int
	VectorDims   int
	FilePath     string
	CPUProfile   string
	MemProfile   string
	ParallelProc bool
	Verbose      bool
}

// Command represents a CLI command with its handler and help text
type Command struct {
	Name        string
	Description string
	Usage       string
	Examples    []string
	Handler     func(flags CommandFlags)
}

// NetworkCreationOutput represents the JSON output for network creation
type NetworkCreationOutput struct {
	Parameters struct {
		Seed       int64  `json:"seed"`
		Layers     int    `json:"layers"`
		Size       int    `json:"size"`
		Dimensions int    `json:"dimensions"`
		Filename   string `json:"filename"`
	} `json:"parameters"`
	Metrics struct {
		CreationTime     string `json:"creation_time"`
		TotalLayers      int    `json:"total_layers"`
		TotalNeurons     int    `json:"total_neurons"`
		TotalConnections int    `json:"total_connections"`
		LayersConnected  int    `json:"layers_connected"`
		FileSize         int64  `json:"file_size_bytes"`
	} `json:"metrics"`
	Network interface{} `json:"network"`
	Status  string      `json:"status"`
}

// BinarySerializationBenchmark benchmarks the binary serialization and deserialization
func BinarySerializationBenchmark(network *Network, filePath string, seed int64) {
	// Benchmark binary serialization
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

	// Print results
	fmt.Println("\nSerialization Benchmark Results:")
	fmt.Printf("Binary save time: %v\n", binarySaveTime)
	fmt.Printf("Binary load time: %v\n", binaryLoadTime)
	fmt.Printf("Binary file size: %v bytes\n", binaryFileSize)
}

// getCommands returns the map of available commands
func getCommands() map[string]Command {
	commands := map[string]Command{
		"create": {
			Name:        "create",
			Description: "Create a new neural network",
			Usage:       "divn create [options]",
			Examples: []string{
				"divn create -layers 4 -size 10 -dims 128 -file mynetwork",
				"divn create -seed 12345 -parallel",
			},
			Handler: createNetwork,
		},
		"save": {
			Name:        "save",
			Description: "Save an existing network to a file",
			Usage:       "divn save [options]",
			Examples: []string{
				"divn save -file mynetwork",
			},
			Handler: saveNetwork,
		},
		"load": {
			Name:        "load",
			Description: "Load a network from file and display information",
			Usage:       "divn load [options]",
			Examples: []string{
				"divn load -file mynetwork",
			},
			Handler: loadNetwork,
		},
		"process": {
			Name:        "process",
			Description: "Process data through the network",
			Usage:       "divn process [options]",
			Examples: []string{
				"divn process -file mynetwork -dims 64",
			},
			Handler: processData,
		},
		"benchmark": {
			Name:        "benchmark",
			Description: "Run performance benchmarks",
			Usage:       "divn benchmark [options]",
			Examples: []string{
				"divn benchmark -layers 3 -size 10 -dims 128",
				"divn benchmark -cpuprofile cpu.prof -memprofile mem.prof",
			},
			Handler: runBenchmark,
		},
		"help": {
			Name:        "help",
			Description: "Display help information for a command",
			Usage:       "divn help [command]",
			Examples: []string{
				"divn help",
				"divn help create",
			},
			Handler: func(flags CommandFlags) {
				// This is handled separately in main()
			},
		},
		"savemap": {
			Name:        "savemap",
			Description: "Save a network with its connection map to separate files",
			Usage:       "divn savemap [options]",
			Examples: []string{
				"divn savemap -file mynetwork",
			},
			Handler: saveNetworkWithConnectionMap,
		},
		"loadmap": {
			Name:        "loadmap",
			Description: "Load a network and its connection map from separate files",
			Usage:       "divn loadmap [options]",
			Examples: []string{
				"divn loadmap -file mynetwork",
			},
			Handler: loadNetworkWithConnectionMap,
		},
	}

	return commands
}

func main() {
	// Get available commands
	commands := getCommands()

	// Check if we have enough arguments
	if len(os.Args) < 2 {
		printUsage(commands)
		os.Exit(1)
	}

	// Get the command
	command := os.Args[1]

	// Handle help command specially
	if command == "help" {
		if len(os.Args) > 2 {
			cmdName := os.Args[2]
			if cmd, exists := commands[cmdName]; exists {
				printCommandHelp(cmd)
			} else {
				fmt.Printf("Unknown command: %s\n", cmdName)
				printUsage(commands)
			}
		} else {
			printUsage(commands)
		}
		os.Exit(0)
	}

	// Check if command exists
	cmd, exists := commands[command]
	if !exists {
		fmt.Printf("Unknown command: %s\n", command)
		printUsage(commands)
		os.Exit(1)
	}

	// Remove the command from arguments to simplify flag parsing
	os.Args = append(os.Args[:1], os.Args[2:]...)

	// Create flags structure with default values
	flags := CommandFlags{
		Seed:         time.Now().UnixNano(),
		LayerCount:   3,
		NeuronSize:   8,
		VectorDims:   64, // Increased for text encoding
		FilePath:     "network",
		CPUProfile:   "",
		MemProfile:   "",
		ParallelProc: false,
		Verbose:      false,
	}

	// Set up command line flags
	flag.Int64Var(&flags.Seed, "seed", flags.Seed, "Random seed for reproducible network generation")
	flag.IntVar(&flags.LayerCount, "layers", flags.LayerCount, "Number of layers in the network")
	flag.IntVar(&flags.NeuronSize, "size", flags.NeuronSize, "Size of neuron grid (size x size)")
	flag.IntVar(&flags.VectorDims, "dims", flags.VectorDims, "Dimensions of neuron vectors")
	flag.StringVar(&flags.FilePath, "file", flags.FilePath, "Base file path for save/load operations (without extension)")
	flag.StringVar(&flags.CPUProfile, "cpuprofile", flags.CPUProfile, "Write CPU profile to file")
	flag.StringVar(&flags.MemProfile, "memprofile", flags.MemProfile, "Write memory profile to file")
	flag.BoolVar(&flags.ParallelProc, "parallel", flags.ParallelProc, "Use parallel processing")
	flag.BoolVar(&flags.Verbose, "verbose", flags.Verbose, "Enable verbose output")

	// Parse command line flags
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

	// Execute the command
	cmd.Handler(flags)

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

// printUsage prints the usage information for all commands
func printUsage(commands map[string]Command) {
	fmt.Println("divn - Dynamic Routing Neural Network")
	fmt.Println("\nUsage: divn <command> [options]")
	fmt.Println("\nCommands:")

	// Get command names and sort them
	var cmdNames []string
	for name := range commands {
		cmdNames = append(cmdNames, name)
	}

	// Print each command with its description
	for _, name := range cmdNames {
		cmd := commands[name]
		fmt.Printf("  %-12s %s\n", name, cmd.Description)
	}

	fmt.Println("\nFor command-specific help, use: divn help <command>")
	fmt.Println("\nGlobal Options:")
	flag.PrintDefaults()
}

// printCommandHelp prints detailed help for a specific command
func printCommandHelp(cmd Command) {
	fmt.Printf("%s - %s\n\n", cmd.Name, cmd.Description)
	fmt.Printf("Usage: %s\n\n", cmd.Usage)

	if len(cmd.Examples) > 0 {
		fmt.Println("Examples:")
		for _, example := range cmd.Examples {
			fmt.Printf("  %s\n", example)
		}
		fmt.Println()
	}

	fmt.Println("Options:")
	flag.PrintDefaults()
}

// createNetwork creates a new network with the specified parameters
func createNetwork(flags CommandFlags) {
	// Create result structure for JSON output
	output := NetworkCreationOutput{}
	output.Parameters.Seed = flags.Seed
	output.Parameters.Layers = flags.LayerCount
	output.Parameters.Size = flags.NeuronSize
	output.Parameters.Dimensions = flags.VectorDims
	output.Parameters.Filename = flags.FilePath + ".json"

	// Record start time for metrics
	startTime := time.Now()

	// Create a new network
	network, err := NewNetwork("Dynamic Routing Network", flags.Seed)
	if err != nil {
		output.Status = fmt.Sprintf("error: %v", err)
		outputJSON(output)
		os.Exit(1)
	}

	// Create layers
	for i := 0; i < flags.LayerCount; i++ {
		layerName := fmt.Sprintf("Layer-%d", i+1)
		layer, err := NewLayer(layerName, flags.NeuronSize, flags.NeuronSize, flags.VectorDims, network.rand)
		if err != nil {
			output.Status = fmt.Sprintf("error: failed to create layer %s: %v", layerName, err)
			outputJSON(output)
			os.Exit(1)
		}

		// Add layer to network
		err = network.AddLayer(layer)
		if err != nil {
			output.Status = fmt.Sprintf("error: failed to add layer %s to network: %v", layerName, err)
			outputJSON(output)
			os.Exit(1)
		}
	}

	// Connect layers - always use full connections
	err = network.ConnectAllLayers()
	if err != nil {
		output.Status = fmt.Sprintf("error: failed to connect layers: %v", err)
		outputJSON(output)
		os.Exit(1)
	}

	// Ensure directory exists
	outputDir := filepath.Dir(flags.FilePath)
	if outputDir != "" && outputDir != "." {
		if err := os.MkdirAll(outputDir, 0755); err != nil {
			output.Status = fmt.Sprintf("error: failed to create output directory %s: %v", outputDir, err)
			outputJSON(output)
			os.Exit(1)
		}
	}

	// Save the network to JSON
	jsonFilePath := flags.FilePath + ".json"
	jsonSerializer := NewJSONSerializer()
	err = jsonSerializer.SaveNetworkToJSON(network, jsonFilePath)
	if err != nil {
		output.Status = fmt.Sprintf("error: failed to save network to JSON: %v", err)
		outputJSON(output)
		os.Exit(1)
	}

	// Calculate total connections
	totalConnections := 0
	for _, layer := range network.Layers {
		for _, neuron := range layer.Neurons {
			totalConnections += len(neuron.Connections)
		}
	}

	// Calculate layers connected (number of layer pairs with connections)
	layersConnected := 0
	layerCount := len(network.Layers)
	if layerCount > 1 {
		// In a fully connected network, each layer is connected to all other layers
		layersConnected = layerCount * (layerCount - 1)
	}

	// Get file size
	fileInfo, err := os.Stat(jsonFilePath)
	var fileSize int64
	if err == nil {
		fileSize = fileInfo.Size()
	}

	// Record metrics
	output.Metrics.CreationTime = time.Since(startTime).String()
	output.Metrics.TotalLayers = len(network.Layers)
	output.Metrics.TotalNeurons = len(network.NeuronCache)
	output.Metrics.TotalConnections = totalConnections
	output.Metrics.LayersConnected = layersConnected
	output.Metrics.FileSize = fileSize

	// Get the full network structure as JSON
	networkJSON := jsonSerializer.networkToJSON(network)
	output.Network = networkJSON
	output.Status = "success"

	// Output JSON result
	outputJSON(output)
}

// outputJSON outputs the result as JSON
func outputJSON(result interface{}) {
	jsonData, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Printf("Error creating JSON output: %v\n", err)
		return
	}
	fmt.Println(string(jsonData))
}

// saveNetwork saves an existing network to a file
func saveNetwork(flags CommandFlags) {
	jsonFilePath := flags.FilePath + ".json"
	fmt.Printf("Loading network from %s...\n", jsonFilePath)

	// Check if file exists
	if _, err := os.Stat(jsonFilePath); os.IsNotExist(err) {
		fmt.Printf("Error: Network file %s does not exist\n", jsonFilePath)
		os.Exit(1)
	}

	// Load the network
	jsonSerializer := NewJSONSerializer()
	network, err := jsonSerializer.LoadNetworkFromJSON(jsonFilePath, flags.Seed)
	if err != nil {
		fmt.Printf("Error loading network: %v\n", err)
		os.Exit(1)
	}

	// Ensure directory exists for output
	outputDir := filepath.Dir(flags.FilePath)
	if outputDir != "" && outputDir != "." {
		if err := os.MkdirAll(outputDir, 0755); err != nil {
			fmt.Printf("Error creating output directory %s: %v\n", outputDir, err)
			os.Exit(1)
		}
	}

	// Save the network
	fmt.Printf("Saving network to %s...\n", jsonFilePath)

	// Use JSON serialization
	err = jsonSerializer.SaveNetworkToJSON(network, jsonFilePath)
	if err != nil {
		fmt.Printf("Error saving network to JSON: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Network saved successfully to %s\n", jsonFilePath)

	if flags.Verbose {
		fmt.Printf("Network summary: %d layers, %d total neurons\n",
			len(network.Layers), len(network.NeuronCache))
	}
}

// loadNetwork loads a network from a file and displays information
func loadNetwork(flags CommandFlags) {
	jsonFilePath := flags.FilePath + ".json"
	fmt.Printf("Loading network from %s...\n", jsonFilePath)

	// Check if file exists
	if _, err := os.Stat(jsonFilePath); os.IsNotExist(err) {
		fmt.Printf("Error: Network file %s does not exist\n", jsonFilePath)
		os.Exit(1)
	}

	// Load the network using JSON serialization
	jsonSerializer := NewJSONSerializer()
	network, err := jsonSerializer.LoadNetworkFromJSON(jsonFilePath, flags.Seed)
	if err != nil {
		fmt.Printf("Error loading network: %v\n", err)
		os.Exit(1)
	}

	// Print network information
	fmt.Printf("\nNetwork Information:\n")
	fmt.Printf("Name: %s\n", network.Name)
	fmt.Printf("UUID: %s\n", network.UUID)
	fmt.Printf("Layers: %d\n", len(network.Layers))
	fmt.Printf("Total Neurons: %d\n", len(network.NeuronCache))

	// Print layer information
	fmt.Printf("\nLayer Details:\n")
	for uuid, layer := range network.Layers {
		fmt.Printf("Layer %s: %s\n", uuid, layer.Name)
		fmt.Printf("  Dimensions: %dx%d\n", layer.Width, layer.Height)
		fmt.Printf("  Neurons: %d\n", len(layer.Neurons))

		if flags.Verbose {
			// Count connections
			totalConnections := 0
			for _, neuron := range layer.Neurons {
				totalConnections += len(neuron.Connections)
			}
			fmt.Printf("  Total Connections: %d\n", totalConnections)
			fmt.Printf("  Avg Connections per Neuron: %.2f\n",
				float64(totalConnections)/float64(len(layer.Neurons)))
		}
	}

	fmt.Printf("\nNetwork loaded successfully from %s\n", jsonFilePath)
}

// processData processes data through the network
func processData(flags CommandFlags) {
	jsonFilePath := flags.FilePath + ".json"
	fmt.Printf("Loading network from %s...\n", jsonFilePath)

	// Check if file exists
	if _, err := os.Stat(jsonFilePath); os.IsNotExist(err) {
		fmt.Printf("Error: Network file %s does not exist\n", jsonFilePath)
		os.Exit(1)
	}

	// Load the network
	jsonSerializer := NewJSONSerializer()
	network, err := jsonSerializer.LoadNetworkFromJSON(jsonFilePath, flags.Seed)
	if err != nil {
		fmt.Printf("Error loading network: %v\n", err)
		os.Exit(1)
	}

	// Get the first layer and neuron for processing
	var startLayerUUID, startNeuronUUID string
	for layerUUID, layer := range network.Layers {
		startLayerUUID = layerUUID
		for neuronUUID := range layer.Neurons {
			startNeuronUUID = neuronUUID
			break
		}
		break
	}

	if startLayerUUID == "" || startNeuronUUID == "" {
		fmt.Println("Error: Could not find a starting layer and neuron")
		os.Exit(1)
	}

	// Create test data
	testData := make([]float64, flags.VectorDims)
	for i := range testData {
		testData[i] = float64(i) / float64(flags.VectorDims)
	}

	// Process data
	fmt.Println("Processing data...")
	maxSteps := 10 // Default max steps

	if flags.Verbose {
		fmt.Printf("Starting from layer UUID: %s, neuron UUID: %s\n",
			startLayerUUID, startNeuronUUID)
		fmt.Printf("Input vector dimensions: %d\n", len(testData))
		fmt.Printf("Maximum processing steps: %d\n", maxSteps)
	}

	result, path, err := network.ProcessData(startLayerUUID, startNeuronUUID, testData, maxSteps)
	if err != nil {
		fmt.Printf("Error processing data: %v\n", err)
		os.Exit(1)
	}

	// Print results
	fmt.Println("\nProcessing Results:")
	fmt.Printf("Path length: %d\n", len(path))
	fmt.Printf("Result dimensions: %d\n", len(result))

	// Print the path if verbose
	if flags.Verbose && len(path) > 0 {
		fmt.Println("\nProcessing Path:")
		for i, neuronUUID := range path {
			fmt.Printf("  Step %d: %s\n", i+1, neuronUUID)
		}
	}

	// Print the result preview
	fmt.Println("\nResult Preview:")
	previewCount := 5
	if len(result) < previewCount {
		previewCount = len(result)
	}
	for i := 0; i < previewCount; i++ {
		fmt.Printf("  Dimension %d: %f\n", i, result[i])
	}

	if len(result) > previewCount {
		fmt.Printf("  ... %d more dimensions\n", len(result)-previewCount)
	}

	fmt.Println("\nProcessing complete.")
}

// runBenchmark runs performance benchmarks
func runBenchmark(flags CommandFlags) {
	fmt.Println("Running benchmarks...")

	if flags.Verbose {
		fmt.Printf("Benchmark configuration: %d layers, %dx%d neurons, %d vector dimensions\n",
			flags.LayerCount, flags.NeuronSize, flags.NeuronSize, flags.VectorDims)
	}

	// Create a network for benchmarking
	network, err := NewNetwork("Benchmark Network", flags.Seed)
	if err != nil {
		fmt.Printf("Error creating network: %v\n", err)
		os.Exit(1)
	}

	// Create layers
	startTime := time.Now()
	for i := 0; i < flags.LayerCount; i++ {
		layerName := fmt.Sprintf("Layer-%d", i+1)
		layer, err := NewLayer(layerName, flags.NeuronSize, flags.NeuronSize, flags.VectorDims, network.rand)
		if err != nil {
			fmt.Printf("Error creating layer %s: %v\n", layerName, err)
			os.Exit(1)
		}

		// Add layer to network
		err = network.AddLayer(layer)
		if err != nil {
			fmt.Printf("Error adding layer %s to network: %v\n", layerName, err)
			os.Exit(1)
		}
	}
	layerCreationTime := time.Since(startTime)

	if flags.Verbose {
		fmt.Printf("Layer creation time: %v\n", layerCreationTime)
	}

	// Connect layers - always use full connections
	fmt.Println("Connecting all layers...")
	connectionStartTime := time.Now()
	err = network.ConnectAllLayers()
	if err != nil {
		fmt.Printf("Error connecting layers: %v\n", err)
		os.Exit(1)
	}
	connectionTime := time.Since(connectionStartTime)

	if flags.Verbose {
		fmt.Printf("Layer connection time: %v\n", connectionTime)
	}

	// Create benchmark directory
	benchmarkDir := filepath.Join(filepath.Dir(flags.FilePath), "benchmark")
	err = os.MkdirAll(benchmarkDir, 0755)
	if err != nil {
		fmt.Printf("Error creating benchmark directory: %v\n", err)
		os.Exit(1)
	}

	// Run serialization benchmark
	benchmarkFilePath := filepath.Join(benchmarkDir, "network")

	// JSON serialization benchmark
	fmt.Println("\nRunning JSON serialization benchmark...")
	jsonSerializer := NewJSONSerializer()

	// Benchmark JSON save
	jsonSaveStart := time.Now()
	jsonFilePath := benchmarkFilePath + ".json"

	err = jsonSerializer.SaveNetworkToJSON(network, jsonFilePath)
	if err != nil {
		fmt.Printf("Error saving network to JSON: %v\n", err)
	} else {
		jsonSaveTime := time.Since(jsonSaveStart)

		// Get JSON file size
		jsonFileInfo, err := os.Stat(jsonFilePath)
		if err != nil {
			fmt.Printf("Error getting JSON file info: %v\n", err)
		} else {
			jsonFileSize := jsonFileInfo.Size()

			// Benchmark JSON load
			jsonLoadStart := time.Now()
			_, err = jsonSerializer.LoadNetworkFromJSON(jsonFilePath, flags.Seed)
			if err != nil {
				fmt.Printf("Error loading network from JSON: %v\n", err)
			} else {
				jsonLoadTime := time.Since(jsonLoadStart)

				// Print results
				fmt.Printf("JSON save time: %v\n", jsonSaveTime)
				fmt.Printf("JSON load time: %v\n", jsonLoadTime)
				fmt.Printf("JSON file size: %v bytes\n", jsonFileSize)
			}
		}
	}

	// Run binary serialization benchmark
	BinarySerializationBenchmark(network, benchmarkFilePath, flags.Seed)

	// Run connection map benchmark
	fmt.Println("\nRunning connection map benchmark...")

	// Extract connection map
	connectionMapStart := time.Now()
	connectionMap := ExtractConnectionMapFromNetwork(network)
	connectionMapTime := time.Since(connectionMapStart)

	// Save connection map
	connectionMapFilePath := benchmarkFilePath + ".connections.json"
	connectionMapSaveStart := time.Now()
	err = SaveConnectionMapToJSON(connectionMap, connectionMapFilePath)
	if err != nil {
		fmt.Printf("Error saving connection map: %v\n", err)
	} else {
		connectionMapSaveTime := time.Since(connectionMapSaveStart)

		// Get connection map file size
		connectionMapFileInfo, err := os.Stat(connectionMapFilePath)
		if err != nil {
			fmt.Printf("Error getting connection map file info: %v\n", err)
		} else {
			connectionMapFileSize := connectionMapFileInfo.Size()

			// Load connection map
			connectionMapLoadStart := time.Now()
			_, err = LoadConnectionMapFromJSON(connectionMapFilePath)
			if err != nil {
				fmt.Printf("Error loading connection map: %v\n", err)
			} else {
				connectionMapLoadTime := time.Since(connectionMapLoadStart)

				// Print results
				fmt.Printf("Connection map extraction time: %v\n", connectionMapTime)
				fmt.Printf("Connection map save time: %v\n", connectionMapSaveTime)
				fmt.Printf("Connection map load time: %v\n", connectionMapLoadTime)
				fmt.Printf("Connection map file size: %v bytes\n", connectionMapFileSize)
			}
		}
	}

	fmt.Println("\nBenchmark complete.")
}
