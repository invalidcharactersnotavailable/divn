package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Neuron represents a processing unit in the network
type Neuron struct {
	ID          string             `json:"id"`
	Value       *Vector            `json:"value"`       // Vector for data transformation
	Resistance  float64            `json:"resistance"`  // Resistance value for routing
	Connections map[string]float64 `json:"connections"` // Map of connected neuron IDs to connection strengths
	mu          sync.RWMutex       `json:"-"`           // Mutex for thread safety
}

// NewNeuron creates a new neuron with the specified vector dimensions using provided random source
func NewNeuron(id string, vectorDims int, r *rand.Rand) (*Neuron, error) {
	if id == "" {
		return nil, fmt.Errorf("neuron ID cannot be empty")
	}

	if r == nil {
		return nil, fmt.Errorf("random source cannot be nil")
	}

	if vectorDims <= 0 {
		return nil, fmt.Errorf("vector dimensions must be positive, got %d", vectorDims)
	}

	vector, err := NewRandomVector(vectorDims, r)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector: %w", err)
	}

	return &Neuron{
		ID:          id,
		Value:       vector,
		Resistance:  r.Float64(), // Initial random resistance
		Connections: make(map[string]float64),
	}, nil
}

// NewNeuronWithUniqueVector creates a neuron with a uniquely dimensioned vector
func NewNeuronWithUniqueVector(id string, vectorDims int, seed int64) (*Neuron, error) {
	if id == "" {
		return nil, fmt.Errorf("neuron ID cannot be empty")
	}

	if vectorDims <= 0 {
		return nil, fmt.Errorf("vector dimensions must be positive, got %d", vectorDims)
	}

	// Create a deterministic random source for resistance
	r := rand.New(rand.NewSource(seed))

	vector, err := NewUniqueVector(vectorDims, seed)
	if err != nil {
		return nil, fmt.Errorf("failed to create unique vector: %w", err)
	}

	return &Neuron{
		ID:          id,
		Value:       vector,
		Resistance:  r.Float64(), // Deterministic resistance based on seed
		Connections: make(map[string]float64),
	}, nil
}

// Connect establishes a connection to another neuron with a given strength
// If the connection already exists, the strength is added to the existing value
func (n *Neuron) Connect(targetID string, strength float64) error {
	if targetID == "" {
		return fmt.Errorf("target ID cannot be empty")
	}

	n.mu.Lock()
	defer n.mu.Unlock()

	// Add to existing connection strength if it exists
	if existingStrength, exists := n.Connections[targetID]; exists {
		n.Connections[targetID] = existingStrength + strength
	} else {
		n.Connections[targetID] = strength
	}

	return nil
}

// GetConnections returns a copy of the neuron's connections
func (n *Neuron) GetConnections() map[string]float64 {
	n.mu.RLock()
	defer n.mu.RUnlock()

	connections := make(map[string]float64, len(n.Connections))
	for id, strength := range n.Connections {
		connections[id] = strength
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

// Layer represents a collection of neurons in a 2D grid
type Layer struct {
	ID      string             `json:"id"`
	Width   int                `json:"width"`   // Width of the neuron grid
	Height  int                `json:"height"`  // Height of the neuron grid
	Neurons map[string]*Neuron `json:"neurons"` // Map of neuron IDs to neurons
	mu      sync.RWMutex       `json:"-"`       // Mutex for thread safety
}

// NewLayer creates a new layer with the specified dimensions and vector size
func NewLayer(id string, width, height, vectorDims int, r *rand.Rand) (*Layer, error) {
	if id == "" {
		return nil, fmt.Errorf("layer ID cannot be empty")
	}

	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid dimensions: width=%d, height=%d", width, height)
	}

	if r == nil {
		return nil, fmt.Errorf("random source cannot be nil")
	}

	layer := &Layer{
		ID:      id,
		Width:   width,
		Height:  height,
		Neurons: make(map[string]*Neuron),
	}

	// Create neurons in a grid pattern
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			neuronID := fmt.Sprintf("%s_n%d_%d", id, x, y)
			neuron, err := NewNeuron(neuronID, vectorDims, r)
			if err != nil {
				return nil, fmt.Errorf("failed to create neuron at (%d,%d): %w", x, y, err)
			}
			layer.Neurons[neuronID] = neuron
		}
	}

	return layer, nil
}

// NewLayerWithUniqueVectors creates a layer with neurons having unique vector dimensions
func NewLayerWithUniqueVectors(id string, width, height, vectorDims int, baseSeed int64) (*Layer, error) {
	if id == "" {
		return nil, fmt.Errorf("layer ID cannot be empty")
	}

	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid dimensions: width=%d, height=%d", width, height)
	}

	layer := &Layer{
		ID:      id,
		Width:   width,
		Height:  height,
		Neurons: make(map[string]*Neuron),
	}

	// Create neurons with unique vectors
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			neuronID := fmt.Sprintf("%s_n%d_%d", id, x, y)
			seed := baseSeed + int64(x*10000+y) // Create a unique seed based on position
			neuron, err := NewNeuronWithUniqueVector(neuronID, vectorDims, seed)
			if err != nil {
				return nil, fmt.Errorf("failed to create neuron at (%d,%d): %w", x, y, err)
			}
			layer.Neurons[neuronID] = neuron
		}
	}

	return layer, nil
}

// GetNeuron returns a neuron by ID
func (l *Layer) GetNeuron(id string) (*Neuron, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	neuron, exists := l.Neurons[id]
	return neuron, exists
}

// ConnectInternalNeurons connects neurons within the layer
func (l *Layer) ConnectInternalNeurons(r *rand.Rand) error {
	if r == nil {
		return fmt.Errorf("random source cannot be nil")
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	// Connect each neuron to its neighbors
	for x := 0; x < l.Width; x++ {
		for y := 0; y < l.Height; y++ {
			sourceID := fmt.Sprintf("%s_n%d_%d", l.ID, x, y)
			source, exists := l.Neurons[sourceID]
			if !exists {
				return fmt.Errorf("source neuron not found: %s", sourceID)
			}

			// Connect to neighbors (including diagonals)
			for dx := -1; dx <= 1; dx++ {
				for dy := -1; dy <= 1; dy++ {
					if dx == 0 && dy == 0 {
						continue // Skip self
					}

					nx, ny := x+dx, y+dy
					if nx >= 0 && nx < l.Width && ny >= 0 && ny < l.Height {
						targetID := fmt.Sprintf("%s_n%d_%d", l.ID, nx, ny)
						// Random initial connection strength
						err := source.Connect(targetID, r.Float64())
						if err != nil {
							return fmt.Errorf("failed to connect %s to %s: %w", sourceID, targetID, err)
						}
					}
				}
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
	ID          string             `json:"id"`
	Layers      map[string]*Layer  `json:"layers"`
	NeuronCache map[string]*Neuron `json:"-"` // Global neuron cache for O(1) lookup
	mu          sync.RWMutex       `json:"-"` // Mutex for thread safety
	rand        *rand.Rand         `json:"-"` // Random source for deterministic operations
}

// NewNetwork creates a new network with the given ID and random seed
func NewNetwork(id string, seed int64) (*Network, error) {
	if id == "" {
		return nil, fmt.Errorf("network ID cannot be empty")
	}

	return &Network{
		ID:          id,
		Layers:      make(map[string]*Layer),
		NeuronCache: make(map[string]*Neuron),
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

	// Check if layer ID already exists
	if _, exists := n.Layers[layer.ID]; exists {
		return fmt.Errorf("layer with ID %s already exists", layer.ID)
	}

	n.Layers[layer.ID] = layer

	// Update neuron cache with all neurons from this layer
	for neuronID, neuron := range layer.Neurons {
		if _, exists := n.NeuronCache[neuronID]; exists {
			return fmt.Errorf("neuron ID conflict: %s already exists in network", neuronID)
		}
		n.NeuronCache[neuronID] = neuron
	}

	return nil
}

// GetNeuron returns a neuron from the cache by ID
func (n *Network) GetNeuron(id string) (*Neuron, bool) {
	n.mu.RLock()
	defer n.mu.RUnlock()

	neuron, exists := n.NeuronCache[id]
	return neuron, exists
}

// ConnectLayers connects neurons between two layers
func (n *Network) ConnectLayers(sourceLayerID, targetLayerID string) error {
	n.mu.Lock()
	defer n.mu.Unlock()

	sourceLayer, sourceExists := n.Layers[sourceLayerID]
	if !sourceExists {
		return fmt.Errorf("source layer not found: %s", sourceLayerID)
	}

	targetLayer, targetExists := n.Layers[targetLayerID]
	if !targetExists {
		return fmt.Errorf("target layer not found: %s", targetLayerID)
	}

	// Connect each neuron in source layer to some neurons in target layer
	for _, sourceNeuron := range sourceLayer.Neurons {
		// Connect to a random subset of neurons in the target layer
		// This can be modified to use different connection strategies
		connectionCount := n.rand.Intn(5) + 1 // 1-5 connections per neuron

		// Get a random sample of target neurons
		targetNeurons := make([]string, 0, len(targetLayer.Neurons))
		for id := range targetLayer.Neurons {
			targetNeurons = append(targetNeurons, id)
		}

		// Shuffle and select a subset
		for i := range targetNeurons {
			j := n.rand.Intn(i + 1)
			targetNeurons[i], targetNeurons[j] = targetNeurons[j], targetNeurons[i]
		}

		// Connect to the selected subset
		for i := 0; i < connectionCount && i < len(targetNeurons); i++ {
			err := sourceNeuron.Connect(targetNeurons[i], n.rand.Float64())
			if err != nil {
				return fmt.Errorf("failed to connect %s to %s: %w", sourceNeuron.ID, targetNeurons[i], err)
			}
		}
	}

	return nil
}

// ProcessData processes input data through the network
// It returns the final transformed data and the path of neuron IDs taken
func (n *Network) ProcessData(startLayerID, startNeuronID string, data []float64, maxSteps int) ([]float64, []string, error) {
	if data == nil {
		return nil, nil, fmt.Errorf("input data cannot be nil")
	}

	if maxSteps <= 0 {
		return nil, nil, fmt.Errorf("maxSteps must be positive")
	}

	// Validate starting point
	n.mu.RLock()
	startLayer, layerExists := n.Layers[startLayerID]
	if !layerExists {
		n.mu.RUnlock()
		return nil, nil, fmt.Errorf("starting layer not found: %s", startLayerID)
	}
	n.mu.RUnlock()

	startNeuron, neuronExists := startLayer.GetNeuron(startNeuronID)
	if !neuronExists {
		return nil, nil, fmt.Errorf("starting neuron not found: %s", startNeuronID)
	}

	// Initialize processing
	currentNeuron := startNeuron
	currentData := data
	path := []string{startNeuronID}

	// Process data through the network
	for step := 0; step < maxSteps; step++ {
		// Transform data using current neuron
		var err error
		currentData, err = currentNeuron.TransformData(currentData)
		if err != nil {
			return nil, path, fmt.Errorf("error transforming data at step %d: %w", step, err)
		}

		// Find the connected neuron with lowest resistance
		var nextNeuronID string
		lowestResistance := math.MaxFloat64

		// Get all connections from the current neuron
		connections := currentNeuron.GetConnections()

		for connectedID := range connections {
			// Use the neuron cache for O(1) lookup
			n.mu.RLock()
			connectedNeuron, exists := n.NeuronCache[connectedID]
			n.mu.RUnlock()

			if exists {
				resistance := connectedNeuron.GetResistance()
				if resistance < lowestResistance {
					lowestResistance = resistance
					nextNeuronID = connectedID
				}
			}
		}

		// If no connected neurons or reached a dead end
		if nextNeuronID == "" {
			break
		}

		// Move to the next neuron using the cache
		n.mu.RLock()
		nextNeuron, exists := n.NeuronCache[nextNeuronID]
		n.mu.RUnlock()

		if !exists {
			return nil, path, fmt.Errorf("neuron not found in cache: %s", nextNeuronID)
		}

		currentNeuron = nextNeuron
		path = append(path, nextNeuronID)
	}

	return currentData, path, nil
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

	// Initialize non-serialized fields
	network.NeuronCache = make(map[string]*Neuron)
	network.rand = rand.New(rand.NewSource(seed))

	// Rebuild the neuron cache
	for _, layer := range network.Layers {
		for neuronID, neuron := range layer.Neurons {
			network.NeuronCache[neuronID] = neuron
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
func (v *Vector) Transform(data []float64) ([]float64, error) {
	if v == nil {
		return nil, fmt.Errorf("vector is nil")
	}

	if data == nil {
		return nil, fmt.Errorf("input data is nil")
	}

	// Simple transformation: element-wise multiplication
	// This can be replaced with more complex transformations
	result := make([]float64, len(v.Dimensions))
	for i := range v.Dimensions {
		if i < len(data) {
			result[i] = v.Dimensions[i] * data[i]
		} else {
			result[i] = v.Dimensions[i]
		}
	}
	return result, nil
}

func main() {
	// Parse command line flags
	var (
		seed       int64
		command    string
		layerCount int
		neuronSize int
		vectorDims int
		filePath   string
	)

	// Set up command line flags
	flag.Int64Var(&seed, "seed", time.Now().UnixNano(), "Random seed for reproducible network generation")
	flag.IntVar(&layerCount, "layers", 3, "Number of layers in the network")
	flag.IntVar(&neuronSize, "size", 8, "Size of neuron grid (size x size)")
	flag.IntVar(&vectorDims, "dims", 4, "Dimensions of neuron vectors")
	flag.StringVar(&filePath, "file", "network.json", "File path for save/load operations")

	// Define subcommands
	createCmd := flag.NewFlagSet("create", flag.ExitOnError)
	loadCmd := flag.NewFlagSet("load", flag.ExitOnError)
	testCmd := flag.NewFlagSet("test", flag.ExitOnError)

	// Parse the main command
	flag.Parse()

	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	command = os.Args[1]

	// Handle subcommands
	switch command {
	case "create":
		createCmd.Parse(os.Args[2:])
		createNetwork(seed, layerCount, neuronSize, vectorDims, filePath)
	case "load":
		loadCmd.Parse(os.Args[2:])
		loadNetwork(seed, filePath)
	case "test":
		testCmd.Parse(os.Args[2:])
		testNetwork(seed)
	default:
		fmt.Printf("Unknown command: %s\n", command)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("dynamic intraconnected vector network")
	fmt.Println("\nUsage:")
	fmt.Println("  ./divn [flags] <command>")
	fmt.Println("\nFlags:")
	fmt.Println("  -seed int     Random seed for reproducible network generation (default: current time)")
	fmt.Println("  -layers int   Number of layers in the network (default: 3)")
	fmt.Println("  -size int     Size of neuron grid (size x size) (default: 8)")
	fmt.Println("  -dims int     Dimensions of neuron vectors (default: 4)")
	fmt.Println("  -file string  File path for save/load operations (default: network.json)")
	fmt.Println("\nCommands:")
	fmt.Println("  create        Create a new network with specified parameters")
	fmt.Println("  load          Load a network from file")
	fmt.Println("  test          Run a test of network operations")
}

func createNetwork(seed int64, layerCount, neuronSize, vectorDims int, filePath string) {
	fmt.Println("Creating a new dynamic network...")
	fmt.Printf("Seed: %d, Layers: %d, Grid Size: %dx%d, Vector Dimensions: %d\n",
		seed, layerCount, neuronSize, neuronSize, vectorDims)

	// Initialize random source with seed for reproducibility
	r := rand.New(rand.NewSource(seed))

	// Create a new network
	network, err := NewNetwork("test_network", seed)
	if err != nil {
		fmt.Printf("Error creating network: %v\n", err)
		os.Exit(1)
	}

	// Create and add layers
	for i := 0; i < layerCount; i++ {
		layerID := fmt.Sprintf("layer%d", i+1)

		var layer *Layer
		var err error

		// Make some layers with unique vectors for variety
		if i%2 == 0 {
			layer, err = NewLayer(layerID, neuronSize, neuronSize, vectorDims, r)
		} else {
			layer, err = NewLayerWithUniqueVectors(layerID, neuronSize, neuronSize, vectorDims, seed+int64(i*1000))
		}

		if err != nil {
			fmt.Printf("Error creating layer %s: %v\n", layerID, err)
			os.Exit(1)
		}

		// Connect neurons within the layer
		err = layer.ConnectInternalNeurons(r)
		if err != nil {
			fmt.Printf("Error connecting neurons in layer %s: %v\n", layerID, err)
			os.Exit(1)
		}

		// Add layer to network
		err = network.AddLayer(layer)
		if err != nil {
			fmt.Printf("Error adding layer %s to network: %v\n", layerID, err)
			os.Exit(1)
		}

		fmt.Printf("Added layer %s: %dx%d grid with %d neurons\n",
			layerID, layer.Width, layer.Height, len(layer.Neurons))
	}

	// Connect layers sequentially
	for i := 0; i < layerCount-1; i++ {
		sourceID := fmt.Sprintf("layer%d", i+1)
		targetID := fmt.Sprintf("layer%d", i+2)

		err := network.ConnectLayers(sourceID, targetID)
		if err != nil {
			fmt.Printf("Error connecting layers %s and %s: %v\n", sourceID, targetID, err)
			os.Exit(1)
		}

		fmt.Printf("Connected layer %s to layer %s\n", sourceID, targetID)
	}

	// Create a circular reference for demonstration (last layer to first)
	if layerCount > 2 {
		sourceID := fmt.Sprintf("layer%d", layerCount)
		targetID := "layer1"

		err := network.ConnectLayers(sourceID, targetID)
		if err != nil {
			fmt.Printf("Error connecting layers %s and %s: %v\n", sourceID, targetID, err)
			os.Exit(1)
		}

		fmt.Printf("Connected layer %s to layer %s (circular reference)\n", sourceID, targetID)
	}

	// Save the network to a file
	err = network.SaveToFile(filePath)
	if err != nil {
		fmt.Printf("Error saving network: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Network created and saved to %s\n", filePath)
}

func loadNetwork(seed int64, filePath string) {
	fmt.Printf("Loading network from %s...\n", filePath)

	// Load the network from file
	network, err := LoadNetworkFromFile(filePath, seed)
	if err != nil {
		fmt.Printf("Error loading network: %v\n", err)
		os.Exit(1)
	}

	// Display some information about the loaded network
	fmt.Printf("Loaded network ID: %s\n", network.ID)
	fmt.Printf("Number of layers: %d\n", len(network.Layers))

	for layerID, layer := range network.Layers {
		fmt.Printf("Layer %s: %dx%d grid with %d neurons\n",
			layerID, layer.Width, layer.Height, len(layer.Neurons))
	}
}

func testNetwork(seed int64) {
	fmt.Println("Testing network operations...")
	fmt.Printf("Using seed: %d\n", seed)

	// Initialize random source with seed for reproducibility
	r := rand.New(rand.NewSource(seed))

	// Create a test network
	network, err := NewNetwork("test_network", seed)
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

	// Get a starting neuron
	startNeuronID := "test_layer_n0_0" // Top-left neuron

	// Create some test data
	testData := []float64{0.5, 0.7}

	// Process the data through the network
	result, path, err := network.ProcessData("test_layer", startNeuronID, testData, 10)
	if err != nil {
		fmt.Printf("Error processing data: %v\n", err)
		os.Exit(1)
	}

	// Display results
	fmt.Println("Data processing complete")
	fmt.Printf("Input data: %v\n", testData)
	fmt.Printf("Output data: %v\n", result)
	fmt.Printf("Path taken: %v\n", path)

	// Save just the layer to demonstrate layer persistence
	err = layer.SaveToFile("test_layer.json")
	if err != nil {
		fmt.Printf("Error saving layer: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Test layer saved to test_layer.json")
}
