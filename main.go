package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// ==================== LOGGER IMPLEMENTATION ====================

// LogLevel represents the verbosity level of logging
type LogLevel int

const (
	LogLevelNone LogLevel = iota
	LogLevelError
	LogLevelInfo
	LogLevelDebug
	LogLevelTrace
)

// Logger provides a configurable logging interface
type Logger struct {
	Level       LogLevel
	Writer      io.Writer
	ShowTime    bool
	mu          sync.Mutex
	startTime   time.Time
	lastLogTime time.Time
	// Minimum time between progress logs (to avoid flooding)
	ProgressInterval time.Duration
}

// NewLogger creates a new logger with the specified level
func NewLogger(level LogLevel, writer io.Writer, showTime bool) *Logger {
	if writer == nil {
		writer = os.Stdout
	}
	now := time.Now()
	return &Logger{
		Level:            level,
		Writer:           writer,
		ShowTime:         showTime,
		startTime:        now,
		lastLogTime:      now,
		ProgressInterval: 500 * time.Millisecond, // Default to 500ms between progress logs
	}
}

// SetLevel changes the logging level
func (l *Logger) SetLevel(level LogLevel) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.Level = level
}

// SetProgressInterval sets the minimum time between progress logs
func (l *Logger) SetProgressInterval(interval time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.ProgressInterval = interval
}

// Error logs an error message
func (l *Logger) Error(format string, args ...interface{}) {
	if l.Level >= LogLevelError {
		l.log("ERROR", format, args...)
	}
}

// Info logs an informational message
func (l *Logger) Info(format string, args ...interface{}) {
	if l.Level >= LogLevelInfo {
		l.log("INFO", format, args...)
	}
}

// Debug logs a debug message
func (l *Logger) Debug(format string, args ...interface{}) {
	if l.Level >= LogLevelDebug {
		l.log("DEBUG", format, args...)
	}
}

// Trace logs a trace message
func (l *Logger) Trace(format string, args ...interface{}) {
	if l.Level >= LogLevelTrace {
		l.log("TRACE", format, args...)
	}
}

// Progress logs a progress message with percentage and ETA
// This function will throttle output based on ProgressInterval
func (l *Logger) Progress(operation string, current, total int, startTime time.Time) {
	if l.Level >= LogLevelInfo {
		l.mu.Lock()
		now := time.Now()
		// Only log if enough time has passed since last progress log
		// or if this is the first or last item
		if current == 1 || current == total ||
			now.Sub(l.lastLogTime) >= l.ProgressInterval {

			l.mu.Unlock() // Unlock before calling log

			if total <= 0 {
				l.Info("%s: Processing item %d", operation, current)
				return
			}

			percentage := float64(current) / float64(total) * 100
			elapsed := time.Since(startTime)

			var eta time.Duration
			if current > 0 {
				eta = time.Duration(float64(elapsed) * (float64(total-current) / float64(current)))
			}

			l.Info("%s: %d/%d (%.1f%%) - ETA: %s",
				operation, current, total, percentage, FormatDuration(eta))

			l.mu.Lock()
			l.lastLogTime = now
		}
		l.mu.Unlock()
	}
}

// log formats and writes a log message
func (l *Logger) log(level, format string, args ...interface{}) {
	l.mu.Lock()
	defer l.mu.Unlock()

	message := fmt.Sprintf(format, args...)

	if l.ShowTime {
		elapsed := time.Since(l.startTime)
		fmt.Fprintf(l.Writer, "[%s][%s] %s\n", FormatDuration(elapsed), level, message)
	} else {
		fmt.Fprintf(l.Writer, "[%s] %s\n", level, message)
	}
}

// Global logger instance
var GlobalLogger = NewLogger(LogLevelInfo, os.Stdout, true)

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

	return &Vector{
		Dimensions: make([]float64, dims),
	}, nil
}

// NewRandomVector creates a new vector with random values using provided random source
func NewRandomVector(dims int, r *rand.Rand) (*Vector, error) {
	if r == nil {
		return nil, fmt.Errorf("random source cannot be nil")
	}

	v, err := NewVector(dims)
	if err != nil {
		return nil, err
	}

	// Fill dimensions with random values in a single loop
	for i := range v.Dimensions {
		v.Dimensions[i] = r.Float64()*2 - 1 // Values between -1 and 1
	}
	GlobalLogger.Trace("Created random vector with %d dimensions", dims)
	return v, nil
}

// NewUniqueVector creates a vector with a unique dimension configuration
// The uniqueness is determined by the seed value
func NewUniqueVector(dims int, seed int64) (*Vector, error) {
	r := rand.New(rand.NewSource(seed))
	v, err := NewRandomVector(dims, r)
	if err != nil {
		return nil, err
	}
	GlobalLogger.Trace("Created unique vector with %d dimensions (seed: %d)", dims, seed)
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
	minLen := min(len(data), len(v.Dimensions))

	// Process the overlapping portion
	for i := 0; i < minLen; i++ {
		result[i] = v.Dimensions[i] * data[i]
	}

	// Copy remaining dimensions if vector is longer than data
	copy(result[minLen:], v.Dimensions[minLen:])

	GlobalLogger.Trace("Transformed data with %d dimensions", len(v.Dimensions))
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
	GlobalLogger.Debug("Resizing vector from %d to %d dimensions", currentDims, newDims)

	// If no change in dimensions, return early
	if newDims == currentDims {
		return nil
	}

	// Create new dimensions slice
	newDimensions := make([]float64, newDims)

	// Copy existing dimensions
	copyLen := min(currentDims, newDims)
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

	vector, err := NewRandomVector(vectorDims, r)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector: %w", err)
	}

	neuronUUID := uuid.NewString()
	GlobalLogger.Trace("Created neuron with UUID %s and %d dimensions", neuronUUID, vectorDims)

	return &Neuron{
		UUID:        neuronUUID,
		Value:       vector,
		Resistance:  r.Float64(),                 // Initial random resistance
		Connections: make(map[string]float64, 8), // Pre-allocate with expected size
	}, nil
}

// NewNeuronWithUniqueVector creates a neuron with a uniquely dimensioned vector
func NewNeuronWithUniqueVector(vectorDims int, seed int64) (*Neuron, error) {
	// Create a deterministic random source for resistance
	r := rand.New(rand.NewSource(seed))

	vector, err := NewUniqueVector(vectorDims, seed)
	if err != nil {
		return nil, fmt.Errorf("failed to create unique vector: %w", err)
	}

	neuronUUID := uuid.NewString()
	GlobalLogger.Trace("Created neuron with UUID %s and unique vector (seed: %d)", neuronUUID, seed)

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
	// Add to existing connection strength if it exists
	n.Connections[targetUUID] += strength
	n.mu.Unlock()

	GlobalLogger.Trace("Connected neuron %s to %s with strength %.4f", n.UUID, targetUUID, strength)
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

	GlobalLogger.Trace("Retrieved %d connections for neuron %s", len(connections), n.UUID)
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

	GlobalLogger.Trace("Updated neuron %s resistance from %.4f to %.4f", n.UUID, n.Resistance, resistance)
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
	layerUUID := uuid.NewString()

	GlobalLogger.Info("Creating layer '%s' with %d neurons (%dx%d grid)", name, capacity, width, height)
	startTime := time.Now()

	layer := &Layer{
		UUID:    layerUUID,
		Name:    name,
		Width:   width,
		Height:  height,
		Neurons: make(map[string]*Neuron, capacity),
	}

	// Create neurons in a grid pattern
	// Only log progress for large layers (>100 neurons)
	logProgress := capacity > 100
	for i := 0; i < capacity; i++ {
		if logProgress && (i+1)%100 == 0 {
			GlobalLogger.Progress("Creating neurons", i+1, capacity, startTime)
		}

		neuron, err := NewNeuron(vectorDims, r)
		if err != nil {
			return nil, fmt.Errorf("failed to create neuron at index %d: %w", i, err)
		}
		layer.Neurons[neuron.UUID] = neuron
	}

	GlobalLogger.Info("Layer '%s' created with %d neurons in %s", name, capacity, FormatDuration(time.Since(startTime)))
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
	layerUUID := uuid.NewString()

	GlobalLogger.Info("Creating layer '%s' with %d unique neurons (%dx%d grid)", name, capacity, width, height)
	startTime := time.Now()

	layer := &Layer{
		UUID:    layerUUID,
		Name:    name,
		Width:   width,
		Height:  height,
		Neurons: make(map[string]*Neuron, capacity),
	}

	// Create neurons with unique vectors
	// Only log progress for large layers (>100 neurons)
	logProgress := capacity > 100
	for i := 0; i < capacity; i++ {
		if logProgress && (i+1)%100 == 0 {
			GlobalLogger.Progress("Creating unique neurons", i+1, capacity, startTime)
		}

		seed := baseSeed + int64(i) // Create a unique seed based on position
		neuron, err := NewNeuronWithUniqueVector(vectorDims, seed)
		if err != nil {
			return nil, fmt.Errorf("failed to create neuron at index %d: %w", i, err)
		}
		layer.Neurons[neuron.UUID] = neuron
	}

	GlobalLogger.Info("Layer '%s' created with %d unique neurons in %s",
		name, capacity, FormatDuration(time.Since(startTime)))
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

	neuronCount := len(neuronUUIDs)
	totalConnections := neuronCount * (neuronCount - 1)
	connectionsMade := 0

	GlobalLogger.Info("Connecting %d neurons internally in layer '%s' (%d connections)",
		neuronCount, l.Name, totalConnections)
	startTime := time.Now()

	// Connect each neuron to ALL other neurons in the layer (fully connected)
	// Only log progress for large connection operations (>1000 connections)
	logProgress := totalConnections > 1000
	progressInterval := max(1, totalConnections/10) // Log at most 10 times

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
			if err := source.Connect(targetUUID, r.Float64()); err != nil {
				return fmt.Errorf("failed to connect %s to %s: %w", sourceUUID, targetUUID, err)
			}
			connectionsMade++

			// Log progress at intervals
			if logProgress && connectionsMade%progressInterval == 0 {
				GlobalLogger.Progress("Connecting neurons", connectionsMade, totalConnections, startTime)
			}
		}
	}

	GlobalLogger.Info("Connected %d neurons in layer '%s' with %d connections in %s",
		neuronCount, l.Name, connectionsMade, FormatDuration(time.Since(startTime)))
	return nil
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

	networkUUID := uuid.NewString()
	GlobalLogger.Info("Creating network '%s' with seed %d", name, seed)

	return &Network{
		UUID:        networkUUID,
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

	GlobalLogger.Info("Adding layer '%s' to network '%s'", layer.Name, n.Name)
	startTime := time.Now()

	n.Layers[layer.UUID] = layer

	// Update neuron cache with all neurons from this layer
	neuronCount := len(layer.Neurons)
	neuronIndex := 0

	// Only log progress for large neuron counts (>1000 neurons)
	logProgress := neuronCount > 1000
	for neuronUUID, neuron := range layer.Neurons {
		if logProgress && (neuronIndex+1)%500 == 0 {
			GlobalLogger.Progress("Caching neurons", neuronIndex+1, neuronCount, startTime)
		}

		if _, exists := n.NeuronCache[neuronUUID]; exists {
			return fmt.Errorf("neuron UUID conflict: %s already exists in network", neuronUUID)
		}
		n.NeuronCache[neuronUUID] = neuron
		neuronIndex++
	}

	GlobalLogger.Info("Added layer '%s' with %d neurons to network in %s",
		layer.Name, neuronCount, FormatDuration(time.Since(startTime)))

	// Connect neurons within the layer (intraconnections)
	connectionStartTime := time.Now()
	GlobalLogger.Info("Connecting neurons within layer '%s'", layer.Name)

	if err := layer.ConnectInternalNeurons(n.rand); err != nil {
		return fmt.Errorf("failed to connect internal neurons in layer %s: %w", layer.Name, err)
	}

	GlobalLogger.Info("Connected neurons within layer '%s' in %s",
		layer.Name, FormatDuration(time.Since(connectionStartTime)))

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

	GlobalLogger.Info("Connecting layer '%s' to layer '%s'", sourceLayer.Name, targetLayer.Name)
	startTime := time.Now()

	// Get all target neuron UUIDs for efficient access
	targetNeurons := make([]string, 0, len(targetLayer.Neurons))
	for uuid := range targetLayer.Neurons {
		targetNeurons = append(targetNeurons, uuid)
	}

	// Connect each neuron in source layer to some neurons in target layer
	sourceNeuronCount := len(sourceLayer.Neurons)
	neuronIndex := 0
	totalConnections := 0

	// Only log progress for large source layers (>100 neurons)
	logProgress := sourceNeuronCount > 100
	for _, sourceNeuron := range sourceLayer.Neurons {
		if logProgress && (neuronIndex+1)%50 == 0 {
			GlobalLogger.Progress("Connecting neurons between layers", neuronIndex+1, sourceNeuronCount, startTime)
		}

		// Connect to a random subset of neurons in the target layer
		connectionCount := n.rand.Intn(5) + 1 // 1-5 connections per neuron

		// Fisher-Yates shuffle for efficient random selection
		for i := 0; i < min(connectionCount, len(targetNeurons)); i++ {
			j := n.rand.Intn(len(targetNeurons)-i) + i
			targetNeurons[i], targetNeurons[j] = targetNeurons[j], targetNeurons[i]

			if err := sourceNeuron.Connect(targetNeurons[i], n.rand.Float64()); err != nil {
				return fmt.Errorf("failed to connect %s to %s: %w", sourceNeuron.UUID, targetNeurons[i], err)
			}
			totalConnections++
		}

		neuronIndex++
	}

	GlobalLogger.Info("Connected layer '%s' to layer '%s' with %d connections in %s",
		sourceLayer.Name, targetLayer.Name, totalConnections, FormatDuration(time.Since(startTime)))
	return nil
}

// ConnectAllLayers connects each layer to every other layer in the network
// This creates a fully connected network where every layer is connected to all others
func (n *Network) ConnectAllLayers() error {
	n.mu.RLock()
	// Get all layer UUIDs
	layerUUIDs := make([]string, 0, len(n.Layers))
	for uuid := range n.Layers {
		layerUUIDs = append(layerUUIDs, uuid)
	}
	n.mu.RUnlock()

	layerCount := len(layerUUIDs)

	// Skip if there's only one or zero layers (nothing to connect)
	if layerCount <= 1 {
		GlobalLogger.Info("Skipping layer connections for network '%s' (only %d layer present)",
			n.Name, layerCount)
		return nil
	}

	totalConnections := layerCount * (layerCount - 1)

	GlobalLogger.Info("Connecting all %d layers in network '%s' (%d layer connections)",
		layerCount, n.Name, totalConnections)
	startTime := time.Now()
	connectionsMade := 0

	// For each layer, connect it to all other layers
	for i, sourceUUID := range layerUUIDs {
		for j, targetUUID := range layerUUIDs {
			// Skip self-connections
			if i == j {
				continue
			}

			// Connect the layers
			if err := n.ConnectLayers(sourceUUID, targetUUID); err != nil {
				n.mu.RLock()
				sourceName := n.Layers[sourceUUID].Name
				targetName := n.Layers[targetUUID].Name
				n.mu.RUnlock()
				return fmt.Errorf("failed to connect layer %s to layer %s: %w",
					sourceName, targetName, err)
			}

			connectionsMade++
			GlobalLogger.Progress("Connecting layers", connectionsMade, totalConnections, startTime)
		}
	}

	GlobalLogger.Info("Connected all %d layers in network '%s' in %s",
		layerCount, n.Name, FormatDuration(time.Since(startTime)))
	return nil
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

	GlobalLogger.Info("Saving network '%s' to file %s", network.Name, filePath)
	startTime := time.Now()

	// Create a temporary file in the same directory
	tempFile := filePath + ".tmp"
	file, err := os.Create(tempFile)
	if err != nil {
		return fmt.Errorf("failed to create temporary file: %w", err)
	}
	defer file.Close()

	// Create a buffered writer for better performance
	writer := bufio.NewWriter(file)

	// Write the opening brace
	if _, err := writer.WriteString("{\n"); err != nil {
		return fmt.Errorf("failed to write opening brace: %w", err)
	}

	// Write version
	if _, err := writer.WriteString(fmt.Sprintf("  \"version\": %d,\n", js.Version)); err != nil {
		return fmt.Errorf("failed to write version: %w", err)
	}

	// Lock the network for reading
	network.mu.RLock()

	// Write UUID and name
	if _, err := writer.WriteString(fmt.Sprintf("  \"uuid\": \"%s\",\n", network.UUID)); err != nil {
		network.mu.RUnlock()
		return fmt.Errorf("failed to write UUID: %w", err)
	}

	if _, err := writer.WriteString(fmt.Sprintf("  \"name\": \"%s\",\n", network.Name)); err != nil {
		network.mu.RUnlock()
		return fmt.Errorf("failed to write name: %w", err)
	}

	// Write layers opening
	if _, err := writer.WriteString("  \"layers\": {\n"); err != nil {
		network.mu.RUnlock()
		return fmt.Errorf("failed to write layers opening: %w", err)
	}

	// Get all layer UUIDs
	layerUUIDs := make([]string, 0, len(network.Layers))
	for uuid := range network.Layers {
		layerUUIDs = append(layerUUIDs, uuid)
	}

	layerCount := len(layerUUIDs)
	GlobalLogger.Info("Serializing %d layers", layerCount)

	// Write each layer
	for i, layerUUID := range layerUUIDs {
		layer := network.Layers[layerUUID]

		GlobalLogger.Progress("Serializing layers", i+1, layerCount, startTime)

		// Write layer opening
		if _, err := writer.WriteString(fmt.Sprintf("    \"%s\": {\n", layerUUID)); err != nil {
			network.mu.RUnlock()
			return fmt.Errorf("failed to write layer opening: %w", err)
		}

		// Write layer properties
		if _, err := writer.WriteString(fmt.Sprintf("      \"uuid\": \"%s\",\n", layer.UUID)); err != nil {
			network.mu.RUnlock()
			return fmt.Errorf("failed to write layer UUID: %w", err)
		}

		if _, err := writer.WriteString(fmt.Sprintf("      \"name\": \"%s\",\n", layer.Name)); err != nil {
			network.mu.RUnlock()
			return fmt.Errorf("failed to write layer name: %w", err)
		}

		if _, err := writer.WriteString(fmt.Sprintf("      \"width\": %d,\n", layer.Width)); err != nil {
			network.mu.RUnlock()
			return fmt.Errorf("failed to write layer width: %w", err)
		}

		if _, err := writer.WriteString(fmt.Sprintf("      \"height\": %d,\n", layer.Height)); err != nil {
			network.mu.RUnlock()
			return fmt.Errorf("failed to write layer height: %w", err)
		}

		// Write neurons opening
		if _, err := writer.WriteString("      \"neurons\": {\n"); err != nil {
			network.mu.RUnlock()
			return fmt.Errorf("failed to write neurons opening: %w", err)
		}

		// Get all neuron UUIDs for this layer
		neuronUUIDs := make([]string, 0, len(layer.Neurons))
		for uuid := range layer.Neurons {
			neuronUUIDs = append(neuronUUIDs, uuid)
		}

		neuronCount := len(neuronUUIDs)
		layerStartTime := time.Now()
		GlobalLogger.Info("Serializing %d neurons in layer '%s'", neuronCount, layer.Name)

		// Write each neuron
		// Only log progress for large neuron counts (>1000 neurons)
		logProgress := neuronCount > 1000
		for j, neuronUUID := range neuronUUIDs {
			if logProgress && (j+1)%500 == 0 {
				GlobalLogger.Progress("Serializing neurons", j+1, neuronCount, layerStartTime)
			}

			neuron := layer.Neurons[neuronUUID]

			// Lock the neuron for reading
			neuron.mu.RLock()

			// Write neuron opening
			if _, err := writer.WriteString(fmt.Sprintf("        \"%s\": {\n", neuronUUID)); err != nil {
				neuron.mu.RUnlock()
				network.mu.RUnlock()
				return fmt.Errorf("failed to write neuron opening: %w", err)
			}

			// Write neuron properties
			if _, err := writer.WriteString(fmt.Sprintf("          \"uuid\": \"%s\",\n", neuron.UUID)); err != nil {
				neuron.mu.RUnlock()
				network.mu.RUnlock()
				return fmt.Errorf("failed to write neuron UUID: %w", err)
			}

			if _, err := writer.WriteString(fmt.Sprintf("          \"resistance\": %f,\n", neuron.Resistance)); err != nil {
				neuron.mu.RUnlock()
				network.mu.RUnlock()
				return fmt.Errorf("failed to write neuron resistance: %w", err)
			}

			// Write vector
			if _, err := writer.WriteString("          \"vector\": ["); err != nil {
				neuron.mu.RUnlock()
				network.mu.RUnlock()
				return fmt.Errorf("failed to write vector opening: %w", err)
			}

			for k, dim := range neuron.Value.Dimensions {
				if k > 0 {
					if _, err := writer.WriteString(", "); err != nil {
						neuron.mu.RUnlock()
						network.mu.RUnlock()
						return fmt.Errorf("failed to write vector separator: %w", err)
					}
				}

				if _, err := writer.WriteString(fmt.Sprintf("%f", dim)); err != nil {
					neuron.mu.RUnlock()
					network.mu.RUnlock()
					return fmt.Errorf("failed to write vector dimension: %w", err)
				}
			}

			if _, err := writer.WriteString("],\n"); err != nil {
				neuron.mu.RUnlock()
				network.mu.RUnlock()
				return fmt.Errorf("failed to write vector closing: %w", err)
			}

			// Write connections opening
			if _, err := writer.WriteString("          \"connections\": {\n"); err != nil {
				neuron.mu.RUnlock()
				network.mu.RUnlock()
				return fmt.Errorf("failed to write connections opening: %w", err)
			}

			// Get all connection UUIDs
			connectionUUIDs := make([]string, 0, len(neuron.Connections))
			for uuid := range neuron.Connections {
				connectionUUIDs = append(connectionUUIDs, uuid)
			}

			// Write each connection
			for k, connectionUUID := range connectionUUIDs {
				strength := neuron.Connections[connectionUUID]

				if _, err := writer.WriteString(fmt.Sprintf("            \"%s\": %f", connectionUUID, strength)); err != nil {
					neuron.mu.RUnlock()
					network.mu.RUnlock()
					return fmt.Errorf("failed to write connection: %w", err)
				}

				if k < len(connectionUUIDs)-1 {
					if _, err := writer.WriteString(",\n"); err != nil {
						neuron.mu.RUnlock()
						network.mu.RUnlock()
						return fmt.Errorf("failed to write connection separator: %w", err)
					}
				} else {
					if _, err := writer.WriteString("\n"); err != nil {
						neuron.mu.RUnlock()
						network.mu.RUnlock()
						return fmt.Errorf("failed to write connection newline: %w", err)
					}
				}
			}

			// Write connections closing
			if _, err := writer.WriteString("          }\n"); err != nil {
				neuron.mu.RUnlock()
				network.mu.RUnlock()
				return fmt.Errorf("failed to write connections closing: %w", err)
			}

			// Write neuron closing
			if j < len(neuronUUIDs)-1 {
				if _, err := writer.WriteString("        },\n"); err != nil {
					neuron.mu.RUnlock()
					network.mu.RUnlock()
					return fmt.Errorf("failed to write neuron closing: %w", err)
				}
			} else {
				if _, err := writer.WriteString("        }\n"); err != nil {
					neuron.mu.RUnlock()
					network.mu.RUnlock()
					return fmt.Errorf("failed to write neuron closing: %w", err)
				}
			}

			// Unlock the neuron
			neuron.mu.RUnlock()
		}

		// Write neurons closing
		if _, err := writer.WriteString("      }\n"); err != nil {
			network.mu.RUnlock()
			return fmt.Errorf("failed to write neurons closing: %w", err)
		}

		// Write layer closing
		if i < len(layerUUIDs)-1 {
			if _, err := writer.WriteString("    },\n"); err != nil {
				network.mu.RUnlock()
				return fmt.Errorf("failed to write layer closing: %w", err)
			}
		} else {
			if _, err := writer.WriteString("    }\n"); err != nil {
				network.mu.RUnlock()
				return fmt.Errorf("failed to write layer closing: %w", err)
			}
		}
	}

	// Unlock the network
	network.mu.RUnlock()

	// Write layers closing
	if _, err := writer.WriteString("  }\n"); err != nil {
		return fmt.Errorf("failed to write layers closing: %w", err)
	}

	// Write the closing brace
	if _, err := writer.WriteString("}\n"); err != nil {
		return fmt.Errorf("failed to write closing brace: %w", err)
	}

	// Flush the buffer to ensure all data is written
	if err := writer.Flush(); err != nil {
		return fmt.Errorf("failed to flush buffer: %w", err)
	}

	// Close the file
	if err := file.Close(); err != nil {
		return fmt.Errorf("failed to close file: %w", err)
	}

	// Rename temporary file to target file (atomic operation)
	if err := os.Rename(tempFile, filePath); err != nil {
		// Try to clean up the temporary file
		os.Remove(tempFile)
		return fmt.Errorf("failed to rename temporary file to %s: %w", filePath, err)
	}

	GlobalLogger.Info("Network '%s' saved to %s in %s",
		network.Name, filePath, FormatDuration(time.Since(startTime)))
	return nil
}

// LoadNetworkFromJSON loads a network from a JSON file
func (js *JSONSerializer) LoadNetworkFromJSON(filePath string, seed int64) (*Network, error) {
	GlobalLogger.Info("Loading network from file %s", filePath)
	startTime := time.Now()

	// Open the JSON file
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open JSON file: %w", err)
	}
	defer file.Close()

	// Get file size for progress reporting
	fileInfo, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to get file info: %w", err)
	}
	fileSize := fileInfo.Size()

	GlobalLogger.Info("Reading network file (%d bytes)", fileSize)

	// Create a buffered reader for better performance
	reader := bufio.NewReader(file)

	// Read the entire file into memory
	// For very large files, we would use streaming JSON parsing
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON file: %w", err)
	}

	GlobalLogger.Info("Parsing JSON data (%d bytes read)", len(data))
	parseStartTime := time.Now()

	// Unmarshal JSON
	var networkJSON NetworkJSON
	if err := json.Unmarshal(data, &networkJSON); err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}

	GlobalLogger.Info("JSON parsed in %s", FormatDuration(time.Since(parseStartTime)))
	conversionStartTime := time.Now()

	// Convert JSON structure to network
	network, err := js.jsonToNetwork(networkJSON, seed)
	if err != nil {
		return nil, fmt.Errorf("failed to convert JSON to network: %w", err)
	}

	GlobalLogger.Info("Network '%s' loaded from %s in %s (parsing: %s, conversion: %s)",
		network.Name, filePath, FormatDuration(time.Since(startTime)),
		FormatDuration(time.Since(parseStartTime)-time.Since(conversionStartTime)),
		FormatDuration(time.Since(conversionStartTime)))
	return network, nil
}

// networkToJSON converts a network to its JSON representation
func (js *JSONSerializer) networkToJSON(network *Network) NetworkJSON {
	network.mu.RLock()
	defer network.mu.RUnlock()

	GlobalLogger.Debug("Converting network '%s' to JSON representation", network.Name)
	startTime := time.Now()

	networkJSON := NetworkJSON{
		Version: js.Version,
		UUID:    network.UUID,
		Name:    network.Name,
		Layers:  make(map[string]LayerJSON, len(network.Layers)),
	}

	// Convert each layer
	layerCount := len(network.Layers)
	layerIndex := 0

	for uuid, layer := range network.Layers {
		if layerIndex > 0 && layerIndex%10 == 0 {
			GlobalLogger.Progress("Converting layers to JSON", layerIndex+1, layerCount, startTime)
		}

		networkJSON.Layers[uuid] = js.layerToJSON(layer)
		layerIndex++
	}

	GlobalLogger.Debug("Converted network to JSON in %s", FormatDuration(time.Since(startTime)))
	return networkJSON
}

// layerToJSON converts a layer to its JSON representation
func (js *JSONSerializer) layerToJSON(layer *Layer) LayerJSON {
	layer.mu.RLock()
	defer layer.mu.RUnlock()

	GlobalLogger.Debug("Converting layer '%s' to JSON representation", layer.Name)
	startTime := time.Now()

	layerJSON := LayerJSON{
		UUID:    layer.UUID,
		Name:    layer.Name,
		Width:   layer.Width,
		Height:  layer.Height,
		Neurons: make(map[string]NeuronJSON, len(layer.Neurons)),
	}

	// Convert each neuron
	neuronCount := len(layer.Neurons)
	neuronIndex := 0

	// Only log progress for large neuron counts (>1000 neurons)
	logProgress := neuronCount > 1000
	for uuid, neuron := range layer.Neurons {
		if logProgress && (neuronIndex+1)%500 == 0 {
			GlobalLogger.Progress("Converting neurons to JSON", neuronIndex+1, neuronCount, startTime)
		}

		layerJSON.Neurons[uuid] = js.neuronToJSON(neuron)
		neuronIndex++
	}

	GlobalLogger.Debug("Converted layer '%s' to JSON in %s", layer.Name, FormatDuration(time.Since(startTime)))
	return layerJSON
}

// neuronToJSON converts a neuron to its JSON representation
func (js *JSONSerializer) neuronToJSON(neuron *Neuron) NeuronJSON {
	neuron.mu.RLock()
	defer neuron.mu.RUnlock()

	neuronJSON := NeuronJSON{
		UUID:        neuron.UUID,
		Resistance:  neuron.Resistance,
		Vector:      make([]float64, len(neuron.Value.Dimensions)),
		Connections: make(map[string]float64, len(neuron.Connections)),
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
	GlobalLogger.Info("Converting JSON to network '%s'", networkJSON.Name)
	startTime := time.Now()

	// Create a new network
	network := &Network{
		UUID:        networkJSON.UUID,
		Name:        networkJSON.Name,
		Layers:      make(map[string]*Layer, len(networkJSON.Layers)),
		NeuronCache: make(map[string]*Neuron),
		rand:        rand.New(rand.NewSource(seed)),
	}

	// Estimate neuron cache size for pre-allocation
	neuronCount := 0
	for _, layerJSON := range networkJSON.Layers {
		neuronCount += len(layerJSON.Neurons)
	}
	network.NeuronCache = make(map[string]*Neuron, neuronCount)

	GlobalLogger.Info("Network has %d layers and approximately %d neurons",
		len(networkJSON.Layers), neuronCount)

	// Convert each layer
	layerCount := len(networkJSON.Layers)
	layerIndex := 0

	for uuid, layerJSON := range networkJSON.Layers {
		GlobalLogger.Progress("Converting layers from JSON", layerIndex+1, layerCount, startTime)

		layer, err := js.jsonToLayer(layerJSON)
		if err != nil {
			return nil, fmt.Errorf("failed to convert layer %s: %w", uuid, err)
		}
		network.Layers[uuid] = layer

		// Add neurons to cache
		neuronCount := len(layer.Neurons)
		neuronIndex := 0
		layerStartTime := time.Now()

		GlobalLogger.Debug("Adding %d neurons from layer '%s' to cache", neuronCount, layer.Name)

		// Only log progress for large neuron counts (>1000 neurons)
		logProgress := neuronCount > 1000
		for neuronUUID, neuron := range layer.Neurons {
			if logProgress && (neuronIndex+1)%500 == 0 {
				GlobalLogger.Progress("Caching neurons", neuronIndex+1, neuronCount, layerStartTime)
			}

			network.NeuronCache[neuronUUID] = neuron
			neuronIndex++
		}

		GlobalLogger.Debug("Added %d neurons from layer '%s' to cache in %s",
			neuronCount, layer.Name, FormatDuration(time.Since(layerStartTime)))

		layerIndex++
	}

	GlobalLogger.Info("Converted JSON to network '%s' in %s",
		network.Name, FormatDuration(time.Since(startTime)))
	return network, nil
}

// jsonToLayer converts a JSON representation to a layer
func (js *JSONSerializer) jsonToLayer(layerJSON LayerJSON) (*Layer, error) {
	GlobalLogger.Debug("Converting JSON to layer '%s'", layerJSON.Name)
	startTime := time.Now()

	// Create a new layer
	layer := &Layer{
		UUID:    layerJSON.UUID,
		Name:    layerJSON.Name,
		Width:   layerJSON.Width,
		Height:  layerJSON.Height,
		Neurons: make(map[string]*Neuron, len(layerJSON.Neurons)),
	}

	// Convert each neuron
	neuronCount := len(layerJSON.Neurons)
	neuronIndex := 0

	// Only log progress for large neuron counts (>1000 neurons)
	logProgress := neuronCount > 1000
	for uuid, neuronJSON := range layerJSON.Neurons {
		if logProgress && (neuronIndex+1)%500 == 0 {
			GlobalLogger.Progress("Converting neurons from JSON", neuronIndex+1, neuronCount, startTime)
		}

		neuron, err := js.jsonToNeuron(neuronJSON)
		if err != nil {
			return nil, fmt.Errorf("failed to convert neuron %s: %w", uuid, err)
		}
		layer.Neurons[uuid] = neuron
		neuronIndex++
	}

	GlobalLogger.Debug("Converted JSON to layer '%s' with %d neurons in %s",
		layerJSON.Name, neuronCount, FormatDuration(time.Since(startTime)))
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
		Connections: make(map[string]float64, len(neuronJSON.Connections)),
	}

	// Copy connections
	for uuid, strength := range neuronJSON.Connections {
		neuron.Connections[uuid] = strength
	}

	return neuron, nil
}

// ==================== TIMING AND PROGRESS TRACKING ====================

// ProgressTracker tracks progress and timing information for network creation
type ProgressTracker struct {
	StartTime                 time.Time
	LayerCreationTimes        []time.Duration
	NeuronConnectionTimes     []time.Duration
	LayerConnectionTimes      []time.Duration
	TotalLayers               int
	CompletedLayers           int
	TotalLayerConnections     int
	CompletedLayerConnections int
	mu                        sync.Mutex // Added mutex for thread safety
}

// NewProgressTracker creates a new progress tracker
func NewProgressTracker(totalLayers int) *ProgressTracker {
	totalConnections := totalLayers * (totalLayers - 1)
	return &ProgressTracker{
		StartTime:             time.Now(),
		LayerCreationTimes:    make([]time.Duration, 0, totalLayers),
		NeuronConnectionTimes: make([]time.Duration, 0, totalLayers),
		LayerConnectionTimes:  make([]time.Duration, 0, totalConnections),
		TotalLayers:           totalLayers,
		TotalLayerConnections: totalConnections,
	}
}

// RecordLayerCreation records the time taken to create a layer
func (pt *ProgressTracker) RecordLayerCreation(duration time.Duration) {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	pt.LayerCreationTimes = append(pt.LayerCreationTimes, duration)
	pt.CompletedLayers++

	GlobalLogger.Info("Layer creation recorded: %s (avg: %s)",
		FormatDuration(duration), FormatDuration(pt.GetAverageLayerCreationTime()))
}

// RecordNeuronConnections records the time taken to connect neurons within a layer
func (pt *ProgressTracker) RecordNeuronConnections(duration time.Duration) {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	pt.NeuronConnectionTimes = append(pt.NeuronConnectionTimes, duration)

	GlobalLogger.Info("Neuron connections recorded: %s (avg: %s)",
		FormatDuration(duration), FormatDuration(pt.GetAverageNeuronConnectionTime()))
}

// RecordLayerConnection records the time taken to connect two layers
func (pt *ProgressTracker) RecordLayerConnection(duration time.Duration) {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	pt.LayerConnectionTimes = append(pt.LayerConnectionTimes, duration)
	pt.CompletedLayerConnections++

	GlobalLogger.Info("Layer connection recorded: %s (avg: %s, %d/%d complete)",
		FormatDuration(duration), FormatDuration(pt.GetAverageLayerConnectionTime()),
		pt.CompletedLayerConnections, pt.TotalLayerConnections)
}

// GetAverageLayerCreationTime returns the average time to create a layer
func (pt *ProgressTracker) GetAverageLayerCreationTime() time.Duration {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	if len(pt.LayerCreationTimes) == 0 {
		return 0
	}

	var total time.Duration
	for _, t := range pt.LayerCreationTimes {
		total += t
	}
	return total / time.Duration(len(pt.LayerCreationTimes))
}

// GetAverageNeuronConnectionTime returns the average time to connect neurons within a layer
func (pt *ProgressTracker) GetAverageNeuronConnectionTime() time.Duration {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	if len(pt.NeuronConnectionTimes) == 0 {
		return 0
	}

	var total time.Duration
	for _, t := range pt.NeuronConnectionTimes {
		total += t
	}
	return total / time.Duration(len(pt.NeuronConnectionTimes))
}

// GetAverageLayerConnectionTime returns the average time to connect two layers
func (pt *ProgressTracker) GetAverageLayerConnectionTime() time.Duration {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	if len(pt.LayerConnectionTimes) == 0 {
		return 0
	}

	var total time.Duration
	for _, t := range pt.LayerConnectionTimes {
		total += t
	}
	return total / time.Duration(len(pt.LayerConnectionTimes))
}

// GetTotalTime returns the total time elapsed since tracking started
func (pt *ProgressTracker) GetTotalTime() time.Duration {
	return time.Since(pt.StartTime)
}

// GetEstimatedTimeRemaining returns the estimated time remaining for the operation
func (pt *ProgressTracker) GetEstimatedTimeRemaining() time.Duration {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	if pt.CompletedLayers == 0 {
		return 0 // Can't estimate yet
	}

	elapsed := time.Since(pt.StartTime)

	// Calculate progress as a percentage
	layerProgress := float64(pt.CompletedLayers) / float64(pt.TotalLayers)

	// If we're still creating layers
	if layerProgress < 1.0 {
		// Estimate based on layer creation progress
		return time.Duration(float64(elapsed) * (1.0/layerProgress - 1.0))
	}

	// If we're connecting layers
	if pt.CompletedLayerConnections < pt.TotalLayerConnections {
		connectionProgress := float64(pt.CompletedLayerConnections) / float64(pt.TotalLayerConnections)
		// We're 50% done with layer creation, 50% with connections
		totalProgress := 0.5 + 0.5*connectionProgress
		return time.Duration(float64(elapsed) * (1.0/totalProgress - 1.0))
	}

	return 0 // All done
}

// FormatDuration formats a duration in a human-readable format
func FormatDuration(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%d ms", d.Milliseconds())
	} else if d < time.Minute {
		return fmt.Sprintf("%.2f sec", d.Seconds())
	} else {
		minutes := int(d.Minutes())
		seconds := int(d.Seconds()) % 60
		return fmt.Sprintf("%d min %d sec", minutes, seconds)
	}
}

// ==================== CLI IMPLEMENTATION ====================

func main() {
	// Define global flags
	verbosityPtr := flag.Int("v", 2, "Verbosity level (0=none, 1=error, 2=info, 3=debug, 4=trace)")
	logFilePtr := flag.String("log", "", "Log file path (default: stdout)")

	// Check if any arguments were provided
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	// Create subcommands
	createCmd := flag.NewFlagSet("create", flag.ExitOnError)
	loadCmd := flag.NewFlagSet("load", flag.ExitOnError)

	// Define flags for load command
	loadFilePath := loadCmd.String("file", "", "Path to the network file to load")
	loadVerbose := loadCmd.Bool("verbose", false, "Display detailed network information")

	// Find the command position
	cmdPos := 1
	for i, arg := range os.Args {
		if arg == "create" || arg == "load" {
			cmdPos = i
			break
		}
	}

	// Parse global flags before the command
	if cmdPos > 1 {
		globalFlags := flag.NewFlagSet("global", flag.ExitOnError)
		globalFlags.IntVar(verbosityPtr, "v", 2, "Verbosity level (0=none, 1=error, 2=info, 3=debug, 4=trace)")
		globalFlags.StringVar(logFilePtr, "log", "", "Log file path (default: stdout)")
		globalFlags.Parse(os.Args[1:cmdPos])
	}

	// Setup logging
	setupLogging(*verbosityPtr, *logFilePtr)

	// Parse command
	switch os.Args[cmdPos] {
	case "create":
		createCmd.Parse(os.Args[cmdPos+1:])
		runCreateCommand()
	case "load":
		loadCmd.Parse(os.Args[cmdPos+1:])
		runLoadCommand(*loadFilePath, *loadVerbose)
	default:
		printUsage()
		os.Exit(1)
	}
}

// setupLogging configures the global logger
func setupLogging(verbosity int, logFilePath string) {
	var logLevel LogLevel
	switch verbosity {
	case 0:
		logLevel = LogLevelNone
	case 1:
		logLevel = LogLevelError
	case 2:
		logLevel = LogLevelInfo
	case 3:
		logLevel = LogLevelDebug
	case 4:
		logLevel = LogLevelTrace
	default:
		logLevel = LogLevelInfo
	}

	var writer io.Writer = os.Stdout

	if logFilePath != "" {
		file, err := os.Create(logFilePath)
		if err != nil {
			log.Printf("Failed to create log file: %v, using stdout instead", err)
		} else {
			writer = file
			// Note: We're not closing the file as it will be used throughout the program
		}
	}

	GlobalLogger = NewLogger(logLevel, writer, true)
	// Set a longer progress interval for less frequent updates
	GlobalLogger.SetProgressInterval(2 * time.Second)
	GlobalLogger.Info("Logging initialized at level %d", verbosity)
}

// printUsage prints the usage information for the CLI
func printUsage() {
	fmt.Println("Neural Network CLI Tool")
	fmt.Println("\nUsage:")
	fmt.Println("  neuralnet [global options] [command] [command options]")
	fmt.Println("\nGlobal Options:")
	fmt.Println("  -v level     Verbosity level (0=none, 1=error, 2=info, 3=debug, 4=trace)")
	fmt.Println("  -log file    Log file path (default: stdout)")
	fmt.Println("\nCommands:")
	fmt.Println("  create       Create a new neural network interactively")
	fmt.Println("  load         Load an existing neural network")
	fmt.Println("\nOptions for 'load':")
	fmt.Println("  -file        Path to the network file to load (required)")
	fmt.Println("  -verbose     Display detailed network information")
	fmt.Println("\nExamples:")
	fmt.Println("  neuralnet -v 3 create")
	fmt.Println("  neuralnet -log network.log load -file ./networks/net96.json")
}

// runCreateCommand handles the interactive creation of a new network
func runCreateCommand() {
	fmt.Println("\nNeural Network Creation Tool")
	fmt.Println("----------------------------")

	// Collect network configuration through interactive prompts
	var networkName string
	var err error

	// Get and validate network name
	for {
		networkName = promptString("Network name?", "")
		err = validateNetworkName(networkName)
		if err == nil {
			break
		}
		fmt.Printf("Error: %v\n", err)
	}

	// Get and validate layer count
	var layerCount int
	for {
		layerCount = promptInt("Number of layers?", 1)
		err = validateLayerCount(layerCount)
		if err == nil {
			break
		}
		fmt.Printf("Error: %v\n", err)
	}

	// Get and validate vector dimensions
	var vectorDims int
	for {
		vectorDims = promptInt("Vector dimensions for neurons?", 64)
		err = validateVectorDimensions(vectorDims)
		if err == nil {
			break
		}
		fmt.Printf("Error: %v\n", err)
	}

	// Collect neuron counts for each layer
	neuronCounts := make([]int, layerCount)
	for i := 0; i < layerCount; i++ {
		defaultValue := getDefaultNeuronCount(i, neuronCounts)

		promptText := fmt.Sprintf("Number of neurons for layer %d?", i+1)
		if i > 0 {
			promptText = fmt.Sprintf("Number of neurons for layer %d? (enter for %d)", i+1, defaultValue)
		}

		for {
			neuronCounts[i] = promptInt(promptText, defaultValue)
			err = validateNeuronCount(neuronCounts[i], i+1)
			if err == nil {
				break
			}
			fmt.Printf("Error: %v\n", err)
		}
	}

	// Get random seed
	seed := promptInt64("Random seed (optional, press Enter for time-based seed)?", time.Now().UnixNano())

	// Get and validate output file path
	var filePath string
	defaultPath := "./" + networkName
	for {
		filePath = promptString("Output file path?", defaultPath+".json")
		err = validateFilePath(filePath)
		if err == nil {
			break
		}
		fmt.Printf("Error: %v\n", err)
	}

	// Display network configuration summary and ask for confirmation
	fmt.Println("\nNetwork Configuration Summary:")
	fmt.Printf("  Name: %s\n", networkName)
	fmt.Printf("  Layers: %d\n", layerCount)
	fmt.Printf("  Vector dimensions: %d\n", vectorDims)
	fmt.Println("  Neuron counts per layer:")
	for i := 0; i < layerCount; i++ {
		fmt.Printf("    Layer %d: %d neurons\n", i+1, neuronCounts[i])
	}
	fmt.Printf("  Random seed: %d\n", seed)
	fmt.Printf("  Output file: %s\n", filePath)

	// Ask for confirmation
	confirmed := promptYesNo("\nProceed with network creation?", true)
	if !confirmed {
		fmt.Println("Network creation cancelled.")
		return
	}

	// Create the network
	fmt.Println("\nCreating network...")
	network, err := createNetwork(networkName, layerCount, vectorDims, neuronCounts, seed)
	if err != nil {
		fmt.Printf("Error creating network: %v\n", err)
		os.Exit(1)
	}

	// Save the network to a file
	fmt.Printf("Saving network to %s...\n", filePath)
	serializer := NewJSONSerializer()
	err = serializer.SaveNetworkToJSON(network, filePath)
	if err != nil {
		fmt.Printf("Error saving network: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nNetwork created and saved successfully to %s\n", filePath)

	// Ensure all output is flushed
	os.Stdout.Sync()

	// Exit explicitly to avoid any potential hanging
	os.Exit(0)
}

// runLoadCommand handles loading a network from a file
func runLoadCommand(filePath string, verbose bool) {
	if filePath == "" {
		fmt.Println("Error: File path is required")
		fmt.Println("Usage: neuralnet load -file <path>")
		os.Exit(1)
	}

	fmt.Printf("Loading network from %s...\n", filePath)
	serializer := NewJSONSerializer()
	network, err := serializer.LoadNetworkFromJSON(filePath, time.Now().UnixNano())
	if err != nil {
		fmt.Printf("Error loading network: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nNetwork loaded successfully: %s\n", network.Name)
	fmt.Printf("UUID: %s\n", network.UUID)
	fmt.Printf("Layers: %d\n", len(network.Layers))
	fmt.Printf("Total neurons: %d\n", len(network.NeuronCache))

	if verbose {
		fmt.Println("\nLayer details:")
		for _, layer := range network.Layers {
			fmt.Printf("  - %s (UUID: %s)\n", layer.Name, layer.UUID)
			fmt.Printf("    Dimensions: %d x %d (%d neurons)\n", layer.Width, layer.Height, len(layer.Neurons))

			// Count connections
			totalConnections := 0
			for _, neuron := range layer.Neurons {
				totalConnections += len(neuron.Connections)
			}
			fmt.Printf("    Connections: %d (avg: %.1f per neuron)\n",
				totalConnections, float64(totalConnections)/float64(len(layer.Neurons)))
		}
	}
}

// createNetwork creates a new network with the specified parameters
func createNetwork(name string, layerCount, vectorDims int, neuronCounts []int, seed int64) (*Network, error) {
	// Create a progress tracker
	tracker := NewProgressTracker(layerCount)

	// Create a new network
	network, err := NewNetwork(name, seed)
	if err != nil {
		return nil, fmt.Errorf("failed to create network: %w", err)
	}

	// Create layers
	for i := 0; i < layerCount; i++ {
		layerName := fmt.Sprintf("Layer%d", i+1)

		// Calculate dimensions for the layer
		width, height := calculateLayerDimensions(neuronCounts[i])

		fmt.Printf("Creating layer %d/%d (%s): %d neurons (%dx%d)...\n",
			i+1, layerCount, layerName, neuronCounts[i], width, height)

		startTime := time.Now()

		// Create the layer
		layer, err := NewLayerWithUniqueVectors(layerName, width, height, vectorDims, seed+int64(i*1000))
		if err != nil {
			return nil, fmt.Errorf("failed to create layer %s: %w", layerName, err)
		}

		// Add the layer to the network
		err = network.AddLayer(layer)
		if err != nil {
			return nil, fmt.Errorf("failed to add layer %s to network: %w", layerName, err)
		}

		duration := time.Since(startTime)
		tracker.RecordLayerCreation(duration)

		fmt.Printf("  Layer created in %s\n", FormatDuration(duration))
		fmt.Printf("  Estimated time remaining: %s\n", FormatDuration(tracker.GetEstimatedTimeRemaining()))
	}

	// Connect all layers
	fmt.Println("\nConnecting layers...")
	startTime := time.Now()
	err = network.ConnectAllLayers()
	if err != nil {
		return nil, fmt.Errorf("failed to connect layers: %w", err)
	}
	duration := time.Since(startTime)

	fmt.Printf("All layers connected in %s\n", FormatDuration(duration))
	fmt.Printf("Total creation time: %s\n", FormatDuration(tracker.GetTotalTime()))

	return network, nil
}

// calculateLayerDimensions calculates the width and height for a layer based on neuron count
func calculateLayerDimensions(neuronCount int) (width, height int) {
	// Try to make the layer as square as possible
	width = int(math.Sqrt(float64(neuronCount)))
	height = (neuronCount + width - 1) / width // Ceiling division
	return
}

// getDefaultNeuronCount returns a default neuron count for a layer
func getDefaultNeuronCount(layerIndex int, existingCounts []int) int {
	if layerIndex == 0 {
		return 64 // Default for first layer
	}
	return existingCounts[layerIndex-1] // Use previous layer's count as default
}

// validateNetworkName validates the network name
func validateNetworkName(name string) error {
	if name == "" {
		return fmt.Errorf("network name cannot be empty")
	}
	return nil
}

// validateLayerCount validates the layer count
func validateLayerCount(count int) error {
	if count <= 0 {
		return fmt.Errorf("layer count must be positive")
	}
	if count > 100 {
		return fmt.Errorf("layer count cannot exceed 100")
	}
	return nil
}

// validateVectorDimensions validates the vector dimensions
func validateVectorDimensions(dims int) error {
	if dims <= 0 {
		return fmt.Errorf("vector dimensions must be positive")
	}
	if dims > 1024 {
		return fmt.Errorf("vector dimensions cannot exceed 1024")
	}
	return nil
}

// validateNeuronCount validates the neuron count for a layer
func validateNeuronCount(count, layerIndex int) error {
	if count <= 0 {
		return fmt.Errorf("neuron count must be positive")
	}
	if count > 10000 {
		return fmt.Errorf("neuron count cannot exceed 10000")
	}
	return nil
}

// validateFilePath validates the output file path
func validateFilePath(path string) error {
	if path == "" {
		return fmt.Errorf("file path cannot be empty")
	}

	// Check if the directory exists
	dir := filepath.Dir(path)
	if dir != "." {
		if _, err := os.Stat(dir); os.IsNotExist(err) {
			// Try to create the directory
			if err := os.MkdirAll(dir, 0755); err != nil {
				return fmt.Errorf("directory does not exist and could not be created: %s", dir)
			}
		}
	}

	// Check if the file has a .json extension
	if !strings.HasSuffix(path, ".json") {
		return fmt.Errorf("file must have a .json extension")
	}

	return nil
}

// promptString prompts the user for a string input
func promptString(prompt, defaultValue string) string {
	reader := bufio.NewReader(os.Stdin)

	if defaultValue != "" {
		fmt.Printf("%s [%s]: ", prompt, defaultValue)
	} else {
		fmt.Printf("%s: ", prompt)
	}

	input, err := reader.ReadString('\n')
	if err != nil {
		fmt.Printf("Error reading input: %v\n", err)
		return defaultValue
	}

	// Trim whitespace and newline
	input = strings.TrimSpace(input)

	if input == "" {
		return defaultValue
	}

	return input
}

// promptInt prompts the user for an integer input
func promptInt(prompt string, defaultValue int) int {
	reader := bufio.NewReader(os.Stdin)

	if defaultValue != 0 {
		fmt.Printf("%s [%d]: ", prompt, defaultValue)
	} else {
		fmt.Printf("%s: ", prompt)
	}

	input, err := reader.ReadString('\n')
	if err != nil {
		fmt.Printf("Error reading input: %v\n", err)
		return defaultValue
	}

	// Trim whitespace and newline
	input = strings.TrimSpace(input)

	if input == "" {
		return defaultValue
	}

	// Parse the input as an integer
	value, err := strconv.Atoi(input)
	if err != nil {
		fmt.Printf("Invalid integer: %v\n", err)
		return defaultValue
	}

	return value
}

// promptInt64 prompts the user for an int64 input
func promptInt64(prompt string, defaultValue int64) int64 {
	reader := bufio.NewReader(os.Stdin)

	if defaultValue != 0 {
		fmt.Printf("%s [%d]: ", prompt, defaultValue)
	} else {
		fmt.Printf("%s: ", prompt)
	}

	input, err := reader.ReadString('\n')
	if err != nil {
		fmt.Printf("Error reading input: %v\n", err)
		return defaultValue
	}

	// Trim whitespace and newline
	input = strings.TrimSpace(input)

	if input == "" {
		return defaultValue
	}

	// Parse the input as an int64
	value, err := strconv.ParseInt(input, 10, 64)
	if err != nil {
		fmt.Printf("Invalid integer: %v\n", err)
		return defaultValue
	}

	return value
}

// promptYesNo prompts the user for a yes/no answer
func promptYesNo(prompt string, defaultValue bool) bool {
	reader := bufio.NewReader(os.Stdin)

	defaultStr := "y"
	if !defaultValue {
		defaultStr = "n"
	}

	fmt.Printf("%s [%s]: ", prompt, defaultStr)

	input, err := reader.ReadString('\n')
	if err != nil {
		fmt.Printf("Error reading input: %v\n", err)
		return defaultValue
	}

	// Trim whitespace and newline
	input = strings.TrimSpace(input)

	if input == "" {
		return defaultValue
	}

	// Check for yes/no variations
	input = strings.ToLower(input)
	if input == "y" || input == "yes" || input == "true" || input == "1" {
		return true
	}
	if input == "n" || input == "no" || input == "false" || input == "0" {
		return false
	}

	// Invalid input, use default
	fmt.Printf("Invalid input, using default: %v\n", defaultValue)
	return defaultValue
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
