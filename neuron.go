package main

import (
	"fmt"
	"math/rand"
	"sync"

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
