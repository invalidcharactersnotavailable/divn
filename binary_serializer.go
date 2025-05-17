package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

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
