package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

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
