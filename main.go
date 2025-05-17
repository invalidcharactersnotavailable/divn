package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"time"
)

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
	FullConnect  bool // Flag for full layer interconnection
	UseBinary    bool // Flag for using binary serialization
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

func main() {
	// Check if we have enough arguments
	if len(os.Args) < 2 {
		printUsageUpdated()
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
		VectorDims:   64, // Increased for text encoding
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
		printUsageUpdated()
	default:
		fmt.Printf("Unknown command: %s\n", command)
		printUsageUpdated()
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

// printUsageUpdated prints the updated usage information
func printUsageUpdated() {
	fmt.Println("Usage: network <command> [options]")
	fmt.Println("\nCommands:")
	fmt.Println("  create                Create a new network")
	fmt.Println("  load                  Load a network from file")
	fmt.Println("  test                  Test network data processing")
	fmt.Println("  benchmark             Run performance benchmarks")
	fmt.Println("  benchmark-serialization  Benchmark serialization formats")
	fmt.Println("  help                  Show this help message")
	fmt.Println("\nOptions:")
	flag.PrintDefaults()
}

// createNetwork creates a new network and saves it to a file
func createNetwork(flags CommandFlags) {
	fmt.Printf("Creating network with seed %d...\n", flags.Seed)

	// Create a new network
	network, err := NewNetwork("TestNetwork", flags.Seed)
	if err != nil {
		fmt.Printf("Error creating network: %v\n", err)
		os.Exit(1)
	}

	// Create layers
	for i := 0; i < flags.LayerCount; i++ {
		layerName := fmt.Sprintf("Layer%d", i+1)
		layer, err := NewLayerWithUniqueVectors(layerName, flags.NeuronSize, flags.NeuronSize, flags.VectorDims, flags.Seed+int64(i))
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

		// Connect neurons within the layer
		err = layer.ConnectInternalNeurons(network.rand)
		if err != nil {
			fmt.Printf("Error connecting neurons in layer %s: %v\n", layerName, err)
			os.Exit(1)
		}
	}

	// Connect layers
	if flags.FullConnect {
		fmt.Println("Connecting all layers...")
		err = network.ConnectAllLayers()
		if err != nil {
			fmt.Printf("Error connecting layers: %v\n", err)
			os.Exit(1)
		}
	}

	// Save network
	fmt.Println("Saving network...")
	if flags.UseBinary {
		// Use binary serialization
		serializer := NewBinarySerializer()
		err = serializer.SaveNetworkToBinary(network, flags.FilePath+".bin")
		if err != nil {
			fmt.Printf("Error saving network to binary: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Network saved to %s.bin\n", flags.FilePath)
	} else {
		// Use JSON serialization
		err = network.SaveToFile(flags.FilePath + ".json")
		if err != nil {
			fmt.Printf("Error saving network to JSON: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Network saved to %s.json\n", flags.FilePath)
	}
}

// loadNetwork loads a network from a file
func loadNetwork(flags CommandFlags) {
	fmt.Printf("Loading network with seed %d...\n", flags.Seed)

	var network *Network
	var err error

	if flags.UseBinary {
		// Use binary deserialization
		serializer := NewBinarySerializer()
		network, err = serializer.LoadNetworkFromBinary(flags.FilePath+".bin", flags.Seed)
		if err != nil {
			fmt.Printf("Error loading network from binary: %v\n", err)
			os.Exit(1)
		}
	} else {
		// Use JSON deserialization
		network, err = LoadNetworkFromFile(flags.FilePath+".json", flags.Seed)
		if err != nil {
			fmt.Printf("Error loading network from JSON: %v\n", err)
			os.Exit(1)
		}
	}

	// Print network information
	fmt.Printf("Network loaded: %s (UUID: %s)\n", network.Name, network.UUID)
	fmt.Printf("Layers: %d\n", len(network.Layers))
	fmt.Printf("Neurons: %d\n", len(network.NeuronCache))

	// Print layer information
	for _, layer := range network.Layers {
		fmt.Printf("Layer: %s (UUID: %s)\n", layer.Name, layer.UUID)
		fmt.Printf("  Dimensions: %d x %d\n", layer.Width, layer.Height)
		fmt.Printf("  Neurons: %d\n", len(layer.Neurons))
	}
}

// testNetwork tests network data processing
func testNetwork(flags CommandFlags) {
	fmt.Printf("Testing network with seed %d...\n", flags.Seed)

	var network *Network
	var err error

	if flags.UseBinary {
		// Use binary deserialization
		serializer := NewBinarySerializer()
		network, err = serializer.LoadNetworkFromBinary(flags.FilePath+".bin", flags.Seed)
		if err != nil {
			fmt.Printf("Error loading network from binary: %v\n", err)
			os.Exit(1)
		}
	} else {
		// Use JSON deserialization
		network, err = LoadNetworkFromFile(flags.FilePath+".json", flags.Seed)
		if err != nil {
			fmt.Printf("Error loading network from JSON: %v\n", err)
			os.Exit(1)
		}
	}

	// Get the first layer and neuron for testing
	var firstLayerUUID, firstNeuronUUID string
	for layerUUID, layer := range network.Layers {
		firstLayerUUID = layerUUID
		for neuronUUID := range layer.Neurons {
			firstNeuronUUID = neuronUUID
			break
		}
		break
	}

	if firstLayerUUID == "" || firstNeuronUUID == "" {
		fmt.Println("Error: No layers or neurons found in network")
		os.Exit(1)
	}

	// Create test data
	testData := make([]float64, flags.VectorDims)
	for i := range testData {
		testData[i] = float64(i) / float64(flags.VectorDims)
	}

	fmt.Printf("Processing data through network starting from layer %s, neuron %s...\n",
		network.Layers[firstLayerUUID].Name, firstNeuronUUID)

	// Process data
	result, path, err := network.ProcessData(firstLayerUUID, firstNeuronUUID, testData, 10)
	if err != nil {
		fmt.Printf("Error processing data: %v\n", err)
		os.Exit(1)
	}

	// Print results
	fmt.Println("Input data:", testData)
	fmt.Println("Output data:", result)
	fmt.Println("Path length:", len(path))
	fmt.Println("Path:", path)
}

// benchmarkNetwork runs performance benchmarks
func benchmarkNetwork(flags CommandFlags) {
	fmt.Printf("Benchmarking network with seed %d...\n", flags.Seed)

	var network *Network
	var err error

	if flags.UseBinary {
		// Use binary deserialization
		serializer := NewBinarySerializer()
		network, err = serializer.LoadNetworkFromBinary(flags.FilePath+".bin", flags.Seed)
		if err != nil {
			fmt.Printf("Error loading network from binary: %v\n", err)
			os.Exit(1)
		}
	} else {
		// Use JSON deserialization
		network, err = LoadNetworkFromFile(flags.FilePath+".json", flags.Seed)
		if err != nil {
			fmt.Printf("Error loading network from JSON: %v\n", err)
			os.Exit(1)
		}
	}

	// Get the first layer and neuron for testing
	var firstLayerUUID, firstNeuronUUID string
	for layerUUID, layer := range network.Layers {
		firstLayerUUID = layerUUID
		for neuronUUID := range layer.Neurons {
			firstNeuronUUID = neuronUUID
			break
		}
		break
	}

	if firstLayerUUID == "" || firstNeuronUUID == "" {
		fmt.Println("Error: No layers or neurons found in network")
		os.Exit(1)
	}

	// Create test data
	testData := make([]float64, flags.VectorDims)
	for i := range testData {
		testData[i] = float64(i) / float64(flags.VectorDims)
	}

	// Benchmark sequential processing
	fmt.Println("\nBenchmarking sequential processing...")
	iterations := 1000
	sequentialStart := time.Now()

	for i := 0; i < iterations; i++ {
		_, _, err := network.ProcessData(firstLayerUUID, firstNeuronUUID, testData, 10)
		if err != nil {
			fmt.Printf("Error in sequential processing: %v\n", err)
			os.Exit(1)
		}
	}

	sequentialTime := time.Since(sequentialStart)
	fmt.Printf("Sequential processing time for %d iterations: %v\n", iterations, sequentialTime)
	fmt.Printf("Average time per iteration: %v\n", sequentialTime/time.Duration(iterations))

	// Benchmark parallel processing
	if flags.ParallelProc {
		fmt.Println("\nBenchmarking parallel processing...")
		batchSize := 1000
		dataInputs := make([][]float64, batchSize)
		for i := range dataInputs {
			dataInputs[i] = make([]float64, flags.VectorDims)
			for j := range dataInputs[i] {
				dataInputs[i][j] = float64(j) / float64(flags.VectorDims)
			}
		}

		parallelStart := time.Now()

		_, _, errors := network.ProcessDataParallel(firstLayerUUID, firstNeuronUUID, dataInputs, 10)
		for _, err := range errors {
			if err != nil {
				fmt.Printf("Error in parallel processing: %v\n", err)
				os.Exit(1)
			}
		}

		parallelTime := time.Since(parallelStart)
		fmt.Printf("Parallel processing time for %d inputs: %v\n", batchSize, parallelTime)
		fmt.Printf("Average time per input: %v\n", parallelTime/time.Duration(batchSize))

		// Calculate speedup
		if batchSize == iterations {
			speedup := float64(sequentialTime) / float64(parallelTime)
			fmt.Printf("Parallel speedup: %.2fx\n", speedup)
		}
	}
}

// benchmarkSerialization benchmarks serialization formats
func benchmarkSerialization(flags CommandFlags) {
	fmt.Printf("Benchmarking serialization with seed %d...\n", flags.Seed)

	// Create a network for benchmarking
	network, err := NewNetwork("BenchmarkNetwork", flags.Seed)
	if err != nil {
		fmt.Printf("Error creating network: %v\n", err)
		os.Exit(1)
	}

	// Create layers
	for i := 0; i < flags.LayerCount; i++ {
		layerName := fmt.Sprintf("Layer%d", i+1)
		layer, err := NewLayerWithUniqueVectors(layerName, flags.NeuronSize, flags.NeuronSize, flags.VectorDims, flags.Seed+int64(i))
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

		// Connect neurons within the layer
		err = layer.ConnectInternalNeurons(network.rand)
		if err != nil {
			fmt.Printf("Error connecting neurons in layer %s: %v\n", layerName, err)
			os.Exit(1)
		}
	}

	// Connect layers
	if flags.FullConnect {
		fmt.Println("Connecting all layers...")
		err = network.ConnectAllLayers()
		if err != nil {
			fmt.Printf("Error connecting layers: %v\n", err)
			os.Exit(1)
		}
	}

	// Run serialization benchmark
	BinarySerializationBenchmark(network, flags.FilePath, flags.Seed)
}
