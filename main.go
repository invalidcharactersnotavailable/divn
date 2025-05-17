package main

import (
	"flag"
	"fmt"
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
	switch command {
	case "create":
		createNetwork(flags)
	case "save":
		saveNetwork(flags)
	case "load":
		loadNetwork(flags)
	case "process":
		processData(flags)
	case "benchmark":
		runBenchmark(flags)
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
	fmt.Println("Usage: divn <command> [options]")
	fmt.Println("\nCommands:")
	fmt.Println("  create     Create a new network")
	fmt.Println("  save       Save an existing network")
	fmt.Println("  load       Load a network from file")
	fmt.Println("  process    Process data through the network")
	fmt.Println("  benchmark  Run performance benchmarks")
	fmt.Println("\nOptions:")
	flag.PrintDefaults()
}

// createNetwork creates a new network with the specified parameters
func createNetwork(flags CommandFlags) {
	fmt.Printf("Creating network with seed %d...\n", flags.Seed)

	// Create a new network
	network, err := NewNetwork("Dynamic Routing Network", flags.Seed)
	if err != nil {
		fmt.Printf("Error creating network: %v\n", err)
		os.Exit(1)
	}

	// Create layers
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

	// Connect layers - always use full connections
	fmt.Println("Connecting all layers...")
	err = network.ConnectAllLayers()
	if err != nil {
		fmt.Printf("Error connecting layers: %v\n", err)
		os.Exit(1)
	}

	// Save the network
	fmt.Printf("Saving network to %s.bin...\n", flags.FilePath)

	// Save using binary serialization
	err = network.SaveToFile(flags.FilePath + ".bin")
	if err != nil {
		fmt.Printf("Error saving network: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Network saved to %s.bin\n", flags.FilePath)
}

// saveNetwork saves an existing network to a file
func saveNetwork(flags CommandFlags) {
	fmt.Printf("Loading network with seed %d...\n", flags.Seed)

	// Load the network
	network, err := LoadNetworkFromFile(flags.FilePath+".bin", flags.Seed)
	if err != nil {
		fmt.Printf("Error loading network: %v\n", err)
		os.Exit(1)
	}

	// Save the network
	fmt.Printf("Saving network to %s.bin...\n", flags.FilePath)

	// Use binary serialization
	serializer := NewBinarySerializer()
	err = serializer.SaveNetworkToBinary(network, flags.FilePath+".bin")
	if err != nil {
		fmt.Printf("Error saving network to binary: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Network saved to %s.bin\n", flags.FilePath)
}

// loadNetwork loads a network from a file
func loadNetwork(flags CommandFlags) {
	fmt.Printf("Loading network with seed %d...\n", flags.Seed)

	var network *Network
	var err error

	// Load the network using binary serialization
	network, err = LoadNetworkFromFile(flags.FilePath+".bin", flags.Seed)
	if err != nil {
		fmt.Printf("Error loading network: %v\n", err)
		os.Exit(1)
	}

	// Print network information
	fmt.Printf("Network loaded: %s\n", network.Name)
	fmt.Printf("UUID: %s\n", network.UUID)
	fmt.Printf("Layers: %d\n", len(network.Layers))
	fmt.Printf("Neurons: %d\n", len(network.NeuronCache))

	// Print layer information
	for uuid, layer := range network.Layers {
		fmt.Printf("Layer %s: %s\n", uuid, layer.Name)
		fmt.Printf("  Dimensions: %dx%d\n", layer.Width, layer.Height)
		fmt.Printf("  Neurons: %d\n", len(layer.Neurons))
	}
}

// processData processes data through the network
func processData(flags CommandFlags) {
	fmt.Printf("Loading network with seed %d...\n", flags.Seed)

	// Load the network
	network, err := LoadNetworkFromFile(flags.FilePath+".bin", flags.Seed)
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

	// Create test data
	testData := make([]float64, flags.VectorDims)
	for i := range testData {
		testData[i] = float64(i) / float64(flags.VectorDims)
	}

	// Process data
	fmt.Println("Processing data...")
	result, path, err := network.ProcessData(startLayerUUID, startNeuronUUID, testData, 10)
	if err != nil {
		fmt.Printf("Error processing data: %v\n", err)
		os.Exit(1)
	}

	// Print results
	fmt.Println("Processing complete.")
	fmt.Printf("Path length: %d\n", len(path))
	fmt.Printf("Result dimensions: %d\n", len(result))

	// Print the first few dimensions of the result
	fmt.Println("Result preview:")
	previewCount := 5
	if len(result) < previewCount {
		previewCount = len(result)
	}
	for i := 0; i < previewCount; i++ {
		fmt.Printf("  Dimension %d: %f\n", i, result[i])
	}
}

// runBenchmark runs performance benchmarks
func runBenchmark(flags CommandFlags) {
	fmt.Println("Running benchmarks...")

	// Create a network for benchmarking
	network, err := NewNetwork("Benchmark Network", flags.Seed)
	if err != nil {
		fmt.Printf("Error creating network: %v\n", err)
		os.Exit(1)
	}

	// Create layers
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

	// Connect layers - always use full connections
	fmt.Println("Connecting all layers...")
	err = network.ConnectAllLayers()
	if err != nil {
		fmt.Printf("Error connecting layers: %v\n", err)
		os.Exit(1)
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
	BinarySerializationBenchmark(network, benchmarkFilePath, flags.Seed)

	fmt.Println("\nBenchmarks complete.")
}
