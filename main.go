package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strings"
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
	return map[string]Command{
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
	}
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
				fmt.Println("\nJSON Serialization Benchmark Results:")
				fmt.Printf("JSON save time: %v\n", jsonSaveTime)
				fmt.Printf("JSON load time: %v\n", jsonLoadTime)
				fmt.Printf("JSON file size: %v bytes\n", jsonFileSize)
			}
		}
	}

	// Run processing benchmark if verbose
	if flags.Verbose {
		fmt.Println("\nRunning processing benchmark...")

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
		processingStartTime := time.Now()
		_, _, err := network.ProcessData(startLayerUUID, startNeuronUUID, testData, 10)
		if err != nil {
			fmt.Printf("Error in processing benchmark: %v\n", err)
		} else {
			processingTime := time.Since(processingStartTime)
			fmt.Printf("Processing time: %v\n", processingTime)
		}
	}

	fmt.Println("\nBenchmarks complete.")
}

// validateFilePath ensures the file path is valid and has the correct extension
func validateFilePath(filePath string, extension string) string {
	// Add extension if not present
	if !strings.HasSuffix(filePath, extension) {
		return filePath + extension
	}
	return filePath
}
