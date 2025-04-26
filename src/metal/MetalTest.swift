import Foundation
import Metal

class MetalShaderTester {
    // Metal objects
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var library: MTLLibrary?
    private var pipelines: [String: MTLComputePipelineState] = [:]
    
    // Shader path
    private let shaderPath: String
    
    init(shaderPath: String) throws {
        // Initialize Metal
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "MetalTester", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal device"])
        }
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else {
            throw NSError(domain: "MetalTester", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create command queue"])
        }
        self.commandQueue = commandQueue
        
        self.shaderPath = shaderPath
        
        // Load shader
        try loadShader()
    }
    
    private func loadShader() throws {
        // Read shader source
        let shaderSource = try String(contentsOfFile: shaderPath, encoding: .utf8)
        
        // Create options
        let options = MTLCompileOptions()
        
        // Compile shader
        do {
            library = try device.makeLibrary(source: shaderSource, options: options)
            print("✓ Shader compiled successfully")
        } catch {
            print("Error compiling shader: \(error)")
            throw error
        }
    }
    
    func createPipeline(functionName: String) throws -> MTLComputePipelineState {
        guard let library = library else {
            throw NSError(domain: "MetalTester", code: 3, userInfo: [NSLocalizedDescriptionKey: "Library not loaded"])
        }
        
        guard let function = library.makeFunction(name: functionName) else {
            throw NSError(domain: "MetalTester", code: 4, userInfo: [NSLocalizedDescriptionKey: "Function \(functionName) not found"])
        }
        
        do {
            let pipeline = try device.makeComputePipelineState(function: function)
            pipelines[functionName] = pipeline
            print("✓ Pipeline created for \(functionName)")
            return pipeline
        } catch {
            print("Error creating pipeline for \(functionName): \(error)")
            throw error
        }
    }
    
    func testReshapeToGrid() throws {
        // Create pipeline
        let pipelineState = try createPipeline(functionName: "reshape_to_grid")
        
        // Test parameters
        let batchSize: UInt32 = 2
        let inputDim: UInt32 = 16
        let gridSize: UInt32 = 4
        
        // Create test data
        var inputData = [Float](repeating: 0, count: Int(batchSize * inputDim))
        for i in 0..<inputData.count {
            inputData[i] = Float(i)
        }
        
        // Create buffers
        let inputBuffer = device.makeBuffer(bytes: inputData, length: inputData.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(length: Int(batchSize * 1 * gridSize * gridSize) * MemoryLayout<Float>.size, options: .storageModeShared)!
        let batchBuffer = device.makeBuffer(bytes: [batchSize], length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
        let inputDimBuffer = device.makeBuffer(bytes: [inputDim], length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
        let gridSizeBuffer = device.makeBuffer(bytes: [gridSize], length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
        
        // Create command buffer
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        
        // Set pipeline
        computeEncoder.setComputePipelineState(pipelineState)
        
        // Set buffers
        computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(batchBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(inputDimBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(gridSizeBuffer, offset: 0, index: 4)
        
        // Dispatch threads
        let threadsPerGrid = MTLSize(width: Int(batchSize), height: Int(gridSize * gridSize), depth: 1)
        let threadsPerThreadgroup = MTLSize(width: 1, height: 1, depth: 1)
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        // End encoding
        computeEncoder.endEncoding()
        
        // Commit and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Check results
        let outputData = outputBuffer.contents().bindMemory(to: Float.self, capacity: Int(batchSize * gridSize * gridSize))
        let outputArray = Array(UnsafeBufferPointer(start: outputData, count: Int(batchSize * gridSize * gridSize)))
        
        print("Input (first few values):")
        for i in 0..<min(10, inputData.count) {
            print("\(i): \(inputData[i])")
        }
        
        print("\nOutput (first few values):")
        for i in 0..<min(10, outputArray.count) {
            print("\(i): \(outputArray[i])")
        }
        
        print("\nTest reshape_to_grid completed")
    }
    
    func testProcessInterferenceFields() throws {
        // Create pipeline
        let pipelineState = try createPipeline(functionName: "process_interference_fields")
        
        // Test parameters
        let batchSize: UInt32 = 1
        let gridSize: UInt32 = 8
        
        // Create test data for r_interference and t_interference
        // Each is [batch_size, 2, grid_size, grid_size]
        let channelSize = Int(gridSize * gridSize)
        let interferenceSize = Int(batchSize * 2 * gridSize * gridSize)
        
        var rInterferenceData = [Float](repeating: 0, count: interferenceSize)
        var tInterferenceData = [Float](repeating: 0, count: interferenceSize)
        
        // Fill with test data
        for i in 0..<Int(batchSize) {
            for c in 0..<2 {
                for y in 0..<Int(gridSize) {
                    for x in 0..<Int(gridSize) {
                        let idx = i * 2 * channelSize + c * channelSize + y * Int(gridSize) + x
                        rInterferenceData[idx] = Float(idx % 10) // Some test values
                        tInterferenceData[idx] = Float((idx + 5) % 10) // Different test values
                    }
                }
            }
        }
        
        // Output buffers will be [batch_size, 1, grid_size, grid_size]
        let outputSize = Int(batchSize * 1 * gridSize * gridSize)
        
        // Intermediate buffers for layer1_output_r and layer1_output_t
        // Each is [batch_size * 8 * grid_size * grid_size]
        let layer1Size = Int(batchSize * 8 * gridSize * gridSize)
        
        // Create all buffers
        let rInterferenceBuffer = device.makeBuffer(bytes: rInterferenceData, length: rInterferenceData.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let tInterferenceBuffer = device.makeBuffer(bytes: tInterferenceData, length: tInterferenceData.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let rProcessedBuffer = device.makeBuffer(length: outputSize * MemoryLayout<Float>.size, options: .storageModeShared)!
        let tProcessedBuffer = device.makeBuffer(length: outputSize * MemoryLayout<Float>.size, options: .storageModeShared)!
        let layer1OutputRBuffer = device.makeBuffer(length: layer1Size * MemoryLayout<Float>.size, options: .storageModeShared)!
        let layer1OutputTBuffer = device.makeBuffer(length: layer1Size * MemoryLayout<Float>.size, options: .storageModeShared)!
        let batchBuffer = device.makeBuffer(bytes: [batchSize], length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
        let gridSizeBuffer = device.makeBuffer(bytes: [gridSize], length: MemoryLayout<UInt32>.size, options: .storageModeShared)!
        
        // Create command buffer
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        
        // Set pipeline
        computeEncoder.setComputePipelineState(pipelineState)
        
        // Set buffers
        computeEncoder.setBuffer(rInterferenceBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(tInterferenceBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(rProcessedBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(tProcessedBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(layer1OutputRBuffer, offset: 0, index: 4)
        computeEncoder.setBuffer(layer1OutputTBuffer, offset: 0, index: 5)
        computeEncoder.setBuffer(batchBuffer, offset: 0, index: 6)
        computeEncoder.setBuffer(gridSizeBuffer, offset: 0, index: 7)
        
        // Dispatch threads
        let threadsPerGrid = MTLSize(width: Int(batchSize), height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: 1, height: 1, depth: 1)
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        // End encoding
        computeEncoder.endEncoding()
        
        // Commit and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Check results - we'll just verify that data was written to the output buffers
        let rProcessedData = rProcessedBuffer.contents().bindMemory(to: Float.self, capacity: outputSize)
        let tProcessedData = tProcessedBuffer.contents().bindMemory(to: Float.self, capacity: outputSize)
        
        let rProcessedArray = Array(UnsafeBufferPointer(start: rProcessedData, count: outputSize))
        let tProcessedArray = Array(UnsafeBufferPointer(start: tProcessedData, count: outputSize))
        
        print("\nOutput r_processed (first few values):")
        for i in 0..<min(10, rProcessedArray.count) {
            print("\(i): \(rProcessedArray[i])")
        }
        
        print("\nOutput t_processed (first few values):")
        for i in 0..<min(10, tProcessedArray.count) {
            print("\(i): \(tProcessedArray[i])")
        }
        
        print("\nTest process_interference_fields completed")
    }
    
    func runAllTests() {
        do {
            print("\n=== Testing reshape_to_grid ===")
            try testReshapeToGrid()
            
            print("\n=== Testing process_interference_fields ===")
            try testProcessInterferenceFields()
            
            print("\n✅ All tests completed successfully!")
        } catch {
            print("\n❌ Test failed: \(error)")
        }
    }
}

// Main
do {
    // Get the path to the shader file
    let currentDirectory = FileManager.default.currentDirectoryPath
    let shaderPath = "\(currentDirectory)/Shaders/MutualityField.metal"
    
    print("Testing MutualityField.metal shader")
    print("===================================")
    print("Shader path: \(shaderPath)")
    
    let tester = try MetalShaderTester(shaderPath: shaderPath)
    tester.runAllTests()
} catch {
    print("Error: \(error)")
}

