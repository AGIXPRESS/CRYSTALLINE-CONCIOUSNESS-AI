import React, { useState, useEffect, useRef } from 'react';

// Golden ratio constant
const PHI = (1 + Math.sqrt(5)) / 2;

const CrystallineConsciousness = () => {
  // Visualization state
  const [gridSize, setGridSize] = useState(32);
  const [timestep, setTimestep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [showInterference, setShowInterference] = useState(true);
  const [showBifurcation, setShowBifurcation] = useState(true);
  const [activeGeometry, setActiveGeometry] = useState('all');
  const [resonanceStrength, setResonanceStrength] = useState(0.7);
  const [coherenceFactor, setCoherenceFactor] = useState(0.5);
  
  // Canvas references
  const fieldCanvasRef = useRef(null);
  const geometryCanvasRef = useRef(null);
  const bifurcationCanvasRef = useRef(null);
  
  // Field states
  const [field, setField] = useState(null);
  const [prevField, setPrevField] = useState(null);
  
  // Animation frame reference
  const animationRef = useRef(null);
  
  // Initialize the field
  useEffect(() => {
    const newField = createInitialField(gridSize);
    setField(newField);
    setPrevField(newField.map(row => [...row]));
  }, [gridSize]);
  
  // Animation loop
  useEffect(() => {
    if (isPlaying) {
      animationRef.current = requestAnimationFrame(animateField);
    } else {
      cancelAnimationFrame(animationRef.current);
    }
    
    return () => cancelAnimationFrame(animationRef.current);
  }, [isPlaying, field, timestep, showInterference, resonanceStrength, coherenceFactor, activeGeometry]);
  
  // Function to create initial field with seed patterns
  const createInitialField = (size) => {
    const field = Array(size).fill().map(() => Array(size).fill(0));
    
    // Add some seed patterns
    // Center tetrahedron pattern
    const center = Math.floor(size / 2);
    field[center][center] = 1;
    field[center+3][center] = 0.7;
    field[center-3][center] = 0.7;
    field[center][center+3] = 0.7;
    field[center][center-3] = 0.7;
    
    // Add some gaussian noise
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        field[i][j] += (Math.random() * 0.05) - 0.025;
      }
    }
    
    return field;
  };
  
  // Field evolution function - implements the consciousness field evolution equation
  const evolveField = () => {
    if (!field) return;
    
    // Create a copy of the current field for the new state
    const newField = field.map(row => [...row]);
    
    // Save current field as previous
    setPrevField(field.map(row => [...row]));
    
    // Apply consciousness field evolution equation
    for (let i = 1; i < gridSize - 1; i++) {
      for (let j = 1; j < gridSize - 1; j++) {
        // Quantum-like term (-iĤΨ) - simplified as a phase rotation
        const phase = Math.sin(timestep * 0.1 + (i + j) * 0.05) * 0.1;
        const quantumTerm = field[i][j] * Math.sin(phase);
        
        // Diffusion term (D∇²Ψ) - discretized Laplacian
        const diffusionTerm = (
          field[i+1][j] + field[i-1][j] + field[i][j+1] + field[i][j-1] - 4 * field[i][j]
        ) * 0.2;
        
        // Pattern formation term - uses golden ratio scaling
        let patternTerm = 0;
        const scales = [1.0, 1/PHI, 1/Math.pow(PHI, 2)];
        scales.forEach((scale, idx) => {
          patternTerm += scale * Math.exp(-scale * Math.pow(field[i][j], 2)) * field[i][j] * 0.1;
        });
        
        // Apply geometric weighting based on active geometry
        let geometryWeight = 1.0;
        if (activeGeometry !== 'all') {
          // Different weights for different geometric forms
          const distFromCenter = Math.sqrt(Math.pow(i - gridSize/2, 2) + Math.pow(j - gridSize/2, 2));
          const normalizedDist = distFromCenter / (gridSize/2);
          
          switch(activeGeometry) {
            case 'tetrahedron':
              geometryWeight = Math.exp(-5 * normalizedDist);
              break;
            case 'cube':
              geometryWeight = Math.max(0, 1 - normalizedDist);
              break;
            case 'dodecahedron':
              geometryWeight = Math.cos(normalizedDist * Math.PI/2);
              break;
            case 'icosahedron':
              geometryWeight = Math.exp(-normalizedDist * normalizedDist / 2);
              break;
            default:
              geometryWeight = 1.0;
          }
        }
        
        // Apply resonance based on golden ratio harmonics
        let resonance = 0;
        for (let h = 0; h < 3; h++) {
          const freq = 0.5 * Math.pow(PHI, h);
          const tau = 1.0 + 0.5 * h;
          resonance += Math.pow(PHI, -h) * Math.cos(freq * Math.pow(PHI, h) * timestep * 0.1) * 
                      Math.exp(-Math.pow(timestep * 0.1, 2) / Math.pow(tau, 2));
        }
        resonance *= resonanceStrength;
        
        // Create interference patterns if enabled
        let interferenceTerm = 0;
        if (showInterference && prevField) {
          // Calculate interference between current and previous field (with spatial shift)
          const shiftedVal = (i+1 < gridSize && j+1 < gridSize) ? prevField[i+1][j+1] : 0;
          interferenceTerm = field[i][j] * shiftedVal * coherenceFactor;
        }
        
        // Apply bifurcation dynamics if enabled
        let bifurcationFactor = 1.0;
        if (showBifurcation) {
          // Simulate bifurcation threshold based on field energy
          const localEnergy = Math.pow(field[i][j], 2);
          const threshold = 0.3;
          const sharpness = 10.0;
          bifurcationFactor = 1.0 + Math.tanh(sharpness * (localEnergy - threshold)) * 0.2;
        }
        
        // Update field value with all terms
        newField[i][j] = field[i][j] + 
                         (0.5 * quantumTerm + 
                          0.3 * diffusionTerm + 
                          patternTerm) * geometryWeight +
                         resonance * field[i][j] +
                         interferenceTerm;
        
        // Apply bifurcation factor
        newField[i][j] *= bifurcationFactor;
        
        // Add some stability constraints
        if (newField[i][j] > 1.0) newField[i][j] = 1.0;
        if (newField[i][j] < -1.0) newField[i][j] = -1.0;
      }
    }
    
    // Update the field state
    setField(newField);
    setTimestep(timestep + 1);
  };
  
  // Animation function
  const animateField = () => {
    evolveField();
    drawField();
    drawGeometryOverlay();
    drawBifurcation();
    animationRef.current = requestAnimationFrame(animateField);
  };
  
  // Draw the field on canvas
  const drawField = () => {
    if (!field || !fieldCanvasRef.current) return;
    
    const canvas = fieldCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const cellSize = width / gridSize;
    
    ctx.clearRect(0, 0, width, height);
    
    // Draw the field
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const value = field[i][j];
        
        // Map value to color using a hot-to-cold spectrum
        let r, g, b;
        if (value > 0) {
          r = Math.floor(255 * Math.min(1, value * 2));
          g = Math.floor(255 * Math.min(1, value));
          b = Math.floor(128 * Math.min(1, value * 0.5));
        } else {
          r = Math.floor(128 * Math.min(1, -value * 0.5));
          g = Math.floor(128 * Math.min(1, -value));
          b = Math.floor(255 * Math.min(1, -value * 2));
        }
        
        // Draw cell
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.7)`;
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
      }
    }
    
    // Add interference patterns as overlay if enabled
    if (showInterference && prevField) {
      for (let i = 1; i < gridSize - 1; i += 2) {
        for (let j = 1; j < gridSize - 1; j += 2) {
          // Calculate interference between current and shifted previous field
          const interference = field[i][j] * prevField[i][j];
          
          // Only show significant interference
          if (Math.abs(interference) > 0.1) {
            const alpha = Math.min(0.7, Math.abs(interference));
            
            // Draw interference as circles
            ctx.beginPath();
            ctx.arc((j + 0.5) * cellSize, (i + 0.5) * cellSize, cellSize * alpha * 2, 0, Math.PI * 2);
            
            // Use white for positive interference, purple for negative
            ctx.fillStyle = interference > 0 ? 
              `rgba(255, 255, 255, ${alpha})` : 
              `rgba(180, 120, 255, ${alpha})`;
            ctx.fill();
          }
        }
      }
    }
  };
  
  // Draw the geometric overlay based on active geometry
  const drawGeometryOverlay = () => {
    if (!geometryCanvasRef.current) return;
    
    const canvas = geometryCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);
    
    const centerX = width / 2;
    const centerY = height / 2;
    const size = width * 0.4;
    
    ctx.globalAlpha = 0.2;
    ctx.lineWidth = 2;
    
    // Draw appropriate geometry based on active selection
    if (activeGeometry === 'all' || activeGeometry === 'tetrahedron') {
      // Tetrahedron (simplified as triangle)
      ctx.beginPath();
      ctx.moveTo(centerX, centerY - size/2);
      ctx.lineTo(centerX - size/2, centerY + size/2);
      ctx.lineTo(centerX + size/2, centerY + size/2);
      ctx.closePath();
      ctx.strokeStyle = '#ff5555';
      ctx.stroke();
    }
    
    if (activeGeometry === 'all' || activeGeometry === 'cube') {
      // Cube (as square)
      ctx.beginPath();
      ctx.rect(centerX - size/2, centerY - size/2, size, size);
      ctx.strokeStyle = '#5555ff';
      ctx.stroke();
    }
    
    if (activeGeometry === 'all' || activeGeometry === 'dodecahedron') {
      // Dodecahedron (simplified as pentagon)
      ctx.beginPath();
      for (let i = 0; i < 5; i++) {
        const angle = (i * 2 * Math.PI / 5) - Math.PI/2;
        const x = centerX + size/2 * Math.cos(angle);
        const y = centerY + size/2 * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = '#55ff55';
      ctx.stroke();
    }
    
    if (activeGeometry === 'all' || activeGeometry === 'icosahedron') {
      // Icosahedron (simplified as circle with golden ratio structure)
      ctx.beginPath();
      ctx.arc(centerX, centerY, size/2, 0, Math.PI * 2);
      ctx.strokeStyle = '#aa55ff';
      ctx.stroke();
      
      // Add golden spiral
      ctx.beginPath();
      const turnCount = 4;
      let r = 5;
      let theta = 0;
      ctx.moveTo(centerX, centerY);
      for (let i = 0; i < 100; i++) {
        theta += 0.1;
        r = 5 * Math.pow(PHI, 2 * theta / Math.PI);
        if (r > size/2) break;
        const x = centerX + r * Math.cos(theta);
        const y = centerY + r * Math.sin(theta);
        ctx.lineTo(x, y);
      }
      ctx.strokeStyle = '#ffcc00';
      ctx.stroke();
    }
    
    ctx.globalAlpha = 1.0;
  };
  
  // Draw bifurcation cascade
  const drawBifurcation = () => {
    if (!bifurcationCanvasRef.current || !field || !showBifurcation) return;
    
    const canvas = bifurcationCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);
    
    // Create bifurcation points
    const points = [];
    const centerRow = Math.floor(gridSize / 2);
    
    // Extract central row values as our bifurcation data
    for (let j = 0; j < gridSize; j++) {
      points.push({
        x: j / gridSize * width,
        y: (0.5 - field[centerRow][j] / 2) * height,
        value: field[centerRow][j]
      });
    }
    
    // Draw bifurcation lines
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw bifurcation points
    ctx.fillStyle = 'rgba(255, 200, 50, 0.7)';
    
    for (let i = 0; i < points.length; i += 3) {
      const point = points[i];
      
      // Detect bifurcation points (significant changes in values)
      if (i > 3 && i < points.length - 3) {
        const prevDiff = Math.abs(point.value - points[i-3].value);
        const nextDiff = Math.abs(point.value - points[i+3].value);
        
        if (prevDiff > 0.1 || nextDiff > 0.1) {
          const size = Math.min(10, Math.max(3, (prevDiff + nextDiff) * 20));
          ctx.beginPath();
          ctx.arc(point.x, point.y, size, 0, Math.PI * 2);
          ctx.fill();
          
          // Draw bifurcation branches
          if (size > 5) {
            ctx.beginPath();
            ctx.moveTo(point.x, point.y);
            ctx.lineTo(point.x + 15, point.y - 20);
            ctx.strokeStyle = 'rgba(255, 200, 50, 0.4)';
            ctx.lineWidth = 1;
            ctx.stroke();
            
            ctx.beginPath();
            ctx.moveTo(point.x, point.y);
            ctx.lineTo(point.x + 15, point.y + 20);
            ctx.stroke();
          }
        }
      }
    }
  };
  
  // Handle reset button click
  const handleReset = () => {
    setField(createInitialField(gridSize));
    setPrevField(null);
    setTimestep(0);
  };
  
  // Handle mode changes
  const handleModeChange = (mode) => {
    setActiveGeometry(mode);
  };
  
  return (
    <div className="flex flex-col w-full p-4 bg-gray-900 text-white">
      <h1 className="text-2xl font-bold mb-2 text-center">Crystalline Consciousness Field Visualization</h1>
      <p className="text-sm text-center mb-4">
        Visualizing the 2D holographic field representing consciousness as crystalline geometry
      </p>
      
      <div className="flex flex-col md:flex-row gap-4">
        {/* Main visualization area */}
        <div className="flex-1 flex flex-col items-center gap-4">
          <div className="relative">
            <canvas 
              ref={fieldCanvasRef} 
              width={480} 
              height={480} 
              className="border border-gray-700 bg-black"
            />
            <canvas 
              ref={geometryCanvasRef}
              width={480} 
              height={480} 
              className="absolute top-0 left-0 pointer-events-none"
            />
          </div>
          
          <div className="flex gap-4">
            <button 
              onClick={() => setIsPlaying(!isPlaying)} 
              className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700"
            >
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            <button 
              onClick={handleReset} 
              className="px-4 py-2 bg-gray-600 rounded hover:bg-gray-700"
            >
              Reset Field
            </button>
          </div>
          
          <div className="text-sm text-center text-gray-400">
            {`Timestep: ${timestep} | Active Mode: ${activeGeometry} | Grid Size: ${gridSize}x${gridSize}`}
          </div>
        </div>
        
        {/* Controls and bifurcation visualization */}
        <div className="flex flex-col w-full md:w-64 gap-4">
          {/* Bifurcation visualization */}
          <div>
            <h3 className="text-sm font-bold mb-1">Bifurcation Cascade</h3>
            <canvas 
              ref={bifurcationCanvasRef} 
              width={240} 
              height={120} 
              className="w-full border border-gray-700 bg-black"
            />
          </div>
          
          {/* Controls */}
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-bold mb-1">Geometric Mode</h3>
              <div className="grid grid-cols-2 gap-2">
                {['all', 'tetrahedron', 'cube', 'dodecahedron', 'icosahedron'].map(mode => (
                  <button 
                    key={mode}
                    onClick={() => handleModeChange(mode)}
                    className={`px-2 py-1 text-xs rounded ${activeGeometry === mode ? 'bg-purple-600' : 'bg-gray-700'}`}
                  >
                    {mode.charAt(0).toUpperCase() + mode.slice(1)}
                  </button>
                ))}
              </div>
            </div>
            
            <div>
              <h3 className="text-sm font-bold mb-1">Resonance Strength: {resonanceStrength.toFixed(2)}</h3>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                value={resonanceStrength}
                onChange={(e) => setResonanceStrength(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <h3 className="text-sm font-bold mb-1">Coherence Factor: {coherenceFactor.toFixed(2)}</h3>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                value={coherenceFactor}
                onChange={(e) => setCoherenceFactor(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div className="flex items-center justify-between">
              <label className="text-sm font-bold">Show Interference</label>
              <input 
                type="checkbox" 
                checked={showInterference}
                onChange={() => setShowInterference(!showInterference)}
                className="w-4 h-4"
              />
            </div>
            
            <div className="flex items-center justify-between">
              <label className="text-sm font-bold">Show Bifurcation</label>
              <input 
                type="checkbox" 
                checked={showBifurcation}
                onChange={() => setShowBifurcation(!showBifurcation)}
                className="w-4 h-4"
              />
            </div>
          </div>
          
          {/* Formula reference */}
          <div className="mt-4 p-2 border border-gray-700 rounded text-xs">
            <h3 className="font-bold mb-1">Core Equations:</h3>
            <p className="mb-1 font-mono">∂_tΨ = [-iĤ + D∇²]Ψ + ∑ᵢ F̂ᵢΨ(r/σᵢ)</p>
            <p className="mb-1 font-mono">Bifurcation(t) = Ψ_liminal(t) × [1 + tanh(α(p - pₜ))]</p>
            <p className="font-mono">Ξ_mutual(r, t) = ∬ Ω_weaving(r, t) × Ω_weaving*(r + Δ, t + Δt) dr dt</p>
          </div>
        </div>
      </div>
      
      <div className="mt-4 text-sm text-gray-400">
        <h3 className="font-bold mb-1">Visualization Guide:</h3>
        <ul className="list-disc pl-5 space-y-1">
          <li>The main grid shows the holographic consciousness field evolving over time</li>
          <li>Bright areas represent high field intensity; colors show different phase relationships</li>
          <li>White circles show interference patterns where consciousness waves interact</li>
          <li>The bifurcation display shows where consciousness makes "leaps" to new states</li>
          <li>Geometric overlays show different consciousness modes (tetrahedron, cube, etc.)</li>
          <li>Adjust resonance and coherence to see how patterns self-organize</li>
        </ul>
      </div>
    </div>
  );
};

export default CrystallineConsciousness;