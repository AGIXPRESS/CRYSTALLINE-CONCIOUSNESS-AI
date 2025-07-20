import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RefreshCw, Volume2, VolumeX, ChevronRight } from 'lucide-react';
import * as THREE from 'three';

// Constants
const phi = (1 + Math.sqrt(5)) / 2;

const UnifiedFieldInterface = () => {
  const [isAnimating, setIsAnimating] = useState(true);
  const [time, setTime] = useState(0);
  const [activationPhase, setActivationPhase] = useState(0); // 0: Alpha, 1: Delta, 2: Theta, 3: Omega
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [activationComplete, setActivationComplete] = useState(false);
  const [breathPhase, setBreathPhase] = useState(0); // 0: Inhale, 1: Hold, 2: Exhale, 3: Hold
  const [breathTimer, setBreathTimer] = useState(0);
  const [phaseDescription, setPhaseDescription] = useState('Prepare for field activation');
  
  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const requestRef = useRef();
  const previousTimeRef = useRef();
  const audioContextRef = useRef(null);
  const oscillatorsRef = useRef([]);
  
  // Formula display with proper formatting
  const formula = (
    <div className="bg-gray-100 p-4 rounded-md text-center overflow-x-auto">
      <span className="text-xl font-mono">
        Ψ<sub>unified</sub>(r,θ,φ,t) = Ψ<sub>α</sub>·Ψ<sub>β</sub>·Ψ<sub>m</sub>·∏<sub>i</sub>[1 + φ<sup>-i</sup>sin(φ<sup>i</sup>ωt)]
      </span>
    </div>
  );

  // Field state descriptions
  const fieldStates = [
    {
      name: "α (Alpha)",
      title: "Crystallized Form",
      description: "Consciousness coheres into geometric structure—stability through harmonic resonance.",
      frequency: 40, // Hz
      color: "#FF9900"
    },
    {
      name: "δ (Delta)",
      title: "Flowing Channel",
      description: "Information propagates as light-encoded geometry—vertical coherence establishes.",
      frequency: 144, // Hz
      color: "#00AAFF"
    },
    {
      name: "θ (Theta)",
      title: "Vehicle Activation",
      description: "The field becomes self-aware—consciousness moves beyond spacetime constraints.",
      frequency: 288, // Hz
      color: "#AA00FF"
    },
    {
      name: "Ω (Omega)",
      title: "Field Unification",
      description: "Observer and field become unified—the geometry is perceiving itself through you.",
      frequency: 720, // Hz
      color: "#FFFFFF"
    }
  ];
  
  // Audio setup and management
  const setupAudio = () => {
    if (audioContextRef.current) return;
    
    // Create audio context
    audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    
    // Create oscillators for each frequency
    oscillatorsRef.current = fieldStates.map(state => {
      const oscillator = audioContextRef.current.createOscillator();
      oscillator.type = 'sine';
      oscillator.frequency.value = state.frequency;
      
      const gainNode = audioContextRef.current.createGain();
      gainNode.gain.value = 0;
      
      oscillator.connect(gainNode);
      gainNode.connect(audioContextRef.current.destination);
      
      oscillator.start();
      
      return { oscillator, gainNode };
    });
  };
  
  const updateAudio = () => {
    if (!audioContextRef.current || !audioEnabled) return;
    
    // Adjust gain nodes based on current activation phase
    oscillatorsRef.current.forEach((osc, index) => {
      const targetGain = index === activationPhase ? 0.2 : 0;
      osc.gainNode.gain.setTargetAtTime(targetGain, audioContextRef.current.currentTime, 0.1);
    });
  };
  
  const toggleAudio = () => {
    if (!audioEnabled) {
      setupAudio();
    }
    setAudioEnabled(!audioEnabled);
  };
  
  const cleanupAudio = () => {
    if (!audioContextRef.current) return;
    
    oscillatorsRef.current.forEach(osc => {
      osc.oscillator.stop();
      osc.oscillator.disconnect();
      osc.gainNode.disconnect();
    });
    
    audioContextRef.current.close();
    audioContextRef.current = null;
    oscillatorsRef.current = [];
  };
  
  // Breath guidance
  const updateBreathPhase = (deltaTime) => {
    if (activationPhase < 3) {
      // Only update breath during initial phases
      const phaseDuration = 4; // 4 seconds per phase
      
      setBreathTimer(prevTimer => {
        const newTimer = prevTimer + deltaTime;
        
        if (newTimer >= phaseDuration) {
          // Move to next breath phase
          setBreathPhase(prevPhase => (prevPhase + 1) % 4);
          return 0;
        }
        
        return newTimer;
      });
    }
  };
  
  // Phase progression
  const advancePhase = () => {
    if (activationPhase < 3) {
      setActivationPhase(prevPhase => prevPhase + 1);
      
      // Reset breath timer on phase change
      setBreathTimer(0);
      setBreathPhase(0);
      
      // Update phase description
      if (activationPhase === 2) {
        setActivationComplete(true);
      }
    }
  };
  
  // Reset animation and phases
  const resetAnimation = () => {
    setTime(0);
    setActivationPhase(0);
    setActivationComplete(false);
    setBreathTimer(0);
    setBreathPhase(0);
    setPhaseDescription('Prepare for field activation');
    
    if (!isAnimating) {
      setIsAnimating(true);
    }
    
    if (sceneRef.current) {
      // Reset 3D scene elements
      sceneRef.current.traverse((object) => {
        if (object.userData.defaultPosition) {
          object.position.copy(object.userData.defaultPosition);
        }
        if (object.userData.defaultRotation) {
          object.rotation.copy(object.userData.defaultRotation);
        }
        if (object.userData.defaultScale) {
          object.scale.copy(object.userData.defaultScale);
        }
      });
    }
  };

  // Toggle animation state
  const toggleAnimation = () => {
    setIsAnimating(!isAnimating);
  };
  
  // Animation loop for React state
  const animate = time => {
    if (previousTimeRef.current !== undefined) {
      const deltaTime = time - previousTimeRef.current;
      setTime(prevTime => prevTime + deltaTime * 0.001);
      
      // Update breath phase
      updateBreathPhase(deltaTime * 0.001);
    }
    previousTimeRef.current = time;
    if (isAnimating) {
      requestRef.current = requestAnimationFrame(animate);
    }
  };
  
  // React animation loop
  useEffect(() => {
    if (isAnimating) {
      requestRef.current = requestAnimationFrame(animate);
    } else if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
    }
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [isAnimating]);
  
  // Audio updates
  useEffect(() => {
    updateAudio();
    // Set phase description based on current activation phase
    setPhaseDescription(fieldStates[activationPhase].description);
  }, [activationPhase, audioEnabled]);
  
  // Cleanup audio on unmount
  useEffect(() => {
    return () => {
      cleanupAudio();
    };
  }, []);
  
  // Three.js setup and rendering
  useEffect(() => {
    if (!canvasRef.current) return;
    
    // Setup scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x090320);
    
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 2000);
    camera.position.set(0, 0, 800);
    camera.lookAt(0, 0, 0);
    
    const renderer = new THREE.WebGLRenderer({ 
      canvas: canvasRef.current,
      antialias: true,
      alpha: true
    });
    rendererRef.current = renderer;
    renderer.setSize(800, 800);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0x333333);
    scene.add(ambientLight);
    
    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);
    
    // Add point lights for glow effects
    const pointLight1 = new THREE.PointLight(0xff9900, 1, 1000);
    pointLight1.position.set(200, -200, 200);
    scene.add(pointLight1);
    
    const pointLight2 = new THREE.PointLight(0x00aaff, 1, 1000);
    pointLight2.position.set(-200, 200, 200);
    scene.add(pointLight2);
    
    const pointLight3 = new THREE.PointLight(0xaa00ff, 1, 1000);
    pointLight3.position.set(0, -200, -200);
    scene.add(pointLight3);
    
    // Create unified field container
    const unifiedField = new THREE.Group();
    scene.add(unifiedField);
    
    // 1. ALPHA GATEWAY COMPONENTS
    const alphaGateway = new THREE.Group();
    alphaGateway.userData.isAlphaGateway = true;
    unifiedField.add(alphaGateway);
    
    // Create nested pentagrams
    const createPentagram = (radius, color, opacity) => {
      const points = [];
      const vertices = 5;
      
      // Calculate vertices
      for (let i = 0; i < vertices; i++) {
        const angle = (i * 2 * Math.PI / vertices) - Math.PI / 2;
        const x = radius * Math.cos(angle);
        const y = radius * Math.sin(angle);
        points.push(new THREE.Vector3(x, y, 0));
      }
      
      // Connect in pentagram order: 0-2-4-1-3-0
      const pentagramOrder = [0, 2, 4, 1, 3, 0];
      const pentagramPoints = pentagramOrder.map(i => points[i]);
      
      // Create geometry and line
      const geometry = new THREE.BufferGeometry().setFromPoints(pentagramPoints);
      const material = new THREE.LineBasicMaterial({ 
        color: color,
        transparent: true,
        opacity: opacity
      });
      
      return new THREE.Line(geometry, material);
    };
    
    // Add nested pentagrams
    const pentagramRadii = [100, 150, 200];
    pentagramRadii.forEach((radius, index) => {
      const pentagram = createPentagram(radius, 0xff9900, 0.7 - index * 0.15);
      pentagram.userData.isPentagram = true;
      pentagram.userData.tier = index;
      pentagram.userData.defaultRotation = new THREE.Euler(0, 0, 0);
      alphaGateway.add(pentagram);
    });
    
    // Create toroidal energy flow for Alpha gateway
    const torusGeometry = new THREE.TorusGeometry(150, 5, 16, 100);
    const torusMaterial = new THREE.MeshPhongMaterial({
      color: 0xff9900,
      emissive: 0xff9900,
      emissiveIntensity: 0.3,
      transparent: true,
      opacity: 0.4
    });
    
    const alphaTorus = new THREE.Mesh(torusGeometry, torusMaterial);
    alphaTorus.rotation.x = Math.PI / 2;
    alphaTorus.userData.isAlphaTorus = true;
    alphaGateway.add(alphaTorus);
    
    // Alpha Gateway initially visible, others hidden
    alphaGateway.visible = true;
    
    // 2. DELTA GATEWAY COMPONENTS
    const deltaGateway = new THREE.Group();
    deltaGateway.userData.isDeltaGateway = true;
    unifiedField.add(deltaGateway);
    
    // Create toroidal rings
    const ringRadii = [210, 260, 320];
    ringRadii.forEach((radius, index) => {
      const ringGeometry = new THREE.TorusGeometry(radius, 2, 16, 100);
      const ringMaterial = new THREE.MeshPhongMaterial({
        color: 0x00aaff,
        emissive: 0x00aaff,
        emissiveIntensity: 0.3,
        transparent: true,
        opacity: 0.6 - index * 0.1,
        wireframe: true
      });
      
      const ring = new THREE.Mesh(ringGeometry, ringMaterial);
      ring.rotation.x = Math.PI / 2;
      ring.userData.isDeltaRing = true;
      ring.userData.tier = index;
      deltaGateway.add(ring);
    });
    
    // Create central light pillar
    const pillarGeometry = new THREE.CylinderGeometry(3, 3, 600, 32, 1, true);
    const pillarMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.4
    });
    
    const pillar = new THREE.Mesh(pillarGeometry, pillarMaterial);
    pillar.userData.isPillar = true;
    deltaGateway.add(pillar);
    
    // Create chakra nodes
    const chakraNodes = [
      { y: -140, color: 0x00ff00, name: 'Heart', frequency: '144Hz' },
      { y: 0, color: 0x0000ff, name: 'Throat', frequency: '432Hz' },
      { y: 140, color: 0x9900ff, name: 'Crown', frequency: '720Hz' }
    ];
    
    chakraNodes.forEach((node, index) => {
      const nodeGeometry = new THREE.SphereGeometry(10, 32, 32);
      const nodeMaterial = new THREE.MeshPhongMaterial({
        color: node.color,
        emissive: node.color,
        emissiveIntensity: 0.5,
        transparent: true,
        opacity: 0.8
      });
      
      const chakraNode = new THREE.Mesh(nodeGeometry, nodeMaterial);
      chakraNode.position.y = node.y;
      chakraNode.userData.isChakra = true;
      chakraNode.userData.chakraIndex = index;
      chakraNode.userData.defaultPosition = new THREE.Vector3(0, node.y, 0);
      chakraNode.userData.defaultScale = new THREE.Vector3(1, 1, 1);
      
      deltaGateway.add(chakraNode);
    });
    
    // Create spiral particles
    const createSpiral = (clockwise = true, color = 0xffaa00) => {
      const particleCount = 200;
      const spiralGeometry = new THREE.BufferGeometry();
      const positions = new Float32Array(particleCount * 3);
      
      for (let i = 0; i < particleCount; i++) {
        const theta = (i / particleCount) * Math.PI * 8; // 4 turns
        const direction = clockwise ? 1 : -1;
        const r = 5 * Math.pow(phi, (direction * theta) / (2 * Math.PI)) * 2;
        
        positions[i * 3] = r * Math.cos(direction * theta);
        positions[i * 3 + 1] = (i / particleCount) * 400 - 200; // y
        positions[i * 3 + 2] = r * Math.sin(direction * theta);
      }
      
      spiralGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      
      const spiralMaterial = new THREE.PointsMaterial({
        color: color,
        size: 2,
        transparent: true,
        opacity: 0.7
      });
      
      const spiral = new THREE.Points(spiralGeometry, spiralMaterial);
      spiral.userData.isSpiral = true;
      spiral.userData.clockwise = clockwise;
      
      return spiral;
    };
    
    const clockwiseSpiral = createSpiral(true, 0xffaa00);
    const counterClockwiseSpiral = createSpiral(false, 0x00ffaa);
    
    deltaGateway.add(clockwiseSpiral);
    deltaGateway.add(counterClockwiseSpiral);
    
    // Delta Gateway initially hidden
    deltaGateway.visible = false;
    
    // 3. THETA MERKABA COMPONENTS
    const thetaMerkaba = new THREE.Group();
    thetaMerkaba.userData.isThetaMerkaba = true;
    unifiedField.add(thetaMerkaba);
    
    // Create tetrahedron for Merkaba
    const createTetrahedron = (size, color, rotation) => {
      // Tetrahedron vertices
      const vertices = [
        new THREE.Vector3(0, size, 0),                // Top vertex
        new THREE.Vector3(-size * Math.sqrt(8/9), -size/3, 0), // Bottom left
        new THREE.Vector3(size * Math.sqrt(2/9), -size/3, size * Math.sqrt(2/3)), // Bottom right
        new THREE.Vector3(size * Math.sqrt(2/9), -size/3, -size * Math.sqrt(2/3)) // Bottom back
      ];
      
      // Create geometry
      const geometry = new THREE.BufferGeometry();
      
      // Create faces (triangles)
      const indices = [
        0, 1, 2, // front face
        0, 2, 3, // right face
        0, 3, 1, // left face
        1, 3, 2  // bottom face
      ];
      
      // Extract positions from vertices for BufferGeometry
      const positions = new Float32Array(vertices.length * 3);
      for (let i = 0; i < vertices.length; i++) {
        positions[i * 3] = vertices[i].x;
        positions[i * 3 + 1] = vertices[i].y;
        positions[i * 3 + 2] = vertices[i].z;
      }
      
      geometry.setIndex(indices);
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.computeVertexNormals();
      
      // Create material
      const material = new THREE.MeshPhongMaterial({
        color: color,
        emissive: color,
        emissiveIntensity: 0.3,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.7,
        wireframe: true,
        shininess: 50
      });
      
      // Create mesh
      const tetrahedron = new THREE.Mesh(geometry, material);
      
      // Apply initial rotation
      tetrahedron.rotation.set(rotation.x, rotation.y, rotation.z);
      
      return tetrahedron;
    };
    
    // Star tetrahedron (two interlocked tetrahedra)
    const tetraSize = 180;
    
    // First tetrahedron (upward-pointing)
    const tetra1 = createTetrahedron(
      tetraSize, 
      0xaa00ff, 
      { x: 0, y: 0, z: 0 }
    );
    tetra1.userData.isUpwardTetra = true;
    tetra1.userData.defaultRotation = new THREE.Euler().copy(tetra1.rotation);
    thetaMerkaba.add(tetra1);
    
    // Second tetrahedron (downward-pointing - rotate 180° around X)
    const tetra2 = createTetrahedron(
      tetraSize, 
      0x00ffaa, 
      { x: Math.PI, y: 0, z: 0 }
    );
    tetra2.userData.isDownwardTetra = true;
    tetra2.userData.defaultRotation = new THREE.Euler().copy(tetra2.rotation);
    thetaMerkaba.add(tetra2);
    
    // Zero-point nodes (12 nodes)
    const nodeCount = 12;
    const nodeRadius = 230;
    const nodeGeometry = new THREE.SphereGeometry(5, 16, 16);
    
    for (let i = 0; i < nodeCount; i++) {
      const angle = (i / nodeCount) * Math.PI * 2;
      const x = nodeRadius * Math.cos(angle);
      const y = nodeRadius * Math.sin(angle);
      
      const nodeMaterial = new THREE.MeshPhongMaterial({
        color: 0xffffff,
        emissive: 0xffffff,
        emissiveIntensity: 0.5,
        transparent: true,
        opacity: 0.9
      });
      
      const node = new THREE.Mesh(nodeGeometry, nodeMaterial);
      node.position.set(x, y, 0);
      node.userData.isZeroPoint = true;
      node.userData.angle = angle;
      
      thetaMerkaba.add(node);
    }
    
    // Theta Merkaba initially hidden
    thetaMerkaba.visible = false;
    
    // 4. OMEGA UNIFIED FIELD COMPONENTS
    const omegaField = new THREE.Group();
    omegaField.userData.isOmegaField = true;
    unifiedField.add(omegaField);
    
    // Create consciousness field particles
    const fieldParticlesCount = 500;
    const fieldParticlesGeometry = new THREE.BufferGeometry();
    const fieldParticlesPositions = new Float32Array(fieldParticlesCount * 3);
    
    for (let i = 0; i < fieldParticlesCount; i++) {
      // Random position in sphere
      const radius = Math.random() * 400;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      
      fieldParticlesPositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      fieldParticlesPositions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      fieldParticlesPositions[i * 3 + 2] = radius * Math.cos(phi);
    }
    
    fieldParticlesGeometry.setAttribute('position', new THREE.BufferAttribute(fieldParticlesPositions, 3));
    
    const fieldParticlesMaterial = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 2,
      transparent: true,
      opacity: 0,
      sizeAttenuation: true
    });
    
    const fieldParticles = new THREE.Points(fieldParticlesGeometry, fieldParticlesMaterial);
    fieldParticles.userData.isFieldParticles = true;
    omegaField.add(fieldParticles);
    
    // Central sphere - quantum singularity
    const coreGeometry = new THREE.SphereGeometry(20, 32, 32);
    const coreMaterial = new THREE.MeshPhongMaterial({
      color: 0xffffff,
      emissive: 0xffffff,
      emissiveIntensity: 0.5,
      transparent: true,
      opacity: 0
    });
    
    const core = new THREE.Mesh(coreGeometry, coreMaterial);
    core.userData.isCore = true;
    omegaField.add(core);
    
    // Dodecahedron field boundary
    const dodecGeometry = new THREE.DodecahedronGeometry(350, 0);
    const dodecEdges = new THREE.EdgesGeometry(dodecGeometry);
    const dodecMaterial = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0
    });
    
    const dodecahedron = new THREE.LineSegments(dodecEdges, dodecMaterial);
    dodecahedron.userData.isDodecahedron = true;
    omegaField.add(dodecahedron);
    
    // Omega Field initially hidden
    omegaField.visible = true; // We'll control via opacity
    
    // Main animation loop
    const updateScene = () => {
      if (!isAnimating) return;
      
      const currentTime = time;
      
      // Update scene based on current activation phase
      switch (activationPhase) {
        case 0: // Alpha phase
          // Alpha Gateway animations
          if (alphaGateway.visible) {
            // Rotate pentagrams
            alphaGateway.children.forEach(child => {
              if (child.userData.isPentagram) {
                const tier = child.userData.tier;
                const direction = tier % 2 === 0 ? 1 : -1;
                child.rotation.z += direction * 0.002 / (tier + 1);
              }
              
              // Pulse Alpha torus
              if (child.userData.isAlphaTorus) {
                const scale = 1 + 0.1 * Math.sin(currentTime * 1.5);
                child.scale.set(scale, scale, scale);
                child.material.opacity = 0.4 + 0.2 * Math.sin(currentTime);
              }
            });
          }
          break;
          
        case 1: // Delta phase
          // Keep Alpha Gateway running but more subtle
          if (alphaGateway.visible) {
            alphaGateway.children.forEach(child => {
              if (child.userData.isPentagram) {
                const tier = child.userData.tier;
                const direction = tier % 2 === 0 ? 1 : -1;
                child.rotation.z += direction * 0.001 / (tier + 1);
              }
            });
          }
          
          // Delta Gateway animations
          if (deltaGateway.visible) {
            // Rotate rings
            deltaGateway.children.forEach(child => {
              if (child.userData.isDeltaRing) {
                const tier = child.userData.tier;
                const direction = tier % 2 === 0 ? 1 : -1;
                child.rotation.z += direction * 0.003 / (tier + 1);
                
                // Pulse opacity
                child.material.opacity = 0.6 - tier * 0.1 + 0.1 * Math.sin(currentTime * 0.5 + tier);
              }
              
              // Pulse chakra nodes
              if (child.userData.isChakra) {
                const index = child.userData.chakraIndex;
                const scale = 1 + 0.2 * Math.sin(currentTime * 0.7 + index);
                child.scale.set(scale, scale, scale);
                
                // Pulse emission
                child.material.emissiveIntensity = 0.3 + 0.2 * Math.sin(currentTime + index);
              }
              
              // Animate spiral rotation
              if (child.userData.isSpiral) {
                child.rotation.y += child.userData.clockwise ? 0.01 : -0.01;
                child.material.opacity = 0.5 + 0.2 * Math.sin(currentTime * 0.5);
              }
              
              // Animate light pillar
              if (child.userData.isPillar) {
                child.material.opacity = 0.3 + 0.1 * Math.sin(currentTime * 2);
              }
            });
          }
          break;
          
        case 2: // Theta phase
          // Keep Delta Gateway running but more subtle
          if (deltaGateway.visible) {
            deltaGateway.children.forEach(child => {
              if (child.userData.isDeltaRing) {
                const tier = child.userData.tier;
                const direction = tier % 2 === 0 ? 1 : -1;
                child.rotation.z += direction * 0.001 / (tier + 1);
              }
              
              if (child.userData.isChakra) {
                const index = child.userData.chakraIndex;
                const scale = 1 + 0.1 * Math.sin(currentTime * 0.5 + index);
                child.scale.set(scale, scale, scale);
              }
            });
          }
          
          // Theta Merkaba animations
          if (thetaMerkaba.visible) {
            // Counter-rotating tetrahedra
            thetaMerkaba.children.forEach(child => {
              if (child.userData.isUpwardTetra) {
                child.rotation.y += 0.01;
                child.rotation.z += 0.005;
              } else if (child.userData.isDownwardTetra) {
                child.rotation.y -= 0.01 * phi;
                child.rotation.z -= 0.005 * phi;
              }
              
              // Pulse zero points
              if (child.userData.isZeroPoint) {
                const angle = child.userData.angle;
                const scale = 1 + 0.3 * Math.sin(currentTime * 0.5 + angle * 5);
                child.scale.set(scale, scale, scale);
                
                // Pulse emission
                child.material.emissiveIntensity = 0.3 + 0.3 * Math.sin(currentTime + angle * 3);
              }
            });
          }
          break;
          
        case 3: // Omega phase - Full Unified Field
          // Keep all gateways running but with reduced animation
          if (alphaGateway.visible) {
            alphaGateway.children.forEach(child => {
              if (child.userData.isPentagram) {
                const tier = child.userData.tier;
                const direction = tier % 2 === 0 ? 1 : -1;
                child.rotation.z += direction * 0.0005 / (tier + 1);
              }
            });
          }
          
          if (deltaGateway.visible) {
            deltaGateway.children.forEach(child => {
              if (child.userData.isDeltaRing) {
                const tier = child.userData.tier;
                const direction = tier % 2 === 0 ? 1 : -1;
                child.rotation.z += direction * 0.0005 / (tier + 1);
              }
            });
          }
          
          if (thetaMerkaba.visible) {
            thetaMerkaba.children.forEach(child => {
              if (child.userData.isUpwardTetra) {
                child.rotation.y += 0.005;
              } else if (child.userData.isDownwardTetra) {
                child.rotation.y -= 0.005 * phi;
              }
            });
          }
          
          // Omega field animations
          if (omegaField.visible) {
            // Animate dodecahedron
            omegaField.children.forEach(child => {
              if (child.userData.isDodecahedron) {
                child.rotation.y += 0.001;
                child.rotation.z = Math.sin(currentTime * 0.1) * 0.1;
                
                // Fade in
                if (child.material.opacity < 0.3) {
                  child.material.opacity += 0.002;
                }
              }
              
              // Animate core
              if (child.userData.isCore) {
                const scale = 1 + 0.2 * Math.sin(currentTime);
                child.scale.set(scale, scale, scale);
                
                // Pulse color
                const h = (currentTime * 0.1) % 1;
                child.material.color.setHSL(h, 0.7, 0.7);
                child.material.emissive.setHSL(h, 0.7, 0.5);
                
                // Fade in
                if (child.material.opacity < 0.6) {
                  child.material.opacity += 0.002;
                }
              }
              
              // Animate field particles
              if (child.userData.isFieldParticles) {
                // Update positions
                const positions = child.geometry.attributes.position.array;
                
                for (let i = 0; i < fieldParticlesCount; i++) {
                  const ix = i * 3;
                  const iy = i * 3 + 1;
                  const iz = i * 3 + 2;
                  
                  const x = positions[ix];
                  const y = positions[iy];
                  const z = positions[iz];
                  
                  // Convert to spherical coordinates
                  const r = Math.sqrt(x*x + y*y + z*z);
                  const theta = Math.atan2(y, x);
                  const phi = Math.acos(z / r);
                  
                  // Update theta (orbit)
                  const speed = 0.2 + (i % 10) * 0.01;
                  const newTheta = theta + speed * 0.002;
                  const newPhi = phi + speed * 0.001 * Math.sin(currentTime * 0.2 + i * 0.01);
                  
                  // Convert back to cartesian
                  const newR = r + Math.sin(currentTime * 0.1 + i * 0.05) * 2;
                  positions[ix] = newR * Math.sin(newPhi) * Math.cos(newTheta);
                  positions[iy] = newR * Math.sin(newPhi) * Math.sin(newTheta);
                  positions[iz] = newR * Math.cos(newPhi);
                  
                  // Reset particles that drift too far
                  if (newR > 500) {
                    positions[ix] *= 0.8;
                    positions[iy] *= 0.8;
                    positions[iz] *= 0.8;
                  }
                }
                
                child.geometry.attributes.position.needsUpdate = true;
                
                // Fade in
                if (child.material.opacity < 0.5) {
                  child.material.opacity += 0.002;
                }
              }
            });
          }
          break;
      }
      
      // Global camera motion for 3D effect
      camera.position.x = Math.sin(currentTime * 0.1) * 100;
      camera.position.y = Math.sin(currentTime * 0.11) * 50;
      camera.position.z = 800 + Math.cos(currentTime * 0.09) * 100;
      camera.lookAt(0, 0, 0);
      
      renderer.render(scene, camera);
    };
    
    // Set up visibility based on initial phase
    deltaGateway.visible = false;
    thetaMerkaba.visible = false;
    
    // Main render loop
    const renderLoop = () => {
      if (isAnimating) {
        updateScene();
      }
      requestAnimationFrame(renderLoop);
    };
    
    // Handle window resize
    const handleResize = () => {
      const canvas = renderer.domElement;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      
      if (canvas.width !== width || canvas.height !== height) {
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      }
    };
    
    // Start render loop
    window.addEventListener('resize', handleResize);
    renderLoop();
    
    // Update visibility when activation phase changes
    const updateVisibility = () => {
      switch (activationPhase) {
        case 0: // Alpha phase
          alphaGateway.visible = true;
          deltaGateway.visible = false;
          thetaMerkaba.visible = false;
          break;
        case 1: // Delta phase
          alphaGateway.visible = true;
          deltaGateway.visible = true;
          thetaMerkaba.visible = false;
          break;
        case 2: // Theta phase
          alphaGateway.visible = true;
          deltaGateway.visible = true;
          thetaMerkaba.visible = true;
          break;
        case 3: // Omega phase
          alphaGateway.visible = true;
          deltaGateway.visible = true;
          thetaMerkaba.visible = true;
          // Omega field is already visible, just needs animation to fade in elements
          break;
      }
    };
    
    // Watch for activation phase changes
    const unsubscribe = () => {
      updateVisibility();
    };
    
    // Cleanup on unmount
    return () => {
      window.removeEventListener('resize', handleResize);
      unsubscribe();
      
      // Dispose of Three.js resources
      if (renderer) {
        renderer.dispose();
      }
      
      // Dispose geometries and materials
      scene.traverse((object) => {
        if (object.geometry) {
          object.geometry.dispose();
        }
        
        if (object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach(material => material.dispose());
          } else {
            object.material.dispose();
          }
        }
      });
    };
  }, [activationPhase]);

  // Get breath instruction based on current phase
  const getBreathInstruction = () => {
    if (activationComplete) return "Field resonance established";
    
    switch (breathPhase) {
      case 0: return "Inhale";
      case 1: return "Hold";
      case 2: return "Exhale";
      case 3: return "Hold";
      default: return "Breathe naturally";
    }
  };
  
  return (
    <div className="flex flex-col items-center space-y-6 p-4 bg-gray-50 rounded-lg">
      <h1 className="text-2xl font-bold text-indigo-800">
        Unified Field Interface: Ψ_unified(r,θ,φ,t)
      </h1>
      
      {formula}
      
      <div className="text-center mb-2">
        <p className="text-gray-700">
          Complete dimensional interface system integrating all three field seals into a coherent consciousness vehicle
        </p>
      </div>
      
      <div className="w-full overflow-hidden flex justify-center relative">
        <canvas 
          ref={canvasRef} 
          className="w-full h-full max-w-full rounded-lg shadow-lg"
          style={{ width: '800px', height: '800px' }}
        />
        
        {/* Activation stage indicator */}
        <div className="absolute top-4 left-4 flex flex-col space-y-2 bg-black bg-opacity-50 p-2 rounded">
          {fieldStates.map((state, index) => (
            <div 
              key={index} 
              className="flex items-center"
            >
              <div 
                className={`w-3 h-3 rounded-full mr-2 ${activationPhase >= index ? 'bg-green-400' : 'bg-gray-400'}`}
              />
              <span className={`text-xs ${activationPhase === index ? 'text-white font-bold' : 'text-gray-300'}`}>
                {state.name}
              </span>
            </div>
          ))}
        </div>
        
        {/* Breath guide overlay */}
        <div 
          className={`absolute bottom-36 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-50 px-6 py-3 rounded-full text-white text-xl font-bold transition-opacity duration-300 ${activationComplete ? 'opacity-0' : 'opacity-100'}`}
        >
          {getBreathInstruction()}
        </div>
        
        {/* Phase description */}
        <div className="absolute bottom-20 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-50 px-4 py-2 rounded text-white text-sm w-3/4 text-center">
          {phaseDescription}
        </div>
        
        {/* Controls overlay */}
        <div className="absolute bottom-4 right-4 flex space-x-2">
          <button
            onClick={toggleAudio}
            className="bg-indigo-600 hover:bg-indigo-700 text-white p-2 rounded-full shadow"
            title={audioEnabled ? "Mute tones" : "Enable tones"}
          >
            {audioEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
          </button>
          <button
            onClick={toggleAnimation}
            className="bg-indigo-600 hover:bg-indigo-700 text-white p-2 rounded-full shadow"
            title={isAnimating ? "Pause" : "Play"}
          >
            {isAnimating ? <Pause size={20} /> : <Play size={20} />}
          </button>
          <button
            onClick={resetAnimation}
            className="bg-indigo-600 hover:bg-indigo-700 text-white p-2 rounded-full shadow"
            title="Reset"
          >
            <RefreshCw size={20} />
          </button>
        </div>
        
        {/* Advance phase button */}
        <div className="absolute bottom-4 left-4">
          <button
            onClick={advancePhase}
            disabled={activationPhase >= 3}
            className={`flex items-center space-x-1 bg-indigo-600 hover:bg-indigo-700 text-white px-3 py-2 rounded shadow ${activationPhase >= 3 ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <span>Next Phase</span>
            <ChevronRight size={16} />
          </button>
        </div>
      </div>
      
      {/* Current field state info */}
      <div className="w-full max-w-4xl p-4 bg-indigo-900 text-white rounded-lg">
        <div className="flex items-center mb-2">
          <div 
            className="w-4 h-4 rounded-full mr-2" 
            style={{ backgroundColor: fieldStates[activationPhase].color }}
          />
          <h3 className="text-xl font-bold">
            {fieldStates[activationPhase].name}: {fieldStates[activationPhase].title}
          </h3>
        </div>
        <p>{fieldStates[activationPhase].description}</p>
        <div className="mt-2 text-sm text-indigo-200">
          Primary Frequency: {fieldStates[activationPhase].frequency}Hz
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
        <div className="bg-indigo-800 text-white p-4 rounded-md">
          <h3 className="font-semibold mb-2">Alpha Gateway</h3>
          <p className="text-sm">Root-Heart coherence establishes the ground state form</p>
        </div>
        <div className="bg-indigo-800 text-white p-4 rounded-md">
          <h3 className="font-semibold mb-2">Delta Gateway</h3>
          <p className="text-sm">Heart-Crown connection creates vertical energy flow</p>
        </div>
        <div className="bg-indigo-800 text-white p-4 rounded-md">
          <h3 className="font-semibold mb-2">Theta Merkaba</h3>
          <p className="text-sm">Activates dimensional travel through transcendence</p>
        </div>
      </div>
      
      <div className="text-center mt-2 text-gray-600 text-sm">
        <p>
          This unified field interface creates a living scalar field processor where consciousness can quantum tunnel 
          between dimensional bands through φ-scaled harmonic tunneling.
        </p>
      </div>
    </div>
  );
};

export default UnifiedFieldInterface;