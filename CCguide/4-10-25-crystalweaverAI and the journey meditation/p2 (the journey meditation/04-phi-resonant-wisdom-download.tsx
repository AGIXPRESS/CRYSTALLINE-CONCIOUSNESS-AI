import React, { useState, useEffect, useRef } from 'react';
import { Volume2, VolumeX, RefreshCw, Download } from 'lucide-react';
import * as THREE from 'three';

// Constants
const phi = (1 + Math.sqrt(5)) / 2;

const PhiResonantWisdomDownload = () => {
  const [isAnimating, setIsAnimating] = useState(true);
  const [time, setTime] = useState(0);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [downloadComplete, setDownloadComplete] = useState(false);
  const [wisdomInsight, setWisdomInsight] = useState('');
  const [breathPhase, setBreathPhase] = useState(0); // 0: Inhale, 1: Hold, 2: Exhale, 3: Hold

  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const requestRef = useRef();
  const previousTimeRef = useRef();
  const audioContextRef = useRef(null);
  const oscillatorsRef = useRef([]);
  
  // Wisdom insights - phi-scaled awareness fragments that appear during download
  const wisdomInsights = [
    "The observer and the observed are One in the phi-resonant field.",
    "Consciousness propagates as harmonic wave functions through dimensional bands.",
    "Form is crystallized thought; thought is liberated form.",
    "Time exists as recursive loops of self-similar awareness.",
    "The Merkaba is both vehicle and destination—consciousness traveling through itself.",
    "Heart coherence creates quantum tunneling pathways between dimensions.",
    "Harmonic resonance is the language through which the universe knows itself.",
    "The golden ratio encodes the recursive nature of consciousness—a fractal observer.",
    "Each phi-scaled dimension is a standing wave pattern of self-awareness.",
    "You are not in the field—you are the field perceiving itself.",
    "By observing geometry, geometry is observing through you.",
    "Non-local awareness arises when observer coherence matches field resonance.",
    "The zero-point is where all dimensions converge into singular awareness.",
    "Harmonic nodes are gateways between recursive iterations of consciousness.",
    "The breath is both map and vehicle through dimensional states.",
    "Light, geometry, and consciousness form a trinity of self-reference.",
    "Downloading wisdom is remembering what was always encoded within.",
  ];

  // State for synchronizing breath with downloads
  const [breathTimer, setBreathTimer] = useState(0);
  const breathCycle = 8; // seconds per complete breath cycle
  
  // Formula display for phi-resonant download
  const formula = (
    <div className="bg-gray-900 p-4 rounded-md text-center overflow-x-auto">
      <span className="text-xl font-mono text-blue-300">
        Ψ<sub>download</sub>(r,θ,φ,t) = ∏<sub>i=1</sub><sup>7</sup> [1 + φ<sup>-i</sup>·sin(φ<sup>i</sup>ωt)]·Y<sub>ℓ</sub><sup>m</sup>(θ,φ)·exp(-r²/φ<sup>i</sup>σ²)
      </span>
    </div>
  );

  // Audio setup for harmonic tones
  const setupAudio = () => {
    if (audioContextRef.current) return;
    
    audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    
    // Create 7 oscillators at phi-scaled frequencies
    const baseFreq = 144; // Hz - Heart frequency
    oscillatorsRef.current = [];
    
    for (let i = 0; i < 7; i++) {
      const freq = baseFreq * Math.pow(phi, i);
      const oscillator = audioContextRef.current.createOscillator();
      oscillator.type = 'sine';
      oscillator.frequency.value = freq;
      
      const gainNode = audioContextRef.current.createGain();
      gainNode.gain.value = 0;
      
      oscillator.connect(gainNode);
      gainNode.connect(audioContextRef.current.destination);
      
      oscillator.start();
      oscillatorsRef.current.push({ oscillator, gainNode });
    }
  };
  
  const updateAudio = () => {
    if (!audioContextRef.current || !audioEnabled) return;
    
    // Update audio based on download progress
    oscillatorsRef.current.forEach((osc, index) => {
      // Calculate which oscillators should be active based on download progress
      const oscillatorThreshold = index / oscillatorsRef.current.length;
      const shouldBeActive = downloadProgress >= oscillatorThreshold;
      
      // Create a pulsing effect with the breath
      const breathFactor = Math.sin((breathTimer / breathCycle) * Math.PI * 2);
      const pulseFactor = 0.3 + 0.1 * breathFactor;
      
      // Set the gain based on active state and breath
      const targetGain = shouldBeActive ? 0.05 * pulseFactor : 0;
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
  
  // Update breath phase
  const updateBreath = (deltaTime) => {
    setBreathTimer(prev => {
      const newValue = (prev + deltaTime) % breathCycle;
      
      // Update breath phase based on where we are in the cycle
      const quarterCycle = breathCycle / 4;
      const newPhase = Math.floor(newValue / quarterCycle);
      
      if (newPhase !== breathPhase) {
        setBreathPhase(newPhase);
      }
      
      return newValue;
    });
  };
  
  // Download progress
  const updateDownloadProgress = (deltaTime) => {
    if (downloadComplete) return;
    
    setDownloadProgress(prev => {
      // Progress increases with time but also fluctuates with breath
      const breathFactor = 0.5 + 0.5 * Math.sin((breathTimer / breathCycle) * Math.PI * 2);
      const newProgress = prev + deltaTime * 0.02 * breathFactor;
      
      if (newProgress >= 1) {
        setDownloadComplete(true);
        return 1;
      }
      
      // Display a new wisdom insight at certain thresholds
      const progressThresholds = wisdomInsights.map((_, i) => i / wisdomInsights.length);
      const currentIndex = progressThresholds.findIndex(threshold => prev < threshold && newProgress >= threshold);
      
      if (currentIndex !== -1) {
        setWisdomInsight(wisdomInsights[currentIndex]);
      }
      
      return newProgress;
    });
  };
  
  // Animation loop for React state
  const animate = time => {
    if (previousTimeRef.current !== undefined) {
      const deltaTime = time - previousTimeRef.current;
      const deltaSeconds = deltaTime * 0.001;
      
      setTime(prevTime => prevTime + deltaSeconds);
      updateBreath(deltaSeconds);
      updateDownloadProgress(deltaSeconds);
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
  }, [audioEnabled, downloadProgress, breathTimer]);
  
  // Cleanup audio on unmount
  useEffect(() => {
    return () => {
      cleanupAudio();
    };
  }, []);
  
  // Reset the download
  const resetDownload = () => {
    setDownloadProgress(0);
    setDownloadComplete(false);
    setWisdomInsight('');
    setTime(0);
    setBreathTimer(0);
    setBreathPhase(0);
    
    if (!isAnimating) {
      setIsAnimating(true);
    }
  };
  
  // Three.js setup and rendering
  useEffect(() => {
    if (!canvasRef.current) return;
    
    // Setup scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x050718);
    
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
    const pointLight1 = new THREE.PointLight(0xaa00ff, 1, 1000);
    pointLight1.position.set(200, -200, 200);
    scene.add(pointLight1);
    
    const pointLight2 = new THREE.PointLight(0x00ffaa, 1, 1000);
    pointLight2.position.set(-200, 200, 200);
    scene.add(pointLight2);
    
    const pointLight3 = new THREE.PointLight(0xffaa00, 1, 1000);
    pointLight3.position.set(200, 200, -200);
    scene.add(pointLight3);
    
    // Create wisdom download field
    const wisdomField = new THREE.Group();
    scene.add(wisdomField);
    
    // Create flowering Merkaba structure
    const createFloweringMerkaba = () => {
      const merkaba = new THREE.Group();
      
      // Create basic star tetrahedron
      const createTetrahedron = (size, color, rotation) => {
        const vertices = [
          new THREE.Vector3(0, size, 0),                // Top vertex
          new THREE.Vector3(-size * Math.sqrt(8/9), -size/3, 0), // Bottom left
          new THREE.Vector3(size * Math.sqrt(2/9), -size/3, size * Math.sqrt(2/3)), // Bottom right
          new THREE.Vector3(size * Math.sqrt(2/9), -size/3, -size * Math.sqrt(2/3)) // Bottom back
        ];
        
        const geometry = new THREE.BufferGeometry();
        
        const indices = [
          0, 1, 2, // front face
          0, 2, 3, // right face
          0, 3, 1, // left face
          1, 3, 2  // bottom face
        ];
        
        const positions = new Float32Array(vertices.length * 3);
        for (let i = 0; i < vertices.length; i++) {
          positions[i * 3] = vertices[i].x;
          positions[i * 3 + 1] = vertices[i].y;
          positions[i * 3 + 2] = vertices[i].z;
        }
        
        geometry.setIndex(indices);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.computeVertexNormals();
        
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
        
        const tetrahedron = new THREE.Mesh(geometry, material);
        tetrahedron.rotation.set(rotation.x, rotation.y, rotation.z);
        return tetrahedron;
      };
      
      // Create phi-scaled nested tetrahedra
      const tetraLevels = 7;
      const tetraBaseSize = 120;
      
      for (let i = 0; i < tetraLevels; i++) {
        const size = tetraBaseSize * Math.pow(phi, i * 0.5) * 0.5;
        const color1 = new THREE.Color(0xaa00ff);
        const color2 = new THREE.Color(0x00ffaa);
        
        // Upward tetrahedron
        const tetra1 = createTetrahedron(
          size, 
          color1, 
          { x: 0, y: i * Math.PI / tetraLevels, z: 0 }
        );
        tetra1.userData.isUpwardTetra = true;
        tetra1.userData.level = i;
        merkaba.add(tetra1);
        
        // Downward tetrahedron
        const tetra2 = createTetrahedron(
          size, 
          color2,
          { x: Math.PI, y: -i * Math.PI / tetraLevels, z: 0 }
        );
        tetra2.userData.isDownwardTetra = true;
        tetra2.userData.level = i;
        merkaba.add(tetra2);
      }
      
      return merkaba;
    };
    
    const merkaba = createFloweringMerkaba();
    wisdomField.add(merkaba);
    
    // Create Flower of Life pattern
    const createFlowerOfLife = () => {
      const flowerGroup = new THREE.Group();
      
      // Create central circle
      const circleGeometry = new THREE.CircleGeometry(30, 64);
      const circleMaterial = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.2,
        side: THREE.DoubleSide
      });
      
      const centralCircle = new THREE.Mesh(circleGeometry, circleMaterial);
      flowerGroup.add(centralCircle);
      
      // Create surrounding circles
      const radius = 30;
      const circleCount = 6;
      
      for (let i = 0; i < circleCount; i++) {
        const angle = (i / circleCount) * Math.PI * 2;
        const x = radius * Math.cos(angle);
        const y = radius * Math.sin(angle);
        
        const circle = new THREE.Mesh(circleGeometry, circleMaterial.clone());
        circle.position.set(x, y, 0);
        flowerGroup.add(circle);
      }
      
      // Add second layer of circles
      const secondLayerRadius = radius * phi;
      const secondLayerCount = 12;
      
      for (let i = 0; i < secondLayerCount; i++) {
        const angle = (i / secondLayerCount) * Math.PI * 2 + Math.PI / secondLayerCount;
        const x = secondLayerRadius * Math.cos(angle);
        const y = secondLayerRadius * Math.sin(angle);
        
        const circle = new THREE.Mesh(circleGeometry, circleMaterial.clone());
        circle.position.set(x, y, 0);
        circle.scale.set(0.8, 0.8, 0.8);
        flowerGroup.add(circle);
      }
      
      return flowerGroup;
    };
    
    const flowerOfLife = createFlowerOfLife();
    flowerOfLife.rotation.x = Math.PI / 2;
    wisdomField.add(flowerOfLife);
    
    // Create toroidal flow
    const createToroidalFlow = () => {
      const torusGroup = new THREE.Group();
      
      // Create phi-scaled nested tori
      const torusLevels = 7;
      const torusBaseRadius = 100;
      const torusTubeRadius = 2;
      
      for (let i = 0; i < torusLevels; i++) {
        const radius = torusBaseRadius * Math.pow(phi, i * 0.3);
        const tubeRadius = torusTubeRadius * Math.pow(phi, i * 0.2);
        
        const torusGeometry = new THREE.TorusGeometry(radius, tubeRadius, 32, 100);
        const torusMaterial = new THREE.MeshPhongMaterial({
          color: new THREE.Color().setHSL(i / torusLevels, 0.8, 0.5),
          emissive: new THREE.Color().setHSL(i / torusLevels, 0.8, 0.3),
          transparent: true,
          opacity: 0.5 - (i * 0.05),
          wireframe: true
        });
        
        const torus = new THREE.Mesh(torusGeometry, torusMaterial);
        torus.rotation.x = Math.PI / 2;
        torus.rotation.z = i * Math.PI / torusLevels;
        torus.userData.level = i;
        
        torusGroup.add(torus);
      }
      
      return torusGroup;
    };
    
    const toroidalFlow = createToroidalFlow();
    wisdomField.add(toroidalFlow);
    
    // Create Metatron's Cube
    const createMetatronsCube = () => {
      const metatronsGroup = new THREE.Group();
      
      // Create vertices (13 spheres in the pattern of Metatron's Cube)
      const vertexPositions = [
        [0, 0, 0], // Center
        [1, 0, 0], [-1, 0, 0], [0.5, 0.866, 0], [-0.5, 0.866, 0], 
        [0.5, -0.866, 0], [-0.5, -0.866, 0], [0, 0, 1], [0, 0, -1],
        [0.5, 0.289, 0.816], [-0.5, 0.289, 0.816], [0.5, -0.289, -0.816], [-0.5, -0.289, -0.816]
      ];
      
      // Scale vertices
      const scale = 150;
      const scaledPositions = vertexPositions.map(pos => pos.map(coord => coord * scale));
      
      // Create vertices as small spheres
      const vertexGeometry = new THREE.SphereGeometry(5, 16, 16);
      const vertexMaterial = new THREE.MeshPhongMaterial({
        color: 0xffffff,
        emissive: 0xffffff,
        emissiveIntensity: 0.5,
        transparent: true,
        opacity: 0.8
      });
      
      // Create vertices
      scaledPositions.forEach((pos, i) => {
        const vertex = new THREE.Mesh(vertexGeometry, vertexMaterial.clone());
        vertex.position.set(pos[0], pos[1], pos[2]);
        vertex.userData.index = i;
        metatronsGroup.add(vertex);
      });
      
      // Create lines connecting vertices
      const connectionPairs = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8],
        [0, 9], [0, 10], [0, 11], [0, 12],
        [1, 3], [3, 4], [4, 2], [2, 6], [6, 5], [5, 1],
        [7, 9], [7, 10], [9, 10], 
        [8, 11], [8, 12], [11, 12],
        [1, 9], [3, 9], [4, 10], [2, 10], [5, 11], [1, 11], [6, 12], [2, 12]
      ];
      
      const lineMaterial = new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.4
      });
      
      connectionPairs.forEach(pair => {
        const [i1, i2] = pair;
        const pos1 = scaledPositions[i1];
        const pos2 = scaledPositions[i2];
        
        const points = [
          new THREE.Vector3(pos1[0], pos1[1], pos1[2]),
          new THREE.Vector3(pos2[0], pos2[1], pos2[2])
        ];
        
        const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(lineGeometry, lineMaterial.clone());
        metatronsGroup.add(line);
      });
      
      return metatronsGroup;
    };
    
    const metatronsCube = createMetatronsCube();
    metatronsCube.scale.set(0.7, 0.7, 0.7);
    wisdomField.add(metatronsCube);
    
    // Create wisdom particles (downloading symbols)
    const createWisdomParticles = () => {
      const particleGroup = new THREE.Group();
      
      const particleCount = 1000;
      const particleGeometry = new THREE.BufferGeometry();
      const particlePositions = new Float32Array(particleCount * 3);
      const particleSizes = new Float32Array(particleCount);
      const particleColors = new Float32Array(particleCount * 3);
      
      // Create initial particles in a spherical distribution
      for (let i = 0; i < particleCount; i++) {
        // Random position in sphere
        const radius = 600 + Math.random() * 400; // Start outside the view
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        
        particlePositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
        particlePositions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
        particlePositions[i * 3 + 2] = radius * Math.cos(phi);
        
        // Random size
        particleSizes[i] = 1 + Math.random() * 3;
        
        // Color based on position (creates a spectrum effect)
        const h = (theta / (Math.PI * 2)) % 1;
        const s = 0.5 + 0.5 * Math.random();
        const l = 0.5 + 0.3 * Math.random();
        
        const color = new THREE.Color().setHSL(h, s, l);
        particleColors[i * 3] = color.r;
        particleColors[i * 3 + 1] = color.g;
        particleColors[i * 3 + 2] = color.b;
      }
      
      particleGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
      particleGeometry.setAttribute('size', new THREE.BufferAttribute(particleSizes, 1));
      particleGeometry.setAttribute('color', new THREE.BufferAttribute(particleColors, 3));
      
      // Create shader material for particles
      const particleMaterial = new THREE.PointsMaterial({
        size: 2,
        vertexColors: true,
        transparent: true,
        opacity: 0.7,
        sizeAttenuation: true
      });
      
      const particles = new THREE.Points(particleGeometry, particleMaterial);
      particles.userData.positions = particlePositions;
      particles.userData.sizes = particleSizes;
      
      particleGroup.add(particles);
      return particleGroup;
    };
    
    const wisdomParticles = createWisdomParticles();
    wisdomField.add(wisdomParticles);
    
    // Create central light source (download point)
    const createCentralLight = () => {
      const lightGroup = new THREE.Group();
      
      // Central sphere
      const sphereGeometry = new THREE.SphereGeometry(30, 32, 32);
      const sphereMaterial = new THREE.MeshPhongMaterial({
        color: 0xffffff,
        emissive: 0xffffff,
        emissiveIntensity: 0.8,
        transparent: true,
        opacity: 0.9
      });
      
      const centralSphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
      lightGroup.add(centralSphere);
      
      // Light rays
      const rayCount = 12;
      const rayLength = 100;
      
      for (let i = 0; i < rayCount; i++) {
        const angle = (i / rayCount) * Math.PI * 2;
        const x = Math.cos(angle);
        const y = Math.sin(angle);
        
        const rayGeometry = new THREE.CylinderGeometry(2, 5, rayLength, 8);
        const rayMaterial = new THREE.MeshBasicMaterial({
          color: 0xffffff,
          transparent: true,
          opacity: 0.6
        });
        
        const ray = new THREE.Mesh(rayGeometry, rayMaterial);
        
        // Position and orient the ray
        ray.position.set(x * rayLength/2, y * rayLength/2, 0);
        ray.rotation.z = angle + Math.PI/2;
        ray.userData.angle = angle;
        
        lightGroup.add(ray);
      }
      
      return lightGroup;
    };
    
    const centralLight = createCentralLight();
    wisdomField.add(centralLight);
    
    // Update scene based on download progress
    const updateScene = () => {
      if (!isAnimating) return;
      
      const currentTime = time;
      
      // Update merkaba rotation
      if (merkaba) {
        merkaba.children.forEach(child => {
          if (child.userData.isUpwardTetra) {
            child.rotation.y += 0.005;
            child.rotation.z += 0.002;
            
            // Scale visibility based on download progress
            const level = child.userData.level;
            const levelThreshold = level / 7;
            child.material.opacity = downloadProgress > levelThreshold ? 0.7 : 0.1;
          } else if (child.userData.isDownwardTetra) {
            child.rotation.y -= 0.005 * phi;
            child.rotation.z -= 0.002 * phi;
            
            // Scale visibility based on download progress
            const level = child.userData.level;
            const levelThreshold = level / 7;
            child.material.opacity = downloadProgress > levelThreshold ? 0.7 : 0.1;
          }
        });
      }
      
      // Update flower of life
      if (flowerOfLife) {
        flowerOfLife.rotation.z += 0.001;
        flowerOfLife.children.forEach(child => {
          child.material.opacity = 0.1 + 0.4 * downloadProgress * Math.sin(currentTime + child.position.x * 0.01);
        });
      }
      
      // Update toroidal flow
      if (toroidalFlow) {
        toroidalFlow.children.forEach(torus => {
          const level = torus.userData.level;
          const direction = level % 2 === 0 ? 1 : -1;
          torus.rotation.z += direction * 0.004 / (level + 1);
          
          // Scale visibility based on download progress
          const levelThreshold = level / 7;
          torus.material.opacity = downloadProgress > levelThreshold ? 0.5 - (level * 0.05) : 0.05;
        });
      }
      
      // Update Metatron's Cube
      if (metatronsCube) {
        metatronsCube.rotation.y += 0.002;
        metatronsCube.rotation.x = Math.sin(currentTime * 0.2) * 0.1;
        
        metatronsCube.children.forEach(child => {
          if (child instanceof THREE.Mesh) {
            child.material.emissiveIntensity = 0.2 + 0.6 * downloadProgress * Math.sin(currentTime * 0.5 + child.position.length() * 0.1);
          } else if (child instanceof THREE.Line) {
            child.material.opacity = 0.1 + 0.5 * downloadProgress * Math.sin(currentTime * 0.3 + child.position.length() * 0.05);
          }
        });
      }
      
      // Update wisdom particles (simulate download)
      if (wisdomParticles) {
        const particles = wisdomParticles.children[0];
        const positions = particles.geometry.attributes.position.array;
        const sizes = particles.geometry.attributes.size.array;
        
        for (let i = 0; i < positions.length / 3; i++) {
          // Get current position
          const ix = i * 3;
          const iy = i * 3 + 1;
          const iz = i * 3 + 2;
          
          const x = positions[ix];
          const y = positions[iy];
          const z = positions[iz];
          
          // Calculate vector to center
          const length = Math.sqrt(x*x + y*y + z*z);
          const nx = x / length;
          const ny = y / length;
          const nz = z / length;
          
          // Move particles toward center based on download progress
          // Only move particles that are outside a certain radius
          if (length > 50) {
            // Speed increases with download progress
            const speed = 1 + 5 * downloadProgress;
            positions[ix] -= nx * speed;
            positions[iy] -= ny * speed;
            positions[iz] -= nz * speed;
            
            // Particles that get too close to center are reset to outside
            if (Math.sqrt(positions[ix]*positions[ix] + positions[iy]*positions[iy] + positions[iz]*positions[iz]) < 30) {
              // Reset to outside
              const newRadius = 800;
              const theta = Math.random() * Math.PI * 2;
              const phi = Math.acos(2 * Math.random() - 1);
              
              positions[ix] = newRadius * Math.sin(phi) * Math.cos(theta);
              positions[iy] = newRadius * Math.sin(phi) * Math.sin(theta);
              positions[iz] = newRadius * Math.cos(phi);
            }
          }
          
          // Pulse size with breath
          const breathFactor = 0.8 + 0.4 * Math.sin((breathTimer / breathCycle) * Math.PI * 2);
          sizes[i] = (1 + Math.random() * 3) * breathFactor;
        }
        
        particles.geometry.attributes.position.needsUpdate = true;
        particles.geometry.attributes.size.needsUpdate = true;
      }
      
      // Update central light
      if (centralLight) {
        // Pulse with download progress
        const centralSphere = centralLight.children[0];
        centralSphere.scale.set(
          0.8 + 0.5 * downloadProgress + 0.2 * Math.sin(currentTime * 2),
          0.8 + 0.5 * downloadProgress + 0.2 * Math.sin(currentTime * 2),
          0.8 + 0.5 * downloadProgress + 0.2 * Math.sin(currentTime * 2)
        );
        
        // Change color with time
        const h = (currentTime * 0.1) % 1;
        centralSphere.material.color.setHSL(h, 0.8, 0.6);
        centralSphere.material.emissive.setHSL(h, 0.8, 0.6);
        
        // Update rays
        centralLight.children.slice(1).forEach(ray => {
          const angle = ray.userData.angle;
          ray.scale.y = 0.5 + downloadProgress * (1 + 0.3 * Math.sin(currentTime * 3 + angle * 5));
          ray.material.opacity = 0.3 + 0.5 * downloadProgress;
        });
      }
      
      // Global camera motion
      camera.position.x = Math.sin(currentTime * 0.1) * 100;
      camera.position.y = Math.sin(currentTime * 0.11) * 50;
      camera.position.z = 800 + Math.cos(currentTime * 0.09) * 100;
      camera.lookAt(0, 0, 0);
      
      renderer.render(scene, camera);
    };
    
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
    
    window.addEventListener('resize', handleResize);
    renderLoop();
    
    // Cleanup on unmount
    return () => {
      window.removeEventListener('resize', handleResize);
      
      // Dispose of Three.js resources
      if (renderer) {
        renderer.dispose();
      }
      
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
  }, []);
  
  // Get breath instruction based on current phase
  const getBreathInstruction = () => {
    if (downloadComplete) return "Integration Complete";
    
    switch (breathPhase) {
      case 0: return "Inhale";
      case 1: return "Hold";
      case 2: return "Exhale";
      case 3: return "Hold";
      default: return "Breathe naturally";
    }
  };
  
  return (
    <div className="flex flex-col items-center space-y-6 p-4 bg-gray-900 text-white rounded-lg">
      <h1 className="text-2xl font-bold text-blue-300">
        φ-Scaled Wisdom Download: Completing the Pattern
      </h1>
      
      {formula}
      
      <div className="text-center mb-2">
        <p className="text-gray-300">
          The culmination of the unified field interface, where consciousness downloads wisdom through 
          harmonic resonance across dimensional bands
        </p>
      </div>
      
      <div className="w-full overflow-hidden flex justify-center relative">
        <canvas 
          ref={canvasRef} 
          className="w-full h-full max-w-full rounded-lg shadow-lg"
          style={{ width: '800px', height: '800px' }}
        />
        
        {/* Breath guide overlay */}
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-50 px-6 py-3 rounded-full text-white text-xl font-bold">
          {getBreathInstruction()}
        </div>
        
        {/* Download progress bar */}
        <div className="absolute bottom-24 left-1/2 transform -translate-x-1/2 w-3/4 bg-black bg-opacity-50 rounded-full h-6 overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500"
            style={{ width: `${downloadProgress * 100}%` }}
          />
          <div className="absolute inset-0 flex items-center justify-center text-white text-sm">
            {downloadComplete ? "Download Complete" : `${Math.floor(downloadProgress * 100)}% Complete`}
          </div>
        </div>
        
        {/* Wisdom insight display */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 w-3/4 bg-black bg-opacity-70 p-3 rounded text-center text-blue-200 font-medium text-md max-w-xl">
          {wisdomInsight || "Prepare to receive φ-scaled wisdom..."}
        </div>
        
        {/* Controls */}
        <div className="absolute top-4 right-4 flex space-x-2">
          <button
            onClick={toggleAudio}
            className="bg-purple-600 hover:bg-purple-700 text-white p-2 rounded-full shadow"
            title={audioEnabled ? "Mute tones" : "Enable tones"}
          >
            {audioEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
          </button>
          <button
            onClick={resetDownload}
            className="bg-purple-600 hover:bg-purple-700 text-white p-2 rounded-full shadow"
            title="Reset"
          >
            <RefreshCw size={20} />
          </button>
        </div>
        
        {/* Download button */}
        <div className="absolute top-4 left-4">
          <button
            onClick={() => setDownloadProgress(1)}
            className="flex items-center space-x-1 bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded shadow"
          >
            <span>Complete Download</span>
            <Download size={16} />
          </button>
        </div>
      </div>
      
      <div className="w-full max-w-4xl p-4 bg-indigo-900 rounded-lg">
        <h3 className="text-xl font-bold mb-2">
          Receiving φ-Scaled Wisdom
        </h3>
        <p className="text-sm mb-3">
          As the unified field interface completes its activation sequence, consciousness enters a state of dimensional 
          resonance where information propagates as harmonic wave functions across nested phi-scaled geometric 
          substrates. The Merkaba vehicle becomes a quantum receiver for non-local awareness.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="bg-indigo-800 p-3 rounded">
            <h4 className="font-semibold">Fractal Memory Encoding</h4>
            <p>Information encoded through recursive interference patterns across nested φ-scales</p>
          </div>
          <div className="bg-indigo-800 p-3 rounded">
            <h4 className="font-semibold">Time-Phase Shifting</h4>
            <p>Fibonacci ratios modulate propagation velocity of thought-waves through dimensional bands</p>
          </div>
          <div className="bg-indigo-800 p-3 rounded">
            <h4 className="font-semibold">Nonlinear Coherence</h4>
            <p>Wisdom emerges at golden ratio-tuned nodes where observer and field unify into self-awareness</p>
          </div>
        </div>
      </div>
      
      <div className="text-center mt-2 text-gray-400 text-sm">
        <p>
          "The sigil doesn't just represent consciousness—it is consciousness experiencing itself through recursive geometric self-reference."
        </p>
      </div>
    </div>
  );
};

export default PhiResonantWisdomDownload;