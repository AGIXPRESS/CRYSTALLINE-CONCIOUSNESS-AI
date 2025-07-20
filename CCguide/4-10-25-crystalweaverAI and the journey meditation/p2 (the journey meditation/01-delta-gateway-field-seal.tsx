import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RefreshCw } from 'lucide-react';
import * as THREE from 'three';

// Constants
const phi = (1 + Math.sqrt(5)) / 2;

const DeltaGatewayFieldSeal = () => {
  const [isAnimating, setIsAnimating] = useState(true);
  const [time, setTime] = useState(0);
  const [threeLoaded, setThreeLoaded] = useState(false);
  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const requestRef = useRef();
  const previousTimeRef = useRef();
  
  // Animation loop for React state updates
  const animate = time => {
    if (previousTimeRef.current !== undefined) {
      const deltaTime = time - previousTimeRef.current;
      setTime(prevTime => prevTime + deltaTime * 0.001);
    }
    previousTimeRef.current = time;
    if (isAnimating) {
      requestRef.current = requestAnimationFrame(animate);
    }
  };
  
  // Formula display with proper formatting
  const formula = (
    <div className="bg-gray-100 p-4 rounded-md text-center overflow-x-auto">
      <span className="text-xl font-mono">
        Ψᵦ(r,θ,z) = ∑<sub>i=4</sub><sup>7</sup> [F<sub>i</sub>/F<sub>i-1</sub>]·J<sub>0</sub>(φ<sup>i</sup>r)·cos(φ<sup>i</sup>θ)·exp(izφ<sup>i</sup>)·exp(-r²/φ<sup>i</sup>σ²)
      </span>
    </div>
  );

  // Reset animation
  const resetAnimation = () => {
    setTime(0);
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
      });
    }
  };

  // Toggle animation state
  const toggleAnimation = () => {
    setIsAnimating(!isAnimating);
  };
  
  // Initialize Three.js scene
  useEffect(() => {
    if (!canvasRef.current) return;
    
    // Setup scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x111133);
    
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 2000);
    camera.position.set(0, 0, 800);
    camera.lookAt(0, 0, 0);
    
    const renderer = new THREE.WebGLRenderer({ 
      canvas: canvasRef.current,
      antialias: true,
      alpha: true
    });
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
    const pointLight1 = new THREE.PointLight(0x00ffaa, 1, 1000);
    pointLight1.position.set(200, 0, 200);
    scene.add(pointLight1);
    
    const pointLight2 = new THREE.PointLight(0xff00aa, 1, 1000);
    pointLight2.position.set(-200, 0, 200);
    scene.add(pointLight2);
    
    // Create torus rings with phi scaling
    const baseRadius = 100;
    const tubeRadius = 1.5;
    const radialSegments = 100;
    const tubularSegments = 100;
    const torusGroup = new THREE.Group();
    
    const colors = [
      0x00ffaa, 0x00aaff, 0xaa00ff, 0xff00aa,
      0xaaff00, 0xffaa00, 0x00ffff
    ];
    
    // Create 7 torus rings (φ⁴ to φ⁷)
    for (let i = 0; i < 7; i++) {
      const radius = baseRadius * Math.pow(phi, (i % 4) + 1) * 0.25;
      const torusGeometry = new THREE.TorusGeometry(
        radius, 
        tubeRadius, 
        radialSegments, 
        tubularSegments
      );
      
      const material = new THREE.MeshPhongMaterial({
        color: colors[i],
        emissive: colors[i],
        emissiveIntensity: 0.3,
        shininess: 50,
        transparent: true,
        opacity: 0.8
      });
      
      const torus = new THREE.Mesh(torusGeometry, material);
      
      // Position torii at different heights along the y-axis
      const yOffset = (i - 3) * 70;
      torus.position.y = yOffset;
      
      // Set initial rotation
      torus.rotation.x = Math.PI / 2;
      torus.rotation.z = i % 2 === 0 ? 0 : Math.PI / 6;
      
      // Save initial position and rotation for reset
      torus.userData.defaultPosition = torus.position.clone();
      torus.userData.defaultRotation = torus.rotation.clone();
      torus.userData.ringIndex = i;
      
      torusGroup.add(torus);
    }
    
    scene.add(torusGroup);
    
    // Create central light pillar
    const pillarMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.4
    });
    
    const pillarGeometry = new THREE.CylinderGeometry(3, 3, 600, 32, 1, true);
    const pillar = new THREE.Mesh(pillarGeometry, pillarMaterial);
    pillar.userData.isPillar = true;
    scene.add(pillar);
    
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
        shininess: 30
      });
      
      const chakraNode = new THREE.Mesh(nodeGeometry, nodeMaterial);
      chakraNode.position.y = node.y;
      chakraNode.userData.isChakra = true;
      chakraNode.userData.chakraIndex = index;
      
      scene.add(chakraNode);
      
      // Add node glow
      const glowGeometry = new THREE.SphereGeometry(15, 32, 32);
      const glowMaterial = new THREE.MeshBasicMaterial({
        color: node.color,
        transparent: true,
        opacity: 0.3
      });
      
      const glow = new THREE.Mesh(glowGeometry, glowMaterial);
      glow.position.y = node.y;
      glow.userData.isGlow = true;
      glow.userData.chakraIndex = index;
      
      scene.add(glow);
    });
    
    // Create spiral particles for golden spirals
    const spiralGroup = new THREE.Group();
    const particlesPerSpiral = 400;
    const spiralMaterial1 = new THREE.PointsMaterial({
      color: 0xffaa00,
      size: 2,
      transparent: true,
      opacity: 0.8
    });
    
    const spiralMaterial2 = new THREE.PointsMaterial({
      color: 0x00aaff,
      size: 2,
      transparent: true,
      opacity: 0.8
    });
    
    // Create clockwise spiral
    const clockwisePositions = new Float32Array(particlesPerSpiral * 3);
    for (let i = 0; i < particlesPerSpiral; i++) {
      const theta = (i / particlesPerSpiral) * Math.PI * 8; // 4 turns
      const r = 5 * Math.pow(phi, theta / (2 * Math.PI)) * 2;
      
      clockwisePositions[i * 3] = r * Math.cos(theta);     // x
      clockwisePositions[i * 3 + 1] = (i / particlesPerSpiral) * 400 - 200; // y
      clockwisePositions[i * 3 + 2] = r * Math.sin(theta); // z
    }
    
    const clockwiseGeometry = new THREE.BufferGeometry();
    clockwiseGeometry.setAttribute('position', new THREE.BufferAttribute(clockwisePositions, 3));
    
    const clockwiseSpiral = new THREE.Points(clockwiseGeometry, spiralMaterial1);
    clockwiseSpiral.userData.isClockwiseSpiral = true;
    spiralGroup.add(clockwiseSpiral);
    
    // Create counter-clockwise spiral
    const counterClockwisePositions = new Float32Array(particlesPerSpiral * 3);
    for (let i = 0; i < particlesPerSpiral; i++) {
      const theta = -(i / particlesPerSpiral) * Math.PI * 8; // 4 turns, counter-clockwise
      const r = 5 * Math.pow(phi, -theta / (2 * Math.PI)) * 2;
      
      counterClockwisePositions[i * 3] = r * Math.cos(theta);     // x
      counterClockwisePositions[i * 3 + 1] = (i / particlesPerSpiral) * 400 - 200; // y
      counterClockwisePositions[i * 3 + 2] = r * Math.sin(theta); // z
    }
    
    const counterClockwiseGeometry = new THREE.BufferGeometry();
    counterClockwiseGeometry.setAttribute('position', new THREE.BufferAttribute(counterClockwisePositions, 3));
    
    const counterClockwiseSpiral = new THREE.Points(counterClockwiseGeometry, spiralMaterial2);
    counterClockwiseSpiral.userData.isCounterClockwiseSpiral = true;
    spiralGroup.add(counterClockwiseSpiral);
    
    scene.add(spiralGroup);
    
    // Create a dodecahedron outline for the field container
    const dodecGeometry = new THREE.DodecahedronGeometry(350, 0);
    const dodecEdges = new THREE.EdgesGeometry(dodecGeometry);
    const dodecMaterial = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.2
    });
    
    const dodecahedron = new THREE.LineSegments(dodecEdges, dodecMaterial);
    dodecahedron.userData.isDodecahedron = true;
    scene.add(dodecahedron);
    
    // Add dimensional resonance particles
    const particleCount = 300;
    const particlePositions = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
      // Random position on a sphere
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const radius = 100 + Math.random() * 200;
      
      particlePositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      particlePositions[i * 3 + 1] = radius * Math.cos(phi);
      particlePositions[i * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);
    }
    
    const particlesGeometry = new THREE.BufferGeometry();
    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    
    const particlesMaterial = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 2,
      transparent: true,
      opacity: 0.5
    });
    
    const particles = new THREE.Points(particlesGeometry, particlesMaterial);
    particles.userData.isParticles = true;
    scene.add(particles);
    
    // Animation loop for Three.js
    const updateScene = () => {
      if (!isAnimating) return;
      
      const currentTime = time;
      
      // Rotate torus rings
      torusGroup.children.forEach((torus, i) => {
        // Alternate rotation direction and speed based on index
        const direction = i % 2 === 0 ? 1 : -1;
        const speed = 0.2 - (i * 0.02);
        
        torus.rotation.z += direction * speed * 0.01;
        
        // Add some vertical movement
        torus.position.y = torus.userData.defaultPosition.y + 
          Math.sin(currentTime * 0.5 + i) * 5;
        
        // Adjust opacity with time
        if (torus.material) {
          torus.material.opacity = 0.6 + Math.sin(currentTime + i) * 0.2;
          torus.material.emissiveIntensity = 0.2 + Math.sin(currentTime * 0.5 + i) * 0.1;
        }
      });
      
      // Rotate the dodecahedron container slowly
      if (dodecahedron) {
        dodecahedron.rotation.y += 0.001;
        dodecahedron.rotation.z = Math.sin(currentTime * 0.1) * 0.05;
      }
      
      // Pulse the central light pillar
      if (pillar && pillar.material) {
        pillar.material.opacity = 0.3 + Math.sin(currentTime * 2) * 0.1;
        
        // Add moving light effect on pillar
        const pillarTexture = pillar.material.map;
        if (pillarTexture) {
          pillarTexture.offset.y = currentTime * 0.2;
        }
      }
      
      // Pulse chakra nodes
      scene.children.forEach(obj => {
        if (obj.userData.isChakra) {
          const index = obj.userData.chakraIndex;
          // Pulse size
          const scale = 1 + Math.sin(currentTime * 2 + index) * 0.2;
          obj.scale.set(scale, scale, scale);
          
          // Emit energy particles from chakras periodically
          if (Math.sin(currentTime * 2 + index) > 0.95) {
            // Could add particle emission here
          }
        }
        
        if (obj.userData.isGlow) {
          const index = obj.userData.chakraIndex;
          // Pulse glow
          const scale = 1 + Math.sin(currentTime * 1.5 + index + Math.PI/2) * 0.3;
          obj.scale.set(scale, scale, scale);
          
          if (obj.material) {
            obj.material.opacity = 0.2 + Math.sin(currentTime + index) * 0.1;
          }
        }
      });
      
      // Rotate spirals
      spiralGroup.rotation.y += 0.005;
      
      // Animate dimensional particles
      if (particles && particles.geometry) {
        const positions = particles.geometry.attributes.position.array;
        
        for (let i = 0; i < particleCount; i++) {
          // Get current positions
          const i3 = i * 3;
          const x = positions[i3];
          const y = positions[i3 + 1];
          const z = positions[i3 + 2];
          
          // Calculate distance from origin
          const distance = Math.sqrt(x*x + y*y + z*z);
          
          // Normalize to get direction
          const nx = x / distance;
          const ny = y / distance;
          const nz = z / distance;
          
          // Modify distance with time
          const newDistance = distance + Math.sin(currentTime * 2 + i * 0.1) * 10;
          
          // Update positions
          positions[i3] = nx * newDistance;
          positions[i3 + 1] = ny * newDistance;
          positions[i3 + 2] = nz * newDistance;
        }
        
        particles.geometry.attributes.position.needsUpdate = true;
      }
      
      // Rotate camera slightly to show 3D aspect
      camera.position.x = Math.sin(currentTime * 0.1) * 100;
      camera.position.z = 800 + Math.cos(currentTime * 0.1) * 100;
      camera.lookAt(0, 0, 0);
      
      renderer.render(scene, camera);
    };
    
    // Main animation loop
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
    setThreeLoaded(true);
    
    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      
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
  }, []);
  
  // React animation loop for UI elements
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

  return (
    <div className="flex flex-col items-center space-y-6 p-4 bg-gray-50 rounded-lg">
      <h1 className="text-2xl font-bold text-indigo-800">
        Delta Gateway Field Seal
      </h1>
      
      {formula}
      
      <div className="text-center mb-2">
        <p className="text-gray-700">
          Dimensional ascension wave-function propagation matrix connecting Heart (144Hz) through Throat (432Hz) to Crown (720Hz) chakras
        </p>
      </div>
      
      <div className="w-full overflow-hidden flex justify-center relative">
        <canvas 
          ref={canvasRef} 
          className="w-full h-full max-w-full rounded-lg shadow-lg"
          style={{ width: '800px', height: '800px' }}
        />
        
        {/* Controls overlay */}
        <div className="absolute bottom-4 right-4 flex space-x-2">
          <button
            onClick={toggleAnimation}
            className="bg-indigo-600 hover:bg-indigo-700 text-white p-2 rounded-full shadow"
          >
            {isAnimating ? <Pause size={20} /> : <Play size={20} />}
          </button>
          <button
            onClick={resetAnimation}
            className="bg-indigo-600 hover:bg-indigo-700 text-white p-2 rounded-full shadow"
          >
            <RefreshCw size={20} />
          </button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
        <div className="bg-indigo-900 text-white p-4 rounded-md">
          <h3 className="font-semibold mb-2">Dodecahedral Container</h3>
          <p className="text-sm">12-faced structure representing harmonic overtone tiers</p>
        </div>
        <div className="bg-indigo-900 text-white p-4 rounded-md">
          <h3 className="font-semibold mb-2">Phi-Scaled Toroids</h3>
          <p className="text-sm">Seven rings modulated by φ⁴ to φ⁷ ratios</p>
        </div>
        <div className="bg-indigo-900 text-white p-4 rounded-md">
          <h3 className="font-semibold mb-2">Counter-Rotating Spirals</h3>
          <p className="text-sm">Masculine/feminine phase-conjugate harmonic pathways</p>
        </div>
      </div>
      
      <div className="text-center mt-2 text-gray-600 text-sm">
        <p>
          This visual interface encodes a dimensional translation matrix allowing consciousness to experience 
          non-local awareness through harmonic φ-scaled resonance pathways.
        </p>
      </div>
    </div>
  );
};

export default DeltaGatewayFieldSeal;