import React, { useRef, useMemo, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Line, Html, Stars } from '@react-three/drei';
import * as THREE from 'three';

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────
interface MeshViewerProps {
  viewMode: 'rgb' | 'heatmap' | 'source';
  tierMode: 'tier1' | 'tier2';
  clipHeight: number;
  showTrajectory: boolean;
  onPointHover?: (info: HoverInfo | null) => void;
}

interface HoverInfo {
  x: number; y: number; z: number;
  confidence: number;
  type: string;
}

// ─────────────────────────────────────────────────────────────
// Synthetic data generator — realistic urban UAV scene
// ─────────────────────────────────────────────────────────────
function generateSceneData(tierMode: 'tier1' | 'tier2', clipHeight: number) {
  const numPoints = tierMode === 'tier1' ? 4000 : 38000;
  const pts: number[] = [];
  const rgbs: number[] = [];
  const heatmaps: number[] = [];
  const sources: number[] = [];
  const confs: number[] = [];

  // UAV trajectory — arc sweep over scene
  const trajectory: [number, number, number][] = [];
  for (let t = 0; t <= 1; t += 0.025) {
    const angle = t * Math.PI * 0.7 - Math.PI * 0.35;
    trajectory.push([
      16 * Math.cos(angle),
      16 * Math.sin(angle),
      11 + Math.sin(t * Math.PI) * 1.5,
    ]);
  }

  const rng = (a: number, b: number) => a + Math.random() * (b - a);
  const noise = (s: number) => (Math.random() - 0.5) * s;

  // ── Region definitions ──────────────────────────────────────
  const regions = [
    // Main building roof — high confidence MVS
    { weight: 0.18, gen: () => ({
      x: rng(-5, 5), y: rng(-4, 4), z: rng(4.0, 5.2),
      r: 0.55 + noise(0.05), g: 0.52 + noise(0.05), b: 0.50 + noise(0.05),
      hmr: 0.05, hmg: 0.75, hmb: 1.0, srr: 0.05, srg: 0.85, srb: 0.9,
      conf: 0.93, type: 'MVS Roof',
    })},
    // Building B (smaller annex)
    { weight: 0.08, gen: () => ({
      x: rng(7, 12), y: rng(-3, 3), z: rng(2.5, 3.5),
      r: 0.60 + noise(0.05), g: 0.55 + noise(0.05), b: 0.48 + noise(0.05),
      hmr: 0.05, hmg: 0.68, hmb: 0.95, srr: 0.05, srg: 0.80, srb: 0.90,
      conf: 0.89, type: 'MVS Roof B',
    })},
    // Building facades (walls) — lower confidence (single-pass limited angle)
    { weight: 0.12, gen: () => ({
      x: rng(-5.5, 5.5), y: rng(-4.5, 4.5), z: rng(0.0, 4.0),
      r: 0.62 + noise(0.08), g: 0.55 + noise(0.08), b: 0.45 + noise(0.08),
      hmr: 0.15, hmg: 0.80, hmb: 0.95, srr: 0.10, srg: 0.80, srb: 0.88,
      conf: 0.78, type: 'MVS Facade',
    })},
    // Road surface — monocular infill
    { weight: 0.22, gen: () => ({
      x: rng(-18, 18), y: rng(-16, 16), z: rng(-0.4, 0.2),
      r: 0.28 + noise(0.04), g: 0.28 + noise(0.04), b: 0.30 + noise(0.04),
      hmr: 0.95, hmg: 0.80, hmb: 0.10, srr: 0.90, srg: 0.70, srb: 0.15,
      conf: 0.62, type: 'Mono Infill Road',
    })},
    // Terrain / grass — monocular
    { weight: 0.18, gen: () => ({
      x: rng(-20, 20), y: rng(-18, 18), z: rng(-0.8, 0.5),
      r: 0.22 + noise(0.06), g: 0.45 + noise(0.06), b: 0.18 + noise(0.06),
      hmr: 0.90, hmg: 0.72, hmb: 0.08, srr: 0.88, srg: 0.65, srb: 0.12,
      conf: 0.60, type: 'Mono Infill Terrain',
    })},
    // Vegetation cluster — monocular
    { weight: 0.09, gen: () => ({
      x: rng(-14, -8) + noise(2), y: rng(6, 12) + noise(2), z: rng(0.5, 3.5),
      r: 0.18 + noise(0.05), g: 0.52 + noise(0.08), b: 0.12 + noise(0.04),
      hmr: 0.88, hmg: 0.72, hmb: 0.08, srr: 0.85, srg: 0.65, srb: 0.12,
      conf: 0.56, type: 'Mono Infill Vegetation',
    })},
    // Occluded wall base — 3DGS hallucinated
    { weight: 0.08, gen: () => ({
      x: rng(-5, 5), y: rng(-4, 4), z: rng(-2.5, -0.5),
      r: 0.72 + noise(0.08), g: 0.42 + noise(0.06), b: 0.25 + noise(0.06),
      hmr: 0.97, hmg: 0.15, hmb: 0.12, srr: 0.97, srg: 0.18, srb: 0.18,
      conf: 0.25, type: '3DGS Hallucinated',
    })},
    // Infrastructure (fences, poles) — hallucinated
    { weight: 0.05, gen: () => ({
      x: rng(-12, 12), y: rng(-12, 12), z: rng(1.5, 5.0),
      r: 0.65 + noise(0.08), g: 0.38 + noise(0.06), b: 0.22 + noise(0.06),
      hmr: 0.97, hmg: 0.18, hmb: 0.15, srr: 0.95, srg: 0.20, srb: 0.20,
      conf: 0.20, type: '3DGS Hallucinated',
    })},
  ];

  // Build cumulative weights
  const cumulative: number[] = [];
  let total = 0;
  for (const r of regions) { total += r.weight; cumulative.push(total); }

  for (let i = 0; i < numPoints; i++) {
    const rand = Math.random() * total;
    const regionIdx = cumulative.findIndex(c => rand <= c);
    const region = regions[Math.max(0, regionIdx)];
    const d = region.gen();

    if (d.z > clipHeight) continue;

    pts.push(d.x, d.y, d.z);
    rgbs.push(d.r, d.g, d.b);
    heatmaps.push(d.hmr, d.hmg, d.hmb);
    sources.push(d.srr, d.srg, d.srb);
    confs.push(d.conf);
  }

  return {
    points: new Float32Array(pts),
    rgbColors: new Float32Array(rgbs),
    heatmapColors: new Float32Array(heatmaps),
    sourceColors: new Float32Array(sources),
    confidences: new Float32Array(confs),
    trajectory,
  };
}

// ─────────────────────────────────────────────────────────────
// UAV drone mesh (simple geometric indicator)
// ─────────────────────────────────────────────────────────────
const UAVDrone: React.FC<{ trajectory: [number, number, number][] }> = ({ trajectory }) => {
  const droneRef = useRef<THREE.Mesh>(null);
  const progress = useRef(0);

  useFrame((_, delta) => {
    progress.current = (progress.current + delta * 0.04) % 1;
    const t = progress.current;
    const idx = Math.min(Math.floor(t * (trajectory.length - 1)), trajectory.length - 2);
    const alpha = (t * (trajectory.length - 1)) - idx;
    const p = trajectory[idx];
    const pn = trajectory[idx + 1];
    if (droneRef.current && p && pn) {
      droneRef.current.position.set(
        p[0] + (pn[0] - p[0]) * alpha,
        p[1] + (pn[1] - p[1]) * alpha,
        p[2] + (pn[2] - p[2]) * alpha,
      );
      droneRef.current.rotation.z = Math.sin(Date.now() * 0.003) * 0.05;
    }
  });

  return (
    <mesh ref={droneRef}>
      <boxGeometry args={[0.6, 0.6, 0.15]} />
      <meshStandardMaterial color="#38bdf8" emissive="#0ea5e9" emissiveIntensity={0.4} />
    </mesh>
  );
};

// ─────────────────────────────────────────────────────────────
// Main point cloud scene
// ─────────────────────────────────────────────────────────────
const PointCloudScene: React.FC<MeshViewerProps> = ({
  viewMode, tierMode, clipHeight, showTrajectory,
}) => {
  const pointsRef = useRef<THREE.Points>(null);

  const { points, rgbColors, heatmapColors, sourceColors, trajectory } = useMemo(
    () => generateSceneData(tierMode, clipHeight),
    [tierMode, clipHeight]
  );

  const activeColors = useMemo(() => {
    if (viewMode === 'heatmap') return heatmapColors;
    if (viewMode === 'source') return sourceColors;
    return rgbColors;
  }, [viewMode, rgbColors, heatmapColors, sourceColors]);

  const pointSize = tierMode === 'tier1' ? 0.22 : 0.07;

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[15, 25, 20]} intensity={1.0} castShadow />
      <pointLight position={[-10, -10, 15]} intensity={0.3} color="#38bdf8" />

      {/* Background stars */}
      <Stars radius={200} depth={60} count={1200} factor={2} saturation={0} fade speed={0.5} />

      {/* Main point cloud */}
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[points, 3]} />
          <bufferAttribute attach="attributes-color" args={[activeColors, 3]} />
        </bufferGeometry>
        <pointsMaterial
          size={pointSize}
          vertexColors
          sizeAttenuation
          transparent
          opacity={0.88}
          depthWrite={false}
        />
      </points>

      {/* Ground reference grid */}
      <gridHelper
        args={[40, 40, '#1e293b', '#0f172a']}
        position={[0, 0, -3]}
        rotation={[Math.PI / 2, 0, 0]}
      />

      {/* Building outline wireframes (reference geometry) */}
      <lineSegments position={[0, 0, 0]}>
        <edgesGeometry args={[new THREE.BoxGeometry(10, 8, 5.2)]} />
        <lineBasicMaterial color="#334155" transparent opacity={0.35} />
      </lineSegments>
      <lineSegments position={[9.5, 0, 0]}>
        <edgesGeometry args={[new THREE.BoxGeometry(5, 6, 3.5)]} />
        <lineBasicMaterial color="#334155" transparent opacity={0.30} />
      </lineSegments>

      {/* UAV trajectory path */}
      {showTrajectory && trajectory.length > 1 && (
        <>
          <Line
            points={trajectory}
            color="#38bdf8"
            lineWidth={2}
            dashed={false}
            transparent
            opacity={0.7}
          />
          <UAVDrone trajectory={trajectory} />
        </>
      )}

      {/* Scan effect ring at flight altitude */}
      {showTrajectory && (
        <mesh position={[0, 0, 10.5]} rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[13.5, 14.5, 64]} />
          <meshBasicMaterial color="#38bdf8" transparent opacity={0.08} side={THREE.DoubleSide} />
        </mesh>
      )}

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.06}
        minDistance={4}
        maxDistance={80}
      />
    </>
  );
};

// ─────────────────────────────────────────────────────────────
// Public export
// ─────────────────────────────────────────────────────────────
export const MeshViewer: React.FC<MeshViewerProps> = (props) => {
  return (
    <Canvas
      camera={{ position: [22, -18, 14], fov: 42, near: 0.1, far: 500 }}
      gl={{ antialias: true, alpha: false }}
      style={{ width: '100%', height: '100%' }}
    >
      <PointCloudScene {...props} />
    </Canvas>
  );
};
