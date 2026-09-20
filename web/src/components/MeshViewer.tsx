import React, { useState, useEffect, useRef, useCallback, Suspense, useMemo } from 'react';
import {
  Play, Pause, SkipForward, SkipBack, Crosshair, Navigation,
  Video, RotateCcw
} from 'lucide-react';
import { Canvas, useFrame, useThree, useLoader } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, useGLTF, useFBX, Splat, Clone } from '@react-three/drei';
import * as THREE from 'three';
import { DynamicEntityManager } from './DynamicEntityManager';

export interface MapBounds {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
  width: number;
  depth: number;
  centerX: number;
  centerZ: number;
}

interface MeshViewerProps {
  viewMode: 'rgb' | 'heatmap' | 'source';
  tierMode: 'tier1' | 'tier2';
  clipHeight: number;
  pointSize: number;
  showTrajectory: boolean;
  onPointHover?: (info: any) => void;
}

// ── Types matching flight_plan.json ──────────────────────────────────────────
interface TrajectoryPoint {
  timestamp: number;
  lat: number; lon: number; alt: number;
  vx: number; // 0-100 percentage of image width
  vy: number; // 0-100 percentage of image height
}
interface FrameEntry { filename: string; frameOffset: number; }
interface Sequence { seqId: string; frameCount: number; frames: FrameEntry[]; trajectoryPoints: TrajectoryPoint[]; }
interface FlightPlan { generatedAt: string; totalImages: number; sequences: Sequence[]; originLatLon: [number, number]; }

// ── pointcloud.json types (semantic terrain schema) ─────────────────────────
interface MeshPlane {
  filename: string;
  x: number; y: number; z: number;
  w: number; h: number; frame: number;
}
interface TerrainPt {
  x: number; y: number;       // world X / Z position (Three.js uses Y-up)
  h: number;                  // height in world units
  label: string;              // 'road' | 'vegetation' | 'building' | 'water' | 'ground'
  color: string;              // hex color
  frame: number;
}
interface ObjectMarker {
  x: number; y: number; z: number;
  w: number; h: number; d: number;
  cls: number; label: string; color: string; conf: number; frame: number;
}
interface PointCloud {
  generatedAt: string;
  depthBackend: string;
  stats: {
    numImages: number;
    numTerrain: number;
    numObjects: number;
    epochAtExport: number;
    mAP50AtExport: number;
    modelName: string;
    sampleStride: number;
  };
  planes: MeshPlane[];
  terrain: TerrainPt[];
  objects: ObjectMarker[];
  entities?: any;
  // Legacy support (old schema fallback)
  boxes?: any[];
}

// ── VisDrone annotation categories ───────────────────────────────────────────
const CATEGORY_LABELS: Record<number, string> = {
  1: 'Pedestrian', 2: 'People', 3: 'Bicycle', 4: 'Car',
  5: 'Van', 6: 'Truck', 7: 'Tricycle', 8: 'Awning-Tricycle',
  9: 'Bus', 10: 'Motor', 11: 'Others',
};
const CATEGORY_COLORS: Record<number, string> = {
  1: '#38bdf8', 2: '#38bdf8', 3: '#f43f5e', 4: '#10b981',
  5: '#f59e0b', 6: '#f59e0b', 7: '#a78bfa', 8: '#a78bfa',
  9: '#f59e0b', 10: '#f43f5e', 11: '#64748b',
};
interface BBox {
  x: number; y: number; w: number; h: number;
  label: string; color: string;
}

function parseAnnotation(txt: string): BBox[] {
  return txt.split('\n').filter(l => l.trim().length > 0).flatMap(line => {
    const parts = line.split(',').map(Number);
    if (parts.length < 6) return [];
    const [left, top, width, height, , category] = parts;
    if (category === 0 || !CATEGORY_LABELS[category]) return [];
    return [{
      x: left, y: top, w: width, h: height,
      label: CATEGORY_LABELS[category],
      color: CATEGORY_COLORS[category] ?? '#64748b'
    }];
  });
}

// ── View filters ──────────────────────────────────────────────────────────────
const VIEW_FILTER: Record<string, string> = {
  heatmap: 'saturate(0.35) brightness(0.65)',
  rgb: 'saturate(1.15) brightness(1.05) contrast(1.02)',
  source: 'grayscale(1) brightness(0.75) contrast(1.15)',
};
const VIEW_TINT: Record<string, string> = {
  heatmap: 'rgba(14,165,233,0.15)',
  rgb: 'transparent',
  source: 'rgba(99,102,241,0.12)',
};

// ── Rendered image rect accounting for object-contain letterboxing ────────────
function getRenderedRect(
  containerW: number, containerH: number, natW: number, natH: number
): { left: number; top: number; w: number; h: number } {
  if (natW === 0 || natH === 0) return { left: 0, top: 0, w: containerW, h: containerH };
  const scale = Math.min(containerW / natW, containerH / natH);
  const w = natW * scale;
  const h = natH * scale;
  return { left: (containerW - w) / 2, top: (containerH - h) / 2, w, h };
}

// ── Scan line ─────────────────────────────────────────────────────────────────
const ScanLine: React.FC = () => {
  const [pos, setPos] = useState(0);
  useEffect(() => {
    const iv = setInterval(() => setPos(p => (p + 0.3) % 100), 16);
    return () => clearInterval(iv);
  }, []);
  return (
    <div style={{
      position: 'absolute', left: 0, right: 0, top: `${pos}%`, height: 1,
      background: 'linear-gradient(90deg,transparent,rgba(56,189,248,0.2),rgba(56,189,248,0.45),rgba(56,189,248,0.2),transparent)',
      pointerEvents: 'none',
    }} />
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Three.js Point Cloud (Tier 2)
// ─────────────────────────────────────────────────────────────────────────────

/** Converts confidence (0-1) to a heatmap RGB color (blue→cyan→green→yellow→red) */
export function confToColor(conf: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, conf));
  if (t < 0.33) {
    const s = t / 0.33;
    return [244, Math.round(63 + s * (158 - 63)), Math.round(94 + s * (11 - 94))]; // rose→amber
  }
  if (t < 0.66) {
    const s = (t - 0.33) / 0.33;
    return [Math.round(245 + s * (56 - 245)), Math.round(158 + s * (189 - 158)), Math.round(11 + s * (248 - 11))]; // amber→cyan
  }
  const s = (t - 0.66) / 0.34;
  return [Math.round(56 - s * 40), Math.round(189 - s * 4), Math.round(248 - s * 119)]; // cyan→emerald
}

/** Source tag color: 1.0=MVS(cyan), ~0.6=mono(amber), ~0.2=3DGS(rose) */
export function sourceToColor(tag: number): [number, number, number] {
  if (tag >= 0.85) return [14, 165, 233];   // MVS – cyan
  if (tag >= 0.5) return [245, 158, 11];   // mono – amber
  return [244, 63, 94];                      // 3DGS – rose
}


// ── Textured image plane (renders the drone photo flat on the ground) ─────────
interface PointCloudMeshProps {
  cloud: PointCloud;
  viewMode: 'rgb' | 'heatmap' | 'source';
  clipHeight: number;
  pointSize: number;
}

const TexturedPlane: React.FC<{ plane: MeshPlane; index: number }> = ({ plane, index }) => {
  const texture = useLoader(THREE.TextureLoader, `/drone/${plane.filename}`);
  texture.colorSpace = THREE.SRGBColorSpace;
  // Stagger each overlapping plane by 2mm vertically based on frame index.
  // This completely eliminates WebGL Z-fighting (flickering/glitching) on overlapping drone passes.
  const yPos = -0.30 + (index * 0.003);

  return (
    <mesh position={[plane.x, yPos, plane.y]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[plane.w, plane.h]} />
      <meshBasicMaterial
        map={texture}
        side={THREE.DoubleSide}
        depthWrite={true}
        polygonOffset={true}
        polygonOffsetFactor={-index}
        polygonOffsetUnits={-index * 2}
      />
    </mesh>
  );
};

// ── Terrain label → geometry type ────────────────────────────────────────────

export function terrainShape(pt: TerrainPt): THREE.BufferGeometry {
  // Use a consistent footprint that covers the sampling stride area.
  // stride=25px * phys_scale(0.05) = 1.25 world units per sample.
  // Make each tile slightly larger (1.3) so there are no gaps between tiles.
  const tileW = 1.3;
  const tileD = 1.3;
  switch (pt.label) {
    case 'vegetation':
      return new THREE.ConeGeometry(tileW * 0.55, pt.h, 6);
    case 'building':
      return new THREE.BoxGeometry(tileW, pt.h, tileD);
    case 'water':
      return new THREE.BoxGeometry(tileW, 0.04, tileD);
    case 'road':
    case 'ground':
    default:
      return new THREE.BoxGeometry(tileW, Math.max(pt.h, 0.06), tileD);
  }
}

// Object marker renderer: thin wireframe outlines for high-confidence detected vehicles
const ObjectMarkers: React.FC<{ objects: ObjectMarker[]; clipHeight: number; viewMode?: string }> = ({ objects, clipHeight, viewMode = 'rgb' }) => {
  const minConf = viewMode === 'heatmap' ? 0.45 : 0.65;
  const filtered = objects.filter(o => o.h <= clipHeight && (o.conf ?? 1) >= minConf).slice(0, 120);

  return (
    <group>
      {filtered.map((obj, i) => {
        let geom: THREE.BufferGeometry;
        if (obj.label === 'Pedestrian' || obj.label === 'People') {
          geom = new THREE.CapsuleGeometry(Math.max(obj.w, obj.d) / 2.5, obj.h / 2, 4, 8);
        } else {
          geom = new THREE.BoxGeometry(obj.w, obj.h, obj.d);
        }
        return (
          <mesh key={`o-${i}`} position={[obj.x, obj.h / 2 + 0.1, obj.y]} geometry={geom}>
            <lineSegments geometry={new THREE.EdgesGeometry(geom)}>
              <lineBasicMaterial color={obj.color} linewidth={1} transparent opacity={0.6} />
            </lineSegments>
          </mesh>
        );
      })}
    </group>
  );
};

// Terrain renderer: renders subtle semantic overlay ONLY when heatmap shading is active
const SemanticTerrain: React.FC<{ terrain: TerrainPt[]; clipHeight: number; viewMode?: string }> = ({ terrain, clipHeight, viewMode = 'rgb' }) => {
  // If in realistic RGB mode, do NOT draw the heavy orange/blue blocks over the photo planes!
  if (viewMode !== 'heatmap') return null;

  const filtered = terrain
    .filter(pt => pt.h <= clipHeight && pt.label !== 'building' && pt.label !== 'vegetation')
    .slice(0, 5000);

  return (
    <group>
      {filtered.map((pt, i) => {
        const geom = new THREE.PlaneGeometry(1.25, 1.25);
        return (
          <mesh key={`t-${i}`} position={[pt.x, 0.03, pt.y]} rotation={[-Math.PI / 2, 0, 0]} geometry={geom}>
            <meshBasicMaterial color={pt.color} transparent opacity={0.28} depthWrite={false} />
          </mesh>
        );
      })}
    </group>
  );
};

/**
 * CityBuildingsOverlay
 * Loads the 63 MB GLB city model. Uses original colors.
 * Automatically scales and centers itself based on the underlying drone images (planes).
 */
const CityBuildingsOverlay: React.FC<{ planes: MeshPlane[] }> = ({ planes }) => {
  const { scene } = useGLTF('/api/assets/buildings.glb');
  const ref = useRef<THREE.Object3D>(null);

  const clonedScene = useMemo(() => scene.clone(true), [scene]);

  useEffect(() => {
    if (!ref.current || !planes || planes.length === 0) return;

    // Calculate the bounding box of the drone images (planes)
    let minX = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxZ = -Infinity;
    planes.forEach(p => {
      const halfW = p.w / 2;
      const halfH = p.h / 2;
      minX = Math.min(minX, p.x - halfW);
      maxX = Math.max(maxX, p.x + halfW);
      minZ = Math.min(minZ, p.y - halfH);
      maxZ = Math.max(maxZ, p.y + halfH);
    });

    const planesW = maxX - minX;
    const planesD = maxZ - minZ;
    const planesCenterX = (maxX + minX) / 2;
    const planesCenterZ = (maxZ + minZ) / 2;

    // Calculate bounding box of the city model (before scaling)
    const box = new THREE.Box3().setFromObject(clonedScene);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    // Scale the city to match the planes dimensions
    const scaleX = planesW / size.x;
    const scaleZ = planesD / size.z;
    const scale = Math.max(scaleX, scaleZ) * 0.95; // Scale to fit most of the area

    ref.current.scale.set(scale, scale, scale);

    // Position it so the center of the city matches the center of the planes
    // Also snap the bottom to y=0
    ref.current.position.set(
      planesCenterX - (center.x * scale),
      0 - (box.min.y * scale),
      planesCenterZ - (center.z * scale)
    );
  }, [clonedScene, planes]);

  return <primitive ref={ref} object={clonedScene} />;
};

/**
 * TreeOverlay — FBX tree model instanced at detected vegetation points.
 */
const TreeOverlay: React.FC<{ terrain: TerrainPt[] }> = ({ terrain }) => {
  const fbx = useFBX('/api/assets/tree.fbx');
  const vegPts = terrain.filter(pt => pt.label === 'vegetation').slice(0, 10);

  if (!fbx || vegPts.length === 0) return null;

  return (
    <group>
      {vegPts.map((pt, i) => (
        <Clone
          key={`tree-${i}`}
          object={fbx}
          scale={[0.003, 0.003, 0.003]}
          position={[pt.x, 0, pt.y]}
        />
      ))}
    </group>
  );
};

/**
 * UAVFlightTrajectory3D
 * Renders a glowing 3D flight corridor in the sky above the reconstructed 3D city:
 * - Continuous 3D CatmullRom spline tube floating at 28-32m altitude
 * - Waypoint spheres with vertical drop guidelines to each captured image plane
 * - Animated drone with forward camera coverage frustum gliding along the path
 */
const UAVFlightTrajectory3D: React.FC<{ planes: MeshPlane[] }> = ({ planes }) => {
  const droneRef = useRef<THREE.Group>(null);

  // Generate 3D waypoints above each image plane along the flight corridor
  const waypoints = useMemo(() => {
    if (!planes || planes.length === 0) return [];
    return planes.map((p, i) => {
      // Gentle realistic survey path centered over the flight corridor
      const swayX = Math.sin((i / Math.max(planes.length, 1)) * Math.PI * 2) * 4.0;
      const altY = 28.0 + Math.sin(i * 0.5) * 2.0; // Suspended 28-30m in the air above buildings
      return new THREE.Vector3(p.x + swayX, altY, p.y);
    });
  }, [planes]);

  const curve = useMemo(() => {
    if (waypoints.length < 2) return null;
    return new THREE.CatmullRomCurve3(waypoints, false, 'catmullrom', 0.2);
  }, [waypoints]);

  const tubeGeometry = useMemo(() => {
    if (!curve) return null;
    return new THREE.TubeGeometry(curve, 80, 0.45, 8, false);
  }, [curve]);

  // Animate the drone along the 3D flight path
  useFrame(({ clock }) => {
    if (!curve || !droneRef.current) return;
    const t = (clock.getElapsedTime() * 0.05) % 1;
    const pos = curve.getPointAt(t);
    const tangent = curve.getTangentAt(t);
    droneRef.current.position.copy(pos);
    droneRef.current.lookAt(pos.clone().add(tangent));
  });

  if (!curve || waypoints.length === 0) return null;

  return (
    <group>
      {/* 1. Glowing 3D Flight Corridor Tube */}
      {tubeGeometry && (
        <mesh geometry={tubeGeometry}>
          <meshStandardMaterial
            color="#38bdf8"
            emissive="#0ea5e9"
            emissiveIntensity={1.8}
            roughness={0.2}
            metalness={0.7}
            transparent
            opacity={0.88}
          />
        </mesh>
      )}

      {/* 2. Waypoint spheres + vertical guide lines down to ground */}
      {waypoints.map((wpt, i) => {
        const lineGeom = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(wpt.x, 0, wpt.z),
          new THREE.Vector3(wpt.x, wpt.y, wpt.z),
        ]);
        return (
          <group key={`wpt3d-${i}`}>
            {/* Vertical laser drop line down to ground */}
            {/* @ts-ignore */}
            <line geometry={lineGeom}>
              <lineBasicMaterial color="#0284c7" transparent opacity={0.35} />
            </line>

            {/* Glowing Waypoint Sphere */}
            <mesh position={[wpt.x, wpt.y, wpt.z]}>
              <sphereGeometry args={[0.9, 16, 16]} />
              <meshStandardMaterial
                color="#38bdf8"
                emissive="#38bdf8"
                emissiveIntensity={2.2}
              />
            </mesh>

            {/* Orbiting Halo Ring */}
            <mesh position={[wpt.x, wpt.y, wpt.z]} rotation={[Math.PI / 2, 0, 0]}>
              <ringGeometry args={[1.3, 1.7, 24]} />
              <meshBasicMaterial
                color="#0ea5e9"
                transparent
                opacity={0.7}
                side={THREE.DoubleSide}
              />
            </mesh>
          </group>
        );
      })}

      {/* 3. Animated UAV Drone with Camera Coverage Frustum */}
      <group ref={droneRef}>
        {/* Drone Center Chassis */}
        <mesh>
          <boxGeometry args={[1.6, 0.4, 1.6]} />
          <meshStandardMaterial color="#0f172a" metalness={0.9} roughness={0.2} />
        </mesh>
        <pointLight color="#38bdf8" intensity={4} distance={25} />

        {/* Diagonal Arms */}
        <mesh rotation={[0, Math.PI / 4, 0]}>
          <boxGeometry args={[3.8, 0.15, 0.25]} />
          <meshStandardMaterial color="#334155" />
        </mesh>
        <mesh rotation={[0, -Math.PI / 4, 0]}>
          <boxGeometry args={[3.8, 0.15, 0.25]} />
          <meshStandardMaterial color="#334155" />
        </mesh>

        {/* 4 Glowing Propellers */}
        {[
          [1.35, 0.25, 1.35],
          [-1.35, 0.25, 1.35],
          [1.35, 0.25, -1.35],
          [-1.35, 0.25, -1.35],
        ].map(([rx, ry, rz], idx) => (
          <mesh key={idx} position={[rx, ry, rz]} rotation={[-Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.75, 0.75, 0.04, 16]} />
            <meshBasicMaterial color="#38bdf8" transparent opacity={0.6} />
          </mesh>
        ))}

        {/* Downward Camera Sensor Frustum (Single-Pass Scanning Beam) */}
        <group rotation={[Math.PI / 3.2, 0, 0]}>
          <mesh position={[0, -12, 0]}>
            <coneGeometry args={[9, 24, 4, 1, true]} />
            <meshBasicMaterial
              color="#38bdf8"
              transparent
              opacity={0.08}
              wireframe
              side={THREE.DoubleSide}
            />
          </mesh>
        </group>
      </group>
    </group>
  );
};

interface PointCloudMeshProps {
  cloud: PointCloud;
  viewMode: 'rgb' | 'heatmap' | 'source';
  clipHeight: number;
  pointSize: number;
  showTrajectory?: boolean;
}

const ReconstructedMesh: React.FC<PointCloudMeshProps> = ({ cloud, viewMode, clipHeight, showTrajectory = true }) => {
  const terrain = cloud.terrain ?? [];
  const objects = cloud.objects ?? cloud.boxes ?? [];

  return (
    <group>
      {/* Ground-truth image planes */}
      {cloud.planes?.map((plane, i) => (
        <TexturedPlane key={i} plane={plane} index={i} />
      ))}
      {/* Semantic terrain geometry: subtle overlay only in heatmap mode */}
      <SemanticTerrain terrain={terrain} clipHeight={clipHeight} viewMode={viewMode} />
      {/* Clean high-confidence object markers */}
      <ObjectMarkers objects={objects} clipHeight={clipHeight} viewMode={viewMode} />
      {/* Dynamic 3D Entities: Individual Buildings & Trees placed at AI-detected coordinates */}
      {cloud.entities?.buildings && cloud.entities.buildings.length > 0 ? (
        <Suspense fallback={null}>
          <DynamicEntityManager entities={cloud.entities} clipHeight={clipHeight} />
        </Suspense>
      ) : (
        <>
          <Suspense fallback={null}>
            <CityBuildingsOverlay planes={cloud.planes ?? []} />
          </Suspense>
          <Suspense fallback={null}>
            <TreeOverlay terrain={terrain} />
          </Suspense>
        </>
      )}
      {/* 3D UAV Flight Trajectory Corridor in the sky */}
      {showTrajectory && (
        <UAVFlightTrajectory3D planes={cloud.planes ?? []} />
      )}
    </group>
  );
};

interface PointCloudViewerProps {
  cloud: PointCloud;
  viewMode: 'rgb' | 'heatmap' | 'source';
  clipHeight: number;
  pointSize: number;
  showTrajectory?: boolean;
  navMode: 'auto' | 'follow';
  mapBounds: MapBounds;
  resetKey: number;
  onZoneChange: (zone: 'glide' | 'rotate') => void;
  onManualTakeover: () => void;
}

/**
 * MapBoundaryVisual
 * Clearly defines the physical boundaries of the drone survey area:
 * - Glowing neon ground perimeter loop
 * - Holographic semi-transparent boundary barrier walls (3.2m tall) with illuminated top rail
 * - 4 Corner beacon towers with pulsating light caps
 */
const MapBoundaryVisual: React.FC<{ bounds: MapBounds }> = ({ bounds }) => {
  const { minX, maxX, minZ, maxZ, width, depth, centerX, centerZ } = bounds;

  const loopPoints = useMemo(() => [
    new THREE.Vector3(minX, 0.08, minZ),
    new THREE.Vector3(maxX, 0.08, minZ),
    new THREE.Vector3(maxX, 0.08, maxZ),
    new THREE.Vector3(minX, 0.08, maxZ),
  ], [minX, maxX, minZ, maxZ]);

  const loopGeometry = useMemo(() => {
    return new THREE.BufferGeometry().setFromPoints([...loopPoints, loopPoints[0]]);
  }, [loopPoints]);

  const wallHeight = 3.2;

  const topRailGeometry = useMemo(() => {
    return new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(minX, wallHeight, minZ),
      new THREE.Vector3(maxX, wallHeight, minZ),
      new THREE.Vector3(maxX, wallHeight, maxZ),
      new THREE.Vector3(minX, wallHeight, maxZ),
      new THREE.Vector3(minX, wallHeight, minZ),
    ]);
  }, [minX, maxX, minZ, maxZ, wallHeight]);

  const cornerPosts = useMemo(() => [
    [minX, minZ],
    [maxX, minZ],
    [maxX, maxZ],
    [minX, maxZ],
  ], [minX, maxX, minZ, maxZ]);

  return (
    <group>
      {/* 1. Glowing ground perimeter line */}
      {/* @ts-ignore */}
      <line geometry={loopGeometry}>
        <lineBasicMaterial color="#38bdf8" linewidth={2} transparent opacity={0.9} />
      </line>

      {/* 2. Top rail line of the boundary fence */}
      {/* @ts-ignore */}
      <line geometry={topRailGeometry}>
        <lineBasicMaterial color="#0ea5e9" linewidth={2} transparent opacity={0.7} />
      </line>

      {/* 3. Holographic Semi-transparent Boundary Walls */}
      {/* North wall */}
      <mesh position={[centerX, wallHeight / 2, minZ]}>
        <planeGeometry args={[width, wallHeight]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={0.08} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      {/* South wall */}
      <mesh position={[centerX, wallHeight / 2, maxZ]}>
        <planeGeometry args={[width, wallHeight]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={0.08} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      {/* West wall */}
      <mesh position={[minX, wallHeight / 2, centerZ]} rotation={[0, Math.PI / 2, 0]}>
        <planeGeometry args={[depth, wallHeight]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={0.08} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      {/* East wall */}
      <mesh position={[maxX, wallHeight / 2, centerZ]} rotation={[0, Math.PI / 2, 0]}>
        <planeGeometry args={[depth, wallHeight]} />
        <meshBasicMaterial color="#38bdf8" transparent opacity={0.08} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>

      {/* 4. Glowing Corner Beacon Pillars */}
      {cornerPosts.map(([cx, cz], i) => (
        <group key={`corner-beacon-${i}`} position={[cx, 0, cz]}>
          <mesh position={[0, wallHeight / 2, 0]}>
            <cylinderGeometry args={[0.22, 0.22, wallHeight, 12]} />
            <meshStandardMaterial color="#0369a1" emissive="#0284c7" emissiveIntensity={0.6} />
          </mesh>
          <mesh position={[0, wallHeight + 0.35, 0]}>
            <sphereGeometry args={[0.42, 16, 16]} />
            <meshStandardMaterial color="#38bdf8" emissive="#38bdf8" emissiveIntensity={2.8} />
          </mesh>
          <mesh position={[0, 0.09, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <ringGeometry args={[0.5, 1.8, 16]} />
            <meshBasicMaterial color="#38bdf8" transparent opacity={0.4} side={THREE.DoubleSide} />
          </mesh>
        </group>
      ))}
    </group>
  );
};

/**
 * CameraNavigationManager
 * Boundary-Aware Camera Controller:
 * - Mouse INSIDE map boundary: Left-click drag GLIDES throughout the map forward/backward/left/right
 * - Mouse OUTSIDE map boundary: Left-click drag ROTATES the entire map 360° in any direction
 * - Scroll wheel zooms smoothly anywhere
 * - Keyboard WASD / Arrow keys glide smoothly
 * - Follow Drone Mode: third-person chase camera locking onto the animated UAV drone
 */
const CameraNavigationManager: React.FC<{
  navMode: 'auto' | 'follow';
  planes?: MeshPlane[];
  mapBounds: MapBounds;
  resetKey: number;
  onZoneChange?: (zone: 'glide' | 'rotate') => void;
  onManualTakeover?: () => void;
}> = ({ navMode, planes, mapBounds, resetKey, onZoneChange, onManualTakeover }) => {
  const { camera, gl } = useThree();
  const controlsRef = useRef<any>(null);
  const keysPressed = useRef<{ [key: string]: boolean }>({});

  // Set initial camera view and handle reset
  useEffect(() => {
    if (controlsRef.current) {
      camera.position.set(0, 52, -38);
      if (controlsRef.current.target) {
        controlsRef.current.target.set(0, 8, 55);
      }
      controlsRef.current.update();
    }
  }, [resetKey, camera]);

  // Raycasting boundary interaction listener on domElement
  useEffect(() => {
    const domElement = gl.domElement;
    if (!domElement) return;

    const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const raycaster = new THREE.Raycaster();
    const intersection = new THREE.Vector3();

    const isInsideBounds = (clientX: number, clientY: number): boolean => {
      const rect = domElement.getBoundingClientRect();
      const x = ((clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(new THREE.Vector2(x, y), camera);
      const hits = raycaster.ray.intersectPlane(groundPlane, intersection);
      if (!hits) return false;

      return (
        intersection.x >= mapBounds.minX &&
        intersection.x <= mapBounds.maxX &&
        intersection.z >= mapBounds.minZ &&
        intersection.z <= mapBounds.maxZ
      );
    };

    const handlePointerDown = (e: PointerEvent) => {
      if (e.button !== 0 || !controlsRef.current) return;
      if (navMode === 'follow') {
        onManualTakeover?.();
      }

      const inside = isInsideBounds(e.clientX, e.clientY);
      onZoneChange?.(inside ? 'glide' : 'rotate');

      if (inside) {
        // INSIDE MAP BOUNDARY -> GLIDE / PAN FORWARD & ACROSS TERRAIN
        controlsRef.current.mouseButtons.LEFT = THREE.MOUSE.PAN;
        controlsRef.current.screenSpacePanning = false;
        controlsRef.current.panSpeed = 1.4;
      } else {
        // OUTSIDE MAP BOUNDARY -> ROTATE THE ENTIRE MAP 360° IN ANY DIRECTION
        controlsRef.current.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
        controlsRef.current.screenSpacePanning = false;
        controlsRef.current.rotateSpeed = 1.0;
      }
    };

    const handlePointerMove = (e: PointerEvent) => {
      if (e.buttons !== 0) return;
      const inside = isInsideBounds(e.clientX, e.clientY);
      onZoneChange?.(inside ? 'glide' : 'rotate');
      domElement.style.cursor = inside ? 'grab' : 'crosshair';
    };

    const handlePointerUp = () => {
      if (!domElement) return;
      domElement.style.cursor = 'default';
    };

    // Use capture phase so controlsRef.current.mouseButtons.LEFT is set before OrbitControls processes it
    domElement.addEventListener('pointerdown', handlePointerDown, { capture: true });
    domElement.addEventListener('pointermove', handlePointerMove, { passive: true });
    window.addEventListener('pointerup', handlePointerUp);

    return () => {
      domElement.removeEventListener('pointerdown', handlePointerDown, { capture: true });
      domElement.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [gl.domElement, camera, mapBounds, navMode, onZoneChange, onManualTakeover]);

  // Compute flight trajectory curve for follow drone mode
  const waypoints = useMemo(() => {
    if (!planes || planes.length === 0) return [];
    return planes.map((p, i) => {
      const swayX = Math.sin((i / Math.max(planes.length, 1)) * Math.PI * 2) * 4.0;
      const altY = 28.0 + Math.sin(i * 0.5) * 2.0;
      return new THREE.Vector3(p.x + swayX, altY, p.y);
    });
  }, [planes]);

  const curve = useMemo(() => {
    if (waypoints.length < 2) return null;
    return new THREE.CatmullRomCurve3(waypoints, false, 'catmullrom', 0.2);
  }, [waypoints]);

  // Keyboard navigation listener (WASD + Arrow keys)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
      keysPressed.current[e.code] = true;
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      keysPressed.current[e.code] = false;
    };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  useFrame(({ clock }, delta) => {
    if (navMode === 'follow' && curve) {
      const t = (clock.getElapsedTime() * 0.05) % 1;
      const pos = curve.getPointAt(t);
      const tangent = curve.getTangentAt(t).normalize();
      const behindCam = pos.clone().sub(tangent.clone().multiplyScalar(26)).add(new THREE.Vector3(0, 15, 0));
      camera.position.lerp(behindCam, 0.08);
      camera.lookAt(pos.clone().add(tangent.clone().multiplyScalar(15)));
      return;
    }

    // Keyboard smooth gliding across map (W/S moves along flight path, A/D strafes)
    const moveSpeed = 65 * delta;
    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);
    forward.y = 0;
    forward.normalize();

    const right = new THREE.Vector3();
    right.crossVectors(camera.up, forward).negate().normalize();

    const moveDelta = new THREE.Vector3();
    if (keysPressed.current['KeyW'] || keysPressed.current['ArrowUp']) {
      moveDelta.add(forward.clone().multiplyScalar(moveSpeed));
    }
    if (keysPressed.current['KeyS'] || keysPressed.current['ArrowDown']) {
      moveDelta.add(forward.clone().multiplyScalar(-moveSpeed));
    }
    if (keysPressed.current['KeyD'] || keysPressed.current['ArrowRight']) {
      moveDelta.add(right.clone().multiplyScalar(moveSpeed));
    }
    if (keysPressed.current['KeyA'] || keysPressed.current['ArrowLeft']) {
      moveDelta.add(right.clone().multiplyScalar(-moveSpeed));
    }

    if (moveDelta.lengthSq() > 0) {
      camera.position.add(moveDelta);
      if (controlsRef.current && controlsRef.current.target) {
        controlsRef.current.target.add(moveDelta);
      }
    }
  });

  if (navMode === 'follow') return null;

  return (
    <OrbitControls
      ref={controlsRef}
      target={[0, 8, 55]}
      enableDamping
      dampingFactor={0.08}
      panSpeed={1.4}
      rotateSpeed={1.0}
      minDistance={5}
      maxDistance={750}
      screenSpacePanning={false}
      maxPolarAngle={Math.PI / 2 - 0.02}
      makeDefault
    />
  );
};

const PointCloudViewer: React.FC<PointCloudViewerProps> = ({
  cloud,
  viewMode,
  clipHeight,
  pointSize,
  showTrajectory = true,
  navMode,
  mapBounds,
  resetKey,
  onZoneChange,
  onManualTakeover,
}) => (
  <Canvas
    style={{ width: '100%', height: '100%', background: '#060b14' }}
    gl={{ antialias: true, alpha: true }}
  >
    <PerspectiveCamera makeDefault position={[0, 52, -38]} fov={50} />
    <ambientLight intensity={1.2} />
    <directionalLight position={[70, 130, 70]} intensity={1.5} castShadow />
    <directionalLight position={[-60, 90, -60]} intensity={0.5} />
    <hemisphereLight args={['#38bdf8', '#0f172a', 0.65]} />
    <Suspense fallback={null}>
      <ReconstructedMesh cloud={cloud} viewMode={viewMode} clipHeight={clipHeight} pointSize={pointSize} showTrajectory={showTrajectory} />
    </Suspense>

    {/* Visually Defined Map Boundaries with glowing perimeter, walls & beacon posts */}
    <MapBoundaryVisual bounds={mapBounds} />

    {/* Dynamic Camera Navigation: Inside boundary -> Glide; Outside boundary -> Rotate 360° */}
    <CameraNavigationManager
      navMode={navMode}
      planes={cloud.planes}
      mapBounds={mapBounds}
      resetKey={resetKey}
      onZoneChange={onZoneChange}
      onManualTakeover={onManualTakeover}
    />

    {/* Background Grid floor outside the survey map */}
    <gridHelper args={[340, 68, '#0f2040', '#0d1a2e']} position={[0, -0.4, 105]} />
  </Canvas>
);

/** Overlay shown when pointcloud.json is missing */
const PointCloudPlaceholder: React.FC<{ epoch: number }> = ({ epoch }) => (
  <div style={{
    width: '100%', height: '100%',
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center',
    background: '#060b14', gap: 16,
  }}>
    <div style={{
      width: 60, height: 60, borderRadius: 14,
      background: 'rgba(56,189,248,0.08)',
      border: '1px solid rgba(56,189,248,0.2)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="1.5">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
      </svg>
    </div>
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: '#f1f5f9', marginBottom: 6 }}>
        3D Point Cloud — Epoch {epoch}
      </div>
      <div style={{ fontSize: 11, color: '#64748b', maxWidth: 280, lineHeight: 1.6 }}>
        Run the reconstruction script to build the 3D cloud from current training weights:
      </div>
      <div style={{
        marginTop: 10, padding: '8px 14px',
        background: 'rgba(15,23,42,0.8)', border: '1px solid rgba(56,189,248,0.2)',
        borderRadius: 7, fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: '#38bdf8',
      }}>
        python scripts/reconstruct_3d.py
      </div>
    </div>
  </div>
);

// ── Tier 2: 3D Reconstruction View ───────────────────────────────────────────

interface Tier2ViewProps {
  viewMode: 'rgb' | 'heatmap' | 'source';
  clipHeight: number;
  pointSize: number;
  showTrajectory?: boolean;
}

const Tier2View: React.FC<Tier2ViewProps> = ({ viewMode, clipHeight, pointSize, showTrajectory = true }) => {
  const [cloud, setCloud] = useState<PointCloud | null>(null);
  const [epoch, setEpoch] = useState(0);
  const [showSplat, setShowSplat] = useState(false);
  const [navMode, setNavMode] = useState<'auto' | 'follow'>('auto');
  const [activeZone, setActiveZone] = useState<'glide' | 'rotate'>('glide');
  const [resetKey, setResetKey] = useState(0);

  useEffect(() => {
    // Load epoch from training_status.json
    fetch('/training_status.json')
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setEpoch(d.epoch ?? 0); })
      .catch(() => { });

    // Load point cloud
    fetch('/pointcloud.json')
      .then(r => r.ok ? r.json() : null)
      .then((d: PointCloud | null) => { if (d) setCloud(d); })
      .catch(() => { });
  }, []);

  // Compute precise map boundaries from drone survey planes
  const mapBounds = useMemo<MapBounds>(() => {
    if (!cloud?.planes || cloud.planes.length === 0) {
      return { minX: -50, maxX: 50, minZ: -30, maxZ: 235, width: 100, depth: 265, centerX: 0, centerZ: 102.5 };
    }
    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;

    cloud.planes.forEach(p => {
      const halfW = p.w / 2;
      const halfH = p.h / 2;
      minX = Math.min(minX, p.x - halfW);
      maxX = Math.max(maxX, p.x + halfW);
      minZ = Math.min(minZ, p.y - halfH);
      maxZ = Math.max(maxZ, p.y + halfH);
    });

    minX -= 1.5;
    maxX += 1.5;
    minZ -= 1.5;
    maxZ += 1.5;

    return {
      minX,
      maxX,
      minZ,
      maxZ,
      width: maxX - minX,
      depth: maxZ - minZ,
      centerX: (minX + maxX) / 2,
      centerZ: (minZ + maxZ) / 2,
    };
  }, [cloud?.planes]);

  if (!cloud) return <PointCloudPlaceholder epoch={epoch} />;

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {showSplat ? (
        <Canvas style={{ width: '100%', height: '100%', background: '#060b14' }}>
          <PerspectiveCamera makeDefault position={[0, 80, 110]} fov={55} />
          <ambientLight intensity={0.8} />
          <Suspense fallback={null}>
            <Splat src="/api/assets/reconstruction.splat" />
          </Suspense>
          <OrbitControls makeDefault />
        </Canvas>
      ) : (
        <PointCloudViewer
          cloud={cloud}
          viewMode={viewMode}
          clipHeight={clipHeight}
          pointSize={pointSize}
          showTrajectory={showTrajectory}
          navMode={navMode}
          mapBounds={mapBounds}
          resetKey={resetKey}
          onZoneChange={setActiveZone}
          onManualTakeover={() => setNavMode('auto')}
        />
      )}

      {/* View Toggle (Semantic vs 3DGS) */}
      <div style={{
        position: 'absolute', top: 50, left: 10, zIndex: 10,
        display: 'flex', gap: 4, background: 'rgba(15,23,42,0.85)',
        backdropFilter: 'blur(8px)',
        padding: 4, borderRadius: 6, border: '1px solid rgba(56,189,248,0.2)'
      }}>
        <button
          onClick={() => setShowSplat(false)}
          style={{
            padding: '4px 12px', fontSize: 11, fontWeight: 600,
            background: !showSplat ? '#38bdf8' : 'transparent',
            color: !showSplat ? '#0f172a' : '#94a3b8',
            border: 'none', borderRadius: 4, cursor: 'pointer',
            transition: 'all 0.2s'
          }}
        >
          Semantic Proxy
        </button>
        <button
          onClick={() => setShowSplat(true)}
          style={{
            padding: '4px 12px', fontSize: 11, fontWeight: 600,
            background: showSplat ? '#38bdf8' : 'transparent',
            color: showSplat ? '#0f172a' : '#94a3b8',
            border: 'none', borderRadius: 4, cursor: 'pointer',
            transition: 'all 0.2s'
          }}
        >
          Photorealistic (3DGS)
        </button>
      </div>

      {/* Top-Right Boundary Navigation Bar */}
      <div style={{
        position: 'absolute', top: 50, right: 14, zIndex: 10,
        display: 'flex', alignItems: 'center', gap: 6, background: 'rgba(15,23,42,0.92)',
        backdropFilter: 'blur(12px)',
        padding: '5px 8px', borderRadius: 9, border: '1px solid rgba(56,189,248,0.25)',
        boxShadow: '0 6px 24px rgba(0,0,0,0.55)',
      }}>
        {/* Dynamic Boundary Zone Status Badge */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '4px 11px', fontSize: 11, fontWeight: 700,
          background: activeZone === 'glide' ? 'rgba(2,132,199,0.2)' : 'rgba(168,85,247,0.2)',
          color: activeZone === 'glide' ? '#38bdf8' : '#c084fc',
          border: activeZone === 'glide' ? '1px solid rgba(56,189,248,0.45)' : '1px solid rgba(192,132,252,0.45)',
          borderRadius: 6,
          transition: 'all 0.2s ease',
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%',
            background: activeZone === 'glide' ? '#38bdf8' : '#c084fc',
            boxShadow: activeZone === 'glide' ? '0 0 8px #38bdf8' : '0 0 8px #c084fc',
          }} />
          {activeZone === 'glide' ? 'INSIDE MAP · DRAG TO GLIDE' : 'OUTSIDE BOUNDARY · DRAG TO ROTATE 360°'}
        </div>

        {/* Follow Drone Button */}
        <button
          onClick={() => setNavMode(m => m === 'follow' ? 'auto' : 'follow')}
          style={{
            display: 'flex', alignItems: 'center', gap: 5,
            padding: '5px 12px', fontSize: 11, fontWeight: 600,
            background: navMode === 'follow' ? '#0ea5e9' : 'transparent',
            color: navMode === 'follow' ? '#ffffff' : '#94a3b8',
            border: 'none', borderRadius: 6, cursor: 'pointer',
            transition: 'all 0.2s',
          }}
          title="Camera dynamically follows the animated UAV drone"
        >
          <Video size={13} />
          {navMode === 'follow' ? 'Following Drone' : 'Follow Drone'}
        </button>

        {/* Reset Camera View */}
        <button
          onClick={() => {
            setNavMode('auto');
            setResetKey(k => k + 1);
          }}
          style={{
            display: 'flex', alignItems: 'center', gap: 5,
            padding: '5px 10px', fontSize: 11, fontWeight: 600,
            background: 'transparent',
            color: '#64748b',
            border: 'none', borderRadius: 6, cursor: 'pointer',
            transition: 'all 0.2s',
          }}
          title="Reset camera view to beginning of flight corridor"
        >
          <RotateCcw size={13} />
          Reset
        </button>
      </div>

      {/* Info overlay */}
      <div style={{
        position: 'absolute', top: 10, left: '50%', transform: 'translateX(-50%)',
        display: 'flex', gap: 8, pointerEvents: 'none', flexWrap: 'wrap', justifyContent: 'center',
      }}>
        {[
          { label: 'EPOCH', value: String(cloud.stats?.epochAtExport ?? (cloud as any).epochAtExport ?? '–') },
          { label: 'mAP50', value: `${(((cloud.stats?.mAP50AtExport ?? (cloud as any).mAP50AtExport ?? 0)) * 100).toFixed(1)}%` },
          { label: 'TERRAIN', value: (cloud.stats?.numTerrain ?? cloud.terrain?.length ?? 0).toLocaleString() },
          { label: 'OBJECTS', value: (cloud.stats?.numObjects ?? cloud.objects?.length ?? 0).toLocaleString() },
          { label: 'DEPTH', value: (cloud.depthBackend ?? 'n/a').toUpperCase() },
          { label: 'MODEL', value: (cloud.stats?.modelName ?? (cloud as any).modelName ?? 'YOLO').toUpperCase() },
        ].map(({ label, value }) => (
          <div key={label} style={{
            background: 'rgba(6,11,20,0.82)', backdropFilter: 'blur(8px)',
            border: '1px solid rgba(56,189,248,0.15)', borderRadius: 5,
            padding: '3px 9px',
            fontFamily: "'JetBrains Mono', monospace", fontSize: 10,
          }}>
            <span style={{ color: '#334155' }}>{label} </span>
            <span style={{ color: '#38bdf8', fontWeight: 700 }}>{value}</span>
          </div>
        ))}
      </div>

      {/* Controls hint */}
      <div style={{
        position: 'absolute', bottom: 12, left: '50%', transform: 'translateX(-50%)',
        fontFamily: "'JetBrains Mono', monospace", fontSize: 10,
        color: '#94a3b8', background: 'rgba(6,11,20,0.88)', backdropFilter: 'blur(10px)',
        padding: '5px 18px', borderRadius: 14, border: '1px solid rgba(56,189,248,0.22)',
        pointerEvents: 'none', boxShadow: '0 4px 18px rgba(0,0,0,0.45)',
        whiteSpace: 'nowrap',
      }}>
        {navMode === 'follow'
          ? '🛸 Following Drone along Flight Corridor · Click anywhere to take manual control'
          : '🖐️ Inside Map Boundary: Left-drag to glide forward & across map · 🔄 Outside Boundary: Left-drag to rotate 360° · Scroll to zoom'}
      </div>

      {/* Scan line */}
      <ScanLine />
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Tier 1: 2D Drone Image Viewer (original MeshViewer logic, bbox FIXED)
// ─────────────────────────────────────────────────────────────────────────────

interface Tier1ViewProps {
  viewMode: 'rgb' | 'heatmap' | 'source';
  showTrajectory: boolean;
}

const Tier1View: React.FC<Tier1ViewProps> = ({ viewMode, showTrajectory }) => {
  const [plan, setPlan] = useState<FlightPlan | null>(null);
  const [seqIdx, setSeqIdx] = useState(0);
  const [frameIdx, setFrameIdx] = useState(0);
  const [imgLoaded, setImgLoaded] = useState(false);
  const [imgNat, setImgNat] = useState({ w: 0, h: 0 });
  const [boxes, setBoxes] = useState<BBox[]>([]);
  const [playing, setPlaying] = useState(true);
  const [containerSize, setContainerSize] = useState({ w: 0, h: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Measure container
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setContainerSize({ w: width, h: height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Load flight plan
  useEffect(() => {
    fetch('/flight_plan.json')
      .then(r => r.json())
      .then((data: FlightPlan) => setPlan(data))
      .catch(e => console.error('Failed to load flight plan:', e));
  }, []);

  const currentSeq = plan?.sequences[seqIdx] ?? null;
  const totalFrames = currentSeq?.frames.length ?? 0;
  const currentFrame = currentSeq?.frames[frameIdx] ?? null;

  // Load annotations
  useEffect(() => {
    if (!currentFrame) return;
    setBoxes([]);
    fetch(`/api/annotation/${currentFrame.filename.replace('.jpg', '.txt')}`)
      .then(r => r.ok ? r.text() : '')
      .then(txt => txt ? setBoxes(parseAnnotation(txt)) : setBoxes([]))
      .catch(() => setBoxes([]));
  }, [currentFrame?.filename]);

  // Playback
  const advance = useCallback(() => {
    setImgLoaded(false);
    setFrameIdx(i => {
      if (i + 1 >= (currentSeq?.frames.length ?? 1)) {
        if (plan) setSeqIdx(s => (s + 1) % plan.sequences.length);
        return 0;
      }
      return i + 1;
    });
  }, [currentSeq, plan]);

  const goBack = useCallback(() => {
    setImgLoaded(false);
    setFrameIdx(i => Math.max(0, i - 1));
  }, []);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(advance, 5000);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, advance]);

  // Trajectory
  const trajectoryPts = currentSeq?.trajectoryPoints ?? [];
  const activeWaypointIdx = trajectoryPts.length > 0
    ? Math.floor((frameIdx / Math.max(totalFrames - 1, 1)) * (trajectoryPts.length - 1))
    : 0;
  const activeWpt = trajectoryPts[activeWaypointIdx];

  // Viewport area (exclude 70px controls bar at bottom)
  const viewH = Math.max(1, containerSize.h - 70);
  const viewW = Math.max(1, containerSize.w);
  const rendered = getRenderedRect(viewW, viewH, imgNat.w || viewW, imgNat.h || viewH);

  // RIGHT / BOTTOM bounds of the rendered image area (absolute in container)
  const renderedRight = rendered.left + rendered.w;
  const renderedBottom = rendered.top + rendered.h;

  const imgUrl = currentFrame ? `/drone/${currentFrame.filename}` : null;
  const depthImgUrl = currentFrame ? `/drone/${currentFrame.filename.replace('.jpg', '_depth.jpg')}` : null;

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%', position: 'relative', background: '#060b14', overflow: 'hidden' }}
    >
      {/* Image */}
      {imgUrl && (
        <img
          key={imgUrl}
          src={imgUrl}
          alt={`Seq ${currentSeq?.seqId} frame ${frameIdx + 1}`}
          onLoad={(e) => {
            setImgLoaded(true);
            setImgNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight });
          }}
          onError={() => setImgLoaded(true)}
          style={{
            position: 'absolute', top: 0, left: 0,
            width: '100%',
            height: `calc(100% - 70px)`,
            objectFit: 'contain',
            objectPosition: 'center',
            filter: VIEW_FILTER[viewMode] ?? 'none',
            opacity: imgLoaded ? 1 : 0,
            transition: 'opacity 0.3s ease, filter 0.4s ease',
          }}
        />
      )}

      {/* Depth Image Overlay for Heatmap mode */}
      {depthImgUrl && (
        <img
          key={depthImgUrl}
          src={depthImgUrl}
          alt={`Depth map for ${frameIdx + 1}`}
          style={{
            position: 'absolute', top: 0, left: 0,
            width: '100%',
            height: `calc(100% - 70px)`,
            objectFit: 'contain',
            objectPosition: 'center',
            opacity: viewMode === 'heatmap' && imgLoaded ? 0.75 : 0,
            transition: 'opacity 0.4s ease',
            mixBlendMode: 'screen',
            pointerEvents: 'none',
          }}
        />
      )}

      {/* Loading shimmer */}
      {!imgLoaded && (
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, bottom: '70px',
          background: 'linear-gradient(135deg,#0c1829,#0f2040,#0c1829)',
        }} />
      )}

      {/* View mode tint */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, bottom: '70px',
        background: VIEW_TINT[viewMode],
        pointerEvents: 'none', transition: 'background 0.4s ease',
      }} />

      {/* ── BOUNDING BOXES — clamped to rendered image rect ────────────── */}
      {imgLoaded && imgNat.w > 0 && boxes.slice(0, 14).map((box, i) => {
        // Convert annotation pixel coords → container-absolute pixels
        const rawLeft = rendered.left + (box.x / imgNat.w) * rendered.w;
        const rawTop = rendered.top + (box.y / imgNat.h) * rendered.h;
        const rawW = (box.w / imgNat.w) * rendered.w;
        const rawH = (box.h / imgNat.h) * rendered.h;

        // Clamp left & top to stay inside rendered rect
        const left = Math.max(rendered.left, rawLeft);
        const top = Math.max(rendered.top, rawTop);

        // Clamp right edge: box must not exceed rendered image right/bottom
        const right = Math.min(renderedRight, rawLeft + Math.max(rawW, 6));
        const bottom = Math.min(renderedBottom, rawTop + Math.max(rawH, 6));

        const w = right - left;
        const h = bottom - top;

        // Skip degenerate boxes (completely outside image area)
        if (w <= 0 || h <= 0) return null;

        return (
          <div key={i} style={{
            position: 'absolute', left, top, width: w, height: h,
            border: `1.5px solid ${box.color}`,
            boxShadow: `0 0 6px ${box.color}40`,
            pointerEvents: 'none',
          }}>
            <div style={{
              position: 'absolute', top: '-15px', left: 0,
              background: `${box.color}dd`, color: '#fff',
              fontSize: 8, fontWeight: 700, padding: '1px 4px', borderRadius: 2,
              whiteSpace: 'nowrap', fontFamily: "'JetBrains Mono', monospace",
            }}>{box.label}</div>
            {/* Corner ticks */}
            {[
              { top: -1, left: -1, borderTop: `2px solid ${box.color}`, borderLeft: `2px solid ${box.color}` },
              { top: -1, right: -1, borderTop: `2px solid ${box.color}`, borderRight: `2px solid ${box.color}` },
              { bottom: -1, left: -1, borderBottom: `2px solid ${box.color}`, borderLeft: `2px solid ${box.color}` },
              { bottom: -1, right: -1, borderBottom: `2px solid ${box.color}`, borderRight: `2px solid ${box.color}` },
            ].map((s, j) => (
              <div key={j} style={{ position: 'absolute', width: 6, height: 6, ...s }} />
            ))}
          </div>
        );
      })}

      {/* ── GPS Trajectory — inside rendered image rect only ─────────── */}
      {showTrajectory && imgLoaded && trajectoryPts.length > 0 && (
        <svg
          style={{
            position: 'absolute',
            left: rendered.left, top: rendered.top,
            width: rendered.w, height: rendered.h,
            pointerEvents: 'none', overflow: 'hidden',
          }}
          viewBox={`0 0 ${rendered.w} ${rendered.h}`}
        >
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
              <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>

          {/* Perspective Flight Corridor Polyline */}
          {(() => {
            const corridorPoints = trajectoryPts.map((_, i) => {
              const prog = i / Math.max(trajectoryPts.length - 1, 1);
              // Naturally follows the road center corridor from horizon (top center) to foreground (bottom center)
              const px = rendered.w * (0.50 + Math.sin(prog * Math.PI * 1.2) * 0.06);
              const py = rendered.h * (0.16 + prog * 0.72);
              return { x: px, y: py, i };
            });

            return (
              <>
                {/* Glowing flight corridor center-line */}
                <polyline
                  points={corridorPoints.map(p => `${p.x},${p.y}`).join(' ')}
                  fill="none"
                  stroke="rgba(56,189,248,0.85)"
                  strokeWidth="3.0"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  filter="url(#glow)"
                />

                {/* Waypoint dots */}
                {corridorPoints.map(p => (
                  <circle
                    key={p.i}
                    cx={p.x}
                    cy={p.y}
                    r={p.i === activeWaypointIdx ? 6 : 2.5}
                    fill={p.i === activeWaypointIdx ? '#38bdf8' : 'rgba(56,189,248,0.6)'}
                    filter={p.i === activeWaypointIdx ? 'url(#glow)' : undefined}
                  />
                ))}

                {/* Active UAV position reticle */}
                {corridorPoints[activeWaypointIdx] && (
                  <g>
                    <circle
                      cx={corridorPoints[activeWaypointIdx].x}
                      cy={corridorPoints[activeWaypointIdx].y}
                      r={14}
                      fill="none"
                      stroke="#38bdf8"
                      strokeWidth="1.5"
                      opacity="0.6"
                    />
                    <circle
                      cx={corridorPoints[activeWaypointIdx].x}
                      cy={corridorPoints[activeWaypointIdx].y}
                      r={22}
                      fill="none"
                      stroke="#38bdf8"
                      strokeWidth="1.0"
                      strokeDasharray="3 3"
                      opacity="0.4"
                    />
                  </g>
                )}
              </>
            );
          })()}
        </svg>
      )}

      {/* Scan line */}
      <ScanLine />

      {/* Corner brackets */}
      {[
        { top: 12, left: 12, borderTop: '2px solid rgba(56,189,248,0.45)', borderLeft: '2px solid rgba(56,189,248,0.45)', width: 20, height: 20 },
        { top: 12, right: 12, borderTop: '2px solid rgba(56,189,248,0.45)', borderRight: '2px solid rgba(56,189,248,0.45)', width: 20, height: 20 },
        { bottom: 82, left: 12, borderBottom: '2px solid rgba(56,189,248,0.45)', borderLeft: '2px solid rgba(56,189,248,0.45)', width: 20, height: 20 },
        { bottom: 82, right: 12, borderBottom: '2px solid rgba(56,189,248,0.45)', borderRight: '2px solid rgba(56,189,248,0.45)', width: 20, height: 20 },
      ].map((s, i) => <div key={i} style={{ position: 'absolute', pointerEvents: 'none', ...s }} />)}

      {/* Crosshair */}
      <div style={{ position: 'absolute', top: `calc(50% - 35px)`, left: '50%', transform: 'translate(-50%,-50%)', pointerEvents: 'none', opacity: 0.25 }}>
        <Crosshair size={24} color="#38bdf8" />
      </div>

      {/* HUD top bar */}
      <div style={{
        position: 'absolute', top: 10, left: '50%', transform: 'translateX(-50%)',
        display: 'flex', gap: 8, pointerEvents: 'none', flexWrap: 'wrap', justifyContent: 'center',
      }}>
        {[
          { label: 'SEQ', value: currentSeq?.seqId ?? '–' },
          { label: 'FRAME', value: totalFrames > 0 ? `${frameIdx + 1}/${totalFrames}` : '–' },
          { label: 'ALT', value: activeWpt ? `${activeWpt.alt.toFixed(1)}m` : '–' },
          { label: 'LAT', value: activeWpt ? activeWpt.lat.toFixed(5) : '–' },
          { label: 'LON', value: activeWpt ? activeWpt.lon.toFixed(5) : '–' },
        ].map(({ label, value }) => (
          <div key={label} style={{
            background: 'rgba(6,11,20,0.82)', backdropFilter: 'blur(8px)',
            border: '1px solid rgba(56,189,248,0.15)', borderRadius: 5,
            padding: '3px 9px', fontFamily: "'JetBrains Mono', monospace", fontSize: 10,
          }}>
            <span style={{ color: '#334155' }}>{label} </span>
            <span style={{ color: '#38bdf8', fontWeight: 700 }}>{value}</span>
          </div>
        ))}
      </div>

      {/* Object count */}
      {boxes.length > 0 && (
        <div style={{
          position: 'absolute', top: 48, right: 12, pointerEvents: 'none',
          background: 'rgba(6,11,20,0.82)', border: '1px solid rgba(56,189,248,0.18)',
          borderRadius: 6, padding: '4px 10px',
          fontFamily: "'JetBrains Mono', monospace", fontSize: 10,
        }}>
          <span style={{ color: '#475569' }}>OBJECTS </span>
          <span style={{ color: '#10b981', fontWeight: 700 }}>{boxes.length}</span>
        </div>
      )}

      {/* Waypoint info */}
      {activeWpt && showTrajectory && (
        <div style={{
          position: 'absolute', top: 48, left: 12, pointerEvents: 'none',
          background: 'rgba(6,11,20,0.82)', border: '1px solid rgba(56,189,248,0.18)',
          borderRadius: 6, padding: '4px 10px',
          display: 'flex', alignItems: 'center', gap: 6,
          fontFamily: "'JetBrains Mono', monospace", fontSize: 10,
        }}>
          <Navigation size={10} color="#38bdf8" />
          <span style={{ color: '#475569' }}>WPT {activeWaypointIdx + 1}/{trajectoryPts.length}</span>
          <span style={{ color: '#38bdf8', fontWeight: 700 }}>t={activeWpt.timestamp.toFixed(1)}s</span>
        </div>
      )}

      {/* Bottom controls bar */}
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0, height: 70,
        background: 'linear-gradient(to top, rgba(6,11,20,0.98) 0%, rgba(6,11,20,0.85) 100%)',
        padding: '8px 14px 10px',
        display: 'flex', flexDirection: 'column', gap: 6,
      }}>
        {/* Sequence selector */}
        {plan && (
          <div style={{ display: 'flex', gap: 5 }}>
            {plan.sequences.map((seq, i) => (
              <button key={seq.seqId}
                onClick={() => { setSeqIdx(i); setFrameIdx(0); setImgLoaded(false); }}
                style={{
                  flex: 1, padding: '3px 0', borderRadius: 4, border: 'none', cursor: 'pointer',
                  fontSize: 8, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
                  background: i === seqIdx ? 'rgba(56,189,248,0.2)' : 'rgba(30,41,59,0.6)',
                  color: i === seqIdx ? '#38bdf8' : '#475569',
                  outline: i === seqIdx ? '1px solid rgba(56,189,248,0.4)' : 'none',
                  transition: 'all 0.2s',
                }}>
                {seq.seqId}
              </button>
            ))}
          </div>
        )}

        {/* Timeline + controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button onClick={goBack} style={btnStyle(false)}><SkipBack size={12} /></button>
          <button onClick={() => setPlaying(p => !p)} style={btnStyle(playing)}>
            {playing ? <Pause size={12} /> : <Play size={12} />}
            <span style={{ fontSize: 10, fontWeight: 700 }}>{playing ? 'LIVE' : 'PAUSED'}</span>
          </button>
          <button onClick={advance} style={btnStyle(false)}><SkipForward size={12} /></button>

          {/* Scrubber */}
          <div
            style={{ flex: 1, height: 3, background: 'rgba(255,255,255,0.08)', borderRadius: 2, cursor: 'pointer', position: 'relative' }}
            onClick={e => {
              if (!currentSeq) return;
              const r = e.currentTarget.getBoundingClientRect();
              const ratio = (e.clientX - r.left) / r.width;
              setFrameIdx(Math.min(Math.floor(ratio * currentSeq.frames.length), currentSeq.frames.length - 1));
              setImgLoaded(false);
            }}
          >
            <div style={{
              position: 'absolute', left: 0, top: 0, bottom: 0,
              width: totalFrames > 0 ? `${((frameIdx + 1) / totalFrames) * 100}%` : '0%',
              background: 'linear-gradient(90deg,#0ea5e9,#6366f1)',
              borderRadius: 2, boxShadow: '0 0 5px rgba(56,189,248,0.4)',
              transition: 'width 0.4s ease',
            }} />
          </div>

          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: '#334155', whiteSpace: 'nowrap' }}>
            {plan ? `${plan.totalImages} frames · ${trajectoryPts.length} wpts` : 'Loading…'}
          </span>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Main component — delegates to Tier1View or Tier2View
// ─────────────────────────────────────────────────────────────────────────────

export const MeshViewer: React.FC<MeshViewerProps> = ({
  viewMode, tierMode, clipHeight, pointSize, showTrajectory,
}) => {
  if (tierMode === 'tier2') {
    return <Tier2View viewMode={viewMode} clipHeight={clipHeight} pointSize={pointSize} showTrajectory={showTrajectory} />;
  }
  return <Tier1View viewMode={viewMode} showTrajectory={showTrajectory} />;
};

function btnStyle(active: boolean): React.CSSProperties {
  return {
    background: active ? 'rgba(14,165,233,0.18)' : 'rgba(30,41,59,0.65)',
    border: `1px solid ${active ? 'rgba(56,189,248,0.4)' : 'rgba(255,255,255,0.07)'}`,
    borderRadius: 6, padding: '5px 12px', cursor: 'pointer',
    color: active ? '#38bdf8' : '#94a3b8',
    display: 'flex', alignItems: 'center', gap: 4,
    boxShadow: active ? '0 0 8px rgba(56,189,248,0.2)' : 'none',
    transition: 'all 0.2s', flexShrink: 0,
  };
}
