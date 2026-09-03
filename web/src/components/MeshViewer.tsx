import React, { useState, useEffect, useRef, useCallback, Suspense } from 'react';
import { Play, Pause, SkipForward, SkipBack, Crosshair, Navigation } from 'lucide-react';
import { Canvas, useFrame, useThree, useLoader } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

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
interface FrameEntry  { filename: string; frameOffset: number; }
interface Sequence    { seqId: string; frameCount: number; frames: FrameEntry[]; trajectoryPoints: TrajectoryPoint[]; }
interface FlightPlan  { generatedAt: string; totalImages: number; sequences: Sequence[]; originLatLon: [number, number]; }

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
  planes:   MeshPlane[];
  terrain:  TerrainPt[];
  objects:  ObjectMarker[];
  // Legacy support (old schema fallback)
  boxes?:   any[];
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
    return [{ x: left, y: top, w: width, h: height,
              label: CATEGORY_LABELS[category],
              color: CATEGORY_COLORS[category] ?? '#64748b' }];
  });
}

// ── View filters ──────────────────────────────────────────────────────────────
const VIEW_FILTER: Record<string, string> = {
  heatmap: 'saturate(0.35) brightness(0.65)',
  rgb:     'saturate(1.15) brightness(1.05) contrast(1.02)',
  source:  'grayscale(1) brightness(0.75) contrast(1.15)',
};
const VIEW_TINT: Record<string, string> = {
  heatmap: 'rgba(14,165,233,0.15)',
  rgb:     'transparent',
  source:  'rgba(99,102,241,0.12)',
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
function confToColor(conf: number): [number, number, number] {
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
function sourceToColor(tag: number): [number, number, number] {
  if (tag >= 0.85) return [14, 165, 233];   // MVS – cyan
  if (tag >= 0.5)  return [245, 158, 11];   // mono – amber
  return [244, 63, 94];                      // 3DGS – rose
}


// ── Textured image plane (renders the drone photo flat on the ground) ─────────
interface PointCloudMeshProps {
  cloud: PointCloud;
  viewMode: 'rgb' | 'heatmap' | 'source';
  clipHeight: number;
  pointSize: number;
}

const TexturedPlane: React.FC<{ plane: MeshPlane; index: number }> = ({ plane }) => {
  const texture = useLoader(THREE.TextureLoader, `/drone/${plane.filename}`);
  texture.colorSpace = THREE.SRGBColorSpace;
  // Move plane to y=-0.3 so it sits BELOW all terrain geometry (which starts at y=0).
  // polygonOffset pushes it further away from the camera to prevent Z-fighting.
  return (
    <mesh position={[plane.x, -0.3, plane.y]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[plane.w, plane.h]} />
      <meshBasicMaterial
        map={texture}
        side={THREE.DoubleSide}
        depthWrite={true}
        polygonOffset={true}
        polygonOffsetFactor={2}
        polygonOffsetUnits={2}
      />
    </mesh>
  );
};

// ── Terrain label → geometry type ────────────────────────────────────────────

function terrainShape(pt: TerrainPt): THREE.BufferGeometry {
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

// Terrain renderer: one mesh per sampled terrain point
const SemanticTerrain: React.FC<{ terrain: TerrainPt[]; clipHeight: number }> = ({ terrain, clipHeight }) => {
  const filtered = terrain.filter(pt => pt.h <= clipHeight);
  return (
    <group>
      {filtered.map((pt, i) => {
        const geom  = terrainShape(pt);
        const color = pt.color;
        const yPos  = pt.h / 2;  // centre of geometry sits at half-height above ground
        const opacity = pt.label === 'road' || pt.label === 'ground' ? 0.55 : 0.75;
        return (
          <mesh key={`t-${i}`} position={[pt.x, yPos, pt.y]} geometry={geom}>
            <meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
            {(pt.label === 'building' || pt.label === 'vegetation') && (
              <lineSegments geometry={new THREE.EdgesGeometry(geom)}>
                <lineBasicMaterial color={color} transparent opacity={0.6} />
              </lineSegments>
            )}
          </mesh>
        );
      })}
    </group>
  );
};

// Object marker renderer: thin wireframe slabs for detected vehicles/people
const ObjectMarkers: React.FC<{ objects: ObjectMarker[]; clipHeight: number }> = ({ objects, clipHeight }) => {
  return (
    <group>
      {objects.filter(o => o.h <= clipHeight).map((obj, i) => {
        let geom: THREE.BufferGeometry;
        if (obj.label === 'Pedestrian' || obj.label === 'People') {
          geom = new THREE.CapsuleGeometry(Math.max(obj.w, obj.d) / 2.5, obj.h / 2, 4, 8);
        } else {
          geom = new THREE.BoxGeometry(obj.w, obj.h, obj.d);
        }
        return (
          <mesh key={`o-${i}`} position={[obj.x, obj.h / 2, obj.y]} geometry={geom}>
            <meshBasicMaterial color={obj.color} transparent opacity={0.15} depthWrite={false} />
            <lineSegments geometry={new THREE.EdgesGeometry(geom)}>
              <lineBasicMaterial color={obj.color} linewidth={1} />
            </lineSegments>
          </mesh>
        );
      })}
    </group>
  );
};

const ReconstructedMesh: React.FC<PointCloudMeshProps> = ({ cloud, clipHeight }) => {
  const groupRef = useRef<THREE.Group>(null);
  const terrain  = cloud.terrain  ?? [];
  const objects  = cloud.objects  ?? cloud.boxes ?? [];

  // Gentle auto-rotate
  useFrame((_, delta) => {
    if (groupRef.current) groupRef.current.rotation.y += delta * 0.025;
  });

  return (
    <group ref={groupRef}>
      {/* Ground-truth image planes */}
      {cloud.planes?.map((plane, i) => (
        <TexturedPlane key={i} plane={plane} index={i} />
      ))}
      {/* Semantic terrain geometry */}
      <SemanticTerrain terrain={terrain} clipHeight={clipHeight} />
      {/* Thin object markers */}
      <ObjectMarkers objects={objects} clipHeight={clipHeight} />
    </group>
  );
};

interface PointCloudViewerProps {
  cloud: PointCloud;
  viewMode: 'rgb' | 'heatmap' | 'source';
  clipHeight: number;
  pointSize: number;
}

const PointCloudViewer: React.FC<PointCloudViewerProps> = ({ cloud, viewMode, clipHeight, pointSize }) => (
  <Canvas
    style={{ width: '100%', height: '100%', background: '#060b14' }}
    gl={{ antialias: true, alpha: true }}
  >
    <PerspectiveCamera makeDefault position={[0, 80, 110]} fov={55} />
    <ambientLight intensity={0.8} />
    <directionalLight position={[50, 100, 50]} intensity={0.6} castShadow />
    <Suspense fallback={null}>
      <ReconstructedMesh cloud={cloud} viewMode={viewMode} clipHeight={clipHeight} pointSize={pointSize} />
    </Suspense>
    <OrbitControls
      enableDamping
      dampingFactor={0.08}
      minDistance={10}
      maxDistance={500}
      makeDefault
    />
      {/* Grid floor scaled to match scene — planes span 96 wide x 85 deep */}
      <gridHelper args={[200, 40, '#0f2040', '#0d1a2e']} />
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
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
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
}

const Tier2View: React.FC<Tier2ViewProps> = ({ viewMode, clipHeight, pointSize }) => {
  const [cloud, setCloud] = useState<PointCloud | null>(null);
  const [epoch, setEpoch] = useState(0);

  useEffect(() => {
    // Load epoch from training_status.json
    fetch('/training_status.json')
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setEpoch(d.epoch ?? 0); })
      .catch(() => {});

    // Load point cloud
    fetch('/pointcloud.json')
      .then(r => r.ok ? r.json() : null)
      .then((d: PointCloud | null) => { if (d) setCloud(d); })
      .catch(() => {});
  }, []);

  if (!cloud) return <PointCloudPlaceholder epoch={epoch} />;

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <PointCloudViewer cloud={cloud} viewMode={viewMode} clipHeight={clipHeight} pointSize={pointSize} />

      {/* Info overlay */}
      <div style={{
        position: 'absolute', top: 10, left: '50%', transform: 'translateX(-50%)',
        display: 'flex', gap: 8, pointerEvents: 'none', flexWrap: 'wrap', justifyContent: 'center',
      }}>
        {[
          { label: 'EPOCH',   value: String(cloud.stats?.epochAtExport ?? cloud.epochAtExport ?? '–') },
          { label: 'mAP50',   value: `${((cloud.stats?.mAP50AtExport ?? cloud.mAP50AtExport ?? 0) * 100).toFixed(1)}%` },
          { label: 'TERRAIN', value: (cloud.stats?.numTerrain ?? cloud.terrain?.length ?? 0).toLocaleString() },
          { label: 'OBJECTS', value: (cloud.stats?.numObjects ?? cloud.objects?.length ?? 0).toLocaleString() },
          { label: 'DEPTH',   value: (cloud.depthBackend ?? 'n/a').toUpperCase() },
          { label: 'MODEL',   value: (cloud.stats?.modelName ?? cloud.modelName ?? 'YOLO').toUpperCase() },
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
        fontFamily: "'JetBrains Mono', monospace", fontSize: 9, color: '#334155',
        pointerEvents: 'none',
      }}>
        drag to rotate · scroll to zoom · right-drag to pan
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
  const [plan, setPlan]               = useState<FlightPlan | null>(null);
  const [seqIdx, setSeqIdx]           = useState(0);
  const [frameIdx, setFrameIdx]       = useState(0);
  const [imgLoaded, setImgLoaded]     = useState(false);
  const [imgNat, setImgNat]           = useState({ w: 0, h: 0 });
  const [boxes, setBoxes]             = useState<BBox[]>([]);
  const [playing, setPlaying]         = useState(true);
  const [containerSize, setContainerSize] = useState({ w: 0, h: 0 });
  const containerRef  = useRef<HTMLDivElement>(null);
  const intervalRef   = useRef<ReturnType<typeof setInterval> | null>(null);

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

  const currentSeq   = plan?.sequences[seqIdx] ?? null;
  const totalFrames  = currentSeq?.frames.length ?? 0;
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
  const trajectoryPts     = currentSeq?.trajectoryPoints ?? [];
  const activeWaypointIdx = trajectoryPts.length > 0
    ? Math.floor((frameIdx / Math.max(totalFrames - 1, 1)) * (trajectoryPts.length - 1))
    : 0;
  const activeWpt = trajectoryPts[activeWaypointIdx];

  // Viewport area (exclude 70px controls bar at bottom)
  const viewH    = Math.max(1, containerSize.h - 70);
  const viewW    = Math.max(1, containerSize.w);
  const rendered = getRenderedRect(viewW, viewH, imgNat.w || viewW, imgNat.h || viewH);

  // RIGHT / BOTTOM bounds of the rendered image area (absolute in container)
  const renderedRight  = rendered.left + rendered.w;
  const renderedBottom = rendered.top  + rendered.h;

  const imgUrl = currentFrame ? `/drone/${currentFrame.filename}` : null;

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
        const rawTop  = rendered.top  + (box.y / imgNat.h) * rendered.h;
        const rawW    = (box.w / imgNat.w) * rendered.w;
        const rawH    = (box.h / imgNat.h) * rendered.h;

        // Clamp left & top to stay inside rendered rect
        const left = Math.max(rendered.left, rawLeft);
        const top  = Math.max(rendered.top,  rawTop);

        // Clamp right edge: box must not exceed rendered image right/bottom
        const right  = Math.min(renderedRight,  rawLeft + Math.max(rawW, 6));
        const bottom = Math.min(renderedBottom, rawTop  + Math.max(rawH, 6));

        const w = right  - left;
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
              { top: -1, left: -1,   borderTop: `2px solid ${box.color}`, borderLeft:   `2px solid ${box.color}` },
              { top: -1, right: -1,  borderTop: `2px solid ${box.color}`, borderRight:  `2px solid ${box.color}` },
              { bottom: -1, left: -1,  borderBottom: `2px solid ${box.color}`, borderLeft:  `2px solid ${box.color}` },
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
              <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
              <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
          </defs>

          {/* Path polyline */}
          <polyline
            points={trajectoryPts.map(p =>
              `${(p.vx / 100) * rendered.w},${(p.vy / 100) * rendered.h}`
            ).join(' ')}
            fill="none"
            stroke="rgba(56,189,248,0.35)"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Waypoint dots */}
          {trajectoryPts.map((p, i) => (
            <circle
              key={i}
              cx={(p.vx / 100) * rendered.w}
              cy={(p.vy / 100) * rendered.h}
              r={i === activeWaypointIdx ? 5 : 2.5}
              fill={i === activeWaypointIdx ? '#38bdf8' : 'rgba(56,189,248,0.5)'}
              filter={i === activeWaypointIdx ? 'url(#glow)' : undefined}
            />
          ))}

          {/* Active UAV ring */}
          {activeWpt && (
            <circle
              cx={(activeWpt.vx / 100) * rendered.w}
              cy={(activeWpt.vy / 100) * rendered.h}
              r={13} fill="none" stroke="#38bdf8" strokeWidth="1.5" opacity="0.4"
            />
          )}
        </svg>
      )}

      {/* Scan line */}
      <ScanLine />

      {/* Corner brackets */}
      {[
        { top: 12, left: 12,   borderTop: '2px solid rgba(56,189,248,0.45)', borderLeft:   '2px solid rgba(56,189,248,0.45)', width: 20, height: 20 },
        { top: 12, right: 12,  borderTop: '2px solid rgba(56,189,248,0.45)', borderRight:  '2px solid rgba(56,189,248,0.45)', width: 20, height: 20 },
        { bottom: 82, left: 12,  borderBottom: '2px solid rgba(56,189,248,0.45)', borderLeft:  '2px solid rgba(56,189,248,0.45)', width: 20, height: 20 },
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
          { label: 'SEQ',   value: currentSeq?.seqId ?? '–' },
          { label: 'FRAME', value: totalFrames > 0 ? `${frameIdx + 1}/${totalFrames}` : '–' },
          { label: 'ALT',   value: activeWpt ? `${activeWpt.alt.toFixed(1)}m` : '–' },
          { label: 'LAT',   value: activeWpt ? activeWpt.lat.toFixed(5) : '–' },
          { label: 'LON',   value: activeWpt ? activeWpt.lon.toFixed(5) : '–' },
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
          <button onClick={goBack}  style={btnStyle(false)}><SkipBack  size={12} /></button>
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
    return <Tier2View viewMode={viewMode} clipHeight={clipHeight} pointSize={pointSize} />;
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
