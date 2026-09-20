import React, { useMemo } from 'react';
import { useLoader } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';

export interface BuildingEntity {
  id: string;
  type: string;
  footprint: {
    cx: number;
    cz: number;
    w: number;
    d: number;
    heading: number;
  };
  measuredHeight: number;
  floorCount: number;
  confidence: number;
  uncertainty?: number;
  lod?: number;
  source?: string;
  frame?: number;
}

export interface TreeEntity {
  id: string;
  position: [number, number, number];
  height: number;
  radius: number;
}

export interface VehicleEntity {
  id: string;
  type: string;
  position: [number, number, number];
  heading: number;
  conf: number;
}

export interface EntitiesData {
  buildings?: BuildingEntity[];
  vegetation?: TreeEntity[];
  vehicles?: VehicleEntity[];
}

interface DynamicEntityManagerProps {
  entities: EntitiesData;
  clipHeight?: number;
}

// ── Model Paths ─────────────────────────────────────────────────────────────
const MODEL_PATHS = {
  office_tower:          '/models/commercial/building-skyscraper-a.glb',
  high_rise_residential: '/models/commercial/building-skyscraper-b.glb',
  commercial_shopfront:  '/models/commercial/building-b.glb',
  apartment_block:       '/models/commercial/building-a.glb',
  villa_house:           '/models/suburban/building-type-a.glb',
  industrial_warehouse:  '/models/commercial/low-detail-building-wide-a.glb',
  mixed_use:             '/models/commercial/building-c.glb',
  tree_large:            '/models/vegetation/tree-large.glb',
  tree_small:            '/models/vegetation/tree-small.glb',
};

// ── Realistic Building Color Palette Fallbacks ──────────────────────────────
const TYPE_ACCENT_COLORS: Record<string, string> = {
  office_tower:          '#38bdf8',  // Cyan glass tint
  high_rise_residential: '#f59e0b',  // Warm amber brick
  commercial_shopfront:  '#10b981',  // Emerald retail trim
  apartment_block:       '#a78bfa',  // Purple slate
  villa_house:           '#fb923c',  // Terracotta roof
  industrial_warehouse:  '#94a3b8',  // Steel gray
  mixed_use:             '#0ea5e9',
};

// ── Single Building Instance with Texture & Real Height ─────────────────────
const DynamicBuildingInstance: React.FC<{
  building: BuildingEntity;
  modelUrl: string;
  colormap: THREE.Texture;
}> = ({ building, modelUrl, colormap }) => {
  const { scene } = useGLTF(modelUrl);

  // Realistic metric height calculation based on building type & floors:
  const actualHeight = useMemo(() => {
    const raw = building.measuredHeight || 3.5;
    switch (building.type) {
      case 'office_tower':
        return Math.max(raw * 3.8, 22.0);
      case 'high_rise_residential':
        return Math.max(raw * 3.4, 20.0);
      case 'apartment_block':
      case 'mixed_use':
        return Math.max(raw * 2.6, 13.0);
      case 'commercial_shopfront':
        return Math.max(raw * 2.2, 9.5);
      case 'industrial_warehouse':
        return Math.max(raw * 1.9, 8.0);
      case 'villa_house':
      default:
        return Math.max(raw * 1.6, 6.0);
    }
  }, [building.measuredHeight, building.type]);

  const { cloned, scale, position, foundationW, foundationD } = useMemo(() => {
    // Clone scene and apply textures + materials
    const clonedScene = scene.clone(true);
    const accentHex = TYPE_ACCENT_COLORS[building.type] || '#38bdf8';
    const accentColor = new THREE.Color(accentHex);

    clonedScene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        if (mesh.material) {
          const origMat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
          const mat = (origMat as THREE.MeshStandardMaterial).clone();

          // Apply texture colormap if available
          mat.map = colormap;
          mat.roughness = 0.55;
          mat.metalness = 0.2;

          // Subtle ambient tint matching architectural style so buildings look rich and distinct
          mat.color.lerp(accentColor, 0.18);
          mat.needsUpdate = true;
          mesh.material = mat;
        }
      }
    });

    const box = new THREE.Box3().setFromObject(clonedScene);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    // Target dimensions from AI reconstruction footprint
    const targetW = Math.max(building.footprint.w, 5.0);
    const targetD = Math.max(building.footprint.d, 5.0);
    const targetH = actualHeight;

    const sx = targetW / Math.max(size.x, 0.5);
    const sy = targetH / Math.max(size.y, 0.5);
    const sz = targetD / Math.max(size.z, 0.5);

    return {
      cloned: clonedScene,
      scale: [sx, sy, sz] as [number, number, number],
      position: [
        building.footprint.cx - center.x * sx,
        -box.min.y * sy,
        building.footprint.cz - center.z * sz,
      ] as [number, number, number],
      foundationW: targetW * 1.04,
      foundationD: targetD * 1.04,
    };
  }, [scene, building, actualHeight, colormap]);

  return (
    <group position={position}>
      {/* 3D Modular Building Mesh */}
      <primitive object={cloned} scale={scale} />

      {/* Dark Slate Base Foundation (snaps building firmly into ground) */}
      <mesh position={[0, 0.1, 0]}>
        <boxGeometry args={[foundationW, 0.2, foundationD]} />
        <meshStandardMaterial color="#1e293b" roughness={0.9} />
      </mesh>
    </group>
  );
};

// ── Single Tree Instance with Natural Foliage ───────────────────────────────
const DynamicTreeInstance: React.FC<{
  tree: TreeEntity;
  modelUrl: string;
  colormap: THREE.Texture;
}> = ({ tree, modelUrl, colormap }) => {
  const { scene } = useGLTF(modelUrl);

  const { cloned, scale, position } = useMemo(() => {
    const clonedScene = scene.clone(true);
    clonedScene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        if (mesh.material) {
          const mat = ((mesh.material as any).clone()) as THREE.MeshStandardMaterial;
          mat.map = colormap;
          mat.color.set('#22c55e'); // Rich green foliage
          mat.needsUpdate = true;
          mesh.material = mat;
        }
      }
    });

    const box = new THREE.Box3().setFromObject(clonedScene);
    const size = box.getSize(new THREE.Vector3());

    const targetH = Math.max(tree.height * 1.5, 3.5);
    const targetR = Math.max(tree.radius * 2.2, 2.5);

    const sx = targetR / Math.max(size.x, 0.5);
    const sy = targetH / Math.max(size.y, 0.5);
    const sz = targetR / Math.max(size.z, 0.5);

    return {
      cloned: clonedScene,
      scale: [sx, sy, sz] as [number, number, number],
      position: [
        tree.position[0],
        -box.min.y * sy,
        tree.position[2],
      ] as [number, number, number],
    };
  }, [scene, tree, colormap]);

  return (
    <group position={position}>
      <primitive object={cloned} scale={scale} />
    </group>
  );
};

// ── Master Entity Manager Component ──────────────────────────────────────────
export const DynamicEntityManager: React.FC<DynamicEntityManagerProps> = ({
  entities,
  clipHeight = 150,
}) => {
  const buildings = entities.buildings ?? [];
  const vegetation = entities.vegetation ?? [];

  // Load colormap texture
  const colormap = useLoader(THREE.TextureLoader, '/models/commercial/colormap.png');
  colormap.colorSpace = THREE.SRGBColorSpace;
  colormap.flipY = false;

  const visibleBuildings = useMemo(() => {
    return buildings.filter((b) => (b.measuredHeight || 3.5) <= clipHeight);
  }, [buildings, clipHeight]);

  const visibleTrees = useMemo(() => {
    return vegetation.slice(0, 70); // Render primary trees cleanly
  }, [vegetation]);

  return (
    <group>
      {/* 3D Buildings with realistic heights & textures */}
      {visibleBuildings.map((bldg) => {
        let modelUrl = MODEL_PATHS.apartment_block;
        if (bldg.type in MODEL_PATHS) {
          modelUrl = MODEL_PATHS[bldg.type as keyof typeof MODEL_PATHS];
        } else if (bldg.measuredHeight > 15) {
          modelUrl = MODEL_PATHS.office_tower;
        } else if (bldg.measuredHeight < 6) {
          modelUrl = MODEL_PATHS.villa_house;
        }

        return (
          <React.Suspense key={bldg.id} fallback={null}>
            <DynamicBuildingInstance
              building={bldg}
              modelUrl={modelUrl}
              colormap={colormap}
            />
          </React.Suspense>
        );
      })}

      {/* 3D Trees */}
      {visibleTrees.map((tree, idx) => {
        const modelUrl = idx % 2 === 0 ? MODEL_PATHS.tree_large : MODEL_PATHS.tree_small;
        return (
          <React.Suspense key={tree.id} fallback={null}>
            <DynamicTreeInstance
              tree={tree}
              modelUrl={modelUrl}
              colormap={colormap}
            />
          </React.Suspense>
        );
      })}
    </group>
  );
};

// Preload models for instant rendering
Object.values(MODEL_PATHS).forEach((url) => {
  useGLTF.preload(url);
});
