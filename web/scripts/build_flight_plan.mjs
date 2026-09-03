/**
 * build_flight_plan.mjs
 *
 * Reads the VisDrone val images directory and GPS telemetry, then outputs
 * a flight_plan.json consumed by the frontend MeshViewer.
 *
 * Run once: node scripts/build_flight_plan.mjs
 * (Re-run whenever new data arrives — does NOT touch training.)
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const IMAGES_DIR = path.join(ROOT, '../VisDrone2019-DET-val/VisDrone2019-DET-val/images');
const GPS_FILE   = path.join(ROOT, '../sample_flight_data/gps_telemetry.json');
const OUT_FILE   = path.join(ROOT, 'public/flight_plan.json');

// ── 1. Parse all image filenames into structured frames ─────────────────────
const files = fs.readdirSync(IMAGES_DIR).filter(f => f.endsWith('.jpg'));

// Filename format: {seqId}_{frameOffset}_{difficulty}_{frameIdx}.jpg
const frames = files.map(fname => {
  const stem = fname.replace('.jpg', '');
  const parts = stem.split('_');
  return {
    filename: fname,
    seqId:       parts[0],
    frameOffset: parseInt(parts[1], 10),
    difficulty:  parts[2],
    frameIdx:    parseInt(parts[3], 10),
  };
});

// ── 2. Group by sequence, sort each group by frameOffset ────────────────────
const bySeq = {};
for (const f of frames) {
  if (!bySeq[f.seqId]) bySeq[f.seqId] = [];
  bySeq[f.seqId].push(f);
}
for (const seq of Object.values(bySeq)) {
  seq.sort((a, b) => a.frameOffset - b.frameOffset);
}

// ── 3. Pick the longest 5 sequences for a rich demo playback ─────────────────
const sequences = Object.entries(bySeq)
  .map(([seqId, frms]) => ({ seqId, frames: frms }))
  .sort((a, b) => b.frames.length - a.frames.length)
  .slice(0, 5);

// ── 4. Build real GPS trajectory from actual 3D positions ────────────────────
//
// GPS telemetry has two position formats:
//   pos_xyz: [x, y, z] in metres (ENU frame) – much larger spread (~15m range)
//   lat_lon_alt: geographic coords – tiny spread (~0.00014° = 15m)
//
// We use pos_xyz for the viewport projection because it has meaningful spatial
// spread across the image.
//
const gps = JSON.parse(fs.readFileSync(GPS_FILE, 'utf8'));

const xs = gps.map(p => p.pos_xyz[0]);
const ys = gps.map(p => p.pos_xyz[1]);
const xMin = Math.min(...xs), xMax = Math.max(...xs);
const yMin = Math.min(...ys), yMax = Math.max(...ys);
const xRange = xMax - xMin || 1;
const yRange = yMax - yMin || 1;

// Map ENU x → image vx (left-right), ENU y → image vy (top-bottom, inverted)
// Clamp to [8, 92] so the path never goes outside the image area.
const PAD = 8;
const exactTrajectoryPoints = gps.map(pt => ({
  timestamp: pt.timestamp,
  lat: pt.lat_lon_alt[0],
  lon: pt.lat_lon_alt[1],
  alt: pt.lat_lon_alt[2],
  vx: PAD + ((pt.pos_xyz[0] - xMin) / xRange) * (100 - 2 * PAD),
  // ENU y increases northward; image y increases downward → flip
  vy: PAD + ((1 - (pt.pos_xyz[1] - yMin) / yRange)) * (100 - 2 * PAD),
}));

// Verify all points are in bounds
for (const p of exactTrajectoryPoints) {
  if (p.vx < 0 || p.vx > 100 || p.vy < 0 || p.vy > 100) {
    console.warn(`  ⚠ Out-of-bounds point: vx=${p.vx.toFixed(1)} vy=${p.vy.toFixed(1)}`);
  }
}

const finalSequences = sequences.map(s => ({
  seqId: s.seqId,
  frameCount: s.frames.length,
  frames: s.frames.map(f => ({
    filename: f.filename,
    frameOffset: f.frameOffset,
  })),
  // Every sequence shares the same real GPS path — this is correct because
  // the trajectory shows the DRONE's planned survey path over any given area.
  trajectoryPoints: exactTrajectoryPoints,
}));

// ── 5. Write output ──────────────────────────────────────────────────────────
const output = {
  generatedAt: new Date().toISOString(),
  totalImages: frames.length,
  sequences: finalSequences,
  originLatLon: [gps[0].lat_lon_alt[0], gps[0].lat_lon_alt[1]],
};

fs.writeFileSync(OUT_FILE, JSON.stringify(output, null, 2));
console.log(`✓ flight_plan.json written → ${OUT_FILE}`);
console.log(`  ${sequences.length} sequences`);
sequences.forEach(s => console.log(`  seq ${s.seqId}: ${s.frames.length} frames`));
console.log(`  Trajectory: ${exactTrajectoryPoints.length} waypoints`);
console.log(`  vx range: [${Math.min(...exactTrajectoryPoints.map(p=>p.vx)).toFixed(1)}, ${Math.max(...exactTrajectoryPoints.map(p=>p.vx)).toFixed(1)}]`);
console.log(`  vy range: [${Math.min(...exactTrajectoryPoints.map(p=>p.vy)).toFixed(1)}, ${Math.max(...exactTrajectoryPoints.map(p=>p.vy)).toFixed(1)}]`);
