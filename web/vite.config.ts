import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import fs from 'fs'
import path from 'path'

const DRONE_IMAGES_DIR = path.resolve(
  __dirname,
  '../VisDrone2019-DET-val/VisDrone2019-DET-val/images'
);

const TRAINING_DIR = path.resolve(
  __dirname,
  '../aero_mesh/training/runs/visdrone_fast'
);

const BUILDINGS_GLB = path.resolve(
  __dirname,
  '../full-gameready-city-buildings/source/full_gameready_city_buildings.glb'
);

const TREE_FBX = path.resolve(
  __dirname,
  '../mystical-x-tree-viii/source/55-3_png.fbx'
);

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'serve-drone-data',
      configureServer(server) {

        // ── Serve real VisDrone images at /drone/<filename> ──────────────
        server.middlewares.use('/drone', (req, res) => {
          const filename = req.url?.replace(/^\//, '') ?? '';
          const filePath = path.join(DRONE_IMAGES_DIR, filename);
          if (fs.existsSync(filePath)) {
            res.setHeader('Content-Type', 'image/jpeg');
            res.setHeader('Cache-Control', 'public, max-age=3600');
            fs.createReadStream(filePath).pipe(res as any);
          } else {
            res.statusCode = 404;
            res.end('Image not found');
          }
        });

        // ── Serve real VisDrone annotations at /api/annotation/<file.txt> ─
        server.middlewares.use('/api/annotation', (req, res) => {
          const filename = req.url?.replace(/^\//, '') ?? '';
          const annotDir = path.resolve(
            DRONE_IMAGES_DIR,
            '../annotations'
          );
          const filePath = path.join(annotDir, filename);
          if (fs.existsSync(filePath)) {
            res.setHeader('Content-Type', 'text/plain');
            res.setHeader('Access-Control-Allow-Origin', '*');
            res.end(fs.readFileSync(filePath, 'utf8'));
          } else {
            res.statusCode = 404;
            res.end('');
          }
        });

        // ── Serve live training results CSV at /api/results ──────────────
        server.middlewares.use('/api/results', (_req, res) => {
          const csvPath = path.join(TRAINING_DIR, 'results.csv');
          if (fs.existsSync(csvPath)) {
            res.setHeader('Content-Type', 'text/csv');
            res.setHeader('Cache-Control', 'no-cache');
            res.end(fs.readFileSync(csvPath));
          } else {
            res.statusCode = 404;
            res.end('Not found');
          }
        });

        // ── Serve YOLO training batch images at /api/batch/<file> ────────
        server.middlewares.use('/api/batch', (req, res) => {
          const filename = req.url?.replace(/^\//, '') ?? '';
          const filePath = path.join(TRAINING_DIR, filename);
          if (fs.existsSync(filePath) && filename.endsWith('.jpg')) {
            res.setHeader('Content-Type', 'image/jpeg');
            res.setHeader('Cache-Control', 'no-cache');
            fs.createReadStream(filePath).pipe(res as any);
          } else {
            res.statusCode = 404;
            res.end('Not found');
          }
        });
        // ── Serve 3D city buildings GLB at /api/assets/buildings.glb ────────
        server.middlewares.use('/api/assets/buildings.glb', (_req, res) => {
          if (fs.existsSync(BUILDINGS_GLB)) {
            res.setHeader('Content-Type', 'model/gltf-binary');
            res.setHeader('Cache-Control', 'public, max-age=86400');
            res.setHeader('Access-Control-Allow-Origin', '*');
            fs.createReadStream(BUILDINGS_GLB).pipe(res as any);
          } else {
            res.statusCode = 404;
            res.end('buildings.glb not found');
          }
        });

        // ── Serve tree FBX at /api/assets/tree.fbx ────────────────────────────
        server.middlewares.use('/api/assets/tree.fbx', (_req, res) => {
          if (fs.existsSync(TREE_FBX)) {
            res.setHeader('Content-Type', 'application/octet-stream');
            res.setHeader('Cache-Control', 'public, max-age=86400');
            res.setHeader('Access-Control-Allow-Origin', '*');
            fs.createReadStream(TREE_FBX).pipe(res as any);
          } else {
            res.statusCode = 404;
            res.end('tree.fbx not found');
          }
        });
      },
    },
  ],
})
