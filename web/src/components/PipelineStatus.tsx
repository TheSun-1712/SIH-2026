import React, { useEffect, useState, useRef } from 'react';
import {
  CheckCircle2, Loader2, Circle, AlertTriangle,
  Cpu, Layers, ScanLine, Zap, Eye, Map, Database,
} from 'lucide-react';

interface Stage {
  id: number;
  name: string;
  label: string;
  detail: string;
  icon: React.ReactNode;
  durationMs: number;
  status: 'pending' | 'running' | 'done' | 'warn';
}

const PIPELINE_STAGES: Omit<Stage, 'status'>[] = [
  {
    id: 1, name: 'ingest', label: 'Frame Extraction & Blur Filter',
    detail: 'Laplacian sharpness · BRISQUE · ffmpeg 2fps',
    icon: <ScanLine size={13} />, durationMs: 1200,
  },
  {
    id: 2, name: 'eskf', label: '15-State ESKF Sensor Fusion',
    detail: 'GPS + IMU + Baro → smooth 6-DoF poses',
    icon: <Cpu size={13} />, durationMs: 900,
  },
  {
    id: 3, name: 'sfm', label: 'SuperPoint+LightGlue SfM',
    detail: 'GPU feature matching · COLMAP Bundle Adj.',
    icon: <Layers size={13} />, durationMs: 2200,
  },
  {
    id: 4, name: 'depth', label: 'Depth Anything V2 Metric',
    detail: 'ViT-L · CUDA · WLS scale-shift alignment',
    icon: <Eye size={13} />, durationMs: 1800,
  },
  {
    id: 5, name: 'motion', label: 'RAFT-Large + SAM2 Motion Mask',
    detail: 'Ego-motion compensation · Instance masking',
    icon: <Zap size={13} />, durationMs: 1500,
  },
  {
    id: 6, name: 'tsdf', label: 'Open3D TSDF Volumetric Fusion',
    detail: 'Scalable voxel hashing · 5cm resolution',
    icon: <Database size={13} />, durationMs: 2000,
  },
  {
    id: 7, name: 'splat', label: '3DGS Occlusion Completion',
    detail: 'Voxel gap detection · KNN color interpolation',
    icon: <Map size={13} />, durationMs: 1600,
  },
  {
    id: 8, name: 'confidence', label: 'Confidence Map + Georeferencing',
    detail: 'Multi-factor scoring · WGS84 transform',
    icon: <CheckCircle2 size={13} />, durationMs: 700,
  },
];

const STATUS_COLORS = {
  pending: '#334155',
  running: '#f59e0b',
  done: '#10b981',
  warn: '#f59e0b',
};

const STATUS_ICONS = {
  pending: <Circle size={13} style={{ color: '#475569' }} />,
  running: <Loader2 size={13} style={{ color: '#f59e0b', animation: 'spin 1s linear infinite' }} />,
  done: <CheckCircle2 size={13} style={{ color: '#10b981' }} />,
  warn: <AlertTriangle size={13} style={{ color: '#f59e0b' }} />,
};

const STATUS_LABELS = {
  pending: 'QUEUED',
  running: 'RUNNING',
  done: 'DONE',
  warn: 'WARN',
};

interface PipelineStatusProps {
  autoRun?: boolean;
  loopInterval?: number; // ms between full re-runs (for demo mode)
}

export const PipelineStatus: React.FC<PipelineStatusProps> = ({
  autoRun = true,
  loopInterval = 18000,
}) => {
  const [stages, setStages] = useState<Stage[]>(
    PIPELINE_STAGES.map(s => ({ ...s, status: 'pending' as const }))
  );
  const [currentIdx, setCurrentIdx] = useState(-1);
  const [totalElapsed, setTotalElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const loopRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const runPipeline = () => {
    setCurrentIdx(-1);
    setTotalElapsed(0);
    setStages(PIPELINE_STAGES.map(s => ({ ...s, status: 'pending' as const })));

    let cumDelay = 200;
    PIPELINE_STAGES.forEach((stage, i) => {
      // Start stage
      setTimeout(() => {
        setCurrentIdx(i);
        setStages(prev => prev.map((s, idx) =>
          idx === i ? { ...s, status: 'running' } : s
        ));
      }, cumDelay);

      cumDelay += stage.durationMs;

      // Complete stage (warn on stage 7 to indicate hallucination)
      const finalStatus: Stage['status'] = stage.id === 7 ? 'warn' : 'done';
      setTimeout(() => {
        setStages(prev => prev.map((s, idx) =>
          idx === i ? { ...s, status: finalStatus } : s
        ));
      }, cumDelay);
    });

    // Track total elapsed
    const interval = setInterval(() => setTotalElapsed(t => t + 100), 100);
    setTimeout(() => clearInterval(interval), cumDelay + 500);
  };

  useEffect(() => {
    if (!autoRun) return;

    runPipeline();

    loopRef.current = setInterval(runPipeline, loopInterval);
    return () => {
      if (loopRef.current) clearInterval(loopRef.current);
    };
  }, [autoRun, loopInterval]);

  const totalDone = stages.filter(s => s.status === 'done' || s.status === 'warn').length;
  const progress = (totalDone / PIPELINE_STAGES.length) * 100;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: '12px', fontWeight: 600, color: '#cbd5e1' }}>
          Pipeline Execution
        </span>
        <span style={{
          fontSize: '11px', color: '#64748b',
          fontVariantNumeric: 'tabular-nums',
        }}>
          {(totalElapsed / 1000).toFixed(1)}s
        </span>
      </div>

      {/* Progress bar */}
      <div style={{
        height: '4px', background: '#1e293b',
        borderRadius: '2px', overflow: 'hidden',
      }}>
        <div style={{
          height: '100%', width: `${progress}%`,
          background: 'linear-gradient(90deg, #0ea5e9, #6366f1)',
          borderRadius: '2px',
          transition: 'width 0.5s ease',
          boxShadow: '0 0 8px rgba(56,189,248,0.5)',
        }} />
      </div>

      {/* Stage list */}
      {stages.map((stage) => (
        <div
          key={stage.id}
          style={{
            display: 'flex', alignItems: 'flex-start', gap: '8px',
            padding: '8px 10px',
            background: stage.status === 'running'
              ? 'rgba(245,158,11,0.08)'
              : stage.status !== 'pending'
              ? 'rgba(15,23,42,0.6)'
              : 'rgba(15,23,42,0.3)',
            borderRadius: '7px',
            border: `1px solid ${stage.status === 'running'
              ? 'rgba(245,158,11,0.3)'
              : stage.status !== 'pending'
              ? 'rgba(255,255,255,0.05)'
              : 'transparent'}`,
            transition: 'all 0.25s ease',
          }}
        >
          {/* Stage icon */}
          <div style={{
            width: '22px', height: '22px', borderRadius: '5px', flexShrink: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: stage.status !== 'pending' ? 'rgba(56,189,248,0.12)' : 'rgba(30,41,59,0.6)',
            color: stage.status !== 'pending' ? '#38bdf8' : '#475569',
          }}>
            {stage.icon}
          </div>

          {/* Stage text */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{
              fontSize: '11.5px', fontWeight: 600,
              color: stage.status === 'pending' ? '#64748b' : '#f1f5f9',
              transition: 'color 0.25s',
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}>
              {`S${stage.id}: ${stage.label}`}
            </div>
            <div style={{ fontSize: '10px', color: '#475569', marginTop: '1px' }}>
              {stage.detail}
            </div>
          </div>

          {/* Status badge */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: '4px',
            fontSize: '10px', fontWeight: 700,
            color: STATUS_COLORS[stage.status],
            flexShrink: 0,
          }}>
            {STATUS_ICONS[stage.status]}
            <span style={{ display: stage.status === 'pending' ? 'none' : 'inline' }}>
              {STATUS_LABELS[stage.status]}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
};
