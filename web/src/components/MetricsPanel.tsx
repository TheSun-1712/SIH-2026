import React, { useMemo } from 'react';
import { TrendingUp, Target, Clock, Layers, MapPin, Cpu } from 'lucide-react';

interface MetricsPanelProps {
  tierMode: 'tier1' | 'tier2';
  viewMode: 'rgb' | 'heatmap' | 'source';
}

// ─── Mini SVG Donut Chart ──────────────────────────────────────
interface DonutSlice {
  value: number;
  color: string;
  label: string;
}

const DonutChart: React.FC<{ slices: DonutSlice[]; size?: number }> = ({
  slices, size = 68,
}) => {
  const r = 24;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * r;

  const total = slices.reduce((s, sl) => s + sl.value, 0);
  let offset = 0;

  const arcs = slices.map((sl) => {
    const dash = (sl.value / total) * circumference;
    const gap = circumference - dash;
    const arc = { ...sl, dash, gap, offset };
    offset += dash;
    return arc;
  });

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#1e293b" strokeWidth={8} />
      {arcs.map((arc, i) => (
        <circle
          key={i}
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke={arc.color}
          strokeWidth={7}
          strokeDasharray={`${arc.dash} ${arc.gap}`}
          strokeDashoffset={-arc.offset + circumference * 0.25}
          style={{ transition: 'stroke-dasharray 0.6s ease' }}
        />
      ))}
      <circle cx={cx} cy={cy} r={18} fill="#090d16" />
    </svg>
  );
};

// ─── Mini bar chart ────────────────────────────────────────────
const MiniBar: React.FC<{
  value: number; max: number; color: string; label: string;
}> = ({ value, max, color, label }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px' }}>
    <span style={{ color: '#64748b', width: '68px', flexShrink: 0 }}>{label}</span>
    <div style={{
      flex: 1, height: '5px', background: '#1e293b', borderRadius: '3px', overflow: 'hidden',
    }}>
      <div style={{
        width: `${Math.min((value / max) * 100, 100)}%`,
        height: '100%', background: color, borderRadius: '3px',
        transition: 'width 0.8s cubic-bezier(.4,0,.2,1)',
        boxShadow: `0 0 6px ${color}80`,
      }} />
    </div>
    <span style={{ color, fontWeight: 700, width: '30px', textAlign: 'right' }}>
      {value.toFixed(1)}
    </span>
  </div>
);

// ─── Stat card ─────────────────────────────────────────────────
const StatCard: React.FC<{
  icon: React.ReactNode; value: string; label: string; color?: string;
}> = ({ icon, value, label, color = '#38bdf8' }) => (
  <div style={{
    display: 'flex', flexDirection: 'column', gap: '4px',
    background: 'rgba(15,23,42,0.6)', padding: '10px 12px',
    borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)',
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#64748b' }}>
      {icon}
      <span style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
        {label}
      </span>
    </div>
    <span style={{ fontSize: '20px', fontWeight: 800, color, letterSpacing: '-0.5px' }}>
      {value}
    </span>
  </div>
);

// ─── Main component ────────────────────────────────────────────
export const MetricsPanel: React.FC<MetricsPanelProps> = ({ tierMode, viewMode }) => {
  const metrics = useMemo(() => {
    const isT2 = tierMode === 'tier2';
    return {
      totalPoints: isT2 ? '38.2K' : '4.0K',
      keyframes: isT2 ? '10 / 15' : '5 / 15',
      spatialCoverage: isT2 ? 96.4 : 78.2,
      rmseM: isT2 ? 0.12 : 0.31,
      eskfDrift: 0.038,
      processingTime: isT2 ? '4m 12s' : '0m 48s',
      gpsAccuracy: '0.84m',
      confidence: {
        observed: isT2 ? 65.2 : 45.0,
        monoInfill: isT2 ? 24.1 : 38.5,
        hallucinated: isT2 ? 10.7 : 16.5,
      },
      backend: isT2 ? 'LightGlue + DA2 V2' : 'SIFT + DA2 V2',
      device: 'CUDA (RTX 3080)',
    };
  }, [tierMode]);

  const donutSlices: DonutSlice[] = [
    { value: metrics.confidence.observed, color: '#38bdf8', label: 'MVS Obs.' },
    { value: metrics.confidence.monoInfill, color: '#f59e0b', label: 'Mono Infill' },
    { value: metrics.confidence.hallucinated, color: '#f43f5e', label: '3DGS Halluc.' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>

      {/* Stat grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
        <StatCard
          icon={<Layers size={11} />}
          value={metrics.totalPoints}
          label="Total Vertices"
          color="#38bdf8"
        />
        <StatCard
          icon={<Target size={11} />}
          value={`${metrics.rmseM}m`}
          label="Est. RMSE"
          color="#10b981"
        />
        <StatCard
          icon={<MapPin size={11} />}
          value={metrics.keyframes}
          label="Keyframes"
          color="#a78bfa"
        />
        <StatCard
          icon={<Clock size={11} />}
          value={metrics.processingTime}
          label="Pipeline Time"
          color="#f59e0b"
        />
      </div>

      {/* Confidence distribution */}
      <div style={{
        background: 'rgba(15,23,42,0.5)', borderRadius: '8px',
        padding: '12px', border: '1px solid rgba(255,255,255,0.05)',
      }}>
        <span style={{ fontSize: '11px', fontWeight: 600, color: '#94a3b8' }}>
          Confidence Distribution
        </span>
        <div style={{
          display: 'flex', alignItems: 'center', gap: '14px', marginTop: '10px',
        }}>
          <DonutChart slices={donutSlices} size={68} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', flex: 1 }}>
            {donutSlices.map((sl) => (
              <div key={sl.label} style={{
                display: 'flex', alignItems: 'center',
                justifyContent: 'space-between', fontSize: '11px',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <div style={{
                    width: '8px', height: '8px', borderRadius: '50%',
                    background: sl.color, boxShadow: `0 0 6px ${sl.color}`,
                  }} />
                  <span style={{ color: '#94a3b8' }}>{sl.label}</span>
                </div>
                <span style={{ color: sl.color, fontWeight: 700 }}>
                  {sl.value.toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Accuracy bars */}
      <div style={{
        background: 'rgba(15,23,42,0.5)', borderRadius: '8px',
        padding: '12px', border: '1px solid rgba(255,255,255,0.05)',
        display: 'flex', flexDirection: 'column', gap: '8px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
          <TrendingUp size={12} style={{ color: '#38bdf8' }} />
          <span style={{ fontSize: '11px', fontWeight: 600, color: '#94a3b8' }}>
            Accuracy Metrics
          </span>
        </div>
        <MiniBar value={metrics.spatialCoverage} max={100} color="#38bdf8" label="Coverage" />
        <MiniBar value={(1 - metrics.rmseM / 2) * 100} max={100} color="#10b981" label="Geo Acc." />
        <MiniBar value={(1 - metrics.eskfDrift / 0.2) * 100} max={100} color="#a78bfa" label="Pose Acc." />
        <MiniBar value={72.5} max={100} color="#f59e0b" label="Completeness" />
      </div>

      {/* GPU / Backend info */}
      <div style={{
        background: 'rgba(15,23,42,0.4)', borderRadius: '8px',
        padding: '10px 12px', border: '1px solid rgba(255,255,255,0.04)',
        display: 'flex', flexDirection: 'column', gap: '5px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Cpu size={11} style={{ color: '#38bdf8' }} />
          <span style={{ fontSize: '10px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase' }}>
            Compute Backend
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
          <span style={{ color: '#64748b' }}>Matcher</span>
          <span style={{ color: '#f1f5f9', fontWeight: 600 }}>{metrics.backend}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
          <span style={{ color: '#64748b' }}>Device</span>
          <span style={{ color: '#10b981', fontWeight: 600 }}>{metrics.device}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
          <span style={{ color: '#64748b' }}>GPS Accuracy</span>
          <span style={{ color: '#f59e0b', fontWeight: 600 }}>{metrics.gpsAccuracy}</span>
        </div>
      </div>
    </div>
  );
};
