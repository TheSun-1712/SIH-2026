import React, { useEffect, useState } from 'react';
import { TrendingUp, Target, Activity, Cpu, CheckCircle2, AlertCircle } from 'lucide-react';

interface MetricsPanelProps {
  tierMode: 'tier1' | 'tier2';
  viewMode: 'rgb' | 'heatmap' | 'source';
}

// ─── Mini bar ─────────────────────────────────────────────────────────────────
const Bar: React.FC<{ value: number; max: number; color: string; label: string; display: string }> = ({
  value, max, color, label, display,
}) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px' }}>
    <span style={{ color: '#64748b', width: '72px', flexShrink: 0, lineHeight: 1.3 }}>{label}</span>
    <div style={{ flex: 1, height: '5px', background: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
      <div style={{
        width: `${Math.min(Math.max((value / max) * 100, 0), 100)}%`,
        height: '100%', background: color, borderRadius: '3px',
        transition: 'width 1s cubic-bezier(.4,0,.2,1)',
        boxShadow: `0 0 6px ${color}80`,
      }} />
    </div>
    <span style={{ color, fontWeight: 700, width: '44px', textAlign: 'right', fontFamily: "'JetBrains Mono', monospace" }}>
      {display}
    </span>
  </div>
);

// ─── Stat card ────────────────────────────────────────────────────────────────
const StatCard: React.FC<{ icon: React.ReactNode; value: string; label: string; sub?: string; color?: string }> = ({
  icon, value, label, sub, color = '#38bdf8',
}) => (
  <div style={{
    display: 'flex', flexDirection: 'column', gap: '3px',
    background: 'rgba(15,23,42,0.6)', padding: '10px 12px',
    borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)',
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '5px', color: '#64748b' }}>
      {icon}
      <span style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{label}</span>
    </div>
    <span style={{ fontSize: '19px', fontWeight: 800, color, letterSpacing: '-0.5px', lineHeight: 1 }}>{value}</span>
    {sub && <span style={{ fontSize: '9.5px', color: '#475569' }}>{sub}</span>}
  </div>
);

// ─── Donut chart ──────────────────────────────────────────────────────────────
const Donut: React.FC<{ slices: { v: number; color: string; label: string }[] }> = ({ slices }) => {
  const r = 24; const c = 34;
  const circ = 2 * Math.PI * r;
  const total = slices.reduce((s, sl) => s + sl.v, 0) || 1;
  let off = 0;
  return (
    <svg width={68} height={68} viewBox="0 0 68 68">
      <circle cx={c} cy={c} r={r} fill="none" stroke="#1e293b" strokeWidth={8} />
      {slices.map((sl, i) => {
        const dash = (sl.v / total) * circ;
        const node = (
          <circle key={i} cx={c} cy={c} r={r} fill="none" stroke={sl.color} strokeWidth={7}
            strokeDasharray={`${dash} ${circ - dash}`}
            strokeDashoffset={-(off) + circ * 0.25}
            style={{ transition: 'stroke-dasharray 0.8s ease' }}
          />
        );
        off += dash;
        return node;
      })}
      <circle cx={c} cy={c} r={18} fill="#090d16" />
    </svg>
  );
};

// ─── Main component ───────────────────────────────────────────────────────────
export const MetricsPanel: React.FC<MetricsPanelProps> = () => {
  const [d, setD] = useState({
    epoch: 0, totalEpochs: 100,
    mAP50: 0, mAP5095: 0,
    precision: 0, recall: 0,
    trainLoss: 0, valLoss: 0,
    trainingMin: 0,
  });
  const [live, setLive] = useState(false);
  // Live training meta from training_status.json
  const [status, setStatus] = useState<{
    modelName: string;
    nc: number;
    classNames: string[];
    nTrain: number;
    nVal: number;
    bestWeights: string;
    epochsCompleted: number;
  }>({
    modelName: '…', nc: 0, classNames: [],
    nTrain: 0, nVal: 0, bestWeights: '', epochsCompleted: 0,
  });

  useEffect(() => {
    // Primary: training_status.json (written by update_training_status.py)
    const fetchStatus = async () => {
      try {
        const res = await fetch('/training_status.json');
        if (!res.ok) return;
        const s = await res.json();
        setStatus({
          modelName:       s.modelName       ?? '…',
          nc:              s.nc              ?? 0,
          classNames:      s.classNames      ?? [],
          nTrain:          s.nTrain          ?? 0,
          nVal:            s.nVal            ?? 0,
          bestWeights:     s.bestWeights     ?? '',
          epochsCompleted: s.epochsCompleted ?? 0,
        });
        setLive(true);
        setD({
          epoch:       s.epoch       ?? 0,
          totalEpochs: s.totalEpochs ?? 100,
          mAP50:       s.mAP50       ?? 0,
          mAP5095:     s.mAP5095     ?? 0,
          precision:   s.precision   ?? 0,
          recall:      s.recall      ?? 0,
          trainLoss:   s.trainLoss   ?? 0,
          valLoss:     s.valLoss     ?? 0,
          trainingMin: s.trainingMin ?? 0,
        });
      } catch { /* ignore */ }
    };

    // Fallback: results.csv (direct parse)
    const fetchCsv = async () => {
      try {
        const res = await fetch('/api/results');
        if (!res.ok) return;
        const csv = await res.text();
        const lines = csv.trim().split('\n');
        if (lines.length < 2) return;
        const last = lines[lines.length - 1].split(',').map(v => parseFloat(v.trim()));
        if (last.length < 10) return;
        setLive(true);
        setD(prev => ({
          ...prev,
          epoch:       last[0] || 0,
          mAP50:       last[7] || 0,
          mAP5095:     last[8] || 0,
          precision:   last[5] || 0,
          recall:      last[6] || 0,
          trainLoss:   last[2] || 0,
          valLoss:     last[9] || 0,
          trainingMin: (last[1] || 0) / 60,
        }));
      } catch { /* ignore */ }
    };

    fetchStatus();
    fetchCsv();
    const iv = setInterval(() => { fetchStatus(); fetchCsv(); }, 5000);
    return () => clearInterval(iv);
  }, []);

  const progressPct = d.totalEpochs > 0 ? (d.epoch / d.totalEpochs) * 100 : 0;
  const mAP50Pct = d.mAP50 * 100;
  const precPct = d.precision * 100;
  const recPct = d.recall * 100;

  // Human-readable confidence label
  const confLabel =
    mAP50Pct >= 25 ? 'Good detection performance'
    : mAP50Pct >= 15 ? 'Model learning — improving'
    : 'Early training phase';

  const donutSlices = [
    { v: precPct, color: '#10b981', label: 'Precision' },
    { v: recPct, color: '#38bdf8', label: 'Recall' },
    { v: Math.max(0, 100 - precPct - recPct * 0.3), color: '#1e293b', label: 'Gap' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>

      {/* Live badge */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: '11px', fontWeight: 600, color: '#94a3b8' }}>Training Monitor</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          {live
            ? <><div style={{ width: 7, height: 7, borderRadius: '50%', background: '#10b981', boxShadow: '0 0 6px #10b981', animation: 'pulse-glow 2s infinite' }} /><span style={{ fontSize: '10px', color: '#10b981', fontWeight: 700 }}>LIVE</span></>
            : <><AlertCircle size={11} color="#f59e0b" /><span style={{ fontSize: '10px', color: '#f59e0b' }}>OFFLINE</span></>
          }
        </div>
      </div>

      {/* Training progress bar */}
      <div style={{ background: 'rgba(15,23,42,0.5)', borderRadius: '8px', padding: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
          <span style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 600 }}>Training Progress</span>
          <span style={{ fontSize: '11px', color: '#38bdf8', fontFamily: "'JetBrains Mono', monospace", fontWeight: 700 }}>
            Epoch {d.epoch} / {d.totalEpochs}
          </span>
        </div>
        <div style={{ height: '8px', background: '#1e293b', borderRadius: '4px', overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: `${progressPct}%`,
            background: 'linear-gradient(90deg, #0ea5e9, #6366f1)',
            borderRadius: '4px',
            transition: 'width 1s ease',
            boxShadow: '0 0 10px rgba(56,189,248,0.4)',
          }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px' }}>
          <span style={{ fontSize: '9.5px', color: '#475569' }}>{progressPct.toFixed(1)}% complete</span>
          <span style={{ fontSize: '9.5px', color: '#475569' }}>{d.trainingMin.toFixed(0)} min elapsed</span>
        </div>
      </div>

      {/* Stat grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
        <StatCard
          icon={<Target size={11} />}
          value={mAP50Pct > 0 ? `${mAP50Pct.toFixed(1)}%` : '—'}
          label="Detection Acc."
          sub="mAP@50 score"
          color="#10b981"
        />
        <StatCard
          icon={<Activity size={11} />}
          value={d.trainLoss > 0 ? d.trainLoss.toFixed(3) : '—'}
          label="Model Loss"
          sub="Lower = better"
          color="#f59e0b"
        />
      </div>

      {/* Precision / Recall donut */}
      <div style={{ background: 'rgba(15,23,42,0.5)', borderRadius: '8px', padding: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <Donut slices={donutSlices} />
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <span style={{ fontSize: '11px', fontWeight: 600, color: '#94a3b8' }}>Detection Quality</span>
            {[
              { label: 'Precision', val: precPct, color: '#10b981' },
              { label: 'Recall', val: recPct, color: '#38bdf8' },
            ].map(({ label, val, color }) => (
              <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <div style={{ width: 8, height: 8, borderRadius: '50%', background: color, boxShadow: `0 0 5px ${color}` }} />
                  <span style={{ color: '#94a3b8' }}>{label}</span>
                </div>
                <span style={{ color, fontWeight: 700 }}>{val > 0 ? `${val.toFixed(1)}%` : '—'}</span>
              </div>
            ))}
            <div style={{ marginTop: '2px', padding: '4px 6px', borderRadius: '4px', background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.15)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <CheckCircle2 size={10} color="#10b981" />
                <span style={{ fontSize: '9.5px', color: '#10b981' }}>{confLabel}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bars */}
      <div style={{ background: 'rgba(15,23,42,0.5)', borderRadius: '8px', padding: '12px', border: '1px solid rgba(255,255,255,0.05)', display: 'flex', flexDirection: 'column', gap: '9px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
          <TrendingUp size={12} color="#38bdf8" />
          <span style={{ fontSize: '11px', fontWeight: 600, color: '#94a3b8' }}>Performance Breakdown</span>
        </div>
        <Bar value={d.mAP5095 * 100} max={100} color="#38bdf8" label="Overall Acc." display={d.mAP5095 > 0 ? `${(d.mAP5095*100).toFixed(1)}%` : '—'} />
        <Bar value={precPct} max={100} color="#10b981" label="Precision" display={precPct > 0 ? `${precPct.toFixed(1)}%` : '—'} />
        <Bar value={recPct} max={100} color="#a78bfa" label="Recall" display={recPct > 0 ? `${recPct.toFixed(1)}%` : '—'} />
        <Bar value={Math.max(0, 3 - d.valLoss) / 3} max={1} color="#f59e0b" label="Val Quality" display={d.valLoss > 0 ? d.valLoss.toFixed(3) : '—'} />
      </div>

      {/* GPU info */}
      <div style={{ background: 'rgba(15,23,42,0.4)', borderRadius: '8px', padding: '10px 12px', border: '1px solid rgba(255,255,255,0.04)', display: 'flex', flexDirection: 'column', gap: '5px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Cpu size={11} color="#38bdf8" />
          <span style={{ fontSize: '10px', fontWeight: 600, color: '#64748b', textTransform: 'uppercase' }}>Hardware &amp; Model</span>
        </div>
        {[
          { label: 'GPU',     value: 'RTX 5060 Laptop · 8GB',                                            color: '#10b981' },
          { label: 'Model',   value: status.modelName !== '…' ? `${status.modelName.toUpperCase()} · VisDrone` : '…', color: '#f1f5f9' },
          { label: 'Classes', value: status.nc > 0 ? `${status.nc} (${status.classNames.slice(0,3).join(', ')}…)` : '…', color: '#94a3b8' },
          { label: 'Train',   value: status.nTrain > 0 ? `${status.nTrain.toLocaleString()} images` : '…',  color: '#94a3b8' },
          { label: 'Val',     value: status.nVal   > 0 ? `${status.nVal.toLocaleString()} images`   : '…',  color: '#94a3b8' },
        ].map(({ label, value, color }) => (
          <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px' }}>
            <span style={{ color: '#64748b' }}>{label}</span>
            <span style={{ color, fontWeight: 600 }}>{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
