import React, { useState, useCallback } from 'react';
import { MeshViewer } from './components/MeshViewer';
import { PipelineStatus } from './components/PipelineStatus';
import { MetricsPanel } from './components/MetricsPanel';
import {
  Cpu, Layers, Eye, ShieldCheck, Compass, Sliders,
  Zap, Radio, Upload, ChevronDown, ChevronUp,
  Satellite, BarChart3, Activity, Info,
} from 'lucide-react';

type ViewMode = 'rgb' | 'heatmap' | 'source';
type TierMode = 'tier1' | 'tier2';
type RightTab = 'pipeline' | 'metrics';

// ─── Tooltip ───────────────────────────────────────────────────
const Tooltip: React.FC<{ text: string; children: React.ReactNode }> = ({
  text, children,
}) => (
  <div className="tooltip-wrap">
    {children}
    <span className="tooltip-box">{text}</span>
  </div>
);

// ─── Section wrapper ───────────────────────────────────────────
const Section: React.FC<{
  title: string; icon?: React.ReactNode;
  collapsible?: boolean; children: React.ReactNode;
}> = ({ title, icon, collapsible = false, children }) => {
  const [open, setOpen] = useState(true);
  return (
    <div className="section-block">
      <div
        className="section-header"
        onClick={collapsible ? () => setOpen(o => !o) : undefined}
        style={{ cursor: collapsible ? 'pointer' : 'default' }}
      >
        <span className="section-title">
          {icon && <span style={{ opacity: 0.8 }}>{icon}</span>}
          {title}
        </span>
        {collapsible && (
          open ? <ChevronUp size={13} style={{ color: '#475569' }} />
               : <ChevronDown size={13} style={{ color: '#475569' }} />
        )}
      </div>
      {open && <div className="section-body">{children}</div>}
    </div>
  );
};

// ─── Main App ──────────────────────────────────────────────────
export const App: React.FC = () => {
  const [viewMode, setViewMode] = useState<ViewMode>('heatmap');
  const [tierMode, setTierMode] = useState<TierMode>('tier2');
  const [clipHeight, setClipHeight] = useState(12);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [pointSize, setPointSize] = useState(1.0);
  const [rightTab, setRightTab] = useState<RightTab>('pipeline');
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file) setUploadedFile(file.name);
  }, []);

  return (
    <div className="dashboard-root">

      {/* ─── LEFT SIDEBAR ─────────────────────────────────── */}
      <aside className="sidebar sidebar-left">

        {/* Logo */}
        <div className="logo-row">
          <div className="logo-icon">
            <Cpu size={18} style={{ color: '#38bdf8' }} />
          </div>
          <div>
            <div className="logo-title">AERO MESH</div>
            <div className="logo-sub">Single-Pass UAV 3D Reconstruction</div>
          </div>
          <span className="badge badge-cyan">v2.0</span>
        </div>

        {/* Upload Zone */}
        <Section title="Input" icon={<Upload size={12} />} collapsible>
          <div
            className={`upload-zone ${dragActive ? 'upload-zone--active' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
          >
            {uploadedFile ? (
              <div className="upload-zone__file">
                <Zap size={14} style={{ color: '#10b981' }} />
                <span>{uploadedFile}</span>
              </div>
            ) : (
              <>
                <Upload size={20} style={{ color: '#38bdf8', marginBottom: '6px' }} />
                <span style={{ fontSize: '12px', color: '#64748b' }}>
                  Drop drone video / PLY here
                </span>
                <span style={{ fontSize: '10px', color: '#334155', marginTop: '2px' }}>
                  .mp4 · .ply · .las · .json
                </span>
              </>
            )}
          </div>
        </Section>

        {/* Tier Mode */}
        <Section title="Delivery Tier" icon={<Zap size={12} style={{ color: '#f59e0b' }} />}>
          <div className="btn-grid">
            <button
              id="btn-tier1"
              className={`btn ${tierMode === 'tier1' ? 'btn-active' : ''}`}
              onClick={() => setTierMode('tier1')}
            >
              <Zap size={12} /> Tier 1
            </button>
            <button
              id="btn-tier2"
              className={`btn ${tierMode === 'tier2' ? 'btn-active' : ''}`}
              onClick={() => setTierMode('tier2')}
            >
              <Layers size={12} /> Tier 2
            </button>
          </div>
          <p className="hint-text">
            {tierMode === 'tier1'
              ? '⚡ Sparse SfM cloud — real-time situational awareness'
              : '💎 Dense TSDF + 3DGS neural completion'}
          </p>
        </Section>

        {/* View Mode */}
        <Section title="Shading Mode" icon={<Eye size={12} style={{ color: '#38bdf8' }} />}>
          {(
            [
              { id: 'heatmap', icon: <ShieldCheck size={12} />, label: 'Confidence Heatmap' },
              { id: 'rgb', icon: <Layers size={12} />, label: 'Realistic RGB' },
              { id: 'source', icon: <Compass size={12} />, label: 'Source Origin Tags' },
            ] as { id: ViewMode; icon: React.ReactNode; label: string }[]
          ).map(m => (
            <button
              key={m.id}
              id={`btn-view-${m.id}`}
              className={`btn btn-full ${viewMode === m.id ? 'btn-active' : ''}`}
              onClick={() => setViewMode(m.id)}
              style={{ marginBottom: '6px' }}
            >
              {m.icon} {m.label}
            </button>
          ))}
        </Section>

        {/* Height Slice */}
        <Section title="Slicing" icon={<Sliders size={12} />} collapsible>
          <div className="slider-row">
            <span className="slider-label">Height Cutoff</span>
            <span className="slider-val">{clipHeight}m</span>
          </div>
          <input
            id="clip-height-slider"
            type="range" min={-3} max={14} step={0.5}
            value={clipHeight}
            onChange={e => setClipHeight(parseFloat(e.target.value))}
            className="slider"
          />

          <div className="slider-row" style={{ marginTop: '10px' }}>
            <span className="slider-label">Point Size</span>
            <span className="slider-val">{pointSize.toFixed(1)}×</span>
          </div>
          <input
            id="point-size-slider"
            type="range" min={0.3} max={3.0} step={0.1}
            value={pointSize}
            onChange={e => setPointSize(parseFloat(e.target.value))}
            className="slider"
          />
        </Section>

        {/* Toggles */}
        <Section title="Overlays" icon={<Radio size={12} style={{ color: '#38bdf8' }} />}>
          <div className="toggle-row">
            <span className="toggle-label">
              <Satellite size={12} /> UAV Trajectory Curve
            </span>
            <label className="switch">
              <input
                id="toggle-trajectory"
                type="checkbox"
                checked={showTrajectory}
                onChange={e => setShowTrajectory(e.target.checked)}
              />
              <span className="switch-thumb" />
            </label>
          </div>
        </Section>

      </aside>

      {/* ─── CENTER VIEWPORT ──────────────────────────────── */}
      <main className="viewport">
        <MeshViewer
          viewMode={viewMode}
          tierMode={tierMode}
          clipHeight={clipHeight}
          pointSize={pointSize}
          showTrajectory={showTrajectory}
        />

        {/* Top overlay badges */}
        <div className="overlay overlay-top-left">
          <span className="badge badge-cyan">6-DoF ESKF Georeferenced</span>
          <span className="badge badge-amber">UTM Zone 43N · WGS84</span>
          <span className="badge" style={{
            background: 'rgba(99,102,241,0.15)',
            color: '#a78bfa',
            border: '1px solid rgba(99,102,241,0.3)',
          }}>SIH PS #26158</span>
        </div>

        {/* Tier indicator */}
        <div className="overlay overlay-top-right">
          <div className="tier-indicator">
            <span className="tier-label">{tierMode === 'tier1' ? 'TIER 1' : 'TIER 2'}</span>
            <span className="tier-sub">
              {tierMode === 'tier1' ? 'Sparse SfM' : 'Dense TSDF+3DGS'}
            </span>
          </div>
        </div>

        {/* Bottom legend (heatmap only) */}
        {viewMode === 'heatmap' && (
          <div className="legend-bar">
            {[
              { color: '#38bdf8', label: 'MVS Observed', sub: '≥90%' },
              { color: '#f59e0b', label: 'Mono Infill', sub: '60–79%' },
              { color: '#f43f5e', label: '3DGS Hallucinated', sub: '≤30%' },
            ].map(item => (
              <div key={item.label} className="legend-item">
                <div className="legend-dot" style={{
                  background: item.color,
                  boxShadow: `0 0 8px ${item.color}`,
                }} />
                <div>
                  <div style={{ fontSize: '11px', color: '#f1f5f9', fontWeight: 600 }}>
                    {item.label}
                  </div>
                  <div style={{ fontSize: '10px', color: '#64748b' }}>{item.sub}</div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Source legend */}
        {viewMode === 'source' && (
          <div className="legend-bar">
            {[
              { color: '#0ea5e9', label: 'Observed MVS' },
              { color: '#f59e0b', label: 'Monocular Depth' },
              { color: '#ef4444', label: '3DGS Inferred' },
            ].map(item => (
              <div key={item.label} className="legend-item">
                <div className="legend-dot" style={{ background: item.color }} />
                <span style={{ fontSize: '11px', color: '#f1f5f9' }}>{item.label}</span>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* ─── RIGHT SIDEBAR ────────────────────────────────── */}
      <aside className="sidebar sidebar-right">

        {/* Tab switcher */}
        <div className="tab-row">
          <button
            id="tab-pipeline"
            className={`tab-btn ${rightTab === 'pipeline' ? 'tab-btn--active' : ''}`}
            onClick={() => setRightTab('pipeline')}
          >
            <Activity size={13} /> Pipeline
          </button>
          <button
            id="tab-metrics"
            className={`tab-btn ${rightTab === 'metrics' ? 'tab-btn--active' : ''}`}
            onClick={() => setRightTab('metrics')}
          >
            <BarChart3 size={13} /> Metrics
          </button>
        </div>

        {/* Tab content */}
        {rightTab === 'pipeline' ? (
          <PipelineStatus autoRun loopInterval={20000} />
        ) : (
          <MetricsPanel tierMode={tierMode} viewMode={viewMode} />
        )}

        {/* Footer info */}
        <div className="sidebar-footer">
          <Info size={11} style={{ color: '#475569', flexShrink: 0 }} />
          <span>
            Best-in-class GPU pipeline: SuperPoint+LightGlue · DA2 Metric · RAFT-L · Open3D TSDF · 3DGS
          </span>
        </div>

      </aside>
    </div>
  );
};

export default App;
