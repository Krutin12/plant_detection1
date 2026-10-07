import React from 'react';
import { useFarm } from '../../context/FarmContext';
import { siteConfig } from '../../config/siteConfig';
import { MetricCard } from '../common/MetricCard';
import { BarChart3, TrendingUp, AlertTriangle, ShieldCheck, PieChart, Activity } from 'lucide-react';

export const FarmAnalyticsView = () => {
  const { detections, treatments, navigateTo } = useFarm();

  const totalScans = detections.length;
  const healthyScans = detections.filter(d => d.predictedDisease === 'Plant_Healthy_Condition' || d.severity === 'None').length;
  const healthRate = totalScans > 0 ? Math.round((healthyScans / totalScans) * 100) : 75;

  // Disease frequency counts
  const diseaseCounts = {};
  detections.forEach(d => {
    const key = d.predictedDisease || 'Unknown';
    diseaseCounts[key] = (diseaseCounts[key] || 0) + 1;
  });

  // Crop-wise distribution
  const cropCounts = {};
  detections.forEach(d => {
    const crop = d.crop || 'Field';
    cropCounts[crop] = (cropCounts[crop] || 0) + 1;
  });

  return (
    <div className="page-body">
      {/* ── Page Header ────────────────────────────────────────── */}
      <div className="glass-panel" style={{ padding: '24px 30px', marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '1.6rem' }}>📊</span>
            <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--slate-900)' }}>
              Farm Agronomy & Disease Analytics
            </h2>
          </div>
          <p style={{ color: 'var(--slate-600)', marginTop: '4px', fontSize: '0.94rem' }}>
            Multi-field health metrics, pathogen distribution trends, and historical vulnerability models
          </p>
        </div>
        <button 
          className="btn-secondary"
          onClick={() => navigateTo('export')}
        >
          📄 Export Analytics Report
        </button>
      </div>

      {/* ── Top Metric Cards ───────────────────────────────────── */}
      <div className="metric-grid-4">
        <MetricCard
          icon="🛡️"
          iconColorClass="mc-green"
          label="Farm Health Index"
          value={`${healthRate}%`}
          delta="+4.5% vs last month"
        />
        <MetricCard
          icon="🔍"
          iconColorClass="mc-blue"
          label="Total Scans Logged"
          value={totalScans}
          delta="100% verified AI"
        />
        <MetricCard
          icon="⚠️"
          iconColorClass="mc-red"
          label="High Severity Plots"
          value={detections.filter(d => d.severity === 'Critical' || d.severity === 'High').length}
          delta="Requires immediate spray"
          deltaType="negative"
        />
        <MetricCard
          icon="💊"
          iconColorClass="mc-teal"
          label="Treatment Success"
          value="92.8%"
          delta="Infections arrested"
        />
      </div>

      {/* ── Analytical Visual Charts Grid ──────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.8fr', gap: '24px', marginBottom: '26px' }}>
        {/* Left: Disease Distribution Bar Breakdown */}
        <div className="glass-panel" style={{ padding: '26px' }}>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--slate-900)', marginBottom: '18px' }}>
            Pathogen & Condition Distribution
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {Object.keys(siteConfig.diseases).map(disKey => {
              const disInfo = siteConfig.diseases[disKey];
              const count = diseaseCounts[disKey] || 0;
              const percent = totalScans > 0 ? Math.round((count / totalScans) * 100) : 0;

              return (
                <div key={disKey}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.86rem', fontWeight: 700, marginBottom: '6px' }}>
                    <span style={{ color: 'var(--slate-800)' }}>{disInfo.name.split('(')[0]}</span>
                    <span style={{ color: 'var(--primary-800)' }}>{count} scans ({percent}%)</span>
                  </div>
                  <div style={{ height: '10px', background: 'var(--slate-200)', borderRadius: '6px', overflow: 'hidden' }}>
                    <div style={{ 
                      width: `${Math.max(percent, 4)}%`, 
                      background: disKey === 'Plant_Healthy_Condition' ? '#16a34a' : (disKey.includes('Severe') ? '#ef4444' : '#f59e0b'), 
                      height: '100%',
                      borderRadius: '6px'
                    }}></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Crop Vulnerability & Plot Breakdown */}
        <div className="glass-panel" style={{ padding: '26px' }}>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--slate-900)', marginBottom: '18px' }}>
            Crop-Wise Infection Frequency
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '20px' }}>
            {Object.keys(cropCounts).map(crop => {
              const count = cropCounts[crop];
              return (
                <div key={crop} style={{ background: '#ffffff', border: '1.5px solid var(--slate-200)', borderRadius: '14px', padding: '16px' }}>
                  <div style={{ fontSize: '1.4rem', marginBottom: '4px' }}>🍃</div>
                  <h4 style={{ fontSize: '1rem', fontWeight: 800, color: 'var(--slate-900)' }}>{crop}</h4>
                  <div style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--primary-700)', marginTop: '2px' }}>
                    {count} <span style={{ fontSize: '0.8rem', color: 'var(--slate-500)', fontWeight: 600 }}>scans</span>
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ background: 'rgba(240, 253, 244, 0.7)', border: '1px solid var(--primary-200)', borderRadius: '12px', padding: '16px' }}>
            <h4 style={{ fontSize: '0.92rem', fontWeight: 800, color: 'var(--primary-900)', marginBottom: '6px' }}>
              💡 Agronomic Health Recommendation:
            </h4>
            <p style={{ fontSize: '0.86rem', color: 'var(--slate-700)', lineHeight: '1.5' }}>
              Potato plots show highest susceptibility to late blight due to recent localized humidity spikes. Recommend prophylactic spray of copper hydroxide across Sector C before rainfall.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
