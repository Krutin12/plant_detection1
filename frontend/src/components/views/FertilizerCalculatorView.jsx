import React, { useState } from 'react';
import { siteConfig } from '../../config/siteConfig';
import { useFarm } from '../../context/FarmContext';
import { Sprout, Calculator, AlertCircle, CheckCircle2, ShieldCheck, Download } from 'lucide-react';

export const FertilizerCalculatorView = () => {
  const { profile } = useFarm();

  const [selectedCrop, setSelectedCrop] = useState('Tomato');
  const [landArea, setLandArea] = useState(profile.totalArea || 5.0);
  const [growthStage, setGrowthStage] = useState('vegetative');
  const [soilPh, setSoilPh] = useState(6.5);
  const [hasDiseaseRisk, setHasDiseaseRisk] = useState(false);

  const cropData = siteConfig.cropNutrients[selectedCrop] || siteConfig.cropNutrients['Tomato'];
  const baseNPK = cropData.baseNPK;
  const stageMultipliers = cropData.stages[growthStage] || { N: 1.0, P: 1.0, K: 1.0 };

  // Calculate Elemental Requirements
  let reqN = baseNPK.N * stageMultipliers.N * landArea;
  let reqP = baseNPK.P * stageMultipliers.P * landArea;
  let reqK = baseNPK.K * stageMultipliers.K * landArea;

  // Disease resistance adjustment
  if (hasDiseaseRisk) {
    reqK *= 1.3; // Boost potassium for cell wall strengthening
    reqN *= 0.8; // Restrict soft tissue nitrogen flush
  }

  // Commercial Fertilizer Conversions
  // Urea: 46% N
  const ureaBags50kg = Math.ceil((reqN / 0.46) / 50);
  const ureaKg = Math.round(reqN / 0.46);

  // DAP: 46% P2O5, 18% N
  const dapBags50kg = Math.ceil((reqP / 0.46) / 50);
  const dapKg = Math.round(reqP / 0.46);

  // MOP: 60% K2O
  const mopBags50kg = Math.ceil((reqK / 0.60) / 50);
  const mopKg = Math.round(reqK / 0.60);

  return (
    <div className="page-body">
      {/* ── Page Header ────────────────────────────────────────── */}
      <div className="glass-panel" style={{ padding: '24px 30px', marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '1.6rem' }}>🌾</span>
            <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--slate-900)' }}>
              Precision Fertilizer & Crop Nutrition Advisor
            </h2>
          </div>
          <p style={{ color: 'var(--slate-600)', marginTop: '4px', fontSize: '0.94rem' }}>
            Calculate exact N-P-K nutrient demand, stage-specific fertilizer bags, and soil pH corrections
          </p>
        </div>
      </div>

      {/* ── Calculator Main Grid ───────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.8fr', gap: '26px' }}>
        {/* Left Column: Form Controls */}
        <div className="glass-panel" style={{ padding: '26px' }}>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--slate-900)', marginBottom: '18px' }}>
            Field & Agronomy Parameters
          </h3>

          <div style={{ marginBottom: '16px' }}>
            <label className="form-label">Crop Specimen:</label>
            <select
              className="input-control"
              value={selectedCrop}
              onChange={(e) => {
                setSelectedCrop(e.target.value);
                const nextStages = Object.keys(siteConfig.cropNutrients[e.target.value]?.stages || {});
                if (nextStages.length > 0) setGrowthStage(nextStages[0]);
              }}
            >
              {Object.keys(siteConfig.cropNutrients).map(cropKey => (
                <option key={cropKey} value={cropKey}>
                  {siteConfig.cropNutrients[cropKey].icon} {siteConfig.cropNutrients[cropKey].name}
                </option>
              ))}
            </select>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label className="form-label">Land Size (Hectares):</label>
            <input
              type="number"
              step="0.1"
              min="0.1"
              max="500"
              className="input-control"
              value={landArea}
              onChange={(e) => setLandArea(parseFloat(e.target.value) || 0.1)}
            />
          </div>

          <div style={{ marginBottom: '16px' }}>
            <label className="form-label">Current Growth Phenophase:</label>
            <select
              className="input-control"
              value={growthStage}
              onChange={(e) => setGrowthStage(e.target.value)}
            >
              {Object.entries(cropData.stages).map(([k, v]) => (
                <option key={k} value={k}>{v.label}</option>
              ))}
            </select>
          </div>

          <div style={{ marginBottom: '18px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <label className="form-label">Soil pH Level: <strong>{soilPh}</strong></label>
              <span style={{ fontSize: '0.8rem', color: (soilPh >= cropData.phRange[0] && soilPh <= cropData.phRange[1]) ? 'var(--primary-700)' : 'var(--amber-500)', fontWeight: 700 }}>
                Ideal: {cropData.phRange[0]} - {cropData.phRange[1]}
              </span>
            </div>
            <input
              type="range"
              min="4.5"
              max="8.5"
              step="0.1"
              value={soilPh}
              onChange={(e) => setSoilPh(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--primary-600)' }}
            />
          </div>

          {/* Disease Adjustment Toggle */}
          <div style={{ background: hasDiseaseRisk ? '#fef3c7' : 'rgba(255,255,255,0.7)', border: hasDiseaseRisk ? '1.5px solid #fde68a' : '1px solid var(--slate-200)', borderRadius: '12px', padding: '14px', marginBottom: '20px', cursor: 'pointer' }}
               onClick={() => setHasDiseaseRisk(!hasDiseaseRisk)}
          >
            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', fontWeight: 700, fontSize: '0.88rem', color: hasDiseaseRisk ? '#92400e' : 'var(--slate-800)' }}>
              <input
                type="checkbox"
                checked={hasDiseaseRisk}
                onChange={(e) => setHasDiseaseRisk(e.target.checked)}
                style={{ width: '18px', height: '18px', accentColor: 'var(--amber-500)' }}
              />
              <span>🛡️ Adjust for Active Disease Stress (+30% K, -20% N)</span>
            </label>
          </div>
        </div>

        {/* Right Column: Fertilizer Output Prescription */}
        <div className="glass-panel" style={{ padding: '26px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <div>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--slate-900)' }}>
                Nutrient Prescription for {landArea} ha of {cropData.name}
              </h3>
              <span style={{ fontSize: '0.82rem', color: 'var(--slate-500)', fontWeight: 600 }}>
                Stage: {cropData.stages[growthStage]?.label}
              </span>
            </div>
            <span style={{ background: '#dcfce7', color: '#15803d', border: '1px solid #86efac', padding: '6px 14px', borderRadius: '20px', fontSize: '0.82rem', fontWeight: 800 }}>
              Precision Formulation
            </span>
          </div>

          {/* Elemental NPK Requirements */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px', marginBottom: '24px' }}>
            <div style={{ background: '#eff6ff', border: '1.5px solid #bfdbfe', borderRadius: '14px', padding: '16px', textAlign: 'center' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 800, color: '#1e40af' }}>Nitrogen (N)</span>
              <div style={{ fontSize: '1.7rem', fontWeight: 800, color: '#1e3a8a', margin: '4px 0' }}>
                {Math.round(reqN)} <span style={{ fontSize: '0.9rem' }}>kg</span>
              </div>
              <span style={{ fontSize: '0.74rem', color: '#3b82f6', fontWeight: 600 }}>Leaf & canopy vigor</span>
            </div>

            <div style={{ background: '#fef3c7', border: '1.5px solid #fde68a', borderRadius: '14px', padding: '16px', textAlign: 'center' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 800, color: '#92400e' }}>Phosphorus (P₂O₅)</span>
              <div style={{ fontSize: '1.7rem', fontWeight: 800, color: '#78350f', margin: '4px 0' }}>
                {Math.round(reqP)} <span style={{ fontSize: '0.9rem' }}>kg</span>
              </div>
              <span style={{ fontSize: '0.74rem', color: '#d97706', fontWeight: 600 }}>Root development & flowering</span>
            </div>

            <div style={{ background: '#f0fdf4', border: '1.5px solid #bbf7d0', borderRadius: '14px', padding: '16px', textAlign: 'center' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 800, color: '#166534' }}>Potassium (K₂O)</span>
              <div style={{ fontSize: '1.7rem', fontWeight: 800, color: '#14532d', margin: '4px 0' }}>
                {Math.round(reqK)} <span style={{ fontSize: '0.9rem' }}>kg</span>
              </div>
              <span style={{ fontSize: '0.74rem', color: '#16a34a', fontWeight: 600 }}>Disease defense & fruit firmness</span>
            </div>
          </div>

          {/* Commercial Fertilizer Bags Translation */}
          <h4 style={{ fontSize: '1rem', fontWeight: 800, color: 'var(--slate-800)', marginBottom: '14px' }}>
            Recommended Commercial Fertilizer Purchase (50kg Bags):
          </h4>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px', marginBottom: '22px' }}>
            <div style={{ background: '#ffffff', border: '1.5px solid var(--slate-200)', borderRadius: '14px', padding: '16px' }}>
              <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--slate-500)' }}>Urea (46% N)</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--slate-900)', marginTop: '4px' }}>
                {ureaBags50kg} <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>bags</span>
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--primary-700)', fontWeight: 700, marginTop: '2px' }}>
                {ureaKg} kg total
              </div>
            </div>

            <div style={{ background: '#ffffff', border: '1.5px solid var(--slate-200)', borderRadius: '14px', padding: '16px' }}>
              <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--slate-500)' }}>DAP (18-46-0)</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--slate-900)', marginTop: '4px' }}>
                {dapBags50kg} <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>bags</span>
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--primary-700)', fontWeight: 700, marginTop: '2px' }}>
                {dapKg} kg total
              </div>
            </div>

            <div style={{ background: '#ffffff', border: '1.5px solid var(--slate-200)', borderRadius: '14px', padding: '16px' }}>
              <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--slate-500)' }}>MOP (0-0-60)</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--slate-900)', marginTop: '4px' }}>
                {mopBags50kg} <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>bags</span>
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--primary-700)', fontWeight: 700, marginTop: '2px' }}>
                {mopKg} kg total
              </div>
            </div>
          </div>

          {/* Essential Micronutrients Checklist */}
          <div style={{ background: 'rgba(240, 253, 244, 0.7)', border: '1px solid var(--primary-200)', borderRadius: '14px', padding: '16px 20px' }}>
            <h4 style={{ fontSize: '0.92rem', fontWeight: 800, color: 'var(--primary-900)', marginBottom: '8px' }}>
              Essential Micronutrients for {cropData.name}:
            </h4>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {cropData.micronutrients.map((micro, i) => (
                <span key={i} style={{ background: '#ffffff', color: '#15803d', border: '1px solid #86efac', padding: '4px 12px', borderRadius: '8px', fontSize: '0.82rem', fontWeight: 700 }}>
                  ✓ {micro}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
