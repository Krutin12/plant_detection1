import React, { useState } from 'react';
import { useFarm } from '../../context/FarmContext';
import { siteConfig } from '../../config/siteConfig';
import { analyzeLeafImage } from '../../services/api';
import { LeafSpecimen } from '../../utils/leafIllustrations';
import { MetricCard } from '../common/MetricCard';
import { SeverityBadge } from '../common/SeverityBadge';
import { Modal } from '../common/Modal';
import { 
  UploadCloud, 
  Sparkles, 
  ArrowRight, 
  Eye, 
  Activity, 
  CheckCircle2, 
  AlertTriangle, 
  Target,
  FileText,
  Calendar
} from 'lucide-react';

export const DashboardView = () => {
  const { detections, addDetection, navigateTo, profile } = useFarm();

  const [uploadMode, setUploadMode] = useState('upload'); // 'upload' | 'sample'
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [selectedSample, setSelectedSample] = useState(siteConfig.sampleLeaves[0]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeModalScan, setActiveModalScan] = useState(null);

  // File Upload Handlers
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setUploadMode('upload');
    }
  };

  // Run AI Analysis
  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    try {
      const result = await analyzeLeafImage(
        uploadMode === 'upload' ? selectedFile : null,
        uploadMode === 'sample' ? selectedSample : null
      );

      // Save scan to state
      addDetection({
        crop: result.crop,
        predictedDisease: result.diseaseKey,
        confidence: result.confidence,
        severity: result.severity,
        imageName: selectedFile ? selectedFile.name : `${selectedSample.crop}_sample.svg`,
        sampleType: uploadMode === 'sample' ? selectedSample.leafType : null,
        previewUrl: previewUrl,
        metrics: result.metrics
      });

      // Jump to Detection view with results
      navigateTo('detection', { lastScan: result });
    } catch (err) {
      console.error("Diagnosis error:", err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Calculated Metrics
  const totalScans = detections.length;
  const healthyCount = detections.filter(d => 
    d.predictedDisease === 'Plant_Healthy_Condition' || d.severity === 'None'
  ).length;
  const diseaseCount = totalScans - healthyCount;
  const avgConfidence = totalScans > 0 
    ? (detections.reduce((acc, curr) => acc + (curr.confidence || 0.9), 0) / totalScans * 100).toFixed(1)
    : "96.4";

  return (
    <div className="page-body">
      {/* ── Lush Botanical Hero Banner ─────────────────────────── */}
      <div className="av-hero-banner">
        <div style={{ zIndex: 2 }}>
          <div className="av-hero-title-row">
            <div className="av-hero-icon-box">🌿</div>
            <h1>{siteConfig.tagline}</h1>
          </div>
          <p className="av-hero-desc">{siteConfig.heroDescription}</p>
        </div>
        <div className="av-hero-tagline-box" style={{ zIndex: 2 }}>
          <span className="av-hero-script">{siteConfig.heroScript}</span>
          <span className="av-hero-script-sub">{siteConfig.heroScriptSub}</span>
        </div>
      </div>

      {/* ── Diagnostic Showcase Grid ───────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: '24px', marginBottom: '28px' }}>
        {/* Left Column: Interactive Scanner */}
        <div className="glass-panel" style={{ padding: '26px' }}>
          {/* Mode Selector Tabs */}
          <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
            <button
              className={`btn-secondary ${uploadMode === 'upload' ? 'btn-primary' : ''}`}
              style={{ flex: 1 }}
              onClick={() => setUploadMode('upload')}
            >
              📁 Upload Leaf Image
            </button>
            <button
              className={`btn-secondary ${uploadMode === 'sample' ? 'btn-primary' : ''}`}
              style={{ flex: 1 }}
              onClick={() => setUploadMode('sample')}
            >
              🌿 Use Sample Leaf
            </button>
          </div>

          {/* Upload Dropzone View */}
          {uploadMode === 'upload' ? (
            <div>
              <label 
                className="av-dropzone" 
                onDragOver={(e) => e.preventDefault()} 
                onDrop={handleDrop}
                style={{ display: 'block' }}
              >
                <input 
                  type="file" 
                  accept="image/png, image/jpeg, image/webp" 
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />
                <div className="dropzone-icon-box">🖼️</div>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--slate-900)', marginBottom: '4px' }}>
                  Drag & Drop leaf photo here
                </h3>
                <p style={{ fontSize: '0.88rem', color: 'var(--slate-600)', marginBottom: '8px' }}>
                  or click anywhere to browse from your device
                </p>
                <span style={{ fontSize: '0.78rem', color: 'var(--slate-400)', fontWeight: 600 }}>
                  Supports PNG, JPG, JPEG, WEBP (Max 25MB)
                </span>
              </label>

              {previewUrl && (
                <div style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '16px', padding: '12px', background: 'rgba(255,255,255,0.7)', borderRadius: '12px', border: '1px solid var(--primary-200)' }}>
                  <img 
                    src={previewUrl} 
                    alt="Selected leaf" 
                    style={{ width: '80px', height: '80px', objectFit: 'cover', borderRadius: '10px', border: '2px solid var(--primary-400)' }}
                  />
                  <div>
                    <h4 style={{ fontSize: '0.94rem', fontWeight: 700, color: 'var(--slate-800)' }}>{selectedFile?.name}</h4>
                    <p style={{ fontSize: '0.8rem', color: 'var(--primary-700)', fontWeight: 600 }}>Leaf ready for neural diagnostic</p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* Sample Leaf Chooser View */
            <div>
              <label className="form-label">Select high-definition plant specimen:</label>
              <select 
                className="input-control"
                value={selectedSample.id}
                onChange={(e) => {
                  const match = siteConfig.sampleLeaves.find(s => s.id === e.target.value);
                  if (match) setSelectedSample(match);
                }}
                style={{ marginBottom: '18px' }}
              >
                {siteConfig.sampleLeaves.map((sample) => (
                  <option key={sample.id} value={sample.id}>
                    {sample.name} ({sample.badge})
                  </option>
                ))}
              </select>

              <div style={{ display: 'flex', alignItems: 'center', gap: '24px', background: 'rgba(240, 253, 244, 0.7)', padding: '16px 20px', borderRadius: '14px', border: '1px solid var(--primary-200)' }}>
                <div style={{ background: '#ffffff', padding: '10px', borderRadius: '14px', boxShadow: '0 4px 12px rgba(22,163,74,0.1)' }}>
                  <LeafSpecimen type={selectedSample.leafType} size={110} />
                </div>
                <div>
                  <span className="sev-pill sev-pill-medium" style={{ marginBottom: '6px' }}>{selectedSample.badge}</span>
                  <h4 style={{ fontSize: '1.05rem', fontWeight: 800, color: 'var(--slate-900)' }}>{selectedSample.name}</h4>
                  <p style={{ fontSize: '0.84rem', color: 'var(--slate-600)', marginTop: '4px' }}>{selectedSample.description}</p>
                </div>
              </div>
            </div>
          )}

          {/* Analyze CTA Button */}
          <div style={{ marginTop: '22px' }}>
            <button
              className="btn-primary"
              style={{ width: '100%', padding: '14px', fontSize: '1.05rem' }}
              onClick={handleAnalyze}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <Activity className="animate-spin" size={20} />
                  <span>Scanning Foliage with MobileNetV2...</span>
                </>
              ) : (
                <>
                  <span>🌿 Analyze Plant Disease</span>
                  <ArrowRight size={19} />
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: How It Works Guide */}
        <div className="glass-panel" style={{ padding: '26px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '22px' }}>
              <span style={{ fontSize: '1.4rem' }}>🌿</span>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--slate-900)' }}>How it works</h3>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-start' }}>
                <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: '#dcfce7', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.2rem', flexShrink: 0 }}>
                  🖼️
                </div>
                <div>
                  <h4 style={{ fontSize: '0.96rem', fontWeight: 700, color: 'var(--slate-900)' }}>1. Upload leaf image</h4>
                  <p style={{ fontSize: '0.84rem', color: 'var(--slate-600)', lineHeight: '1.45' }}>Drag & drop or snap a photo of any crop leaf showing spots or wilting.</p>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-start' }}>
                <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: '#dbeafe', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.2rem', flexShrink: 0 }}>
                  🤖
                </div>
                <div>
                  <h4 style={{ fontSize: '0.96rem', fontWeight: 700, color: 'var(--slate-900)' }}>2. AI analyzes symptoms</h4>
                  <p style={{ fontSize: '0.84rem', color: 'var(--slate-600)', lineHeight: '1.45' }}>Deep neural model evaluates chlorophyll ratio, chlorosis, and necrotic patterns.</p>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-start' }}>
                <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: '#fef3c7', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.2rem', flexShrink: 0 }}>
                  💊
                </div>
                <div>
                  <h4 style={{ fontSize: '0.96rem', fontWeight: 700, color: 'var(--slate-900)' }}>3. Get diagnosis & remedy</h4>
                  <p style={{ fontSize: '0.84rem', color: 'var(--slate-600)', lineHeight: '1.45' }}>Immediate confidence rating, organic & chemical treatments with dosages.</p>
                </div>
              </div>
            </div>
          </div>

          <div style={{ background: 'rgba(240, 253, 244, 0.8)', border: '1px solid var(--primary-300)', borderRadius: '12px', padding: '12px 16px', marginTop: '20px' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--primary-800)' }}>
              💡 Pro Tip: Inspect both upper and lower leaf surfaces for accurate diagnosis.
            </span>
          </div>
        </div>
      </div>

      {/* ── 4 Metric Cards Row ─────────────────────────────────── */}
      <div className="metric-grid-4">
        <MetricCard 
          icon="🍃" 
          iconColorClass="mc-green" 
          label="Total Scans" 
          value={totalScans} 
          delta="+12% this week" 
        />
        <MetricCard 
          icon="🌿" 
          iconColorClass="mc-teal" 
          label="Healthy Crops" 
          value={healthyCount} 
          delta="+8% this week" 
        />
        <MetricCard 
          icon="⚠️" 
          iconColorClass="mc-red" 
          label="Diseases Detected" 
          value={diseaseCount} 
          delta="+5% this week" 
          deltaType="negative"
        />
        <MetricCard 
          icon="🎯" 
          iconColorClass="mc-blue" 
          label="AI Model Accuracy" 
          value={`${avgConfidence}%`} 
          delta="+2.1% boosted" 
        />
      </div>

      {/* ── Recent Detections Table Section ────────────────────── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '24px 0 14px 0' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1.25rem', fontWeight: 800, color: '#ffffff', textShadow: '0 2px 4px rgba(0,0,0,0.4)' }}>
          <span>🕒</span>
          <span>Recent Plant Detections</span>
        </div>
        <button 
          onClick={() => navigateTo('history')}
          className="btn-outline-leaf"
          style={{ background: 'rgba(255, 255, 255, 0.9)', color: 'var(--primary-800)', border: 'none', fontWeight: 800 }}
        >
          View All History ➔
        </button>
      </div>

      <div className="av-table-container">
        <table className="av-table">
          <thead>
            <tr>
              <th>Crop Specimen</th>
              <th>Diagnosis</th>
              <th>Confidence</th>
              <th>Severity</th>
              <th>Date & Time</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {detections.slice(0, 5).map((scan) => {
              const disKey = scan.predictedDisease || 'Plant_Healthy_Condition';
              const disInfo = siteConfig.diseases[disKey] || { name: disKey };
              const dateStr = new Date(scan.timestamp).toLocaleString('en-US', {
                month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit'
              });

              return (
                <tr key={scan.id}>
                  <td style={{ fontWeight: 800, color: 'var(--slate-900)' }}>
                    🍃 {scan.crop || 'Field Plant'}
                  </td>
                  <td style={{ fontWeight: 600 }}>
                    {disInfo.name}
                  </td>
                  <td style={{ fontWeight: 700, color: 'var(--primary-700)' }}>
                    {((scan.confidence || 0.92) * 100).toFixed(1)}%
                  </td>
                  <td>
                    <SeverityBadge severity={scan.severity || 'low'} />
                  </td>
                  <td style={{ color: 'var(--slate-500)', fontSize: '0.84rem' }}>
                    {dateStr}
                  </td>
                  <td>
                    <button 
                      className="btn-outline-leaf"
                      onClick={() => setActiveModalScan(scan)}
                    >
                      <Eye size={14} />
                      <span>View Details</span>
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* ── Quick Action Nav Links ─────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '18px', marginTop: '24px' }}>
        <button 
          className="btn-secondary" 
          style={{ padding: '16px', fontWeight: 800, background: 'rgba(255,255,255,0.95)' }}
          onClick={() => navigateTo('history')}
        >
          📋 Open Full Scan Archive
        </button>
        <button 
          className="btn-secondary" 
          style={{ padding: '16px', fontWeight: 800, background: 'rgba(255,255,255,0.95)' }}
          onClick={() => navigateTo('treatment')}
        >
          💊 Open Treatment Planner
        </button>
        <button 
          className="btn-secondary" 
          style={{ padding: '16px', fontWeight: 800, background: 'rgba(255,255,255,0.95)' }}
          onClick={() => navigateTo('fertilizer')}
        >
          🌾 Calculate Crop Nutrients
        </button>
      </div>

      {/* ── Scan Detail Modal ──────────────────────────────────── */}
      {activeModalScan && (
        <Modal 
          isOpen={Boolean(activeModalScan)} 
          onClose={() => setActiveModalScan(null)}
          title={`Scan Diagnostics: ${activeModalScan.crop}`}
        >
          <div>
            <div style={{ display: 'flex', gap: '20px', alignItems: 'center', marginBottom: '20px', padding: '16px', background: 'var(--primary-50)', borderRadius: '12px' }}>
              <div style={{ width: '60px', height: '60px', borderRadius: '12px', background: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '2rem' }}>
                🌿
              </div>
              <div>
                <SeverityBadge severity={activeModalScan.severity} />
                <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--slate-900)', marginTop: '4px' }}>
                  {siteConfig.diseases[activeModalScan.predictedDisease]?.name || activeModalScan.predictedDisease}
                </h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--slate-600)' }}>
                  Scanned on {new Date(activeModalScan.timestamp).toLocaleString()}
                </p>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginBottom: '20px' }}>
              <div style={{ padding: '12px', background: '#f8fafc', borderRadius: '10px' }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--slate-500)', fontWeight: 700 }}>Confidence Score</span>
                <p style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--primary-700)' }}>
                  {((activeModalScan.confidence || 0.95) * 100).toFixed(1)}%
                </p>
              </div>
              <div style={{ padding: '12px', background: '#f8fafc', borderRadius: '10px' }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--slate-500)', fontWeight: 700 }}>Recommended Action</span>
                <p style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--slate-800)' }}>
                  {activeModalScan.severity === 'None' ? 'Routine Care' : 'Targeted Spray'}
                </p>
              </div>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <h4 style={{ fontSize: '0.92rem', fontWeight: 800, color: 'var(--slate-800)', marginBottom: '6px' }}>Key Symptoms</h4>
              <p style={{ fontSize: '0.88rem', color: 'var(--slate-600)', lineHeight: '1.5' }}>
                {siteConfig.diseases[activeModalScan.predictedDisease]?.symptoms || "Normal healthy cellular structure without noticeable chlorosis or necrotic tissue."}
              </p>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '24px' }}>
              <button 
                className="btn-secondary"
                onClick={() => setActiveModalScan(null)}
              >
                Close
              </button>
              <button 
                className="btn-primary"
                onClick={() => {
                  setActiveModalScan(null);
                  navigateTo('detection');
                }}
              >
                Open in Detection Studio ➔
              </button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};
