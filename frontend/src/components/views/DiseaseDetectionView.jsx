import React, { useState, useEffect } from 'react';
import { useFarm } from '../../context/FarmContext';
import { siteConfig } from '../../config/siteConfig';
import { analyzeLeafImage } from '../../services/api';
import { LeafSpecimen } from '../../utils/leafIllustrations';
import { SeverityBadge } from '../common/SeverityBadge';
import { Modal } from '../common/Modal';
import { 
  ScanLine, 
  Upload, 
  Sparkles, 
  ShieldAlert, 
  CheckCircle, 
  Pill, 
  Leaf, 
  FileDown, 
  Activity, 
  Info,
  Calendar,
  AlertCircle
} from 'lucide-react';

export const DiseaseDetectionView = () => {
  const { lastScanResult, addDetection, addTreatment, navigateTo } = useFarm();

  const [activeTab, setActiveTab] = useState('organic'); // 'organic' | 'chemical' | 'prevention'
  const [selectedCrop, setSelectedCrop] = useState('Tomato');
  const [uploadMode, setUploadMode] = useState('sample'); // 'upload' | 'sample'
  const [selectedSample, setSelectedSample] = useState(siteConfig.sampleLeaves[0]);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewSrc, setPreviewSrc] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [currentDiagnosis, setCurrentDiagnosis] = useState(null);
  const [isTreatmentModalOpen, setIsTreatmentModalOpen] = useState(false);

  // Prefill from last scan or default
  useEffect(() => {
    if (lastScanResult) {
      setCurrentDiagnosis(lastScanResult);
      setSelectedCrop(lastScanResult.crop || 'Tomato');
    } else {
      // Default initial scan to tomato early blight
      const initialKey = "Early_Disease_Symptoms";
      const info = siteConfig.diseases[initialKey];
      setCurrentDiagnosis({
        diseaseKey: initialKey,
        diseaseName: info.name,
        crop: "Tomato",
        confidence: 0.942,
        severity: info.severity,
        urgency: info.urgency,
        symptoms: info.symptoms,
        causes: info.causes,
        prevention: info.prevention,
        organicTreatment: info.organicTreatment,
        chemicalTreatment: info.chemicalTreatment,
        metrics: { chlorophyllIndex: 68, necrosisIndex: 22, chlorosisIndex: 18, qualityScore: 95 }
      });
    }
  }, [lastScanResult]);

  // Handle User File Upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedFile(file);
      setPreviewSrc(URL.createObjectURL(file));
      setUploadMode('upload');
    }
  };

  // Perform Leaf Scan
  const handleRunScanner = async () => {
    setIsScanning(true);
    try {
      const res = await analyzeLeafImage(
        uploadMode === 'upload' ? uploadedFile : null,
        uploadMode === 'sample' ? selectedSample : null
      );

      // Add selected crop override if given
      if (selectedCrop) res.crop = selectedCrop;

      setCurrentDiagnosis(res);
      addDetection({
        crop: res.crop,
        predictedDisease: res.diseaseKey,
        confidence: res.confidence,
        severity: res.severity,
        imageName: uploadedFile ? uploadedFile.name : `${selectedSample.name}.svg`,
        metrics: res.metrics
      });
    } catch (err) {
      console.error(err);
    } finally {
      setIsScanning(false);
    }
  };

  // Add Treatment Log Form State
  const [trtProductName, setTrtProductName] = useState('');
  const [trtDosage, setTrtDosage] = useState('2.5 ml / Liter water');
  const [trtNotes, setTrtNotes] = useState('');

  const handleCreateTreatment = (e) => {
    e.preventDefault();
    if (!currentDiagnosis) return;

    addTreatment({
      crop: currentDiagnosis.crop,
      disease: currentDiagnosis.diseaseKey,
      treatmentType: activeTab === 'chemical' ? 'chemical' : 'organic',
      productName: trtProductName || (activeTab === 'chemical' ? 'Mancozeb 75% WP' : 'Cold-Pressed Neem Oil'),
      dosage: trtDosage,
      areaTreated: "2.5 ha",
      cost: 55.0,
      plannedDate: new Date(Date.now() + 7 * 86400000).toISOString(),
      notes: trtNotes || `Treatment initiated for ${currentDiagnosis.diseaseName}`
    });

    setIsTreatmentModalOpen(false);
    navigateTo('treatment');
  };

  if (!currentDiagnosis) {
    return <div className="page-body"><div className="glass-panel" style={{ padding: '30px', color: 'white' }}>Loading Pathology Engine...</div></div>;
  }

  const diseaseDetail = siteConfig.diseases[currentDiagnosis.diseaseKey] || currentDiagnosis;

  return (
    <div className="page-body">
      {/* ── Diagnostic Studio Header ───────────────────────────── */}
      <div className="glass-panel" style={{ padding: '24px 30px', marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '1.6rem' }}>🔬</span>
            <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--slate-900)' }}>
              AI Plant Pathology Diagnostic Studio
            </h2>
          </div>
          <p style={{ color: 'var(--slate-600)', marginTop: '4px', fontSize: '0.94rem' }}>
            MobileNetV2 high-resolution cellular feature scanner with chlorophyll spectroscopy
          </p>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button 
            className="btn-secondary"
            onClick={() => navigateTo('history')}
          >
            📋 Scan History
          </button>
        </div>
      </div>

      {/* ── Main Studio Grid ───────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.8fr', gap: '26px' }}>
        {/* Left Column: Specimen Input & Scanner Controls */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--slate-900)', marginBottom: '16px' }}>
            1. Select Foliage Specimen
          </h3>

          {/* Mode Switcher */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
            <button
              className={`btn-secondary ${uploadMode === 'sample' ? 'btn-primary' : ''}`}
              style={{ flex: 1, padding: '10px' }}
              onClick={() => setUploadMode('sample')}
            >
              🌿 Reference Samples
            </button>
            <button
              className={`btn-secondary ${uploadMode === 'upload' ? 'btn-primary' : ''}`}
              style={{ flex: 1, padding: '10px' }}
              onClick={() => setUploadMode('upload')}
            >
              📁 Device File
            </button>
          </div>

          {/* Crop Selector */}
          <div style={{ marginBottom: '16px' }}>
            <label className="form-label">Associated Crop:</label>
            <select 
              className="input-control"
              value={selectedCrop}
              onChange={(e) => setSelectedCrop(e.target.value)}
            >
              {siteConfig.defaultProfile.primaryCrops.map(crop => (
                <option key={crop} value={crop}>{crop}</option>
              ))}
            </select>
          </div>

          {/* Specimen Preview Box */}
          <div className="scanner-active-box" style={{ 
            background: 'linear-gradient(135deg, rgba(240, 253, 244, 0.9), rgba(220, 252, 231, 0.8))',
            border: '2px solid var(--primary-300)',
            borderRadius: '16px',
            padding: '24px',
            textAlign: 'center',
            marginBottom: '18px',
            minHeight: '230px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            {isScanning && <div className="scanner-laser-line"></div>}

            {uploadMode === 'sample' ? (
              <div>
                <LeafSpecimen type={selectedSample.leafType} size={150} />
                <div style={{ marginTop: '10px', fontWeight: 800, color: 'var(--primary-900)' }}>
                  {selectedSample.name}
                </div>
              </div>
            ) : previewSrc ? (
              <img 
                src={previewSrc} 
                alt="Uploaded specimen" 
                style={{ maxHeight: '180px', maxWidth: '100%', objectFit: 'contain', borderRadius: '12px' }}
              />
            ) : (
              <label style={{ cursor: 'pointer' }}>
                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleFileUpload} 
                  style={{ display: 'none' }}
                />
                <div className="dropzone-icon-box" style={{ margin: '0 auto 10px auto' }}>📁</div>
                <div style={{ fontWeight: 700, color: 'var(--slate-800)' }}>Click to Upload Leaf Photo</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--slate-500)' }}>PNG, JPG or WEBP</div>
              </label>
            )}
          </div>

          {/* Sample Dropdown when in sample mode */}
          {uploadMode === 'sample' && (
            <div style={{ marginBottom: '18px' }}>
              <label className="form-label">Specimen Library:</label>
              <select
                className="input-control"
                value={selectedSample.id}
                onChange={(e) => {
                  const s = siteConfig.sampleLeaves.find(item => item.id === e.target.value);
                  if (s) {
                    setSelectedSample(s);
                    setSelectedCrop(s.crop);
                  }
                }}
              >
                {siteConfig.sampleLeaves.map(sample => (
                  <option key={sample.id} value={sample.id}>
                    {sample.name} — {sample.badge}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Run Diagnostic Button */}
          <button
            className="btn-primary"
            style={{ width: '100%', padding: '14px', fontSize: '1.05rem' }}
            onClick={handleRunScanner}
            disabled={isScanning}
          >
            {isScanning ? (
              <>
                <Activity className="animate-spin" size={20} />
                <span>Processing Neural Weights...</span>
              </>
            ) : (
              <>
                <ScanLine size={20} />
                <span>Run AI Deep Diagnostic</span>
              </>
            )}
          </button>
        </div>

        {/* Right Column: In-Depth Diagnostic Results Card */}
        <div className="glass-panel" style={{ padding: '28px' }}>
          {/* Header Result Bar */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', borderBottom: '1.5px solid var(--slate-200)', paddingBottom: '18px', marginBottom: '20px' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                <span style={{ fontSize: '0.9rem', fontWeight: 800, color: 'var(--primary-700)', textTransform: 'uppercase' }}>
                  {currentDiagnosis.crop} Diagnosis
                </span>
                <span style={{ color: 'var(--slate-400)' }}>•</span>
                <SeverityBadge severity={currentDiagnosis.severity} />
              </div>
              <h2 style={{ fontSize: '1.7rem', fontWeight: 800, color: 'var(--slate-900)', lineHeight: '1.2' }}>
                {diseaseDetail.name}
              </h2>
            </div>

            {/* Confidence Score Pill */}
            <div style={{ textAlign: 'right', background: 'var(--primary-50)', border: '1.5px solid var(--primary-300)', padding: '10px 18px', borderRadius: '16px' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--primary-800)', textTransform: 'uppercase' }}>
                AI Confidence
              </span>
              <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--primary-700)', lineHeight: '1.1' }}>
                {((currentDiagnosis.confidence || 0.95) * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Cellular Feature Ratio Meters */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px', marginBottom: '22px' }}>
            <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', padding: '12px 14px', borderRadius: '12px' }}>
              <span style={{ fontSize: '0.78rem', color: '#166534', fontWeight: 700 }}>🍃 Chlorophyll Index</span>
              <div style={{ fontSize: '1.3rem', fontWeight: 800, color: '#14532d', marginTop: '2px' }}>
                {currentDiagnosis.metrics?.chlorophyllIndex || 74}%
              </div>
              <div style={{ height: '6px', background: '#dcfce7', borderRadius: '4px', overflow: 'hidden', marginTop: '6px' }}>
                <div style={{ width: `${currentDiagnosis.metrics?.chlorophyllIndex || 74}%`, background: '#22c55e', height: '100%' }}></div>
              </div>
            </div>

            <div style={{ background: '#fffbeb', border: '1px solid #fde68a', padding: '12px 14px', borderRadius: '12px' }}>
              <span style={{ fontSize: '0.78rem', color: '#92400e', fontWeight: 700 }}>🟡 Chlorosis Yellowing</span>
              <div style={{ fontSize: '1.3rem', fontWeight: 800, color: '#78350f', marginTop: '2px' }}>
                {currentDiagnosis.metrics?.chlorosisIndex || 16}%
              </div>
              <div style={{ height: '6px', background: '#fef3c7', borderRadius: '4px', overflow: 'hidden', marginTop: '6px' }}>
                <div style={{ width: `${currentDiagnosis.metrics?.chlorosisIndex || 16}%`, background: '#f59e0b', height: '100%' }}></div>
              </div>
            </div>

            <div style={{ background: '#fef2f2', border: '1px solid #fecaca', padding: '12px 14px', borderRadius: '12px' }}>
              <span style={{ fontSize: '0.78rem', color: '#991b1b', fontWeight: 700 }}>🟤 Necrotic Tissue</span>
              <div style={{ fontSize: '1.3rem', fontWeight: 800, color: '#7f1d1d', marginTop: '2px' }}>
                {currentDiagnosis.metrics?.necrosisIndex || 10}%
              </div>
              <div style={{ height: '6px', background: '#fee2e2', borderRadius: '4px', overflow: 'hidden', marginTop: '6px' }}>
                <div style={{ width: `${currentDiagnosis.metrics?.necrosisIndex || 10}%`, background: '#ef4444', height: '100%' }}></div>
              </div>
            </div>
          </div>

          {/* Symptoms & Etiology Summary */}
          <div style={{ background: 'rgba(255, 255, 255, 0.75)', border: '1px solid var(--slate-200)', borderRadius: '14px', padding: '16px 20px', marginBottom: '22px' }}>
            <h4 style={{ fontSize: '0.94rem', fontWeight: 800, color: 'var(--slate-800)', marginBottom: '6px' }}>
              Observed Symptoms:
            </h4>
            <p style={{ fontSize: '0.88rem', color: 'var(--slate-600)', lineHeight: '1.5' }}>
              {diseaseDetail.symptoms}
            </p>
          </div>

          {/* Tabbed Treatment Protocol */}
          <div style={{ marginBottom: '24px' }}>
            <div style={{ display: 'flex', gap: '8px', borderBottom: '2px solid var(--slate-200)', paddingBottom: '8px', marginBottom: '16px' }}>
              <button
                className={`btn-secondary ${activeTab === 'organic' ? 'btn-primary' : ''}`}
                style={{ padding: '8px 18px', fontSize: '0.88rem' }}
                onClick={() => setActiveTab('organic')}
              >
                🌿 Organic Protocol
              </button>
              <button
                className={`btn-secondary ${activeTab === 'chemical' ? 'btn-primary' : ''}`}
                style={{ padding: '8px 18px', fontSize: '0.88rem' }}
                onClick={() => setActiveTab('chemical')}
              >
                💊 Chemical Formulation
              </button>
              <button
                className={`btn-secondary ${activeTab === 'prevention' ? 'btn-primary' : ''}`}
                style={{ padding: '8px 18px', fontSize: '0.88rem' }}
                onClick={() => setActiveTab('prevention')}
              >
                🛡️ Cultural Prevention
              </button>
            </div>

            {/* Organic Tab Content */}
            {activeTab === 'organic' && (
              <div style={{ background: '#f0fdf4', border: '1.5px solid var(--primary-200)', borderRadius: '14px', padding: '20px' }}>
                <h4 style={{ fontSize: '0.96rem', fontWeight: 800, color: '#166534', marginBottom: '10px' }}>
                  Recommended Biological Products:
                </h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '14px' }}>
                  {(diseaseDetail.organicTreatment?.products || []).map((prod, idx) => (
                    <span key={idx} style={{ background: '#ffffff', color: '#15803d', border: '1px solid #86efac', padding: '6px 12px', borderRadius: '8px', fontSize: '0.84rem', fontWeight: 700 }}>
                      ✓ {prod}
                    </span>
                  ))}
                </div>
                <div style={{ fontSize: '0.86rem', color: '#1e293b', lineHeight: '1.5', marginBottom: '8px' }}>
                  <strong>Application Guidance:</strong> {diseaseDetail.organicTreatment?.instructions}
                </div>
                <div style={{ fontSize: '0.82rem', color: '#15803d', fontWeight: 600 }}>
                  <strong>Safety:</strong> {diseaseDetail.organicTreatment?.safety}
                </div>
              </div>
            )}

            {/* Chemical Tab Content */}
            {activeTab === 'chemical' && (
              <div style={{ background: '#eff6ff', border: '1.5px solid #bfdbfe', borderRadius: '14px', padding: '20px' }}>
                <h4 style={{ fontSize: '0.96rem', fontWeight: 800, color: '#1e40af', marginBottom: '10px' }}>
                  Targeted Fungicide Formulations:
                </h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '14px' }}>
                  {(diseaseDetail.chemicalTreatment?.products || []).map((prod, idx) => (
                    <span key={idx} style={{ background: '#ffffff', color: '#1d4ed8', border: '1px solid #93c5fd', padding: '6px 12px', borderRadius: '8px', fontSize: '0.84rem', fontWeight: 700 }}>
                      💊 {prod}
                    </span>
                  ))}
                </div>
                <div style={{ fontSize: '0.86rem', color: '#1e293b', lineHeight: '1.5', marginBottom: '8px' }}>
                  <strong>Dosage & Frequency:</strong> {diseaseDetail.chemicalTreatment?.instructions}
                </div>
                <div style={{ fontSize: '0.82rem', color: '#dc2626', fontWeight: 700 }}>
                  <strong>Pre-Harvest Waiting Period (PHI):</strong> {diseaseDetail.chemicalTreatment?.safety}
                </div>
              </div>
            )}

            {/* Prevention Tab Content */}
            {activeTab === 'prevention' && (
              <div style={{ background: '#f8fafc', border: '1.5px solid #e2e8f0', borderRadius: '14px', padding: '20px' }}>
                <h4 style={{ fontSize: '0.96rem', fontWeight: 800, color: '#334155', marginBottom: '10px' }}>
                  Farm Sanitation & Long-Term Prevention:
                </h4>
                <p style={{ fontSize: '0.88rem', color: '#475569', lineHeight: '1.6' }}>
                  {diseaseDetail.prevention}
                </p>
              </div>
            )}
          </div>

          {/* Action Row */}
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <button
              className="btn-primary"
              onClick={() => setIsTreatmentModalOpen(true)}
            >
              <Pill size={18} />
              <span>Log Treatment to History</span>
            </button>
            <button
              className="btn-secondary"
              onClick={() => navigateTo('fertilizer')}
            >
              <span>🌾 Check Fertilizer Needs</span>
            </button>
            <button
              className="btn-secondary"
              onClick={() => navigateTo('export')}
            >
              <FileDown size={18} />
              <span>Export Pathology Report</span>
            </button>
          </div>
        </div>
      </div>

      {/* ── Treatment Logging Modal ────────────────────────────── */}
      {isTreatmentModalOpen && (
        <Modal
          isOpen={isTreatmentModalOpen}
          onClose={() => setIsTreatmentModalOpen(false)}
          title={`Log Treatment: ${currentDiagnosis.crop}`}
        >
          <form onSubmit={handleCreateTreatment}>
            <div style={{ marginBottom: '16px' }}>
              <label className="form-label">Crop & Diagnosis:</label>
              <input 
                className="input-control" 
                value={`${currentDiagnosis.crop} — ${diseaseDetail.name}`} 
                disabled 
              />
            </div>

            <div style={{ marginBottom: '16px' }}>
              <label className="form-label">Selected Remedy / Product Name:</label>
              <input 
                className="input-control" 
                placeholder="e.g. Neem Oil 5ml/L or Ridomil Gold 2g/L"
                value={trtProductName}
                onChange={(e) => setTrtProductName(e.target.value)}
                required
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginBottom: '16px' }}>
              <div>
                <label className="form-label">Dosage Concentration:</label>
                <input 
                  className="input-control" 
                  value={trtDosage}
                  onChange={(e) => setTrtDosage(e.target.value)}
                  required
                />
              </div>
              <div>
                <label className="form-label">Plot Acreage Treated:</label>
                <input 
                  className="input-control" 
                  defaultValue="2.5 ha"
                />
              </div>
            </div>

            <div style={{ marginBottom: '20px' }}>
              <label className="form-label">Agronomist Application Notes:</label>
              <textarea 
                className="input-control" 
                rows="3"
                placeholder="Spray conditions, weather, equipment used..."
                value={trtNotes}
                onChange={(e) => setTrtNotes(e.target.value)}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
              <button 
                type="button" 
                className="btn-secondary"
                onClick={() => setIsTreatmentModalOpen(false)}
              >
                Cancel
              </button>
              <button type="submit" className="btn-primary">
                Save & Open Treatment Log
              </button>
            </div>
          </form>
        </Modal>
      )}
    </div>
  );
};
