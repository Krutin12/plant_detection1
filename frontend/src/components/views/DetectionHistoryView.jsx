import React, { useState } from 'react';
import { useFarm } from '../../context/FarmContext';
import { siteConfig } from '../../config/siteConfig';
import { SeverityBadge } from '../common/SeverityBadge';
import { Modal } from '../common/Modal';
import { LeafSpecimen } from '../../utils/leafIllustrations';
import { 
  Search, 
  Filter, 
  Trash2, 
  Eye, 
  Download, 
  FileSpreadsheet, 
  Pill,
  ArrowUpDown
} from 'lucide-react';

export const DetectionHistoryView = () => {
  const { detections, deleteDetection, clearDetections, navigateTo, setLastScanResult } = useFarm();

  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCropFilter, setSelectedCropFilter] = useState('ALL');
  const [selectedSeverityFilter, setSelectedSeverityFilter] = useState('ALL');
  const [activeModalRecord, setActiveModalRecord] = useState(null);

  // Filter Logic
  const filtered = detections.filter(item => {
    const matchesSearch = 
      (item.crop || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
      (item.predictedDisease || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
      (siteConfig.diseases[item.predictedDisease]?.name || '').toLowerCase().includes(searchQuery.toLowerCase());

    const matchesCrop = selectedCropFilter === 'ALL' || item.crop === selectedCropFilter;
    const matchesSeverity = selectedSeverityFilter === 'ALL' || (item.severity || '').toLowerCase() === selectedSeverityFilter.toLowerCase();

    return matchesSearch && matchesCrop && matchesSeverity;
  });

  const cropsList = Array.from(new Set(detections.map(d => d.crop).filter(Boolean)));

  const handleExportCSV = () => {
    const headers = ["ID", "Crop", "Diagnosis", "Confidence", "Severity", "Timestamp"];
    const rows = filtered.map(f => [
      f.id,
      f.crop,
      siteConfig.diseases[f.predictedDisease]?.name || f.predictedDisease,
      `${((f.confidence || 0.9) * 100).toFixed(1)}%`,
      f.severity,
      f.timestamp
    ]);
    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `agrivision_detections_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="page-body">
      {/* ── Page Header ────────────────────────────────────────── */}
      <div className="glass-panel" style={{ padding: '24px 30px', marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '1.6rem' }}>📋</span>
            <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--slate-900)' }}>
              Plant Detection History
            </h2>
          </div>
          <p style={{ color: 'var(--slate-600)', marginTop: '4px', fontSize: '0.94rem' }}>
            Archived pathology records, historical severity trends, and symptom tracking
          </p>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button 
            className="btn-secondary"
            onClick={handleExportCSV}
            title="Download CSV"
          >
            <Download size={16} />
            <span>Export CSV</span>
          </button>
          <button 
            className="btn-primary"
            onClick={() => navigateTo('detection')}
          >
            <span>+ New AI Scan</span>
          </button>
        </div>
      </div>

      {/* ── Filters & Search Control Bar ───────────────────────── */}
      <div className="glass-panel" style={{ padding: '20px 24px', marginBottom: '20px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1.2fr 1.2fr auto', gap: '16px', alignItems: 'center' }}>
          {/* Search Input */}
          <div style={{ position: 'relative' }}>
            <input 
              type="text" 
              className="input-control" 
              placeholder="Search by crop, disease name or symptoms..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{ paddingLeft: '38px' }}
            />
            <Search size={18} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--slate-400)' }} />
          </div>

          {/* Crop Filter */}
          <div>
            <select 
              className="input-control"
              value={selectedCropFilter}
              onChange={(e) => setSelectedCropFilter(e.target.value)}
            >
              <option value="ALL">All Crops ({cropsList.length})</option>
              {cropsList.map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

          {/* Severity Filter */}
          <div>
            <select 
              className="input-control"
              value={selectedSeverityFilter}
              onChange={(e) => setSelectedSeverityFilter(e.target.value)}
            >
              <option value="ALL">All Severity Levels</option>
              <option value="none">Healthy Only</option>
              <option value="low">Low Risk</option>
              <option value="medium">Medium Risk</option>
              <option value="critical">Critical / High</option>
            </select>
          </div>

          {/* Reset Filters / Clear */}
          <button 
            className="btn-secondary"
            onClick={() => {
              setSearchQuery('');
              setSelectedCropFilter('ALL');
              setSelectedSeverityFilter('ALL');
            }}
          >
            Reset Filters
          </button>
        </div>
      </div>

      {/* ── Table of Records ───────────────────────────────────── */}
      <div className="av-table-container">
        {filtered.length === 0 ? (
          <div style={{ padding: '60px 20px', textAlign: 'center' }}>
            <div style={{ fontSize: '3rem', marginBottom: '12px' }}>🍃</div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--slate-800)' }}>No matching scans found</h3>
            <p style={{ color: 'var(--slate-500)', marginTop: '4px' }}>Try adjusting your filters or run a new scan from the Dashboard.</p>
          </div>
        ) : (
          <table className="av-table">
            <thead>
              <tr>
                <th>Crop Specimen</th>
                <th>Diagnosis & Pathogen</th>
                <th>Confidence</th>
                <th>Severity</th>
                <th>Date & Time</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((record) => {
                const disease = siteConfig.diseases[record.predictedDisease] || { name: record.predictedDisease };
                const dateStr = new Date(record.timestamp).toLocaleString('en-US', {
                  month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit'
                });

                return (
                  <tr key={record.id}>
                    <td style={{ fontWeight: 800, color: 'var(--slate-900)' }}>
                      <span style={{ marginRight: '6px' }}>🍃</span>
                      {record.crop || 'Field Plant'}
                    </td>
                    <td>
                      <div style={{ fontWeight: 700, color: 'var(--slate-900)' }}>{disease.name}</div>
                      <div style={{ fontSize: '0.78rem', color: 'var(--slate-500)' }}>{record.imageName || 'specimen_sample.png'}</div>
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontWeight: 800, color: 'var(--primary-700)', minWidth: '45px' }}>
                          {((record.confidence || 0.94) * 100).toFixed(1)}%
                        </span>
                        <div style={{ width: '60px', height: '6px', background: 'var(--slate-200)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ width: `${(record.confidence || 0.94) * 100}%`, background: 'var(--primary-600)', height: '100%' }}></div>
                        </div>
                      </div>
                    </td>
                    <td>
                      <SeverityBadge severity={record.severity} />
                    </td>
                    <td style={{ color: 'var(--slate-600)', fontSize: '0.84rem' }}>
                      {dateStr}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'inline-flex', gap: '6px' }}>
                        <button
                          className="btn-outline-leaf"
                          onClick={() => setActiveModalRecord(record)}
                          title="View Diagnosis"
                        >
                          <Eye size={14} />
                          <span>Details</span>
                        </button>
                        <button
                          className="btn-outline-leaf"
                          style={{ color: '#ef4444', borderColor: '#fca5a5' }}
                          onClick={() => deleteDetection(record.id)}
                          title="Delete Record"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* ── Detail Modal ───────────────────────────────────────── */}
      {activeModalRecord && (
        <Modal
          isOpen={Boolean(activeModalRecord)}
          onClose={() => setActiveModalRecord(null)}
          title={`Scan Archive Details: ${activeModalRecord.crop}`}
        >
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', background: 'var(--primary-50)', padding: '16px', borderRadius: '12px', marginBottom: '20px' }}>
              <div style={{ fontSize: '2.4rem' }}>🌿</div>
              <div>
                <SeverityBadge severity={activeModalRecord.severity} />
                <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--slate-900)', marginTop: '4px' }}>
                  {siteConfig.diseases[activeModalRecord.predictedDisease]?.name || activeModalRecord.predictedDisease}
                </h3>
                <span style={{ fontSize: '0.84rem', color: 'var(--slate-500)' }}>
                  Scanned on {new Date(activeModalRecord.timestamp).toLocaleString()}
                </span>
              </div>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <h4 style={{ fontSize: '0.92rem', fontWeight: 800, color: 'var(--slate-800)', marginBottom: '4px' }}>Diagnosis Symptoms</h4>
              <p style={{ fontSize: '0.88rem', color: 'var(--slate-600)', lineHeight: '1.5' }}>
                {siteConfig.diseases[activeModalRecord.predictedDisease]?.symptoms || "Optimal foliar tissue health with active photosynthesis."}
              </p>
            </div>

            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ fontSize: '0.92rem', fontWeight: 800, color: 'var(--slate-800)', marginBottom: '4px' }}>Recommended Treatment</h4>
              <p style={{ fontSize: '0.88rem', color: 'var(--primary-800)', lineHeight: '1.5' }}>
                {siteConfig.diseases[activeModalRecord.predictedDisease]?.organicTreatment?.instructions || "Maintain standard nutrient fertigation and regular monitoring."}
              </p>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
              <button 
                className="btn-secondary"
                onClick={() => setActiveModalRecord(null)}
              >
                Close
              </button>
              <button 
                className="btn-primary"
                onClick={() => {
                  setLastScanResult(activeModalRecord);
                  setActiveModalRecord(null);
                  navigateTo('detection');
                }}
              >
                Open in Pathology Studio ➔
              </button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};
