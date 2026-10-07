import React, { useState } from 'react';
import { useFarm } from '../../context/FarmContext';
import { siteConfig } from '../../config/siteConfig';
import { FileSpreadsheet, Download, Printer, FileText, CheckCircle2 } from 'lucide-react';

export const ExportReportsView = () => {
  const { profile, detections, treatments } = useFarm();

  const [reportType, setReportType] = useState('full'); // full | detection | treatment
  const [downloadSuccess, setDownloadSuccess] = useState(false);

  const handleDownloadJSON = () => {
    const reportData = {
      generatedAt: new Date().toISOString(),
      farmProfile: profile,
      scansCount: detections.length,
      treatmentsCount: treatments.length,
      detections: detections,
      treatments: treatments
    };
    const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(reportData, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", jsonStr);
    downloadAnchor.setAttribute("download", `agrivision_report_${new Date().toISOString().slice(0, 10)}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
    setDownloadSuccess(true);
    setTimeout(() => setDownloadSuccess(false), 3000);
  };

  const handleDownloadCSV = () => {
    const headers = ["Scan_ID", "Crop", "Diagnosis", "Confidence", "Severity", "Timestamp"];
    const rows = detections.map(d => [
      d.id,
      d.crop,
      siteConfig.diseases[d.predictedDisease]?.name || d.predictedDisease,
      `${((d.confidence || 0.95) * 100).toFixed(1)}%`,
      d.severity,
      d.timestamp
    ]);
    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `crop_scans_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    setDownloadSuccess(true);
    setTimeout(() => setDownloadSuccess(false), 3000);
  };

  const handlePrintPDF = () => {
    window.print();
  };

  return (
    <div className="page-body">
      {/* ── Page Header ────────────────────────────────────────── */}
      <div className="glass-panel" style={{ padding: '24px 30px', marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '1.6rem' }}>📄</span>
            <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--slate-900)' }}>
              Export Agronomic Pathology Reports
            </h2>
          </div>
          <p style={{ color: 'var(--slate-600)', marginTop: '4px', fontSize: '0.94rem' }}>
            Generate executive compliance summaries, PDF farm certificates, and raw dataset exports
          </p>
        </div>
      </div>

      {downloadSuccess && (
        <div style={{ background: '#dcfce7', border: '1.5px solid #86efac', borderRadius: '12px', padding: '14px 20px', color: '#166534', fontWeight: 700, marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <CheckCircle2 size={18} />
          <span>Report file generated and downloaded successfully!</span>
        </div>
      )}

      {/* ── Export Controls & Preview Grid ─────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.8fr', gap: '26px' }}>
        {/* Left Column: Format Options */}
        <div className="glass-panel" style={{ padding: '26px' }}>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--slate-900)', marginBottom: '18px' }}>
            Select Report Specifications
          </h3>

          <div style={{ marginBottom: '18px' }}>
            <label className="form-label">Report Category:</label>
            <select 
              className="input-control"
              value={reportType}
              onChange={(e) => setReportType(e.target.value)}
            >
              <option value="full">Full Estate Crop Health Audit</option>
              <option value="detection">Pathology Scan Log Only</option>
              <option value="treatment">Treatment & Input Expenditures</option>
            </select>
          </div>

          <div style={{ marginBottom: '24px' }}>
            <label className="form-label">Reporting Period:</label>
            <select className="input-control" defaultValue="all">
              <option value="7">Last 7 Days (Immediate Scouting)</option>
              <option value="30">Last 30 Days (Monthly Trend)</option>
              <option value="all">Complete Season (Full History)</option>
            </select>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <button 
              className="btn-primary"
              style={{ width: '100%', padding: '14px' }}
              onClick={handlePrintPDF}
            >
              <Printer size={18} />
              <span>Print / Save as PDF Certificate</span>
            </button>

            <button 
              className="btn-secondary"
              style={{ width: '100%', padding: '14px' }}
              onClick={handleDownloadCSV}
            >
              <FileSpreadsheet size={18} />
              <span>Download CSV Spreadsheet</span>
            </button>

            <button 
              className="btn-secondary"
              style={{ width: '100%', padding: '14px' }}
              onClick={handleDownloadJSON}
            >
              <Download size={18} />
              <span>Download Full JSON Archive</span>
            </button>
          </div>
        </div>

        {/* Right Column: Live Printable Preview Document */}
        <div className="glass-panel" style={{ padding: '32px', background: '#ffffff', color: '#0f172a' }}>
          <div style={{ borderBottom: '2px solid #e2e8f0', paddingBottom: '20px', marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#15803d' }}>
                🌿 AgriVision AI Pathology Report
              </div>
              <div style={{ fontSize: '0.86rem', color: '#64748b', marginTop: '2px' }}>
                Official Agronomic Plant Health Evaluation
              </div>
            </div>
            <div style={{ textAlign: 'right', fontSize: '0.82rem', color: '#64748b' }}>
              <div><strong>Date:</strong> {new Date().toLocaleDateString()}</div>
              <div><strong>Status:</strong> Active Certification</div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', background: '#f8fafc', padding: '16px', borderRadius: '12px', marginBottom: '20px', fontSize: '0.88rem' }}>
            <div><strong>Farm Name:</strong> {profile.farmName}</div>
            <div><strong>Lead Farmer:</strong> {profile.farmerName}</div>
            <div><strong>Location:</strong> {profile.location}</div>
            <div><strong>Total Land Size:</strong> {profile.totalArea} Hectares</div>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 800, color: '#0f172a', marginBottom: '8px' }}>
              Executive Summary & Diagnostics:
            </h4>
            <p style={{ fontSize: '0.86rem', color: '#475569', lineHeight: '1.6' }}>
              A total of <strong>{detections.length}</strong> crop specimens have been inspected using deep neural pathology scanning. 
              <strong> {detections.filter(d => d.severity === 'None').length}</strong> specimens were determined optimal and healthy, 
              while <strong> {detections.filter(d => d.severity !== 'None').length}</strong> specimens displayed localized fungal or physiological stress requiring corrective intervention.
            </p>
          </div>

          <div>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 800, color: '#0f172a', marginBottom: '8px' }}>
              Recent Critical Scans:
            </h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
              <thead>
                <tr style={{ background: '#f1f5f9', borderBottom: '1px solid #cbd5e1', textAlign: 'left' }}>
                  <th style={{ padding: '8px 10px' }}>Crop</th>
                  <th style={{ padding: '8px 10px' }}>Diagnosis</th>
                  <th style={{ padding: '8px 10px' }}>Severity</th>
                  <th style={{ padding: '8px 10px' }}>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {detections.slice(0, 4).map(d => (
                  <tr key={d.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 700 }}>{d.crop}</td>
                    <td style={{ padding: '8px 10px' }}>{siteConfig.diseases[d.predictedDisease]?.name || d.predictedDisease}</td>
                    <td style={{ padding: '8px 10px' }}>{d.severity}</td>
                    <td style={{ padding: '8px 10px' }}>{((d.confidence || 0.95) * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
