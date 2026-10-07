import React, { useState } from 'react';
import { useFarm } from '../../context/FarmContext';
import { siteConfig } from '../../config/siteConfig';
import { MetricCard } from '../common/MetricCard';
import { Modal } from '../common/Modal';
import { 
  Pill, 
  Plus, 
  CheckCircle2, 
  Clock, 
  DollarSign, 
  Layers, 
  Trash2, 
  Filter,
  Check
} from 'lucide-react';

export const TreatmentHistoryView = () => {
  const { treatments, addTreatment, updateTreatmentStatus, deleteTreatment, navigateTo } = useFarm();

  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [filterType, setFilterType] = useState('ALL'); // ALL | organic | chemical | integrated

  // Form State
  const [formCrop, setFormCrop] = useState('Tomato');
  const [formDisease, setFormDisease] = useState('Early_Disease_Symptoms');
  const [formType, setFormType] = useState('organic');
  const [formProduct, setFormProduct] = useState('');
  const [formDosage, setFormDosage] = useState('5 ml / L');
  const [formArea, setFormArea] = useState('2.0 ha');
  const [formCost, setFormCost] = useState('45');
  const [formNotes, setFormNotes] = useState('');

  const handleFormSubmit = (e) => {
    e.preventDefault();
    addTreatment({
      crop: formCrop,
      disease: formDisease,
      treatmentType: formType,
      productName: formProduct,
      dosage: formDosage,
      areaTreated: formArea,
      cost: parseFloat(formCost) || 0,
      notes: formNotes,
      plannedDate: new Date().toISOString()
    });
    setIsAddModalOpen(false);
    // Reset form
    setFormProduct('');
    setFormNotes('');
  };

  const filtered = treatments.filter(t => 
    filterType === 'ALL' || t.treatmentType === filterType
  );

  // Metrics
  const activeCount = treatments.filter(t => t.status === 'in_progress').length;
  const completedCount = treatments.filter(t => t.status === 'completed').length;
  const totalCost = treatments.reduce((acc, curr) => acc + (curr.cost || 0), 0);

  return (
    <div className="page-body">
      {/* ── Page Header ────────────────────────────────────────── */}
      <div className="glass-panel" style={{ padding: '24px 30px', marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '1.6rem' }}>💊</span>
            <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--slate-900)' }}>
              Crop Treatment Management
            </h2>
          </div>
          <p style={{ color: 'var(--slate-600)', marginTop: '4px', fontSize: '0.94rem' }}>
            Track fungicide sprays, biological inoculants, costs, and crop recovery schedules
          </p>
        </div>
        <div>
          <button 
            className="btn-primary"
            onClick={() => setIsAddModalOpen(true)}
          >
            <Plus size={18} />
            <span>+ Log New Treatment</span>
          </button>
        </div>
      </div>

      {/* ── Top Metrics ────────────────────────────────────────── */}
      <div className="metric-grid-4">
        <MetricCard
          icon="⏳"
          iconColorClass="mc-amber"
          label="Active Treatments"
          value={activeCount}
          delta="Under monitoring"
        />
        <MetricCard
          icon="✅"
          iconColorClass="mc-green"
          label="Completed Cycles"
          value={completedCount}
          delta="Successfully cured"
        />
        <MetricCard
          icon="💵"
          iconColorClass="mc-blue"
          label="Total Treatment Spend"
          value={`$${totalCost.toFixed(0)}`}
          delta="Budget tracked"
        />
        <MetricCard
          icon="🌾"
          iconColorClass="mc-teal"
          label="Protected Plots"
          value={`${treatments.length} logs`}
          delta="Full coverage"
        />
      </div>

      {/* ── Treatment Filter Tabs ──────────────────────────────── */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
        {['ALL', 'organic', 'chemical', 'integrated'].map((type) => (
          <button
            key={type}
            className={`btn-secondary ${filterType === type ? 'btn-primary' : ''}`}
            style={{ textTransform: 'capitalize', padding: '8px 18px', fontSize: '0.88rem' }}
            onClick={() => setFilterType(type)}
          >
            {type === 'ALL' ? 'All Formulations' : `${type} Treatments`}
          </button>
        ))}
      </div>

      {/* ── Treatments Table ───────────────────────────────────── */}
      <div className="av-table-container">
        {filtered.length === 0 ? (
          <div style={{ padding: '60px 20px', textAlign: 'center' }}>
            <div style={{ fontSize: '3rem', marginBottom: '12px' }}>💊</div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--slate-800)' }}>No treatments logged</h3>
            <p style={{ color: 'var(--slate-500)', marginTop: '4px' }}>Click "+ Log New Treatment" above to record your first crop remedy.</p>
          </div>
        ) : (
          <table className="av-table">
            <thead>
              <tr>
                <th>Crop & Plot</th>
                <th>Target Pathology</th>
                <th>Method</th>
                <th>Applied Formulation & Dosage</th>
                <th>Status</th>
                <th>Cost</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((t) => {
                const diseaseName = siteConfig.diseases[t.disease]?.name || t.disease;
                const isDone = t.status === 'completed';

                return (
                  <tr key={t.id}>
                    <td style={{ fontWeight: 800, color: 'var(--slate-900)' }}>
                      🍃 {t.crop || 'Field'}
                      <div style={{ fontSize: '0.75rem', color: 'var(--slate-500)', fontWeight: 500 }}>
                        Area: {t.areaTreated || '2.0 ha'}
                      </div>
                    </td>
                    <td>
                      <div style={{ fontWeight: 700, color: 'var(--slate-800)' }}>{diseaseName}</div>
                      <div style={{ fontSize: '0.78rem', color: 'var(--slate-500)' }}>
                        {t.notes ? (t.notes.length > 50 ? t.notes.slice(0, 50) + '...' : t.notes) : 'Standard spray cycle'}
                      </div>
                    </td>
                    <td>
                      <span style={{ 
                        display: 'inline-block',
                        padding: '3px 10px', 
                        borderRadius: '20px', 
                        fontSize: '0.78rem', 
                        fontWeight: 700,
                        textTransform: 'uppercase',
                        background: t.treatmentType === 'organic' ? '#dcfce7' : (t.treatmentType === 'chemical' ? '#dbeafe' : '#fef3c7'),
                        color: t.treatmentType === 'organic' ? '#15803d' : (t.treatmentType === 'chemical' ? '#1d4ed8' : '#b45309'),
                      }}>
                        {t.treatmentType}
                      </span>
                    </td>
                    <td>
                      <div style={{ fontWeight: 700, color: 'var(--slate-900)' }}>{t.productName}</div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--primary-700)', fontWeight: 600 }}>{t.dosage}</div>
                    </td>
                    <td>
                      <button
                        onClick={() => updateTreatmentStatus(t.id, isDone ? 'in_progress' : 'completed')}
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '6px',
                          padding: '5px 12px',
                          borderRadius: '8px',
                          border: isDone ? '1px solid #86efac' : '1px solid #fde68a',
                          background: isDone ? '#f0fdf4' : '#fffbeb',
                          color: isDone ? '#15803d' : '#92400e',
                          fontWeight: 700,
                          fontSize: '0.8rem',
                          cursor: 'pointer'
                        }}
                      >
                        {isDone ? <Check size={14} /> : <Clock size={14} />}
                        <span>{isDone ? 'Completed' : 'In Progress'}</span>
                      </button>
                    </td>
                    <td style={{ fontWeight: 800, color: 'var(--slate-900)' }}>
                      ${t.cost || 0}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn-outline-leaf"
                        style={{ color: '#ef4444', borderColor: '#fca5a5' }}
                        onClick={() => deleteTreatment(t.id)}
                        title="Delete log"
                      >
                        <Trash2 size={14} />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* ── Add Treatment Modal ─────────────────────────────────── */}
      {isAddModalOpen && (
        <Modal
          isOpen={isAddModalOpen}
          onClose={() => setIsAddModalOpen(false)}
          title="Log New Crop Treatment"
        >
          <form onSubmit={handleFormSubmit}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginBottom: '16px' }}>
              <div>
                <label className="form-label">Crop Type:</label>
                <select 
                  className="input-control"
                  value={formCrop}
                  onChange={(e) => setFormCrop(e.target.value)}
                >
                  {siteConfig.defaultProfile.primaryCrops.map(c => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="form-label">Target Pathology:</label>
                <select
                  className="input-control"
                  value={formDisease}
                  onChange={(e) => setFormDisease(e.target.value)}
                >
                  {Object.keys(siteConfig.diseases).map(k => (
                    <option key={k} value={k}>{siteConfig.diseases[k].name}</option>
                  ))}
                </select>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginBottom: '16px' }}>
              <div>
                <label className="form-label">Treatment Type:</label>
                <select
                  className="input-control"
                  value={formType}
                  onChange={(e) => setFormType(e.target.value)}
                >
                  <option value="organic">Organic / Biological</option>
                  <option value="chemical">Chemical / Systemic</option>
                  <option value="integrated">Integrated IPM</option>
                </select>
              </div>

              <div>
                <label className="form-label">Estimated Cost ($):</label>
                <input 
                  type="number"
                  className="input-control"
                  value={formCost}
                  onChange={(e) => setFormCost(e.target.value)}
                />
              </div>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <label className="form-label">Product Name & Formulation:</label>
              <input 
                className="input-control"
                placeholder="e.g. Copper Hydroxide 2g/L or Neem Oil"
                value={formProduct}
                onChange={(e) => setFormProduct(e.target.value)}
                required
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginBottom: '16px' }}>
              <div>
                <label className="form-label">Application Dosage:</label>
                <input 
                  className="input-control"
                  value={formDosage}
                  onChange={(e) => setFormDosage(e.target.value)}
                  required
                />
              </div>

              <div>
                <label className="form-label">Plot Area Treated:</label>
                <input 
                  className="input-control"
                  value={formArea}
                  onChange={(e) => setFormArea(e.target.value)}
                  required
                />
              </div>
            </div>

            <div style={{ marginBottom: '22px' }}>
              <label className="form-label">Field Observations & Weather:</label>
              <textarea 
                className="input-control"
                rows="3"
                placeholder="Observed symptom progression, spray nozzles used..."
                value={formNotes}
                onChange={(e) => setFormNotes(e.target.value)}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
              <button 
                type="button" 
                className="btn-secondary"
                onClick={() => setIsAddModalOpen(false)}
              >
                Cancel
              </button>
              <button type="submit" className="btn-primary">
                Save Treatment Log
              </button>
            </div>
          </form>
        </Modal>
      )}
    </div>
  );
};
