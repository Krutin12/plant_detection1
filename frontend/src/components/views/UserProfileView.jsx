import React, { useState } from 'react';
import { useFarm } from '../../context/FarmContext';
import { siteConfig } from '../../config/siteConfig';
import { User, MapPin, Phone, Mail, Sparkles, Check, RotateCcw } from 'lucide-react';

export const UserProfileView = () => {
  const { profile, updateProfile, resetToDefaults } = useFarm();

  const [formData, setFormData] = useState({ ...profile });
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [newCropInput, setNewCropInput] = useState('');

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleAddCrop = () => {
    if (newCropInput.trim() && !formData.primaryCrops.includes(newCropInput.trim())) {
      setFormData(prev => ({
        ...prev,
        primaryCrops: [...prev.primaryCrops, newCropInput.trim()]
      }));
      setNewCropInput('');
    }
  };

  const handleRemoveCrop = (cropToRemove) => {
    setFormData(prev => ({
      ...prev,
      primaryCrops: prev.primaryCrops.filter(c => c !== cropToRemove)
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    updateProfile(formData);
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 3000);
  };

  return (
    <div className="page-body">
      {/* ── Page Header ────────────────────────────────────────── */}
      <div className="glass-panel" style={{ padding: '24px 30px', marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '1.6rem' }}>👤</span>
            <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--slate-900)' }}>
              Farmer & Farm Estate Profile
            </h2>
          </div>
          <p style={{ color: 'var(--slate-600)', marginTop: '4px', fontSize: '0.94rem' }}>
            Manage farm acreage, soil characteristics, crop portfolio, and personalized agronomist settings
          </p>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button 
            type="button"
            className="btn-secondary"
            onClick={() => {
              if (window.confirm("Reset all settings & profile to default?")) {
                resetToDefaults();
                setFormData(siteConfig.defaultProfile);
              }
            }}
          >
            <RotateCcw size={16} />
            <span>Reset Defaults</span>
          </button>
        </div>
      </div>

      {/* ── Main Edit Form ─────────────────────────────────────── */}
      <form onSubmit={handleSubmit} className="glass-panel" style={{ padding: '32px' }}>
        {saveSuccess && (
          <div style={{ background: '#dcfce7', border: '1.5px solid #86efac', borderRadius: '12px', padding: '14px 20px', color: '#166534', fontWeight: 700, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Check size={18} />
            <span>Profile settings successfully saved! Topbar and Sidebar have been updated.</span>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '22px', marginBottom: '22px' }}>
          <div>
            <label className="form-label">Farmer Full Name:</label>
            <input
              type="text"
              name="farmerName"
              className="input-control"
              value={formData.farmerName || ''}
              onChange={handleChange}
              required
            />
          </div>

          <div>
            <label className="form-label">Farm / Estate Name:</label>
            <input
              type="text"
              name="farmName"
              className="input-control"
              value={formData.farmName || ''}
              onChange={handleChange}
              required
            />
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '22px', marginBottom: '22px' }}>
          <div>
            <label className="form-label">Farm Geographical Location:</label>
            <input
              type="text"
              name="location"
              className="input-control"
              value={formData.location || ''}
              onChange={handleChange}
            />
          </div>

          <div>
            <label className="form-label">Total Arable Area (Hectares):</label>
            <input
              type="number"
              step="0.1"
              name="totalArea"
              className="input-control"
              value={formData.totalArea || ''}
              onChange={handleChange}
              required
            />
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '22px', marginBottom: '22px' }}>
          <div>
            <label className="form-label">Contact Phone:</label>
            <input
              type="text"
              name="phone"
              className="input-control"
              value={formData.phone || ''}
              onChange={handleChange}
            />
          </div>

          <div>
            <label className="form-label">Agronomist / Farmer Email:</label>
            <input
              type="email"
              name="email"
              className="input-control"
              value={formData.email || ''}
              onChange={handleChange}
            />
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '22px', marginBottom: '22px' }}>
          <div>
            <label className="form-label">Farming Philosophy / Practice:</label>
            <select
              name="farmingType"
              className="input-control"
              value={formData.farmingType || 'Integrated Sustainable'}
              onChange={handleChange}
            >
              <option value="Organic Biological">Organic Biological</option>
              <option value="Integrated Sustainable">Integrated Sustainable (IPM)</option>
              <option value="Conventional High-Yield">Conventional High-Yield</option>
            </select>
          </div>

          <div>
            <label className="form-label">Predominant Soil Classification:</label>
            <input
              type="text"
              name="soilType"
              className="input-control"
              value={formData.soilType || 'Loamy Alluvial'}
              onChange={handleChange}
            />
          </div>
        </div>

        {/* Primary Crops Tag Management */}
        <div style={{ marginBottom: '28px' }}>
          <label className="form-label">Primary Cultivated Crops:</label>
          <div style={{ display: 'flex', gap: '10px', marginBottom: '12px' }}>
            <input
              type="text"
              className="input-control"
              placeholder="Add another crop (e.g. Soybean, Mustard, Rice)..."
              value={newCropInput}
              onChange={(e) => setNewCropInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); handleAddCrop(); }}}
              style={{ maxWidth: '380px' }}
            />
            <button
              type="button"
              className="btn-secondary"
              onClick={handleAddCrop}
            >
              + Add Crop
            </button>
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {(formData.primaryCrops || []).map(crop => (
              <span 
                key={crop}
                style={{ 
                  background: '#f0fdf4', 
                  border: '1.5px solid #86efac', 
                  color: '#15803d', 
                  padding: '6px 14px', 
                  borderRadius: '9999px',
                  fontWeight: 700,
                  fontSize: '0.86rem',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px'
                }}
              >
                <span>🍃 {crop}</span>
                <button
                  type="button"
                  onClick={() => handleRemoveCrop(crop)}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#166534', fontWeight: 800, fontSize: '0.9rem' }}
                >
                  ✕
                </button>
              </span>
            ))}
          </div>
        </div>

        {/* Save Button */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '14px' }}>
          <button type="submit" className="btn-primary" style={{ padding: '14px 32px', fontSize: '1.02rem' }}>
            <Check size={18} />
            <span>Save Profile Changes</span>
          </button>
        </div>
      </form>
    </div>
  );
};
