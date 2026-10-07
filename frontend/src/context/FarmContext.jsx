import React, { createContext, useContext, useState, useEffect } from 'react';
import { siteConfig } from '../config/siteConfig';

const FarmContext = createContext();

export const FarmProvider = ({ children }) => {
  // Navigation State
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // Profile State (with LocalStorage persistence)
  const [profile, setProfile] = useState(() => {
    try {
      const saved = localStorage.getItem('agri_profile');
      return saved ? JSON.parse(saved) : siteConfig.defaultProfile;
    } catch {
      return siteConfig.defaultProfile;
    }
  });

  // Detection History State
  const [detections, setDetections] = useState(() => {
    try {
      const saved = localStorage.getItem('agri_detections');
      return saved ? JSON.parse(saved) : siteConfig.initialDetections;
    } catch {
      return siteConfig.initialDetections;
    }
  });

  // Treatment Logs State
  const [treatments, setTreatments] = useState(() => {
    try {
      const saved = localStorage.getItem('agri_treatments');
      return saved ? JSON.parse(saved) : siteConfig.initialTreatments;
    } catch {
      return siteConfig.initialTreatments;
    }
  });

  // Most Recent Scan Result (For instant inspection)
  const [lastScanResult, setLastScanResult] = useState(null);

  // Sync to localStorage
  useEffect(() => {
    localStorage.setItem('agri_profile', JSON.stringify(profile));
  }, [profile]);

  useEffect(() => {
    localStorage.setItem('agri_detections', JSON.stringify(detections));
  }, [detections]);

  useEffect(() => {
    localStorage.setItem('agri_treatments', JSON.stringify(treatments));
  }, [treatments]);

  // Actions
  const navigateTo = (pageId, additionalState = null) => {
    if (additionalState && additionalState.lastScan) {
      setLastScanResult(additionalState.lastScan);
    }
    setCurrentPage(pageId);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(prev => !prev);
  };

  const updateProfile = (updatedFields) => {
    setProfile(prev => ({ ...prev, ...updatedFields }));
  };

  const addDetection = (newScan) => {
    const formatted = {
      id: `scan-${Date.now()}`,
      timestamp: new Date().toISOString(),
      ...newScan
    };
    setDetections(prev => [formatted, ...prev]);
    setLastScanResult(formatted);
    return formatted;
  };

  const deleteDetection = (id) => {
    setDetections(prev => prev.filter(item => item.id !== id));
  };

  const clearDetections = () => {
    setDetections([]);
  };

  const addTreatment = (newTreatment) => {
    const formatted = {
      id: `trt-${Date.now()}`,
      timestamp: new Date().toISOString(),
      status: 'in_progress',
      ...newTreatment
    };
    setTreatments(prev => [formatted, ...prev]);
    return formatted;
  };

  const updateTreatmentStatus = (id, newStatus) => {
    setTreatments(prev => prev.map(t => t.id === id ? { ...t, status: newStatus } : t));
  };

  const deleteTreatment = (id) => {
    setTreatments(prev => prev.filter(t => t.id !== id));
  };

  const resetToDefaults = () => {
    setProfile(siteConfig.defaultProfile);
    setDetections(siteConfig.initialDetections);
    setTreatments(siteConfig.initialTreatments);
    setLastScanResult(null);
    localStorage.removeItem('agri_profile');
    localStorage.removeItem('agri_detections');
    localStorage.removeItem('agri_treatments');
  };

  return (
    <FarmContext.Provider value={{
      currentPage,
      isSidebarOpen,
      profile,
      detections,
      treatments,
      lastScanResult,
      navigateTo,
      toggleSidebar,
      updateProfile,
      addDetection,
      deleteDetection,
      clearDetections,
      addTreatment,
      updateTreatmentStatus,
      deleteTreatment,
      setLastScanResult,
      resetToDefaults
    }}>
      {children}
    </FarmContext.Provider>
  );
};

export const useFarm = () => {
  const context = useContext(FarmContext);
  if (!context) {
    throw new Error('useFarm must be used within a FarmProvider');
  }
  return context;
};
