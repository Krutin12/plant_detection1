import React from 'react';
import { FarmProvider, useFarm } from './context/FarmContext';
import { Topbar } from './components/common/Topbar';
import { Sidebar } from './components/common/Sidebar';

// Views
import { DashboardView } from './components/views/DashboardView';
import { DiseaseDetectionView } from './components/views/DiseaseDetectionView';
import { DetectionHistoryView } from './components/views/DetectionHistoryView';
import { TreatmentHistoryView } from './components/views/TreatmentHistoryView';
import { FertilizerCalculatorView } from './components/views/FertilizerCalculatorView';
import { FarmAnalyticsView } from './components/views/FarmAnalyticsView';
import { UserProfileView } from './components/views/UserProfileView';
import { ExportReportsView } from './components/views/ExportReportsView';
import { DataManagementView } from './components/views/DataManagementView';

import './config/theme.css';

const MainRouter = () => {
  const { currentPage } = useFarm();

  const renderView = () => {
    switch (currentPage) {
      case 'dashboard':
        return <DashboardView />;
      case 'detection':
        return <DiseaseDetectionView />;
      case 'history':
        return <DetectionHistoryView />;
      case 'treatment':
        return <TreatmentHistoryView />;
      case 'fertilizer':
        return <FertilizerCalculatorView />;
      case 'analytics':
        return <FarmAnalyticsView />;
      case 'profile':
        return <UserProfileView />;
      case 'export':
        return <ExportReportsView />;
      case 'data':
        return <DataManagementView />;
      default:
        return <DashboardView />;
    }
  };

  return (
    <div className="app-viewport">
      <div className="app-layout">
        <Sidebar />
        <div className="main-content-wrap">
          <Topbar />
          <main>
            {renderView()}
          </main>
        </div>
      </div>
    </div>
  );
};

export default function App() {
  return (
    <FarmProvider>
      <MainRouter />
    </FarmProvider>
  );
}
