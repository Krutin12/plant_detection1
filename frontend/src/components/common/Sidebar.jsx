import React from 'react';
import { useFarm } from '../../context/FarmContext';
import { siteConfig } from '../../config/siteConfig';
import { 
  LayoutDashboard, 
  ScanLine, 
  ClipboardList, 
  Pill, 
  Sprout, 
  BarChart3, 
  UserCircle2, 
  FileSpreadsheet, 
  Database,
  X,
  ChevronRight,
  ArrowRight
} from 'lucide-react';

const iconMap = {
  LayoutDashboard: LayoutDashboard,
  ScanLine: ScanLine,
  ClipboardList: ClipboardList,
  Pill: Pill,
  Sprout: Sprout,
  BarChart3: BarChart3,
  UserCircle2: UserCircle2,
  FileSpreadsheet: FileSpreadsheet,
  Database: Database
};

export const Sidebar = () => {
  const { currentPage, navigateTo, isSidebarOpen, toggleSidebar, profile } = useFarm();

  if (!isSidebarOpen) {
    return null;
  }

  return (
    <aside className="av-sidebar">
      {/* Brand Header */}
      <div className="av-sidebar-header">
        <div className="av-brand-wrap">
          <div className="av-brand-logo">🌿</div>
          <div className="av-brand-text">
            <h2>{siteConfig.appName}</h2>
            <p>{siteConfig.appSubtitle}</p>
          </div>
        </div>
        <button 
          onClick={toggleSidebar}
          style={{
            background: 'var(--slate-100)',
            border: '1px solid var(--slate-200)',
            borderRadius: '8px',
            width: '32px',
            height: '32px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            color: 'var(--slate-600)'
          }}
          title="Close Sidebar"
        >
          <X size={18} />
        </button>
      </div>

      {/* Active Farm Card */}
      <div 
        className="av-farm-badge-box"
        onClick={() => navigateTo('profile')}
        title="View Farm Details"
      >
        <div className="av-farm-header-row">
          <span className="av-farm-status-tag">● Active Farm</span>
          <ArrowRight size={14} color="var(--primary-600)" />
        </div>
        <div className="av-farm-name-txt">{profile.farmName || 'Green Valley Farm'}</div>
        <div className="av-farm-sub-txt">
          <span>👤 {profile.farmerName}</span> • <span>{profile.totalArea || 0} ha</span>
        </div>
      </div>

      {/* Nav Menu Items */}
      <ul className="av-nav-list">
        {siteConfig.navItems.map((item) => {
          const IconComponent = iconMap[item.icon] || LayoutDashboard;
          const isActive = currentPage === item.id;

          return (
            <li key={item.id}>
              <button
                className={`av-nav-item-btn ${isActive ? 'active' : ''}`}
                onClick={() => navigateTo(item.id)}
              >
                <div className="av-nav-icon-wrap">
                  <IconComponent size={19} />
                </div>
                <span style={{ flex: 1 }}>{item.label}</span>
                {isActive && <ChevronRight size={16} opacity={0.7} />}
              </button>
            </li>
          );
        })}
      </ul>

      {/* Botanical Footer */}
      <div className="av-sidebar-footer">
        <div style={{ fontSize: '1.25rem', marginBottom: '4px' }}>🌿🌱</div>
        <div className="av-botanical-quote">Healthier Crops</div>
        <div className="av-botanical-subquote">for a greener future</div>
      </div>
    </aside>
  );
};
