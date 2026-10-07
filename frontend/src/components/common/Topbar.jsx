import React from 'react';
import { useFarm } from '../../context/FarmContext';
import { siteConfig } from '../../config/siteConfig';
import { Menu, Bell, Sparkles, User, Sun, Moon } from 'lucide-react';

export const Topbar = () => {
  const { toggleSidebar, profile, currentPage, navigateTo } = useFarm();

  const currentItem = siteConfig.navItems.find(item => item.id === currentPage) || siteConfig.navItems[0];

  return (
    <header className="av-topbar">
      <div className="av-topbar-left">
        <button 
          className="av-sidebar-toggle-btn"
          onClick={toggleSidebar}
          title="Toggle Sidebar Menu"
        >
          <Menu size={18} />
          <span>Menu</span>
        </button>

        <div className="av-topbar-title-wrap">
          <div className="av-topbar-icon-box">
            🌿
          </div>
          <h1 className="av-topbar-title">{currentItem.label}</h1>
        </div>
      </div>

      <div className="av-topbar-right">
        {/* Real-Time AI Status Badge */}
        <div className="av-status-pill">
          <span className="av-status-dot"></span>
          <span>{siteConfig.activeStatusText}</span>
        </div>

        {/* Farmer Profile Chip */}
        <div 
          className="av-user-chip"
          onClick={() => navigateTo('profile')}
          title="Open Farmer Profile"
        >
          <div className="av-user-avatar">
            <User size={16} />
          </div>
          <span>{profile.farmerName || 'Farmer'}</span>
          <span style={{ fontSize: '0.75rem', color: 'var(--slate-400)' }}>▾</span>
        </div>
      </div>
    </header>
  );
};
