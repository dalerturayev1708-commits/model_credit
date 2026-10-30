import React from 'react';
import {
  LayoutDashboard,
  Database,
  Cpu,
  BarChart3,
  CreditCard,
  Layers
} from 'lucide-react';

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'dataset', label: 'Dataset', icon: Database },
  { id: 'models', label: 'Model Comparison', icon: Cpu },
  { id: 'visualizations', label: 'Visualizations', icon: BarChart3 },
  { id: 'predictions', label: 'Predictions', icon: CreditCard },
];

export default function Sidebar({ activeTab, setActiveTab }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="logo-icon">
          <Layers size={22} />
        </div>
        <div className="logo-text">
          <h1>DefaultGuard</h1>
          <span>ML Credit Analytics</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              className={`nav-item ${isActive ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <Icon size={18} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <p>UCI Machine Learning</p>
        <div className="dataset-badge">Dataset ID: #350</div>
      </div>
    </aside>
  );
}
