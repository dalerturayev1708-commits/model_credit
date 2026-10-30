import React from 'react';

export default function MetricCard({ title, value, subtitle, icon: Icon, color }) {
  return (
    <div className="metric-card">
      <div className="metric-info">
        <span>{title}</span>
        <h3>{value !== undefined && value !== null ? value : '—'}</h3>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {Icon && (
        <div className="metric-icon-wrap" style={color ? { color, backgroundColor: `${color}15` } : {}}>
          <Icon size={24} />
        </div>
      )}
    </div>
  );
}
