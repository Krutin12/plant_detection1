import React from 'react';

export const MetricCard = ({
  icon,
  iconColorClass = "mc-green",
  label,
  value,
  delta,
  deltaType = "positive" // positive | negative | neutral
}) => {
  return (
    <div className="av-metric-card">
      <div className={`av-metric-circle ${iconColorClass}`}>
        {icon}
      </div>
      <div className="av-metric-info">
        <div className="av-metric-label">{label}</div>
        <div className="av-metric-num">{value}</div>
        {delta && (
          <div className={`av-metric-delta ${deltaType === 'positive' ? 'delta-green' : 'delta-red'}`}>
            <span>{deltaType === 'positive' ? '↗' : '↘'}</span>
            <span>{delta}</span>
          </div>
        )}
      </div>
    </div>
  );
};
