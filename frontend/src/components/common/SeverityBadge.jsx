import React from 'react';

export const SeverityBadge = ({ severity }) => {
  const sev = (severity || 'unknown').toLowerCase();
  
  if (sev === 'none' || sev === 'healthy' || sev === 'optimal') {
    return (
      <span className="sev-pill sev-pill-healthy">
        <span>●</span> Healthy
      </span>
    );
  }

  if (sev === 'low' || sev === 'mild') {
    return (
      <span className="sev-pill sev-pill-low">
        <span>●</span> Low Severity
      </span>
    );
  }

  if (sev === 'medium' || sev === 'moderate') {
    return (
      <span className="sev-pill sev-pill-medium">
        <span>▲</span> Medium Risk
      </span>
    );
  }

  if (sev === 'high' || sev === 'critical' || sev === 'severe') {
    return (
      <span className="sev-pill sev-pill-high">
        <span>⚠️</span> Critical
      </span>
    );
  }

  return (
    <span className="sev-pill sev-pill-low">
      <span>●</span> {severity}
    </span>
  );
};
