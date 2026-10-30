import React from 'react';

export default function ConfusionMatrix({ matrix, modelName }) {
  if (!matrix || matrix.length !== 2) {
    return <p style={{ fontSize: '12px', color: '#64748b' }}>Matritsa mavjud emas</p>;
  }

  const tn = matrix[0][0];
  const fp = matrix[0][1];
  const fn = matrix[1][0];
  const tp = matrix[1][1];
  const total = tn + fp + fn + tp;

  const pct = (val) => (total > 0 ? ((val / total) * 100).toFixed(1) : '0.0');

  return (
    <div className="cm-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '12px', fontWeight: 600, color: '#334155' }}>
          Confusion Matrix ({modelName})
        </span>
        <span style={{ fontSize: '11px', color: '#64748b' }}>
          Jami test namunalari: {total.toLocaleString()}
        </span>
      </div>

      <div className="cm-grid">
        <div className="cm-cell header"></div>
        <div className="cm-cell header">Pred: Non-Def (0)</div>
        <div className="cm-cell header">Pred: Default (1)</div>

        <div className="cm-cell header" style={{ textAlign: 'right', paddingRight: '6px' }}>
          Actual: 0
        </div>
        <div className="cm-cell tn" title="True Negative: Non-Default correctly identified">
          <span className="cm-value">{tn.toLocaleString()}</span>
          <span className="cm-label">TN ({pct(tn)}%)</span>
        </div>
        <div className="cm-cell fp" title="False Positive: Non-Default incorrectly predicted as Default">
          <span className="cm-value">{fp.toLocaleString()}</span>
          <span className="cm-label">FP ({pct(fp)}%)</span>
        </div>

        <div className="cm-cell header" style={{ textAlign: 'right', paddingRight: '6px' }}>
          Actual: 1
        </div>
        <div className="cm-cell fn" title="False Negative: Actual Default missed (High Risk Error!)">
          <span className="cm-value">{fn.toLocaleString()}</span>
          <span className="cm-label">FN ({pct(fn)}%)</span>
        </div>
        <div className="cm-cell tp" title="True Positive: Default correctly caught">
          <span className="cm-value">{tp.toLocaleString()}</span>
          <span className="cm-label">TP ({pct(tp)}%)</span>
        </div>
      </div>
    </div>
  );
}
