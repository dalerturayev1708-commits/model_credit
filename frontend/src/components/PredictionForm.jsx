import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  Send,
  AlertTriangle,
  CheckCircle,
  HelpCircle,
  ShieldCheck,
  ShieldAlert,
  Info
} from 'lucide-react';
import api from '../api';

const PRESETS = {
  good: {
    label: "A'lo mijoz (Past xavf)",
    values: {
      X1: 250000, X2: 2, X3: 1, X4: 2, X5: 34,
      X6: -1, X7: -1, X8: -1, X9: -1, X10: -1, X11: -1,
      X12: 12000, X13: 10000, X14: 9500, X15: 8000, X16: 6000, X17: 5000,
      X18: 12000, X19: 10000, X20: 9500, X21: 8000, X22: 6000, X23: 5000
    }
  },
  risky: {
    label: "Kechiktiruvchi mijoz (Yuqori xavf)",
    values: {
      X1: 30000, X2: 1, X3: 3, X4: 1, X5: 28,
      X6: 2, X7: 2, X8: 2, X9: 1, X10: 0, X11: 0,
      X12: 29000, X13: 28500, X14: 28000, X15: 27500, X16: 27000, X17: 26500,
      X18: 0, X19: 500, X20: 0, X21: 1000, X22: 500, X23: 0
    }
  },
  moderate: {
    label: "O'rtacha xavf (Aylanma kredit)",
    values: {
      X1: 100000, X2: 2, X3: 2, X4: 1, X5: 42,
      X6: 0, X7: 0, X8: 0, X9: 0, X10: 0, X11: 0,
      X12: 45000, X13: 43000, X14: 41000, X15: 39000, X16: 37000, X17: 35000,
      X18: 3000, X19: 2500, X20: 2500, X21: 2000, X22: 2000, X23: 2000
    }
  }
};

export default function PredictionForm({ featuresMeta }) {
  const [selectedModel, setSelectedModel] = useState('random_forest');
  const [formData, setFormData] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Initialize form with defaults from metadata
  useEffect(() => {
    if (featuresMeta && featuresMeta.length > 0) {
      const initial = {};
      featuresMeta.forEach((feat) => {
        initial[feat.id] = feat.default;
      });
      setFormData(initial);
    }
  }, [featuresMeta]);

  const handleInputChange = (fieldId, val) => {
    setFormData((prev) => ({
      ...prev,
      [fieldId]: Number(val),
    }));
  };

  const applyPreset = (presetKey) => {
    if (PRESETS[presetKey]) {
      setFormData({ ...PRESETS[presetKey].values });
      setPredictionResult(null);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await api.predict(selectedModel, formData);
      setPredictionResult(res);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || "Bashorat qilishda xatolik yuz berdi.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div className="card">
        <div className="card-title">
          <h3>Mijoz Kredit Qaytarmaslik Xavfini Bashorat Qilish</h3>
          <p>
            Mijozning 23 ta ko'rsatkichini kiriting yoki tayyor shablonlardan birini tanlang va tanlangan ML modeli orqali tahlil qiling.
          </p>
        </div>

        {/* Presets & Model Selector Bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px', flexWrap: 'wrap', gap: '14px' }}>
          <div className="presets-bar" style={{ margin: 0 }}>
            <span>Tayyor shablonlar:</span>
            {Object.entries(PRESETS).map(([key, p]) => (
              <button
                key={key}
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => applyPreset(key)}
              >
                <Sparkles size={12} color="#2563eb" />
                {p.label}
              </button>
            ))}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '13px', fontWeight: 600, color: '#334155' }}>
              Qo'llaniladigan Model:
            </span>
            <select
              className="form-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{ fontWeight: 600, color: '#2563eb' }}
            >
              <option value="random_forest">Random Forest (Tavsiya etiladi)</option>
              <option value="decision_tree">Decision Tree</option>
              <option value="knn">K-Nearest Neighbors (KNN)</option>
            </select>
          </div>
        </div>
      </div>

      {error && (
        <div className="notification-banner error">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        </div>
      )}

      <div className="pred-layout">
        {/* Form Inputs Grid */}
        <form onSubmit={handleSubmit} className="card" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <h4 style={{ fontSize: '16px', fontWeight: 700, borderBottom: '1px solid #e2e8f0', paddingBottom: '10px' }}>
            Mijoz Ko'rsatkichlari (23 ta belgi)
          </h4>

          <div className="form-grid">
            {featuresMeta && featuresMeta.map((feat) => {
              const val = formData[feat.id] !== undefined ? formData[feat.id] : feat.default;

              return (
                <div key={feat.id} className="form-group">
                  <label htmlFor={feat.id}>
                    <span>{feat.name}</span>
                    <span className="uz-hint">{feat.label_uz}</span>
                  </label>

                  {feat.type === 'categorical' && feat.options ? (
                    <select
                      id={feat.id}
                      className="form-select"
                      value={val}
                      onChange={(e) => handleInputChange(feat.id, e.target.value)}
                    >
                      {feat.options.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      id={feat.id}
                      type="number"
                      step={feat.step || 1}
                      min={feat.min}
                      max={feat.max}
                      className="form-input"
                      value={val}
                      onChange={(e) => handleInputChange(feat.id, e.target.value)}
                      required
                    />
                  )}
                </div>
              );
            })}
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '10px' }}>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading}
              style={{ minWidth: '180px' }}
            >
              <Send size={16} />
              <span>{loading ? 'Hisoblanmoqda...' : 'Predict (Bashorat qilish)'}</span>
            </button>
          </div>
        </form>

        {/* Prediction Results Card */}
        <div>
          {predictionResult ? (
            <div className={`result-card ${predictionResult.risk_assessment.category}`}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '12px', fontWeight: 600, color: '#64748b' }}>
                  Model: {predictionResult.model_name}
                </span>
                <span className={`risk-badge ${predictionResult.risk_assessment.category}`}>
                  {predictionResult.prediction === 1 ? <ShieldAlert size={16} /> : <ShieldCheck size={16} />}
                  {predictionResult.risk_assessment.level}
                </span>
              </div>

              <div>
                <span style={{ fontSize: '13px', color: '#64748b' }}>Bashorat qilingan holat:</span>
                <h3 style={{ fontSize: '22px', fontWeight: 800, color: predictionResult.prediction === 1 ? '#dc2626' : '#166534', marginTop: '4px' }}>
                  {predictionResult.prediction_label}
                </h3>
              </div>

              {/* Probability Meter */}
              <div className="prob-meter-container">
                <div className="prob-labels">
                  <span>Kredit qaytarmaslik ehtimoli:</span>
                  <span style={{ color: '#0f172a', fontWeight: 700 }}>
                    {predictionResult.probabilities.default_percentage}%
                  </span>
                </div>
                <div className="prob-bar-wrapper">
                  <div
                    className={`prob-bar-fill ${predictionResult.risk_assessment.category}`}
                    style={{ width: `${Math.min(predictionResult.probabilities.default_percentage, 100)}%` }}
                  />
                </div>
                <div className="prob-labels" style={{ fontSize: '11px', color: '#94a3b8' }}>
                  <span>0% (Xavfsiz)</span>
                  <span>50% (Chegara)</span>
                  <span>100% (Default)</span>
                </div>
              </div>

              <div style={{ background: '#f8fafc', padding: '14px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <p style={{ fontSize: '13px', fontWeight: 600, color: '#1e293b' }}>
                  Xulosa va tavsiya:
                </p>
                <p style={{ fontSize: '13px', color: '#475569', marginTop: '4px', lineHeight: 1.5 }}>
                  {predictionResult.risk_assessment.explanation_uz}
                </p>
              </div>

              <div className="disclaimer-box">
                <div style={{ display: 'flex', gap: '6px', alignItems: 'flex-start' }}>
                  <Info size={16} color="#64748b" style={{ flexShrink: 0, marginTop: '2px' }} />
                  <span>{predictionResult.disclaimer}</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="card" style={{ textAlign: 'center', padding: '50px 20px', color: '#64748b' }}>
              <HelpCircle size={40} color="#94a3b8" style={{ margin: '0 auto 14px' }} />
              <h4 style={{ fontSize: '16px', fontWeight: 600, color: '#334155' }}>
                Natija bu yerda ko'rinadi
              </h4>
              <p style={{ fontSize: '13px', marginTop: '6px' }}>
                Chap tomondagi ko'rsatkichlarni to'ldirib, «Predict» tugmasini bosing yoki yuqoridagi tayyor shablonlardan birini tanlang.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
