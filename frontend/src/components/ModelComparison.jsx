import React from 'react';
import { Download, RefreshCw, Award, Clock, Zap, CheckCircle2 } from 'lucide-react';
import ConfusionMatrix from './ConfusionMatrix';

const MODEL_KEYS = [
  { key: 'knn', label: 'KNN', defaultName: 'K-Nearest Neighbors' },
  { key: 'random_forest', label: 'Random Forest', defaultName: 'Random Forest' },
  { key: 'decision_tree', label: 'Decision Tree', defaultName: 'Decision Tree' }
];

export default function ModelComparison({
  modelsData,
  onTrainAll,
  isTraining,
  onDownloadModel,
  downloadingModel
}) {
  const models = modelsData?.models || {};
  const isTrained = modelsData?.trained;
  const bestModelKey = modelsData?.best_model_by_f1?.model_name;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <h3>Machine Learning Modellari Taqqoslashi</h3>
            <p>
              UCI ID: 350 ma'lumotlarida o'qitilgan KNN, Random Forest va Decision Tree algoritmlarining sinov natijalari.
            </p>
          </div>
          <button
            className="btn btn-primary"
            onClick={onTrainAll}
            disabled={isTraining}
          >
            <RefreshCw size={16} className={isTraining ? 'spinner' : ''} />
            <span>{isTraining ? "Modellar o'qitilmoqda..." : "Train All Models (Qayta o'qitish)"}</span>
          </button>
        </div>

        {!isTrained ? (
          <div style={{ textAlign: 'center', padding: '40px 20px', color: '#64748b' }}>
            <p style={{ fontSize: '15px', marginBottom: '16px' }}>
              Modellar hali o'qitilmagan yoki yuklanmagan. Haqiqiy UCI ma'lumotlarida modellarni o'qitish uchun tugmani bosing.
            </p>
            <button
              className="btn btn-primary"
              onClick={onTrainAll}
              disabled={isTraining}
            >
              <RefreshCw size={16} className={isTraining ? 'spinner' : ''} />
              <span>O'qitishni boshlash</span>
            </button>
          </div>
        ) : (
          <div className="models-grid">
            {MODEL_KEYS.map(({ key }) => {
              const model = models[key];
              if (!model) return null;

              const metrics = model.metrics || {};
              const isBest = bestModelKey && model.display_name.includes(bestModelKey);
              const isDownloading = downloadingModel === key;

              return (
                <div key={key} className={`model-card ${isBest ? 'is-best' : ''}`}>
                  {isBest && (
                    <div className="best-badge">
                      <Award size={12} style={{ display: 'inline', marginRight: '4px' }} />
                      Eng yaxshi F1
                    </div>
                  )}

                  <div className="model-header">
                    <div>
                      <h4>{model.display_name}</h4>
                      <p>{model.description}</p>
                    </div>
                  </div>

                  <div className="metrics-table">
                    <div className="metric-row">
                      <span className="metric-label">Accuracy (Umumiy aniqlik):</span>
                      <span className="metric-val">{(metrics.accuracy * 100).toFixed(2)}%</span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-label">Precision (Default = 1):</span>
                      <span className="metric-val highlight">{(metrics.precision * 100).toFixed(2)}%</span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-label">Recall (Qamrab olish):</span>
                      <span className="metric-val highlight">{(metrics.recall * 100).toFixed(2)}%</span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-label">F1-Score (Garmonik o'rtacha):</span>
                      <span className="metric-val highlight" style={{ color: '#2563eb', fontWeight: 800 }}>
                        {(metrics.f1 * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-label">ROC-AUC:</span>
                      <span className="metric-val">
                        {metrics.roc_auc ? (metrics.roc_auc * 100).toFixed(2) + '%' : 'N/A'}
                      </span>
                    </div>
                    <div className="metric-row" style={{ borderTop: '1px dashed #cbd5e1', paddingTop: '6px', marginTop: '2px' }}>
                      <span className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Clock size={12} /> O'qitish vaqti:
                      </span>
                      <span className="metric-val">{metrics.training_time}s</span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-label" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Zap size={12} /> Test bashorat vaqti:
                      </span>
                      <span className="metric-val">{metrics.prediction_time}s</span>
                    </div>
                  </div>

                  {/* Confusion Matrix */}
                  <ConfusionMatrix
                    matrix={metrics.confusion_matrix}
                    modelName={model.display_name}
                  />

                  {/* Download Button */}
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => onDownloadModel(key)}
                    disabled={isDownloading}
                    style={{ marginTop: 'auto' }}
                  >
                    <Download size={14} className={isDownloading ? 'spinner' : ''} />
                    <span>
                      {isDownloading
                        ? 'Yuklab olinmoqda...'
                        : `Download ${model.display_name.split(' ')[0]} (.joblib)`}
                    </span>
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
