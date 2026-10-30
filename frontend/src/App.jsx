import React, { useState, useEffect } from 'react';
import {
  Users,
  Layers,
  Award,
  CheckCircle,
  AlertCircle,
  Database,
  ArrowRight,
  Download,
  Info,
  ExternalLink
} from 'lucide-react';
import api from './api';
import Sidebar from './components/Sidebar';
import MetricCard from './components/MetricCard';
import ModelComparison from './components/ModelComparison';
import Charts from './components/Charts';
import PredictionForm from './components/PredictionForm';
import './styles.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  
  // Data states
  const [summaryData, setSummaryData] = useState(null);
  const [modelsData, setModelsData] = useState(null);
  const [vizData, setVizData] = useState(null);
  
  // Action states
  const [isTraining, setIsTraining] = useState(false);
  const [downloadingModel, setDownloadingModel] = useState(null);
  const [notification, setNotification] = useState(null);

  const showNotification = (type, message) => {
    setNotification({ type, message });
    setTimeout(() => {
      setNotification(null);
    }, 6000);
  };

  // Check health and load initial data
  const loadAllData = async () => {
    try {
      await api.checkHealth();
      setBackendStatus('online');

      // Fetch in parallel
      const [datasetRes, modelsRes, vizRes] = await Promise.allSettled([
        api.getDatasetSummary(),
        api.getModels(),
        api.getVisualizations(),
      ]);

      if (datasetRes.status === 'fulfilled') setSummaryData(datasetRes.value);
      if (modelsRes.status === 'fulfilled') setModelsData(modelsRes.value);
      if (vizRes.status === 'fulfilled') setVizData(vizRes.value);
    } catch (err) {
      setBackendStatus('offline');
      showNotification('error', "Backend serverga ulanib bo'lmadi (http://localhost:8000). Server ishlayotganini tekshiring.");
    }
  };

  useEffect(() => {
    loadAllData();
  }, []);

  // Train models handler
  const handleTrainAll = async () => {
    setIsTraining(true);
    showNotification('info', "Modellar haqiqiy UCI ma'lumotlar to'plamida o'qitilmoqda, kuting...");
    try {
      const trainRes = await api.trainModels();
      setModelsData(trainRes);
      
      // Refresh visualizations with new importances
      const freshViz = await api.getVisualizations();
      setVizData(freshViz);

      showNotification('success', "Barcha 3 ta model muvaffaqiyatli o'qitildi va saqlandi!");
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || "Modellarni o'qitishda xatolik yuz berdi.";
      showNotification('error', msg);
    } finally {
      setIsTraining(false);
    }
  };

  // Download model handler
  const handleDownloadModel = async (modelKey) => {
    setDownloadingModel(modelKey);
    try {
      await api.downloadModelFile(modelKey);
      showNotification('success', `${modelKey.toUpperCase()} model fayli (.joblib) yuklab olindi.`);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || "Model faylini yuklab olishda xatolik yuz berdi.";
      showNotification('error', msg);
    } finally {
      setDownloadingModel(null);
    }
  };

  const trainedModelsCount = modelsData?.models ? Object.keys(modelsData.models).length : 0;
  const bestModelInfo = modelsData?.best_model_by_f1;

  return (
    <div className="app-container">
      {/* Sidebar */}
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Area */}
      <div className="main-wrapper">
        {/* Top Header */}
        <header className="top-header">
          <div className="header-title">
            <h2>Credit Default Prediction</h2>
            <p>Machine Learning Model Evaluation Dashboard</p>
          </div>

          <div className="header-actions">
            <div className={`status-indicator ${backendStatus === 'online' ? 'online' : 'offline'}`}>
              <span className="pulse-dot"></span>
              <span>
                {backendStatus === 'online'
                  ? 'Backend Online (Port 8000)'
                  : backendStatus === 'offline'
                  ? 'Backend Offline'
                  : 'Tekshirilmoqda...'}
              </span>
            </div>
          </div>
        </header>

        {/* Global Notification Banner */}
        {notification && (
          <div className={`notification-banner ${notification.type}`}>
            <span>{notification.message}</span>
            <button
              style={{ background: 'none', border: 'none', cursor: 'pointer', fontWeight: 700 }}
              onClick={() => setNotification(null)}
            >
              ✕
            </button>
          </div>
        )}

        {/* Dynamic Body Content */}
        <main className="content-body">
          {/* 1. DASHBOARD VIEW */}
          {activeTab === 'dashboard' && (
            <>
              {/* Summary Metrics Row */}
              <div className="metrics-grid">
                <MetricCard
                  title="Jami Mijozlar (Total Clients)"
                  value={summaryData?.total_rows?.toLocaleString() || '30,000'}
                  subtitle="UCI Dataset 350 to'liq yozuvlari"
                  icon={Users}
                  color="#2563eb"
                />
                <MetricCard
                  title="Belgilar Soni (Total Features)"
                  value={summaryData?.total_features || 23}
                  subtitle="Demografik va to'lov ko'rsatkichlari"
                  icon={Layers}
                  color="#10b981"
                />
                <MetricCard
                  title="Eng Yaxshi Model (Best by F1)"
                  value={bestModelInfo ? bestModelInfo.model_name : 'Aniqlanmagan'}
                  subtitle={
                    bestModelInfo
                      ? `F1-Score: ${(bestModelInfo.f1_score * 100).toFixed(2)}%`
                      : "Modellarni o'qiting"
                  }
                  icon={Award}
                  color="#f59e0b"
                />
                <MetricCard
                  title="Tayyor Modellar (Trained)"
                  value={`${trainedModelsCount} / 3`}
                  subtitle="KNN, Random Forest, Decision Tree"
                  icon={CheckCircle}
                  color="#8b5cf6"
                />
              </div>

              {/* Quick Action Banner */}
              <div className="card" style={{ background: 'linear-gradient(135deg, #1e293b, #0f172a)', color: '#ffffff' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
                  <div>
                    <h3 style={{ fontSize: '18px', fontWeight: 700, color: '#ffffff' }}>
                      Modellarni boshqarish va yangi mijoz xavfini baholash
                    </h3>
                    <p style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>
                      UCI Default of Credit Card Clients ma'lumotlarida 3 xil ML modelini o'qiting, metrikalarni solishtiring yoki mijoz kredit xavfini bashorat qiling.
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: '10px' }}>
                    <button
                      className="btn btn-primary"
                      onClick={() => setActiveTab('predictions')}
                    >
                      <span>Bashorat qilish</span>
                      <ArrowRight size={14} />
                    </button>
                    <button
                      className="btn btn-secondary"
                      onClick={handleTrainAll}
                      disabled={isTraining}
                      style={{ color: '#0f172a' }}
                    >
                      <span>{isTraining ? "O'qitilmoqda..." : "Train All"}</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Model Comparison Preview */}
              <ModelComparison
                modelsData={modelsData}
                onTrainAll={handleTrainAll}
                isTraining={isTraining}
                onDownloadModel={handleDownloadModel}
                downloadingModel={downloadingModel}
              />
            </>
          )}

          {/* 2. DATASET VIEW */}
          {activeTab === 'dataset' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              <div className="card">
                <div className="card-header">
                  <div className="card-title">
                    <h3>UCI Default of Credit Card Clients Dataset (ID: 350)</h3>
                    <p>
                      Tayvan banklaridagi 30,000 mijozning 2005-yil aprel-sentabr oylaridagi to'lov tarixlari va kredit qaytarmaslik holati.
                    </p>
                  </div>
                  <a
                    href="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
                    target="_blank"
                    rel="noreferrer"
                    className="btn btn-secondary btn-sm"
                  >
                    <span>UCI Manzilini ochish</span>
                    <ExternalLink size={14} />
                  </a>
                </div>

                <div className="metrics-grid" style={{ marginBottom: '20px' }}>
                  <div className="metric-card">
                    <div className="metric-info">
                      <span>Jami Qatorlar</span>
                      <h3>{summaryData?.total_rows?.toLocaleString() || 30000}</h3>
                      <p>Mijoz hisoblari</p>
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-info">
                      <span>Non-Default (0)</span>
                      <h3 style={{ color: '#2563eb' }}>
                        {summaryData?.target_distribution?.non_default_0?.toLocaleString() || 23364}
                      </h3>
                      <p>{summaryData?.target_distribution?.non_default_pct || 77.88}% — To'lovni bajargan</p>
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-info">
                      <span>Default (1)</span>
                      <h3 style={{ color: '#dc2626' }}>
                        {summaryData?.target_distribution?.default_1?.toLocaleString() || 6636}
                      </h3>
                      <p>{summaryData?.target_distribution?.default_pct || 22.12}% — Kechiktirgan (Xavf)</p>
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-info">
                      <span>Yo'qotilgan Qiymatlar (Missing)</span>
                      <h3 style={{ color: '#10b981' }}>
                        {summaryData?.missing_values_total || 0}
                      </h3>
                      <p>To'liq va toza ma'lumotlar</p>
                    </div>
                  </div>
                </div>

                <h4 style={{ fontSize: '16px', fontWeight: 700, margin: '20px 0 12px' }}>
                  Xususiyatlar Lug'ati (Feature Dictionary)
                </h4>
                <div className="table-responsive">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Standart Nomi</th>
                        <th>O'zbekcha Nomi</th>
                        <th>Turi</th>
                        <th>Tavsifi</th>
                      </tr>
                    </thead>
                    <tbody>
                      {summaryData?.features?.map((feat) => (
                        <tr key={feat.id}>
                          <td><strong>{feat.id}</strong></td>
                          <td><code>{feat.name}</code></td>
                          <td>{feat.label_uz}</td>
                          <td>
                            <span style={{
                              padding: '2px 8px',
                              borderRadius: '4px',
                              fontSize: '11px',
                              fontWeight: 600,
                              backgroundColor: feat.type === 'categorical' ? '#fef3c7' : '#eff6ff',
                              color: feat.type === 'categorical' ? '#92400e' : '#1e40af'
                            }}>
                              {feat.type}
                            </span>
                          </td>
                          <td style={{ fontSize: '12px', color: '#475569' }}>{feat.description}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* 3. MODEL COMPARISON VIEW */}
          {activeTab === 'models' && (
            <ModelComparison
              modelsData={modelsData}
              onTrainAll={handleTrainAll}
              isTraining={isTraining}
              onDownloadModel={handleDownloadModel}
              downloadingModel={downloadingModel}
            />
          )}

          {/* 4. VISUALIZATIONS VIEW */}
          {activeTab === 'visualizations' && (
            <Charts modelsData={modelsData} vizData={vizData} />
          )}

          {/* 5. PREDICTIONS VIEW */}
          {activeTab === 'predictions' && (
            <PredictionForm featuresMeta={summaryData?.features} />
          )}
        </main>
      </div>
    </div>
  );
}
