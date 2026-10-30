import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { AlertCircle, HelpCircle } from 'lucide-react';

const COLORS = ['#2563eb', '#dc2626', '#10b981', '#f59e0b', '#8b5cf6'];

export default function Charts({ modelsData, vizData }) {
  const models = modelsData?.models || {};

  // 1. Prepare Model Metrics Comparison Bar Chart Data
  const metricsComparisonData = [
    {
      metric: 'Accuracy (Aniqlik)',
      KNN: models.knn ? Number((models.knn.metrics.accuracy * 100).toFixed(1)) : 0,
      'Random Forest': models.random_forest ? Number((models.random_forest.metrics.accuracy * 100).toFixed(1)) : 0,
      'Decision Tree': models.decision_tree ? Number((models.decision_tree.metrics.accuracy * 100).toFixed(1)) : 0,
    },
    {
      metric: 'Precision (Default)',
      KNN: models.knn ? Number((models.knn.metrics.precision * 100).toFixed(1)) : 0,
      'Random Forest': models.random_forest ? Number((models.random_forest.metrics.precision * 100).toFixed(1)) : 0,
      'Decision Tree': models.decision_tree ? Number((models.decision_tree.metrics.precision * 100).toFixed(1)) : 0,
    },
    {
      metric: 'Recall (Qamrov)',
      KNN: models.knn ? Number((models.knn.metrics.recall * 100).toFixed(1)) : 0,
      'Random Forest': models.random_forest ? Number((models.random_forest.metrics.recall * 100).toFixed(1)) : 0,
      'Decision Tree': models.decision_tree ? Number((models.decision_tree.metrics.recall * 100).toFixed(1)) : 0,
    },
    {
      metric: 'F1-Score',
      KNN: models.knn ? Number((models.knn.metrics.f1 * 100).toFixed(1)) : 0,
      'Random Forest': models.random_forest ? Number((models.random_forest.metrics.f1 * 100).toFixed(1)) : 0,
      'Decision Tree': models.decision_tree ? Number((models.decision_tree.metrics.f1 * 100).toFixed(1)) : 0,
    },
  ];

  // 2. Target distribution data
  const targetDist = vizData?.target_distribution || [
    { name: "Non-Default (0: To'lagan)", count: 23364, percentage: 77.9, fill: '#2563eb' },
    { name: "Default (1: Qaytarmagan)", count: 6636, percentage: 22.1, fill: '#dc2626' }
  ];

  // 3. Top correlations data
  const correlations = vizData?.top_correlations || [];

  // 4. Feature importances
  const rfImportances = vizData?.feature_importances?.random_forest || [];
  const dtImportances = vizData?.feature_importances?.decision_tree || [];
  const knnInfo = vizData?.feature_importances?.knn;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div className="card">
        <div className="card-title">
          <h3>Vizual Tahlil va Metrikalar Grafikasi</h3>
          <p>
            Modellarning samaradorligi, kredit qaytarmaslik taqsimoti va qaror qabul qilishda eng muhim belgilari.
          </p>
        </div>
      </div>

      <div className="charts-grid">
        {/* Chart 1: Model Metrics Comparison */}
        <div className="chart-card">
          <div className="chart-header">
            <h4>1. Modellar Samaradorligi Taqqoslashi (%)</h4>
            <p>Accuracy, Precision, Recall va F1-Score ko'rsatkichlarining uchta model bo'yicha solishtirilishi.</p>
          </div>
          <div style={{ width: '100%', height: 320 }}>
            <ResponsiveContainer>
              <BarChart data={metricsComparisonData} margin={{ top: 20, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="metric" tick={{ fontSize: 12, fill: '#64748b' }} />
                <YAxis unit="%" domain={[0, 100]} tick={{ fontSize: 12, fill: '#64748b' }} />
                <Tooltip formatter={(value) => [`${value}%`]} />
                <Legend />
                <Bar dataKey="KNN" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Random Forest" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Decision Tree" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: Target Class Distribution */}
        <div className="chart-card">
          <div className="chart-header">
            <h4>2. Maqsadli Sinf Taqsimoti (Target Distribution)</h4>
            <p>Mijozlarning kreditni o'z vaqtida to'laganligi (0) va to'lamaganligi (1) nisbati.</p>
          </div>
          <div style={{ width: '100%', height: 320, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={targetDist}
                  dataKey="count"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  innerRadius={50}
                  label={({ name, percentage }) => `${percentage}%`}
                >
                  {targetDist.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill || COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value, name, item) => [`${value.toLocaleString()} ta (${item.payload.percentage}%)`, name]} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 3: Top Feature Correlations */}
        <div className="chart-card">
          <div className="chart-header">
            <h4>3. Kredit Defaulti bilan Eng Kuchli Korrelyatsiya</h4>
            <p>Qaysi ko'rsatkichlar kredit qaytarmaslik xavfi bilan eng yuqori chiziqli bog'liqlikka ega.</p>
          </div>
          <div style={{ width: '100%', height: 340 }}>
            <ResponsiveContainer>
              <BarChart
                layout="vertical"
                data={correlations.slice(0, 7)}
                margin={{ top: 10, right: 30, left: 60, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis type="number" domain={[-0.4, 0.4]} tick={{ fontSize: 12, fill: '#64748b' }} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 12, fill: '#334155' }} />
                <Tooltip formatter={(val, name, item) => [val, `${item.payload.label_uz || item.payload.name}`]} />
                <Bar
                  dataKey="correlation"
                  fill="#2563eb"
                  name="Korrelyatsiya koeffitsienti"
                  radius={[0, 4, 4, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 4: Random Forest Feature Importance */}
        <div className="chart-card">
          <div className="chart-header">
            <h4>4. Random Forest: Muhim Belgilar (Feature Importance)</h4>
            <p>100 ta daraxt ansambli qarorida eng ko'p ta'sir ko'rsatgan 8 ta xususiyat.</p>
          </div>
          <div style={{ width: '100%', height: 340 }}>
            {rfImportances.length > 0 ? (
              <ResponsiveContainer>
                <BarChart
                  layout="vertical"
                  data={rfImportances.slice(0, 8)}
                  margin={{ top: 10, right: 30, left: 60, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" unit="%" tick={{ fontSize: 12, fill: '#64748b' }} />
                  <YAxis dataKey="feature_name" type="category" tick={{ fontSize: 12, fill: '#334155' }} />
                  <Tooltip formatter={(val, name, item) => [`${val}%`, `${item.payload.label_uz || item.payload.label}`]} />
                  <Bar
                    dataKey="importance_pct"
                    fill="#10b981"
                    name="Muhimlik darajasi (%)"
                    radius={[0, 4, 4, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#64748b' }}>
                <p>Muhimlik grafigini ko'rish uchun avval modellarni o'qiting.</p>
              </div>
            )}
          </div>
        </div>

        {/* Chart 5: Decision Tree Feature Importance */}
        <div className="chart-card">
          <div className="chart-header">
            <h4>5. Decision Tree: Muhim Belgilar (Feature Importance)</h4>
            <p>Chuqurligi 5 bo'lgan qaror daraxti shoxlanishida asosiy rol o'ynagan belgilar.</p>
          </div>
          <div style={{ width: '100%', height: 340 }}>
            {dtImportances.length > 0 ? (
              <ResponsiveContainer>
                <BarChart
                  layout="vertical"
                  data={dtImportances.slice(0, 8)}
                  margin={{ top: 10, right: 30, left: 60, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" unit="%" tick={{ fontSize: 12, fill: '#64748b' }} />
                  <YAxis dataKey="feature_name" type="category" tick={{ fontSize: 12, fill: '#334155' }} />
                  <Tooltip formatter={(val, name, item) => [`${val}%`, `${item.payload.label_uz || item.payload.label}`]} />
                  <Bar
                    dataKey="importance_pct"
                    fill="#f59e0b"
                    name="Muhimlik darajasi (%)"
                    radius={[0, 4, 4, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#64748b' }}>
                <p>Daraxt modelini o'qitgandan so'ng grafik paydo bo'ladi.</p>
              </div>
            )}
          </div>
        </div>

        {/* Note / Explainability for KNN */}
        <div className="chart-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-start' }}>
            <div style={{ background: '#eff6ff', color: '#2563eb', padding: '12px', borderRadius: '10px' }}>
              <HelpCircle size={24} />
            </div>
            <div>
              <h4 style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a' }}>
                Nega KNN modeli uchun Feature Importance mavjud emas?
              </h4>
              <p style={{ fontSize: '13px', color: '#475569', marginTop: '6px', lineHeight: 1.6 }}>
                <strong>K-Nearest Neighbors (KNN)</strong> – bu «lazy learning» (xotiraga asoslangan) algoritmi. U ma'lumotlar ustida daraxtlar singari ajratuvchi qoidalar yoki og'irlik koeffitsientlari (weights) o'rnatmaydi.
              </p>
              <p style={{ fontSize: '13px', color: '#475569', marginTop: '6px', lineHeight: 1.6 }}>
                Aksincha, har bir yangi mijoz kiritilganda barcha 23 ta ko'rsatkich bo'yicha ko'p o'lchovli fazoda Evklid masofasini (Euclidean distance) hisoblab, eng yaqin 5 ta qo'shniga qarab baho beradi. Shu sababli standart KNN algoritmlarida qat'iy «feature importance» ko'rsatkichi mavjud emas.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
