import axios from 'axios';

// Base API configuration
const API_BASE_URL = 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60s timeout for model training
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Check backend connectivity
  checkHealth: async () => {
    const res = await client.get('/api/health');
    return res.data;
  },

  // Fetch dataset metadata and summary statistics
  getDatasetSummary: async () => {
    const res = await client.get('/api/dataset/summary');
    return res.data;
  },

  // Train all three models (KNN, Random Forest, Decision Tree)
  trainModels: async () => {
    const res = await client.post('/api/train');
    return res.data;
  },

  // Get saved models metadata and metrics
  getModels: async () => {
    const res = await client.get('/api/models');
    return res.data;
  },

  // Download a trained model .joblib file
  downloadModelFile: async (modelKey) => {
    const res = await client.get(`/api/models/${modelKey}/download`, {
      responseType: 'blob',
    });
    // Create a temporary link element to trigger file download in browser
    const url = window.URL.createObjectURL(new Blob([res.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `${modelKey}_model.joblib`);
    document.body.appendChild(link);
    link.click();
    link.parentNode.removeChild(link);
    window.URL.revokeObjectURL(url);
  },

  // Make a single prediction
  predict: async (modelKey, features) => {
    const res = await client.post(`/api/predict/${modelKey}`, { features });
    return res.data;
  },

  // Get data for charts and visualizations
  getVisualizations: async () => {
    const res = await client.get('/api/visualizations');
    return res.data;
  },
};

export default api;
