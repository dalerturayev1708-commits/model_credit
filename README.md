# Credit Card Default Prediction Dashboard

A modern, clean, and fully functional Full Stack Machine Learning web application built using the exact **UCI Default of Credit Card Clients dataset (ID: 350)**.

This application allows users to train and evaluate 3 machine learning algorithms (**KNN**, **Random Forest**, and **Decision Tree**), compare their performance metrics, visualize confusion matrices and feature importances, download trained `.joblib` model pipelines, and run real-time credit default risk predictions through a responsive React dashboard.

---

## 1. Project Overview & Architecture

```
credit-default-dashboard/
│
├── backend/
│   ├── main.py                     # FastAPI server, endpoints, CORS, validation
│   ├── requirements.txt            # Python backend dependencies
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── data.py                 # UCI repo loader, data caching, preprocessing
│   │   ├── train.py                # Model training, Pipeline configs, joblib serialization
│   │   └── evaluate.py             # Accuracy, Precision, Recall, F1, ROC-AUC, CM
│   └── models/                     # Saved .joblib model files & metadata
│       ├── knn_model.joblib
│       ├── random_forest_model.joblib
│       ├── decision_tree_model.joblib
│       └── metadata.json
│
├── frontend/
│   ├── package.json                # React, Vite, Axios, Recharts, Lucide React
│   ├── vite.config.js              # Vite server & proxy configuration
│   ├── index.html                  # HTML template with Inter typography
│   └── src/
│       ├── main.jsx                # React root mount
│       ├── App.jsx                 # Main application state and page switcher
│       ├── api.js                  # Axios client for FastAPI communication
│       ├── styles.css              # Modern Vanilla CSS design system
│       └── components/
│           ├── Sidebar.jsx         # Navigation sidebar (Dashboard, Dataset, etc.)
│           ├── MetricCard.jsx      # Summary KPI cards
│           ├── ModelComparison.jsx # Model cards, metrics, and download buttons
│           ├── ConfusionMatrix.jsx # Interactive 2x2 confusion matrix heatmaps
│           ├── Charts.jsx          # Recharts visualizations (Comparison, Importances)
│           └── PredictionForm.jsx  # Interactive prediction form with presets
│
└── README.md                       # Documentation and usage guide
```

---

## 2. Dataset Information

The project uses the real **UCI Default of Credit Card Clients Dataset (ID: 350)**:
- **Source**: [https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Target Variable ($Y$)**: Default payment next month:
  - `0`: **Non-Default** (Customer paid bill on time) — $77.9\%$ ($23,364$ clients).
  - `1`: **Default** (Customer delayed or failed to pay) — $22.1\%$ ($6,636$ clients).
- **Features ($X_1 \dots X_{23}$)**:
  - Demographics: Credit limit (`LIMIT_BAL`), Gender (`SEX`), Education (`EDUCATION`), Marital status (`MARRIAGE`), Age (`AGE`).
  - Repayment History: 6 months of past repayment statuses (`PAY_0` to `PAY_6`).
  - Billing Statements: 6 months of bill amounts (`BILL_AMT1` to `BILL_AMT6`).
  - Previous Payments: 6 months of payment amounts (`PAY_AMT1` to `PAY_AMT6`).
- **Data Splitting**:
  - `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
  - Exactly $24,000$ training samples and $6,000$ held-out test samples.
  - Stratification ensures class balance is strictly preserved across train and test partitions.

---

## 3. Implemented Machine Learning Models

### Model 1: K-Nearest Neighbors (KNN)
- **Pipeline**: Scikit-Learn `Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))])`.
- **StandardScaler**: Scaled inside the pipeline to prevent data leakage and ensure fair distance calculation between financial balances and categorical codes.
- **Distance Weighting**: Closer customer payment histories contribute higher voting power.

### Model 2: Random Forest
- **Classifier**: `RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')`.
- **Balanced Class Weighting**: Automatically adjusts weights inversely proportional to class frequencies to combat the $78/22$ default class imbalance, ensuring high recall on defaulting customers.

### Model 3: Decision Tree
- **Classifier**: `DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')`.
- **Interpretability**: Constrained to depth 5 to avoid overfitting while producing clean, interpretable branching rules.

---

## 4. Evaluation Metrics & Benchmark Results

On the held-out test set ($6,000$ clients), all models were evaluated on the same split:

| Model | Accuracy | Precision (Default) | Recall (Default) | F1-Score (Default) | ROC-AUC | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN (k=5)** | $78.97\%$ | $53.51\%$ | $36.10\%$ | $43.15\%$ | $0.6999$ | $0.02\text{ s}$ |
| **Random Forest** | $\mathbf{80.02\%}$ | $\mathbf{55.48\%}$ | $48.83\%$ | $\mathbf{51.94\%}$ | $\mathbf{0.7589}$ | $0.82\text{ s}$ |
| **Decision Tree (d=5)** | $77.08\%$ | $48.42\%$ | $\mathbf{55.16\%}$ | $51.57\%$ | $0.7571$ | $0.12\text{ s}$ |

> [!NOTE]
> Metrics are calculated primarily for the **positive default class ($Y=1$)** because correctly flagging defaulting clients is the critical business objective of credit risk management.

---

## 5. Technology Stack

### Backend
- **Python 3.10+**
- **FastAPI**: Asynchronous, production-grade REST API framework.
- **Uvicorn**: Lightning-fast ASGI server.
- **Scikit-Learn**: Model pipelines, estimators, and evaluation metrics.
- **Pandas & NumPy**: Data processing and statistical analysis.
- **Joblib**: Model serialization and disk persistence.
- **ucimlrepo**: Direct integration with the UCI Machine Learning repository.

### Frontend
- **React 18 / 19**
- **Vite**: Ultra-fast build tool and dev server.
- **JavaScript (ES Modules)**
- **Vanilla CSS**: Clean design system (no Tailwind, no CSS bloat).
- **Axios**: HTTP API client.
- **Recharts**: Responsive SVG charts (Bar, Pie, Heatmaps).
- **Lucide React**: Clean icons.

---

## 6. Installation & Startup Instructions

### Recommended Workspace
Set your workspace to:
```
C:\Users\User\.gemini\antigravity\scratch\credit-default-dashboard
```

### Step 1: Start the Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

- API Base URL: `http://localhost:8000`
- Interactive Swagger UI: `http://localhost:8000/docs`

### Step 2: Start the Frontend (React + Vite)

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

- Web Dashboard URL: `http://localhost:5173`

---

## 7. Backend API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Backend and dataset health status |
| `GET` | `/api/dataset/summary` | Dataset metadata, target distribution, feature stats |
| `POST` | `/api/train` | Trains all 3 models, evaluates on test split, saves `.joblib` files |
| `GET` | `/api/models` | Returns trained models and their metrics |
| `GET` | `/api/models/{model_name}/download` | Downloads `knn`, `random_forest`, or `decision_tree` `.joblib` file |
| `POST` | `/api/predict/{model_name}` | Predicts default risk for 23 input features with probabilities |
| `GET` | `/api/visualizations` | Aggregated data for Recharts (Correlations, Importances, Target dist) |

---

## 8. How to Use the Application

1. **Dashboard Overview**: Check backend connectivity and view overall dataset statistics.
2. **Train Models**: Click **"Train All Models"** to train KNN, Random Forest, and Decision Tree on the real UCI dataset.
3. **Compare Performance**: Compare accuracy, precision, recall, F1, ROC-AUC, and visual confusion matrix heatmaps.
4. **Download Models**: Click **"Download Model (.joblib)"** on any card to save the fitted model file to your local computer.
5. **Interactive Prediction**:
   - Go to the **Predictions** tab.
   - Choose one of the preset profiles (*A'lo mijoz*, *Kechiktiruvchi mijoz*, or *O'rtacha xavf*) or enter custom values.
   - Select the model and click **"Predict"**.
   - Review the predicted class, default probability percentage bar, and risk assessment breakdown.

---

## 9. Educational Disclaimer
This dashboard is developed for educational and machine learning demonstration purposes. It does not replace formal underwriting audits or statutory credit bureau regulations.
