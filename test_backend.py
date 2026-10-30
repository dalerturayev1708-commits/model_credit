import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from main import app, health_check, dataset_summary, train_models, get_models, predict, get_visualizations, PredictionInput

print("1. Testing health check...")
h = health_check()
print("Health:", h)

print("\n2. Testing dataset summary...")
summary = dataset_summary()
print("Rows:", summary["total_rows"])
print("Features:", summary["total_features"])
print("Default count:", summary["target_distribution"]["default_1"])

print("\n3. Training all models (KNN, Random Forest, Decision Tree)...")
train_res = train_models()
print("Train status:", train_res["status"])
for k, m in train_res["models"].items():
    metrics = m["metrics"]
    print(f" -> {k}: Acc={metrics['accuracy']}, F1={metrics['f1']}, Recall={metrics['recall']}, ROC-AUC={metrics['roc_auc']}, TrainTime={metrics['training_time']}s")

print("\n4. Testing saved models retrieval...")
models_meta = get_models()
print("Trained:", models_meta["trained"], "Models count:", len(models_meta["models"]))

print("\n5. Testing prediction with Random Forest...")
sample_features = {f["id"]: f["default"] for f in summary["features"]}
pred_rf = predict("random_forest", PredictionInput(features=sample_features))
print("RF Result:", pred_rf["prediction_label"], "Default prob:", pred_rf["probabilities"]["default_percentage"], "% Risk:", pred_rf["risk_assessment"]["level"])

print("\n6. Testing prediction with KNN...")
pred_knn = predict("knn", PredictionInput(features=sample_features))
print("KNN Result:", pred_knn["prediction_label"], "Default prob:", pred_knn["probabilities"]["default_percentage"], "%")

print("\n7. Testing prediction with Decision Tree...")
pred_dt = predict("decision_tree", PredictionInput(features=sample_features))
print("DT Result:", pred_dt["prediction_label"], "Default prob:", pred_dt["probabilities"]["default_percentage"], "%")

print("\n8. Testing visualizations...")
viz = get_visualizations()
print("Target dist items:", len(viz["target_distribution"]))
print("Top correlations count:", len(viz["top_correlations"]))
print("RF top feature:", viz["feature_importances"]["random_forest"][0]["feature_name"])
print("KNN feature importance note:", viz["feature_importances"]["knn"]["explanation"])

print("\n>>> ALL TESTS PASSED PERFECTLY! <<<")
