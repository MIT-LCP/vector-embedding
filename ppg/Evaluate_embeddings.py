import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
embeddings_dir = "/content/drive/My Drive/mimic3_ppg_data/papagei_embeddings"
mapping_file = "/content/drive/My Drive/mimic3_ppg_data/waveform_clinical_mapping.csv"

# Load embedding info
embedding_info_file = os.path.join(embeddings_dir, "embedding_info.csv")
if os.path.exists(embedding_info_file):
    embedding_info = pd.read_csv(embedding_info_file)
    print(f"Loaded embedding info for {len(embedding_info)} patients")
    print("\nEmbedding info summary:")
    print(embedding_info.head())
else:
    print("Embedding info file not found.")
    embedding_info = pd.DataFrame(columns=['patient_id', 'subject_id', 'num_segments', 'embedding_dim'])

# Get list of embedding files
embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('_embeddings.npy')]
print(f"\nFound {len(embedding_files)} embedding files")

# Load mapping
mapping_df = pd.read_csv(mapping_file)
print(f"Loaded mapping with {len(mapping_df)} entries")

# Basic statistics on segments per patient
if 'num_segments' in embedding_info.columns:
    print("\nSegments per patient statistics:")
    print(embedding_info['num_segments'].describe())

    # Plot distribution of segments per patient
    plt.figure(figsize=(10, 6))
    sns.histplot(embedding_info['num_segments'], bins=30, kde=True)
    plt.title('Distribution of Segments per Patient')
    plt.xlabel('Number of Segments')
    plt.ylabel('Frequency')
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True, alpha=0.3)
    plt.show()

# Examine a few random embedding files
np.random.seed(42)  # For reproducibility
sample_indices = np.random.choice(len(embedding_files), size=min(3, len(embedding_files)), replace=False)

for idx in sample_indices:
    file = embedding_files[idx]
    file_path = os.path.join(embeddings_dir, file)

    # Load embeddings
    embeddings = np.load(file_path)

    # Basic statistics
    print(f"\n=== Sample embeddings from {file} ===")
    print(f"Shape: {embeddings.shape}")

    # Check for NaN values
    nan_count = np.isnan(embeddings).sum()
    if nan_count > 0:
        print(f"NaN values: {nan_count} ({nan_count/embeddings.size:.2%})")
        print(f"Mean (excluding NaNs): {np.nanmean(embeddings):.4f}")
        print(f"Std (excluding NaNs): {np.nanstd(embeddings):.4f}")
    else:
        print("No NaN values found")
        print(f"Mean: {np.mean(embeddings):.4f}")
        print(f"Std: {np.std(embeddings):.4f}")

    print(f"Min: {np.nanmin(embeddings):.4f}")
    print(f"Max: {np.nanmax(embeddings):.4f}")

    # Replace NaNs with 0 for visualization
    embeddings_clean = np.nan_to_num(embeddings, nan=0.0)

    # Plot heatmap of first few embeddings (first 20 dimensions)
    plt.figure(figsize=(12, 6))
    sample_rows = min(5, embeddings.shape[0])
    sns.heatmap(embeddings_clean[:sample_rows, :20], cmap='viridis',
                xticklabels=5, yticklabels=True)
    plt.title(f'First {sample_rows} embeddings (first 20 dimensions) from {file}')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Segment Index')
    plt.show()

    # Plot embedding vector profiles for a few segments
    plt.figure(figsize=(12, 6))
    for i in range(min(5, embeddings.shape[0])):
        plt.plot(embeddings_clean[i, :50], label=f'Segment {i}')
    plt.title(f'First 50 dimensions of embeddings from {file}')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # If there are many segments, plot embedding changes over time
    if embeddings.shape[0] > 10:
        # Take mean across dimensions for each segment
        mean_values = np.nanmean(embeddings, axis=1)

        plt.figure(figsize=(12, 6))
        plt.plot(mean_values)
        plt.title(f'Mean Embedding Value Over Time ({file})')
        plt.xlabel('Segment Index (chronological)')
        plt.ylabel('Mean Embedding Value')
        plt.grid(True, alpha=0.3)
        plt.show()

# Simple analysis of embedding statistics across patients
print("\n=== Overall Embedding Statistics ===")

# Collect metrics across all files
all_metrics = []

for file in embedding_files:
    file_path = os.path.join(embeddings_dir, file)
    patient_id = file.split('_')[0]

    try:
        # Load embeddings
        embeddings = np.load(file_path)

        # Calculate metrics
        nan_count = np.isnan(embeddings).sum()
        mean_val = np.nanmean(embeddings)
        std_val = np.nanstd(embeddings)
        min_val = np.nanmin(embeddings)
        max_val = np.nanmax(embeddings)

        # Get subject ID from mapping
        subject_id = mapping_df[mapping_df['waveform_id'] == patient_id]['subject_id'].iloc[0]

        # Add to metrics list
        all_metrics.append({
            'patient_id': patient_id,
            'subject_id': subject_id,
            'num_segments': embeddings.shape[0],
            'nan_percentage': (nan_count / embeddings.size) * 100 if embeddings.size > 0 else 0,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val
        })
    except Exception as e:
        print(f"Error analyzing {file}: {str(e)}")

# Convert to DataFrame
metrics_df = pd.DataFrame(all_metrics)

# Summary statistics
print("Summary of embedding statistics across all patients:")
print(metrics_df[['num_segments', 'nan_percentage', 'mean', 'std', 'min', 'max']].describe())

# Plot histogram of mean values
plt.figure(figsize=(10, 6))
sns.histplot(metrics_df['mean'], bins=20, kde=True)
plt.title('Distribution of Mean Embedding Values Across Patients')
plt.xlabel('Mean Embedding Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Highlight patients with unusual embedding statistics
print("\n=== Patients with Notable Embedding Characteristics ===")

# Patients with highest/lowest mean values
print("\nPatients with highest mean embedding values:")
print(metrics_df.sort_values('mean', ascending=False).head(5)[['patient_id', 'subject_id', 'mean']])

print("\nPatients with lowest mean embedding values:")
print(metrics_df.sort_values('mean').head(5)[['patient_id', 'subject_id', 'mean']])

# Patients with highest NaN percentages
if metrics_df['nan_percentage'].max() > 0:
    print("\nPatients with highest NaN percentages:")
    print(metrics_df.sort_values('nan_percentage', ascending=False).head(5)[['patient_id', 'subject_id', 'nan_percentage']])

print("\n=== Next Steps with These Embeddings ===")
print("1. Classification: Train a classifier to predict clinical outcomes from embeddings")
print("2. Clustering: Group patients with similar PPG patterns")
print("3. Feature Importance: Identify which embedding dimensions correlate with outcomes")
print("4. Time Series Analysis: Track changes in embeddings over time for long recordings")
print("5. Anomaly Detection: Identify unusual PPG patterns compared to the population")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                            auc, precision_recall_curve, mean_squared_error, r2_score)
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
base_dir = "/content/drive/My Drive/mimic3_ppg_data"
embeddings_dir = os.path.join(base_dir, "papagei_embeddings")
final_dataset_path = os.path.join(base_dir, "final_modeling_dataset.csv")
results_dir = os.path.join(base_dir, "model_results")
os.makedirs(results_dir, exist_ok=True)

# Load the final dataset that links patients to outcomes
print("Loading final dataset...")
final_df = pd.read_csv(final_dataset_path)
print(f"Loaded data for {len(final_df)} patients")

# Function to load embeddings for a patient
def load_patient_embedding(patient_id):
    """Load the embedding file for a given patient ID and return the mean embedding"""
    embedding_path = os.path.join(embeddings_dir, f"{patient_id}_embeddings.npy")
    if os.path.exists(embedding_path):
        embeddings = np.load(embedding_path)
        # Use mean across all segments
        if np.isnan(embeddings).any():
            # Handle NaN values if present
            mean_embedding = np.nanmean(embeddings, axis=0)
            # Replace any remaining NaNs with 0
            mean_embedding = np.nan_to_num(mean_embedding, nan=0.0)
        else:
            mean_embedding = np.mean(embeddings, axis=0)
        return mean_embedding
    else:
        print(f"Warning: No embedding file found for {patient_id}")
        return None

# Load embeddings for all patients
print("Loading embeddings for all patients...")
X_data = []
patient_ids = []

for idx, row in final_df.iterrows():
    patient_id = row['patient_id']
    embedding = load_patient_embedding(patient_id)
    if embedding is not None:
        X_data.append(embedding)
        patient_ids.append(patient_id)

# Convert to numpy array
X = np.array(X_data)
print(f"Loaded embeddings with shape: {X.shape}")

# Create a dataframe with only the patients for which we have embeddings
modeling_df = final_df[final_df['patient_id'].isin(patient_ids)].copy()
print(f"Final modeling dataset includes {len(modeling_df)} patients")

# Define outcome variables
y_mortality = modeling_df['mortality'].values
y_los = modeling_df['icu_los'].values

# Save patient details for later reference
patient_details = modeling_df[['patient_id', 'subject_id', 'mortality', 'icu_los', 'careunit']].copy()
patient_details.to_csv(os.path.join(results_dir, "patient_details.csv"), index=False)

##################################################
# TASK 1: CLASSIFICATION - MORTALITY PREDICTION
##################################################
print("\n=== TASK 1: MORTALITY PREDICTION ===")

# Given the small dataset, we'll use cross-validation instead of a single train/test split
# Also, we'll try both Random Forest and Logistic Regression

# Setup cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Classifiers to try
classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

# Results storage
mortality_results = {}

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")

    # Create pipeline with preprocessing
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])

    # Cross-validation scores
    cv_scores = cross_val_score(pipe, X, y_mortality, cv=cv, scoring='roc_auc')
    print(f"Cross-validation ROC AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit on all data for feature importance (especially for Random Forest)
    pipe.fit(X, y_mortality)

    # If it's RandomForest, extract feature importance
    if name == 'RandomForest':
        feature_importances = pipe.named_steps['classifier'].feature_importances_
        importance_df = pd.DataFrame({
            'feature_idx': range(len(feature_importances)),
            'importance': feature_importances
        }).sort_values('importance', ascending=False)

        # Save top features
        importance_df.head(50).to_csv(os.path.join(results_dir, "mortality_feature_importance.csv"), index=False)

        # Plot top 20 important features
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature_idx', data=importance_df.head(20))
        plt.title('Top 20 Important Features for Mortality Prediction')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "mortality_feature_importance.png"))
        plt.close()

    # Store results
    mortality_results[name] = {
        'cv_scores': cv_scores,
        'mean_auc': cv_scores.mean(),
        'model': pipe
    }

# Select best model for final evaluation
best_model_name = max(mortality_results, key=lambda x: mortality_results[x]['mean_auc'])
best_model = mortality_results[best_model_name]['model']
print(f"\nBest model for mortality prediction: {best_model_name} (AUC: {mortality_results[best_model_name]['mean_auc']:.3f})")

# Predict probabilities for all patients (for visualization)
y_prob = best_model.predict_proba(X)[:, 1]

# Save predictions with patient details
mortality_preds = patient_details.copy()
mortality_preds['predicted_probability'] = y_prob
mortality_preds.to_csv(os.path.join(results_dir, "mortality_predictions.csv"), index=False)

# Plot prediction distribution by actual mortality
plt.figure(figsize=(10, 6))
sns.histplot(
    data=mortality_preds, x='predicted_probability', hue='mortality',
    bins=20, element='step', stat='density', common_norm=False
)
plt.title('Distribution of Mortality Predictions by Actual Outcome')
plt.xlabel('Predicted Probability of Mortality')
plt.ylabel('Density')
plt.savefig(os.path.join(results_dir, "mortality_prediction_distribution.png"))
plt.close()

##################################################
# TASK 2: REGRESSION - ICU LENGTH OF STAY PREDICTION
##################################################
print("\n=== TASK 2: ICU LENGTH OF STAY PREDICTION ===")

# Models to try
regressors = {
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Results storage
los_results = {}

for name, reg in regressors.items():
    print(f"\nTraining {name}...")

    # Create pipeline with preprocessing
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', reg)
    ])

    # Cross-validation
    cv_scores = cross_val_score(pipe, X, y_los, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    print(f"Cross-validation RMSE: {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f} days")

    # Fit on all data
    pipe.fit(X, y_los)

    # If RandomForest, get feature importance
    if name == 'RandomForest':
        feature_importances = pipe.named_steps['regressor'].feature_importances_
        importance_df = pd.DataFrame({
            'feature_idx': range(len(feature_importances)),
            'importance': feature_importances
        }).sort_values('importance', ascending=False)

        # Save top features
        importance_df.head(50).to_csv(os.path.join(results_dir, "los_feature_importance.csv"), index=False)

        # Plot top 20 important features
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature_idx', data=importance_df.head(20))
        plt.title('Top 20 Important Features for ICU Length of Stay Prediction')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "los_feature_importance.png"))
        plt.close()

    # Store results
    los_results[name] = {
        'cv_scores': rmse_scores,
        'mean_rmse': rmse_scores.mean(),
        'model': pipe
    }

# Select best model for final evaluation
best_model_name = min(los_results, key=lambda x: los_results[x]['mean_rmse'])
best_model = los_results[best_model_name]['model']
print(f"\nBest model for LOS prediction: {best_model_name} (RMSE: {los_results[best_model_name]['mean_rmse']:.3f} days)")

# Predict for all patients
y_pred_los = best_model.predict(X)

# Calculate R²
r2 = r2_score(y_los, y_pred_los)
print(f"R² on full dataset: {r2:.3f}")

# Save predictions with patient details
los_preds = patient_details.copy()
los_preds['predicted_los'] = y_pred_los
los_preds.to_csv(os.path.join(results_dir, "los_predictions.csv"), index=False)

# Plot actual vs predicted
plt.figure(figsize=(10, 8))
plt.scatter(y_los, y_pred_los, alpha=0.7)
plt.plot([0, max(y_los)], [0, max(y_los)], 'r--')
plt.xlabel('Actual ICU Length of Stay (days)')
plt.ylabel('Predicted ICU Length of Stay (days)')
plt.title('Actual vs Predicted ICU Length of Stay')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(results_dir, "los_actual_vs_predicted.png"))
plt.close()

##################################################
# TASK 3: CLUSTERING - PATIENT GROUPS BASED ON PPG PATTERNS
##################################################
print("\n=== TASK 3: CLUSTERING PATIENTS BY PPG PATTERNS ===")

# First, standardize the data
X_scaled = StandardScaler().fit_transform(X)

# Dimensionality reduction for visualization
# First reduce with PCA to 50 dimensions, then apply t-SNE
print("Applying dimensionality reduction...")
pca = PCA(n_components=min(50, X.shape[0], X.shape[1]))
X_pca = pca.fit_transform(X_scaled)
print(f"PCA explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(X)-1))
X_tsne = tsne.fit_transform(X_pca)

# Try different numbers of clusters
k_range = range(2, min(8, len(X)+1))
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(results_dir, "clustering_elbow_curve.png"))
plt.close()

# Choose optimal k (this is a simplified approach)
# In a real application, you might want a more sophisticated method
# or manual selection based on domain knowledge
k_diffs = np.diff(inertias)
k_diffs_percent = k_diffs / inertias[:-1]
optimal_k_idx = np.argmax(k_diffs_percent) + 1
optimal_k = k_range[optimal_k_idx]

print(f"Optimal number of clusters based on elbow method: {optimal_k}")

# Apply KMeans with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Save cluster assignments with patient details
cluster_results = patient_details.copy()
cluster_results['cluster'] = clusters
cluster_results.to_csv(os.path.join(results_dir, "patient_clusters.csv"), index=False)

# Plot clusters using t-SNE
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', s=100, alpha=0.8)
plt.colorbar(scatter, label='Cluster')

# Add patient IDs as annotations
for i, patient_id in enumerate(patient_ids):
    plt.annotate(patient_id, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=8)

plt.title('Patient Clusters Based on PPG Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.savefig(os.path.join(results_dir, "patient_clusters_tsne.png"))
plt.close()

# Analyze outcomes by cluster
print("\nOutcomes by cluster:")
cluster_stats = cluster_results.groupby('cluster').agg({
    'icu_los': ['mean', 'std', 'min', 'max', 'count'],
    'mortality': ['mean', 'sum', 'count']
})
print(cluster_stats)

# Save cluster statistics
cluster_stats.to_csv(os.path.join(results_dir, "cluster_statistics.csv"))

# Plot mortality by cluster
plt.figure(figsize=(10, 6))
sns.barplot(x='cluster', y='mortality', data=cluster_results)
plt.title('Mortality Rate by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Mortality Rate')
plt.ylim(0, 1)
plt.savefig(os.path.join(results_dir, "mortality_by_cluster.png"))
plt.close()

# Plot LOS distribution by cluster
plt.figure(figsize=(12, 6))
sns.boxplot(x='cluster', y='icu_los', data=cluster_results)
plt.title('ICU Length of Stay by Cluster')
plt.xlabel('Cluster')
plt.ylabel('ICU Length of Stay (days)')
plt.savefig(os.path.join(results_dir, "los_by_cluster.png"))
plt.close()

print("\n=== Analysis Complete ===")
print(f"All results saved to: {results_dir}")
print("\nSummary of findings:")
print(f"1. Mortality Prediction: Best model: {best_model_name}, AUC: {mortality_results[best_model_name]['mean_auc']:.3f}")
print(f"2. ICU LOS Prediction: Best model: {best_model_name}, RMSE: {los_results[best_model_name]['mean_rmse']:.3f} days")
print(f"3. Patient Clustering: Identified {optimal_k} distinct patient groups based on PPG patterns")
