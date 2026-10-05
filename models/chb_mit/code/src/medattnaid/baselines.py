import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def _metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def supervised_bilstm_reference(X, y):
    features = extract_features(X)
    train_x, test_x, train_y, test_y = train_test_split(
        features, y, test_size=0.30, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), random_state=42, max_iter=500)
    clf.fit(train_x, train_y)
    return _metrics(test_y, clf.predict(test_x))


def feature_kmeans(X, y):
    features = extract_features(X)
    scaled = StandardScaler().fit_transform(features)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(scaled, y)
    selected = np.argsort(rf.feature_importances_)[-min(30, scaled.shape[1]):]
    labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(scaled[:, selected])
    return _metrics(y, align_clusters(labels, y))


def tskmeans_dtw_reference(X, y):
    try:
        from tslearn.clustering import TimeSeriesKMeans
        labels = TimeSeriesKMeans(n_clusters=2, metric="dtw", max_iter=10, random_state=42, n_init=5).fit_predict(X)
    except Exception:
        labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X.reshape(len(X), -1))
    return _metrics(y, align_clusters(labels, y))


def extract_features(X):
    X = np.asarray(X).reshape(len(X), -1)
    gradients = np.diff(X, axis=1)
    percentiles = np.percentile(X, [20, 40, 60, 80], axis=1).T
    time_to_percentile = np.stack([
        (row[:, None] >= row_percentiles[None, :]).argmax(axis=0)
        for row, row_percentiles in zip(X, percentiles)
    ])
    return np.column_stack([
        X.mean(axis=1),
        X.var(axis=1),
        X.std(axis=1),
        X.min(axis=1),
        X.max(axis=1),
        np.ptp(X, axis=1),
        skew(X, axis=1, nan_policy="omit"),
        kurtosis(X, axis=1, nan_policy="omit"),
        time_to_percentile,
        gradients.mean(axis=1),
        gradients.max(axis=1),
        gradients.std(axis=1),
        np.percentile(gradients, 75, axis=1) - np.percentile(gradients, 25, axis=1),
    ])


def align_clusters(labels, y):
    labels = np.asarray(labels)
    y = np.asarray(y)
    cluster0_normal = np.sum((labels == 0) & (y == 0))
    cluster1_normal = np.sum((labels == 1) & (y == 0))
    normal_cluster = 0 if cluster0_normal >= cluster1_normal else 1
    return (labels != normal_cluster).astype(int)
