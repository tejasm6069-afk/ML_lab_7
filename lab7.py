import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import DecisionBoundaryDisplay

# A1. Entropy Calculation
def calculate_entropy(y):
    """Calculates the entropy of a dataset."""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
    return entropy

# A2. Gini Index Calculation
def calculate_gini(y):
    """Calculates the Gini index of a dataset."""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities ** 2)
    return gini

# A4. Binning Function (Overloaded via Default Parameters)
def bin_continuous_feature(feature, bins=4, strategy='width'):
    """Converts continuous features to categorical using equal width or frequency binning."""
    if strategy == 'width':
        # Equal width binning
        return pd.cut(feature, bins=bins, labels=False)
    elif strategy == 'frequency':
        # Equal frequency binning
        return pd.qcut(feature, q=bins, labels=False, duplicates='drop')
    else:
        raise ValueError("Strategy must be 'width' or 'frequency'")

# A3. Information Gain Calculation
def information_gain(y, feature_splits):
    """Calculates information gain for a given feature split."""
    total_entropy = calculate_entropy(y)
    weighted_entropy = 0
    
    for split_val in np.unique(feature_splits):
        subset_y = y[feature_splits == split_val]
        weight = len(subset_y) / len(y)
        weighted_entropy += weight * calculate_entropy(subset_y)
        
    return total_entropy - weighted_entropy

def find_best_split(X, y):
    """Detects the best feature for the root node based on Information Gain."""
    best_gain = -1
    best_feature = None
    
    for col in X.columns:
        # Convert continuous to categorical for impurity measurement
        if np.issubdtype(X[col].dtype, np.number):
            binned_col = bin_continuous_feature(X[col], bins=4, strategy='width')
        else:
            binned_col = X[col]
            
        gain = information_gain(y, binned_col)
        if gain > best_gain:
            best_gain = gain
            best_feature = col
            
    return best_feature, best_gain

# A5. Custom Decision Tree Module (Simplified Recursive Structure)
class CustomDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth) or X.empty:
            return np.bincount(y).argmax() if len(y) > 0 else None
            
        best_feature, _ = find_best_split(X, y)
        if best_feature is None:
            return np.bincount(y).argmax()
            
        node = {'feature': best_feature, 'children': {}}
        binned_feat = bin_continuous_feature(X[best_feature]) if np.issubdtype(X[best_feature].dtype, np.number) else X[best_feature]
        
        for val in np.unique(binned_feat):
            mask = binned_feat == val
            node['children'][val] = self.fit(X[mask].drop(columns=[best_feature]), y[mask], depth + 1)
            
        self.tree = node
        return node

# Main Program Execution
if __name__ == "__main__":
    # Project Context: FoodLinkAI Expiration Urgency
    np.random.seed(42)
    data = pd.DataFrame({
        'days_to_expiry': np.random.uniform(1, 14, 100),
        'storage_temp': np.random.uniform(2, 25, 100),
        'humidity_level': np.random.uniform(30, 90, 100),
        'urgency_score': np.random.choice([0, 1], 100) # 0: Low Urgency, 1: High Urgency
    })
    
    X = data[['days_to_expiry', 'storage_temp', 'humidity_level']]
    y = data['urgency_score']

    print(f"Dataset Entropy: {calculate_entropy(y):.4f}")
    print(f"Dataset Gini Index: {calculate_gini(y):.4f}")
    
    best_feat, info_gain = find_best_split(X, y)
    print(f"Best Root Node Feature: {best_feat} (Gain: {info_gain:.4f})")

    # A6 & A7. Visualization using sklearn for robust plotting
    dt_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    dt_classifier.fit(X, y)
    
    plt.figure(figsize=(12, 8))
    plot_tree(dt_classifier, feature_names=X.columns, class_names=['Low', 'High'], filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()

    # Decision Boundary (using 2 features)
    X_2d = X[['days_to_expiry', 'storage_temp']]
    dt_2d = DecisionTreeClassifier(max_depth=3).fit(X_2d, y)
    
    DecisionBoundaryDisplay.from_estimator(dt_2d, X_2d, response_method="predict", cmap=plt.cm.RdBu, alpha=0.5)
    plt.scatter(X_2d.iloc[:, 0], X_2d.iloc[:, 1], c=y, edgecolor="k", cmap=plt.cm.RdBu)
    plt.title("Decision Boundary (FoodLinkAI Urgency)")
    plt.xlabel("Days to Expiry")
    plt.ylabel("Storage Temperature")
    plt.show()

    # A8. Hyper-parameter Tuning
    param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 4, 5], 'min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    grid_search.fit(X, y)
    print("Best Hyper-parameters:", grid_search.best_params_)
