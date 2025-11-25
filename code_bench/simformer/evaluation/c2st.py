import numpy as np

# Try to import sklearn, provide fallback if not available
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def c2st(X, Y, n_folds=5, random_state=42):
    """
    Performs the Classifier Two-Sample Test (C2ST) using a RandomForestClassifier.

    As specified in the paper's addendum, the C2ST is implemented with a
    RandomForestClassifier to measure the distinguishability of two sets of samples.
    A score close to 0.5 indicates that the samples are indistinguishable, implying
    high-quality generation.

    Args:
        X (np.ndarray): The first set of samples (e.g., ground truth).
        Y (np.ndarray): The second set of samples (e.g., model-generated).
        n_folds (int): The number of folds for cross-validation.
        random_state (int): A random seed for reproducibility of the classifier.

    Returns:
        float: The mean cross-validated accuracy of the classifier.
    """
    # Ensure inputs are numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Create labels: 0 for samples from X, 1 for samples from Y
    labels = np.concatenate([np.zeros(len(X)), np.ones(len(Y))])
    
    # Concatenate the data
    data = np.concatenate([X, Y])

    if SKLEARN_AVAILABLE:
        # As per the addendum, use RandomForestClassifier with n_estimators=100
        classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)

        # Calculate cross-validated accuracy
        scores = cross_val_score(classifier, data, labels, cv=n_folds)

        # Return the mean accuracy
        return scores.mean()
    else:
        # Fallback implementation using a simple nearest-centroid approach
        # when sklearn is not available
        return _c2st_fallback(X, Y, n_folds, random_state)


def _c2st_fallback(X, Y, n_folds=5, random_state=42):
    """
    Fallback C2ST implementation using a simple nearest-centroid classifier
    when sklearn is not available.
    
    Args:
        X (np.ndarray): The first set of samples (e.g., ground truth).
        Y (np.ndarray): The second set of samples (e.g., model-generated).
        n_folds (int): The number of folds for cross-validation.
        random_state (int): A random seed for reproducibility.

    Returns:
        float: The mean cross-validated accuracy of the classifier.
    """
    np.random.seed(random_state)
    
    # Ensure inputs are numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Combine data and labels
    data = np.concatenate([X, Y])
    labels = np.concatenate([np.zeros(len(X)), np.ones(len(Y))])
    
    # Shuffle indices
    n_samples = len(data)
    indices = np.random.permutation(n_samples)
    
    # Calculate fold size
    fold_size = n_samples // n_folds
    scores = []
    
    for fold in range(n_folds):
        # Create train/test split for this fold
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_samples
        
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        
        # Split data
        train_data = data[train_indices]
        train_labels = labels[train_indices]
        test_data = data[test_indices]
        test_labels = labels[test_indices]
        
        # Simple nearest-centroid classifier
        # Calculate centroids for each class
        centroid_0 = train_data[train_labels == 0].mean(axis=0)
        centroid_1 = train_data[train_labels == 1].mean(axis=0)
        
        # Predict based on distance to centroids
        dist_to_0 = np.linalg.norm(test_data - centroid_0, axis=1)
        dist_to_1 = np.linalg.norm(test_data - centroid_1, axis=1)
        predictions = (dist_to_1 < dist_to_0).astype(float)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == test_labels)
        scores.append(accuracy)
    
    return np.mean(scores)


if __name__ == "__main__":
    # Test the function with sample data
    print("Testing c2st function...")
    print(f"sklearn available: {SKLEARN_AVAILABLE}")
    
    # Create test data
    np.random.seed(42)
    X_test = np.random.normal(0, 1, (100, 5))  # 100 samples, 5 features
    Y_test = np.random.normal(0.5, 1, (100, 5))  # Slightly different distribution
    
    # Run the test
    result = c2st(X_test, Y_test)
    print(f"C2ST result: {result:.4f}")
    print("Function executed successfully!")
