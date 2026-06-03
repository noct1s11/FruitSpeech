import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils.feature_extraction import extract_features

# Path settings
DATASET_DIR = "dataset"
MODEL_PATH = os.path.join("models", "fruit_classifier.pkl")

def train(model_type='SVM'):
    X = []
    y = []
    
    # Check if dataset directory exists
    if not os.path.exists(DATASET_DIR):
        msg = f"Dataset directory '{DATASET_DIR}' not found. Please create it and add audio files."
        print(msg)
        return {'status': 'error', 'message': msg}
        
    print("Loading dataset and extracting features...")
    
    # Iterate through fruit categories (subdirectories)
    categories = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    
    if not categories:
        msg = "No fruit categories found inside the dataset/ folder."
        print(msg)
        return {'status': 'error', 'message': msg}
        
    for category in categories:
        category_path = os.path.join(DATASET_DIR, category)
        files = [f for f in os.listdir(category_path) if f.endswith(".wav")]
        
        print(f"Processing '{category}' ({len(files)} files)...")
        for file in files:
            file_path = os.path.join(category_path, file)
            features = extract_features(file_path)
            
            if features is not None:
                X.append(features)
                y.append(category)
                
    if not X:
        msg = "No training samples found. Please record WAV files in dataset/ subfolders first."
        print(msg)
        return {'status': 'error', 'message': msg}
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset loaded: {len(X)} samples.")
    
    # Split dataset into train and test sets
    classes = sorted(list(set(y)))
    counts = [np.sum(y == c) for c in classes]
    can_stratify = len(classes) > 1 and min(counts) > 1 and int(len(X) * 0.2) >= len(classes)
    
    # If the total samples is extremely small or less than number of classes,
    # we don't split to avoid train_test_split errors.
    if len(X) > len(classes) and len(X) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if can_stratify else None
        )
    else:
        # Use all data for both train and test if data is too small
        X_train, X_test, y_train, y_test = X, X, y, y
    
    if model_type == 'MLP':
        print("Training Multi-Layer Perceptron (MLP) classifier...")
        classifier = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    else:
        print("Training Support Vector Machine (SVM) classifier...")
        classifier = SVC(kernel='linear', probability=True, random_state=42)
        
    classifier.fit(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    print(f"Training complete. Test Accuracy: {accuracy * 100:.2f}%")
    
    import time
    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save model along with metadata
    model_data = {
        'classifier': classifier,
        'model_type': model_type,
        'classes': classes,
        'accuracy': accuracy,
        'num_samples': len(X),
        'report': report_dict,
        'confusion_matrix': cm.tolist(),
        'trained_at': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {MODEL_PATH}")
    
    return {
        'status': 'success',
        'accuracy': accuracy,
        'report': report_dict,
        'confusion_matrix': cm.tolist(),
        'classes': classes,
        'num_samples': len(X)
    }

if __name__ == "__main__":
    train()
