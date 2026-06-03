import os
import sys
import pickle
from utils.feature_extraction import extract_features
from utils.tts import speak

MODEL_PATH = os.path.join("models", "fruit_classifier.pkl")

def predict(audio_path, return_dict=False):
    """
    Predicts the fruit class of an audio file.
    
    Parameters:
        audio_path (str): Path to the audio WAV file.
        return_dict (bool): If True, returns a dictionary of results. Otherwise, returns prediction label string.
        
    Returns:
        str or dict: The prediction label or a dictionary with prediction, confidence, and probabilities.
    """
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found. Please train the model first by running train_model.py.")
        speak("Model klasifikasi belum dilatih.", lang='id')
        return None
        
    # Check if audio path exists
    if not os.path.exists(audio_path):
        print(f"Audio file '{audio_path}' not found.")
        return None
        
    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
        
    # Handle both new dictionary format and older formats
    if isinstance(model_data, dict) and 'classifier' in model_data:
        model = model_data['classifier']
    else:
        model = model_data
        
    print(f"Extracting features from {audio_path}...")
    features = extract_features(audio_path)
    
    if features is None:
        print("Failed to extract features.")
        return None
        
    # Reshape features to fit model input (1 sample, n features)
    features = features.reshape(1, -1)
    
    # Predict
    prediction = model.predict(features)[0]
    
    # Check if model supports predict_proba
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = model.classes_
        max_prob = max(probabilities)
        prob_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    else:
        # Fallback if probability is not enabled
        probabilities = [1.0]
        classes = [prediction]
        max_prob = 1.0
        prob_dict = {prediction: 1.0}
    
    print(f"\nResult: Predicted Fruit is '{prediction}' with confidence {max_prob * 100:.2f}%")
    
    # Speak the result in Indonesian (default command line behavior)
    if not return_dict:
        response_text = f"Suara yang didengar adalah {prediction}."
        speak(response_text, lang='id')
        return prediction
        
    return {
        'prediction': prediction,
        'confidence': max_prob,
        'probabilities': prob_dict
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_file>")
        print("Example: python predict.py test.wav")
    else:
        audio_file = sys.argv[1]
        predict(audio_file)
