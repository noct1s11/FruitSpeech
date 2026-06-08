import librosa
import numpy as np

def extract_features(file_path, n_mfcc=13, sample_rate=16000):
    """
    Extracts enhanced MFCC features from an audio file (mean, std, delta mean, delta std).
    
    Parameters:
        file_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCC features to extract.
        sample_rate (int): Sample rate to resample the audio to.
        
    Returns:
        numpy.ndarray: Flat feature vector (52 dimensions).
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=sample_rate)
        
        # Trim silence from the beginning and end
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Fallback if the trimmed audio is too short or empty
        if len(y_trimmed) < 1600:
            y_trimmed = y
            
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc)
        
        # Calculate Delta MFCCs (velocity)
        delta_mfccs = librosa.feature.delta(mfccs)
        
        # Aggregate features across the time axis (mean and standard deviation)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        mfcc_std = np.std(mfccs.T, axis=0)
        delta_mean = np.mean(delta_mfccs.T, axis=0)
        delta_std = np.std(delta_mfccs.T, axis=0)
        
        # Combine all features into a 1D vector (52 features)
        features_combined = np.hstack([mfcc_mean, mfcc_std, delta_mean, delta_std])
        
        return features_combined
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
