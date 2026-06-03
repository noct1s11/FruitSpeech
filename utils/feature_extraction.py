import librosa
import numpy as np

def extract_features(file_path, n_mfcc=13, sample_rate=16000):
    """
    Extracts MFCC features from an audio file.
    
    Parameters:
        file_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCC features to extract.
        sample_rate (int): Sample rate to resample the audio to.
        
    Returns:
        numpy.ndarray: Flat feature vector (mean of MFCCs over time).
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=sample_rate)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Take the mean across the time axis to get a 1D feature vector
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
        return mfccs_processed
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
