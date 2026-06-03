import sounddevice as sd
import soundfile as sf
import os

def record_audio(filename, duration=2.0, sample_rate=16000):
    """
    Records audio from the microphone and saves it as a WAV file.
    
    Parameters:
        filename (str): Path to save the audio file.
        duration (float): Duration of recording in seconds.
        sample_rate (int): Sampling rate of the recording.
    """
    print(f"Recording '{filename}' for {duration} seconds... Speak now!")
    # Record audio in mono (channels=1)
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    print("Recording finished.")
    
    # Ensure directory exists
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    # Save the recorded audio as a WAV file
    sf.write(filename, recording, sample_rate, subtype='PCM_16')
    print(f"Saved to {filename}")

if __name__ == "__main__":
    # Test recording
    test_file = "test.wav"
    record_audio(test_file, duration=2.0)
    if os.path.exists(test_file):
        os.remove(test_file)
        print("Test file cleaned up.")
