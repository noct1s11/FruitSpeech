from gtts import gTTS
import os
import subprocess
import librosa
import soundfile as sf

def generate_tts_audio(text, lang='id', speed='normal', gender='perempuan'):
    """
    Generates text-to-speech audio using gTTS and applies DSP (pitch-shift and time-stretch)
    using librosa to customize speed and gender.
    
    Parameters:
        text (str): The text to synthesize.
        lang (str): Language code (default 'id').
        speed (str): Speed option ('slow', 'normal', 'fast').
        gender (str): Gender option ('laki-laki', 'perempuan').
        
    Returns:
        tuple: (y, sr) where y is the audio time series (numpy array) and sr is the sampling rate.
    """
    # 1. Generate base speech with gTTS (always female by default)
    tts = gTTS(text=text, lang=lang)
    temp_mp3 = "temp_raw_gtts.mp3"
    tts.save(temp_mp3)
    
    try:
        # 2. Load the audio file
        y, sr = librosa.load(temp_mp3, sr=None)
    finally:
        # Clean up the raw gTTS output
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
            
    # 3. Apply speed adjustment (time stretch)
    if speed == 'slow':
        # Stretch audio (stretch factor < 1.0 slows down, e.g. 0.75)
        y = librosa.effects.time_stretch(y=y, rate=0.75)
    elif speed == 'fast':
        # Stretch audio (stretch factor > 1.0 speeds up, e.g. 1.3)
        y = librosa.effects.time_stretch(y=y, rate=1.3)
        
    # 4. Apply gender adjustment (pitch shift)
    if gender == 'laki-laki':
        # Shift pitch down for male voice (e.g. -3.5 semitones)
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-3.5)
    elif gender == 'perempuan':
        # Shift pitch up slightly for a clearer female voice
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=0.8)
        
    return y, sr

def speak(text, lang='id', speed='normal', gender='perempuan', filename="temp_tts.wav"):
    """
    Converts text to speech, saves to filename, and plays it locally using standard macOS command 'afplay'
    or a fallback method.
    
    Parameters:
        text (str): The text to be spoken.
        lang (str): Language code (default is 'id' for Indonesian).
        speed (str): Speed option ('slow', 'normal', 'fast').
        gender (str): Gender option ('laki-laki', 'perempuan').
        filename (str): Path to write the output wav file.
    """
    print(f"TTS (speed={speed}, gender={gender}): {text}")
    try:
        y, sr = generate_tts_audio(text, lang=lang, speed=speed, gender=gender)
        
        # Save the audio file as a WAV file
        sf.write(filename, y, sr, subtype='PCM_16')
        
        # Play the audio file using afplay (macOS native command)
        if os.name == 'posix':
            subprocess.run(["afplay", filename])
        else:
            print(f"[TTS Output]: {text}")
            
        # Clean up the file if it's the default temporary one
        if filename == "temp_tts.wav" and os.path.exists(filename):
            os.remove(filename)
            
    except Exception as e:
        print(f"Error in TTS: {e}")

if __name__ == "__main__":
    # Test TTS in Indonesian
    speak("Halo! Selamat datang di program pengenalan suara buah.", speed='normal', gender='laki-laki')
