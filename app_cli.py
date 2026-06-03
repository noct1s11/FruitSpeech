import os
from utils.audio_recorder import record_audio
from utils.tts import speak
from train_model import train
from predict import predict

# Available fruit classes
FRUITS = [
    "apel", "jeruk", "mangga", "pisang", "semangka", 
    "melon", "anggur", "pepaya", "nanas", "stroberi"
]

def print_header(title):
    print("=" * 50)
    print(f"{title:^50}")
    print("=" * 50)

def main_menu():
    while True:
        print_header("FruitSpeech - Deteksi Suara Buah")
        print("1. Rekam Sampel Dataset (Record Training Audio)")
        print("2. Latih Model (Train Classification Model)")
        print("3. Uji Coba Langsung (Record & Predict)")
        print("4. Keluar (Exit)")
        print("-" * 50)
        
        choice = input("Pilih menu (1-4): ").strip()
        
        if choice == '1':
            record_dataset_menu()
        elif choice == '2':
            print_header("Melatih Model...")
            train()
            input("\nTekan Enter untuk kembali ke menu utama...")
        elif choice == '3':
            predict_live_menu()
        elif choice == '4':
            print("\nTerima kasih telah menggunakan FruitSpeech!")
            speak("Terima kasih", lang='id')
            break
        else:
            print("Pilihan tidak valid, coba lagi.\n")

def record_dataset_menu():
    print_header("Rekam Sampel Dataset")
    print("Pilih buah yang ingin direkam:")
    for idx, fruit in enumerate(FRUITS, 1):
        print(f"{idx:2d}. {fruit.capitalize()}")
    print("0. Kembali ke Menu Utama")
    print("-" * 50)
    
    try:
        choice = int(input("Pilih nomor buah: ").strip())
        if choice == 0:
            return
        if 1 <= choice <= len(FRUITS):
            fruit_name = FRUITS[choice - 1]
            category_dir = os.path.join("dataset", fruit_name)
            os.makedirs(category_dir, exist_ok=True)
            
            # Find next file index
            existing_files = [f for f in os.listdir(category_dir) if f.endswith(".wav")]
            next_idx = len(existing_files) + 1
            filename = os.path.join(category_dir, f"{fruit_name}_{next_idx:03d}.wav")
            
            # Voice prompt
            speak(f"Ucapkan kata: {fruit_name}", lang='id')
            
            # Record
            record_audio(filename, duration=2.0)
            print(f"Sampel berhasil disimpan di {filename}\n")
            
        else:
            print("Pilihan tidak valid.\n")
    except ValueError:
        print("Masukkan angka yang benar.\n")
    
    input("Tekan Enter untuk melanjutkan...")

def predict_live_menu():
    print_header("Uji Coba Langsung (Prediksi Suara)")
    temp_predict_file = "temp_predict.wav"
    
    speak("Silakan bicara setelah aba-aba", lang='id')
    # Record test sample
    record_audio(temp_predict_file, duration=2.0)
    
    # Predict
    if os.path.exists(temp_predict_file):
        predict(temp_predict_file)
        # Clean up
        os.remove(temp_predict_file)
    else:
        print("Gagal merekam suara.")
        
    input("\nTekan Enter untuk kembali ke menu utama...")

if __name__ == "__main__":
    # Say hello at start
    speak("Halo! Selamat datang di FruitSpeech.", lang='id')
    main_menu()
