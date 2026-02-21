from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    MAX_CHUNK_SIZE: int = 1024 * 1024  # 1MB
    ALLOWED_ORIGINS: list[str] = ["*"]
    
    # Audio Processing Settings
    SAMPLE_RATE: int = 16000
    N_MELS: int = 128
    HOP_LENGTH: int = 512
    N_FFT: int = 2048
    MIN_DURATION: float = 0.1  # seconds
    MAX_DURATION: float = 30.0  # seconds
    
    # ML Inference Settings
    MODEL_PATH: str = "models/deepfake_voice_detector_repaired.pth"
    # MFCC params must match training (see deepfake_voice_detector.ipynb)
    MODEL_SAMPLE_RATE: int = 22050   # model was trained at 22 050 Hz
    MFCC_N_MFCC: int = 40            # number of MFCC coefficients
    MFCC_MAX_LEN: int = 174          # fixed time-frame length (~4 s @ 22 050 Hz)
    
    # Risk Engine Settings
    RISK_WINDOW_SIZE: int = 10
    RISK_THRESHOLD_LOW: float = 0.3
    RISK_THRESHOLD_HIGH: float = 0.7
    
    # Phase 4: Robustness Settings
    VAD_ENERGY_THRESHOLD: float = 0.01
    PREDICTION_SMOOTHING_FACTOR: float = 0.3
    MAX_CHUNKS_PER_SECOND: int = 10
    CHUNK_OVERLAP_SAMPLES: int = 1600  # 100ms at 16kHz
    ROBUSTNESS_SNR_THRESHOLD: float = 15.0  # dB
    APP_VERSION: str = "1.0.0"

    class Config:
        env_file = ".env"

settings = Settings()
