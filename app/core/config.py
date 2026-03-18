from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    MODEL_PATH: str = "ml/model.pt"

    ALLOWED_ORIGINS: str = "http://localhost:3000"

    # ── Mapping TARGET_COLS → trường API ──────────────────────────────────
    # Tên cột trong TARGET_COLS tương ứng với plant_signal_a và plant_signal_b
    # Xem TARGET_COLS trong notebook training của bạn
    PLANT_SIGNAL_A_COL: str = "attr_3"   # deep head index 2
    PLANT_SIGNAL_B_COL: str = "attr_6"   # deep head index 5

    # Giá trị tối đa model có thể trả về cho plant signals
    # Dùng để normalize về 0–99
    PLANT_SIGNAL_MAX: int = 99

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]


settings = Settings()
