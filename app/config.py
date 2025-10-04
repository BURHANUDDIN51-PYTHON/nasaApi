from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    This class loads environment variables from a .env file.
    It uses pydantic-settings for robust validation and type hints.
    """
    
    # Define your configuration variables here with type hints.
    # The variable names must match the names in your .env file (case-insensitive).
    GROQ_API_KEY: str
    PINECONE_API_KEY: str
    GEMINI_API_KEY: str
    
    # This tells pydantic-settings to load variables from a .env file.
    model_config = SettingsConfigDict(env_file=".env")


# Create a single instance of the Settings class.
# You can import this `settings` object throughout your application
# to access your configuration variables.
settings = Settings()
