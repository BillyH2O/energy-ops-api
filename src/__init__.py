import os
from dotenv import load_dotenv

load_dotenv()

for env_var in ["ENV", "MLFLOW_SERVER", "MLFLOW_REGISTRY_NAME", "GOOGLE_APPLICATION_CREDENTIALS"]:
    if not os.getenv(env_var):
        raise Exception(f"Environment variable {env_var} must be defined.")