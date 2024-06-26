"""
Load the environment variables from .env file to set the credentials
"""

from dotenv import load_dotenv
import os
from wasabi import Printer
msg = Printer()

class Config:
    """Read the environment variables from .env file to set the credentials"""
    def __init__(self):
        load_dotenv()
        self.roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
    
    def validate(self):
        """Check if the environment file is se correctly"""
        if self.roboflow_api_key is None:
            raise ValueError("Roboflow API Key is not set")
        return True

config = Config()

if __name__ == "__main__":
    try:
        if config.validate():
            msg.good("Configuración válida")
    except ValueError as e:
        msg.fail(e)
        raise ValueError("Configuración inválida")
