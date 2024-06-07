"""
Download the agave hd dataset from roboflow universe, API key is needed
"""
import os
from roboflow import Roboflow
from utils.config import config
from wasabi import Printer

class RoboflowDownloader:
    def __init__(self):
        self.printer = Printer()
        self.printer.info(f"Initializing RoboflowDownloader with API Key: {config.roboflow_api_key}")
        if config.validate():
            self.rf = Roboflow(api_key=config.roboflow_api_key)
        else:
            raise ValueError("Invalid configuration")
    
    def download_dataset(self, workspace_name, project_name, version_number, dataset_format, download_path):
        self.printer.info("Initializing download process...")
        
        # Verificar conexión y autenticación
        # TODO: Move the download folder, now its downloaded in root
        try:
            self.printer.info(f"Connecting to workspace: {workspace_name}")
            project = self.rf.workspace(workspace_name).project(project_name)
            self.printer.info(f"Accessing project: {project_name}")
            version = project.version(version_number)
            self.printer.info(f"Accessing version: {version_number}")
        except Exception as e:
            self.printer.fail(f"Error accessing project or version: {e}")
            return None
        
        self.printer.info(f"Workspace: {workspace_name}, Project: {project_name}, Version: {version_number}")
        
        dataset_path = os.path.join(download_path, f"{project_name}_{version_number}")
        
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path, exist_ok=True)
            self.printer.info(f"Created directory: {dataset_path}")
            try:
                self.printer.info(f"Downloading dataset in format: {dataset_format}")
                # Asegurarse de que la descarga se guarde en la ubicación especificada
                dataset = version.download(dataset_format)
                
                # Mover los archivos descargados a la ubicación especificada
                for item in os.listdir(version.location):
                    s = os.path.join(version.location, item)
                    d = os.path.join(dataset_path, item)
                    if os.path.isdir(s):
                        os.makedirs(d, exist_ok=True)
                    else:
                        os.rename(s, d)
                
                self.printer.good(f"Dataset downloaded successfully and saved to {dataset_path}")
                return dataset
            except Exception as e:
                self.printer.fail(f"Error downloading dataset: {e}")
                return None
        else:
            self.printer.warn(f"Dataset already exists at {dataset_path}. Download skipped.")
            return None

if __name__ == "__main__":
    downloader = RoboflowDownloader()
    dataset = downloader.download_dataset("agave", "agavehd", 1, "voc", "data/datasets")
    if dataset is None:
        print("Download process encountered an issue.")
    else:
        print("Download process completed successfully.")
