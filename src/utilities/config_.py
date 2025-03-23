import os
from pathlib import Path
import src.utilities as utils


# Paths
root_path = Path(__file__).parent.parent.parent.resolve()
log_path = root_path / "logs"
config_path = root_path / "configs"
train_data_path = root_path / "data" / "training"
morgan_train_data_path = root_path / "data" / "training" / "morgan"
ferrero_train_data_path = root_path / "data" / "training" / "ferrero"
predicted_data_path = root_path / "data" / "predicted"


class ConfigManager(object):
    """
    Config Manager to manage main configurations
    and store them as attributes depending on
    the environment
    """

    def __init__(self, config_file="main_config.yaml"):
        # load main_config
        self.params = utils.read_yaml(os.path.join(config_path, "main_config.yaml"))