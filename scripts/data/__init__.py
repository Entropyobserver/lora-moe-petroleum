def __init__(self, config: dict):
    self.config = config
    project_root = Path(__file__).resolve().parent.parent.parent
    self.data_dir = project_root / config["paths"]["data_dir"]