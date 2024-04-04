import munch
import toml


class GlobalConfig:
    @classmethod
    def load_global_config(filepath: str = "project_config.toml"):
        return munch.munchify(toml.load(filepath))

    @classmethod
    def save_global_config(new_config, filepath: str = "project_config.toml"):
        with open(filepath, "w") as file:
            toml.dump(new_config, file)

    config = load_global_config()
    data_config = config.data
    train_config = config.train
    model_config = config.model
