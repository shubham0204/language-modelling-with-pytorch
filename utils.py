import pickle


def save_dict_as_pickle(data: dict, filename: str):
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def load_dict_from_pickle(filename) -> dict:
    with open(filename, "rb") as file:
        return pickle.load(file)
