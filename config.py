import munch
import toml


def load_global_config( filepath : str = "project_config.toml" ):
    return munch.munchify( toml.load( filepath ) )

def save_global_config( new_config , filepath : str = "project_config.toml" ):
    with open( filepath , "w" ) as file:
        toml.dump( new_config , file )

if __name__ == "__main__":
    config = load_global_config()
    print( config.train.batch_size )
    config.data.vocab_size = 113000
    save_global_config( config )
