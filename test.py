import configparser

config = configparser.ConfigParser()
config.read('config_run.ini')
print(config["Movielens_Basic"]["FILE_MOVIE_MISSING_PATH"])