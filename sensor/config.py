from dataclasses import dataclass
import pymongo
@dataclass
class EnvironmentVariable:
    MONGODB_URL = "mongodb+srv://aakash:aakash@cluster0.4gxtdut.mongodb.net/?retryWrites=true&w=majority"

env_var = EnvironmentVariable()

mongo_client = pymongo.MongoClient(env_var.MONGODB_URL)