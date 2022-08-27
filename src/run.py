from utils import logger
from generate import generate_datasets
from processing import preprocess_datasets
from training import train_model




if __name__ == "__main__":
    logger.info("FERv3 is starting...")

    # download, generate, and preprocess datasets
    generate_datasets()
    preprocess_datasets()
    
    # run model training 
    train_model()



