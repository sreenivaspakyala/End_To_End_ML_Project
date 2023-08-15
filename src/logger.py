import logging  # make sure your logger file doesn't have the same name
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(),'logs')
os.makedirs(log_path,exist_ok=True)

# log file path after creating the logs directory
LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
)

if __name__ == '__main__':
    print('Logging has started...')
    logging.info('Logging has Started.....')