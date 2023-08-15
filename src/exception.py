import sys
from src.logger import logging

def error_details(error, error_detail:sys):
    _,_, exc_tb = error_detail.exc_info()   # tb-> traceback object
    filename = exc_tb.tb_frame.f_code.co_filename
    linenum = exc_tb.tb_lineno
    error_msg = "An error has occured in the python script [{0}] at line number [{1}] "\
         "with error message [{2}]".format(filename,linenum,str(error))
    return error_msg

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_details(error_message,error_detail=error_detail)
    
    def __str__(self) -> str:
        return self.error_message


if __name__ == '__main__':
    try:
        c = 1/0
    except Exception as err:
        logging.error(err)
        raise CustomException(err,sys)