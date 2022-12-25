import sys


def exception_message_detail(exception, exception_detail: sys):
    _, _, exception_traceback = exception_detail.exc_info()
    filename = exception_traceback.tb_frame.f_code.co_filename
    exception_message = "Exception get raised in python script name [{0}] in line number [{1}] exception message - [{2}].".format(str(filename), exception_traceback.tb_lineno, str(exception))
    return exception_message


class SensorException(Exception):
    def __init__(self, exception_message, exception_detail=sys):
        self.exception_message = exception_message_detail(exception_message, exception_detail)

    def __str__(self):
        return self.exception_message
