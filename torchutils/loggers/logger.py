import logging as ori_logging
from colorama import Fore

def _reformat_msg(msg):
    if len(msg) == 1:
        return msg[0]
    else:
        return [i for i in msg]
    
class LogWrapper:
    
    _logger = ori_logging.getLogger("logger")
    _format = ori_logging.Formatter(f'%(asctime)s %(message)s')
    _handler = ori_logging.StreamHandler()
    _handler.setFormatter(_format)
    _logger.addHandler(_handler)
    _logger.setLevel(ori_logging.INFO)

    @staticmethod
    def setLevel(level):
        __class__._logger.setLevel(getattr(ori_logging, level))

    @staticmethod
    def addFileHandler(fn):
        hanlder = ori_logging.FileHandler(fn)
        __class__._logger.addHandler(hanlder)
    
    @staticmethod
    def info(*msg):
        __class__._logger.info(Fore.GREEN + '[    INFO   ] ' + str(_reformat_msg(msg)) + Fore.RESET)
    
    @staticmethod
    def debug(*msg):
        __class__._logger.debug(Fore.CYAN + '[   DEBUG   ] ' + str(_reformat_msg(msg)) + Fore.RESET)
    
    @staticmethod
    def error(*msg):
        __class__._logger.error(Fore.RED + '[   ERROR   ] ' + str(_reformat_msg(msg)) + Fore.RESET)
    
    @staticmethod
    def warn(*msg):
        __class__._logger.warn(Fore.YELLOW + '[    WARN   ] ' + str(_reformat_msg(msg)) + Fore.RESET)
    
    @staticmethod
    def warning(*msg):
        __class__._logger.warning(Fore.YELLOW + '[    WARN   ] ' + str(_reformat_msg(msg)) + Fore.RESET)
    
    @staticmethod
    def fatal(*msg):
        __class__._logger.fatal(Fore.BLUE + '[   FATAL   ] ' + str(_reformat_msg(msg)) + Fore.RESET)
    
    @staticmethod
    def exception(*msg):
        __class__._logger.exception(Fore.RED + '[ EXCEPTION ] ' + str(_reformat_msg(msg)) + Fore.RESET)

    @staticmethod
    def critical(*msg):
        __class__._logger.critical(Fore.MAGENTA + '[  CRITICAL ] ' + str(_reformat_msg(msg)) + Fore.RESET)

LogWrapper.setLevel('INFO')
logging = LogWrapper