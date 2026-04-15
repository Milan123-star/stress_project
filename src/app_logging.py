import logging

logger=logging.getLogger("stress")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

File_handler=logging.FileHandler("errors.log")
File_handler.setLevel("ERROR")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
File_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(File_handler)