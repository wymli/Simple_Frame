
class Logger:
    def __init__(self , log_path):
        self.log_path = log_path
        self.buffer = ""
        self.BUFFER_LEN = 4396

    def log(self , str):
        print(str)
        self.buffer += "\n"+str
        if len(self.buffer) > self.BUFFER_LEN:
            self.flush()

    def flush(self):
        with open(self.log_path , "a") as f:
            f.write(self.buffer)
