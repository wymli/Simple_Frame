
class Logger:
    def __init__(self , log_path):
        self.log_path = log_path
        self.buffer = []
        self.BUFFER_LEN = 10

    def log(self , str):
        print(str)
        self.buffer.append(str)
        if len(self.buffer) > self.BUFFER_LEN:
            self.flush()
        return self

    def flush(self):
        buf_ = "\n".join(self.buffer)
        with open(self.log_path , "a") as f:
            f.write(buf_)
