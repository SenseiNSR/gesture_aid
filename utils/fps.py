import time

class FPSMonitor:
    def __init__(self):
        self.pTime = 0
        self.fps = 0
        
    def update(self):
        cTime = time.time()
        if cTime - self.pTime > 0:
            self.fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        return int(self.fps)
