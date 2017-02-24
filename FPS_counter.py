import time

class FPS_counter:
    def __init__(self, limit=0):
        self._time = time.time() # seconds since epoch
        self._dt = 0 # seconds
        self._min_dt = 1 / limit if limit > 0 else 0
        self._limited = False
         
    def _get_dt(self):
        return time.time() - self._time
       
    def update(self):
        dt = self._get_dt()
        if dt < self._min_dt:
            time.sleep(self._min_dt - dt)
            self._time += self._min_dt
            self._dt = self._dt * 0.9 + self._min_dt * 0.1
            self._limited = True
        else:
            self._time += dt
            self._dt = self._dt * 0.9 + dt * 0.1
            self._limited = False
    
    @property
    def dt_remaining(self):
        return max(self._min_dt - self._get_dt(), 0)
    
    @property
    def last_dt(self):
        return self._dt
        
    def __str__(self):
        return '{:.2f}fps{}'.format(1 / self._dt, '+' if self._limited else '-')
        
    
