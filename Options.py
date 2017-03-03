import time

class Cycle:    
    def __init__(self, name, key, options, current=0, auto_cycle=False, cycle_time=[]):
        self._name = name
        self._key = key
        self._current = current
        self._options = options
        self._timer = 0
        self._auto_cycle = auto_cycle
        self._cycle_time = cycle_time
        
    def update(self, key, dt):
        if key == ord(self._key):
            self.cycle();
            time.sleep(0.5)
            self._timer = 0
            return True
        elif self._auto_cycle:
            self._timer += dt
            if self._timer > self._cycle_time[self._current]:
                self._timer = 0
                self.cycle()
        return False
        
    def cycle(self):
        self._current = (self._current + 1) % len(self._options)
        
    @property
    def current(self):
        return self._current
    
    @property
    def current_name(self):
        return self._options[self._current]
        
    def __str__(self):
        return "{}({})={}".format(self._name, self._key, 
                self._options[self._current])
    
    
class Range:
    def __init__(self, name, keys, range, step, current=0):
        assert len(keys) == 2
        assert len(range) == 2
        self._name = name
        self._keys = keys
        self._current = max(min(current, range[1]), range[0])
        self._range = range
        self._step = step
                
    def update(self, key, dt):
        if key == ord(self._keys[0]):            
            self._current = max(self._current - self._step, self._range[0])
            return True        
        if key == ord(self._keys[1]):
            self._current = min(self._current + self._step, self._range[1])
            return True
        else:
            return False
        
    @property
    def current(self):
        return self._current
    
    def __str__(self):
        return "{}({}/{})={:.2f}".format(self._name, self._keys[0], 
                self._keys[1], self._current)