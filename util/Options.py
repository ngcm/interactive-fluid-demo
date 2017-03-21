import time

class Cycle:
    def __init__(self, name, key, options,
        initial=0,
        cached_update=False,
        auto_cycle=False, cycle_time=[]):
        self._name = name
        self._key = key
        self._options = options
        self._current = initial
        self._timer = 0
        self._auto_cycle = auto_cycle
        self._cycle_time = cycle_time
        self._changed = True

    def update(self, key, dt):
        if key == self._key:
            self.cycle()
            return True
        elif self._auto_cycle:
            self._timer += dt
            if self._timer > self._cycle_time[self._current]:
                self.cycle()
        return False

    def cycle(self):
        self._timer = 0
        self._changed = True
        self._current = (self._current + 1) % len(self._options)

    def get_has_changed(self, reset_change_flag=False):
        if self._changed:
            if reset_change_flag:
                self._changed = False
            return True
        else:
            return False

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
    def __init__(self, name, keys, range, step,
        current=0):
        assert len(keys) == 2
        assert len(range) == 2
        self._name = name
        self._keys = keys
        self._current = max(min(current, range[1]), range[0])
        self._range = range
        self._step = step
        self._changed = True

    def update(self, key, dt):
        old_value = self._current
        if key == self._keys[0]:
            self._current = max(self._current - self._step, self._range[0])
        elif key == self._keys[1]:
            self._current = min(self._current + self._step, self._range[1])

        if old_value != self._current:
            self._changed = True
            return True
        else:
            return False

    def get_has_changed(self, reset_change_flag=False):
        if self._changed:
            if reset_change_flag:
                self._changed = False
            return True
        else:
            return False

    @property
    def current(self):
        return self._current

    def __str__(self):
        return "{}({}/{})={:.2f}".format(self._name, self._keys[0],
                self._keys[1], self._current)
