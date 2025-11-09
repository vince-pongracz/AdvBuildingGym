from collections import deque


class TemporalFeatureBuffer:
    def __init__(self, window_size=12 * 5):
        self.window_size = window_size
        self.delta_history = deque(maxlen=window_size)
        self.action_history = deque(maxlen=window_size)

    def reset(self):
        self.delta_history.clear()
        self.action_history.clear()

    def append(self, delta, action):
        self.delta_history.append(delta)
        self.action_history.append(action)

    def get_padded_deltas(self):
        history = list(self.delta_history)
        while len(history) < self.window_size:
            history.insert(0, 0.0)
        return history

    def get_padded_actions(self):
        history = list(self.action_history)
        while len(history) < self.window_size:
            history.insert(0, 0.0)
        return history
