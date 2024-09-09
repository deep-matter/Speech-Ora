
from collections import deque

class Chat:
    """
    Handles the chat using a circular buffer to avoid OOM issues.
    """

    def __init__(self, size):
        self.init_chat_message = None
        self.buffer = deque(maxlen=size)

    def append(self, item):
        self.buffer.append(item)

    def init_chat(self, init_chat_message):
        self.init_chat_message = init_chat_message

    def to_list(self):
        if self.init_chat_message:
            return [self.init_chat_message] + list(self.buffer)
        else:
            return list(self.buffer)
