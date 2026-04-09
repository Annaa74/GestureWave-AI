import pyautogui
import time

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


class ActionExecutor:
    def __init__(self, safe_top_bar=90, enable_click_actions=True, training_mode=False):
        self.safe_top_bar = safe_top_bar
        self.enable_click_actions = enable_click_actions
        self.training_mode = training_mode
        self.simulated_action = ""
        self.action_time = 0.0

    def _log_action(self, name):
        if self.training_mode:
            self.simulated_action = name
            self.action_time = time.perf_counter()

    def safe_to_click(self, y):
        return self.enable_click_actions and y > self.safe_top_bar

    def move_cursor(self, x, y):
        if not self.training_mode:
            pyautogui.moveTo(x, y, _pause=False)

    def left_click(self, x, y):
        self._log_action("Left Click")
        if self.safe_to_click(y):
            if not self.training_mode:
                pyautogui.click(x, y)

    def right_click(self, x, y):
        self._log_action("Right Click")
        if self.safe_to_click(y):
            if not self.training_mode:
                pyautogui.rightClick(x, y)

    def double_click(self, x, y):
        self._log_action("Double Click")
        if self.safe_to_click(y):
            if not self.training_mode:
                pyautogui.doubleClick(x, y)

    def scroll_up(self, amount):
        self._log_action("Scroll Up")
        if not self.training_mode:
            pyautogui.scroll(amount)

    def scroll_down(self, amount):
        self._log_action("Scroll Down")
        if not self.training_mode:
            pyautogui.scroll(-amount)

    def zoom_in(self):
        self._log_action("Zoom In")
        if not self.training_mode:
            pyautogui.hotkey("ctrl", "+")

    def zoom_out(self):
        self._log_action("Zoom Out")
        if not self.training_mode:
            pyautogui.hotkey("ctrl", "-")