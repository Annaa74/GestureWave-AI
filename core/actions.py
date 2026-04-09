import pyautogui
import time

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


class ActionExecutor:
    def __init__(self, safe_top_bar=90, enable_click_actions=True):
        self.safe_top_bar = safe_top_bar
        self.enable_click_actions = enable_click_actions

    def safe_to_click(self, y):
        return self.enable_click_actions and y > self.safe_top_bar

    def move_cursor(self, x, y):
        pyautogui.moveTo(x, y, _pause=False)

    def left_click(self, x, y):
        if self.safe_to_click(y):
            pyautogui.click(x, y)

    def right_click(self, x, y):
        if self.safe_to_click(y):
            pyautogui.rightClick(x, y)

    def double_click(self, x, y):
        if self.safe_to_click(y):
            pyautogui.doubleClick(x, y)

    def scroll_up(self, amount):
        pyautogui.scroll(amount)

    def scroll_down(self, amount):
        pyautogui.scroll(-amount)

    def zoom_in(self):
        pyautogui.hotkey("ctrl", "+")

    def zoom_out(self):
        pyautogui.hotkey("ctrl", "-")