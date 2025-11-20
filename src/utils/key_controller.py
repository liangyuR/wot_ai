from pynput import keyboard
class KeyController:
    """
    键盘控制类：封装基本的按键操作接口
    """
    def __init__(self):
        self._controller = keyboard.Controller()

    def press(self, key):
        """按下指定按键"""
        self._controller.press(key)

    def release(self, key):
        """释放指定按键"""
        self._controller.release(key)

    def tap(self, key):
        """点击（按下并释放）指定按键"""
        self._controller.press(key)
        self._controller.release(key)

    def type(self, text: str):
        """输入一段字符串"""
        self._controller.type(text)


if __name__ == "__main__":
    import time
    key_controller = KeyController()
    key_controller.tap('w')
    time.sleep(1)
    key_controller.tap('s')
    time.sleep(1)
    key_controller.tap('a')
    time.sleep(1)
    key_controller.tap('d')
    time.sleep(1)
    key_controller.tap('w')
    time.sleep(1)