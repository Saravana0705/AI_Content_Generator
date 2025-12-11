class Interface:
    def __init__(self):
        self.input_data = None

    def get_user_input(self, input_text):
        self.input_data = input_text
        return self.input_data