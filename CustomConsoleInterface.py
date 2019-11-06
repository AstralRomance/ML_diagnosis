from PyInquirer import prompt, print_json
from pprint import pprint
from PyInquirer import style_from_dict, Token, prompt, Separator
from examples import custom_style_2

from Parser import DataPreparer

class CustomConsoleInterface:
    def make_checkbox(self, choices, message, name):
        questions = [
            {
                'type': 'checkbox',
                'qmark': 'o',
                'message': message,
                'name': name,
                'choices': choices
            }
        ]

        selected_columns = self._make_menu(questions)
        return selected_columns

    def make_list(self, choices, message, name):
        questions = [
            {
                'type': 'list',
                'message': message,
                'name': name,
                'choices': choices
            }
        ]
        selected_columns = self._make_menu(questions)
        return selected_columns

    def _make_menu(self, questions):
        answers = prompt(questions, style=custom_style_2)
        return answers

