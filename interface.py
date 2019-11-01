from PyInquirer import prompt, print_json
from pprint import pprint
from PyInquirer import style_from_dict, Token, prompt, Separator
from examples import custom_style_2

from Parser import Parser


parser = Parser('datasetxls.xlsx')
parser.parse()
print(*parser.get_dataset_unmodified.keys())
#t = '\n'.join(parser.get_dataset_unmodified.keys())
#print(t)
questions = [
    {
        'type': 'checkbox',
        'qmark': '😃',
        'message': 'choose useless',
        'name': 'useless_columns',
        'choices': [{'name': i} for i in parser.get_dataset_unmodified.keys()]
    }
]

answers = prompt(questions, style=custom_style_2)
parser.remove_useless(*answers.values())
print(parser.dataset_no_useless)