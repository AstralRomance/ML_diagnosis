from PyInquirer import prompt, print_json
from pprint import pprint
from PyInquirer import style_from_dict, Token, prompt, Separator
from examples import custom_style_2

from Parser import Parser


parser = Parser('datasetxls.xlsx')
parser.parse()
print(*parser.get_dataset_unmodified.keys())

questions = [
    {
        'type': 'checkbox',
        'qmark': '😃',
        'message': 'Select toppings',
        'name': 'toppings',
        'choices': [
            Separator('Cells of dataset'),
            {
                'name' : [i for i in parser.get_dataset_unmodified.keys()]
            }

        ],
        'validate': lambda answer: 'You must choose at least one topping.' \
            if len(answer) == 0 else True
    }
]

answers = prompt(questions, style=custom_style_2)
pprint(answers)
