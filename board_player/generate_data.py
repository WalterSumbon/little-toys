import random
from .game import Game, SlightlySmarterAI

def saves(board, choice):
    lines = []
    for line in board:
        lines.append(''.join([str(x) for x in line]))
    lines = ''.join(lines)
    return lines + str(choice) + '\n'

def loads(s):
    pass

def play_with_human(model):
    game = Game()
    while not game.is_over():
        while True:
            print(game.plot())
            choice = int(input('Enter your choice: '))
            try:
                choice = choice.split(',')
                choice = (int(choice[0]), int(choice[1]))
            except:
                print('Invalid input. Please enter two numbers separated by comma')
                continue

            if(game.place(choice)):
                break
            else:
                print('Invalid choice. Place already taken.')
        
        model.choose(game.board)

