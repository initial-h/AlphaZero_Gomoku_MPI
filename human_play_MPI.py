# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:45:36 2018

@author: initial-h
"""
'''
write a root parallel mcts and vote a move like ensemble way
why i do this:
when i train the network, i should try some parameter settings 
and train for a while to compare which is better,
so there are many semi-finished model get useless,it's waste of computation resource 
even though i can continue to train based on the semi-finished model. 
i write the parallel way using MPI,
so that each rank can load different model and then vote the next move to play, besides, 
you can also weights each model to get the weighted next move(i don't do it here but it's easy to realize).

and also each rank can load the same model and vote the next move, 
besides the upper benifit ,it can also improve the strength and save the playout time by parallel.
some other parallel ways can find in《Parallel Monte-Carlo Tree Search》.
'''

from game_board import Board
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorlayer import PolicyValueNet
from mpi4py import MPI
from collections import Counter
from GUI_v1_4 import GUI

# how  to run :
# mpiexec -np 2 python -u human_play_mpi.py

#　MPI setting
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# game setting
n_in_row = 5
width, height = 11, 11

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board,is_selfplay=False,print_probs_value=0):
        # no use params in the func : is_selfplay,print_probs_value
        # just to stay the same with AI's API
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move,_ = self.get_action(board)
        return move,None

    def __str__(self):
        return "Human {}".format(self.player)

def graphic(board, player1=1, player2=2):
    '''
    Draw the board and show game info
    '''
    width = board.width
    height = board.height

    print("Player", player1, "with X".rjust(3))
    print("Player", player2, "with O".rjust(3))
    print(board.states)
    print()
    print(' ' * 2, end='')
    # rjust()
    # http://www.runoob.com/python/att-string-rjust.html
    for x in range(width):
        print("{0:4}".format(x), end='')
    # print('\r\n')
    print('\r')
    for i in range(height - 1, -1, -1):
        print("{0:4d}".format(i), end='')
        for j in range(width):
            loc = i * width + j
            p = board.states.get(loc, -1)
            if p == player1:
                print('X'.center(4), end='')
            elif p == player2:
                print('O'.center(4), end='')
            else:
                print('-'.center(4), end='')
        # print('\r\n') # new line
        print('\r')


board= Board(width=width,height=height,n_in_row=n_in_row)

# init model here
# if you want to load different models in each rank,
# you can assign it here ,like
# if rank == 0 : model_file = '...'
# if rank == 1 : model_file = '...'

model_file='model_11_11_5/best_policy.model'
best_policy = PolicyValueNet(board_width=width,board_height=height,block=19,init_model=model_file,cuda=True)
alpha_zero_player = MCTSPlayer(policy_value_function=best_policy.policy_value_fn_random,
                         action_fc=best_policy.action_fc_test,
                         evaluation_fc=best_policy.evaluation_fc2_test,
                         c_puct=5,
                         n_playout=400,
                         is_selfplay=False)

player1 = Human()
player2 = alpha_zero_player
# player2 = MCTS_Pure(5,200)

def start_play(start_player=0, is_shown=1):
    # run a gomoku game with AI in terminal
    bcast_move = -1

    # init game and player
    board.init_board()
    player2.reset_player()

    end = False
    if rank == 0 and is_shown:
        # draw board in terminal
        graphic(board=board)

    if start_player == 0:
        # human first to play
        if rank == 0:
            bcast_move,move_probs = player1.get_action(board=board,is_selfplay=False,print_probs_value=False)
        # bcast the move to other ranks
        bcast_move = comm.bcast(bcast_move, root=0)
        # print('!'*10,rank,bcast_move)

        # human do move
        board.do_move(bcast_move)

        if rank == 0:
            # print move index
            print(board.move_to_location(bcast_move))
            if is_shown:
                graphic(board=board)

    while True:

        # reset the search tree
        player2.reset_player()
        # AI's turn
        if rank == 0:
            # print prior probabilities
            gather_move, move_probs = player2.get_action(board=board,is_selfplay=False,print_probs_value=True)
        else:
            gather_move, move_probs = player2.get_action(board=board, is_selfplay=False, print_probs_value=False)

        gather_move_list = comm.gather(gather_move, root=0)

        if rank == 0:
            # gather ecah rank's move and get the most selected one
            print('list is', gather_move_list)
            bcast_move = Counter(gather_move_list).most_common()[0][0]

        # bcast the move to other ranks
        bcast_move = comm.bcast(bcast_move, root=0)
        # print('!' * 10, rank, bcast_move)

        # AI do move
        board.do_move(bcast_move)
        # print('rank:', rank, board.availables)

        if rank == 0:
            print(board.move_to_location(bcast_move))
            if is_shown:
                graphic(board=board)
        end, winner = board.game_end()

        # check if game end
        if end:
            if rank == 0:
                if winner != -1:
                    print("Game end. Winner is ", winner)
                else:
                    print("Game end. Tie")
            break

        # human's turn
        if rank == 0:
            bcast_move, move_probs = player1.get_action(board=board)

        # bcast the move to other ranks
        bcast_move = comm.bcast(bcast_move, root=0)
        # print('!'*10,rank,bcast_move)

        # human do move
        board.do_move(bcast_move)
        # print('rank:', rank, board.availables)

        if rank == 0:
            print(board.move_to_location(bcast_move))
            if is_shown:
                graphic(board=board)
        end, winner = board.game_end()

        # check if game end
        if end:
            if rank == 0:
                if winner != -1:
                    print("Game end. Winner is ", winner)
                else:
                    print("Game end. Tie")
            break

def start_play_with_UI(start_player=0):
    # run a gomoku game with AI in GUI
    bcast_move = -1

    # init game and player
    board.init_board()
    player2.reset_player()

    current_player_num = start_player
    restart = 0
    end = False
    if rank == 0:
        SP = start_player
        UI = GUI(board.width)

    while True:

        if rank == 0:
            if current_player_num == 0:
                UI.show_messages('Your turn')
            else:
                UI.show_messages('AI\'s turn')

        # AI's turn
        if current_player_num == 1 and not end:
            # reset the search tree
            player2.reset_player()
            if rank == 0:
                # print prior probabilities
                gather_move, move_probs = player2.get_action(board=board, is_selfplay=False, print_probs_value=True)
            else:
                gather_move, move_probs = player2.get_action(board=board, is_selfplay=False, print_probs_value=False)

            gather_move_list = comm.gather(gather_move, root=0)
            # print('list is', gather_move_list)

            if rank == 0:
                # gather ecah rank's move and get the most selected one
                print('list is', gather_move_list)
                bcast_move = Counter(gather_move_list).most_common()[0][0]
                # print(board.move_to_location(bcast_move))

        # human's turn
        else:
            if rank == 0:
                inp = UI.get_input()
                if inp[0] == 'move' and not end:
                    if type(inp[1]) != int:
                        bcast_move = UI.loc_2_move(inp[1])
                    else:
                        bcast_move = inp[1]

                elif inp[0] == 'RestartGame':
                    UI.restart_game()
                    restart = SP+1

                elif inp[0] == 'ResetScore':
                    UI.reset_score()
                    continue

                elif inp[0] == 'quit':
                    restart = 'exit'

                elif inp[0] == 'SwitchPlayer':
                    SP = (SP + 1) % 2
                    UI.restart_game(False)
                    UI.reset_score()
                    restart = SP+1

                else:
                    # print('ignored inp:', inp)
                    continue

        restart = comm.bcast(restart, root=0)

        if not end and not restart:
            # bcast the move to other ranks
            bcast_move = comm.bcast(bcast_move, root=0)
            # print('!'*10,rank,bcast_move)
            if rank == 0:
                print(board.move_to_location(bcast_move))
                UI.render_step(bcast_move, board.current_player)

            # human do move
            board.do_move(bcast_move)
            # print('rank:', rank, board.availables)

            current_player_num = (current_player_num + 1) % 2
            end, winner = board.game_end()

            # check if game end
            if end:
                if rank == 0:
                    if winner != -1:
                        print("Game end. Winner is ", winner)
                        UI.add_score(winner)
                    else:
                        print("Game end. Tie")
        else:
            if restart:
                if restart == 'exit':
                    exit()
                board.init_board()
                player2.reset_player()
                current_player_num = restart-1
                restart = 0
                end = False


if __name__ == '__main__':
    # start_play(start_player=0,is_shown=True)
    start_play_with_UI()


