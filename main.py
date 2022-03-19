import matplotlib.pyplot as plt
import numpy as np
from state import State
from mcts import MCTS
from state import detection_kernels
from scipy.signal import convolve2d
from network import Network
import torch
from torch.nn.functional import normalize

# game logical variables:
# First player: color=red, number on the borad=1, turn=0, type: search algorithm
# Second player: color=green, number on the borad=-1, turn=1, type: search algorithm
ROW_COUNT=6
COLUMN_COUNT=7
FPLAYER_PIECE=1
SPLAYER_PIECE=-1
Num_Sim_Per_Move=200
Num_Training_Games=100000
batch_size=50

def end_of_game(board):
    for kernel in detection_kernels:
        result_board = convolve2d(board, kernel, mode="valid")
        if 4 in result_board or -4 in result_board:
            return "win or loss"

    if len(np.where(board == 0)[0]) == 0:
        return "draw"

    return "not ended"


def is_valid_loc(board,row,col):
    return board[row][col]==0


def self_play(brain):
    # set game parameters nd show the first status of the game
    game_over = False
    game_board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    turn = 0
    # instantiate tree and the root details
    root= State(parent=None,isroot=True,board=game_board,cp=torch.tensor(-1),N=torch.tensor(1),prior_policy=torch.tensor(0))
    action_probs,val = brain.predict(root.get_board())
    root.set_inferences(val,action_probs)
    tree = MCTS(root,Num_Sim_Per_Move,brain)
    # game loop
    while not game_over:
        # get the next move from players and change their boards
        if turn == 0:
            # player1 has to play
            row, cul = tree.search()
        else:
            # player2 has to play
            row, cul = tree.search()

        # put the new piece on the board
        if is_valid_loc(game_board, row, cul):
            if turn == 0:
                game_board[row][cul] = FPLAYER_PIECE
            else:
                game_board[row][cul] = SPLAYER_PIECE
        else:
            raise Exception("The move is not valid.")

        # check if game ends
        result = end_of_game(game_board)
        if result != "not ended":
            return tree.buffer(result)

        # change turn
        turn = 1 - turn

def plot_loss(value_loss,policy_loss):
    plt.clf()
    plt.plot(range(len(value_loss)), value_loss,label="value loss")
    plt.plot(range(len(policy_loss)), policy_loss,label="policy loss")
    plt.pause(0.001)
    plt.legend()
    plt.title("loss per batch")
    plt.xlabel("batch")
    plt.ylabel("loss")

def update(brain,vloss,ploss):
    brain.optimizer.zero_grad()
    ploss.backward(retain_graph=True)
    vloss.backward()
    brain.optimizer.step()

def save_net(brain):
    torch.save(brain.state_dict(),"brain.pt")

def train():
    counter=0
    policy_loss_pl=[]
    value_loss_pl=[]
    value_loss=0
    policy_loss=0
    brain=Network(input_shape=ROW_COUNT*COLUMN_COUNT,number_of_actions=COLUMN_COUNT)
    while counter<Num_Training_Games:
        vl,pl=self_play(brain)
        value_loss=value_loss+vl
        policy_loss=policy_loss+pl
        if counter%batch_size==0:
            policy_loss_pl.append(policy_loss.item()/batch_size)
            value_loss_pl.append(value_loss.item()/batch_size)
            plot_loss(value_loss_pl,policy_loss_pl)
            update(brain, value_loss, policy_loss)
            value_loss=0
            policy_loss=0
        counter+=1
    save_net(brain)



if __name__ == '__main__':
    train()