import matplotlib.pyplot as plt
import numpy as np
from state import State
from mcts import MCTS
from state import detection_kernels
from scipy.signal import convolve2d
from network import Network
import torch
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# game logical variables:
# First player: color=red, number on the borad=1, turn=0, type: search algorithm
# Second player: color=green, number on the borad=-1, turn=1, type: search algorithm
ROW_COUNT = 6
COLUMN_COUNT = 7
FPLAYER_PIECE = 1
SPLAYER_PIECE = -1
Num_Sim_Per_Move = 100
Num_Training_Games = 5000
batch_size = 100


def end_of_game(board):
    for kernel in detection_kernels:
        result_board = convolve2d(board, kernel, mode="valid")
        if 4 in result_board or -4 in result_board:
            return "win or loss"

    if len(np.where(board == 0)[0]) == 0:
        return "draw"

    return "not ended"


def is_valid_loc(board, row, col):
    return board[row][col] == 0


def self_play(brain):
    # set game parameters nd show the first status of the game
    game_over = False
    game_board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    turn = 0
    # instantiate tree and the root details
    root = State(parent=None, isroot=True, board=game_board, cp=-1, N=1,
                 prior_policy=torch.tensor(0))
    action_probs, val = brain(root.get_board())
    root.set_inferences(val, action_probs)
    tree = MCTS(root, Num_Sim_Per_Move, brain)
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


def plot_loss(value_loss, policy_loss):
    with open("/content/drive/MyDrive/Connect4_AlphaZero/value loss", "wb") as fp:
        pickle.dump(value_loss, fp)
    with open("/content/drive/MyDrive/Connect4_AlphaZero/policy loss", "wb") as fp:
        pickle.dump(policy_loss, fp)

    plt.clf()
    fig_v, ax_v = plt.subplots()
    ax_v.plot(range(len(value_loss)), value_loss, label="value loss")
    ax_v.legend()
    plt.title("loss per batch")
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.savefig('/content/drive/MyDrive/Connect4_AlphaZero/value loss')
    plt.close(fig_v)

    fig_p, ax_p = plt.subplots()
    ax_p.plot(range(len(policy_loss)), policy_loss, label="policy loss")
    ax_p.legend()
    plt.title("loss per batch")
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.savefig('/content/drive/MyDrive/Connect4_AlphaZero/policy loss')
    plt.close(fig_p)

def update(brain, vloss, ploss):
    brain.optimizer.zero_grad()
    vloss.backward(retain_graph=True)
    ploss.backward()
    brain.optimizer.step()
    save_net(brain)


def save_net(brain):
    torch.save(brain.state_dict(), "/content/drive/MyDrive/Connect4_AlphaZero/brain.pt")


def train(load=False):
    value_buffer = []
    policy_buffer = []
    counter = 0

    if not load:
        brain = Network().to(device)
        policy_loss_pl = []
        value_loss_pl = []
    else:
        with open("/content/drive/MyDrive/Connect4_AlphaZero/value loss", "rb") as fp:
             value_loss_pl= pickle.load(fp)
        with open("/content/drive/MyDrive/Connect4_AlphaZero/policy loss", "rb") as fp:
             policy_loss_pl= pickle.load(fp)
        brain = Network().to(device)
        brain.load_state_dict(torch.load("/content/drive/MyDrive/Connect4_AlphaZero/brain.pt"))

    while counter < Num_Training_Games:
        vl, pl= self_play(brain)

        value_buffer.extend(vl)
        policy_buffer.extend(pl)

        if counter % batch_size == 0 and counter != 0:

            value_loss=sum(value_buffer)/len(value_buffer)
            policy_loss=sum(policy_buffer)/len(policy_buffer)

            policy_loss_pl.append(policy_loss.item())
            value_loss_pl.append(value_loss.item())
            plot_loss(value_loss_pl, policy_loss_pl)

            update(brain, value_loss, policy_loss)

            value_buffer = []
            policy_buffer = []
            print("number of played games: "+str(counter))

        counter += 1
    save_net(brain)


if __name__ == '__main__':
    train()