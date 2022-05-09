import numpy as np
from state import State
from alphazero_mcts import AlphaZero_MCTS 
from state import detection_kernels
from scipy.signal import convolve2d
from network import Network
import torch
from torch.nn.utils import clip_grad_norm_
from uct_mcts import UCT_MCTS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# game logical variables:
# First player: color=red, number on the borad=1, turn=0, type: search algorithm
# Second player: color=green, number on the borad=-1, turn=1, type: search algorithm
ROW_COUNT = 6
COLUMN_COUNT = 7
FPLAYER_PIECE = 1
SPLAYER_PIECE = -1
Num_Sim_Per_Move = 300
Num_Training_Games =300000
batch_size = 100
Max_Norm=1
epoch_num=50


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

def save_loss(value_loss, policy_loss,loss):
    f= open('loss.txt', 'a')
    f.write("value: "+str(value_loss)+"   policy: "+str(policy_loss)+"   loss: "+str(loss)+"\n")
    f.close()

def save_game_size(min_length,mean_length,max_length):
    f= open('sizes.txt', 'a')
    f.write("min: "+str(min_length)+"   mean: "+str(mean_length)+"   max: "+str(max_length)+"\n")
    f.close()

def save_counters(counter):
    f= open('counters.txt', 'a')
    f.write("game: "+str(counter)+"\n")
    f.close()

def self_play(brain,counter):
    # set game parameters nd show the first status of the game
    game_over = False
    game_board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    turn = 0
    # instantiate tree and the root details
    root = State(parent=None, isroot=True, board=game_board, cp=1, N=1,
                 prior_policy=torch.tensor(0))
    root.encode_board()
    action_probs, val = brain(root.get_encoded_board())
    root.set_inferences(val, action_probs)
    tree = AlphaZero_MCTS(root, Num_Sim_Per_Move, brain,counter)
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


def update(brain,buffer):
    values=[i[1] for i in buffer]
    policies=[i[2] for i in buffer]
    states=np.reshape(np.array([i[0] for i in buffer]),(len(buffer),3,6,7))
    for i in range(epoch_num):
        policy_pred,val_pred=brain(states)
        value_loss=torch.sum((torch.stack(values)-val_pred)**2)
        policy_loss=torch.sum(-torch.stack(policies)*policy_pred.log())
        loss=value_loss+policy_loss
        loss=loss/len(buffer)
        brain.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        clip_grad_norm_(brain.parameters(), Max_Norm)
        brain.optimizer.step()
    brain.scheduler.step()
    
    save_loss(value_loss.item()/len(buffer),policy_loss.item()/len(buffer),loss.item())


def save_net(brain):
    torch.save(brain.state_dict(), "brain.pt")
    torch.save(brain.optimizer.state_dict(),"optimizer.pt")


def train(load=False):
    buffer = []
    min_length=ROW_COUNT*COLUMN_COUNT
    max_length=0

    if not load:
        brain = Network().to(device)
        counter = 1
    else:
        print("brain is loading...")
        counter=7601
        brain = Network().to(device)
        brain.load_state_dict(torch.load("brain.pt"))
        brain.optimizer.load_state_dict(torch.load("optimizer.pt"))

    while counter <= Num_Training_Games:
        temp_buffer=self_play(brain,counter)
        buffer.extend(temp_buffer)

        if len(temp_buffer)<min_length:
            min_length=len(temp_buffer)
        if len(temp_buffer)>max_length:
            max_length=len(temp_buffer)
        
        if counter % batch_size == 0:
            update(brain, buffer)
            save_game_size(min_length,len(buffer)//batch_size,max_length)
            save_counters(counter)
            save_net(brain)
            buffer=[]
            min_length=ROW_COUNT*COLUMN_COUNT
            max_length=0
        
        counter += 1


def game_loop():
    game_over=False
    game_board=np.zeros((ROW_COUNT,COLUMN_COUNT))
    turn=0

    # instantiate players
    root1 = State(parent=None, isroot=True, board=game_board, cp=1, N=1,
                 prior_policy=torch.tensor(0))    
    root2 = State(parent=None, isroot=True, board=game_board, cp=1, N=1,
                 prior_policy=torch.tensor(0))
    
    print("brain is loading...")
    brain = Network().to(device)
    brain.load_state_dict(torch.load("brain.pt"))
    brain.optimizer.load_state_dict(torch.load("optimizer.pt"))
    brain.eval()

    root1.encode_board()
    action_probs, val = brain(root1.get_encoded_board())
    root1.set_inferences(val, action_probs)

    player1=AlphaZero_MCTS(root1, Num_Sim_Per_Move, brain,0)
    player2=UCT_MCTS(root2,UCT_MCTS.simulation1,3)
    counter=1
    while not game_over:
        #get the next move from players and change their boards
        if turn==0:
            #player1 has to play
            row,cul=player1.search()
            player2.change_root(row,cul)
        else:
            #player2 has to play
            row,cul=player2.search()
            player1.change_root(row,cul)

        #put the new piece on the board and show it
        if is_valid_loc(game_board,row,cul):
            if turn==0:
                game_board[row][cul] = FPLAYER_PIECE
            else:
                game_board[row][cul] = SPLAYER_PIECE
        else:
            raise Exception("The move is not valid.")

        #check if game ends and announce the winner
        result=end_of_game(game_board)
        if result=="win or loss":
            game_over=True
            if turn==0:
                #player1 wins
                return "AlphaZero wins",counter
            else:
                return "UCT wins",counter
        
        if game_over=="draw":
            return "draw",counter
        
        #change turn and print it
        turn=1-turn
        counter=counter+1


def test():
    test_games=100
    draws=0
    wins=0
    losses=0
    game_lengthes=[]
    for i in range(test_games):
        outcome,length=game_loop()
        print("game: "+str(i)+"  outcome: "+outcome)
        game_lengthes.append(length)
        if outcome=="draw":
            draws=draws+1
        elif outcome=="UCT wins":
            losses=losses+1
        else:
            wins=wins+1
    print("*"*20)
    print("draws: "+ str(draws))
    print("wins: "+ str(wins))
    print("losses: "+ str(losses))
    f= open('match.txt', 'a')
    f.write("draw: "+str(draws)+"   wins: "+str(wins)+"   losses: "+str(losses)+"\n")
    for i in range(len(game_lengthes)):
        f.write(str(game_lengthes[i])+"\n")
    f.close()

if __name__ == '__main__':
    #train()
    test()