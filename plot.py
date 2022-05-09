import sys

import matplotlib.pyplot as plt
import numpy as np



def read_text(file):
    with open(file) as f:
        lines = f.readlines()
    loss=[]
    for i in lines:
        entries=i.split()
        loss.append([float(entries[1]),float(entries[3]),float(entries[5])])
    return loss


def draw_loss(loss):
    plt.figure()
    plt.plot(range(1,loss.shape[0]+1), loss[:,0], label='min')
    plt.plot(range(1,loss.shape[0]+1), loss[:,1], label='average')
    plt.plot(range(1,loss.shape[0]+1), loss[:,2], label='max')
    plt.xlabel("checkpoint", fontsize=12)
    plt.ylabel("loss", fontsize=12)
    #plt.ylim([0, 600])
    #plt.title("AlphaZero Training loss", fontsize=12)
    plt.grid(True)
    plt.legend(loc='best', bbox_to_anchor=(.75, 0.5, 0.0, 0.0))
    plt.show()


def match_stats(file):
    with open(file) as f:
        lines = f.readlines()
    outcomes=[int(lines[0].split()[1]),int(lines[0].split()[3]),int(lines[0].split()[5])]
    lengthes=[]
    for i in range(1,len(lines)):
        lengthes.append(int(lines[i]))
    return outcomes,lengthes

def draw_match_stats(stats):
    plt.figure()
    #ax = fig.add_axes(["draw", "win","loss"])
    langs = ["draw", "win","loss"]
    students = [stats[0][0],stats[0][1],stats[0][2]]
    plt.bar(langs, students)
    plt.xlabel("outcome", fontsize=12)
    plt.ylabel("number of games", fontsize=12)
    plt.show()

def draw_game_sizes(stats):
    plt.figure()
    lengthes=np.asarray(stats[1])
    print(lengthes.shape)
    plt.plot(range(1, lengthes.shape[0] + 1), lengthes, label='game length',color='green')

    plt.xlabel("game", fontsize=12)
    plt.ylabel("length", fontsize=12)
    # plt.ylim([0, 600])
    # plt.title("AlphaZero Training loss", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #loss=np.asarray(read_text("results/300_8_128_0.003_with scheduling/sizes.txt"))
    stats=match_stats("results/match.txt")
    #draw_loss(loss)
    draw_match_stats(stats)
    #draw_game_sizes(stats)