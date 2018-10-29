#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import numpy as np
import Backgammon as BG
import torch 
from torch.autograd import Variable

# Global variables
n = 29
device = torch.device('cpu')
# cuda will only create a significant speedup for large/deep networks and batched training
# device = torch.device('cuda') 
# randomly initialized weights with zeros for the biases
w1 = Variable(torch.randn(n*n,2*n, device = device, dtype=torch.float), requires_grad = True)
b1 = Variable(torch.zeros((n*n,1), device = device, dtype=torch.float), requires_grad = True)
w2 = Variable(torch.randn(1,n*n, device = device, dtype=torch.float), requires_grad = True)
b2 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)


def flip_board(board_copy):
    #flips the game board and returns a new copy
    idx = np.array([0,24,23,22,21,20,19,18,17,16,15,14,13,
    12,11,10,9,8,7,6,5,4,3,2,1,26,25,28,27])
    flipped_board = -np.copy(board_copy[idx])
        
    return flipped_board

def flip_move(move):
    if len(move)!=0:
        for m in move:
            for m_i in range(2):
                m[m_i] = np.array([0,24,23,22,21,20,19,18,17,16,15,14,13,
                                12,11,10,9,8,7,6,5,4,3,2,1,26,25,28,27])[m[m_i]]        
    return move

def ice_hot_encoding(board):
    ice_hot = np.zeros( 2 * len(board) )
    ice_hot[np.where(board == 1)[0] ] = 1
    ice_hot[len(board) + np.where(board == -1)[0] ] = 1
    return ice_hot

def action(board_copy,dice,player,i):
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    if player == -1: 
        board_copy = flip_board(board_copy)
        
    possible_moves, possible_boards = BG.legal_moves(board_copy, dice, player = 1)
    na = np.size(possible_moves)
    va = np.zeros(na)
    
    # if there are no moves available
    if len(possible_moves) == 0: 
        return [] 
    
    for board in possible_boards:
        # encode the board to create the input
        x = Variable(torch.tensor(ice_hot_encoding(board), dtype = torch.float, device = device)).view(2 * n,1)
        # now do a forward pass to evaluate the board's after-state value
        h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
        va[i] = y.sigmoid()

    # Virkar i lista ?
    move = possible_moves[np.random.randint(len(possible_moves))]
    #move = possible_moves[np.random.choice(np.arange(0, na), p=va/np.sum(va))]
    if player == -1: 
        move = flip_move(move)

    return move

def feed_forward(board):
    return 0

def learnit(numgames, lam, alpha, alpha1, alpha2):
    gamma = 1 # for completeness
    # play numgames games for training
    for games in range(0, numgames):
        board = BG.init_board() # initialize the board
        player = np.random.randint(2)*2-1 # which player begins?
    
        # now we initilize all the eligibility traces for the neural network
        Z_w1 = torch.zeros(w1.size(), device = device, dtype = torch.float)
        Z_b1 = torch.zeros(b1.size(), device = device, dtype = torch.float)
        Z_w2 = torch.zeros(w2.size(), device = device, dtype = torch.float)
        Z_b2 = torch.zeros(b2.size(), device = device, dtype = torch.float)
        
        count = 0
        while not BG.game_over(board) and not BG.check_for_error(board):
            dice = BG.roll_dice()
            
            # if 
            for i in range(1+int(dice[0] == dice[1])):
                board_copy = np.copy(board)
                move = action(board_copy, dice, player, i)
                
                if len(move) != 0:
                    for m in move:
                        board = BG.update_board(board, m, player)
      
            if (count > 1):
                # here we have player 2 updating the neural-network (2 layer feed forward with Sigmoid units)
                x = Variable(torch.tensor(ice_hot_encoding(board_copy), dtype = torch.float, device = device)).view(2*n,1)
                # now do a forward pass to evaluate the new board's after-state value
                h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
                h_sigmoid = h.sigmoid() # squash this with a sigmoid function
                y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
                y_sigmoid = y.sigmoid() # squash this with a sigmoid function
                target = y_sigmoid.detach().cpu().numpy()
                
                # lets also do a forward past for the old board, this is the state we will update
                h = torch.mm(w1,xold) + b1 # matrix-multiply x with input weight w1 and add bias
                h_sigmoid = h.sigmoid() # squash this with a sigmoid function
                y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
                y_sigmoid = y.sigmoid() # squash the output
                delta2 = 0 + gamma * target - y_sigmoid.detach().cpu().numpy() # this is the usual TD error
                # using autograd and the contructed computational graph in pytorch compute all gradients
                y_sigmoid.backward()
                # update the eligibility traces using the gradients
                Z_w2 = gamma * lam * Z_w2 + w2.grad.data
                Z_b2 = gamma * lam * Z_b2 + b2.grad.data
                Z_w1 = gamma * lam * Z_w1 + w1.grad.data
                Z_b1 = gamma * lam * Z_b1 + b1.grad.data
                # zero the gradients
                w2.grad.data.zero_()
                b2.grad.data.zero_()
                w1.grad.data.zero_()
                b1.grad.data.zero_()
                # perform now the update for the weights
                delta2 =  torch.tensor(delta2, dtype = torch.float, device = device)
                w1.data = w1.data + alpha1 * delta2 * Z_w1
                b1.data = b1.data + alpha1 * delta2 * Z_b1
                w2.data = w2.data + alpha2 * delta2 * Z_w2
                b2.data = b2.data + alpha2 * delta2 * Z_b2
                # we need to keep track of the last board state visited by the players
            
            if(not BG.game_over(board)):
                xold = Variable(torch.tensor(ice_hot_encoding(board_copy), dtype=torch.float, device = device)).view(2*n,1)
            # swap players
            player = -player
            count += 1
        
        # The game epsiode has ended and we know the outcome of the game, and can find the terminal rewards
        reward = 1
        #update fyrir winner
        # these are basically the same updates as in the inner loop but for the final-after-states (sold and xold)        
        # and then for the neural network:
        h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
        y_sigmoid = y.sigmoid() # squash the output
        delta2 = reward + gamma * 0 - y_sigmoid.detach().cpu().numpy()  # this is the usual TD error
        # using autograd and the contructed computational graph in pytorch compute all gradients
        y_sigmoid.backward()
        # update the eligibility traces
        Z_w2 = gamma * lam * Z_w2 + w2.grad.data
        Z_b2 = gamma * lam * Z_b2 + b2.grad.data
        Z_w1 = gamma * lam * Z_w1 + w1.grad.data
        Z_b1 = gamma * lam * Z_b1 + b1.grad.data
        # zero the gradients
        w2.grad.data.zero_()
        b2.grad.data.zero_()
        w1.grad.data.zero_()
        b1.grad.data.zero_()
        # perform now the update of weights
        delta2 =  torch.tensor(delta2, dtype = torch.float, device = device)
        w1.data = w1.data + alpha1 * delta2 * Z_w1
        b1.data = b1.data + alpha1 * delta2 * Z_b1
        w2.data = w2.data + alpha2 * delta2 * Z_w2
        b2.data = b2.data + alpha2 * delta2 * Z_b2
        
        # update fyrir lúser
        reward = -1
        # these are basically the same updates as in the inner loop but for the final-after-states (sold and xold)        
        # and then for the neural network:
        h = torch.mm(w1,xold) + b1 # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
        y_sigmoid = y.sigmoid() # squash the output
        delta2 = reward + gamma * 0 - y_sigmoid.detach().cpu().numpy()  # this is the usual TD error
        # using autograd and the contructed computational graph in pytorch compute all gradients
        y_sigmoid.backward()
        # update the eligibility traces
        Z_w2 = gamma * lam * Z_w2 + w2.grad.data
        Z_b2 = gamma * lam * Z_b2 + b2.grad.data
        Z_w1 = gamma * lam * Z_w1 + w1.grad.data
        Z_b1 = gamma * lam * Z_b1 + b1.grad.data
        # zero the gradients
        w2.grad.data.zero_()
        b2.grad.data.zero_()
        w1.grad.data.zero_()
        b1.grad.data.zero_()
        # perform now the update of weights
        delta2 =  torch.tensor(delta2, dtype = torch.float, device = device)
        w1.data = w1.data + alpha1 * delta2 * Z_w1
        b1.data = b1.data + alpha1 * delta2 * Z_b1
        w2.data = w2.data + alpha2 * delta2 * Z_w2
        b2.data = b2.data + alpha2 * delta2 * Z_b2
 


alpha = 0.1 # step size for tabular learning
alpha1 = 0.01 # step sizes using for the neural network (first layer)
alpha2 = 0.01 # (second layer)
lam = 0.4 # lambda parameter in TD(lam-bda)

import time
start = time.time()
training_steps = 10
learnit(training_steps, lam, alpha, alpha1, alpha2)
end = time.time()
print(end - start)
