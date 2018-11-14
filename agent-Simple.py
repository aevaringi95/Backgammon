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
import flipped_agent
from torch.autograd import Variable
import time


#flips the game board and returns a new copy
def flip_board(board_copy):
    idx = np.array([0,24,23,22,21,20,19,18,17,16,15,14,13,
    12,11,10,9,8,7,6,5,4,3,2,1,26,25,28,27])
    flipped_board = -np.copy(board_copy[idx])
    return flipped_board

# flips the move
def flip_move(move):
    if len(move)!=0:
        for m in move:
            for m_i in range(2):
                m[m_i] = np.array([0,24,23,22,21,20,19,18,17,16,15,14,13,
                                12,11,10,9,8,7,6,5,4,3,2,1,26,25,28,27])[m[m_i]]        
    return move


# takes in the board, outputs a binary representation of the board.
# the vector contains every triangle on the board. The first seven elements for
# each triangle represent the state of player one on said triangle. The next seven
# represent player 2. The first element represents if the player has no checker on
# said triangle, the second represents one checker etc. The seventh element 
# represents that there are six or more checkers on said triangle. Then next
# 14 elements are for the next triangle and so forth.
# example: [0,0,1,0,0,0,0] means that there are 2 checkers on this triangle.
def ice_hot_encoding(board):
    ice_hot = np.zeros( 2 * (len(board)-1) * 7)
    for i in range(1,len(board)):
        k = board[i].astype(np.int64) # number of checkers on triangle i
        # if it´s player 1.
        if(k > 0):
            if(k > 5):
                ice_hot[6 + (i-1)*14] = 1
                ice_hot[(i-1)*14 + 7] = 1
            else:
                ice_hot[k + (i-1)*14] = 1
                ice_hot[(i-1)*14 + 7] = 1
        # if it's player -1.
        elif(k < 0):
            if(k < -5):
                ice_hot[6 + (i-1)*14 + 7] = 1
                ice_hot[(i-1)*14] = 1
            else:
                ice_hot[-k + (i-1)*14 + 7] = 1
                ice_hot[(i-1)*14] = 1
        
        # if there is no player on said triangle.
        elif(k == 0):
            ice_hot[k + (i-1)*14] = 1
            ice_hot[np.abs(k) + (i-1)*14 + 7] = 1
    
    return ice_hot

"""
input: board, dice, player, i and learning
output:
    if learning == true: move
    else: move, probability, index for the move in possible_moves
"""
"""
def action(board_copy, dice, player, i, learning = False):
    
    if player == -1: 
        board_copy = flip_board(board_copy)
        
    possible_moves, possible_boards = BG.legal_moves(board_copy, dice, player = 1)
    na = len(possible_moves)
    # stores the output of the NN
    values = Variable(torch.zeros(na, device = device, dtype=torch.float), requires_grad = False)
    # if there are no moves available, returns no move but evaluates the board with NN
    if len(possible_moves) == 0: 
        # encode the board to create the input for the NN
        x = Variable(torch.tensor(ice_hot_encoding(board_copy), dtype = torch.float, device = device)).view(2*(n-1)*7,1)
        # forward pass to evaluate the board's after-state value
        values_temp = feed_forward_th(x)
        prob_temp = values_temp.softmax(dim=0)
        #prob_temp = torch.tensor([values_temp],dtype=torch.float, device = device,requires_grad=True)
        move_index = torch.tensor([0], device = device)
        if learning == True:
            return [], prob_temp, move_index
        else:    
            return [] 
    
    j = 0
    # Evaluates every possible board with NN and stores it in values
    for board in possible_boards:
        # encode the board to create the input for the NN
        x = Variable(torch.tensor(ice_hot_encoding(board), dtype = torch.float, device = device)).view(2*(n-1)*7,1)
        # forward pass to evaluate the board's after-state value
        values[j] = feed_forward_th(x)
        j+=1
    
    prob = values.softmax(dim = 0)
    #prob = torch.tensor(prob,dtype=torch.float, device = device,requires_grad=True)
    move_index = torch.multinomial(prob, num_samples  = 1)
    move_index = Variable(move_index, requires_grad=False)
    move = possible_moves[move_index]
    if player == -1: 
        move = flip_move(move)
    if learning == True:
        return move, prob, move_index
    
    return move
"""

"""
input: board, dice, player, i and learning
output:
    if learning == true: move
    else: move, probability, index for the move in possible_moves
"""
def action(board_copy,dice,player,i, learning = False):
    if player == -1: 
        board_copy = flip_board(board_copy)
    
    # Get every possible move and board
    xtheta_mean = torch.zeros((len(theta),1))
    possible_moves, possible_boards = BG.legal_moves(board_copy, dice, player = 1)
    na = len(possible_moves)
    one_hot_boards = np.zeros((2*(n-1)*7,na))
    j = 0
    # if there are no moves available
    if len(possible_moves) == 0: 
        x = Variable(torch.tensor(ice_hot_encoding(board_copy), dtype = torch.float, device = device)).view(2*(n-1)*7,1)
        h_sigmoid = feed_forward_th(x)
        pi = torch.mm(theta,h_sigmoid).softmax(0)
        xtheta_mean = h_sigmoid*pi.item()
        if learning == True:
            return [], xtheta_mean
        else:    
            return []
    
    for board in possible_boards:
        # encode the board to create the input for the NN
        x = Variable(torch.tensor(ice_hot_encoding(board), dtype = torch.float, device = device)).view(2*(n-1)*7,1)
        one_hot_boards[:,j] = x[:,0]
        j+=1
    # select the move from a distribution
    X=Variable(torch.tensor(one_hot_boards, dtype = torch.float,device = device))
    h=feed_forward_th(X)
    h_sigmoid=h.sigmoid()
    pi = torch.mm(theta,h_sigmoid).softmax(1)
    xtheta_mean = torch.sum(torch.mm(h_sigmoid,torch.diagflat(pi)),1)
    xtheta_mean = torch.unsqueeze(xtheta_mean,1)
    move_index = torch.multinomial(pi, num_samples  = 1)
    move = possible_moves[move_index]
    if player == -1: 
        move = flip_move(move)
    
    if learning == True:
        return move, xtheta_mean
    
    return move



# zero the gradients for the critic, w.
def zero_gradients_critic():
    w2.grad.data.zero_()
    b2.grad.data.zero_()
    w1.grad.data.zero_()
    b1.grad.data.zero_()
    
# zero the gradients for the actor, theta.
def zero_gradients_actor():
    theta_2.grad.data.zero_()
    thetab2.grad.data.zero_()
    theta_1.grad.data.zero_()
    thetab1.grad.data.zero_()
    
     
# update the eligibility traces for the critic, w
def update_eligibility_w(gamma, lam_w, zw1, zb1, zw2, zb2):
    zw2 = gamma * lam_w * zw2 + w2.grad.data
    zb2 = gamma * lam_w * zb2 + b2.grad.data
    zw1 = gamma * lam_w * zw1 + w1.grad.data
    zb1 = gamma * lam_w * zb1 + b1.grad.data
    return zw2, zb2, zw1, zb1

# update the eligibility traces for the actor, theta
def update_eligibility_th(gamma, lam_theta, ztheta1, zthetab1, ztheta2, zthetab2,I):
    ztheta2 = gamma * lam_theta * ztheta2 + I*theta_2.grad.data
    zthetab2 = gamma * lam_theta * zthetab2 + I*thetab2.grad.data
    ztheta1 = gamma * lam_theta * ztheta1 + I*theta_1.grad.data
    zthetab1 = gamma * lam_theta * zthetab1 + I*thetab1.grad.data
    return ztheta2, zthetab2, ztheta1, zthetab1

# feed forward for the critic's NN
def feed_forward_w(x):
    h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.sigmoid() # squash this with a sigmoid function
    y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
    y_sigmoid = y.sigmoid() # squash the output
    return y_sigmoid,h_sigmoid

# feed forward for the actor's NN
def feed_forward_th(x):
    h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight theta_1 and add bias
    h_sigmoid = h.sigmoid() # squash this with a sigmoid function
    return h

"""
input: number of games, lambda for w, lambda for theta, alpha1 for w1/theta_1 and alpha2 for w2/theta_2
This function is used to train the NN for backgammon with self-play. 
"""
def learnit(numgames, lam_w, lam_th, alpha_w, alpha_th):
    gamma = 1 # for completeness
    # play numgames games for training
    for games in range(0, numgames):
        board = BG.init_board() # initialize the board
        player = np.random.randint(2)*2-1 # which player begins?
        
        randomPlayer=-1
        learningPlayer=1
    
        # initilize all the eligibility traces for the NN for critic w
        Z_w1 = torch.zeros(w1.size(), device = device, dtype = torch.float)
        Z_b1 = torch.zeros(b1.size(), device = device, dtype = torch.float)
        Z_w2 = torch.zeros(w2.size(), device = device, dtype = torch.float)
        Z_b2 = torch.zeros(b2.size(), device = device, dtype = torch.float)
        
        if games % 100 == 0:
            print(games)
        count = 0
        delta = 0
        # play a game
        while not BG.game_over(board) and not BG.check_for_error(board):
            
            dice = BG.roll_dice()
            for i in range(1+int(dice[0] == dice[1])):
                if player == randomPlayer:
                    move = flipped_agent.action(np.copy(board),dice,player,i) 
                else:
                    move, xtheta  = action(np.copy(board), dice, player, i, True)
                if len(move) != 0:
                    for m in move:
                        board = BG.update_board(board, m, player)
                # if the player gets a double and wins the game in the first move. 
                if BG.game_over(board):
                    break
            
            # check to see if the game is over
            if BG.game_over(board):
                break
            
            
            # only update after the first two moves, because we are using afterstates
            # and both players have to make one move first. 
            if (count > 1 and player==learningPlayer):
                
                
                # Ice-hot encoding of the board
                x = Variable(torch.tensor(ice_hot_encoding(board), dtype = torch.float, device = device)).view(2*(n-1)*7,1)
                
                #Feed forward w-NN
                target,h_sigmoidOLD = feed_forward_w(x)

                #Feed forward old state using w-NN
                old_target,h_sigmoid = feed_forward_w(xold)
                delta = 0 + gamma * target.detach().cpu().numpy() - old_target.detach().cpu().numpy() # this is the usual TD error
                # using autograd and the contructed computational graph in pytorch compute all gradients
                old_target.backward()
                # update the eligibility traces using the gradients
                delta =  torch.tensor(delta, dtype = torch.float, device = device)
                Z_w2, Z_b2, Z_w1, Z_b1 = update_eligibility_w(gamma, lam_w, Z_w1, Z_b1, Z_w2, Z_b2)
                # zero the gradients
                zero_gradients_critic() 
                # perform the update for the weights for the critic, w
                w1.data = w1.data + alpha_w * delta * Z_w1
                b1.data = b1.data + alpha_w * delta * Z_b1
                w2.data = w2.data + alpha_w * delta * Z_w2
                b2.data = b2.data + alpha_w * delta * Z_b2
                
                #Update theta
                grad_ln_pi = h_sigmoid - xtheta
                theta.data = theta.data + alpha_th*delta*grad_ln_pi.view(1,len(grad_ln_pi))
            
            # keep track of the last state the player was in
            # keep track of the last state
            if (count < 2 and player==learningPlayer):
                xold = Variable(torch.tensor(ice_hot_encoding(board), dtype=torch.float, device = device)).view(2*(n-1)*7,1)
            if (count >= 2 and player==learningPlayer):   
                xold = x
            
            # swap players
            player = -player
            count += 1         
        # The game episode has ended and we know the outcome of the game, and can find the terminal rewards
        reward = -1*player  
        # update for the winner
        # these are basically the same updates as in the inner loop but for the final-after-states (x and xold)        
        # and then for the neural network:
        win_target,h_sigmoid= feed_forward_w(x)
        
        delta = reward + gamma * 0 - win_target.detach().cpu().numpy()  # this is the usual TD error
        delta =  torch.tensor(delta, dtype = torch.float, device = device)
        # using autograd and the contructed computational graph in pytorch compute all gradients
        win_target.backward()
        # update the eligibility traces
        Z_w2, Z_b2, Z_w1, Z_b1 = update_eligibility_w(gamma, lam_w, Z_w1, Z_b1, Z_w2, Z_b2)
        # zero the gradients
        zero_gradients_critic()
        # perform now the update of weights
        w1.data = w1.data + alpha_w * delta * Z_w1
        b1.data = b1.data + alpha_w * delta * Z_b1
        w2.data = w2.data + alpha_w * delta * Z_w2
        b2.data = b2.data + alpha_w * delta * Z_b2
        
        # Update theta
        grad_ln_pi = h_sigmoid - xtheta
        theta.data = theta.data + alpha_th*delta*grad_ln_pi.view(1,len(grad_ln_pi))
        

"""
The NN playes one game vs a random player, 
returns the winner.
Used to test whether the NN is improving its game.
"""
def play_a_game_random(commentary = False):
    board = BG.init_board() # initialize the board
    player = np.random.randint(2)*2-1 # which player begins?
    randomPlayer = -1
    while not BG.game_over(board) and not BG.check_for_error(board):
        if commentary: print("lets go player ",player)
        
        # roll dice
        dice = BG.roll_dice()
        if commentary: print("rolled dices:", dice)
            
        # make a move (2 moves if the same number appears on the dice)
        for i in range(1+int(dice[0] == dice[1])):
            board_copy = np.copy(board) 

            if player == randomPlayer:
                move = flipped_agent.action(board_copy,dice,player,i) 
            else:
                move = action(board_copy,dice,player,i)
            
            # update the board
            if len(move) != 0:
                for m in move:
                    board = BG.update_board(board, m, player)
            
            # give status after every move:         
            if commentary: 
                print("move from player",player,":")
                BG.pretty_print(board)
                
        # players take turns 
        player = -player
            
    # return the winner
    return -1*player



# Global variables
n = 29
nodes = 400
device = torch.device('cpu')
# cuda will only create a significant speedup for large/deep networks and batched training
#device = torch.device('cuda') 
# randomly initialized weights, with zeros for the biases, for the critic
w1 = Variable(torch.randn(nodes,7*(n-1)*2, device = device, dtype=torch.float), requires_grad = True)
b1 = Variable(torch.zeros(nodes,1, device = device, dtype=torch.float), requires_grad = True)
w2 = Variable(torch.randn(1,nodes, device = device, dtype=torch.float), requires_grad = True)
b2 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)

#randomly initialized weights, with zeros for the biases, for the actor
theta = torch.ones((1,nodes), device = device, dtype=torch.float)

alpha_w = 0.1 # step sizes using for the neural network (first layer)
alpha_th = 0.05 # (second layer)
lam_w = 0.4 # lambda parameter in TD(lam-bda)
lam_th = 0.4

# compete for "competition_games" vs a random player, 
# then train for "training_steps" using self-play.
for i in range(0,20):
    start = time.time()
    wins_for_player_1 = 0
    loss_for_player_1 = 0
    competition_games = 1
    for j in range(competition_games):
        winner = play_a_game_random(commentary = False)
        if (winner == 1):
            wins_for_player_1 += 1.0
        else:
            loss_for_player_1 += 1.0

    end = time.time()
    print(end - start)
    print(wins_for_player_1, loss_for_player_1)
    
    start = time.time()
    training_steps = 2000
    learnit(training_steps, lam_w,lam_th, alpha_w, alpha_th)
    end = time.time()
    print(end - start)
    
