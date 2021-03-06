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
from random import randrange

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
    ice_hot = np.zeros(8*24+4)
    for i in range(1,(len(board))):
        k = board[i].astype(np.int64)
        if i <= 24:
            # if it´s a positive player.
            if(k > 0):
                ice_hot[0 + (i-1)*8] = 1
                if(k > 1):
                    ice_hot[1 + (i-1)*8] = 1
                    if(k>2):
                        ice_hot[2 + (i-1)*8] = 1
                        if(k>3):
                            ice_hot[3 + (i-1)*8] = (k-3)/2
            # if it's a negetive player                
            if(k < 0):
                ice_hot[0 + 4 + (i-1)*8] = 1
                if(k < -1):
                    ice_hot[1 + 4 + (i-1)*8] = 1
                    if(k<-2):
                        ice_hot[2 + 4 + (i-1)*8] = 1
                        if(k<-3):
                            ice_hot[3 + 4 + (i-1)*8] = (-k-3)/2
        elif i == 25:
            ice_hot[0+(i-1)*8] = k/2
        elif i == 26:
            ice_hot[1+(i-2)*8] = -k/2       
        elif i == 27:
            ice_hot[2+(i-3)*8] = k/15          
        elif i == 28:
            ice_hot[3+(i-4)*8] = -k/15
    
    return ice_hot

def action(board_copy,epsilon,dice,player,i):
    if player == -1: 
        board_copy = flip_board(board_copy)
        
    possible_moves, possible_boards = BG.legal_moves(board_copy, dice, player = 1)
    na = len(possible_moves)
    va = np.zeros(na)
    j = 0
    
    # if there are no moves available
    if na == 0: 
        return []
    if (np.random.uniform() < epsilon):
        move = possible_moves[randrange(na)]
        if player == -1: 
            move = flip_move(move)        
        return move
    
    for board in possible_boards:
        # encode the board to create the input
        x = Variable(torch.tensor(ice_hot_encoding(board), dtype = torch.float, device = device)).view(encSize,1)
        # now do a forward pass to evaluate the board's after-state value
        va[j] = feed_forward_w(x)
        j+=1
    move = possible_moves[np.argmax(va)]
    if player == -1: 
        move = flip_move(move)
    return move

# zero the gradients for the critic.
def zero_gradients_critic():
    w3.grad.data.zero_()
    b3.grad.data.zero_()
    w2.grad.data.zero_()
    b2.grad.data.zero_()
    w1.grad.data.zero_()
    b1.grad.data.zero_()
    
# zero the gradients for the actor.
def zero_gradients_actor():
    th2.grad.data.zero_()
    th_b2.grad.data.zero_()
    th1.grad.data.zero_()
    th_b1.grad.data.zero_()
    
# update the eligibility traces
def update_eligibility_w(gamma, lam_w, zw1, zb1, zw2, zb2, zw3, zb3):
    zw3 = gamma * lam_w * zw3 + w3.grad.data
    zb3 = gamma * lam_w * zb3 + b3.grad.data
    zw2 = gamma * lam_w * zw2 + w2.grad.data
    zb2 = gamma * lam_w * zb2 + b2.grad.data
    zw1 = gamma * lam_w * zw1 + w1.grad.data
    zb1 = gamma * lam_w * zb1 + b1.grad.data
    return zw3, zb3, zw2, zb2, zw1, zb1

def update_eligibility_th(gamma, lam_w, zw1, zb1, zw2, zb2,I):
    zw2 = gamma * lam_w * zw2 + I*th2.grad.data
    zb2 = gamma * lam_w * zb2 + I*th_b2.grad.data
    zw1 = gamma * lam_w * zw1 + I*th1.grad.data
    zb1 = gamma * lam_w * zb1 + I*th_b1.grad.data
    return zw2, zb2, zw1, zb1

def feed_forward_w(x):
    h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.sigmoid() # squash this with a sigmoid function
    h2 = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
    h2_sigmoid = h2.sigmoid() # squash the output
    y = torch.mm(w3,h2_sigmoid) + b3 # multiply with the output weights w2 and add bias
    y_sigmoid = y.sigmoid() # squash the output
    return y_sigmoid

def feed_forward_th(x):
    h = torch.mm(th1,x) + th_b1 # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.sigmoid() # squash this with a sigmoid function
    y = torch.mm(th2,h_sigmoid) + th_b2 # multiply with the output weights w2 and add bias
    return y

"""
def TabularDynaQ(gamma, epsilon, alpha, n):
    nw = 6
    mw = 9
    Q = np.zeros((nw+2,mw+2,4))
    A = np.zeros((nw+2,mw+2,4,2), dtype = int)
    B = np.zeros((nw+2,mw+2,4,1), dtype = int)
    Observed = np.zeros((nw+2,mw+2,4), dtype = bool)
    forever = True
    episode = 0
    stepcount = np.zeros(50) # lets keep track of the steps needed to reach our terminal state!
    while forever == True:
        s = np.array([3,1]) # initial state
        sterminal = np.array([1,9]) # the goal state
        while np.all(s==sterminal) == False:
            stepcount[episode] += 1
            a = epsilongreedy(epsilon,s,Q)
            snew, R = takeaction(s,a)
            Q[s[0],s[1],a] += alpha*(R+gamma*np.max(Q[snew[0],snew[1],:])-Q[s[0],s[1],a])
            A[s[0],s[1],a,:] = snew
            B[s[0],s[1],a,0] = R
            Observed[s[0],s[1],a] = True # Bookkeepping, record that we have visited this (state, action) before
            ob = np.where(Observed == True) # which have been observed?
            nob = len(ob[0]) # the number of observations
            skeep = snew # this is missing in the Dyna-Q code p. 164 in book
            akeep = a
            for repeat in range(n): # in search, planning or dreaming mode:
                choice = np.random.choice(range(nob),1)[0]
                s = [ob[0][choice], ob[1][choice]]
                a = ob[2][choice]
                R = B[s[0],s[1],a,0] 
                snew = A[s[0],s[1],a,:]
                Q[s[0],s[1],a] += alpha*(R+gamma*np.max(Q[snew[0],snew[1],:])-Q[s[0],s[1],a])
            s = skeep # restore the true state before planning phase
            a = akeep
        episode += 1 
        if episode >= 50:
            forever = False # nothing is forever :)
    return Q, stepcount
"""

def learnitDyna(numgames, epsilon, lam_w, alpha_w, gamma, numthink):
    A = np.zeros(4)
    for games in range(0, numgames):
        board = BG.init_board() # initialize the board
        player = np.random.randint(2)*2-1 # which player begins?
        count = 0
        delta = 0
        # now we initilize all the eligibility traces for the neural network
        Z_w1 = torch.zeros(w1.size(), device = device, dtype = torch.float)
        Z_b1 = torch.zeros(b1.size(), device = device, dtype = torch.float)
        Z_w2 = torch.zeros(w2.size(), device = device, dtype = torch.float)
        Z_b2 = torch.zeros(b2.size(), device = device, dtype = torch.float)
        Z_w3 = torch.zeros(w3.size(), device = device, dtype = torch.float)
        Z_b3 = torch.zeros(b3.size(), device = device, dtype = torch.float)
        
        Z_w1_flip = torch.zeros(w1.size(), device = device, dtype = torch.float)
        Z_b1_flip = torch.zeros(b1.size(), device = device, dtype = torch.float)
        Z_w2_flip = torch.zeros(w2.size(), device = device, dtype = torch.float)
        Z_b2_flip = torch.zeros(b2.size(), device = device, dtype = torch.float)
        Z_w3_flip = torch.zeros(w3.size(), device = device, dtype = torch.float)
        Z_b3_flip = torch.zeros(b3.size(), device = device, dtype = torch.float)
        
        if games % 100 == 0:
            print(games)
        
        #play a game
        while not BG.game_over(board) and not BG.check_for_error(board):
            dice = BG.roll_dice()
            
            for i in range(1+int(dice[0] == dice[1])):
                move = action(np.copy(board), epsilon, dice, player, i)
                
                if len(move) != 0:
                    for m in move:
                        board = BG.update_board(board, m, player)
                #tvenna og vinnur i fyrri leik. BREAK!!!!
                if BG.game_over(board):
                    break
                        
            if BG.game_over(board):
                winner = player
                break
            
            if player == -1:
                board = flip_board(np.copy(board))
                
                
            if (count > 1):
                if player == -1:
                    #One-hot encoding of the board
                    move_fliptemp=move
                    x_fliptemp = ice_hot_encoding(board)
                    xflip = Variable(torch.tensor(x_fliptemp, dtype = torch.float, device = device)).view(encSize,1)
                    
                    #Feed forward w-nn for old and new
                    target= feed_forward_w(xflip)
                    old_target= feed_forward_w(xflipold)
                    delta = 0 + gamma * target.detach().cpu().numpy() - old_target.detach().cpu().numpy() # this is the usual TD error
                    # using autograd and the contructed computational graph in pytorch compute all gradients
                    old_target.backward()
                    # update the eligibility traces using the gradients
                    Z_w3_flip, Z_b3_flip, Z_w2_flip, Z_b2_flip, Z_w1_flip, Z_b1_flip = update_eligibility_w(gamma, lam_w, Z_w1_flip, Z_b1_flip, Z_w2_flip, Z_b2_flip, Z_w3_flip, Z_b3_flip)
                    # zero the gradients
                    zero_gradients_critic()
                    # perform now the update for the weights
                    delta =  torch.tensor(delta, dtype = torch.float, device = device)
                    w1.data = w1.data + alpha_w * delta * Z_w1_flip
                    b1.data = b1.data + alpha_w * delta * Z_b1_flip
                    w2.data = w2.data + alpha_w * delta * Z_w2_flip
                    b2.data = b2.data + alpha_w * delta * Z_b2_flip
                    w3.data = w3.data + alpha_w * delta * Z_w3_flip
                    b3.data = b3.data + alpha_w * delta * Z_b3_flip
                    # append to the model, for the first time we create A, else we just stack on it. 
                    if count == 2 and games == 0:
                        A = np.array([[x_fliptempold], [move], [x_fliptemp], 0])
                    else:
                        add_to_model = np.array([[x_fliptempold], [move], [x_fliptemp], 0])
                        A = np.vstack((A, add_to_model))                    


                else:
                    #One-hot encoding of the board
                    move_temp = move
                    x_temp = ice_hot_encoding(board)
                    x = Variable(torch.tensor(x_temp, dtype = torch.float, device = device)).view(encSize,1)
                    
                    #Feed forward w-nn for old and new
                    target= feed_forward_w(x)
                    old_target = feed_forward_w(xold)
                    delta = 0 + gamma * target.detach().cpu().numpy() - old_target.detach().cpu().numpy() # this is the usual TD error
                    # using autograd and the contructed computational graph in pytorch compute all gradients
                    old_target.backward()
                    # update the eligibility traces using the gradients
                    Z_w3, Z_b3, Z_w2, Z_b2, Z_w1, Z_b1 = update_eligibility_w(gamma, lam_w, Z_w1, Z_b1, Z_w2, Z_b2, Z_w3, Z_b3)
                    # zero the gradients
                    zero_gradients_critic()
                    # perform now the update for the weights
                    delta =  torch.tensor(delta, dtype = torch.float, device = device)
                    w1.data = w1.data + alpha_w * delta * Z_w1
                    b1.data = b1.data + alpha_w * delta * Z_b1
                    w2.data = w2.data + alpha_w * delta * Z_w2
                    b2.data = b2.data + alpha_w * delta * Z_b2
                    w3.data = w3.data + alpha_w * delta * Z_w3
                    b3.data = b3.data + alpha_w * delta * Z_b3           
                    # append to the model, for the first time we create A, else we just stack on it. 
                    if count == 2 and games == 0:
                        A = np.array([[x_tempold], [move], [x_temp], 0])
                    else:
                        add_to_model = np.array([[x_tempold], [move], [x_temp], 0])
                        A = np.vstack((A, add_to_model))
                
                
                if count > 2:
                    for thought in range(0,numthink):
                        state_indx = np.random.choice(A.shape[0])
                        state, move_temp, statenew, rewardtemp = A[state_indx]
                        
                        if statenew == 0:
                            #Feed forward old state
                            state = Variable(torch.tensor(state, dtype = torch.float, device = device)).view(encSize,1)
                            old_target1 = feed_forward_w(state)
                            delta2 = rewardtemp + 0 - old_target1.detach().cpu().numpy()
                        else: 
                            state = Variable(torch.tensor(state, dtype = torch.float, device = device)).view(encSize,1)
                            statenew = Variable(torch.tensor(statenew, dtype = torch.float, device = device)).view(encSize,1)
                            #Feed forward w-nn
                            target1 = feed_forward_w(statenew)
                            #Feed forward old state
                            old_target1 = feed_forward_w(state)
                            delta2 = 0 + gamma * target1.detach().cpu().numpy() - old_target1.detach().cpu().numpy() # this is the usual TD error
                        
                        # using autograd and the contructed computational graph in pytorch compute all gradients
                        old_target1.backward()
                        # zero the gradients
                        zero_gradients_critic()
                        # perform now the update for the weights
                        delta2 =  torch.tensor(delta2, dtype = torch.float, device = device)
                        w1.data = w1.data + alpha_w * delta2 * w1.grad.data
                        b1.data = b1.data + alpha_w * delta2 * b1.grad.data
                        w2.data = w2.data + alpha_w * delta2 * w2.grad.data
                        b2.data = b2.data + alpha_w * delta2 * b2.grad.data
                        w3.data = w3.data + alpha_w * delta * w3.grad.data
                        b3.data = b3.data + alpha_w * delta * b3.grad.data
                        
                        
                
                
                
            if (count < 2):
                if player == -1:
                    x_fliptempold = ice_hot_encoding(board)
                    xflipold = Variable(torch.tensor(ice_hot_encoding(board), dtype=torch.float, device = device)).view(encSize,1)
                else:
                    x_tempold = ice_hot_encoding(board)
                    xold = Variable(torch.tensor(ice_hot_encoding(board), dtype=torch.float, device = device)).view(encSize,1)
            else:
                if player == -1:
                    x_fliptempold = x_fliptemp
                    xflipold = Variable(torch.tensor(xflip, dtype=torch.float, device = device)).view(encSize,1)
                else:
                    x_tempold = x_temp
                    xold = Variable(torch.tensor(x, dtype=torch.float, device = device)).view(encSize,1)
            
            if player == -1:
                board = flip_board(np.copy(board))
            # swap players
            player = -player
            count += 1
            
            
        if winner==1:
            reward = 1
            reward_flip = 0
            move_temp = move
        else:
            reward = 0
            reward_flip = 1
            move_fliptemp = move

        
       
        
        #update fyrir player 1  
        #Feed forward old state using w-NN
        old_target = feed_forward_w(xold)
        delta = reward + 0 - old_target.detach().cpu().numpy() # this is the usual TD error
        # using autograd and the contructed computational graph in pytorch compute all gradients
        old_target.backward()
        # update the eligibility traces using the gradients
        delta =  torch.tensor(delta, dtype = torch.float, device = device)
        Z_w3, Z_b3, Z_w2, Z_b2, Z_w1, Z_b1 = update_eligibility_w(gamma, lam_w, Z_w1, Z_b1, Z_w2, Z_b2, Z_w3, Z_b3)
        # zero the gradients
        zero_gradients_critic() 
        # perform the update for the weights for the critic, w
        w1.data = w1.data + alpha_w * delta * Z_w1
        b1.data = b1.data + alpha_w * delta * Z_b1
        w2.data = w2.data + alpha_w * delta * Z_w2
        b2.data = b2.data + alpha_w * delta * Z_b2
        w3.data = w3.data + alpha_w * delta * Z_w3
        b3.data = b3.data + alpha_w * delta * Z_b3
        
        add_to_model = np.array([[x_tempold], [move_temp], 0, reward])
        A = np.vstack((A, add_to_model))
        
        
        
        
        #Feed forward old state using w-NN
        flip_target = feed_forward_w(xflipold)
        delta = reward_flip + 0 - flip_target.detach().cpu().numpy() # this is the usual TD error
        # using autograd and the contructed computational graph in pytorch compute all gradients
        flip_target.backward()
        # update the eligibility traces using the gradients
        delta =  torch.tensor(delta, dtype = torch.float, device = device)
        Z_w3_flip, Z_b3_flip, Z_w2_flip, Z_b2_flip, Z_w1_flip, Z_b1_flip = update_eligibility_w(gamma, lam_w, Z_w1_flip, Z_b1_flip, Z_w2_flip, Z_b2_flip, Z_w3_flip, Z_b3_flip)
        # zero the gradients
        zero_gradients_critic() 
        # perform the update for the weights for the critic, w
        w1.data = w1.data + alpha_w * delta * Z_w1_flip
        b1.data = b1.data + alpha_w * delta * Z_b1_flip
        w2.data = w2.data + alpha_w * delta * Z_w2_flip
        b2.data = b2.data + alpha_w * delta * Z_b2_flip
        w3.data = w3.data + alpha_w * delta * Z_w3_flip
        b3.data = b3.data + alpha_w * delta * Z_b3_flip
        
        add_to_model = np.array([[x_fliptempold], [move_fliptemp], 0, reward_flip])
        A = np.vstack((A, add_to_model))

def play_a_game_random(commentary = False):
    board = BG.init_board() # initialize the board
    if commentary: BG.pretty_print(board)
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
                move = action(board_copy, 0, dice, player, i)
            
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
nodes = 100
nodes2 = 80
device = torch.device('cpu')
alpha_w= 0.1
lam_w = 0.7 # lambda parameter in TD(lam-bda)
epsilon = 0
encSize = 8*24+4
numberofthink = 30
gamma=1
# cuda will only create a significant speedup for large/deep networks and batched training
#device = torch.device('cuda') 
# randomly initialized weights with zeros for the biases
competition_games = 50
training_steps = 1000
Load_data = True
loadtrainstep = 92000

# randomly initialized weights, with zeros for the biases, for the critic
if Load_data == True:
    w1 = torch.load('./Data/Dyna/NN/w1_trained_'+str(loadtrainstep)+'.pth')
    w2 = torch.load('./Data/Dyna/NN/w2_trained_'+str(loadtrainstep)+'.pth')
    b1 = torch.load('./Data/Dyna/NN/b1_trained_'+str(loadtrainstep)+'.pth')
    b2 = torch.load('./Data/Dyna/NN/b2_trained_'+str(loadtrainstep)+'.pth')
    b3 = torch.load('./Data/Dyna/NN/b3_trained_'+str(loadtrainstep)+'.pth')
    w3 = torch.load('./Data/Dyna/NN/w3_trained_'+str(loadtrainstep)+'.pth')
    #wins_against_random = np.concatenate(wins_against_random, np.zeros(training_steps), 0)
else:
    loadtrainstep = 0  
    w1 = Variable(0.01*torch.randn(nodes,encSize, device = device, dtype=torch.float), requires_grad = True)
    b1 = Variable(torch.zeros(nodes,1, device = device, dtype=torch.float), requires_grad = True)
    w2 = Variable(0.01*torch.randn(nodes2,nodes, device = device, dtype=torch.float), requires_grad = True)
    b2 = Variable(torch.zeros((nodes2,1), device = device, dtype=torch.float), requires_grad = True)
    w3 = Variable(0.01*torch.randn(1,nodes2, device = device, dtype=torch.float), requires_grad = True)
    b3 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)

for i in range(0,100):
    start = time.time()
    wins_for_player_1 = 0
    loss_for_player_1 = 0
    for j in range(competition_games):
        winner = play_a_game_random(commentary = False)
        if (winner == 1):
            wins_for_player_1 += 1.0
        else:
            loss_for_player_1 += 1.0
    
    end = time.time()
    print(end - start)
    print(wins_for_player_1, loss_for_player_1)
    
    np.save('./Data/Dyna/NN/wins_against_random'+str(loadtrainstep+(i)*training_steps)+'.npy', wins_for_player_1)
    
    start = time.time()
    learnitDyna(training_steps, epsilon, lam_w, alpha_w, gamma, numberofthink)
    end = time.time()
    print(end - start)
    
    torch.save(w1, './Data/Dyna/NN/w1_trained_'+str(loadtrainstep+(i+1)*training_steps)+'.pth')
    torch.save(w2, './Data/Dyna/NN/w2_trained_'+str(loadtrainstep+(i+1)*training_steps)+'.pth')
    torch.save(w3, './Data/Dyna/NN/w3_trained_'+str(loadtrainstep+(i+1)*training_steps)+'.pth')
    torch.save(b1, './Data/Dyna/NN/b1_trained_'+str(loadtrainstep+(i+1)*training_steps)+'.pth')
    torch.save(b2, './Data/Dyna/NN/b2_trained_'+str(loadtrainstep+(i+1)*training_steps)+'.pth')
    torch.save(b3, './Data/Dyna/NN/b3_trained_'+str(loadtrainstep+(i+1)*training_steps)+'.pth')
    


#0.01 alpha
#0.4 Lambda Gekk vel
#n = 30
#Epsilon 0
#3000 leikir. Fór úr 250 í 331 win af 500.
    
    