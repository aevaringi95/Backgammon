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
n = 27
device = torch.device('cpu')
# cuda will only create a significant speedup for large/deep networks and batched training
# device = torch.device('cuda') 
# randomly initialized weights with zeros for the biases
w1 = Variable(torch.randn(n*n,2*n, device = device, dtype=torch.float), requires_grad = True)
b1 = Variable(torch.zeros((n*n,1), device = device, dtype=torch.float), requires_grad = True)
w2 = Variable(torch.randn(1,n*n, device = device, dtype=torch.float), requires_grad = True)
b2 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)



def ice_hot_encoding(board):
    ice_hot = np.zeros( 2 * len(board) )
    ice_hot[np.where(board == 1)[0] ] = 1
    ice_hot[len(board) + np.where(board == -1)[0] ] = 1
    return ice_hot

def action(board_copy,dice,player,i):
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    possible_moves, possible_boards = BG.legal_moves(board_copy, dice, player)
    na = np.size(possible_moves)
    va = np.zeros(na)
    for board in possible_boards:
        # encode the board to create the input
        x = Variable(torch.tensor(ice_hot_encoding(board), dtype = torch.float, device = device)).view(2 * n,1)
        # now do a forward pass to evaluate the board's after-state value
        h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
        va[i] = y.sigmoid()
    


    # if there are no moves available
    if len(possible_moves) == 0: 
        return [] 
    
    # make the best move according to the policy
    
    # policy missing, returns a random move for the time being
    #
    #
    #
    #
    #
    move = possible_moves[np.random.randint(len(possible_moves))]

    return move

def feed_forward(board):
    return 0

def learnit(numgames, lam, alpha, V, alpha1, alpha2):
    gamma = 1 # for completeness
    # play numgames games for training
    for games in range(0, numgames):
        board = BG.init_board() # initialize the board
        player = np.random.randint(2)*2-1 # which player begins?
    
        # we will use TD(lambda) and so we need to use eligibility traces
        S = [] # no after-state for table V, visited after-states is an empty list
        E = np.array([]) # eligibility traces for table V
        # now we initilize all the eligibility traces for the neural network
        Z_w1 = torch.zeros(w1.size(), device = device, dtype = torch.float)
        Z_b1 = torch.zeros(b1.size(), device = device, dtype = torch.float)
        Z_w2 = torch.zeros(w2.size(), device = device, dtype = torch.float)
        Z_b2 = torch.zeros(b2.size(), device = device, dtype = torch.float)
        
        while not BG.game_over(board) and not BG.check_for_error(board):
            dice = BG.roll_dice()
            
            for i in range(1+int(dice[0] == dice[1])):
                action1 = action(np.copy(board), dice, player, i)
                
                if len(move) != 0:
                for m in move:
                    board = update_board(board, m, player)
            ##########################################################
                # perform move and update board
                board[action] = player
            if (1 == iswin(board, player)): # has this player won?
                winner = player
                break # bail out of inner game loop
            # once both player have performed at least one move we can start doing updates
            if (1 < move):
                if tableplayer == player: # here we have player 1 updating the table V
                    s = hashit(board) # get index to table for this new board
                    delta = 0 + gamma * V[s] - V[sold]
                    E = np.append(E,1) # add trace to this state (note all new states are unique else we would +1)
                    S.append(sold)     # keep track of this state also
                    V[S] = V[S] + delta * alpha * E # the usual tabular TD(lambda) update
                    E = gamma * lam * E
                else: # here we have player 2 updating the neural-network (2 layer feed forward with Sigmoid units)
                    x = Variable(torch.tensor(one_hot_encoding(board, player), dtype = torch.float, device = device)).view(2*9,1)
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
            if tableplayer == player:
                sold = hashit(board)
            else:
                xold = Variable(torch.tensor(one_hot_encoding(board, player), dtype=torch.float, device = device)).view(2*9,1)
                # swap players
                player = getotherplayer(player)

        # The game epsiode has ended and we know the outcome of the game, and can find the terminal rewards
        if winner == tableplayer:
            reward = 0
        elif winner == getotherplayer(tableplayer):
            reward = 1
        else:
            reward = 0.5
        # Now we perform the final update (terminal after-state value is zero)
        # these are basically the same updates as in the inner loop but for the final-after-states (sold and xold)
        # first for the table (note if reward is 0 this player actually won!):
        delta = (1.0 - reward) + gamma * 0 - V[sold]
        E = np.append(E,1) # add one to the trace (recall unique states)
        S.append(sold)
        V[S] = V[S] + delta * alpha * E
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
        
        
    ###############################################################
    
    board = init_board() # initialize the board
    player = np.random.randint(2)*2-1 # which player begins?
    
    # play on
    while not game_over(board) and not check_for_error(board):
        if commentary: print("lets go player ",player)
        
        # roll dice
        dice = roll_dice()
        if commentary: print("rolled dices:", dice)
            
        # make a move (2 moves if the same number appears on the dice)
        for i in range(1+int(dice[0] == dice[1])):
            board_copy = np.copy(board) 

            # make the move (agent vs agent):
            move = agent.action(board_copy,dice,player,i) 
            
            # if you're playing vs random agent:
#            if player == 1:
#                move = agent.action(board_copy,dice,player,i)
#            elif player == -1:
#                move = random_agent(board_copy,dice,player,i) 
            
            # update the board
            if len(move) != 0:
                for m in move:
                    board = update_board(board, m, player)
            
            # give status after every move:         
            if commentary: 
                print("move from player",player,":")
                pretty_print(board)
                
        # players take turns 
        player = -player
            