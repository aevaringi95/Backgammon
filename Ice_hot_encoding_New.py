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
