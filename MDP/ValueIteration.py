# -*- coding: utf-8 -*-
import numpy

def sum_values(matrix, end_states):
    dims = matrix.shape
    sum_total = 0
    for i in range(1,dims[0]-1):
        for j in range(1,dims[1]-1):
            if end_states[i-1,j-1] != 1 and matrix[i,j] < 100:
                sum_total+=matrix[i,j]
    return sum_total


environment = numpy.array([[0, 0, 0, 1],
                           [0, 0, 0, -1],
                           [0, 0, 0, 0,]])
    
walls = numpy.array([[100, 100, 100, 100, 100, 100],
                     [100, 0, 0, 0, 1, 100],
                     [100, 0, 100, 0, -1, 100],
                     [100, 0, 0, 0, 0, 100], 
                     [100, 100, 100, 100, 100, 100]], numpy.float32)
    
end_states = numpy.matrix([[0, 0, 0, 1],
                           [0, 0, 0, 1], 
                           [0, 0, 0, 0]])
    
action_action_probabilities = numpy.matrix([[0.8, 0, 0.1, 0.1],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]],numpy.float32)

#actions = ['north', 'south', 'east', 'west']

previous_value = sum_values(walls, end_states)
current_value = 100
policy = []
gamma = 0.9
epsilon = 0.01
dims_env = walls.shape


while abs(current_value-previous_value) > epsilon:
    previous_value = current_value
    for row in range(1, dims_env[0]-1):
        for col in range(1, dims_env[1]-1):
            action_values = [0]*4
            if end_states[row-1,col-1] != 1 and walls[row,col] < 100:
                action_values[0] = action_action_probabilities[0,0]*gamma*walls[row-1 if walls[row-1 , col] < 100 else row, col] +\
                                    action_action_probabilities[0,1]*gamma*walls[row+1 if walls[row+1,col] < 100 else row, col]+\
                                    action_action_probabilities[0,2]*gamma*walls[row, col+1 if walls[row, col+1] < 100 else col]+\
                                    action_action_probabilities[0,3]*gamma*walls[row,col-1 if walls[row,col-1] < 100 else col]
            
                action_values[1] = action_action_probabilities[1,0]*gamma*walls[row-1 if walls[row-1 , col] < 100 else row, col] +\
                                    action_action_probabilities[1,1]*gamma*walls[row+1 if walls[row+1,col] < 100 else row, col]+\
                                    action_action_probabilities[1,2]*gamma*walls[row, col+1 if walls[row, col+1] < 100 else col]+\
                                    action_action_probabilities[1,3]*gamma*walls[row,col-1 if walls[row,col-1] < 100 else col]
                                    
                action_values[2] = action_action_probabilities[2,0]*gamma*walls[row-1 if walls[row-1 , col] < 100 else row, col] +\
                                    action_action_probabilities[2,1]*gamma*walls[row+1 if walls[row+1,col] < 100 else row, col]+\
                                    action_action_probabilities[2,2]*gamma*walls[row, col+1 if walls[row, col+1] < 100 else col]+\
                                    action_action_probabilities[2,3]*gamma*walls[row,col-1 if walls[row,col-1] < 100 else col]
                                    
                action_values[3] = action_action_probabilities[3,0]*gamma*walls[row-1 if walls[row-1 , col] < 100 else row, col] +\
                                    action_action_probabilities[3,1]*gamma*walls[row+1 if walls[row+1,col] < 100 else row, col]+\
                                    action_action_probabilities[3,2]*gamma*walls[row, col+1 if walls[row, col+1] < 100 else col]+\
                                    action_action_probabilities[3,3]*gamma*walls[row,col-1 if walls[row,col-1] < 100 else col]
                walls[row,col] = max(action_values)
                
    current_value = sum_values(walls, end_states)     