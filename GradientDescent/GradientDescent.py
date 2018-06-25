# -*- coding: utf-8 -*-

# solve stationary point for f(x) = x^4 - 3x^3 + 5x + 2
# X_n+1 = X_n + gamma*df/dx(X_n)

gamma = 0.01
precision = 0.0001
prev_x = 6
current_x = 0

df = lambda x: 4*x**3 - 9*x**2 + 5

while abs(current_x-prev_x) > precision:
    prev_x = current_x
    current_x = prev_x - gamma*df(prev_x)
    
print(current_x)
