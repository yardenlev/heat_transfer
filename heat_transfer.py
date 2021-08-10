'''
Author: Yarden Levenberg
Email: Yarden.lev@gmail.com
-------------------------------------------------------
Short Description:
This is a solution for the heat equation using implicit method
on a XY-plane [m]
-------------------------------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt
import math


# physical consts
s = 0.00625  # = sigma_x = sigma_y [m]
k = 1.786 * (10**(-3))  # [m^2 / s]
range_box = [0, 1.5]  # [m] same for x and y
h = 0.05  # step size [m]
dt = 0.1  # [s]
end_t = 60  # [s]
a = (h**2) / (k * dt)  # alpha
n = int((range_box[1] - range_box[0]) / h) - 1

# initial conditions
T_t0 = 10  # [°c]
# boundary condition for 60 seconds
T_y0 = 100  # [°c]
T_y1_5 = 10  # [°c]
# other variables
map_c = 'coolwarm'  # c-map colors min(blue-white-red)max


# F(x,y,t) gaussian heat sink
def f(x, y, t):
    return -(10**(-4)) * math.exp(-((x-1)**2)/(2 * s**2)) * math.exp(-0.1 * t) * math.exp(-((y-0.5)**2)/(2 * s**2))


# this function returns two matrix that you can multiply to return to the original matrix
# those matrix are lower triangle and upper triangle
def lu_upper_lower(mat):
    rank = len(mat)
    lower = np.zeros((rank, rank))
    upper = np.zeros((rank, rank))
    for j in range(rank):
        for i in range(0, j + 1):
            sum = 0
            for k in range(i):
                sum += (lower[i][k] * upper[k][j])
            upper[i][j] = mat[i][j] - sum
        for i in range(j, rank):
            if i == j:
                lower[i][i] = 1
            else:
                sum = 0
                for k in range(j):
                    sum += (lower[i][k] * upper[k][j])
                lower[i][j] = float((mat[i][j] - sum) / upper[j][j])
    return upper, lower


# this function calculate the answer of the lower & upper matrices and b vector and returns a new solution vector
def lu_decomposition_answer(upper, lower, b_vector):
    rank = len(upper)
    y = []
    for i in range(rank):
        y.append([0] * len(upper[0]))
    y[0] = b_vector[0] / lower[0][0]
    for i in range(1, rank):
        mini_sum = 0
        for j in range(0, i):
            mini_sum += lower[i][j] * y[j]
        y[i] = (b_vector[i] - mini_sum) / lower[i][i]
    x = [0 for i in range(rank)]
    x[rank - 1] = y[rank - 1] / upper[rank - 1][rank - 1]
    for k in range(rank - 2, -1, -1):
        mini_sum_2 = 0
        for p in range(k + 1, rank):
            mini_sum_2 += upper[k][p] * x[p]
        x[k] = (y[k] - mini_sum_2) / upper[k][k]
    return x


# this function create the matrix that would be used to solve the question using the LUD method in an implicit way
def create_matrix_for_lu():
    matrix = np.zeros((n**2, n**2))
    np.fill_diagonal(matrix, 4 + a)
    for i in range(len(matrix)-1):
        if i < len(matrix) - n:
            matrix[i][i + n] = -1
            matrix[i + n][i] = -1
        if (i+1) % n != 0:
            matrix[i][i+1] = -1
            matrix[i+1][i] = -1
    return matrix


# this function create a b vector using the boundary conditions of the question
def create_b_vector(matrix, t):
    b = np.zeros((n**2))
    p = 0
    for i in range(1, n+1):
        for j in range(1, n+1):
            b[p] = a * matrix[i][j] + a * dt * f(i*h, j*h, t)
            if i == 1:
                b[p] += matrix[0][j]
            if i == n:
                b[p] += matrix[i+1][j]
            if j == 1:
                b[p] += matrix[i][0]
            if j == n:
                b[p] += matrix[i][j+1]
            p += 1
    return b


# this function create the boundary initial conditions for this questions
def initial_conditions():
    matrix = np.full((n+2, n+2), T_t0, dtype=float)
    matrix[:, -1] = [100 - (60 * (i * h)) for i in range(len(matrix[:, -1]))]
    matrix[:int(0.8/h), 0] = [100 - (112.5 * (i * h)) for i in range(int(0.8/h))]
    matrix[int(0.8/h):, 0] = 10
    matrix[:][0] = T_y0
    matrix[:][-1] = T_y1_5
    return matrix


# update matrix with the solution vector
def update_matrix(matrix, vector):
    p = 0
    for i in range(1, n+1):
        for j in range(1, n+1):
            matrix[i][j] = vector[p]
            p += 1
    return matrix


# plot boundaries of the model
def plot_formatting():
    plt.rcParams["figure.figsize"] = (7, 7)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.grid(b=True, which='major', color='gray', linestyle='--')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    plt.title("Heat equation in using implicit method")
    ax.set_xlim((range_box[0], range_box[1]))
    ax.set_ylim((range_box[1], range_box[0]))


# this is main
def main():
    heat_matrix = initial_conditions()
    lu_matrix = create_matrix_for_lu()
    upper, lower = lu_upper_lower(lu_matrix)
    print("LU matrix complete")
    time_series = np.around(np.arange(dt, end_t + dt, dt), decimals=int(dt**-1))
    for t in time_series:
        print("t = ", t)
        b = create_b_vector(heat_matrix, t)
        temperature_vector = lu_decomposition_answer(upper, lower, b)
        heat_matrix = update_matrix(heat_matrix, temperature_vector)
        if t % 1 == 0:
            plot_formatting()
            ims = plt.imshow(np.flip(heat_matrix, 0), extent=(range_box[0], range_box[1], range_box[0], range_box[1]), cmap=map_c,
                             vmin=10, vmax=100)
            plt.text(range_box[0]+0.05, range_box[1]-0.05, "t =" + str(t) + " [s]")
            plt.colorbar(ims)
            # plt.savefig('heat2D_t='+ str(t) + ".png")
            plt.show()
            # plt.close()


main()


