import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from math import sin, cos
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
import copy

def generateVecFromEllipse(a, b, n):
  
    t = np.linspace(0, 2 * math.pi, 100) 
    t = np.reshape(t, (t.shape[0], 1))
    
    xVec = np.zeros((t.shape))
    yVec = np.zeros((t.shape))
    for i in range(t.shape[0]):
        xVec[i] = (a + n * np.random.randn()) * math.cos(t[i])
        yVec[i] = (b + n * np.random.randn()) * math.sin(t[i])
    
    data = np.concatenate((xVec, yVec),  axis=1)
    return data
  
def plotData(dataOrg):
  
    fig, ax = plt.subplots(ncols = 1, figsize=(5, 5))
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    ax.plot(dataOrg[:, 0], dataOrg[:, 1])
    plt.show()

def calcRigidTranformation(MatA, MatB):
    A, B = np.copy(MatA), np.copy(MatB)

    centroid_A = np.mean(A, axis=0, dtype=np.float)
    centroid_B = np.mean(B, axis=0, dtype=np.float)

    A -= centroid_A
    B -= centroid_B

    H = np.dot(A.T, B)
    U, S, V = np.linalg.svd(H)
    R = np.dot(V.T, U.T)
    T = np.dot(-R, centroid_A) + centroid_B

    return R, T

class ICP(object):
    def __init__(self, pointsA, pointsB):
        self.pointsA = pointsA
        self.pointsB = pointsB
        self.kdtree = KDTree(self.pointsA)

    def calculate(self, iter):
        old_points = np.copy(self.pointsB)
        new_points = np.copy(self.pointsB)

        for i in range(iter):
            neighbor_idx = self.kdtree.query(old_points)[1]
            targets = self.pointsA[neighbor_idx]
            R, T = calcRigidTranformation(old_points, targets)
            new_points = np.dot(R, old_points.T).T + T

            if  np.sum(np.abs(old_points - new_points)) < 0.000000001:
                break

            old_points = np.copy(new_points)

        print("R is {}\n".format(R))
        print("T is {}\n".format(T))
        return new_points

def icp_test(x, y):
    Y, X = np.mgrid[0:100:5, 0:100:5]
    Z = Y ** 2 + X ** 2
    # A = np.vstack([Y.reshape(-1), X.reshape(-1), Z.reshape(-1)]).T
    A = x

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot(A[:,1], A[:,0], A[:,2], "o", color="#cccccc", ms=4, mew=0.5)
    # plt.show()

    R = np.array([
        [cos(50), -sin(50), 0],
        [sin(50), cos(50), 0],
        [0.0, 0.0, 1.0]
    ])

    T = np.array([5.0, 20.0, 10.0])
    # B = np.dot(R, A.T).T + T
    B = y

    icp = ICP(A, B)
    points = icp.calculate(3000)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_label("x - axis")
    ax.set_label("y - axis")
    ax.set_label("z - axis")

    ax.plot(A[:,1], A[:,0], A[:,2], "o", c="b", ms=4, mew=0.5)
    ax.plot(points[:,1], points[:,0], points[:,2], "o", c="g", ms=4, mew=0.5)
    ax.plot(B[:,0], B[:,1], B[:,2], "o", c="r", ms=4, mew=0.5)

    plt.show()

if __name__ == "__main__":
    print("Ellipse Drawing Sample")
    
    data_true = generateVecFromEllipse(5, 2, 0)
    data_val = generateVecFromEllipse(5, 2, 0.2)
    data_3d_true = np.zeros([100, 3])
    data_3d_valid = np.zeros([100, 3])

    data_3d_true[:, :2] = data_true
    data_3d_valid[:, :2] = data_val
    data_3d_valid[:, 2] = 0.5 * np.random.randn(100)

    rot_vec = 2 * np.random.random_sample(3) - 1
    rot_vec /= np.linalg.norm(rot_vec)

    rot = Rotation.from_rotvec(rot_vec)

    data_3d_valid_trans = (np.dot(rot.as_matrix(), data_3d_valid.T)).T

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(data_3d_true[:, 0], data_3d_true[:, 1], data_3d_true[:, 2], c="b")
    # ax.scatter(data_3d_valid_trans[:, 0], data_3d_valid_trans[:, 1], data_3d_valid_trans[:, 2], c="r")
    # ax.set_xlim(-6, 6)
    # ax.set_ylim(-6, 6)
    # ax.set_zlim(-6, 6)

    # plt.show()

    icp_test(data_3d_true, data_3d_valid_trans)