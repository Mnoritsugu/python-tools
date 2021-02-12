import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
# %matplotlib inline
# 法線ベクトル及びオフセット値

def sample():
  A = 0.0; B = 0.0; C = 1.0; D = -1.0
  # point cloud 作成
  points = []
  for x in np.arange(0, 10, 0.1):
    for y in np.arange(0, 10, 0.1):
      for z in np.arange(0, 10, 0.1):
        if(A*x + B*y + C*z + D ==0):
          points.append([x, y, z])
  points = np.asarray(points).T
  # points = np.asarray(points).T
  # グラフ表示
  fig = plt.figure()
  ax = fig.add_subplot(111, projection = '3d')
  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")
  ax.plot(points[0],points[1],points[2])
  plt.show()

def main():
  #xy平面を作成
  x = 2 * np.random.rand(1000) - 1
  y = 2 * np.random.rand(1000) - 1
  z = 0

  size = len(x)
  origin_face = np.zeros([size, 3])

  for i in range(size):
    origin_face[i] = np.array([x[i], y[i], z])
  
  # fig = plt.figure()
  # ax = fig.add_subplot(111, projection = '3d')
  # ax.set_xlabel("X")
  # ax.set_ylabel("Y")
  # ax.set_zlabel("Z")
  # ax.scatter3D(origin_face[:, 0], origin_face[:, 1], origin_face[:, 2])
  # plt.show()

  #座標変換
  axis = np.array([1, 1, 1])
  theta = math.radians(45)
  rotvec = axis * theta / np.linalg.norm(axis * theta)
  t = np.array([0, 0, 1])

  R = Rotation.from_rotvec(rotvec)
  face = np.dot(R.as_matrix(), origin_face.T) + np.tile(np.array(t), (1000, 1)).T
  face = face.T

  face += np.random.normal(0, 0.05, (1000, 3))

  fig = plt.figure()
  ax = fig.add_subplot(111, projection = '3d')
  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")
  ax.scatter3D(face[:, 0], face[:, 1], face[:, 2], color = 'red')
  # plt.show()

  pred_plane = ransac(face)

  # fig = plt.figure(figsize = (8, 6))
  # ax = fig.add_subplot(111, projection='3d')

  # x,y,z軸の範囲
  xr = [-1, 1]
  yr = [-1, 1]
  zr = [-1, 1]

  plot_plane(ax, pred_plane, xr, yr, zr)
  plt.show()

def ransac(points,
            n = 5,
            k = 100,
            t = 0.15,
            d = 500):
  good_models = []
  good_model_errors = []
  iterations = 0
  max_num_inliner = 0
  while iterations < k:
      sample = points[np.random.choice(len(points), n)]
      param = face_detect(sample)

      num_inliner = 0
      for i in range(len(points)):
        dist = calc_distance(points[i], param)
        if dist <= t:
          num_inliner += 1
      
      if max_num_inliner < num_inliner:
        good_model = param
        max_num_inliner = num_inliner

      if max_num_inliner > d:
        break

      iterations += 1
  
  print(max_num_inliner)

  return good_model

def face_detect(data):
  #dataから最小二乗法で平面を近似する
  A = data
  b = data[:, 2] * -1
  for i in range(len(A)):
    A[i, 2] = 1
  
  temp = np.dot(A.T, A)
  temp = np.dot(np.linalg.inv(temp), A.T)
  x = np.dot(temp, b)

  return np.array([x[0], x[1], 1, x[2]])

def calc_distance(point, param):
  #点と平面の距離を計算する
  return np.linalg.norm(point[0]*param[0] + point[1]*param[1] + point[2]*param[2] + param[3]) / np.linalg.norm(param[:3])

def plot_plane(axes, param, xrange, yrange, zrange,
               pcolor="blue", alpha=0.5):
    # px+qy+rz+s=0をプロットする関数
    # axes：サブプロット
    # param：p,q,r,sのリストまたはタプルなど
    # xrange,yrange,zrange：x軸,y軸,z軸の範囲
    # pcolor,alpha：平面の色,透明度

    # 軸ラベルの設定
    axes.set_xlabel("x", fontsize = 16)
    axes.set_ylabel("y", fontsize = 16)
    axes.set_zlabel("z", fontsize = 16)

    # 軸範囲の設定
    axes.set_xlim(xrange[0], xrange[1])
    axes.set_ylim(yrange[0], yrange[1])
    axes.set_zlim(zrange[0], zrange[1])

    # 格子点の作成
    x = np.arange(xrange[0], xrange[1], 0.2)
    y = np.arange(yrange[0], yrange[1], 0.2)
    xx, yy = np.meshgrid(x, y)

    # 平面の方程式
    zz = -(param[0]*xx + param[1]*yy + param[3]) / param[2]

    # 平面をプロット
    axes.plot_surface(xx, yy, zz, color=pcolor, alpha=alpha)

def test():
  x = 2 * np.random.rand(1000) - 1
  y = 2 * np.random.rand(1000) - 1
  z = 1

  size = len(x)
  origin_face = np.zeros([size, 3])

  for i in range(size):
    origin_face[i] = np.array([x[i], y[i], z])

  param = face_detect(origin_face)
  # param = [face[0], face[1], 1, face[2]]

  # FigureとAxes
  fig = plt.figure(figsize = (8, 6))
  ax = fig.add_subplot(111, projection='3d')

  # x,y,z軸の範囲
  xr = [-5, 5]
  yr = [-5, 5]
  zr = [-5, 5]

  # p,q,r,sを設定
  # param = [2, -1, 3, 0]

  # 平面2x-y+3z=0をプロット
  plot_plane(ax, param, xr, yr, zr)
  plt.show()

  print(calc_distance(np.array([0, 0, 0]), param))

if __name__ == "__main__":
  main()
  # test()