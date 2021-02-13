#レーベンバーグ・マーカート法を利用して最適値を探索する

import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

from flat_detect import calc_distance, plot_plane

def calc_j(param, point, u):
  param[3] = u
  return calc_distance(point, param)**2 / 2

def main():
  #XY平面とZ軸上の点が与えられたとき，XY平面を平行移動させてZ軸上の点が平面上に来るようなパラメータを探索する
  plane_param = [0, 0, 1, 0]
  point = np.array([0, 0, 3])

  c = 0.0001
  u = 1
  J = calc_j(plane_param, point, u)

  while True:
    delta_J = point[2] - u
    H = 1
    delta_u = (point[2] - u) / (1 + c)
    u_dash = u + delta_u
    J_dash = calc_j(plane_param, point, u_dash)

    if J_dash > J:
      c = c*10
      J = J_dash
      u = u_dash
    else:
      c = c/10
      J = J_dash
      u = u_dash
    
    if abs(delta_u) < 0.000001:
      break
  
  plane_param[3] = -u
  print(plane_param)

  fig = plt.figure(figsize = (8, 6))
  ax = fig.add_subplot(111, projection='3d')

  xr = [-5, 5]
  yr = [-5, 5]
  zr = [-5, 5]

  plot_plane(ax, plane_param, xr, yr, zr)
  plt.show()

if __name__ == '__main__':
  main()