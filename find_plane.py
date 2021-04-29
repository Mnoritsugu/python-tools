import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D

def create_sample(nv, d, N):
  """ nv: np.array of (a, b, c), |nv| = 1 """
  """ nv*r + d = 0 """
  # dxyzはノイズ
  dxyz = np.random.random(3*N).reshape(3, N)*0.05
  x = np.random.random(N)*2 - 1.0 + dxyz[0, :]
  y = np.random.random(N)*2 - 1.0 + dxyz[1, :]
  z = - (x*nv[0] + y*nv[1] + d)/nv[2] + dxyz[2, :]
  return (x, y, z)

def print_xyz_meshlab(xs, ys, zs):
  for v in zip(xs, ys, zs):
    print("{} {} {}".format(*v))

def make_plane(p):
  v1 = p[1] - p[0]
  v2 = p[2] - p[0]
  norm = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
  d = np.inner(norm, p[0]) * -1
  return norm, d

def calc_distance_plane2point(norm, d, p):
  return abs(np.inner(norm, p) + d)/np.linalg.norm(norm)

def find_plane(xs, ys, zs):
  r = np.c_[xs, ys, zs]
  # サンプル点群の重心、x, y, zの３つの成分
  c = np.mean(r, axis=0)
  # サンプル点群の重心を原点に移動
  r0 = r - c
  # SVD分解
  u, s, v = np.linalg.svd(r0)
  # sの最小値に対応するベクトルを取り出す
  nv = v[-1, :]
  # サンプル点群の平面と原点との距離
  ds = np.dot(r, nv)
  param = np.r_[nv, -np.mean(ds)]
  print('SVD result: ', param)

def find_plane_ransac(xs, ys, zs):
  r = np.c_[xs, ys, zs]

  iteration = 10000
  inliner_thread = 0.01
  max_inliner = 0

  for i in range(iteration):
    if i > 1 and i%2000 ==0:
      print("i = ", i)
    # ランダムに抽出した点から平面を決定する
    random_r = r[np.random.choice(len(r), 3)]
    norm, d = make_plane(random_r)
    
    inliner = 0
    
    for j in range(len(r)):
      # print("j = ", j)
      # インライアーの数を計算
      dist = calc_distance_plane2point(norm, d, r[j])
      if dist < inliner_thread:
        inliner += 1
    
    if inliner > max_inliner:
      max_inliner = inliner
      plane_param = np.append(norm, d)
  
  print("max inliner: ", max_inliner)
  print("ransac result: ", plane_param)

  return plane_param

def main():
  phi0 = random.random()*np.pi
  theta0 = random.random()*np.pi
  v0 = np.array([math.sin(phi0)*math.cos(theta0), math.sin(phi0)*math.sin(theta0), math.cos(phi0)])
  d0 = random.random()*10.0
  xs, ys, zs = create_sample(v0, d0, 2000)
  print("{} x + {} y + {} z + {} = 0".format(*v0, d0))
  print("----")

  # 可視化
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.set_box_aspect((1, 1, 1))

  ax.scatter(xs, ys, zs, s=10, color='b')
  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")

  # plt.show()

  find_plane(xs, ys, zs)
  plane_param = find_plane_ransac(xs, ys, zs)

  # 予測平面をプロットする
  p_param = 5
  p_base = np.array([[p_param, p_param, 0],
                     [-p_param, p_param, 0],
                     [-p_param, -p_param, 0],
                     [p_param, -p_param, 0]])
  
  axis = np.cross(np.array([0.0, 0.0, 1.0]), plane_param[:3]) / np.linalg.norm(np.cross(np.array([0.0, 0.0, 1.0]), plane_param[:3]))
  angle = math.acos(np.inner(np.array([0.0, 0.0, 1.0]), plane_param[:3]) / (np.linalg.norm(plane_param[:3])))

  # print(angle)
  # print(axis)

  rot = Rotation.from_rotvec(axis*angle)
  # print(calc_distance_plane2point(plane_param[:3], plane_param[3], np.zeros(3)))
  # t = calc_distance_plane2point(plane_param[:3], plane_param[3], np.zeros(3)) * plane_param[:3]
  t = plane_param[:3] * plane_param[3] * -1
  p_base_plot = (np.dot(rot.as_matrix(), p_base.T)).T + np.tile(t, (4, 1))
  wire_def = np.array([[p_base_plot[0], p_base_plot[1], p_base_plot[2], p_base_plot[3], p_base_plot[0]]])
  wire = ax.plot_wireframe(wire_def[:, :, 0], wire_def[:, :, 1], wire_def[:, :, 2], color='r')

  plt.show()

if __name__ == '__main__':
  main()