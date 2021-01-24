
import copy
import math

import numpy as np


def main():
  # csv読み込み
  data_dir = 'C:/Users/Noritsugu/Desktop/PythonStudy/'
  import_data_path = data_dir + 'test.csv'

  raw_data = np.loadtxt(import_data_path, delimiter=',', encoding='utf-8_sig')

  # インデックス行を追加
  result_data = np.zeros([len(raw_data), 10])
  
  for i in range(len(raw_data)):
    result_data[i, 0] = i + 1

  result_data[:, 1:7] = copy.deepcopy(raw_data)
  # 距離を算出
  for i in range(len(raw_data)):
    result_data[i, 7] = np.nan
    result_data[i, 8] = math.sqrt(result_data[i, 1]**2 + result_data[i, 2]**2 + result_data[i, 3]**2)
    result_data[i, 9] = math.sqrt(result_data[i, 4]**2 + result_data[i, 5]**2 + result_data[i, 6]**2)

  # 別名で保存
  data_save_path = data_dir + 'test_result.csv'
  np.savetxt(data_save_path, result_data, delimiter=',')

if __name__ == '__main__':
  main()