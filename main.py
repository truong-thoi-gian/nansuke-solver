import copy
import getopt
import math
import numpy as np
import pycosat
import random
import sys
import time
import matplotlib.pyplot as plt

def main(argv):
  try:
    opts, args = getopt.getopt(argv, '', ["size="])
  except getopt.GetoptError:
    print('Argument error, check -h | --help')
    sys.exit(2)

  type = "binomial"
  for opt, arg in opts: 
    if opt in ("--size"):
      size = int(arg)
  
  xAxis = {'var':[],'clauses':[],'size': []}
  yAxis = {'time': [], 'clauses': []}
  random_c = 10
  false_list = []
  for size in range(10,25):
    print("start",size)
    list_c = []
    list_l = []
    list_t = []
    for t in range(0,1):
      for i in range(1,random_c + 1):
        while True:
          a = random_array(size, t)
          tr, inputs, puzzle = validate_array(a, size)
          if tr == True:
            break
        #print("Questions\n", puzzle, "\nNumbers:\n", inputs)
        resc,time,c,l = solve_sat_problem(puzzle, type, size, inputs)
        list_c.append(c)
        list_l.append(l)
        list_t.append(time)
        #print("Compare:\n", np.array(a), "\n",np.array_equiv(np.array(a),resc))
        if np.array_equiv(np.array(a),resc) == False:
          false_list.append((a,resc,inputs))
    xAxis['var'].append(int(sum(list_l)/random_c))
    xAxis['size'].append(size)
    xAxis['clauses'].append(int(sum(list_c)/random_c))
    yAxis['time'].append(int(sum(list_t)/random_c))
    yAxis['clauses'].append(int(sum(list_c)/random_c))
    
  
  figure, axis = plt.subplots(1, 1)
  
  #axis[0].plot(xAxis['var'], yAxis['time'])
  #axis[0].set_title("Time with var")

  # For Clause with var
  axis.plot(xAxis['size'], yAxis['time'])
  axis.set_title("Time with size")
  if len(false_list) > 0:
    print("failed\n")
    #for i,j,k in false_list:
    #  print(np.array(i),"\n",j,"\n",k)
  plt.show()


def generate_sample(size,t):
  while True:
    a = random_array(size, t)
    tr, inputs, puzzle = validate_array(a, size)
    if tr == True:
      break
  return (puzzle, inputs)

def generate_sat_result(puzzle, type, size, inputs):
  resc,time,c,l = solve_sat_problem(puzzle, type, size, inputs)
  return (resc, time, c, l)
    
            
def help():
  print('Usage:')
  print('main.py --size=3')
  sys.exit()
def append_list(results, item):
  if len(item) > 1:
      string_ints = [str(int) for int in item]
      results.append(int(''.join(string_ints)))
  if len(results) != len(list(set(results))):
    print("failed", results)
    return (False, list(set(results)))
  print("true", results)
  return (True, list(set(results)))
# Validate if data is valid for nansuke problem rules
def validate_array(a, siz):
  results = []
  ac = np.array(a)
  for i in range(0, siz):
    row_num = []
    col_num = []
    for j in range(0, siz):
      if a[i][j] > 0:
        row_num.append(a[i][j])
        ac[i][j] = 1
      else:
        check,results = append_list(results,row_num)
        if check == False:
          return (False, list(set(results)), ac)
        row_num = []
      if a[j][i] > 0:
        col_num.append(a[j][i])
        ac[j][i] = 1
      else:
        check,results = append_list(results,col_num)
        if check == False:
          return (False, list(set(results)), ac)
        col_num = []
    check,results = append_list(results,row_num)
    if check == False:
          return (False, list(set(results)), ac)
    check,results = append_list(results,col_num)
    if check == False:
          return (False, list(set(results)), ac)
  # check if duplicated number (> 9)
  if len(results) == 0:
    return (False, list(set(results)), ac)
  return (True, list(set(results)), ac)
   
# Random problem test data
def random_array(siz, t = -1):
  ca = np.random.rand(siz, siz).reshape(1, siz * siz)
  for i in range(0, siz * siz):
    ca[0][i] = random.randint(1, 9)
  if t < 0:
    hint = random.randint(1, siz * siz - 1)
  else:
    hint = t
  list = random.sample(range(1, siz * siz - 1), hint)
  for h in list:
    ca[0][h] = 0

  ca = ca.reshape(siz, siz).astype(int).tolist()
  return ca

def convert_array(problemset):
  new = np.array(problemset.board)
  new[new == None] = 0
  new.astype(int)
  return new.tolist()
  
def solve_sat_problem(problemset, type, size, inputs):
  result = problemset
  resu = solve(result, type, size, inputs) 
  #print('Answer:\n', resu)
  return resu
    
def v(i, j, d, size, data=[]): 
  sat_size = size
  return sat_size * 9 * (i - 1) + 9 * (j - 1) + d

def get_string(n, k):
  res_s = []
  for item in n:
    res_s.append(str(item)[k])
  return list(set(res_s))

def filter_string(n, k, index, value):
  res_s = []
  for item in n:
    if str(item)[index] == str(value):
      res_s.append(str(item)[k])
  return list(set(res_s))

def nansuke_clauses_binomial(data, inputs, size):
  res = []
  sat_size = int(size)
  # for all cells, ensure that the each cell:
  for i in range(1, sat_size + 1):
    for j in range(1, sat_size + 1):
      # denotes (at least) one of the 9 digits (1 clause)
      if (data[i - 1][j - 1] > 0):
        res.append([v(i, j, d, size, data) for d in range(1, 10)])
        # does not denote two different digits at once (9 clauses)
        for d in range(1, 10):
          for dp in range(d + 1, 10):
            res.append([-v(i, j, d, size, data), -v(i, j, dp, size, data)])
                      
                      
  #process { 'inputs': [1982,1657], 'pos': [[(1,1),(1,2),(1,3),(1,4)],[(1,1),(2,1),(3,1),(4,1)]]]}
  def valid(item, length): 
    inputs = list(set(item['inputs']))
    pos = item['pos']
    for r in range(0, length):
      char_list = get_string(inputs, r)
      for k in pos:
        h, t = k[r]
        new_c = []
        for c in char_list:
          new_c.append(v(h, t, int(c), size))
          for c_i in range(r, length):
            c_relate = filter_string(inputs, c_i, r, c)
            new_c_r = [-v(h, t, int(c), size)]
            for c_r in c_relate:
              hr, tr = k[c_i]
              new_c_r.append(v(hr, tr, int(c_r), size))
              
            res.append(new_c_r)
        res.append(new_c)
    #print("res",res)
    for input in inputs:
      for r in range(0, length):
        new_c = []
        for ki in range(0, len(pos)):
          k = pos[ki]
          h, t = k[r]
          new_c.append(v(h, t, int(str(input)[r]), size))
        res.append(new_c)
    # Do not need to valid duplicate for only one digits
    if length == 1:
      return
    # Valid dupliate number
    for input in inputs:
      for ki in range(0, len(pos)):
        k = pos[ki]
        new_c = []
        for r in range(0, length):
          h, t = k[r]
          new_c.append(-v(h, t, int(str(input)[r]), size))
        for hi in range(ki + 1, len(pos)):
          kh = pos[hi]
          new_c_r = copy.copy(new_c)
          for r in range(0, length):
            hr, tr = kh[r]
            new_c_r.append(-v(hr, tr, int(str(input)[r]), size))
          res.append(new_c_r)
    
  # construct problem into format {1: { inputs: [1,2], pos: [[(1,1),(2,2)]]}}                  
  all_data = {}
  # init and update 'inputs' list
  for input in inputs:
    number_length = math.ceil(math.log10(input))
    if input == 1:
      number_length = 1
    if number_length not in all_data:
      all_data[number_length] = {'inputs': [], 'pos': []}
    all_data[number_length]['inputs'].append(input)
      
  # update 'pos' list
  for i in range(1, sat_size + 1):
    # Update row
    row_sum = 0
    row_tmp = []
    for j in range(1, sat_size + 1):
      if data[i - 1][j - 1] > 0:
        row_tmp.append((i, j))
        row_sum += 1
      else:
        if row_sum > 1 and row_tmp not in all_data[row_sum]['pos']:
          all_data[row_sum]['pos'].append(row_tmp)
        row_sum = 0
        row_tmp = []
    # check if cell is existed in list
    if row_sum > 1 and row_tmp not in all_data[row_sum]['pos']:
      all_data[row_sum]['pos'].append(row_tmp)
          
    # Update column
    col_sum = 0
    col_tmp = []
    for j in range(1, sat_size + 1):
      if data[j - 1][i - 1] > 0:
        col_tmp.append((j, i))
        col_sum += 1
      else:
        if col_sum > 1 and col_tmp not in all_data[col_sum]['pos']:
          all_data[col_sum]['pos'].append(col_tmp)
        col_sum = 0
        col_tmp = []
    # check if cell is existed in list
    if col_sum > 1 and col_tmp not in all_data[col_sum]['pos']:
      all_data[col_sum]['pos'].append(col_tmp)
  #print(all_data)
  # ensure each row, column is filled by prepared number
  for data_length, data_item in all_data.items():
    valid(data_item, data_length)
  return res
#Reduces Nansuke problem to a SAT clauses 
def nansuke_clauses(type, size, data, inputs): 
  prepare_type = 'nansuke_clauses_' + type
  return globals()[prepare_type](data, inputs, size)

def solve(grid, type, size, inputs):
  #solve a Nansuke problem
  clauses = nansuke_clauses(type, size, grid, inputs)
  sat_size = int(size)
    
  # Print number SAT clause  
  numclause = len(clauses)
  resc = []
  for clause in clauses:
    for i in clause:
      resc.append(np.abs(i))
  resc = list(set(resc))
  #print("P CNF number of clauses: " + str(numclause) + " ; number of variables: ", len(resc))
    
  # solve the SAT problem
  start = time.time()
  sol = set(pycosat.solve(clauses))
  end = time.time()
  #print("Time: " + str(end - start))
    
  def read_cell(i, j, size, grid):
    # return the digit of cell i, j according to the solution
    for d in range(1, 10):
      if v(i, j, d, size, grid) in sol:
        return d
    return 0
    
  results = np.array(grid)
  for i in range(1, sat_size + 1):
    for j in range(1, sat_size + 1):
      results[i - 1][j - 1] = read_cell(i, j, size, grid)
    
  return (results, int(1000*(end - start)), numclause, len(resc))


if __name__ == '__main__':
  from pprint import pprint
    
  if(len(sys.argv[1:]) == 0):
    print('Argument error, check --help')
  else:
    main(sys.argv[1:])
