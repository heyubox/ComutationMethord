from memory_profiler import profile

# matrix operations
@profile
def mul_matrix(MA,MB):
  if len(MA[0])!=len(MB):
    print('dim error')
    return
  return [[sum(map(lambda a: a[0]*a[1], zip(l, s))) for l in zip(*MB)] for s in MA ]


def naive_swap(M,row_zero_index):
  swapA = M[row_zero_index]
  # 需要与index下的非0行交换
  for i in range(row_zero_index+1,len(M)):
    if M[i][row_zero_index]!=0:
      return swap(M,i,row_zero_index)
  print("不满足初等变换条件")
  return[[]]

# @profile  
def transpose(M):
  Mt = []
  for i in zip(*M):
    Mt.append(list(i))
  return Mt


# row operations
def sub(rowA,rowB):
  return [i-j for i,j in zip(rowA,rowB)]
def add(rowA,rowB):
  return [i+j for i,j in zip(rowA,rowB)]
def div(rowA,rowB):
  return [i/j for i,j in zip(rowA,rowB)]
def times(rowA,rowB):
  return [i*j for i,j in zip(rowA,rowB)]
def div_dig(row,dig):
  dig_l = [dig]*len(row)
  return div(row,dig_l)
def times_dig(row,dig):
  dig_l = [dig]*len(row)
  return times(row,dig_l)

# @profile
def naive_tran(rowA,rowB,main_dig):
  '''
  @rowA updaterowA
  @rowB 主元
  @main_dig主元位置
  '''
  rowA_dig = rowA[main_dig]
  rowB_dig = rowB[main_dig]
  if rowA_dig == 0:
    return rowA
  fac = rowB_dig/rowA_dig
  rowA_update = sub(rowB.copy(),times_dig(rowA.copy(),fac))
  return rowA_update

def swap(A,i,j):
  tmp = A[i]
  A[i] = A[j]
  A[j] = tmp
  return A


# 求逆
@profile
def r(M):
  resM = M.copy()
  row_element = len(M[0])
  for index in range(len(M)):
    new_row = [0]*row_element
    new_row[index] = 1
    resM[index] = resM[index]+new_row
  for main_i in range(len(M)):
    main_row = resM[main_i].copy()
    diag_dig = resM[main_i][main_i]
    for tra in range(len(M)):
      if tra == main_i:
        continue
      if diag_dig == 0:
        resM = naive_swap(resM,main_i)
        if resM == [[]]:
          print("不满足求逆条件")
          return [[]]
      tran_row = resM[tra].copy()
      resM[tra]=naive_tran(main_row,tran_row,main_i)
    resM[main_i] = div_dig(resM[main_i],resM[main_i][main_i])
  return [row[row_element:] for row in resM] 

def try_float(x):
    try:
        return float(x)
    except ValueError:
        return x

f="winequality-red.csv"
@profile
def load_file(file_name):
  import csv
  readin = []
  with open(file_name) as csvfile:
      spamreader = csv.reader(csvfile, delimiter=';')
      for row in spamreader:
        readin.append([try_float(i) for i in row])
  readin = readin[1:]
  return readin
readin = load_file(f)

def into_group(data_X,data_Y,group_num):
  num_per_group = int(len(data_X)/group_num)
  x=[]
  y=[]
  for i in range(group_num):
    x.append(data_X[i*num_per_group:(i+1)*num_per_group])
    y.append(data_Y[i*num_per_group:(i+1)*num_per_group])
  return x,y
# @profile
def regression_func(coff,X,y):
  # X y are batch
  tar = transpose(y.copy())[0]
  pred = []
  for i in X:
    pred.append(sum(times(coff,i)))
  sqrt_error = [ (t-p)**2 for t,p in zip(tar,pred)]
  error = sum(sqrt_error)/len(sqrt_error)
  return error, pred
@profile
def train_regression(batch_X,batch_y,musk):
  test_X = None
  test_y = None
  train_X = []
  train_y = []
  index = -1
  for x,y in zip(batch_X,batch_y):
    index+=1
    if musk[index] == 0:
      test_X = x 
      test_y = y 
      continue
    else:
      train_X+=x
      train_y+=y
  theta = transpose(mul_matrix(mul_matrix(r(mul_matrix(transpose(train_X),train_X)),transpose(train_X)),train_y))[0]
  # train_error
  train_error,_ = regression_func(theta,train_X,train_y)
  # test_error
  test_error,_ = regression_func(theta,test_X,test_y)
  return train_error,test_error

@profile
def main():
  data_X = [ [1]+row[:-1] for row in readin]
  data_y = [ [row[-1]] for row in readin]

  batch_X,batch_y = into_group(data_X,data_y,5)

  train_error,test_error = train_regression(batch_X,batch_y,musk=[1,1,1,0,1])

  import time
  # import matplotlib.pyplot as plt
  train_errors=[]
  test_errors=[]
  time_start=time.time()

  for musk_test in range(1):
    musk = [1]*5
    musk[musk_test]=0
    train_error,test_error = train_regression(batch_X,batch_y,musk)
    train_errors.append(train_error)
    test_errors.append(test_error)

  time_end=time.time()

  print('avg-time cost',(time_end-time_start)/5,'s')

if __name__ == "__main__":
  main()

  # plt.plot(train_errors,'-^')
  # plt.plot(test_errors,'-o')
  # plt.legend(['train_errors','test_errors'])
  # plt.show()

