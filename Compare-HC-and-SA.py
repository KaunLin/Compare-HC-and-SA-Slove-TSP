
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import sys
import time
import math
import random
import itertools
import pandas as pd
import time
import matplotlib.pyplot as plt


# In[2]:


def readfile(dic):
    with open('C:/Users/qscf6/Desktop/eil51.tsp') as f:
        r = f.read()
        read_line = r.split('\n')               
        for i in range(len(read_line)):         
            read_element = read_line[i].split()
            dic[int(read_element[0])] = [int(read_element[1])]
            dic[int(read_element[0])].append(int(read_element[2]))
        f.close()


# In[3]:


#產生隨機順序的初始 sequence
def initial_sequence(num):
    seq = [n for n in range(1,num + 1)]
    random.shuffle(seq)
    
    return seq

#依照搜尋半徑,隨機產生 swap case
#Main idea:利用 shuffle 達到隨機與省略"重複判斷"
def random_swap_case(domain,neighbors_size,number_of_case):
    case = []
    #產生 1~domain 的 list
    temp = [n for n in range(1,domain+1)]
    
    for i in range(number_of_case):
        random.shuffle(temp)
        case.append([temp[j] for j in range(neighbors_size)])
   
    return case

#依照 Case 取得原 sequence 內 Index
def get_position(seq,case):
    position = []
    
    for i in range(len(case)):
        position.append(seq.index(case[i]))
        
    return position

#依照該次的 case 交換
def swap_by_case(seq,case,position):
    temp = seq[:]
    
    for i in range(len(case)):
        temp[position[i]] = case[i]
        
    return temp

#因是Symmetic，所以先把各城鎮距離算出來，省下每次重新計算的時間
def calculate_distance_table(dic):
    dx = 0
    dy = 0
    distance_table = []
    for i in range(1,len(dic) + 1):
        temp = [0] * 51
        for j in range(i,len(dic) + 1):
            dx = dic[i][0] - dic[j][0]
            dy = dic[i][1] - dic[j][1]
            temp[j - 1] = math.sqrt(dx**2 + dy**2)
            
        distance_table.append(temp)
        
    return distance_table
            

#計算該 seqence 總路徑長(利用查表)
def sequence_total_distance(seq,distance_table):
    dist = 0
    index1 = 0
    index2 = 0
    for i in range(len(seq)):
        if seq[i] > seq[(i + 1) % len(seq)]:
            index1 = seq[(i + 1) % len(seq)] - 1
            index2 = seq[i] - 1
        else:
            index1 = seq[i] - 1
            index2 = seq[(i + 1) % len(seq)] - 1
        
        dist += distance_table[index1][index2]
        #dist += distance_table[seq[i] - 1][seq[(i + 1) % len(seq)] - 1]
        
    return dist
    
def determine_for_SA(neighbors,current_sequence,shortest_sequence,current_temperature,distance_table):
    index = random.randint(0,len(neighbors) - 1)
    position = get_position(current_sequence,neighbors[index])
    random.shuffle(neighbors[index])
    temp = swap_by_case(current_sequence,neighbors[index],position)
    
    value = sequence_total_distance(temp,distance_table) - sequence_total_distance(current_sequence,distance_table)
    
    if value <= 0:
        current_sequence = temp
        shortest_sequence = current_sequence[:]
        
    else:
        r = random.random()
        if math.exp((-10) * value / current_temperature) >= r:
            current_sequence = temp
    
    return current_sequence,shortest_sequence
def determine_for_HC(neighbor_sequence,shortest_sequence,distance_table):
    if sequence_total_distance(neighbor_sequence,distance_table) < sequence_total_distance(shortest_sequence,distance_table):
        shortest_sequence = neighbor_sequence[:]
        
    return shortest_sequence


# In[4]:


def Simulated_Annealing(number_of_samples,size_of_neighbor,start_temperature,end_temperature,decrease_ratio,distance_table):    
    result = []
    history = []
    historysequence = []
    
    for current_samples in range(number_of_samples):
        current_sequence = initial_sequence(len(distance_table))
        neighbors = []
        
        current_temperature = start_temperature
        
        shortest_sequence = []
        
        count = 0
        
        while current_temperature > end_temperature:
            best_value = sequence_total_distance(shortest_sequence,distance_table)
            neighbors = random_swap_case(len(distance_table),size_of_neighbor,10)
            
            current_sequence,shortest_sequence = determine_for_SA(neighbors,current_sequence,shortest_sequence,current_temperature,distance_table)
            
            history.append(sequence_total_distance(current_sequence,distance_table))
            historysequence.append(shortest_sequence)
                
            if sequence_total_distance(current_sequence,distance_table) >= best_value:
                count += 1
            
            else:
                count = 0
                
            if count == 10:
                current_temperature *= decrease_ratio
                count = 0
                
        result.append(sequence_total_distance(current_sequence,distance_table))
                
    return result,history,historysequence

def Hill_Climbing(number_of_samples,size_of_neighbors,distance_table):
    print("開始時間:",time.localtime(time.time()))
    result = []
    history = []
    historysequence = []
    
    for current_sample in range(number_of_samples):
        shortest_sequence = initial_sequence(len(distance_table))

        shortest_in_neighbors = shortest_sequence[:]
    
        while True:
            for combination in itertools.combinations([n for n in range(1,len(distance_table) + 1)],size_of_neighbors):
                position = get_position(shortest_sequence,combination)
                for case in itertools.permutations(combination):
                    temp_neighbor_sequence = swap_by_case(shortest_sequence,case,position)
                    
                    shortest_in_neighbors = determine_for_HC(temp_neighbor_sequence,shortest_in_neighbors,distance_table)
                    
            if sequence_total_distance(shortest_in_neighbors,distance_table) >= sequence_total_distance(shortest_sequence,distance_table):
                history.append(sequence_total_distance(shortest_sequence,distance_table))
                historysequence.append(shortest_sequence)
                break
            
            else:
                shortest_sequence = shortest_in_neighbors[:]
                history.append(sequence_total_distance(shortest_sequence,distance_table))
                historysequence.append(shortest_sequence)
                
        result.append(sequence_total_distance(shortest_sequence,distance_table))
    
    print("結束時間:",time.localtime(time.time()))
    
    return result,history,historysequence


# In[5]:


dic = {}
readfile(dic)

distance_table = []
distance_table = calculate_distance_table(dic)
SAresult,SAhistory,SAhistorysequence = Simulated_Annealing(1,2,100,10,0.99,distance_table)
HCresult,HChistory,HChistorysequence = Hill_Climbing(1,2,distance_table)
print(len(SAhistorysequence))


# In[6]:



def SA_animate(i):
    plt.subplot(2,1,2)
    plt.cla()
    plt.title('sa')
    x = []
    y = []
    for j in range (1,len(dic)+2):
        x.append(dic[SAhistorysequence[i][j%51]][0])
        y.append(dic[SAhistorysequence[i][j%51]][1])
    line, = plt.plot(x,y,".")
    line, = plt.plot(x, y, color='r')
    return line,

def HC_animate(i):
    plt.subplot(2,1,1)
    plt.cla()
    plt.title('hc')
    x = []
    y = []
    for j in range (1,len(dic)+2):
        x.append(dic[HChistorysequence[i][j%51]][0])
        y.append(dic[HChistorysequence[i][j%51]][1])
    line, = plt.plot(x,y,".")
    line, = plt.plot(x, y, color='r')
    return line,


# ## 偶爾會壞掉(兩個只有第一個會run的情況)，要在多測試幾次就會好了(兩個都在run的情況)

# In[7]:



an = []
from matplotlib.animation import FuncAnimation
fig = plt.figure(figsize=(8,10))

an.append( FuncAnimation(fig = fig,func = HC_animate,frames = len(HChistorysequence),interval = 1,blit = True))


an.append( FuncAnimation(fig = fig,func = SA_animate,frames = len(SAhistorysequence),interval = 1,blit = True)) 
plt.show()

