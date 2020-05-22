#import package yang diperlukan
import numpy as np
import pandas as pd
import random as rd
from random import randint
import matplotlib.pyplot as plt #untuk grafik

# masukan kapasitas tas
max_tas = int(input('muatan maksimal tas : '))
# masukan banyak barang tersedia
n_barang = int(input('jumlah barang : '))
# no urut barang
barang_no = np.arange(1,n_barang + 1)
berat=[]
profit=[]
for i in range(n_barang):
    print("barang ke-",i+1)
    x = int(input('masukkan bobot / berat barang  : '))
    berat.append(x)
    y = int(input('Masukkan value / profit barang  : '))
    profit.append(y)
print('list barang')
print('No.barang   berat   profit')
for i in range(barang_no.shape[0]):
    print('{0}          {1}         {2}\n'.format(barang_no[i], berat[i], profit[i]))

#pembentukakn populasi awal
populasi = n_barang *2
ukuran_populasi = (populasi, barang_no.shape[0]) #ukuran matriks
print('ukuran populasi = {}'.format(ukuran_populasi))
populasi_awal = np.random.randint(2, size = ukuran_populasi)
populasi_awal = populasi_awal.astype(int)
jumlah_gen = 100
print('populasi awal: \n{}'.format(populasi_awal))

#fungsi untuk menghitung nilai fitness
def get_fitness(berat,  profit, populasi, max_tas):
    fitness = np.empty(populasi.shape[0])
    for i in range(populasi.shape[0]):
        profit_value = np.sum(populasi[i] * profit)
        berat_value = np.sum(populasi[i] * berat)
        if berat_value <= max_tas:
            #jika berat kurang dari batas maksimal tas
            fitness[i] = profit_value
        else :
            fitness[i] = 0
    return fitness.astype(int)

#fungsi seleksi
def selection(fitness, no_parent, populasi):
    fitness = list(fitness)
    parent = np.empty((no_parent, populasi.shape[1]))
    for i in range(no_parent):
        index_maxfitness = np.where(fitness == np.max(fitness)) #index fitness dengan nilai terbesar
        parent[i,:] = populasi[index_maxfitness[0][0], :]
        fitness[index_maxfitness[0][0]] = -999999
    return parent

#fungsi crossover
def crossover(parent, no_offspring):
    offspring = np.empty((no_offspring, parent.shape[1]))
    titik_cross = int(parent.shape[1]/2)
    rate = 0.8
    i=0
    while (parent.shape[0] < no_offspring):
        index_parent1 = i%parent.shape[0]
        index_parent2 = (i+1)%parent.shape[0]
        x = rd.random()
        if x > rate:
            continue
        index_parent1 = i%parent.shape[0]
        index_parent2 = (i+1)%parent.shape[0]
        offspring[i,0:titik_cross] = parent[index_parent1,0:titik_cross]
        offspring[i,titik_cross:] = parent[index_parent2,titik_cross:]
        i=+1
    return offspring

#fungsi mutasi
def mutation(offspring):
    mutant = np.empty((offspring.shape))
    rate = 0.4
    for i in range(mutant.shape[0]):
        nilai_random = rd.random()
        mutant[i,:] = offspring[i,:]
        if nilai_random > rate:
            continue
        new_nilai_random = randint(0,offspring.shape[1]-1)
        if mutant[i,new_nilai_random] == 0 :
            mutant[i,new_nilai_random] = 1
        else :
            mutant[i,new_nilai_random] = 0
    return mutant

#fungsi optimize
def optimize(berat, profit, populasi, ukuran_populasi, no_generasi, max_tas):
    parameter, catatan_fitness = [], []
    no_parent = int(ukuran_populasi[0] / 2)
    no_offspring = ukuran_populasi[0] - no_parent
    for i in range(no_generasi):
        fitness = get_fitness(berat, profit, populasi, max_tas)
        catatan_fitness.append(fitness)
        parent = selection(fitness, no_parent, populasi)
        offspring = crossover(parent, no_offspring)
        mutant = mutation(offspring)
        populasi[0:parent.shape[0], :] = parent
        populasi[parent.shape[0]:, :] = mutant

    print('generasi terakhir: \n{}\n'.format(populasi))
    fitness_gen_terakhir = get_fitness(berat, profit, populasi, max_tas)
    print('fitness gen terakhir: \n{}\n'.format(fitness_gen_terakhir))
    max_fitness = np.where(fitness_gen_terakhir == np.max(fitness_gen_terakhir))
    parameter.append(populasi[max_fitness[0][0], :])
    return parameter, catatan_fitness

#main
parameter, fitness_history = optimize(berat, profit, populasi_awal, ukuran_populasi, jumlah_gen, max_tas)
print('Optimized parameter  \n{}'.format(parameter))
selected_items = barang_no * parameter
print('\nItem terpilih untuk dimasukkan ke dalam tas : ')
for i in range(selected_items.shape[1]):
  if selected_items[0][i] != 0:
     print('{}\n'.format(selected_items[0][i]))


#plot
fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
fitness_history_max = [np.max(fitness) for fitness in fitness_history]
plt.plot(list(range(jumlah_gen)), fitness_history_mean, label = 'Mean Fitness')
plt.plot(list(range(jumlah_gen)), fitness_history_max, label = 'Max Fitness')
plt.legend()
plt.title('Fitness through the generations')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.show()
print(np.asarray(fitness_history).shape)