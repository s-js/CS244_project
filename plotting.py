import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


#4a
ex = "4a"
parameters = [
    {
        'xs': np.array(list(range(3, 8)))/5,
        'ys': np.load(ex+"_Blue.npy"),
        'label': '3:1 Port-ratio',
        'color': 'blue',
        'marker': 'x'
    },    
    {   
        'xs': np.array(list(range(4, 9)))/5,
        'ys': np.load(ex+"_Red.npy"),
        'label': '2:1 Port-ratio',
        'color': 'red',
        'marker': '+'
    },
    {
        'xs': np.array(list(range(3, 9)))/5,
        'ys': np.load("4b_Green.npy"),
        'label': '3:2 Port-ratio',
        'color': 'green',
        'marker': 'o'
    }
]


plt.figure()
for i in range(len(parameters)):
    if parameters[i]['ys'] is None:
        continue
    plt.plot(parameters[i]['xs'], parameters[i]['ys'], marker=parameters[i]['marker'],
             label=parameters[i]['label'], color=parameters[i]['color'])
plt.grid(linestyle='--', color='lightgray')
plt.ylabel('Throughput\n(Ratio to Upper-bound)')
plt.xlabel(
    'Number of Servers at Large Switches \n (Ratio to Expected Under Random Distribution)')
plt.legend(loc='upper right')
plt.savefig(ex+".svg")
plt.show()




x_axis = np.array(list(range(3, 9)))/5
#4b
ex="4b"
parameters=[
        {
        'ys': np.load(ex+"_Blue.npy"),
        'label': '20 smaller switches',
        'color': 'blue',
        'marker': 'x'
     } ,   
     {    

        'ys': np.load(ex+"_Red.npy"),
        'label': '30 smaller switches',
        'color': 'red',
        'marker': '+'
     },
    {
        'ys': np.load(ex+"_Green.npy"),
        'label': '40 smaller switches',
        'color': 'green',
        'marker': 'o'
     }

]


plt.figure()
for i in range(len(parameters)):
    if parameters[i]['ys'] is None:
        continue
    plt.plot(x_axis, parameters[i]['ys'], marker=parameters[i]['marker'],
                label=parameters[i]['label'], color=parameters[i]['color'])
plt.grid(linestyle='--', color='lightgray')
plt.ylabel('Throughput\n(Ratio to Upper-bound)')
plt.xlabel(
    'Number of Servers at Large Switches \n (Ratio to Expected Under Random Distribution)')
plt.legend(loc='upper right')
plt.savefig(ex+".svg")
plt.show()

#4c
ex = "4c"
parameters = [
    {
        'ys': np.load(ex+"_Red.npy"),
        'label': '480 Servers',
        'color': 'red',
        'marker': '+'
    },
    {
        'ys': np.load("4b_Red.npy"),
        'label': '510 Servers',
        'color': 'green',
        'marker': 'o'
    },
    {
        'ys': np.load(ex+"_Blue.npy"),
        'label': '540 Servers',
        'color': 'blue',
        'marker': 'x'
    }
]


plt.figure()
for i in range(len(parameters)):
    if parameters[i]['ys'] is None:
        continue
    plt.plot(x_axis, parameters[i]['ys'], marker=parameters[i]['marker'],
             label=parameters[i]['label'], color=parameters[i]['color'])
plt.grid(linestyle='--', color='lightgray')
plt.ylabel('Throughput\n(Ratio to Upper-bound)')
plt.xlabel(
    'Number of Servers at Large Switches \n (Ratio to Expected Under Random Distribution)')
plt.legend(loc='upper right')
plt.savefig(ex+".svg")
plt.show()


#6c
ex = "6c"
parameters = [
    {
        'xs': np.array(list(range(1, 8)))/5,
        'ys': np.load(ex+"_Blue.npy"),
        'label': '300 Servers',
        'color': 'blue',
        'marker': 'x'
    },     
    {
        'xs': np.array(list(range(2, 7)))/5,
        'ys': np.load(ex+"_Green.npy"),
        'label': '500 Servers',
        'color': 'green',
        'marker': 'o'
    },    

    {
        'xs': np.array(list(range(2, 7)))/5,
        'ys': np.load(ex+"_Red.npy"),
        'label': '700 Servers',
        'color': 'red',
        'marker': '+'
    }
]


plt.figure()
for i in range(len(parameters)):
    if parameters[i]['ys'] is None:
        continue
    plt.plot(parameters[i]['xs'], parameters[i]['ys'], marker=parameters[i]['marker'],
         label=parameters[i]['label'], color=parameters[i]['color'])
plt.grid(linestyle='--', color='lightgray')
plt.ylabel('Throughput\n(Ratio to Upper-bound)')
plt.xlabel(
    'Cross-cluster Links \n (Ratio to Expected Under Random Distribution)')
plt.legend(loc='upper right')
plt.savefig(ex+".png")
plt.show()

