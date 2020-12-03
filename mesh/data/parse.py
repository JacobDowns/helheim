import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.interpolate import interp1d
import dill as dill
import json
from scipy.signal import savgol_filter
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull
  
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
"""
## Name:DomainOutline 
## Icon:0 
# Points Count  Value 
5 1.000000 
# X pos Y pos 
"""

with open('helheim.json') as json_file:
    data = np.array(json.load(json_file))

    #hull = ConvexHull(data)
    #plt.plot(data[hull.vertices,0], data[hull.vertices,1], 'r--', lw=2)
    #plt.show()

    #data =np.delete(data, -3, axis = 0)

    print(data.shape)

    #i = 21
    #x0 = data[i,0]
    #y0 = data[i,1]
    #x1 = data[i+1,0]
    #y1 = data[i+1,1]

    #print(x0)
    #data[i+1,0] = x0

    plt.plot(data[:,0], data[:,1], 'ko-')
    plt.show()

    """

    #data[i+1,0] = x0
    #print(x0)

    plt.plot(data[:,0], data[:,1], 'ko-')
    dx = x1 - x0
    dy = y1 - y0

    x = np.linspace(x0, x1, 100)
    y = (dy / dx)*(x - x0) + y0

    
    print(dy / dx)
    print(x0, y0)

    plt.plot(x, y, 'ro-')
    plt.show()
    """
    
    #print([data[i,0], data[i+1,0]], [data[i,1], data[i+1,1]])
    #plt.plot([data[i,0], data[i+1,0]], [data[i,1], data[i+1,1]], 'ko')
    #plt.show()
    #quit()

    
    with open("helheim_domain.exp", "ab") as f:
        f.write(b"## Name:DomainOutline\n")
        f.write(b"## Icon:0\n")
        f.write(b"# Points Count  Value\n")
        f.write(b"44 1.000000\n")
        f.write(b"# X pos Y pos\n")
        np.savetxt(f, data)
    
