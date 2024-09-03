import numpy as np
import matplotlib.pyplot as plt
import scipy
import os

U = np.load("U_sim.npy")
V = np.load("V_sim.npy")

x_dim = U.shape[0]
y_dim = U.shape[1]
ntsteps = U.shape[2]

data = np.zeros(shape=(ntsteps-2,2,2,x_dim,y_dim)) #ndataset, input/output pair, U/V,x,y

t = 0
for i in range(2,ntsteps):
  #Inputs
  data[t,0,0] = U[:,:,i-1]
  data[t,0,1] = V[:,:,i-1]

  #outputs
  data[t,1,0] = U[:,:,i]
  data[t,1,1] = V[:,:,i]

  print(t,i,i-1)
  t+=1

x = data[:,0,0,:,:]
y = data[:,1,0,:,:]

# Select the first entry of x and y
x_entry = x[22,:,:]
y_entry = y[22,:,:]
print(x_entry[200,50],y_entry[200,50],U[200,50,23],U[200,50,24])

# Create a meshgrid for x and y coordinates
x_coords = np.linspace(0, 4, x_entry.shape[0])
y_coords = np.linspace(0, 1, x_entry.shape[1])

X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

# Create a figure with 2 subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 2))
levels = np.linspace(0,1,10)
# Plot the first entry of x
contourf1 = axs[0].contourf(X, Y, x_entry)
fig.colorbar(contourf1, ax=axs[0])
axs[0].set_title('x[0]')
axs[0].axis('equal')

# Plot the first entry of y
contourf2 = axs[1].contourf(X, Y, y_entry)
fig.colorbar(contourf2, ax=axs[1])
axs[1].set_title('y[0]')
axs[1].axis('equal')

# Display the figure
plt.savefig("sanity_check.png")

np.save("data.npy", data)
print("Data saved. Shape:", data.shape)
print("Max and min values:", np.max(data), np.min(data))