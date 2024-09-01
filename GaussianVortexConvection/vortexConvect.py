import numpy as np
import time
import matplotlib.pyplot as plt
from hydra import initialize, compose
from omegaconf import DictConfig
from numba import jit

def apply_boundary_conditions(bc_type, U, V):
    # Apply inlet Dirichlet boundary condition for both types
    if bc_type == 1:
        print("Boundary Conditions are dirichlet")
        U[-1, :] = 1
        U[:, 0] = 1
        U[:, -1] = 1
        V[0, :] = 0
        V[-1, :] = 0
        V[:, 0] = 0
        V[:, -1] = 0
    elif bc_type == 2:
        print("Boundary Conditions are neumann")
        U[-1, :] = U[-2, :]
        U[:, 0] = U[:, 1]
        U[:, -1] = U[:, -2]
        V[0, :] = 0
        V[-1, :] = V[-2, :]
        V[:, 0] = V[:, 1]
        V[:, -1] = V[:, -2]

    y = np.linspace(0, 1, U.shape[1])  # assuming y ranges from 0 to 1
    frequencies = np.array([0.5, 1])
    amplitudes = np.array([0.5, 0.25])
    phase = (np.pi/180)*np.array([0.0, 90.0])
    U[0, :] = sum(amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * y + phase[i]) for i in range(len(frequencies)))

    return U, V

@jit(nopython=True)
def apply_neumann_bc(U, V, nx, ny):
    # Apply Neumann boundary conditions
    U[-1, :] = U[-2, :]
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]

    V[-1, :] = V[-2, :]
    V[:, 0] = V[:, 1]
    V[:, -1] = V[:, -2]

    return U, V

@jit(nopython=True)
def update_U_V(U, V, Ukp1, Vkp1, Uk, Vk, mu, dt, dx, dy, nx, ny, w):
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            Ukp1[i, j] = (mu * dt / dx**2) * (Ukp1[i+1, j] + Ukp1[i-1, j]) + \
                         (mu * dt / dy**2) * (Ukp1[i, j+1] + Ukp1[i, j-1]) + \
                         U[i, j] - ((U[i, j] * dt) / (2 * dx)) * (U[i+1, j] - U[i-1, j]) - \
                         ((V[i, j] * dt) / (2 * dy)) * (U[i, j+1] - U[i, j-1])
            Ukp1[i, j] = Ukp1[i, j] / (1 + 2 * mu * dt / dx**2 + 2 * mu * dt / dy**2)
            Ukp1[i, j] = Uk[i, j] + w * (Ukp1[i, j] - Uk[i, j])

            Vkp1[i, j] = (mu * dt / dx**2) * (Vkp1[i+1, j] + Vkp1[i-1, j]) + \
                         (mu * dt / dy**2) * (Vkp1[i, j+1] + Vkp1[i, j-1]) + \
                         V[i, j] - ((U[i, j] * dt) / (2 * dx)) * (V[i+1, j] - V[i-1, j]) - \
                         ((V[i, j] * dt) / (2 * dy)) * (V[i, j+1] - V[i, j-1])
            Vkp1[i, j] = Vkp1[i, j] / (1 + 2 * mu * dt / dx**2 + 2 * mu * dt / dy**2)
            Vkp1[i, j] = Vk[i, j] + w * (Vkp1[i, j] - Vk[i, j])

    return Ukp1, Vkp1

@jit(nopython=True)
def calculate_vorticity(vor, V, U, dx, dy, nx, ny):
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            vor[i, j] = ((V[i+1, j] - V[i-1, j]) / (2 * dx)) - ((U[i, j+1] - U[i, j-1]) / (2 * dy))
    return vor


def main(cfg: DictConfig):
    # Extract configuration parameters
    nx = cfg.simulation.domain.nx
    ny = cfg.simulation.domain.ny
    lx = cfg.simulation.domain.lx
    ly = cfg.simulation.domain.ly
    w = cfg.simulation.solver.w
    errormax = cfg.simulation.solver.errormax
    itermax = cfg.simulation.solver.itermax
    ntsteps = cfg.simulation.settings.ntsteps
    dt = cfg.simulation.settings.dt
    mu = cfg.simulation.settings.mu
    bc = cfg.simulation.boundary_conditions.bc

    Vt = cfg.simulation.vortex_variables.Vt
    x0 = cfg.simulation.vortex_variables.x0
    y0 = cfg.simulation.vortex_variables.y0
    r0 = cfg.simulation.vortex_variables.r0

    dx = lx / nx
    dy = ly / ny

    # Allocate arrays
    x = np.zeros(nx)
    y = np.zeros(ny)
    U = np.zeros((nx, ny))
    V = np.zeros((nx, ny))
    Ukp1 = np.zeros((nx, ny))
    Vkp1 = np.zeros((nx, ny))
    vor = np.zeros((nx, ny))

    # Initialize the domain
    x[0] = 0
    y[0] = 0
    for i in range(1, nx):
        x[i] = x[i-1] + dx
    for j in range(1, ny):
        y[j] = y[j-1] + dy

    for j in range(1, ny-1):
        for i in range(1, nx-1):
            r = np.sqrt((x[i] - x0)**2 + (y[j] - y0)**2)
            U[i, j] +=  1 - Vt * (y[j] - y0) * np.exp((1-(r/r0)**2)/2)
            V[i, j] += Vt * (x[i] - x0) * np.exp((1-(r/r0)**2)/2)

    # Initialize the second vortex at (x0 + 0.2, y0 + 0.2)
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            r = np.sqrt((x[i] - (x0 + 0.2))**2 + (y[j] - (y0 + 0.2))**2)
            U[i, j] += - Vt * (y[j] - (y0 + 0.2)) * np.exp((1-(r/r0)**2)/2)
            V[i, j] += Vt * (x[i] - (x0 + 0.2)) * np.exp((1-(r/r0)**2)/2)

    # Initialize the second vortex at (x0 + 0.2, y0 - 0.2)
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            r = np.sqrt((x[i] - (x0 + 0.2))**2 + (y[j] - (y0 - 0.2))**2)
            U[i, j] += - Vt * (y[j] - (y0 - 0.2)) * np.exp((1-(r/r0)**2)/2)
            V[i, j] += Vt * (x[i] - (x0 + 0.2)) * np.exp((1-(r/r0)**2)/2)

    # Initialize the third vortex at (x0 + 0.4, y0)
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            r = np.sqrt((x[i] - (x0 + 0.4))**2 + (y[j] - y0)**2)
            U[i, j] += - Vt * (y[j] - y0) * np.exp((1-(r/r0)**2)/2)
            V[i, j] += Vt * (x[i] - (x0 + 0.4)) * np.exp((1-(r/r0)**2)/2)

    U, V = apply_boundary_conditions(bc, U, V)

    Uk = np.copy(U)
    Vk = np.copy(V)
    Ukp1 = np.copy(U)
    Vkp1 = np.copy(V)


    vor = np.zeros((nx, ny))
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            vor[i, j] = ((V[i+1, j] - V[i-1, j]) / (2 * dx)) - ((U[i, j+1] - U[i, j-1]) / (2 * dy))

    X,Y = np.meshgrid(x,y,indexing="ij")

    levels = np.linspace(-0.5,1,16)
    plt.figure(figsize=(10, 4))
    plt.contourf(X, Y, vor, cmap='viridis')
    plt.colorbar()
    plt.title('Vorticity Contour')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0,lx])
    plt.ylim([0,ly])
    plt.axis("equal")
    plt.savefig("data/initialization.png")
    plt.close()

    # Plot the inlet boundary condition
    plt.figure(figsize=(6, 4))
    plt.plot(U[0, :], y)
    plt.xlabel('U[0, :]')
    plt.ylabel('y')
    plt.title('Inlet Boundary Condition')
    plt.axis("equal")
    plt.savefig("data/inletBC.png")

    with open(f'data/init.dat', 'w') as f:
      f.write('TITLE = "Vortex Convect"\n')
      f.write('VARIABLES = "X", "Y", "u", "v", "Vorticity"\n')
      f.write(f'ZONE T="BIG ZONE", I={nx}, J={ny}, DATAPACKING=POINT\n')
      for j in range(ny):
          for i in range(nx):
              f.write(f"{x[i]} {y[j]} {U[i, j]} {V[i, j]} {vor[i, j]}\n")

    start_time = time.time()

    iteration_counts = []

    U_sim = np.expand_dims(np.copy(U),axis=2)
    V_sim = np.expand_dims(np.copy(V),axis=2)

    print("U_sim : ",U_sim.shape)


    for t in range(ntsteps):
        print(t)
        errorx = 1
        errory = 1
        error = 1
        iter = 0

        while error > errormax:
          Ukp1, Vkp1 = update_U_V(U, V, Ukp1, Vkp1, Uk, Vk, mu, dt, dx, dy, nx, ny, w)

          U, V = apply_neumann_bc(U, V, nx, ny)

          errorx = np.max(np.abs(Ukp1 - Uk))
          Uk = np.copy(Ukp1)
          errory = np.max(np.abs(Vkp1 - Vk))
          Vk = np.copy(Vkp1)
          error = max(errorx, errory)
          iter += 1

        iteration_counts.append(iter)
        U = np.copy(Ukp1)
        V = np.copy(Vkp1)

        if t % cfg.simulation.settings.write_interval == 0:
          U_sim = np.append(U_sim,np.expand_dims(U, axis=2),axis=2)
          V_sim = np.append(V_sim,np.expand_dims(V, axis=2),axis=2)

          vor = np.zeros((nx, ny))
          vor = calculate_vorticity(vor, V, U, dx, dy, nx, ny)
          with open(f'data/data_%i.dat'%t, 'w') as f:
            f.write('TITLE = "Vortex Convect"\n')
            f.write('VARIABLES = "X", "Y", "u", "v", "Vorticity"\n')
            f.write(f'ZONE T="BIG ZONE", I={nx}, J={ny}, DATAPACKING=POINT\n')
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{x[i]} {y[j]} {U[i, j]} {V[i, j]} {vor[i, j]}\n")
    
    plt.figure()
    plt.plot(range(ntsteps), iteration_counts)
    plt.xlabel('Time step')
    plt.ylabel('Number of iterations')
    plt.title('Number of iterations per time step')
    plt.savefig("data/iter_per_tsteps.png")

    print("U_sim : ",U_sim.shape)
    np.save("U_sim.npy",U_sim)

    print("U_sim : ",V_sim.shape)
    np.save("V_sim.npy",V_sim)

    elapsed_time = time.time() - start_time
    print(f'Elapsed time = {elapsed_time:.2f} seconds')

if __name__ == "__main__":
    with initialize(config_path=".", version_base='1.3'):
        cfg = compose(config_name="config")
        main(cfg)