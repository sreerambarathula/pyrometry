import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import LineString, Polygon
from itertools import product
from scipy.integrate import quad
from sklearn.metrics import mean_absolute_percentage_error
from joblib import Parallel, delayed

def cell_centers(Grid_Size):
    lattice = 0.5 / Grid_Size
    x = y = np.arange(lattice, 1, 2 * lattice)
    X, Y = np.meshgrid(x, y)
    cells = np.array([X.ravel(), Y.ravel()]).T
    x1 = x4 = cells[:, 0] - lattice
    x2 = x3 = cells[:, 0] + lattice
    y1 = y2 = cells[:, 1] + lattice
    y4 = y3 = cells[:, 1] - lattice
    cells = np.hstack((cells, x1[:, np.newaxis], y1[:, np.newaxis], x2[:, np.newaxis], y2[:, np.newaxis], 
                       x3[:, np.newaxis], y3[:, np.newaxis], x4[:, np.newaxis], y4[:, np.newaxis]))
    cells2 = cells[:, 0:2]
    C1 = cells2[cells2[:, 0] == x[0]]
    C1[:, 0] -= x[0]
    C2 = C1.copy()
    C2[:, 0] += 1
    C3 = C1[:, ::-1]
    C4 = C2[:, ::-1]
    sensors = np.vstack((C1, C2, C3, C4))
    paths = [list(map(tuple, pair)) for pair in product(sensors, repeat=2)]
    elements = [list(map(tuple, cells[i, 2:10].reshape(4, 2))) for i in range(len(cells))]
    
    # Compute length matrix with bounding box filtering and parallelization
    path_boxes = [LineString(path).bounds for path in paths]
    cell_boxes = [Polygon(element).bounds for element in elements]
    length_mat = compute_length_matrix_parallel(paths, elements, path_boxes, cell_boxes)
    
    return cells, sensors, paths, elements, length_mat

def intersection_length(line, rect_points):
    line = LineString(line)
    rectangle = Polygon(rect_points)
    intersection = line.intersection(rectangle)
    if intersection.is_empty:
        return 0 
    elif intersection.geom_type == 'LineString':
        return intersection.length 
    elif intersection.geom_type == 'MultiLineString':
        return sum(segment.length for segment in intersection)
    elif intersection.geom_type == 'Point':
        return 0
    else:
        return 0

# Optimized function to calculate length matrix with bounding box filtering and parallelization
def compute_length_matrix_parallel(paths, elements, path_boxes, cell_boxes):
    def compute_entry(i, j):
        # Check bounding boxes for overlap
        if (path_boxes[i][2] < cell_boxes[j][0] or path_boxes[i][0] > cell_boxes[j][2] or 
            path_boxes[i][3] < cell_boxes[j][1] or path_boxes[i][1] > cell_boxes[j][3]):
            return 0
        else:
            return intersection_length(paths[i], elements[j])

    # Use joblib to parallelize the intersection length calculations
    return np.array(Parallel(n_jobs=-1)(
        delayed(compute_entry)(i, j) for i in range(len(paths)) for j in range(len(elements))
    )).reshape(len(paths), len(elements))

def tof_analytical(start, end):
    gamma = 1.4
    R = 287  # J/(kg K)

    def temperature_profile(x, y):
        return 500 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2))

    def speed_of_sound(x, y):
        T = temperature_profile(x, y)
        return np.sqrt(gamma * R * T)
    
    def integrand(t):
        x0, y0 = start
        x1, y1 = end
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        dxdt = x1 - x0
        dydt = y1 - y0
        ds_dt = np.sqrt(dxdt**2 + dydt**2)
        c = speed_of_sound(x, y)
        return ds_dt / c

    time, _ = quad(integrand, 0, 1)
    return time

# Parallel TOF calculation for all paths
def parallel_tof(paths):
    return np.array(Parallel(n_jobs=-1)(delayed(tof_analytical)(i[0], i[1]) for i in paths)).reshape(-1, 1)

# Function to calculate temperature reconstruction error for each grid size
def calculate_for_grid(grid):
    centers, sensors, paths, elements, length_mat = cell_centers(grid)
    tof_experimental = parallel_tof(paths)
    grid_temp = 500 * np.exp(-((centers[:, 0] - 0.5) ** 2) - ((centers[:, 1] - 0.5) ** 2))
    grid_sound = np.sqrt(1.4 * 287 * grid_temp)
    grid_reciprocal_sound = (1 / grid_sound).reshape(-1, 1)
    tof_grid = np.matmul(length_mat, grid_reciprocal_sound)
    f = np.matmul(np.linalg.pinv(length_mat), tof_experimental)
    c_square = (1 / f) ** 2
    temp_pinv = c_square / (1.4 * 287)
    error = mean_absolute_percentage_error(grid_temp, temp_pinv)
    print("The error in temperature reconstruction for grid size", grid, ":", error)
    return error

# Parallel execution over grid sizes
grid_sizes = np.arange(2, 32, 1)
errors = Parallel(n_jobs=-1)(delayed(calculate_for_grid)(grid) for grid in grid_sizes)
