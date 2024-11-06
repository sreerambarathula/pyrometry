import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import LineString, Polygon
from itertools import product
from scipy.integrate import quad
from sklearn.metrics import mean_absolute_percentage_error
from joblib import Parallel, delayed
import pandas as pd

# Core function definitions as given

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

def save_to_csv(data, filename_prefix, grid, snr=None):
    filename = f"{filename_prefix}_grid_{grid}"
    if snr is not None:
        filename += f"_SNR_{snr}"
    filename += ".csv"
    pd.DataFrame(data).to_csv(filename, index=False, header=False)
    print(f"Saved {filename}")

# Function to save static grid data components
def save_grid_data(centers, sensors, paths, elements, length_mat, grid):
    save_to_csv(centers, "centers", grid)
    save_to_csv(sensors, "sensors", grid)
    save_to_csv(paths, "paths", grid)
    save_to_csv(elements, "elements", grid)
    save_to_csv(length_mat, "length_mat", grid)

def add_noise(tof, snr_db):
    signal_power = np.mean(tof**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.normal(size=tof.shape)
    return tof + noise

# Function to compute and save discretization error
def compute_discretization_error(length_mat, grid_reciprocal_sound, tof_experimental_clean):
    # Calculate TOF grid and discretization error
    tof_grid = np.matmul(length_mat, grid_reciprocal_sound)
    #discretization_error = np.abs(tof_experimental_clean - tof_grid)
    
    # Calculate MAPE of the discretization error with respect to tof_experimental_clean
    discretization_mape = mean_absolute_percentage_error(tof_experimental_clean, tof_grid)
    
    return discretization_mape

def apply_reconstruction_method(length_mat, tof, method="pseudo_inverse"):
    if method == "pseudo_inverse":
        f = np.matmul(np.linalg.pinv(length_mat), tof)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    c_square = (1 / f)**2
    temp_reconstructed = c_square / (1.4 * 287)
    return temp_reconstructed

def calculate_error(grid_temp, temp_reconstructed):
    error = mean_absolute_percentage_error(grid_temp, temp_reconstructed)
    return error

# Function to compute grid temperature and reconstructed temperature, then save them
def compute_and_save_temp_data(grid, centers, length_mat, tof_experimental_clean, snr=None, method="pseudo_inverse"):
    # Calculate grid_temp based on the given formula
    grid_temp = 500 * np.exp(-((centers[:, 0] - 0.5) ** 2) - ((centers[:, 1] - 0.5) ** 2))
    grid_sound = np.sqrt(1.4 * 287 * grid_temp)
    grid_reciprocal_sound = (1 / grid_sound).reshape(-1, 1)

    # Generate noisy TOF if SNR is specified
    tof = add_noise(tof_experimental_clean, snr) if snr is not None else tof_experimental_clean

    # Apply reconstruction method and calculate error
    temp_reconstructed = apply_reconstruction_method(length_mat, tof, method=method)
    error = calculate_error(grid_temp, temp_reconstructed)

    # Prepare data for saving
    temp_with_coords = np.hstack((centers[:, :2], temp_reconstructed.reshape(-1, 1)))
    save_to_csv(temp_with_coords, f"temp_pinv", grid, snr)

    return error, grid_temp, temp_reconstructed, grid_reciprocal_sound

def process_grid_with_snr(grid, snr_values, method="pseudo_inverse"):
    # Generate or retrieve all required components
    centers, sensors, paths, elements, length_mat = cell_centers(grid)
    tof_experimental_clean = parallel_tof(paths)

    # Save static grid data (centers, sensors, etc.)
    save_grid_data(centers, sensors, paths, elements, length_mat, grid)
    condition_number = np.linalg.cond(length_mat)
    
    # Compute clean data and discretization error MAPE
    error_clean, grid_temp_clean, temp_pinv_clean, grid_reciprocal_sound = compute_and_save_temp_data(grid, centers, length_mat, tof_experimental_clean, method=method)
    discretization_mape = compute_discretization_error(length_mat, grid_reciprocal_sound, tof_experimental_clean)
    
    # Save clean data
    save_to_csv(np.hstack((centers[:, :2], grid_temp_clean.reshape(-1, 1))), "grid_temp_clean", grid)
    save_to_csv(np.hstack((centers[:, :2], temp_pinv_clean.reshape(-1, 1))), "temp_pinv_clean", grid)
    
    results = [(grid, "Clean", error_clean, condition_number, discretization_mape)]

    # Process noisy data for each SNR level
    for snr in snr_values:
        error_noisy, _, _, _ = compute_and_save_temp_data(grid, centers, length_mat, tof_experimental_clean, snr=snr, method=method)
        # Discretization MAPE only applies to the clean data, so we reuse it for each SNR entry
        results.append((grid, snr, error_noisy, condition_number, discretization_mape))
        

    # Save summary CSV for this grid with all SNR values
    summary_df = pd.DataFrame(results, columns=['Grid Size', 'SNR (dB)', 'Error', 'Condition Number', 'Discretization MAPE'])
    summary_filename = f"summary_grid_{grid}.csv"
    summary_df.to_csv(summary_filename, index=False)
  

    return results

# List of SNR values to test
snr_values = [10, 20, 30]  # Add more SNR values as needed

# Parallel execution over grid sizes
grid_sizes = np.arange(2, 5, 1)
all_results = Parallel(n_jobs=-1)(delayed(process_grid_with_snr)(grid, snr_values) for grid in grid_sizes)

# Combine all results into a single summary file for all grids
all_results_flat = [item for sublist in all_results for item in sublist]  # Flatten results
final_summary_df = pd.DataFrame(all_results_flat, columns=['Grid Size', 'SNR (dB)', 'Error', 'Condition Number', 'Discretization MAPE'])
final_summary_df.to_csv("final_summary_results_with_snr.csv", index=False)
