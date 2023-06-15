import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from skimage.morphology import disk
import scipy.ndimage as nd
from skimage import measure
from scipy import interpolate

from utils.image import class_to_dt, mask_array, get_distance_mask, erode_image

class Particles():
    """
    Container class for cones / particles
    """
    def __init__(self, positions, cy : float, cx : float):
        self.positions = positions
        self.cy = cy
        self.cx = cx

        self.compute_distances_cdc()
        self.compute_distances_local()
        #self.compute_num_neighbors()

    def remove_particles(self, keep):
        """
        Only keep particles for which keep mask
        is True
        """
        self.positions = self.positions[keep]
        self.distances_cdc = self.distances_cdc[keep]

        self.compute_distances_cdc()
        self.compute_distances_local()

    def add_particles(self, new_particles):
        """
        Add array of new particles of shape (n,2) to 
        the set of particles
        """
        self.positions = np.concatenate([self.positions, new_particles], axis=0)

        self.compute_distances_cdc()
        self.compute_distances_local()

    def compute_distances_cdc(self):
        """
        Recompute the distances to the CDC
        """
        y, x = self.positions[:,0] - self.cy, self.positions[:,1] - self.cx
        distances = np.sqrt(y ** 2 + x ** 2)
        self.distances_cdc = distances

    def compute_distances_local(self):
        """
        Compute local mean distances between cones (and STD)
        """
        local_area = 128
        n_neighbors = 3
        cones = self.positions

        # Compute neighbors
        nbrs = NearestNeighbors(n_neighbors=local_area+1).fit(cones) # No particle is a neighbor to itself, hence +1
        distances, indices = nbrs.kneighbors(cones)

        # First iteration: mean of each particle to its neighbors
        neighbor_distances = distances[:,1:n_neighbors+1] # No particle is a neighbor to itself, hence +1
        mean_distances = neighbor_distances.mean(axis=1)

        # Second iteration: mean of all neighbors = local area
        local_area_distances = mean_distances[indices]
        self.distances_local_mean = local_area_distances.mean(axis=1)
        self.distances_local_std = local_area_distances.std(axis=1)

    # def compute_num_neighbors(self):
    #     """
    #     Compute number of neighbors within a multiple of 
    #     the mean local distance
    #     """
    #     local_area = 128
    #     cones = self.positions
    #     factor = 1.0

    #     # Compute neighbors
    #     nbrs = NearestNeighbors(n_neighbors=local_area+1).fit(cones) # No particle is a neighbor to itself, hence +1
    #     distances, indices = nbrs.kneighbors(cones)

    #     # Compute number of neighbors
    #     self.num_neighbors = np.sum((distances[:,1:] / (self.distances_local_mean + factor * self.distances_local_std)[:,None]) < 1, axis=1)

    def get_mask(self, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        pos = np.round(self.positions).astype(int)
        for (y,x) in pos:
            if y >= 0 and y < shape[0]:
                if x >= 0 and x < shape[1]:
                    mask[y,x] = 1
        return mask

class Force():
    """
    Basic force class
    """
    def __init__(self):
        pass

    def apply(self, particles, additive=0):
        return particles.positions + additive
    
class Visualizer():
    """
    Visualized particles and the underlying vector field
    using a pyplot animation
    """
    def __init__(self, solver, min = None, max = None, vector_field = None):
        self.solver = solver

        assert (min is not None and max is not None) or vector_field is not None, "Either (min,max) or vector_field must be provided"

        self.vector_field = vector_field
        self.min = min if min is not None else np.array([0, 0])
        self.max = max if max is not None else np.array([vector_field.shape[0]-1, vector_field.shape[1]-1])

    def show_animated(self, frames, interval):
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.animation = FuncAnimation(self.fig, self.__update, frames=frames, interval=interval)
        #self.animation.save('optimization_cones.gif', writer='pillow')
        plt.show()

    def __update(self, i):
        self.ax.clear()

        self.solver.step()
        data = self.solver.particles.positions

        self.ax.set_title(f"Step {i}")

        if self.vector_field is not None:
            magnitude = np.sqrt((self.vector_field ** 2).sum(axis=2))
            self.ax.imshow(magnitude)

        self.ax.scatter(data[:,1], data[:,0], s=1, c="red") # Order is (x,y)

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        self.ax.set_xlim(self.min[1], self.max[1])
        self.ax.set_ylim(self.min[0], self.max[0])

class InternalEnergy(Force):
    """
    Represent internal energy between particles 
    based on expected distances between cones
    at a given eccentricity
    """
    def __init__(self, factor_energy, factor_deviation, factor_scale, particles=None, distance_information=None, considered_neighbors=16, pixels_per_degree=600):
        super(InternalEnergy, self).__init__()
        self.factor_energy = factor_energy
        self.factor_deviation = factor_deviation
        self.factor_scale = factor_scale
        self.particles = particles
        self.distance_information = distance_information
        self.considered_neighbors = considered_neighbors
        self.pixels_per_degree = pixels_per_degree
        self.internal_energy_sum = 0.0

    def get_internal_energy(self, particles):
        """
        Get the internal energy for each particle
        """
        positions = particles.positions
        distances_cdc = particles.distances_cdc
        n_neighbors = self.considered_neighbors #128

        # Get optimal distances and allowed deviations
        if self.distance_information is not None:
            means, stds = self.distance_information["dist"], self.distance_information["std"]
            optimal_distances = means[(distances_cdc / self.pixels_per_degree * 60).astype(int)] # Pixel to arcmin for indexing!
            allowed_deviations = self.factor_deviation * stds[(distances_cdc / self.pixels_per_degree * 60).astype(int)]
        if self.particles is not None:
            optimal_distances = self.particles.distances_local_mean
            allowed_deviations = self.factor_deviation * self.particles.distances_local_std

        # Compute nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(positions) # No particle is a neighbor to itself, hence +1
        distances, indices = nbrs.kneighbors(positions)
        distances, indices = distances[:,1:], indices[:,1:] # No particle is a neighbor to itself
        neighbors = positions[indices]

        # Get internal energy contribution of neighbors
        optimal_distances = np.stack([optimal_distances for _ in range(n_neighbors)], axis=1)
        allowed_deviations = np.stack([allowed_deviations for _ in range(n_neighbors)], axis=1)

        shifted_distances = distances - optimal_distances # Shift to 0
        evaluated_gaussian = self.factor_scale * 1.0 / (allowed_deviations * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (shifted_distances / allowed_deviations) ** 2)

        too_small = distances < optimal_distances - 3 * allowed_deviations # Mask: distance way too small
        too_large = distances > optimal_distances + 3 * allowed_deviations # Mask: distance way too large
        in_range = np.ones_like(distances) - (too_small | too_large) # Mask: distance is okay

        internal_energy = in_range * -1.0 * evaluated_gaussian # Negative contribution with "good" distance
        internal_energy += too_small * (1.0 - distances / (optimal_distances - 3 * allowed_deviations)) # Maximum punishment at 0 distance, linear decrease

        return internal_energy.sum(axis=1)

    def apply(self, particles, additive=0):
        positions = particles.positions + additive
        distances_cdc = particles.distances_cdc
        n_neighbors = self.considered_neighbors #128

        # Get optimal distances and allowed deviations
        if self.distance_information is not None:
            means, stds = self.distance_information["dist"], self.distance_information["std"]
            optimal_distances = means[(distances_cdc / self.pixels_per_degree * 60).astype(int)] # Pixel to arcmin for indexing!
            allowed_deviations = self.factor_deviation * stds[(distances_cdc / self.pixels_per_degree * 60).astype(int)]
        if self.particles is not None:
            optimal_distances = self.particles.distances_local_mean
            allowed_deviations = self.factor_deviation * self.particles.distances_local_std

        # Compute nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(positions) # No particle is a neighbor to itself, hence +1
        distances, indices = nbrs.kneighbors(positions)
        distances, indices = distances[:,1:], indices[:,1:] # No particle is a neighbor to itself
        neighbors = positions[indices]

        # Get internal energy contribution of neighbors
        optimal_distances = np.stack([optimal_distances for _ in range(n_neighbors)], axis=1)
        allowed_deviations = np.stack([allowed_deviations for _ in range(n_neighbors)], axis=1)

        shifted_distances = distances - optimal_distances # Shift to 0
        evaluated_gaussian = self.factor_scale * 1.0 / (allowed_deviations * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (shifted_distances / allowed_deviations) ** 2)

        too_small = distances < optimal_distances - 3 * allowed_deviations # Mask: distance way too small
        too_large = distances > optimal_distances + 3 * allowed_deviations # Mask: distance way too large
        in_range = np.ones_like(distances) - (too_small | too_large) # Mask: distance is okay

        internal_energy = in_range * -1.0 * evaluated_gaussian # Negative contribution with "good" distance
        internal_energy += too_small * (1.0 - distances / (optimal_distances - 3 * allowed_deviations)) # Maximum punishment at 0 distance, linear decrease
        
        # No influence on particles further away!
        # internal_energy += too_large * 0.001 * (distances - optimal_distances + 3 * allowed_deviations) # Slowly increasing punishment with distance
        
        # Compute internal energy per particle and sum
        self.internal_energy_sum = internal_energy.sum() #axis=1)

        # Move according to internal energy contribution
        towards = (internal_energy < 0.0) & (distances > optimal_distances) # Move towards negative contributions if they are further away than optimal distance
        away = (internal_energy > 0.0) | (distances < optimal_distances) # Move away from positive contributions AND if the distance is smaller than the optimal distance

        # Compute direction towards neighbors
        stacked = np.stack([positions for _ in range(n_neighbors)], axis=1)
        directions = neighbors - stacked
        # Normalize direction?
        #norm = np.linalg.norm(directions, axis=2)
        #directions = directions / np.stack([norm,norm],axis=2) # Normalize directions

        # Compute shift directions
        towards = np.stack([towards, towards], axis=2)
        away = np.stack([away, away], axis=2)
        shift_directions = self.factor_energy * (towards * directions - away * directions)
        shift_directions = np.mean(shift_directions, axis=1)

        return shift_directions
    
class ExternalEnergy(Force):
    """
    Represent external energy of the particles
    based on the underlying (predicted) DT
    """
    def __init__(self, factor, energy, gradient, min=None, max=None, interpolation_mode="fast_cubic_b_splines"):
        super(ExternalEnergy, self).__init__()
        self.factor = factor
        self.energy = energy
        self.gradient = gradient
        self.min = min if min is not None else np.array([0, 0])
        self.max = max if max is not None else np.array([gradient.shape[0]-1, gradient.shape[1]-1])
        self.dims = gradient.shape
        self.external_energy_sum = 0.0
        self.interpolation_mode = interpolation_mode

        if self.interpolation_mode == "cubic_b_splines" or self.interpolation_mode == "pc_hermitian" or self.interpolation_mode == "quintic_b_splines":
            anchor = 0.5
            if self.interpolation_mode == "cubic_b_splines":
                method = "cubic"
            if self.interpolation_mode == "pc_hermitian":
                method = "pchip"
            if self.interpolation_mode == "quintic_b_splines":
                method = "quintic"
            #method = "pchip" # "cubic" # "quintic"
            # pchip = https://scipy.github.io/devdocs/reference/generated/scipy.interpolate.PchipInterpolator.htm
            # Paper: "A Method for Constructing Local Monotone Piecewise Cubic Interpolants", Fritsch et al., 1984
            # https://epubs.siam.org/doi/abs/10.1137/0905021?journalCode=sijcd4, 
            y = np.arange(anchor, self.energy.shape[0], step=1)
            x = np.arange(anchor, self.energy.shape[1], step=1)
            self.interpolator = interpolate.RegularGridInterpolator((y,x), self.energy, method=method)

    # def get_external_energy(self, particles, anchor=0.5, dx=0.0, dy=0.0):
    #     # Bilinear interpolation
    #     # a, b, c, d = particles.positions.copy(), particles.positions.copy(), particles.positions.copy(), particles.positions.copy()

    #     # a[:,0] = a[:,0] - anchor + dy
    #     # a[:,1] = a[:,1] - anchor + dx

    #     # b[:,0] = b[:,0] - anchor + dy
    #     # b[:,1] = b[:,1] + anchor + dx

    #     # c[:,0] = c[:,0] + anchor + dy
    #     # c[:,1] = c[:,1] - anchor + dx

    #     # d[:,0] = d[:,0] + anchor + dy
    #     # d[:,1] = d[:,1] + anchor + dx

    #     # # Get energies at positions / pixels
    #     # a = self.energy[a[:,0].astype(int), a[:,1].astype(int)]
    #     # b = self.energy[b[:,0].astype(int), b[:,1].astype(int)]
    #     # c = self.energy[c[:,0].astype(int), c[:,1].astype(int)]
    #     # d = self.energy[d[:,0].astype(int), d[:,1].astype(int)]

    #     # # Interpolate
    #     # factors = (particles.positions + anchor + np.array([dy,dx])) % 1
    #     # factor_vertical, factor_horizontal = factors[:,0], factors[:,1]
    #     # first_horizontal = factor_horizontal * a + (1.0 - factor_horizontal) * b
    #     # second_horizontal = factor_horizontal * c + (1.0 - factor_horizontal) * d
    #     # bilinear = factor_vertical * first_horizontal + (1.0 - factor_vertical) * second_horizontal

    #     # return bilinear
    #     ypos = np.arange(anchor, self.energy.shape[0], step=1)
    #     xpos = np.arange(anchor, self.energy.shape[1], step=1)
    #     #print(len(ypos), len(xpos), self.energy.shape)
    #     interpolated_energy = interpolate.interp2d(xpos, ypos, self.energy, kind="cubic")

    #     result = np.array([interpolated_energy(x,y) for (y,x) in particles.positions])
    #     return result.flatten()
    #     #print("X")
    #     #return interpolated_energy(particles.positions[:,0] + dy, particles.positions[:,1] + dx)

    def get_external_energy_and_gradient(self, particles, anchor=0.5, delta=1e-2):
        # Get interpolated energy by cubic splines
        # ypos = np.arange(anchor, self.energy.shape[0], step=1)
        # xpos = np.arange(anchor, self.energy.shape[1], step=1)

        # BAD: interp2d uses a function which is not designed for interpolation in the background!
        # SURFIT from FITPACK - works fine as the predicted distance transform has no "evil" outliers, still...
        # interpolated_energy = interpolate.interp2d(xpos, ypos, self.energy, kind="cubic")
        # energies = np.array([interpolated_energy(x,y) for (y,x) in particles.positions]).flatten()
        # energies_top = np.array([interpolated_energy(x,y-delta) for (y,x) in particles.positions]).flatten()
        # energies_bottom = np.array([interpolated_energy(x,y+delta) for (y,x) in particles.positions]).flatten()
        # energies_left = np.array([interpolated_energy(x-delta,y) for (y,x) in particles.positions]).flatten()
        # energies_right = np.array([interpolated_energy(x+delta,y) for (y,x) in particles.positions]).flatten()

        # TOO SLOW with RegularGridInterpolator (all "cubic", "quintic" and "pchip")
        # Reason: Inefficient implementation https://github.com/scipy/scipy/blob/main/scipy/interpolate/_rgi.py:436
        # cites "A note on piecewise linear and multilinear table interpolation in many dimensions", Weiser et al., 1988
        # https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf
        if self.interpolation_mode == "cubic_b_splines" or self.interpolation_mode == "pc_hermitian" or self.interpolation_mode == "quintic_b_splines":
            energies = self.interpolator(particles.positions)
            energies_top = self.interpolator(particles.positions - np.array([delta, 0]))
            energies_bottom = self.interpolator(particles.positions + np.array([delta, 0]))
            energies_left = self.interpolator(particles.positions - np.array([0, delta]))
            energies_right = self.interpolator(particles.positions + np.array([0, delta]))

        # interpn - wrapper for RegularGridInterpolator (TOO SLOW)
        # method = "pchip" #"cubic" #"quintic"
        # energies = interpolate.interpn((ypos, xpos), self.energy, particles.positions, method=method)
        # energies_top = interpolate.interpn((ypos, xpos), self.energy, particles.positions - np.array([delta, 0]), method=method)
        # energies_bottom = interpolate.interpn((ypos, xpos), self.energy, particles.positions + np.array([delta, 0]), method=method)
        # energies_left = interpolate.interpn((ypos, xpos), self.energy, particles.positions - np.array([0, delta]), method=method)
        # energies_right = interpolate.interpn((ypos, xpos), self.energy, particles.positions - np.array([0, delta]), method=method)

        # FASTER THAN ABOVE (CUBIC/QUINTIC B-SPLINES + bilinear)
        # "Fast B-spline transforms for continuous image representation and interpolation", Unser et al., 1991
        # https://ieeexplore.ieee.org/document/75515, http://bigwww.epfl.ch/publications/unser9102.pdf
        # Scipy implementation: https://github.com/scipy/scipy/blob/main/scipy/ndimage/src/ni_splines.c
        if self.interpolation_mode == "fast_cubic_b_splines" or self.interpolation_mode == "fast_quintic_b_splines" or self.interpolation_mode == "bilinear":
            if self.interpolation_mode == "bilinear":
                order = 1
            if self.interpolation_mode == "fast_cubic_b_splines":
                order = 3
            if self.interpolation_mode == "fast_quintic_b_splines":
                order = 5
            #order = 3 #5
            anchor = 0.5
            energies = nd.map_coordinates(self.energy, (particles.positions - np.array([anchor,anchor])).T, order=order)
            energies_top = nd.map_coordinates(self.energy, (particles.positions - np.array([anchor + delta, anchor])).T, order=order)
            energies_bottom = nd.map_coordinates(self.energy, (particles.positions - np.array([anchor - delta, anchor])).T, order=order)
            energies_left = nd.map_coordinates(self.energy, (particles.positions - np.array([anchor, anchor + delta])).T, order=order)
            energies_right = nd.map_coordinates(self.energy, (particles.positions - np.array([anchor, anchor - delta])).T, order=order)

        # Compute gradient by central differences
        grad_y = (energies_bottom - energies_top) / (2.0 * delta)
        grad_x = (energies_right - energies_left) / (2.0 * delta)
        gradients = -1.0 * np.stack([grad_y, grad_x], axis=1)

        return energies, gradients

    def apply(self, particles, additive=0):
        # indices = (particles.positions + additive - self.min) / (self.max - self.min)
        # indices = indices * np.array([self.dims[0], self.dims[1]])
        # indices[:,0] = np.clip(indices[:,0], 0, self.dims[0] - 1)
        # indices[:,1] = np.clip(indices[:,1], 0, self.dims[1] - 1)
        # indices = np.round(indices).astype(int)
        # indices = np.round(particles.positions).astype(int)

        # # Get energies at particle positions
        # energies = self.get_external_energy(particles)

        # # Get gradient
        # delta = 1e-4 #0.125 #0.5 #0.25
        # energies_top = self.get_external_energy(particles, dy=-delta)
        # energies_bottom = self.get_external_energy(particles, dy=delta)
        # energies_left = self.get_external_energy(particles, dx=-delta)
        # energies_right = self.get_external_energy(particles, dx=delta)

        # grad_y = (energies_bottom - energies_top) / (2.0 * delta)
        # grad_x = (energies_right - energies_left) / (2.0 * delta)

        # gradient = np.stack([grad_y, grad_x], axis=1)

        # #energies = self.energy[indices[:,0], indices[:,1]]   
        # self.external_energy_sum = energies.sum()
        #self.external_energy_sum = np.sqrt(np.sum(self.vector_field[indices[:,0], indices[:,1]] ** 2)) # Magnitude of the gradient

        energies, gradients = self.get_external_energy_and_gradient(particles, anchor=0.5, delta=1e-4)
        self.external_energy_sum = energies.sum()

        return self.factor * gradients
        #return self.factor * self.gradient[indices[:,0], indices[:,1]] # Follow the gradient

class Minimizer():
    """
    Minimize the internal and external energy
    """
    def __init__(self, particles : Particles, internal : InternalEnergy, external : ExternalEnergy):
        self.particles = particles
        self.internal = internal
        self.external = external

    def step(self):
        positions = self.particles.positions

        internal_shift = self.internal.apply(self.particles)
        external_shift = self.external.apply(self.particles)

        self.particles.positions = positions + internal_shift + external_shift # Weighting is done internally

    def get_energy(self):
        return self.internal.internal_energy_sum, self.external.external_energy_sum

    def steps(self, n_steps, progress_bar=False, verbose=False):
        iterator = tqdm(range(n_steps)) if progress_bar else range(n_steps)
        for _ in iterator:
            self.step()
            if verbose:
                print(self.get_energy())

class ParticleBasedPostProcessing():
    def __init__(self, cones, dt, image, distance_information, alpha=0.5, beta=1.0, gamma=1.0, step_size=1.0, visualize=False, 
                 visualize_frames=200, interpolation_mode="fast_cubic_b_splines"):
        self.distance_information = None
        self.__rebuild(cones, dt, image, distance_information, alpha, beta, gamma, step_size, interpolation_mode)
        if visualize:
            vis = Visualizer(self.minimizer, vector_field=self.gradient)
            vis.show_animated(visualize_frames, 200)

    def __get_gradient(self, array):
        gradient = np.array(np.gradient(array))
        gradient = gradient.transpose((1,2,0))
        return gradient

    def get_discrete_local_mean_distances_map(self):
        """
        Get a discrete map of the local mean distances
        """
        positions = self.particles.positions
        local_mean_distances = self.particles.distances_local_mean
        local_std_distances = self.particles.distances_local_std
        distances_map = np.zeros(self.dt.shape)

        rounded_positions = np.round(positions).astype(int)
        distances_map[rounded_positions[:,0], rounded_positions[:,1]] = local_mean_distances - self.beta * 3.0 * local_std_distances

        s = nd.generate_binary_structure(2,1)
        for _ in range(10):
            distances_map = nd.grey_dilation(distances_map, footprint=s)
        return distances_map

    def get_discrete_internal_energy_map(self, pixels_per_degree=600):
        """
        Get a discrete map of the internal energy to locate
        regions in which a FN may be present
        """
        factor_scale = self.gamma
        factor_deviation = self.beta

        internal_energy_map = np.zeros(self.dt.shape)
        positions = self.particles.positions
        distances_cdc = self.particles.distances_cdc

        # Get optimal distances and allowed deviations
        if self.distance_information is not None:
            means, stds = self.distance_information["dist"], self.distance_information["std"]
            distances_arcmin = np.clip((distances_cdc / pixels_per_degree * 60).astype(int), 0, 90-1)
            optimal_distances = means[distances_arcmin] # Pixel to arcmin for indexing!
            allowed_deviations = factor_deviation * stds[distances_arcmin]
        if self.particles is not None:
            optimal_distances = self.particles.distances_local_mean
            allowed_deviations = factor_deviation * self.particles.distances_local_std

        # Get size of patches to insert into internal energy map
        patch_size = int(np.ceil(np.max(optimal_distances + 3 * allowed_deviations)))

        def get_internal_energy_patch(mean, std):
            # Get radius disk
            result = np.array(np.meshgrid(np.arange(-patch_size, patch_size+1), np.arange(-patch_size, patch_size+1)))
            result = np.sqrt(np.sum(result ** 2, axis=0))

            # Get masks
            smaller_mean_3std = result < mean - 3.0 * std
            larger_mean_3std = result > mean + 3.0 * std
            in_range = 1 - (smaller_mean_3std | larger_mean_3std)

            # Compute internal energy 
            evaluated_gaussian = factor_scale * 1.0 / (std * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (result / std) ** 2)

            # Merge different masks
            return in_range * -1.0 * evaluated_gaussian + smaller_mean_3std * (1.0 - result / (mean - 3.0 * std))
            
        # Add internal energies to map for each particle
        for (position, mean, std) in zip(positions, optimal_distances, allowed_deviations):
            patch = get_internal_energy_patch(mean, std)
            y, x = int(np.round(position[0])), int(np.round(position[1]))
            internal_energy_map[y-patch_size:y+patch_size+1, x-patch_size:x+patch_size+1] += patch # Important: Sum of patches!

        return internal_energy_map

    def __rebuild(self, cones=None, dt=None, image=None, distance_information=None, alpha=None, beta=None, gamma=None, step_size=None, interpolation_mode=None):
        # Update params
        self.cones = cones if cones is not None else self.cones
        self.dt = dt if dt is not None else self.dt
        self.image = image if image is not None else self.image
        self.distance_information = distance_information if distance_information is not None else self.distance_information
        self.alpha = alpha if alpha is not None else self.alpha
        self.beta = beta if beta is not None else self.beta
        self.gamma = gamma if gamma is not None else self.gamma
        self.step_size = step_size if step_size is not None else self.step_size
        self.interpolation_mode = interpolation_mode if interpolation_mode is not None else self.interpolation_mode

        # Create particles
        #locs = np.argwhere(cones == 1)
        #cy, cx = np.mean(locs[:,0]), np.mean(locs[:,1])
        # dm = get_distance_mask(cones.shape, cy, cx, 0.0, increase_per_pixel=1.0)
        # distances = dm[locs[:,0], locs[:,1]]
        cy, cx = np.mean(self.cones, axis=0)
        self.particles = Particles(cones, cy, cx) #, distances)

        # External
        #dt = class_to_dt(self.cones)
        #masked_dt = mask_array(dt, image, True, fill=1)
        self.gradient = self.__get_gradient(dt) # Use actual network "DT" output
        # plt.imshow(np.sqrt(np.sum(gradient ** 2, axis=2)))
        # plt.colorbar()
        # plt.show()
        factor_external = self.step_size * (1.0 - self.alpha)
        self.external = ExternalEnergy(factor_external, dt, -1.0 * self.gradient, interpolation_mode=interpolation_mode) # NEGATIVE GRADIENT FFS!!!

        # Internal
        factor_internal = self.step_size * (self.alpha / 2.0)
        factor_deviation = self.beta
        factor_scale = self.gamma
        #self.internal = InternalEnergy(factor_internal, factor_deviation, self.distance_information, considered_neighbors=128)
        self.internal = InternalEnergy(factor_internal, factor_deviation, factor_scale, particles=self.particles, considered_neighbors=128)

        # Minimizer
        self.minimizer = Minimizer(self.particles, self.internal, self.external)

    def optimize_locations(self, steps, progress_bar, verbose):
        """
        Optimize locations of particles based on internal and
        external terms
        """
        self.minimizer.steps(steps, progress_bar, verbose)

    def add_particles(self, verbose=False):
        """
        Try to add FN based on internal energy
        """
        internal_energy_map = self.get_discrete_internal_energy_map()
        internal_energy_map = mask_array(internal_energy_map, self.dt, True, fill=np.max(internal_energy_map)) # Remove background
        local_dmask = self.get_discrete_local_mean_distances_map()

        decide = self.dt < local_dmask / 2.0
        decide = (internal_energy_map < 0.0) * decide #(local_dmask / 2.0) #(dmask / 2.0)

        # Postprocess: remove ring
        local_dmask, applied = erode_image(local_dmask > 0, iterations=8, better_mask=True, apply_to=[decide])
        decide = applied[0]

        # Counteract noise
        decide = nd.binary_dilation(decide, iterations=16)

        # Get positions
        def get_pos(binary_map):
            clusters, num_labels = measure.label(binary_map, connectivity=2, return_num=True)
            pos = []
            for l in range(1,num_labels+1):
                indices = np.argwhere(clusters == l)
                y, x = indices[:,0], indices[:,1]
                mean_y, mean_x = np.mean(y), np.mean(x)
                if self.image[int(mean_y), int(mean_x)] > 0.5:
                    pos.append([mean_y, mean_x])
            return np.array(pos)
        
        pos = get_pos(decide)

        if verbose:
            print(f"Add_particles: {len(self.particles.positions)} + {len(pos)}")
        if len(pos) > 0:
            self.particles.add_particles(pos)

    def remove_particles(self, verbose=False):
        """
        Try to remove FP based on internal energy
        # distances
        """
        # Remove particles whose internal energy is larger than some threshold
        internal_energy = self.internal.get_internal_energy(self.particles)
        epsilon = 0.0
        keep_mask = internal_energy < epsilon # Particle is supported by other particles

        # Remove particles which are too close
        if verbose:
            print(f"Remove_particles: {len(self.particles.positions)} - {(1 - keep_mask).sum()}")
        self.particles.remove_particles(keep_mask)

    def postprocess(self, loops=10, steps=10, progress_bar=False, verbose=False, add_particles=True, remove_particles=True):
        """
        Perform particle based post processing for the given number of
        steps using the parameters specified at initialization time
        """
        for loop in range(loops):
            # Optimize locations
            self.optimize_locations(steps, progress_bar, verbose)

            # Do not add / remove particles after last optimization
            if loop < loops - 1:
                # Add particles
                if add_particles:
                    self.add_particles(verbose=verbose)

                # Remove particles
                if remove_particles:
                    self.remove_particles(verbose=verbose)

        # Return resulting mask
        return self.particles.get_mask(self.dt.shape)
    
    def improve(self, max_loops=20, steps=10, progress_bar=False, verbose=False, add_particles=True, remove_particles=True):
        """
        Optimize until no further improvement or max_loops is reached
        """
        best_internal_energy = 1e9
        best_external_energy = 1e9
        for loop in range(max_loops):
            # Optimize locations
            self.optimize_locations(steps, progress_bar, verbose)

             # Do not add / remove particles after last optimization
            if loop < max_loops - 1:
                # Add particles
                if add_particles:
                    self.add_particles(verbose)

                # Remove particles
                if remove_particles:
                    self.remove_particles(verbose)

            # Check conditions
            if self.internal.internal_energy_sum < best_internal_energy:
                best_internal_energy = self.internal.internal_energy_sum
            if self.external.external_energy_sum < best_external_energy:
                best_external_energy = self.external.external_energy_sum

            if self.external.external_energy_sum > 1.05 * best_external_energy: # External energy is positive
                return (loop+1) * steps, best_internal_energy, best_external_energy

            if self.internal.internal_energy_sum > 0.95 * best_internal_energy: # Internal energy is negative
                return (loop+1) * steps, best_internal_energy, best_external_energy
        
        return max_loops * steps, best_internal_energy, best_external_energy
# Useful:
# https://people.cs.uchicago.edu/~glk/talks/pdf/Kindlmann-OptimizingParticles-KAUST-2010.pdf
# http://people.cs.uchicago.edu/~glk/pubs/pdf/Kindlmann-ScaleSpaceParticles-VIS-2009.pdf
