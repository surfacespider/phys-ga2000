import plotting
import methods
# Example: Plot 2D Airy disk with a custom aperture diameter and circles
plotting.plot_2d_airy_disk(wavelength=0.5, aperture_diameter=1, max_radius_factor=2,grid_size=2000)

# -------------------------------------

# Example: Comparing Analytical Airy Disk (Using J1) and Airy disk obtained from FFT
# I'm having trouble with this code. The residuals between the analytical and FFT solution don't seem to work align with aperture_diameters other than 1.
# I think it has something to do with the Coefficiant A in the J1 formula. I also tried normalizing the two graphs but it didn't work.
wavelength = 1
aperture_diameter = 1
aperture_pixels = 300
max_radius_factor = 2

# Example: Plot side-by-side comparison
plotting.plot_2d_airy_disk_vs_convolution(wavelength, aperture_diameter, aperture_pixels, max_radius_factor)

# Trying other values

wavelength = 0.5
aperture_diameter = 2
aperture_pixels = 300
max_radius_factor = 2

# Example: Plot side-by-side comparison
plotting.plot_2d_airy_disk_vs_convolution(wavelength, aperture_diameter, aperture_pixels, max_radius_factor)

# --------------------------------------

# Example: Fourier Transform on Ideal Telescope vs. Cassegrain Telescope

# User-defined input parameters
resolution = 2000
imagesizepixels = (resolution, resolution)
aperture_scaling_factor = 50
initial_image_size = resolution/aperture_scaling_factor
diam_app = 600
diam_obs = 300
thickness = 100
wavelength = 30  # Adjust the wavelength as needed

# Create circular initial image
circular_aperture = methods.create_circular_aperture(imagesizepixels, initial_image_size) # This is the input image (.ie a star)

# Create Cassegrain telescope aperture
telescope_aperture = methods.make_cassegrain_telescope(imagesizepixels, diam_app, diam_obs, thickness)

# Calculate PSF
psf, kx, ky = methods.calculate_psf(telescope_aperture, wavelength)

# Convolve PSF with circular aperture
resulting_image = methods.convolve_images(circular_aperture, psf)

# Create Ideal telescope aperture
telescope_aperture_ideal = methods.make_ideal_telescope(imagesizepixels, diam_app)

# Calculate PSF
psf_ideal, kx_ideal, ky_ideal = methods.calculate_psf(telescope_aperture_ideal, wavelength)

# Convolve PSF with Cassegrain Telescope
resulting_image_ideal = methods.convolve_images(circular_aperture, psf_ideal)

plotting.plot_results(circular_aperture, telescope_aperture_ideal, psf_ideal, resulting_image_ideal,
             telescope_aperture, psf, resulting_image, kx_ideal, ky_ideal, kx, ky)

# --------------------------------------------------

# Example: Gaussian wrinkling of source problem 3 part 2

plotting.plot_results_gaussian()