import methods
import matplotlib.pyplot as plt
import numpy as np

def plot_2d_airy_disk(wavelength, aperture_diameter, grid_size=100, max_radius_factor=2.0):
    """Plot a 2D Airy disk with specified wavelength and aperture diameter.

    Args:
        wavelength (float): Wavelength of light.
        aperture_diameter (float): Diameter of the aperture.
        grid_size (int, optional): Number of grid points. Defaults to 1000.
        max_radius_factor (float, optional): Factor to determine the maximum radius of the plot. Defaults to 2.0.
    """
    # Calculate the maximum radius based on the provided factor
    max_radius = max_radius_factor * aperture_diameter / 2

    x = np.linspace(-max_radius, max_radius, grid_size)
    y = np.linspace(-max_radius, max_radius, grid_size)
    x, y = np.meshgrid(x, y)

    intensity = methods.airy_disk_intensity(x, y, wavelength, aperture_diameter)


    plt.imshow(intensity, cmap='cividis', extent=[-max_radius, max_radius, -max_radius, max_radius], origin='lower')
    plt.title(f'2D Airy Disk for Wavelength λ = {wavelength} units and Aperture Diameter D = {aperture_diameter} units')
    plt.xlabel('X (Aperture Diameter)')
    plt.ylabel('Y (Aperture Diameter)')
    plt.colorbar(label='Intensity')

    # Set ticks and tick labels for x and y axes
    tick_values = np.linspace(-max_radius, max_radius, 5)
    plt.xticks(tick_values)
    plt.yticks(tick_values)

    # Add the original aperture circle in blue
    aperture_circle = plt.Circle((0, 0), aperture_diameter/2, color='blue', fill=False, linestyle='solid', label='Aperture Circle')

    # Add a circle with the radius of the known first zero of the airy disk function
    circle_diameter = 1.22 * wavelength / aperture_diameter
    circle = plt.Circle((0, 0), circle_diameter/2, color='red', fill=False, linestyle='dashed', label='First Zero of Airy Disk')

    plt.gca().add_patch(aperture_circle)
    plt.gca().add_patch(circle)

    # Add legend
    plt.legend()

    plt.show()
    


def plot_2d_airy_disk_vs_convolution(wavelength, aperture_diameter, aperture_pixels=500, max_radius_factor=2.0):
    circular_aperture = methods.create_circular_aperture(aperture_pixels, aperture_diameter)

    max_radius = max_radius_factor * aperture_diameter / 2

    x_airy = np.linspace(-max_radius, max_radius, aperture_pixels)
    y_airy = np.linspace(-max_radius, max_radius, aperture_pixels)
    x_airy, y_airy = np.meshgrid(x_airy, y_airy)

    intensity_airy = methods.airy_disk_intensity(x_airy, y_airy, wavelength, aperture_diameter)

    plt.figure(figsize=(18, 5)) 
    
    plt.subplot(1, 4, 1)
    plt.imshow(intensity_airy, cmap='cividis', extent=[-max_radius, max_radius, -max_radius, max_radius], origin='lower')
    plt.title(f'2D Airy Disk')
    plt.xlabel('X (Aperture Diameter)')
    plt.ylabel('Y (Aperture Diameter)')

    tick_values = np.linspace(-max_radius, max_radius, 5)
    plt.xticks(tick_values)
    plt.yticks(tick_values)

    aperture_circle = plt.Circle((0, 0), aperture_diameter/2, color='blue', fill=False, linestyle='solid', label='Aperture Circle')
    plt.gca().add_patch(aperture_circle)

    # Add a red circle for the first zero of the Airy disk
    first_zero_diameter = 1.22 * wavelength / aperture_diameter
    first_zero_circle = plt.Circle((0, 0), first_zero_diameter/2, color='red', fill=False, linestyle='dashed', label='First Zero of Airy Disk')
    plt.gca().add_patch(first_zero_circle)

    plt.legend()

    # Ensure equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Display current wavelength and aperture diameter
    plt.text(0.5,  -aperture_diameter*max_radius_factor*0.8, f'Wavelength: {wavelength} units\nAperture Diameter: {aperture_diameter} units',
             ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))

    plt.subplot(1, 4, 2)
    psf_convolution = methods.convolve_images(circular_aperture, intensity_airy)
    plt.imshow(psf_convolution, cmap='cividis', extent=[-max_radius, max_radius, -max_radius, max_radius], origin='lower')
    plt.title(f'Point Spread Function')
    plt.xlabel('X (Aperture Diameter)')
    plt.ylabel('Y (Aperture Diameter)')

    # Add a red circle for the first zero of the Airy disk
    first_zero_diameter = 1.22 * wavelength / aperture_diameter
    
    aperture_circle = plt.Circle((0, 0), aperture_diameter/2, color='blue', fill=False, linestyle='solid', label='Aperture Circle')
    plt.gca().add_patch(aperture_circle)
    first_zero_circle = plt.Circle((0, 0), first_zero_diameter/2, color='red', fill=False, linestyle='dashed', label='First Zero of Airy Disk')
    plt.gca().add_patch(first_zero_circle)
    plt.legend()

    # Ensure equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Display current wavelength and aperture diameter
    plt.text(0, -aperture_diameter*max_radius_factor*0.8, f'Wavelength: {wavelength} units\nAperture Diameter: {aperture_diameter} units',
             ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))

    plt.subplot(1,4,3)
    residuals = psf_convolution-(intensity_airy)
    plt.imshow(residuals, cmap='cividis', extent=[-max_radius, max_radius, -max_radius, max_radius], origin='lower')
    plt.title(f'Residuals from 2D Airy disk - PSF')
    plt.xlabel('X (Aperture Diameter)')
    plt.ylabel('Y (Aperture Diameter)')
    # Add a red circle for the first zero of the Airy disk
    first_zero_diameter = 1.22 * wavelength / aperture_diameter
    
    aperture_circle = plt.Circle((0, 0), aperture_diameter/2, color='blue', fill=False, linestyle='solid', label='Aperture Circle')
    plt.gca().add_patch(aperture_circle)
    first_zero_circle = plt.Circle((0, 0), first_zero_diameter/2, color='red', fill=False, linestyle='dashed', label='First Zero of Airy Disk')
    plt.gca().add_patch(first_zero_circle)
    plt.legend()

    # Ensure equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Display current wavelength and aperture diameter
    plt.text(0, -aperture_diameter*max_radius_factor*0.8, f'Wavelength: {wavelength} units\nAperture Diameter: {aperture_diameter} units',
             ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))
   
    plt.subplot(1,4,4)
    # Choose the central row for the slice
    central_row = residuals.shape[0] // 2
    slice_values = residuals[central_row, :]

    # Plot the slice
    plt.plot(slice_values, color='black', label='Slice')
    plt.axhline(0, color='red', linestyle='--', label='Zero Line')
    plt.title('Slice of Residuals')
    plt.xlabel('X (Pixel Value)')
    plt.ylabel('Residual Intensity')
    plt.legend()
    plt.show()
    
    plt.tight_layout()
    plt.show()
    

def plot_results(circular_aperture, telescope_aperture_ideal, psf_ideal, resulting_image_ideal,
                 telescope_aperture, psf, resulting_image, kx_ideal, ky_ideal, kx, ky):
    plt.figure(figsize=(15, 7))

    plt.subplot(2, 5, 1), plt.imshow(circular_aperture, cmap='cividis')
    plt.title("Input Image")

    plt.subplot(2, 5, 2), plt.imshow(telescope_aperture_ideal, cmap='cividis')
    plt.title('Ideal Telescope Aperture')

    plt.subplot(2, 5, 3), plt.imshow(np.log10(psf_ideal), cmap='cividis', extent=[kx_ideal.min(), kx_ideal.max(), ky_ideal.min(), ky_ideal.max()])
    plt.xlabel('kx (xf/λf)')
    plt.ylabel('ky (yf/λf)')
    plt.title('PSF_Ideal')

    plt.subplot(2, 5, 4), plt.imshow(np.log10(resulting_image_ideal), cmap="cividis")
    plt.title('Resulting Image after Convolution_Ideal')

    plt.subplot(2, 5, 7), plt.imshow(telescope_aperture, cmap='cividis')
    plt.title('Cassegrain Aperture')

    plt.subplot(2, 5, 8), plt.imshow(np.log10(psf), cmap='cividis', extent=[kx.min(), kx.max(), ky.min(), ky.max()])
    plt.xlabel('kx (xf/λf)')
    plt.ylabel('ky (yf/λf)')
    plt.title('PSF')

    plt.subplot(2, 5, 9), plt.imshow(np.log10(resulting_image), cmap='cividis')
    plt.title('Resulting Image after Convolution')
    plt.tight_layout()
    plt.show()

def plot_results_gaussian():
    #gaussian wrinkling of source problem 3 part 2

    # User-defined input parameters
    resolution = 2000
    imagesizepixels = (resolution, resolution)
    aperture_scaling_factor = 30
    initial_image_size = resolution/aperture_scaling_factor
    diam_app_mm = 300
    diam_obs_mm = 150
    thickness_mm = 50
    wavelength = 500  # Adjust the wavelength as needed


    # Calculated parameters
    diam_app = diam_app_mm
    diam_obs = diam_obs_mm
    thickness = thickness_mm

    # Create circular initial image
    input_image = methods.create_circular_aperture(imagesizepixels, initial_image_size)

    # Create telescope aperture
    telescope_aperture = methods.make_cassegrain_telescope(imagesizepixels, diam_app, diam_obs, thickness)

    print(telescope_aperture)

    # Calculate PSF
    psf, kx, ky = methods.calculate_psf(telescope_aperture, wavelength)

    # Convolve PSF with circular aperture
    resulting_image = methods.convolve_images(input_image, psf)

    # Plot the circular initial image
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 5, 1), plt.imshow(input_image, cmap='cividis')
    plt.title("Input Image")

    # Plot the ideal telescope
    telescope_aperture_ideal = methods.make_ideal_telescope(imagesizepixels, diam_app)
    plt.subplot(2, 5, 2), plt.imshow(telescope_aperture_ideal, cmap='cividis')
    plt.title('Ideal Telescope Aperture')

    gaussian_noise = methods.gaussian_random_field(Pk = lambda k: k**2, size = resolution) #make input function have correct dependence for cut off exponential decay when ~20cm

    #psf_ideal, kx_ideal, ky_ideal = calculate_psf(telescope_aperture_ideal, wavelength)
    aperture_test = telescope_aperture_ideal+gaussian_noise.real
    psf_ideal, kx_ideal, ky_ideal = methods.calculate_psf(aperture_test, wavelength)
    plt.subplot(2, 5, 3), plt.imshow(np.log10(psf_ideal), cmap='cividis', extent=[kx_ideal.min(), kx_ideal.max(), ky_ideal.min(), ky_ideal.max()])

    plt.xlabel('kx (xf/λf)')
    plt.ylabel('ky (yf/λf)')
    plt.title('PSF_Ideal')

    resulting_image_ideal = methods.convolve_images(input_image, psf_ideal)
    plt.subplot(2, 5, 4), plt.imshow(np.log10(resulting_image_ideal), cmap="cividis")
    plt.title('Resulting Image after Convolution_Ideal'), plt.xticks([]), plt.yticks([])

    # plot cassegrain
    plt.subplot(2, 5, 6), plt.imshow(input_image, cmap='cividis')
    plt.title("Input Image")


    plt.subplot(2, 5, 7), plt.imshow(telescope_aperture, cmap='cividis')
    plt.title('Cassegrain Aperture'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 5, 8), plt.imshow(np.log10(psf), cmap='cividis', extent=[kx.min(), kx.max(), ky.min(), ky.max()])
    plt.xlabel('kx (xf/λf)')
    plt.ylabel('ky (yf/λf)')
    plt.title('PSF')

    plt.subplot(2, 5, 9), plt.imshow(np.log10(resulting_image), cmap='cividis')
    plt.title('Resulting Image after Convolution'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()