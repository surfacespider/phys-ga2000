import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
import cv2

def airy_disk_intensity(x, y, wavelength, aperture_diameter):
    """Calculate the intensity of an Airy disk at given positions (x, y).

    Args:
        x (numpy.ndarray): Array of x-coordinates.
        y (numpy.ndarray): Array of y-coordinates.
        wavelength (float): Wavelength of light.
        aperture_diameter (float): Diameter of the aperture.

    Returns:
        numpy.ndarray: Intensity values for the given coordinates.
    """
    r = np.sqrt(x**2 + y**2)
    k = 2 * np.pi / wavelength
    return (2 * j1(k * r * aperture_diameter) / (k * r * aperture_diameter))**2


def airy_disk_intensity(x, y, wavelength, aperture_diameter):
    r = np.sqrt(x**2 + y**2)
    k = 2 * np.pi / wavelength
    return (2 * j1(k * r * aperture_diameter) / (k * r * aperture_diameter))**2

def create_circular_aperture(aperture_pixels, aperture_diameter):
    circular_aperture = np.zeros((aperture_pixels, aperture_pixels))
    center = aperture_pixels // 2
    y, x = np.ogrid[:aperture_pixels, :aperture_pixels]
    mask = (x - center)**2 + (y - center)**2 <= (aperture_diameter / 2)**2
    circular_aperture[mask] = 1
    return circular_aperture

def plot_2d_airy_disk(wavelength, aperture_diameter, grid_size=100, max_radius_factor=2.0):
    max_radius = max_radius_factor * aperture_diameter / 2
    x = np.linspace(-max_radius, max_radius, grid_size)
    y = np.linspace(-max_radius, max_radius, grid_size)
    x, y = np.meshgrid(x, y)
    intensity = airy_disk_intensity(x, y, wavelength, aperture_diameter)

    plt.imshow(intensity, cmap='cividis', extent=[-max_radius, max_radius, -max_radius, max_radius], origin='lower')
    plt.title(f'2D Airy Disk for Wavelength λ = {wavelength} units and Aperture Diameter D = {aperture_diameter} units')
    plt.xlabel('X (Aperture Diameter)')
    plt.ylabel('Y (Aperture Diameter)')

    tick_values = np.linspace(-max_radius, max_radius, 5)
    plt.xticks(tick_values)
    plt.yticks(tick_values)

    aperture_circle = plt.Circle((0, 0), aperture_diameter/2, color='blue', fill=False, linestyle='solid', label='Aperture Circle')
    plt.gca().add_patch(aperture_circle)

    plt.legend()

    # Ensure equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Display current wavelength and aperture diameter
    plt.text(0.5,  -aperture_diameter*max_radius_factor*0.8, f'Wavelength: {wavelength} units\nAperture Diameter: {aperture_diameter} units',
             ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))

    plt.show()

def convolve_images(image1, image2):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(image1) * np.fft.fft2(image2)).real)

def create_circular_aperture(imagesizepixels, diameter):
    aperture = np.zeros(imagesizepixels)
    center_x, center_y = imagesizepixels[0] // 2, imagesizepixels[1] // 2
    y, x = np.ogrid[:imagesizepixels[0], :imagesizepixels[1]]
    mask = (x - center_x)**2 + (y - center_y)**2 <= (diameter / 2)**2
    aperture[mask] = 1
    return aperture

def draw_circle(array, cx, cy, radius):
    y, x = np.ogrid[:array.shape[0], :array.shape[1]]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    array[mask] = 0  # Set the circular aperture to black
    return array

def draw_vanes(array, cx, cy, thickness):
    # Draw horizontal and vertical vanes
    cv2.line(array, (cx, 0), (cx, array.shape[0]), 0, thickness)
    cv2.line(array, (0, cy), (array.shape[1], cy), 0, thickness)
    return array

def make_cassegrain_telescope(imagesizepixels, diam_app, diam_obs, thickness):
    im = np.zeros(imagesizepixels)
    center_x, center_y = imagesizepixels[0] // 2, imagesizepixels[1] // 2

    # Make circular aperture
    aperture = np.full_like(im, 255)  # Initialize with a white background
    aperture = draw_circle(aperture, center_x, center_y, diam_app // 2)

    # Make central obstruction
    central_obstruction = np.full_like(im, 255)
    central_obstruction = draw_circle(central_obstruction, center_x, center_y, diam_obs // 2)
    telescope_aperture = im + aperture - central_obstruction

    # Combine all components
    telescope_aperture = draw_vanes(telescope_aperture, center_x, center_y, thickness)
    max_value = np.max(telescope_aperture)
    telescope_aperture = max_value - telescope_aperture
    return telescope_aperture

def make_ideal_telescope(imagesizepixels, diam_app):
    im = np.zeros(imagesizepixels)
    center_x, center_y = imagesizepixels[0] // 2, imagesizepixels[1] // 2

    # Make circular aperture
    aperture = np.full_like(im, 255)  # Initialize with a white background
    aperture = draw_circle(aperture, center_x, center_y, diam_app // 2)
    telescope_aperture = im + aperture
    max_value = np.max(telescope_aperture)
    telescope_aperture = max_value - telescope_aperture
    return telescope_aperture

def convolve_images(image1, image2):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(image1) * np.fft.fft2(image2)).real)

def calculate_psf(telescope_aperture, wavelength):
    fft_result = np.fft.fftshift(np.fft.fft2(telescope_aperture))
    fft_result = np.abs(fft_result)
    kx, ky = np.meshgrid(np.fft.fftfreq(telescope_aperture.shape[1]) / wavelength, np.fft.fftfreq(telescope_aperture.shape[0]) / wavelength)
    psf = (fft_result)**2
    return psf, kx, ky

def fftIndgen(n):
    a = range(0, int(n/2+1))
    a = [-i for i in a]
    b = reversed(range(1, int(n/2)))   
    b = [-i for i in b]
    return a + b

def gaussian_random_field(Pk = lambda k : k**-3.0, size = 100):
    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))
    noise = np.fft.fft2(np.random.normal(size = (size, size)))
    amplitude = np.zeros((size,size))
    for i, kx in enumerate(fftIndgen(size)):
        for j, ky in enumerate(fftIndgen(size)):            
            amplitude[i, j] = Pk2(kx, ky)
                            
    return np.fft.ifft2(noise * amplitude)