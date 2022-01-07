"""Create a visualisation of the mandelbrot set"""



################################### MODULES ###################################
import cmath
import numpy as np
import numpy.typing as npt
import time

import matplotlib.animation as anim
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.axes import Axes



############################## SUPPORT FUNCTIONS ##############################
def point_on_cardioid(t: float) -> complex:
    """Return a point that lies on the main cardioid, where t is a real number."""
    return np.exp(t*1j) * (2-np.exp(t*1j))/4 

def rotate_by(z: complex, angle: float) -> complex:
    """Rotate z around the origin by an angle"""
    return z * cmath.rect(1,angle)

def find_zoompoint(seed: complex, precision: int = 16, 
                   search_square_size: int = 5, n_max: int = 200, 
                   r_max: int = 2) -> complex:
    """Try to find a good point to zoom in on.
    
    The goal is to find a point in the Mandelbrot set that is surrounded by
    points that aren't in the set. (i.e. find a point on the set's boundary)

    Parameters
    ----------
    seed
        Place to start searching.
    precision
        Number of decimal places to search up to.
    search_square_size
        Search `search_square_size**2` points in a grid at each step.
    n_max
        As per `mandelbrot_escape_numbers`
    r_max
        As per `mandelbrot_escape_numbers`
    
    Notes
    -----
    Density 'supersampling' explanation: this will turn the actual grid into a
    'super-sampled' grid, where a super-sampled point is only 0 all its actual
    surrounding points are 0. The goal is to find points that are 0, but lie 
    adjacent to non-zero points.

    """
    # RESTRICTIONS
    if abs(seed) > r_max:
        return 0
    if precision > 16: 
        # Past this point you might need a double (in complex numbers' real
        # and imaginary part) - any more then rotate_by() will not rotate when
        # precision gets too high
        print("max precision is 16. Using that instead of {}".format(precision))
        precision = 16


    # DIRECTIONS TO CHECK
    dirs = []
    for y in np.linspace(-1, 1, search_square_size):
        col = []
        for x in np.linspace(-1, 1, search_square_size):
            col.append(complex(x,y))
        dirs.append(col)
    dirs = np.array(dirs)

    # SEARCH
    current_total_rotations = 0
    prec = 1
    val = seed
    while prec <= precision:
        new =  val + dirs/10**prec
        escape_ns = mandelbrot_escape_numbers(new, n_max=n_max, r_max=r_max)

        # DENSITY 'SUPERSAMPLING'
        base = np.zeros_like(escape_ns)

        # Densities - top,right,bottom,left
        top_left, top_right, bottom_right, bottom_left = np.copy(base), np.copy(base), np.copy(base), np.copy(base)
        top_left[:-1, :-1] = escape_ns[1:, 1:]
        top_right[:-1, 1:] = escape_ns[1:, :-1]
        bottom_right[1:, 1:] = escape_ns[:-1, :-1]
        bottom_left[1:,:-1] = escape_ns[:-1,1:]

        # Weight the border heavily - top,right,bottom,left
        base[:1, :] = 4*n_max + 1
        base[:,-1:] = 4*n_max + 1
        base[-1:, :] = 4*n_max + 1
        base[:, :1] = 4*n_max + 1

        # Total density
        # For each point, this describes that point's surroundings (sort of
        # like divergence in calculus).
        density = base + top_left + top_right  + bottom_right + bottom_left 


        # SELECT REASONABLE DIRECTION
        mask1 = np.logical_and(0 < density, density < 4*n_max + 1)
        if np.any(mask1):
            mask = np.equal(density, np.amin(density[mask1]))
        else:
            mask = mask1 # No reasonable values
        possible_vals = new[mask]
        np.random.shuffle(possible_vals) # Randomise the choice if multiple points have the same escape nummber

        # Find a suitable value
        new_found = False
        for i, possible_val in enumerate(possible_vals):
            if possible_val != val:
                new_found = True
                val = possible_val
                break

        # LAST RESORT: TRY TO ROTATE CLOCKWISE (up to a maximum)
        # TODO: try both clockwise & anticlockwise and choose which one is
        # better.
        if not new_found and current_total_rotations < 2*np.pi:
            rotate_angle = 1/10**prec / abs(val) # s.t. arc length ~ 1/10**prec
            val = rotate_by(val, rotate_angle)
            current_total_rotations += rotate_angle

        # Increase precision iff total density is low enough
        elif len(new[mask1]) < (search_square_size-1)**2 / 2:
            prec += 1

    return val



def mandelbrot_escape_numbers(domain: np.ndarray[complex, np.dtype[complex]], n_max: int = 200,
                              r_max: int = 2) -> np.ndarray[complex, np.dtype[complex]]:
    """Mandelbrot 'condition'.
    
    Any element in domain that has a non-zero 'escape number' is assumed to
    have a sequence that diverges to infinity, and is hence not in the
    Mandelbrot set.

    Return the number of iterations, for each element in domain (an array of 
    complex numbers), to reach `abs(element) > r_max` where each element is 
    repeatedly subject to: 
        `element = element**2 + original_value`
    up to `n_max` times.

    Parameters
    ----------
    domain
        2D array of complex numbers to find escape numbers of.
    n_max
        Maximum number of iterations to check
    r_max
        Maximum 'radius', which if a complex number reaches, it is assumed the
        sequence diverges.
    
    """
    domain = np.copy(domain)
    result = np.zeros_like(domain, dtype=int)
    step = np.copy(domain)

    for i in range(0, n_max+1):
        have_escaped = abs(step) > r_max
        result[have_escaped] = i
        step[have_escaped], domain[have_escaped] = 0, 0 # no longer get incremented
        step = step**2 + domain
    return result

def men_one_element(z: complex , n_max: int = 200, r_max: float = 2):
    """As `mandelbrot_escape_numbers`, but for a single number. Left as a 
    demonstration."""
    zn = z
    for n in range(n_max):
        if abs(zn) > r_max:
            return n
        zn = zn*zn + z
    return 0



################################ VISUALISATION ################################
def mandelbrot_square(z_min: complex, z_max: complex, n: int = 500, 
                      n_max: int = 100, show_plot: bool = False, 
                      axis: Axes = None, overall_extent: list[float] = None) -> AxesImage:
    """Visualise the mandelbrot set in a square matrix 
    
    Colours correspond to the number of iterations required to have
        (abs(z) > r_max=2), modulo 200.
    
    Parameters
    ----------
    z_min
        Bottom-left corner of complex domain to visualise
    z_max
        Top-right corner of complex domain to visualise
    n
        Split the domain length and height by `n` points
    n_max
        As per `mandelbrot_escape_numbers`
    show_plot
        If `True`, show the plot in a popup
    axis
        Axis to plot to
    overall_extent
        Parameter to `plt.imshow` to ensure consistency across frames in an
        animation
    
    """
    # MANDELBROT ITERATION
    real_domain = np.linspace(z_min.real, z_max.real, n)
    imag_domain = np.linspace(z_min.imag, z_max.imag, n)
    domain = [ [complex(x,y) for x in real_domain] for y in imag_domain ]
    domain = np.array(domain)
    escape_ns = mandelbrot_escape_numbers(domain, n_max)
    
    # COLOURING
    # TODO: try different functions to map result to [0,1], and set 
    # vmax = highest value possible with that function.
    viewmapper = 1
    if viewmapper == 1:
        escape_ns, vmax = escape_ns%200, 200
    elif viewmapper == 2:
        escape_ns, vmax = np.e**(-1/escape_ns), 1
    else:
        escape_ns, vmax = escape_ns, n_max
    # Colormap, some good ones: None, 'hsv' - https://matplotlib.org/stable/tutorials/colors/colormaps.html
    cmap = None

    # CREATING GRAPH
    local_extent = [z_min.real, z_max.real, z_min.imag, z_max.imag]
    extent = overall_extent or local_extent
    plot = axis or plt
    image = plot.imshow(escape_ns, extent=extent, aspect='equal', vmin=0, vmax=vmax, cmap=cmap)

    if show_plot:
        plt.ylabel("Im")
        plt.xlabel("Re")
        plt.show()
    return image


def create_animation(centre: complex, 
                     range_: complex, 
                     num_frames: int, 
                     n_max: int = 100, 
                     zoom_factor: float = 2, 
                     num_pixels: int = 500, 
                     frame_rate: float = 2, 
                     output_filename: str = None):
    """Create a zoom animation

    Parameters
    ----------
    centre
        Centre point to zoom in on.
    range_
        Complex number representing the range in real and imaginary directions.
        Both `range_.real` and `range_.complex` should be positive.
    num_frames
        Number of frames to render.
    n_max
        As per `mandelbrot_escape_numbers`.
    zoom_factor
        Factor to zoom in by in each successive frame.
    num_pixels
        Number of pixels along width & height.
    frame_rate
        Frames per second
    output_filename
        Save a gif to `output_filename` if not None. (EXCLUDE the `.gif` file
        extension).

    NOTES
    -----
    The displayed range is not correct!

    TODO:
    - Display the axes correctly (correct orientation and scale) for each frame
    - 'use_dynamic_stepsize' -> increase n_max as the zoom_factor increases, or
      'n_max':None -> choose n_max dynamically (should be done in
      mandelbrot_square).

    """
    print("|=============================|")
    print(" starting to create animation")
    t0 = time.process_time()

    fig = plt.figure() # tight and constrained don't get rid of all the whitespace
    #plt.ylabel("Im") # not working
    #plt.xlabel("Re")

    z_min = centre - range_/2
    z_max = centre + range_/2
    overall_extent = [z_min.real, z_max.real, z_min.imag, z_max.imag]
    ax = plt.axes(xlim=(z_min.real, z_max.real), ylim=(z_min.imag, z_max.imag))
    ax.axis('off') # Hide the axes bugs
    
    # CREATE FRAMES
    ims = []
    for zoom_level in range(1,num_frames+1):
        r = range_ / (zoom_factor ** zoom_level)
        im = mandelbrot_square(
            z_min = centre - r/2, 
            z_max = centre + r/2, 
            n = num_pixels, 
            n_max = n_max, 
            show_plot = False, 
            axis = ax, 
            overall_extent = overall_extent
        )
        ims.append([im])
        print("  ++ finished frame {}".format(zoom_level))

    # COMBINE INTO ANIMATION
    interval = round(1000 / frame_rate)
    animation = anim.ArtistAnimation(fig, ims, interval=interval, repeat=True, blit=True)

    # DISPLAY OR SAVE
    print(" mandelbrot animation completed")
    if output_filename == None:
        plt.show()
    else:
        animation.save(output_filename+".gif", fps=frame_rate)
        print(" saved mandelbrot animation as '{}.gif'".format(output_filename))
    
    t1 = time.process_time()
    print(" time taken: {:.3f}".format(t1-t0))
    print("|=============================|")
    return



###############################################################################
if __name__ == '__main__':
    INTERESTING_POINTS = [
        {"centre":-0.5 + 0j, "range":3 + 3j}, #overview
        {"centre":-0.756 - 0.09j, "range":1+1j},
        {"centre":-0.77568377 + 0.13646737j, "range":1+1j}, # good for zoom
        {"centre":-0.74364388703715870475219150611477 + 0.131825904205311970493132056385139j, "range":5 + 3j}, #really good one (from wikipedia)
        {"centre": point_on_cardioid(1+np.pi/2) * (1+np.pi*1e-7), "range":6+6j}, # Boundary of main cardioid (replace 2 with any real value). The 1.xxxxx is a factor to extend the cardioid to the actual domain seen. Adjust it for better results
        {"centre":-0.645 - 0.385j, "range":1+1j},
        {"centre":-0.77568377 + 0.13646737j, "range": 5 + 3j}
        ]

    if True:
        poi = find_zoompoint(-0.45 - .05j)
        range_ = 5+3j
        print("found an interesting point:", poi)
    else:
        poi, range_ = INTERESTING_POINTS[3].values()


    create_animation(poi, range_, 5, zoom_factor=2, frame_rate=1.5, n_max=1000, output_filename="zoom_result")
    #mandelbrot_square(poi-range_/2, poi+range_/2, 1000, showplot=True) #Produce a single image