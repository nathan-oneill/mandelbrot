"""
Create a visualisation of the mandelbrot set

last updated: 2021-04-23
"""




## MODULES ##
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import cmath




## SUPPORT FUNCTIONS ##
def point_on_cardioid(t: float) -> complex:
    """Return a point that lies on the main cardioid, where t is a real number."""
    return np.exp(t*1j) * (2-np.exp(t*1j))/4 

def rotate_by(z: complex, angle: float) -> complex:
    """Rotate z around the origin by an angle"""
    #return z * np.exp(angle * 1j) #max precision 15
    return z * cmath.rect(1,angle) #max precision 16

def find_zoompoint(seed: complex, precision: int = 16, search_square_size: int = 5, n_max: int = 200, r_max: int = 2) -> complex:
    """
    Try to find a good point to zoom in on. The goal is to find a point in
    the Mandelbrot set that is surrounded by points that aren't in the set.
    (i.e. find a point on the set's boundary)

    @param {complex} seed : place to start searching
    @param {int} precision : number of decimal places to search up to
    """
    # Restrictions
    if abs(seed) > r_max:
        return 0
    if precision > 16: # past this point you might need a double (in complex numbers' real and imaginary part) - any more than rotate_by() will not rotate when precision gets too high
        print("max precision is 16. Using that instead of {}".format(precision))
        precision = 16


    # Directions to check
    dirs = []
    for y in np.linspace(-1, 1, search_square_size):
        col = []
        for x in np.linspace(-1, 1, search_square_size):
            col.append(complex(x,y))
        dirs.append(col)
    dirs = np.array(dirs)

    # Search
    current_total_rotations = 0
    prec = 1
    val = seed
    while prec <= precision:
        new =  val + dirs/10**prec
        escape_ns = mandelbrot_escape_numbers(new, n_max=n_max, r_max=r_max)

        # Density 'supersampling'
        """
        Explanation: this will turn the actual grid into a 'super-sampled' grid,
        where a super-sampled point is only 0 all its actual surrounding points 
        are 0.
        The goal is to find points that are 0, but lie adjacent to non-zero points.
        """
        base = np.zeros_like(escape_ns)
        # densities - top,right,bottom,left
        top_left, top_right, bottom_right, bottom_left = np.copy(base), np.copy(base), np.copy(base), np.copy(base)
        top_left[:-1, :-1] = escape_ns[1:, 1:]
        top_right[:-1, 1:] = escape_ns[1:, :-1]
        bottom_right[1:, 1:] = escape_ns[:-1, :-1]
        bottom_left[1:,:-1] = escape_ns[:-1,1:]

        # weight the border heavily - top,right,bottom,left
        base[:1, :] = 4*n_max + 1
        base[:,-1:] = 4*n_max + 1
        base[-1:, :] = 4*n_max + 1
        base[:, :1] = 4*n_max + 1

        # total density
        density = base + top_left + top_right  + bottom_right + bottom_left # For each point, this describes that point's surroundings (sort of like divergence in calculus)


        # Select reasonable directions to try
        mask1 = np.logical_and(0 < density, density < 4*n_max + 1)
        if np.any(mask1):
            mask = np.equal(density, np.amin(density[mask1]))
        else:
            mask = mask1 # no reasonable values
        possible_vals = new[mask]
        np.random.shuffle(possible_vals) #randomise the choice if multiple points have the same escape nummber

        # Find a suitable value
        new_found = False
        for i, possible_val in enumerate(possible_vals):
            if possible_val != val:
                new_found = True
                val = possible_val
                break

        #last resort - try to rotate cloclwise (up to a maximum) +++ TODO: try both clockwise & anticlockwise and choose which one is better
        if not new_found and current_total_rotations < 2*np.pi:
            rotate_angle = 1/10**prec / abs(val) # s.t. arc length ~ 1/10**prec
            val = rotate_by(val, rotate_angle)
            current_total_rotations += rotate_angle

        # Increase precision iff total density is low enough
        elif len(new[mask1]) < (search_square_size-1)**2 / 2:
            prec += 1

    return val



def mandelbrot_escape_numbers(domain: 'array_like', n_max: int = 200, r_max: int = 2) -> np.array:
    """
    Mandelbrot 'condition'. Any element in domain that has a non-zero
    'escape number' is assumed to have a sequence that diverges to infinity,
    and is hence not in the Mandelbrot set.

    Return the number of iterations, for each element in domain (an array of 
    complex numbers), to reach abs(element) > r_max where each element is 
    repeatedly subject to: 
        element = element**2 + original_value
    up to n_max times.
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
    """
    As mandelbrot_escape_numbers, but for a single number. Left as a 
    demonstration. Repeatedly do:
        zn+1 = zn^2 + z
    to in a sequence of complex numbers zn
    """
    zn = z
    for n in range(n_max):
        if abs(zn) > r_max:
            return n
        zn = zn*zn + z
    return 0




## MAIN FUNCTIONS ##
def mandelbrot_square(z_min, z_max, n=500, n_max = 100, showplot=False, axis=None, overall_extent=None):
    """
    Visualise the mandelbrot set in a square matrix from z_min to z_max of size
    n**2.
    The colours correspond to the number of iterations required to have
        (abs(z) > r_max=2), modulo 200.

    @param {complex} z_min : min z value
    @param {complex} z_max : max z value
    @param {int} n : number of numbers to check length-wise and height-wise
    @param {int} n_max : max number of iterations to check in mandelbrot_check() before declaring a value will not diverge to infinity
    @param {Axis} axis : axis to plot to
    @param {list} overall_extent : used in animating
    """
    # Mandelbrot iteration
    real_domain = np.linspace(z_min.real, z_max.real, n)
    imag_domain = np.linspace(z_min.imag, z_max.imag, n)
    domain = [ [complex(x,y) for x in real_domain] for y in imag_domain ]
    domain = np.array(domain)
    escape_ns = mandelbrot_escape_numbers(domain, n_max)
    
    # Enabling detail to be viewed +++ try different functions to map result to [0,1], and set vmax = highest value possible with that function
    viewmapper = 1
    if viewmapper == 1:
        escape_ns, vmax = escape_ns%200, 200
    elif viewmapper == 2:
        escape_ns, vmax = np.e**(-1/escape_ns), 1
    else:
        escape_ns, vmax = escape_ns, n_max
    # Colormap, some good ones: None, 'hsv' - https://matplotlib.org/stable/tutorials/colors/colormaps.html
    cmap = None

    # Creating graph
    extent = overall_extent or [z_min.real, z_max.real, z_min.imag, z_max.imag]
    plot = axis or plt
    image = plot.imshow(escape_ns, extent=extent, aspect='equal', vmin=0, vmax=vmax, cmap=cmap)

    if showplot:
        plt.ylabel("Im")
        plt.xlabel("Re")
        plt.show()
    return image


def create_animation(centre: complex, range_: complex, num_frames: int, 
                     n_max: int = 100, 
                     zoom_factor: float = 2, 
                     num_pixels: int = 500, 
                     frame_rate: float = 2, 
                     output_filename: str = None):
    """
    Create an animation

    NOTE: the displayed range is not correct!

    TODO:
        - 'use_dynamic_stepsize' -> increase n_max as the zoom_factor increases
        - or 'n_max':None -> choose n_max dynamically (should be done in 
          mandelbrot_square)
    
    @param {complex} centre : centre of the plot
    @param {complex} range_ : complex number representing the range. Both 
                              range_.real and range_.imag should be positive
    @param {float} zoom_factor : factor to zoom by in each successive frame
    @param {int} num_pixels : number of pixels along width & height
    @param {int} num_frames : number of frames to render
    @param {float} frame_rate : frames per second
    @param {str} output_filename : save a gif to output_filename if not None. (DO NOT SPECIFY .gif for now)
    """
    print("|=============================|")
    print(" starting to create animation")
    t0 = time.process_time()

    fig = plt.figure()
    plt.ylabel("Im") # not working
    plt.xlabel("Re")

    z_min = centre - range_/2
    z_max = centre + range_/2
    overall_extent = [z_min.real, z_max.real, z_min.imag, z_max.imag]
    ax = plt.axes(xlim=(z_min.real, z_max.real), ylim=(z_min.imag, z_max.imag))

    
    # Create Frames
    ims = []
    for zoom_level in range(1,num_frames+1):
        r = range_ / (zoom_factor ** zoom_level)
        im = mandelbrot_square(centre-r/2, centre+r/2, num_pixels, n_max, False, ax, overall_extent)
        ims.append([im])
        print("  ++ finished frame {}".format(zoom_level))

    # Create Animation
    interval = round(1000 / frame_rate)
    animation = anim.ArtistAnimation(fig, ims, interval=interval, repeat=True, blit=True)

    # Display or Save
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








if __name__ == '__main__':
    import time

    # https://en.wikipedia.org/wiki/Misiurewicz_point and https://math.stackexchange.com/questions/2181175/value-to-use-as-center-of-mandelbrot-set-zoom for finding interesting points
    INTERESTING_POINTS = [
        {"centre":-0.5 + 0j, "range":3 + 3j}, #overview
        {"centre":-0.756 - 0.09j, "range":1+1j},
        {"centre":-0.77568377 + 0.13646737j, "range":1+1j}, # good for zoom
        {"centre":-0.74364388703715870475219150611477 + 0.131825904205311970493132056385139j, "range":5 + 3j}, #really good one (from wikipedia)
        {"centre": point_on_cardioid(1+np.pi/2) * (1+np.pi*1e-7), "range":6+6j}, # Boundary of main cardioid (replace 2 with any real value). The 1.xxxxx is a factor to extend the cardioid to the actual domain seen. Adjust it for better results
        {"centre":-0.645 - 0.385j, "range":1+1j},
        {"centre":-0.77568377 + 0.13646737j, "range": 5 + 3j}
        ]
    OUTPUT_FILENAME = r"C:\Users\Nathan\Desktop\mandelbrotzoom"

    if True:
        poi = find_zoompoint(-0.0045 - .0005j)
        range_ = 5+3j
        print("found an interesting point:", poi)
    else:
        poi, range_ = INTERESTING_POINTS[3].values()


    #create_animation(poi, range_, 30, n_max=1000, output_filename=OUTPUT_FILENAME)
    mandelbrot_square(poi-range_/2, poi+range_/2, 1000, showplot=True) #Produce a single image