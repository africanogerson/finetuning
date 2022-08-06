from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from skimage.morphology import erosion, dilation
from skimage.morphology.binary import binary_closing
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from scipy import interpolate
from scipy import ndimage as ndi
from skimage.measure import label, regionprops, find_contours


def ffdmForeground(im, cflag=False):

    def histc(x, bins):
        map_to_bins = np.digitize(x, bins)
        r = np.zeros(bins.shape)
        for i in map_to_bins:
            r[i-1] += 1
        return r, map_to_bins

    def gaus(x, a1, b1, c1):
        return a1 * np.exp(-((x - b1) / c1) ** 2)

    def get_lower(x, nbins=1000):

        # Minimum and maximum intensities
        x_min = np.amin(x)
        x_max = np.amax(x)

        # Relative frequency:
        xi = np.linspace(x_min, x_max, nbins)
        n, _ = histc(x, xi)

        # Smooth histogram:
        n = signal.convolve(n, signal.windows.gaussian(25, 4.8), mode='same')
        n = n / np.max(n)

        # Find threshold by fitting a gaussian to the histogram peak(s) located below the mean
        xsup = np.minimum(np.mean(x), np.maximum(np.percentile(x, 30), 0.2))
        ipeaks, _ = signal.find_peaks(n*(xi <= xsup), height=0.35)

        if len(ipeaks) == 1:
            select = (n > 0.35) & (xi < xsup)
            p, p_cov = curve_fit(gaus, xi[select], n[select])
            _xth = p[1] + np.sqrt((p[2]**2)*(np.log(p[0]) - np.log(0.05)))
        elif len(ipeaks) > 1:
            # find minimum between peaks
            n_min = np.min(n[np.arange(np.min(ipeaks), np.max(ipeaks) + 1)])
            i_min = np.argwhere(n == n_min)[0]

            # adjust second peak
            select = (xi >= xi[i_min]) & (n > 0.35) & (xi < xsup)
            p, _ = curve_fit(gaus, xi[select], n[select])
            _xth = p[1] + np.sqrt((p[2] ** 2) * (np.log(p[0]) - np.log(0.05)))
        else:
            n_max = np.max(n)
            i_th = np.argwhere(n < 0.05*n_max)[0]
            _xth = xi[i_th]

        return _xth

    def clean_mask(_mask0):

        # Remove first and last row
        _mask0[0, :] = False
        _mask0[-1, :] = False
        kernel = np.ones((5, 5), np.uint8)
        _mask0 = erosion(_mask0, kernel)

        # keep biggest region
        cc = label(_mask0, connectivity=2)
        stats = regionprops(cc)
        areas = np.array([region.area for region in stats])
        idx = np.argmax(areas)
        _mask0 = cc == idx+1

        # Remove spurious holes:
        _mask0 = dilation(_mask0, kernel)
        _mask0 = binary_closing(_mask0, kernel)
        _mask = ndi.binary_fill_holes(_mask0)

        return _mask

    # Find threshold
    xth = get_lower(im.flatten('F'))

    # find mask:
    mask0 = im >= max([xth, 0])

    # remove artifacts and holes in the mask
    mask = clean_mask(mask0)

    return mask


def get_contour(mask, npts=100):

    # Find contour points:
    contours = find_contours(mask, 0.5)
    f = lambda i: len(contours[i])
    idx = max(range(len(contours)), key=f)
    roi = np.zeros_like(mask, dtype=np.bool)
    roi[:, 0:int(mask.shape[1]*0.05)] = True
    edges = canny(mask, 0, mask=roi)
    tested_angles = np.linspace(-5 * np.pi/180, 5 * np.pi/180, 10)
    h, theta, d = hough_line(edges, theta=tested_angles)
    origin = np.array((0, mask.shape[1]))
    x, y = contours[idx][:, 1], contours[idx][:, 0]
    remov = np.zeros_like(x, dtype=np.bool)
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=1, threshold=100)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        m = (y1 - y0) / (origin[1] - origin[0])
        b = y1 - m * origin[1]
        remov = np.logical_or(remov, (abs(y - (m*x+b)) < 250))
    ys = contours[idx][~remov, 0]
    xs = contours[idx][~remov, 1]

    # sub-sample and smooth contour
    n = len(xs)
    s = np.linspace(1, n, npts)
    fx = interpolate.PchipInterpolator(np.linspace(1, n, n), xs)
    xs = smooth(fx(s))
    fy = interpolate.PchipInterpolator(np.linspace(1, n, n), ys)
    ys = smooth(fy(s))

    xs = xs / (0.5 * mask.shape[1]) - 1
    ys = ys / (0.5 * mask.shape[0]) - 1

    contour = np.stack([ys, xs], axis=1)

    return contour.astype('float32')


def smooth(arr, n=5):
    dims = len(arr.shape)
    s = signal.convolve(arr, np.ones((2*n+1,)*dims), mode='same')
    d = signal.convolve(np.ones_like(arr), np.ones((2*n+1,)*dims), mode='same')
    return s / d
