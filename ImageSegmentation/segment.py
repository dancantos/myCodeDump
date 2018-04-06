import matplotlib.pyplot as plt
import requests
import numpy as np
import sys
import time

from annotator import Annotator
from graph import FlowGraph
from PIL import Image
from io import BytesIO

def getImage(imgURL=None):
    """Load an image as numpy array from the internet or a file
    For example, try this with
    http://www.python.org/static/community_logos/python-logo.png
    """
    if not imgURL: imgURL=input("URL or filename of image:")
    try:
        if imgURL.startswith("http"):
            response = requests.get(imgURL)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(imgURL) # assume it's a regular file
    except:
        print("ERROR: failed to load",imgURL)
    return np.asarray(img)

def rgb2gray(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

if __name__ == "__main__":

    # hyper parameters. May modify to change the behaviour of the model
    # l determines the importance of the foreground and background probability terms
    # bins determines how many bins to use in the histograms
    l = 1
    num_bins = 2

    # Get image
    if len(sys.argv) == 2:
        imgURL = sys.argv[1]
    else:
        imgURL = "http://www.python.org/static/community_logos/python-logo.png"
    img = getImage(imgURL)

    #annotate it with foreground and background points
    annotator = Annotator(img)

    # extract foreground and background points
    fg = annotator.fg
    bg = annotator.bg

    if (len(fg) == 0) or (len(bg) == 0):
        print("No Foreground or Background points selected, exiting")
        sys.exit()

    #time Imitalisation
    start = time.clock()

    print("Converting to grayscale and computing statistics")
    #compute intensities (grayscale) and collect statistics
    gray = rgb2gray(img)
    selfg = [  gray[i][j] for (j, i) in fg ]
    selbg = [ gray[i][j] for (j, i) in bg ]
    # selected = [ gray[i][j] for (j, i) in fg + bg ]
    # meanf = np.mean(selfg)
    # minf = min(selfg)
    # maxf = max(selfg)
    # meanb = np.mean(selbg)
    # minb = min(selbg)
    # maxb = max(selbg)
    o = np.std(gray)
    of = np.std(selfg)
    ob = np.std(selbg)

    forehist, forebins = np.histogram(selfg, bins = num_bins)
    foreprobs = [i/len(selfg) for i in forehist]
    backhist, backbins = np.histogram(selbg, bins = num_bins)
    backprobs = [i/len(selfg) for i in backhist]

    #collect metedata and initialise graph
    dims = img.shape
    rows = dims[0]
    cols = dims[1]
    zero = [0 for _ in range(dims[2])]
    size = rows*cols
    g = FlowGraph(size + 2)
    g.source = size+1
    g.sink = size


    #specific functions

    def edgecap(I1, I2):
        return np.exp(-(I1 - I2)**2 / (2 * (o**2)))

    def RpObj(I):
        for i in range(len(forebins)-1):
            if forebins[i] <= I and I < forebins[i+1]:
                if foreprobs[i] == 0.0:
                    return 100
                return -np.log(foreprobs[i])
        return 100

    def RpBkg(I):
        for i in range(len(backbins)-1):
            if backbins[i] <= I and I <= backbins[i+1]:
                if backprobs[i] == 0.0:
                    return 100
                return -np.log(backprobs[i])
        return 100

    print("Initialising Graph")

    # initialise graph, setting initial edge weights
    for i in range(rows):
        for j in range(cols):
            u = i*cols + j
            I1 = gray[i][j]
            cap = 0
            if i < rows - 1:
                v = (i+1)*cols + j
                I2 = gray[i+1][j]
                c = edgecap(I1, I2)
                g.add_edge(u, v, c)
            if j < cols - 1:
                v = i*cols + j + 1
                I2 = gray[i][j+1]
                c = edgecap(I1, I2)
                g.add_edge(u, v, c)

    # find pixel node with maximum sum of edge capacities
    max_nbhd = 0
    for u in range(size-2):
        s = 0
        for edge in g.adjlist[u]:
            s += edge.cap
        if s > max_nbhd:
            max_nbhd = s

    #set K
    K = max_nbhd + 1

    print("Connecting to Source and Sink and rescaling edges")
    #connect each node with source and sink with necessary capacities
    for i in range(rows):
        for j in range(cols):
            u = i*cols + j
            I = gray[i][j]
            if (j, i) in fg:
                g.add_edge(g.source, u, 51)
            elif (j, i) in bg:
                g.add_edge(u, g.sink, 51)
            else:
                g.add_edge(g.source, u, l*RpBkg(I))
                g.add_edge(u, g.sink, l*RpObj(I))
            #rescale edges and convert to integer
            for edge in g.adjlist[u]:
                edge.cap = int(np.ceil(edge.cap * 50 / K))

    setup_time = time.clock() - start

    print("Finding Minimum Cut")

    #find the cut
    start = time.clock()
    max_flow, flow, S, T = g.PreflowPush()
    solver_time = time.clock() - start

    print("Cut found, finalizing")

    #make two copies of the original image
    foreimg = np.copy(img)
    backimg = np.copy(img)

    #black out background pixels in the foreground image
    for pixel in S[:-1]:
        row = pixel//cols
        col = pixel%cols
        backimg[row][col] = np.copy(zero)

    #black out foreground pixels in background image
    for pixel in T[:-1]:
        row = pixel//cols
        col = pixel%cols
        foreimg[row][col] = np.copy(zero)

    #save the images
    foreimg = Image.fromarray(foreimg)
    foreimg.save("Foreground.png")

    backimg = Image.fromarray(backimg)
    backimg.save("Background.png")

    #Print times for important parts
    print("Initialization time: {}s".format(setup_time))
    print("Minimum Cut time:    {}s".format(solver_time))
    print("Total time:          {}s".format(setup_time + solver_time))

    print("Foreground and Background separated and saved as")
    print("Foreground.png")
    print("Background.png")