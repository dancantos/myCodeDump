import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pickle

class Annotator(object):
    """Crude annotation class to allow forground and background of a picture to be marked
    Note: the delete functionality if not working properly
    """
    def __init__(self, img=None):
        self.fg = [] # set of foreground points (x,y) coordinates
        self.bg = [] # background points (x,y)
        self.mode = "x"
        self.drawing=True
        if img is not None:
            self.annotate(img)

    def store(self,event):
        if not event.inaxes:
            return
        if self.mode == "f":
            self.fg.append( (int(event.xdata),int(event.ydata)) )
            self.axes.plot(self.fg[-1][0],self.fg[-1][1],".r")
        elif self.mode== "b":
            self.bg.append( (int(event.xdata),int(event.ydata)) )
            self.axes.plot(self.bg[-1][0],self.bg[-1][1],".b")
        else:
            return
        plt.draw()

    def key_press(self,event):
        if event.key in "fFbBcC":
            self.mode = event.key.lower()
        elif event.key in 'dD':
            if self.mode == "f":
                self.fg.pop()
            elif self.mode=="b":
                self.bg.pop()
            else:
                return
            l = self.axes.lines.pop()
            l.remove()
            self.axes.plot()
            plt.draw()

    def mouse_press(self,event):
        self.drawing=True
        self.store(event)

    def mouse_release(self,event):
        self.drawing=False

    def mouse_move(self, event):
        if self.drawing:
            self.store(event)

    def register(self):
        plt.connect('motion_notify_event', self.mouse_move)
        plt.connect('button_press_event', self.mouse_press)
        plt.connect('button_release_event', self.mouse_release)
        plt.connect('key_press_event', self.key_press)

    def saveToFile(self,filename):
        if self.fig:
            plt.close(self.fig)
        self.fig=None
        self.axes=None
        pickle.dump(self,open(filename,"wb"))

    def annotate(self,img):
        "Given an image, annotate it"
        print("Press f for foreground, b for background,\n",
        " c cancel picking, d delete last, mouse button to pick a point")
        self.img=img # save image
        self.fig, self.axes = plt.subplots()
        plt.axis("off")
        self.axes.imshow(img)
        self.register()
        if self.fg:
            self.axes.plot([x for x,y in self.fg],[y for x,y in self.fg],".r")
        if self.bg:
            self.axes.plot([x for x,y in self.bg],[y for x,y in self.bg],".b")
            self.axes.plot()
        plt.show()

    def restoreFromFile(filename):
        return pickle.load(open(filename,"rb"))

