#!/usr/bin/env python
# coding: utf-8

# # import 

# In[142]:


# %% Setup
import pickle
from tqdm.auto import tqdm
import numpy as np # linear algebra
import pandas as pd
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from collections import Counter
import copy
from itertools import product, permutations, combinations, combinations_with_replacement
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
sys.setrecursionlimit(int(2e5))

# data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
data_path = Path('data')
train_path = data_path / 'training'
eval_path = data_path / 'evaluation'
test_path = data_path / 'test'

train_tasks = { task.stem: json.load(task.open()) for task in train_path.iterdir() if task.suffix.lower() == ".json"} 
eval_tasks = { task.stem: json.load(task.open()) for task in eval_path.iterdir() if task.suffix.lower() == ".json"}
test_tasks = { task.stem: json.load(task.open()) for task in test_path.iterdir() if task.suffix.lower() == ".json"}

data = test_tasks

cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

def plot_pictures(pictures, labels):
    fig, axs = plt.subplots(1, len(pictures), figsize=(2*len(pictures),32))
    for i, (pict, label) in enumerate(zip(pictures, labels)):
        axs[i].imshow(np.array(pict), cmap=cmap, norm=norm)
        axs[i].set_title(label)
    plt.show()

def plot_sample(sample, predict=None):
    """
    This function plots a sample. sample is an object of the class Sample.
    predict is any matrix (numpy ndarray).
    """
    if predict is None:
        plot_pictures([sample.inMatrix.m, sample.outMatrix.m], ['Input', 'Output'])
    else:
        plot_pictures([sample.inMatrix.m, sample.outMatrix.m, predict], ['Input', 'Output', 'Predict'])

def plot_task(task):
    """
    Given a task (in its original format), this function plots all of its
    matrices.
    """
    len_train = len(task['train'])
    len_test  = len(task['test'])
    len_max   = max(len_train, len_test)
    length    = {'train': len_train, 'test': len_test}
    fig, axs  = plt.subplots(len_max, 4, figsize=(15, 15*len_max//4))
    for col, mode in enumerate(['train', 'test']):
        for idx in range(length[mode]):
            axs[idx][2*col+0].axis('off')
            axs[idx][2*col+0].imshow(task[mode][idx]['input'], cmap=cmap, norm=norm)
            axs[idx][2*col+0].set_title(f"Input {mode}, {np.array(task[mode][idx]['input']).shape}")
            try:
                axs[idx][2*col+1].axis('off')
                axs[idx][2*col+1].imshow(task[mode][idx]['output'], cmap=cmap, norm=norm)
                axs[idx][2*col+1].set_title(f"Output {mode}, {np.array(task[mode][idx]['output']).shape}")
            except:
                pass
        for idx in range(length[mode], len_max):
            axs[idx][2*col+0].axis('off')
            axs[idx][2*col+1].axis('off')
    plt.tight_layout()
    plt.axis('off')
    plt.show()

def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

##############################################################################
# %% CORE OBJECTS

# %% Frontiers
class Frontier:
    """
    A Frontier is defined as a straight line with a single color that crosses
    all of the matrix. For example, if the matrix has shape MxN, then a
    Frontier will have shape Mx1 or 1xN. See the function "detectFrontiers"
    for details in the implementation.
    
    ...
    
    Attributes
    ----------
    color: int
        The color of the frontier
    directrion: str
        A character ('h' or 'v') determining whether the frontier is horizontal
        or vertical
    position: tuple
        A 2-tuple of ints determining the position of the upper-left pixel of
        the frontier
    """
    def __init__(self, color, direction, position):
        """
        direction can be 'h' or 'v' (horizontal, vertical)
        color, position and are all integers
        """
        self.color = color
        self.direction = direction
        self.position = position
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False
        
def detectFrontiers(m):
    """
    Returns a list of the Frontiers detected in the matrix m (numpy.ndarray).
    """
    frontiers = []
    
    # Horizontal lines
    if m.shape[0]>1:
        for i in range(m.shape[0]):
            color = m[i, 0]
            isFrontier = True
            for j in range(m.shape[1]):
                if color != m[i,j]:
                    isFrontier = False
                    break
            if isFrontier:
                frontiers.append(Frontier(color, 'h', i))
            
    # Vertical lines
    if m.shape[1]>1:
        for j in range(m.shape[1]):
            color = m[0, j]
            isFrontier = True
            for i in range(m.shape[0]):
                if color != m[i,j]:
                    isFrontier = False
                    break
            if isFrontier:
                frontiers.append(Frontier(color, 'v', j))
            
    return frontiers

# %% Grids
class Grid:
    """
    An object of the class Grid is basically a collection of frontiers that
    have all the same color.
    It is useful to check, for example, whether the cells defined by the grid
    always have the same size or not.
    
    ...
    
    Attributes
    ----------
    color: int
        The color of the grid
    m: numpy.ndarray
        The whole matrix
    frontiers: list
        A list of all the frontiers the grid is composed of
    cells: list of list of 2-tuples
        cells can be viewed as a 2-dimensional matrix of 2-tuples (Matrix, 
        position). The first element is an object of the class Matrix, and the
        second element is the position of the cell in m.
        Each element represents a cell of the grid.
    shape: tuple
        A 2-tuple of ints representing the number of cells of the grid
    nCells: int
        Number of cells of the grid
    cellList: list
        A list of all the cells
    allCellsSameShape: bool
        Determines whether all the cells of the grid have the same shape (as
        matrices).
    cellShape: tuple
        Only defined if allCellsSameShape is True. Shape of the cells.
    allCellsHaveOneColor: bool
        Determines whether the ALL of the cells of the grid are composed of
        pixels of the same color
    """
    def __init__(self, m, frontiers):
        self.color = frontiers[0].color
        self.m = m
        self.frontiers = frontiers
        hPositions = [f.position for f in frontiers if f.direction == 'h']
        hPositions.append(-1)
        hPositions.append(m.shape[0])
        hPositions.sort()
        vPositions = [f.position for f in frontiers if f.direction == 'v']
        vPositions.append(-1)
        vPositions.append(m.shape[1])
        vPositions.sort()
        # cells is a matrix (list of lists) of 2-tuples (Matrix, position)
        self.cells = []
        hShape = 0
        vShape = 0
        for h in range(len(hPositions)-1):
            if hPositions[h]+1 == hPositions[h+1]:
                continue
            self.cells.append([])
            for v in range(len(vPositions)-1):
                if vPositions[v]+1 == vPositions[v+1]:
                    continue
                if hShape == 0:
                    vShape += 1
                self.cells[hShape].append((Matrix(m[hPositions[h]+1:hPositions[h+1], \
                                                   vPositions[v]+1:vPositions[v+1]], \
                                                 detectGrid=False), \
                                          (hPositions[h]+1, vPositions[v]+1)))
            hShape += 1
            
        self.shape = (hShape, vShape) # N of h cells x N of v cells
        self.cellList = []
        for cellRow in range(len(self.cells)):
            for cellCol in range(len(self.cells[0])):
                self.cellList.append(self.cells[cellRow][cellCol])
        self.allCellsSameShape = len(set([c[0].shape for c in self.cellList])) == 1
        if self.allCellsSameShape:
            self.cellShape = self.cells[0][0][0].shape
            
        self.nCells = len(self.cellList)
            
        # Check whether each cell has one and only one color
        self.allCellsHaveOneColor = True
        for c in self.cellList:
            if c[0].nColors!=1:
                self.allCellsHaveOneColor = False
                break
        
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([f in other.frontiers for f in self.frontiers])
        else:
            return False

# %% Shapes and subclasses
class Shape:
    """
    An object of the class Shape is meant to represent a connected entity of a
    Matrix. Its main attribute is "m", a 2-dimensional numpy-ndarray of type
    np.uint8, in which all the entries have the color of the entity from the
    Matrix to be represented, and the rest of the elements are equal to 255.
    The attribute m has the smallest shape possible.
    For example, in the matrix [[1, 1], [1, 3]] we might detect two different
    Shapes: [[1, 1], [1, 255]] and [[3]].
    ...
    Main attributes
    ---------------
    m: numpy.ndarray
        Described above.
    nPixels: int
        Number of pixels of the Shape. This is, number of elements that are
        different from 255 in m.
    position: tuple (int, int)
        Position of the Shape in the original Matrix (upper-left corner of m).
    background: int
        Background color of the original Matrix.
    isBorder: bool
        Checks whether the Shape touches the border of the original Matrix.
    colors: set
        Set of colors present in the Shape.
    nColors: int
        Number of colors present in the Shape.
    colorCount: collections.Counter
        Dictionary whose keys are the colors that appear in the Shape, and
        their values represent the amount of pixels with that color.
    isSquare: bool
        Determines whether the Shape is a full square or not.
    isRectangle: bool
        Determines whether the Shape is a full rectangle or not.
        
    Methods
    -------
    hasSameShape(other, sameColor=False, samePosition=False, rotation=False,
                 mirror=False, scaling=False)
        Checks whether the Shape "other" is the same Shape as the current one,
        modulo the given parameters.
    
    """
    def __init__(self, m, xPos, yPos, background, isBorder):
        # pixels is a 2xn numpy array, where n is the number of pixels
        self.m = m
        self.nPixels = m.size - np.count_nonzero(m==255)
        self.background = background
        self.shape = m.shape
        self.position = (xPos, yPos)
        self.pixels = set([(i,j) for i,j in np.ndindex(m.shape) if m[i,j]!=255])
            
        # Is the shape in the border?
        self.isBorder = isBorder
        
        # Which colors does the shape have?
        self.colors = set(np.unique(m)) - set([255])
        self.nColors = len(self.colors)
        if self.nColors==1:
            self.color = next(iter(self.colors))

        self.colorCount = Counter(self.m.flatten()) + Counter({0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0})
        del self.colorCount[255]

        # Symmetries
        self.lrSymmetric = np.array_equal(self.m, np.fliplr(self.m))
        self.udSymmetric = np.array_equal(self.m, np.flipud(self.m))
        if self.m.shape[0] == self.m.shape[1]:
            self.d1Symmetric = np.array_equal(self.m, self.m.T)
            self.d2Symmetric = np.array_equal(np.fliplr(self.m), (np.fliplr(self.m)).T)
        else:
            self.d1Symmetric = False
            self.d2Symmetric = False

        self.isRectangle = 255 not in np.unique(m)
        self.isSquare = self.isRectangle and self.shape[0]==self.shape[1]
        
        if self.isRectangle and self.nColors > 1:
            self.subshapes = detectShapes(self.m, background=self.colorCount.most_common(1)[0][0],\
                                              singleColor=True, diagonals=False)
        self.nHoles = self.getNHoles()
        
        if self.nColors==1:
            self.isFullFrame = self.isFullFrame()
            self.isPartialFrame = self.isPartialFrame()
        else:
            self.isFullFrame = False
            self.isPartialFrame = False
        
        if self.nColors==1:
            self.boolFeatures = []
            for c in range(10):
                self.boolFeatures.append(self.color==c)
            self.boolFeatures.append(self.isBorder)
            self.boolFeatures.append(not self.isBorder)
            self.boolFeatures.append(self.lrSymmetric)
            self.boolFeatures.append(self.udSymmetric)
            self.boolFeatures.append(self.d1Symmetric)
            self.boolFeatures.append(self.d2Symmetric)
            self.boolFeatures.append(self.isSquare)
            self.boolFeatures.append(self.isRectangle)
            for nPix in range(1,30):
                self.boolFeatures.append(self.nPixels==nPix)
            self.boolFeatures.append((self.nPixels%2)==0)
            self.boolFeatures.append((self.nPixels%2)==1)
    
    def hasSameShape(self, other, sameColor=False, samePosition=False, rotation=False, \
                     mirror=False, scaling=False):
        """
        Checks whether the Shape "other" is the same Shape as the current one,
        modulo the given parameters.
        ...
        Parameters
        ----------
        other: Shape
            Shape to be compared with the current one.
        sameColor: bool
            True if we require the colors to be the same. Default is False.
        samePosition: bool
            True if we require both Shapes to be in the same position of the
            Matrix. Default is False.
        rotation: bool
            True if we allow one Shape to be the rotated version of the other
            Shape. Rotations of 90, 180 and 270 degrees are considered.
            Default is False.
        mirror: bool
            True if we allow one Shape to be the other one mirrored. Only
            Left-Right and Up-Down mirrorings are always considered. Default
            is False.
        scaling: bool
            True if we allow one shape to be equal to the other Shape modulo
            scaling. Default is False.
        """
        if samePosition:
            if self.position != other.position:
                return False
        if sameColor:
            m1 = self.m
            m2 = other.m
        else:
            m1 = self.shapeDummyMatrix()
            m2 = other.shapeDummyMatrix()
        if scaling and m1.shape!=m2.shape:
            def multiplyPixels(matrix, factor):
                m = np.zeros(tuple(s * f for s, f in zip(matrix.shape, factor)), dtype=np.uint8)
                for i,j in np.ndindex(matrix.shape):
                    for k,l in np.ndindex(factor):
                        m[i*factor[0]+k, j*factor[1]+l] = matrix[i,j]
                return m
            
            if (m1.shape[0]%m2.shape[0])==0 and (m1.shape[1]%m2.shape[1])==0:
                factor = (int(m1.shape[0]/m2.shape[0]), int(m1.shape[1]/m2.shape[1]))
                m2 = multiplyPixels(m2, factor)
            elif (m2.shape[0]%m1.shape[0])==0 and (m2.shape[1]%m1.shape[1])==0:
                factor = (int(m2.shape[0]/m1.shape[0]), int(m2.shape[1]/m1.shape[1]))
                m1 = multiplyPixels(m1, factor)
            elif rotation and (m1.shape[0]%m2.shape[1])==0 and (m1.shape[1]%m2.shape[0])==0:
                factor = (int(m1.shape[0]/m2.shape[1]), int(m1.shape[1]/m2.shape[0]))
                m2 = multiplyPixels(m2, factor)
            elif rotation and (m2.shape[0]%m1.shape[1])==0 and (m2.shape[1]%m1.shape[0])==0:
                factor = (int(m2.shape[0]/m1.shape[1]), int(m2.shape[1]/m1.shape[0]))
                m1 = multiplyPixels(m1, factor)
            else:
                return False
        if rotation and not mirror:
            if any([np.array_equal(m1, np.rot90(m2,x)) for x in range(1,4)]):
                return True
        if mirror and not rotation:
            if np.array_equal(m1, np.fliplr(m2)) or np.array_equal(m1, np.flipud(m2)):
                return True
        if mirror and rotation:
            for x in range(1, 4):
                if any([np.array_equal(m1, np.rot90(m2,x))\
                        or np.array_equal(m1, np.fliplr(np.rot90(m2,x))) for x in range(0,4)]):
                    return True               
                
        return np.array_equal(m1,m2)
    
    def __eq__(self, other):
        """
        Two Shapes are considered equal if their matrices m are equal.
        """
        if isinstance(other, self.__class__):
            if self.shape != other.shape:
                return False
            return np.array_equal(self.m, other.m)
        else:
            return False
 
    def isSubshape(self, other, sameColor=False, rotation=False, mirror=False):
        """
        Checks whether the Shape "other" is a subShape of the current one,
        modulo the given parameters.
        ...
        Parameters
        ----------
        other: Shape
            Shape to be compared with the current one.
        sameColor: bool
            True if we require the colors to be the same. Default is False.
        rotation: bool
            True if we allow one Shape to be the rotated version of the other
            Shape. Rotations of 90, 180 and 270 degrees are considered.
            Default is False.
        mirror: bool
            True if we allow one Shape to be the other one mirrored. Only
            Left-Right and Up-Down mirrorings are always considered. Default
            is False.
        """
        #return positions
        if rotation:
            m1 = self.m
            for x in range(1,4):
                if Shape(np.rot90(m1,x), 0, 0, 0, self.isBorder).isSubshape(other, sameColor, False, mirror):
                    return True
        if mirror == 'lr':
            if Shape(self.m[::,::-1], 0, 0, 0, self.isBorder).isSubshape(other, sameColor, rotation, False):
                return True
        if mirror == 'ud':
            if Shape(self.m[::-1,::], 0, 0, 0, self.isBorder).isSubshape(other, sameColor, rotation, False):
                return True
        if sameColor:
            if hasattr(self,'color') and hasattr(other,'color') and self.color != other.color:
                return False
        if any(other.shape[i] < self.shape[i] for i in [0,1]):
            return False
        
        for yIn in range(other.shape[1] - self.shape[1] + 1):
            for xIn in range(other.shape[0] - self.shape[0] + 1):
                if sameColor:
                    if np.all(np.logical_or((self.m == other.m[xIn: xIn + self.shape[0], yIn: yIn + self.shape[1]]),\
                                            self.m==255)):
                        return True
                else:
                    if set([tuple(np.add(ps,[xIn,yIn])) for ps in self.pixels]) <= other.pixels:
                        return True
        return False
    
    def shapeDummyMatrix(self):
        """
        Returns the smallest possible matrix containing the shape. The values
        of the matrix are ones and zeros, depending on whether the pixel is a
        Shape pixel or not.
        """
        return (self.m!=255).astype(np.uint8) 
    
    def hasFeatures(self, features):
        """
        Given a list of features, this function returns True if the current
        Shape has the given features. Otherwise, it returns False.
        The parameter features is a list of boolean values.
        """
        for i in range(len(features)):
            if features[i] and not self.boolFeatures[i]:
                return False
        return True

    def getNHoles(self):
        """
        Returns the number of holes of the Shape.
        """
        nHoles = 0
        m = self.m
        seen = np.zeros((self.shape[0], self.shape[1]), dtype=np.bool)
        def isInHole(i,j):
            if i<0 or j<0 or i>self.shape[0]-1 or j>self.shape[1]-1:
                return False
            if seen[i,j] or m[i,j] != 255:
                return True
            seen[i,j] = True
            ret = isInHole(i+1,j)*isInHole(i-1,j)*isInHole(i,j+1)*isInHole(i,j-1)
            return ret
        for i,j in np.ndindex(m.shape):
            if m[i,j] == 255 and not seen[i,j]:
                if isInHole(i,j):
                    nHoles += 1
        return nHoles        

    def isPartialFrame(self):
        """
        Checks whether the Shape is a partial Frame.
        """
        if self.shape[0] < 4 or self.shape[1] < 4 or len(self.pixels) < 4:
            return False
        if len(np.unique(self.m[1:-1,1:-1])) > 1 or self.color in np.unique(self.m[1:-1,1:-1]):
            return False
        return True
    
    def isFullFrame(self):
        """
        Checks whether the Shape is a full Frame.
        """
        if self.shape[0]<3 or self.shape[1]<3:
            return False
        for i in range(self.shape[0]):
            if self.m[i,0]==255 or self.m[i,self.shape[1]-1]==255:
                return False
        for j in range(self.shape[1]):
            if self.m[0,j]==255 or self.m[self.shape[0]-1,j]==255:
                return False
            
        # We require fullFrames to have less than 20% of the pixels inside the
        # frame of the same color of the frame
        
        if self.nPixels - 2*(self.shape[0]+self.shape[1]-2) < 0.2*(self.shape[0]-2)*(self.shape[1]-2):
            return True
        
        return False

def detectShapesByColor(x, background):
    shapes = []
    for c in range(10):
        if c == background or c not in x:
            continue
        mc = np.zeros(x.shape, dtype=int)
        mc[x==c] = c
        mc[x!=c] = 255
        x1, x2, y1, y2 = 0, mc.shape[0]-1, 0, mc.shape[1]-1
        while x1 <= x2 and np.all(mc[x1,:] == 255):
            x1 += 1 
        while x2 >= x1 and np.all(mc[x2,:] == 255):
            x2 -= 1
        while y1 <= y2 and np.all(mc[:,y1] == 255):
            y1 += 1
        while y2 >= y1 and np.all(mc[:,y2] == 255):
            y2 -= 1
        m = mc[x1:x2+1,y1:y2+1]
        s = Shape(m.copy(), x1, y1, background, False)
        shapes.append(s)
    return shapes

def detectShapes(x, background, singleColor=False, diagonals=False):
    """
    Given a numpy array x (2D), returns a list of the Shapes present in x.
    ...
    Parameters
    ----------
    x: numpy.ndarray
        The matrix we want to detect Shapes from.
    background: int
        The background color of the matrix
    singleColor: bool
        True if we want all the shapes to have only one color. Default is
        False.
    diagonals: bool
        True if we allow pixels of the Shape to be connected diagonally.
        Default is False.
    """
    # Helper function to add pixels to a shape
    def addPixelsAround(i,j):
        def addPixel(i,j):
            if i < 0 or j < 0 or i > iMax or j > jMax or seen[i,j] == True:
                return
            if singleColor:
                if x[i,j] != color:
                    return
                newShape[i,j] = color
            else:
                if x[i,j] == background:
                    return
                newShape[i,j] = x[i,j]
            seen[i,j] = True                
            addPixelsAround(i,j)
        
        addPixel(i-1,j)
        addPixel(i+1,j)
        addPixel(i,j-1)
        addPixel(i,j+1)
        
        if diagonals:
            addPixel(i-1,j-1)
            addPixel(i-1,j+1)
            addPixel(i+1,j-1)
            addPixel(i+1,j+1)
            
    def crop(matrix):
        ret = matrix.copy()
        for k in range(x.shape[0]):
            if any(matrix[k,:] != 255): # -1==255 for dtype=np.uint8
                x0 = k
                break
        for k in reversed(range(x.shape[0])):
            if any(matrix[k,:] != 255): # -1==255 for dtype=np.uint8
                x1 = k
                break
        for k in range(x.shape[1]):
            if any(matrix[:,k] != 255): # -1==255 for dtype=np.uint8
                y0 = k
                break
        for k in reversed(range(x.shape[1])):
            if any(matrix[:,k] != 255): # -1==255 for dtype=np.uint8
                y1 = k
                break
        return ret[x0:x1+1,y0:y1+1], x0, y0
                
    shapes = []
    seen = np.zeros(x.shape, dtype=bool)
    iMax = x.shape[0]-1
    jMax = x.shape[1]-1
    for i, j in np.ndindex(x.shape):
        if seen[i,j] == False:
            seen[i,j] = True
            if not singleColor and x[i,j]==background:
                continue
            newShape = np.full((x.shape), -1, dtype=np.uint8)
            newShape[i,j] = x[i,j]
            if singleColor:
                color = x[i][j]
            addPixelsAround(i,j)
            m, xPos, yPos = crop(newShape)
            isBorder = xPos==0 or yPos==0 or (xPos+m.shape[0]==x.shape[0]) or (yPos+m.shape[1]==x.shape[1])
            s = Shape(m.copy(), xPos, yPos, background, isBorder)
            shapes.append(s)
    return shapes

def detectIsolatedPixels(matrix, dShapeList):
    pixList = []
    for sh in dShapeList:
        if sh.nPixels > 1 or sh.color == matrix.backgroundColor:
            continue
        else:
            cc = set()
            for i,j in np.ndindex(3, 3):
                if i - 1 + sh.position[0] < matrix.shape[0] and i - 1 + sh.position[0] >= 0 \
                        and j - 1 + sh.position[1] < matrix.shape[1] and j - 1 + sh.position[1] >= 0:
                    cc  = cc.union(set([matrix.m[i - 1 + sh.position[0],j - 1 + sh.position[1]]]))
            if len(cc) == 2:
                pixList.append(sh)
    return pixList

# %% Class Matrix
class Matrix():
    """
    An object of the class Matrix stores all the relevant information about
    any matrix, be it input matrix, output matrix, from the training samples
    or from the test samples.
    ...
    Main Attributes
    ----------
    m: numpy.ndarray
        The matrix
    shape: tuple (int, int)
        Shape of m
    colors: set
        Colors present in the matrix
    colorCount: collections.Counter
        Dictionary containing the number of appearances of each color.
    backgroundColor: int
        The background color of the matrix
    shapes: list (of Shapes)
        List of Shapes present in the matrix. The Shapes have only one color,
        and cannot contain diagonal connections.
    dShapes: list (of Shapes)
        List of Shapes present in the matrix. The Shapes have only one color,
        and allow diagonal connections.
    multicolorShapes: list (of Shapes)
        List of Shapes present in the matrix. The Shapes can have many colors,
        and cannot contain diagonal connections. They just cannot contain the
        background color.
    multicolorDShapes: list (of Shapes)
        List of Shapes present in the matrix. The Shapes can have many colors,
        and allow diagonal connections. They just cannot contain the
        background color.
    frontiers: list (of Frontiers)
        List of Frontiers of the matrix.
    isGrid: bool
        True if the matrix contains a symmetric grid. False otherwise.
    grid: Grid
        The Grid present in the matrix (there can only be one). This attribute
        is only defined if isGrid.
    isAsymmetricGrid: bool
        True if the matrix contains an asymmetric grid. False otherwise.
    asymmetricGrid: Grid
        The asymmetric Grid present in the matrix (there can only be one).
        This attribute is only defined if isAsymmetricGrid.
    """
    def __init__(self, m, detectGrid=True, backgroundColor=None):
        if type(m) == Matrix:
            return m
        
        self.m = np.array(m)
        
        # interesting properties:
        
        # Dimensions
        self.shape = self.m.shape
        self.nElements = self.m.size
        
        # Counter of colors
        self.colorCount = self.getColors()
        self.colors = set(self.colorCount.keys())
        self.nColors = len(self.colorCount)
        
        # Background color
        if backgroundColor==None:
            self.backgroundColor = max(self.colorCount, key=self.colorCount.get)
        else:
            self.backgroundColor = backgroundColor
        
        # Shapes
        self.shapes = detectShapes(self.m, self.backgroundColor, singleColor=True)
        self.nShapes = len(self.shapes)
        self.dShapes = detectShapes(self.m, self.backgroundColor, singleColor=True, diagonals=True)
        self.nDShapes = len(self.dShapes)
        self.fullFrames = [shape for shape in self.shapes if shape.isFullFrame]
        self.fullFrames = sorted(self.fullFrames, key=lambda x: x.shape[0]*x.shape[1], reverse=True)
        self.shapesByColor = detectShapesByColor(self.m, self.backgroundColor)
        self.partialFrames = [shape for shape in self.shapesByColor if shape.isPartialFrame]
        self.isolatedPixels = detectIsolatedPixels(self, self.dShapes)
        self.nIsolatedPixels = len(self.isolatedPixels)
        
        self.shapeColorCounter = Counter([s.color for s in self.shapes])
        self.blanks = []
        for s in self.shapes:
            if s.isRectangle and self.shapeColorCounter[s.color]==1:
                self.blanks.append(s)
            
        # Frontiers
        self.frontiers = detectFrontiers(self.m)
        self.frontierColors = [f.color for f in self.frontiers]
        if len(self.frontiers) == 0:
            self.allFrontiersEqualColor = False
        else: self.allFrontiersEqualColor = (self.frontierColors.count(self.frontiers[0]) ==\
                                         len(self.frontiers))
        # Check if it's a grid and the dimensions of the cells
        self.isGrid = False
        self.isAsymmetricGrid = False
        if detectGrid:
            for fc in set(self.frontierColors):
                possibleGrid = [f for f in self.frontiers if f.color==fc]
                possibleGrid = Grid(self.m, possibleGrid)
                if possibleGrid.nCells>1:
                    if possibleGrid.allCellsSameShape:
                        self.grid = copy.deepcopy(possibleGrid)
                        self.isGrid = True
                        self.asymmetricGrid = copy.deepcopy(possibleGrid)
                        self.isAsymmetricGrid = True
                        break
                    else:
                        self.asymmetricGrid = copy.deepcopy(possibleGrid)
                        self.isAsymmetricGrid=True
                        
        # Shape-based backgroundColor
        if not self.isGrid:
            for shape in self.shapes:
                if shape.shape==self.shape:
                    self.backgroundColor = shape.color
                    break
        # Define multicolor shapes based on the background color
        self.multicolorShapes = detectShapes(self.m, self.backgroundColor)
        self.multicolorDShapes = detectShapes(self.m, self.backgroundColor, diagonals=True)
        self.dummyMatrix = (self.m!=self.backgroundColor).astype(np.uint8) 
        # Symmetries
        self.lrSymmetric = np.array_equal(self.m, np.fliplr(self.m))
        # Up-Down
        self.udSymmetric = np.array_equal(self.m, np.flipud(self.m))
        # Diagonals (only if square)
        if self.m.shape[0] == self.m.shape[1]:
            self.d1Symmetric = np.array_equal(self.m, self.m.T)
            self.d2Symmetric = np.array_equal(np.fliplr(self.m), (np.fliplr(self.m)).T)
        else:
            self.d1Symmetric = False
            self.d2Symmetric = False
        self.totalSymmetric = self.lrSymmetric and self.udSymmetric and \
        self.d1Symmetric and self.d2Symmetric
        
        self.fullBorders = []
        for f in self.frontiers:
            if f.color != self.backgroundColor:
                if f.position==0:
                    self.fullBorders.append(f)
                elif (f.direction=='h' and f.position==self.shape[0]-1) or\
                (f.direction=='v' and f.position==self.shape[1]-1):
                    self.fullBorders.append(f)
               
        self.isVertical = False
        self.isHorizontal = False
        if len(self.frontiers)!=0:
            self.isVertical = all([f.direction=='v' for f in self.frontiers])
            self.isHorizontal = all([f.direction=='h' for f in self.frontiers])
    
    def getColors(self):
        unique, counts = np.unique(self.m, return_counts=True)
        return dict(zip(unique, counts))
    
    def getShapes(self, color=None, bigOrSmall=None, isBorder=None, diag=False):
        """
        Return a list of the shapes meeting the required specifications.
        """
        if diag:
            candidates = self.dShapes
        else:
            candidates = self.shapes
        if color != None:
            candidates = [c for c in candidates if c.color == color]
        if isBorder==True:
            candidates = [c for c in candidates if c.isBorder]
        if isBorder==False:
            candidates = [c for c in candidates if not c.isBorder]
        if len(candidates) ==  0:
            return []
        sizes = [c.nPixels for c in candidates]
        if bigOrSmall == "big":
            maxSize = max(sizes)
            return [c for c in candidates if c.nPixels==maxSize]
        elif bigOrSmall == "small":
            minSize = min(sizes)
            return [c for c in candidates if c.nPixels==minSize]
        else:
            return candidates
        
    def followsColPattern(self):
        """
        This function checks whether the matrix follows a pattern of lines or
        columns being always the same (task 771 for example).
        Meant to be used for the output matrix mainly.
        It returns a number (length of the pattern) and "row" or "col".
        """
        m = self.m.copy()
        col0 = m[:,0]
        for i in range(1,int(m.shape[1]/2)+1):
            if np.all(col0 == m[:,i]):
                isPattern=True
                for j in range(i):
                    k=0
                    while k*i+j < m.shape[1]:
                        if np.any(m[:,j] != m[:,k*i+j]):
                            isPattern=False
                            break
                        k+=1
                    if not isPattern:
                        break
                if isPattern:
                    return i
        return False
    
    def followsRowPattern(self):
        m = self.m.copy()
        row0 = m[0,:]
        for i in range(1,int(m.shape[0]/2)+1):
            if np.all(row0 == m[i,:]):
                isPattern=True
                for j in range(i):
                    k=0
                    while k*i+j < m.shape[0]:
                        if np.any(m[j,:] != m[k*i+j,:]):
                            isPattern=False
                            break
                        k+=1
                    if not isPattern:
                        break
                if isPattern:
                    return i
        return False
    
    def isUniqueShape(self, shape):
        count = 0
        for sh in self.shapes:
            if sh.hasSameShape(shape):
                count += 1
        if count==1:
            return True
        return False
    
    def getShapeAttributes(self, backgroundColor=0, singleColor=True, diagonals=True):
        '''
        Returns list of shape attributes that matches list of shapes
        Add:
            - is border
            - has neighbors
            - is reference
            - is referenced
        '''
        if singleColor: 
            if diagonals:   
                shapeList = [sh for sh in self.dShapes]
            else:   
                shapeList = [sh for sh in self.shapes]
            if len([sh for sh in shapeList if sh.color != backgroundColor]) == 0:
                return [set() for sh in shapeList]
        else:
            if diagonals: 
                shapeList = [sh for sh in self.multicolorDShapes]
            else:
                shapeList = [sh for sh in self.multicolorShapes]
            if len(shapeList) == 0:
                return [set()]
        attrList =[[] for i in range(len(shapeList))]
        if singleColor:
            cc = Counter([sh.color for sh in shapeList])
        if singleColor:
            sc = Counter([sh.nPixels for sh in shapeList if sh.color != backgroundColor])
        else:
            sc = Counter([sh.nPixels for sh in shapeList])
        largest, smallest, mcopies, mcolors = -1, 1000, 0, 0
        if singleColor:
            maxH, minH = max([sh.nHoles for sh in shapeList if sh.color != backgroundColor]),\
                            min([sh.nHoles for sh in shapeList if sh.color != backgroundColor])
        ila, ism = [], []
        for i in range(len(shapeList)):
            #color count
            if singleColor:
                if shapeList[i].color == backgroundColor:
                    attrList[i].append(-1)
                    continue
                else:
                    attrList[i].append(shapeList[i].color)
            else:
                attrList[i].append(shapeList[i].nColors)
                if shapeList[i].nColors > mcolors:
                    mcolors = shapeList[i].nColors
            #copies
            if singleColor:
                attrList[i] = [np.count_nonzero([np.all(shapeList[i].pixels == osh.pixels) for osh in shapeList])] + attrList[i]
                if attrList[i][0] > mcopies:
                    mcopies = attrList[i][0]
            else: 
                attrList[i] = [np.count_nonzero([shapeList[i] == osh for osh in shapeList])] + attrList[i]
                if attrList[i][0] > mcopies:
                    mcopies = attrList[i][0]
            #unique color?
            if singleColor:
                if cc[shapeList[i].color] == 1:
                    attrList[i].append('UnCo')
            #more of x color?
            if not singleColor:
                for c in range(10):
                    if shapeList[i].colorCount[c] > 0 and  shapeList[i].colorCount[c] == max([sh.colorCount[c] for sh in shapeList]):
                        attrList[i].append('mo'+str(c))    
            #largest?
            if len(shapeList[i].pixels) >= largest:
                ila += [i]
                if len(shapeList[i].pixels) > largest:
                    largest = len(shapeList[i].pixels)
                    ila = [i]
            #smallest?
            if len(shapeList[i].pixels) <= smallest:
                ism += [i]
                if len(shapeList[i].pixels) < smallest:
                    smallest = len(shapeList[i].pixels)
                    ism = [i]
            #unique size
            if sc[shapeList[i].nPixels] == 1 and len(sc) == 2:
                attrList[i].append('UnSi')
            #symmetric?
            if shapeList[i].lrSymmetric:
                attrList[i].append('LrSy')
            else:
                attrList[i].append('NlrSy')
            if shapeList[i].udSymmetric:
                attrList[i].append('UdSy')
            else:
                attrList[i].append('NudSy')
            if shapeList[i].d1Symmetric: 
                attrList[i].append('D1Sy')
            else:
                attrList[i].append('ND1Sy')
            if shapeList[i].d2Symmetric:
                attrList[i].append('D2Sy')
            else:
                attrList[i].append('ND2Sy')
            attrList[i].append(shapeList[i].position)
            #pixels
            if len(shapeList[i].pixels) == 1:
                attrList[i].append('PiXl')
            #holes
            if singleColor:
                if maxH>minH:
                    if shapeList[i].nHoles == maxH:
                        attrList[i].append('MoHo')
                    elif shapeList[i].nHoles == minH:
                        attrList[i].append('LeHo')           
            #is referenced by a full/partial frame?
                if any((shapeList[i].position[0] >= fr.position[0] and shapeList[i].position[1] >= fr.position[1]\
                        and shapeList[i].position[0] + shapeList[i].shape[0] <= fr.position[0] + fr.shape[0] and\
                        shapeList[i].position[1] + shapeList[i].shape[1] <= fr.position[1] + fr.shape[1] and\
                        shapeList[i].color != fr.color) for fr in self.partialFrames):
                    attrList[i].append('IsRef')
                if any((shapeList[i].position[0] >= fr.position[0] and shapeList[i].position[1] >= fr.position[1]\
                        and shapeList[i].position[0] + shapeList[i].shape[0] <= fr.position[0] + fr.shape[0] and\
                        shapeList[i].position[1] + shapeList[i].shape[1] <= fr.position[1] + fr.shape[1] and\
                        shapeList[i].color != fr.color) for fr in self.fullFrames):
                    attrList[i].append('IsFRef')
    
        if len(ism) == 1:
            attrList[ism[0]].append('SmSh')
        if len(ila) == 1:
            attrList[ila[0]].append('LaSh')
        for i in range(len(shapeList)):
            if len(attrList[i]) > 0 and attrList[i][0] == mcopies:
                attrList[i].append('MoCo')
        if not singleColor:
            for i in range(len(shapeList)):
                if len(attrList[i]) > 0 and attrList[i][1] == mcolors:
                    attrList[i].append('MoCl')
        if [l[0] for l in attrList].count(1) == 1:
            for i in range(len(shapeList)):
                if len(attrList[i]) > 0 and attrList[i][0] == 1:
                    attrList[i].append('UnSh')
                    break
        return [set(l[1:]) for l in attrList]
            

# %% Class Sample
class Sample():
    """
    An object of the class Sample stores all the information refering to a
    training or test sample. Almost all of the attributes are only set if the
    sample is a training sample or the init parameter "submission" is set to
    False.
    ...
    Main Attributes
    ---------------
    inMatrix: Matrix
        The input Matrix.
    outMatrix: Matrix
        The output Matrix.
    sameShape: bool
        True if inMatrix and outMatrix have the same shape. False otherwise.
    colors: set
        Colors present in either the input Matrix or the output Matrix.
    commonColors: set
        Colors present in both the inputMatrix and the output Matrix.
    fixedColors: set
        Colors that appear both in the input and in the output, and in the
        exact same pixels. Only defined if sameShape.
    changedInColors: set
        Colors that are not fixed and are such that, if a pixel in the output
        matrix has that color, then it has the same color in the input matrix.
        Only defined if sameShape.
    changedOutColors: set
        Colors that are not fixed and are such that, if a pixel in the input
        matrix has that color, then it has the same color in the output matrix.
        Only defined if sameShape.
    sameColorCount: bool
        True if the input and the output matrices have the same color count.
    commonShapes: list (of Shapes)
        Shapes that appear both in the input matrix and the output matrix.
    gridIsUnchanged: bool
        True if there is the exact same Grid both in the input and in the
        output matrices.
    """
    def __init__(self, s, trainOrTest, submission=False, backgroundColor=None):
        
        self.inMatrix = Matrix(s['input'], backgroundColor=backgroundColor)
        
        if trainOrTest == "train" or submission==False:
            self.outMatrix = Matrix(s['output'], backgroundColor=backgroundColor)
                    
            # We want to compare the input and the output
            # Do they have the same dimensions?
            self.sameHeight = self.inMatrix.shape[0] == self.outMatrix.shape[0]
            self.sameWidth = self.inMatrix.shape[1] == self.outMatrix.shape[1]
            self.sameShape = self.sameHeight and self.sameWidth
            
            # Is the input shape a factor of the output shape?
            # Or the other way around?
            if not self.sameShape:
                if (self.inMatrix.shape[0] % self.outMatrix.shape[0]) == 0 and \
                (self.inMatrix.shape[1] % self.outMatrix.shape[1]) == 0 :
                    self.outShapeFactor = (int(self.inMatrix.shape[0]/self.outMatrix.shape[0]),\
                                           int(self.inMatrix.shape[1]/self.outMatrix.shape[1]))
                if (self.outMatrix.shape[0] % self.inMatrix.shape[0]) == 0 and \
                (self.outMatrix.shape[1] % self.inMatrix.shape[1]) == 0 :
                    self.inShapeFactor = (int(self.outMatrix.shape[0]/self.inMatrix.shape[0]),\
                                          int(self.outMatrix.shape[1]/self.inMatrix.shape[1]))

            # Is one a subset of the other? for now always includes diagonals
            self.inSmallerThanOut = all(self.inMatrix.shape[i] <= self.outMatrix.shape[i] for i in [0,1]) and not self.sameShape
            self.outSmallerThanIn = all(self.inMatrix.shape[i] >= self.outMatrix.shape[i] for i in [0,1]) and not self.sameShape
    
            #R: Is the output a shape (faster than checking if is a subset?
    
            if self.outSmallerThanIn:
                #check if output is the size of a multicolored shape
                self.outIsInMulticolorShapeSize = any((sh.shape == self.outMatrix.shape) for sh in self.inMatrix.multicolorShapes)
                self.outIsInMulticolorDShapeSize = any((sh.shape == self.outMatrix.shape) for sh in self.inMatrix.multicolorDShapes)
            self.commonShapes, self.commonDShapes, self.commonMulticolorShapes, self.commonMulticolorDShapes = [], [], [], []
            if len(self.inMatrix.shapes) < 15 or len(self.outMatrix.shapes) < 10:
                self.commonShapes = self.getCommonShapes(diagonal=False, sameColor=True,\
                                                     multicolor=False, rotation=True, scaling=True, mirror=True)
            if len(self.inMatrix.dShapes) < 15 or len(self.outMatrix.dShapes) < 10:
                self.commonDShapes = self.getCommonShapes(diagonal=True, sameColor=True,\
                                                      multicolor=False, rotation=True, scaling=True, mirror=True)
            if len(self.inMatrix.multicolorShapes) < 15 or len(self.outMatrix.multicolorShapes) < 10:
                self.commonMulticolorShapes = self.getCommonShapes(diagonal=False, sameColor=True,\
                                                               multicolor=True, rotation=True, scaling=True, mirror=True)
            if len(self.inMatrix.multicolorDShapes) < 15 or len(self.outMatrix.multicolorDShapes) < 10:
                self.commonMulticolorDShapes = self.getCommonShapes(diagonal=True, sameColor=True,\
                                                                multicolor=True, rotation=True, scaling=True, mirror=True)

            # Which colors are there in the sample?
            self.colors = set(self.inMatrix.colors | self.outMatrix.colors)
            self.commonColors = set(self.inMatrix.colors & self.outMatrix.colors)
            self.nColors = len(self.colors)
            # Do they have the same colors?
            self.sameColors = len(self.colors) == len(self.commonColors)
            # Do they have the same number of colors?
            self.sameNumColors = self.inMatrix.nColors == self.outMatrix.nColors
            # Does output contain all input colors or viceversa?
            self.inHasOutColors = self.outMatrix.colors <= self.inMatrix.colors  
            self.outHasInColors = self.inMatrix.colors <= self.outMatrix.colors
            if self.sameShape:
                # Which pixels changes happened? How many times?
                self.changedPixels = Counter()
                self.sameColorCount = self.inMatrix.colorCount == self.outMatrix.colorCount
                for i, j in np.ndindex(self.inMatrix.shape):
                    if self.inMatrix.m[i,j] != self.outMatrix.m[i,j]:
                        self.changedPixels[(self.inMatrix.m[i,j], self.outMatrix.m[i,j])] += 1
                # Are any of these changes complete? (i.e. all pixels of one color are changed to another one)
                self.completeColorChanges = set(change for change in self.changedPixels.keys() if\
                                             self.changedPixels[change]==self.inMatrix.colorCount[change[0]] and\
                                             change[0] not in self.outMatrix.colorCount.keys())
                self.allColorChangesAreComplete = len(self.changedPixels) == len(self.completeColorChanges)
                # Does any color never change?
                self.changedInColors = set(change[0] for change in self.changedPixels.keys())
                self.changedOutColors = set(change[1] for change in self.changedPixels.keys())
                self.unchangedColors = set(x for x in self.colors if x not in set.union(self.changedInColors, self.changedOutColors))
                # Colors that stay unchanged
                self.fixedColors = set(x for x in self.colors if x not in set.union(self.changedInColors, self.changedOutColors))
            
            if self.sameShape and self.sameColorCount:
                self.sameRowCount = True
                for r in range(self.inMatrix.shape[0]):
                    _,inCounts = np.unique(self.inMatrix.m[r,:], return_counts=True)
                    _,outCounts = np.unique(self.outMatrix.m[r,:], return_counts=True)
                    if not np.array_equal(inCounts, outCounts):
                        self.sameRowCount = False
                        break
                self.sameColCount = True
                for c in range(self.inMatrix.shape[1]):
                    _,inCounts = np.unique(self.inMatrix.m[:,c], return_counts=True)
                    _,outCounts = np.unique(self.outMatrix.m[:,c], return_counts=True)
                    if not np.array_equal(inCounts, outCounts):
                        self.sameColCount = False
                        break
                    
            # Shapes in the input that are fixed
            if self.sameShape:
                self.fixedShapes = []
                for sh in self.inMatrix.shapes:
                    if sh.color in self.fixedColors:
                        continue
                    shapeIsFixed = True
                    for i,j in np.ndindex(sh.shape):
                        if sh.m[i,j] != 255:
                            if self.outMatrix.m[sh.position[0]+i,sh.position[1]+j]!=sh.m[i,j]:
                                shapeIsFixed=False
                                break
                    if shapeIsFixed:
                        self.fixedShapes.append(sh)
                    
            # Frames
            self.commonFullFrames = [f for f in self.inMatrix.fullFrames if f in self.outMatrix.fullFrames]
            if len(self.inMatrix.fullFrames)==1:
                frameM = self.inMatrix.fullFrames[0].m.copy()
                frameM[frameM==255] = self.inMatrix.fullFrames[0].background
                if frameM.shape==self.outMatrix.shape:
                    self.frameIsOutShape = True
                elif frameM.shape==(self.outMatrix.shape[0]+1, self.outMatrix.shape[1]+1):
                    self.frameInsideIsOutShape = True
            
            # Grids
            # Is the grid the same in the input and in the output?
            self.gridIsUnchanged = self.inMatrix.isGrid and self.outMatrix.isGrid \
            and self.inMatrix.grid == self.outMatrix.grid
            # Does the shape of the grid cells determine the output shape?
            if hasattr(self.inMatrix, "grid") and self.inMatrix.grid.allCellsSameShape:
                self.gridCellIsOutputShape = self.outMatrix.shape == self.inMatrix.grid.cellShape
            # Does the shape of the input determine the shape of the grid cells of the output?
            if hasattr(self.outMatrix, "grid") and self.outMatrix.grid.allCellsSameShape:
                self.gridCellIsInputShape = self.inMatrix.shape == self.outMatrix.grid.cellShape
            # Do all the grid cells have one color?
            if self.gridIsUnchanged:
                self.gridCellsHaveOneColor = self.inMatrix.grid.allCellsHaveOneColor and\
                                             self.outMatrix.grid.allCellsHaveOneColor
            # Asymmetric grids
            self.asymmetricGridIsUnchanged = self.inMatrix.isAsymmetricGrid and self.outMatrix.isAsymmetricGrid \
            and self.inMatrix.asymmetricGrid == self.outMatrix.asymmetricGrid
            if self.asymmetricGridIsUnchanged:
                self.asymmetricGridCellsHaveOneColor = self.inMatrix.asymmetricGrid.allCellsHaveOneColor and\
                self.outMatrix.asymmetricGrid.allCellsHaveOneColor
            
            # Is there a blank to fill?
            self.inputHasBlank = len(self.inMatrix.blanks)>0
            if self.inputHasBlank:
                for s in self.inMatrix.blanks:
                    if s.shape == self.outMatrix.shape:
                        self.blankToFill = s
             
            # Does the output matrix follow a pattern?
            self.followsRowPattern = self.outMatrix.followsRowPattern()
            self.followsColPattern = self.outMatrix.followsColPattern()
            
            # Full borders and horizontal/vertical
            if self.sameShape:
                self.commonFullBorders = []
                for inBorder in self.inMatrix.fullBorders:
                    for outBorder in self.outMatrix.fullBorders:
                        if inBorder==outBorder:
                            self.commonFullBorders.append(inBorder)
                
                self.isHorizontal = self.inMatrix.isHorizontal and self.outMatrix.isHorizontal
                self.isVertical = self.inMatrix.isVertical and self.outMatrix.isVertical

    def getCommonShapes(self, diagonal=True, multicolor=False, sameColor=False, samePosition=False, rotation=False, \
                     mirror=False, scaling=False):
        comSh = []
        if diagonal:
            if not multicolor:
                ishs = self.inMatrix.dShapes
                oshs = self.outMatrix.dShapes
            else:
                ishs = self.inMatrix.multicolorDShapes
                oshs = self.outMatrix.multicolorDShapes
        else:
            if not multicolor:
                ishs = self.inMatrix.shapes
                oshs = self.outMatrix.shapes
            else:
                ishs = self.inMatrix.multicolorShapes
                oshs = self.outMatrix.multicolorShapes
        #Arbitrary: shapes have size < 100 and > 3
        for ish in ishs:
            outCount = 0
            if len(ish.pixels) < 4 or len(ish.pixels) > 100:
                continue
            for osh in oshs:
                if len(osh.pixels) < 4 or len(osh.pixels) > 100:
                    continue
                if ish.hasSameShape(osh, sameColor=sameColor, samePosition=samePosition,\
                                    rotation=rotation, mirror=mirror, scaling=scaling):
                    outCount += 1
            if outCount > 0:
                comSh.append((ish, np.count_nonzero([ish.hasSameShape(ish2, sameColor=sameColor, samePosition=samePosition,\
                                    rotation=rotation, mirror=mirror, scaling=scaling) for ish2 in ishs]), outCount))
        return comSh

# %% Class Task
class Task():
    """
    An object of the class Task stores all the relevant information about an
    ARC task.
    ...
    Main Attributes
    ---------------
    task: dict
        The task given in the standard ARC format.
    index: str
        The name of the task, in hexadecimal characters.
    submission: bool
        True if we don't know the output matrices of the test sample.
    trainSamples: list (of Samples)
        List of the training Samples of the task.
    testSamples: list (of Samples)
        List of the test Samples of the task.
    nTrain: int
        Number of train Samples.
    nTest: int
        Number of test Samples.
    sameInShape: bool
        True if all the input matrices have the same shape. False otherwise.
    sameOutShape: bool
        True if all the output matrices have the same shape. False otherwise.
    sameIOShapes: bool
        True if the input matrices have the same shape as the output matrices
        for every sample. False otherwise.
    colors: set
        The colors that appear in the task.
    totalInColors: set
        The colors that appear in at least one of the input matrices.
    commonInColors: set
        The colors that appear in all the input matrices.
    totalOutColors: set
        The colors that appear in at least one of the output matrices.
    commonOutColors: set
        The colors that appear in all the output matrices.
    fixedColors: set
        The fixedColors that are common to every training Sample.
    commonChangedInColors: set
        The changedInColors that are common to every training Sample.
    commonChangedOutColors: set
        The changedOutColors that are common to every training Sample.
    backgroundColor: int
        If all the input matrices have the same background color, this
        attribute represents it. Otherwise, it is set to -1.
    """
    def __init__(self, t, i, submission=False, backgrounds=None):
        self.task = t
        self.index = i
        self.submission = submission
        
        if backgrounds==None:
            self.trainSamples = [Sample(s, "train", submission) for s in t['train']]
            self.testSamples = [Sample(s, "test", submission) for s in t['test']]
        else:
            self.trainSamples = [Sample(t["train"][s], "train", submission, backgrounds["train"][s]) for s in range(len(t['train']))]
            self.testSamples = [Sample(t["test"][s], "test", submission, backgrounds["test"][s]) for s in range(len(t['test']))]
        
        self.nTrain = len(self.trainSamples)
        self.nTest = len(self.testSamples)
        
        # Common properties I want to know:
        
        # Dimension:
        # Do all input/output matrices have the same shape?
        inShapes = [s.inMatrix.shape for s in self.trainSamples]
        self.sameInShape = self.allEqual(inShapes)
        if self.sameInShape:
            self.inShape = self.trainSamples[0].inMatrix.shape
        outShapes = [s.outMatrix.shape for s in self.trainSamples]
        self.sameOutShape = self.allEqual(outShapes)
        if self.sameOutShape:
            self.outShape = self.trainSamples[0].outMatrix.shape
            
        # Do all output matrices have the same shape as the input matrix?
        self.sameIOShapes = all([s.sameShape for s in self.trainSamples])
        
        # Are the input/output matrices always squared?
        self.inMatricesSquared = all([s.inMatrix.shape[0] == s.inMatrix.shape[1] \
                                      for s in self.trainSamples+self.testSamples])
        self.outMatricesSquared = all([s.outMatrix.shape[0] == s.outMatrix.shape[1] \
                                       for s in self.trainSamples])
    
        # Are shapes of in (out) matrices always a factor of the shape of the 
        # out (in) matrices?
        if all([hasattr(s, 'inShapeFactor') for s in self.trainSamples]):
            if self.allEqual([s.inShapeFactor for s in self.trainSamples]):
                self.inShapeFactor = self.trainSamples[0].inShapeFactor
            elif all([s.inMatrix.shape[0]**2 == s.outMatrix.shape[0] and \
                      s.inMatrix.shape[1]**2 == s.outMatrix.shape[1] \
                      for s in self.trainSamples]):
                self.inShapeFactor = "squared"
            elif all([s.inMatrix.shape[0]**2 == s.outMatrix.shape[0] and \
                      s.inMatrix.shape[1] == s.outMatrix.shape[1] \
                      for s in self.trainSamples]):
                self.inShapeFactor = "xSquared"
            elif all([s.inMatrix.shape[0] == s.outMatrix.shape[0] and \
                      s.inMatrix.shape[1]**2 == s.outMatrix.shape[1] \
                      for s in self.trainSamples]):
                self.inShapeFactor = "ySquared"
            elif all([s.inMatrix.shape[0]*s.inMatrix.nColors == s.outMatrix.shape[0] and \
                     s.inMatrix.shape[1]*s.inMatrix.nColors == s.outMatrix.shape[1] \
                     for s in self.trainSamples]):
                self.inShapeFactor = "nColors"
            elif all([s.inMatrix.shape[0]*(s.inMatrix.nColors-1) == s.outMatrix.shape[0] and \
                     s.inMatrix.shape[1]*(s.inMatrix.nColors-1) == s.outMatrix.shape[1] \
                     for s in self.trainSamples]):
                self.inShapeFactor = "nColors-1"
        if all([hasattr(s, 'outShapeFactor') for s in self.trainSamples]):
            if self.allEqual([s.outShapeFactor for s in self.trainSamples]):
                self.outShapeFactor = self.trainSamples[0].outShapeFactor
                
        # Is the output always smaller?
        self.outSmallerThanIn = all(s.outSmallerThanIn for s in self.trainSamples)
        self.inSmallerThanOut = all(s.inSmallerThanOut for s in self.trainSamples)
        
        # Symmetries:
        # Are all outputs LR, UD, D1 or D2 symmetric?
        self.lrSymmetric = all([s.outMatrix.lrSymmetric for s in self.trainSamples])
        self.udSymmetric = all([s.outMatrix.udSymmetric for s in self.trainSamples])
        self.d1Symmetric = all([s.outMatrix.d1Symmetric for s in self.trainSamples])
        self.d2Symmetric = all([s.outMatrix.d2Symmetric for s in self.trainSamples])
        
        # Colors
        # How many colors are there in the input? Is it always the same number?
        # How many colors are there in the output? Is it always the same number?
        self.sameNumColors = all([s.sameNumColors for s in self.trainSamples])
        self.nInColors = [s.inMatrix.nColors for s in self.trainSamples] + \
        [s.inMatrix.nColors for s in self.testSamples]
        self.sameNInColors = self.allEqual(self.nInColors)
        self.nOutColors = [s.outMatrix.nColors for s in self.trainSamples]
        self.sameNOutColors = self.allEqual(self.nOutColors)
        # Which colors does the input have? Union and intersection.
        self.inColors = [s.inMatrix.colors for s in self.trainSamples+self.testSamples]
        self.commonInColors = set.intersection(*self.inColors)
        self.totalInColors = set.union(*self.inColors)
        # Which colors does the output have? Union and intersection.
        self.outColors = [s.outMatrix.colors for s in self.trainSamples]
        self.commonOutColors = set.intersection(*self.outColors)
        self.totalOutColors = set.union(*self.outColors)
        # Which colors appear in every sample?
        self.sampleColors = [s.colors for s in self.trainSamples]
        self.commonSampleColors = set.intersection(*self.sampleColors)
        self.almostCommonColors = self.commonSampleColors.copy()
        if self.nTrain > 3:
            for color in range(10):
                if color not in self.almostCommonColors:
                    apps = 0
                    for i in range(self.nTrain):
                        if color in self.trainSamples[i].colors:
                            apps += 1
                    if apps == self.nTrain-1:
                        self.almostCommonColors.add(color)
        # Input colors of the test samples
        self.testInColors = [s.inMatrix.colors for s in self.testSamples]
        # Are there the same number of colors in every sample?
        self.sameNSampleColors = self.allEqual([len(sc) for sc in self.sampleColors]) and\
        all([len(s.inMatrix.colors | self.commonOutColors) <= len(self.sampleColors[0]) for s in self.testSamples])
        # How many colors are there in total? Which ones?
        self.colors = self.totalInColors | self.totalOutColors
        self.nColors = len(self.colors)
        # Does the output always have the same colors as the input?
        if self.sameNumColors:
            self.sameIOColors = all([i==j for i,j in zip(self.inColors, self.outColors)])
        if self.sameIOShapes:
            # Do the matrices have the same color count?
            self.sameColorCount = all([s.sameColorCount for s in self.trainSamples])
            if self.sameColorCount:
                self.sameRowCount = all([s.sameRowCount for s in self.trainSamples])
                self.sameColCount = all([s.sameColCount for s in self.trainSamples])
            # Which color changes happen? Union and intersection.
            cc = [set(s.changedPixels.keys()) for s in self.trainSamples]
            self.colorChanges = set.union(*cc)
            self.commonColorChanges = set.intersection(*cc)
            # Does any color always change? (to and from)
            self.changedInColors = [s.changedInColors for s in self.trainSamples]
            self.commonChangedInColors = set.intersection(*self.changedInColors)
            self.changedOutColors = [s.changedOutColors for s in self.trainSamples]
            self.commonChangedOutColors = set.intersection(*self.changedOutColors)
            self.commonOnlyChangedInColors = self.commonChangedInColors - set.union(*self.changedOutColors)
            # Complete color changes
            self.completeColorChanges = [s.completeColorChanges for s in self.trainSamples]
            self.commonCompleteColorChanges = set.intersection(*self.completeColorChanges)
            self.allColorChangesAreComplete = all([s.allColorChangesAreComplete for s in self.trainSamples])
            # Are there any fixed colors?
            self.fixedColors = set.intersection(*[s.fixedColors for s in self.trainSamples])
            self.fixedColors2 = set.union(*[s.fixedColors for s in self.trainSamples]) - \
            set.union(*[s.changedInColors for s in self.trainSamples]) -\
            set.union(*[s.changedOutColors for s in self.trainSamples])
            # Does any color never change?
            if self.commonChangedInColors == set(self.changedInColors[0]):
                self.unchangedColors = set(range(10)) - self.commonChangedInColors
            else:
                self.unchangedColors = [s.unchangedColors for s in self.trainSamples]
                self.unchangedColors = set.intersection(*self.unchangedColors)
                
        # Is the number of pixels changed always the same?
        """
        if self.sameIOShapes:
            self.sameChanges = self.allEqual([s.diffPixels for s in self.trainSamples])
        """
            
        #R: is output a shape in the input
        self.outIsInMulticolorShapeSize = False
        self.outIsInMulticolorDShapeSize = False

        if all([(hasattr(s, "outIsInMulticolorShapeSize") and s.outIsInMulticolorShapeSize) for s in self.trainSamples]):
             self.outIsInMulticolorShapeSize = True
        if all([(hasattr(s, "outIsInMulticolorDShapeSize") and s.outIsInMulticolorDShapeSize) for s in self.trainSamples]):
             self.outIsInMulticolorDShapeSize = True
             
        self.nCommonInOutShapes = min(len(s.commonShapes) for s in self.trainSamples)
        self.nCommonInOutDShapes = min(len(s.commonDShapes) for s in self.trainSamples) 
        self.nCommonInOutMulticolorShapes = min(len(s.commonMulticolorShapes) for s in self.trainSamples)
        self.nCommonInOutMulticolorDShapes = min(len(s.commonMulticolorDShapes) for s in self.trainSamples) 
        
        if self.sameIOShapes:
            self.fixedShapes = []
            for s in self.trainSamples:
                for shape in s.fixedShapes:
                    self.fixedShapes.append(shape)
            self.fixedShapeFeatures = []
            nFeatures = len(self.trainSamples[0].inMatrix.shapes[0].boolFeatures)
            for i in range(nFeatures):
                self.fixedShapeFeatures.append(True)
            for shape in self.fixedShapes:
                self.fixedShapeFeatures = [shape.boolFeatures[i] and self.fixedShapeFeatures[i] \
                                             for i in range(nFeatures)]
        
        # Grids:
        self.inputIsGrid = all([s.inMatrix.isGrid for s in self.trainSamples+self.testSamples])
        self.outputIsGrid = all([s.outMatrix.isGrid for s in self.trainSamples])
        self.hasUnchangedGrid = all([s.gridIsUnchanged for s in self.trainSamples])
        if all([hasattr(s, "gridCellIsOutputShape") for s in self.trainSamples]):
            self.gridCellIsOutputShape = all([s.gridCellIsOutputShape for s in self.trainSamples])
        if all([hasattr(s, "gridCellIsInputShape") for s in self.trainSamples]):
            self.gridCellIsInputShape = all([s.gridCellIsInputShape for s in self.trainSamples])
        if self.hasUnchangedGrid:
            self.gridCellsHaveOneColor = all([s.gridCellsHaveOneColor for s in self.trainSamples])
            self.outGridCellsHaveOneColor = all([s.outMatrix.grid.allCellsHaveOneColor for s in self.trainSamples])
        # Asymmetric grids
        self.inputIsAsymmetricGrid = all([s.inMatrix.isAsymmetricGrid for s in self.trainSamples+self.testSamples])
        self.hasUnchangedAsymmetricGrid = all([s.asymmetricGridIsUnchanged for s in self.trainSamples])
        if self.hasUnchangedAsymmetricGrid:
            self.assymmetricGridCellsHaveOneColor = all([s.asymmetricGridCellsHaveOneColor for s in self.trainSamples])
            self.outAsymmetricGridCellsHaveOneColor = all([s.outMatrix.asymmetricGrid.allCellsHaveOneColor for s in self.trainSamples])
        
        # Background color
        
        # Is there always a background color? Which one?
        if self.allEqual([s.inMatrix.backgroundColor for s in self.trainSamples]) and\
        self.trainSamples[0].inMatrix.backgroundColor == self.testSamples[0].inMatrix.backgroundColor:
            self.backgroundColor = self.trainSamples[0].inMatrix.backgroundColor
        elif self.hasUnchangedAsymmetricGrid and all([s.inMatrix.asymmetricGrid.nCells>6 for s in self.trainSamples]):
            self.backgroundColor = self.trainSamples[0].inMatrix.asymmetricGrid.color
            for sample in self.trainSamples:
                sample.inMatrix.backgroundColor = self.backgroundColor
                sample.outMatrix.backgroundColor = self.backgroundColor
            for sample in self.testSamples:
                sample.inMatrix.backgroundColor = self.backgroundColor
        else:
            self.backgroundColor = -1
            
        self.orderedColors = self.orderColors()
        
        # Shapes:
        # Does the task ONLY involve changing colors of shapes?
        if self.sameIOShapes:
            self.onlyShapeColorChanges = True
            for s in self.trainSamples:
                nShapes = s.inMatrix.nShapes
                if s.outMatrix.nShapes != nShapes:
                    self.onlyShapeColorChanges = False
                    break
                for shapeI in range(nShapes):
                    if not s.inMatrix.shapes[shapeI].hasSameShape(s.outMatrix.shapes[shapeI]):
                        self.onlyShapeColorChanges = False
                        break
                if not self.onlyShapeColorChanges:
                    break
            
            # Get a list with the number of pixels shapes have
            if self.onlyShapeColorChanges:
                nPixels = set()
                for s in self.trainSamples:
                    for shape in s.inMatrix.shapes:
                        nPixels.add(shape.nPixels)
                self.shapePixelNumbers =  list(nPixels)
                
        #R: Are there any common input shapes accross samples?
        self.commonInShapes = []
        for sh1 in self.trainSamples[0].inMatrix.shapes:
            if sh1.color == self.trainSamples[0].inMatrix.backgroundColor:
                continue
            addShape = True
            for s in range(1,self.nTrain):
                if not any([sh1.pixels == sh2.pixels for sh2 in self.trainSamples[s].inMatrix.shapes]):
                    addShape = False
                    break
            if addShape and sh1 not in self.commonInShapes:
                self.commonInShapes.append(sh1)

        self.commonInDShapes = []
        for sh1 in self.trainSamples[0].inMatrix.dShapes:
            if sh1.color == self.trainSamples[0].inMatrix.backgroundColor:
                continue
            addShape = True
            for s in range(1,self.nTrain):
                if not any([sh1.pixels == sh2.pixels for sh2 in self.trainSamples[s].inMatrix.dShapes]):
                    addShape = False
                    break
            if addShape:
                self.commonInDShapes.append(sh1)
        #Does the task use the information of isolated pixels?
        #if all(s.inMatrix.nIsolatedPixels)
        #Does the input always consist in two shapes?
        self.twoShapeTask = (False, False, False, False, 1)
        if all(len(s.inMatrix.multicolorDShapes)==2 for s in self.trainSamples):
            self.twoShapeTask = (True, True, True, False, 1)
            if all(s.inMatrix.multicolorDShapes[0].shape == s.inMatrix.multicolorDShapes[1].shape for s in self.trainSamples):
                self.twoShapeTask = (True, True, True, True, 1)
                
        elif all(len(s.inMatrix.multicolorShapes)==2 for s in self.trainSamples):
            self.twoShapeTask = (True, True, False, False, 1)
            if all(s.inMatrix.multicolorShapes[0].shape == s.inMatrix.multicolorShapes[1].shape for s in self.trainSamples):
                self.twoShapeTask = (True, True, False, True, 1)
        elif all(len(s.inMatrix.dShapes)==2 for s in self.trainSamples):
            self.twoShapeTask = (True, False, True, False, 1)
            if all(s.inMatrix.dShapes[0].shape == s.inMatrix.dShapes[1].shape for s in self.trainSamples):
                self.twoShapeTask = (True, False, True, True, 1)
        if self.inputIsGrid:
            if all(s.inMatrix.grid.nCells == 2 for s in self.trainSamples):
                self.twoShapeTask = (True, False, False, True, 2)
        elif  all((s.inMatrix.shape[0]*2 == s.inMatrix.shape[1] or\
                  s.inMatrix.shape[0] == s.inMatrix.shape[1]*2) for s in self.trainSamples):
                self.twoShapeTask = (True, False, False, True, 3)
                
        #Are all output matrices equal mod nonBackgroundColor?
        if self.sameOutShape:
            self.sameOutDummyMatrix = all(np.all(self.trainSamples[0].outMatrix.dummyMatrix==s.outMatrix.dummyMatrix) for s in self.trainSamples)
        # Frames
        self.hasFullFrame = all([len(s.inMatrix.fullFrames)>0 for s in self.trainSamples])
        self.hasPartialFrame = all([len(s.inMatrix.partialFrames)>0 for s in self.trainSamples])
        # Is the task about filling a blank?
        self.fillTheBlank =  all([hasattr(s, 'blankToFill') for s in self.trainSamples])
                
        # Do all output matrices follow a pattern?
        self.followsRowPattern = all([s.followsRowPattern != False for s in self.trainSamples])
        self.followsColPattern = all([s.followsColPattern != False for s in self.trainSamples])
        if self.followsRowPattern:
            self.rowPatterns = [s.outMatrix.followsRowPattern() for s in self.trainSamples]
        if self.followsColPattern:
            self.colPatterns = [s.outMatrix.followsColPattern() for s in self.trainSamples]
        
        # Full Borders / Requires vertical-horizontal rotation
        if self.sameIOShapes:
            if self.submission:
                self.hasOneFullBorder = all([len(s.commonFullBorders)==1 for s in self.trainSamples])
            else:
                self.hasOneFullBorder = all([hasattr(s, 'commonFullBorders') and len(s.commonFullBorders)==1 for s in self.trainSamples+self.testSamples])
            self.requiresHVRotation = False
            if not (self.allEqual([s.isHorizontal for s in self.trainSamples]) or \
                    self.allEqual([s.isVertical for s in self.trainSamples])):    
                self.requiresHVRotation = all([s.isHorizontal or s.isVertical for s in self.trainSamples])
        
    def allEqual(self, x):
        """
        x is a list.
        Returns true if all elements of x are equal.
        """
        if len(x) == 0:
            return False
        return x.count(x[0]) == len(x)
    
    def orderColors(self):
        """
        The aim of this function is to give the colors a specific order, in
        order to do the OHE in the right way for every sample.
        """
        orderedColors = []
        # 1: Colors that appear in every sample, input and output, and never
        # change. Only valid if t.sameIOShapes
        if self.sameIOShapes:
            for c in self.fixedColors:
                if all([c in sample.inMatrix.colors for sample in self.testSamples]):
                    orderedColors.append(c)
        # 2: Colors that appear in every sample and are always changed from,
        # never changed to.
            for c in self.commonChangedInColors:
                if c not in self.commonChangedOutColors:
                    if all([c in sample.inMatrix.colors for sample in self.testSamples]):
                        if c not in orderedColors:
                            orderedColors.append(c)
        # 3: Colors that appear in every sample and are always changed to,
        # never changed from.
            for c in self.commonChangedOutColors:
                if not all([c in sample.inMatrix.colors for sample in self.trainSamples]):
                    if c not in orderedColors:
                        orderedColors.append(c)
        # 4: Add the background color.
        if self.backgroundColor != -1:
            if self.backgroundColor not in orderedColors:
                orderedColors.append(self.backgroundColor)
        # 5: Other colors that appear in every input.
        for c in self.commonInColors:
            if all([c in sample.inMatrix.colors for sample in self.testSamples]):
                if c not in orderedColors:
                    orderedColors.append(c)
        # 6: Other colors that appear in every output.
        for c in self.commonOutColors:
            if not all([c in sample.inMatrix.colors for sample in self.trainSamples]):
                if c not in orderedColors:
                    orderedColors.append(c)
                
        # TODO Dealing with grids and frames
        
        return orderedColors   
    
#############################################################################
# %% Models
        
class OneConvModel(nn.Module):
    """
    Simple CNN model consisting of only one 2d convolution. Input and output
    tensors have the same shape.
    """
    def __init__(self, ch=10, kernel=3, padVal = -1):
        super(OneConvModel, self).__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=kernel, bias=0)
        self.pad = nn.ConstantPad2d(int((kernel-1)/2), padVal)
        
    def forward(self, x, steps=1):
        for _ in range(steps):
            x = self.conv(self.pad(x))
        return x
    
class LinearModel(nn.Module):
    """
    Model consisting only of a fully connected layer and a given number of
    channels.
    """
    def __init__(self, inSize, outSize, ch):
        super(LinearModel, self).__init__()
        self.inSize = inSize
        self.outSize = outSize
        self.ch = ch
        self.fc = nn.Linear(inSize[0]*inSize[1]*ch, outSize[0]*outSize[1]*ch)
        
    def forward(self, x):
        x = x.view(1, self.inSize[0]*self.inSize[1]*self.ch)
        x = self.fc(x)
        x = x.view(1, self.ch, self.outSize[0]*self.outSize[1])
        return x
    
class LinearModelDummy(nn.Module): #(dummy = 2 channels)
    """
    Model consisting only of a fully connected layer and two channels.
    """
    def __init__(self, inSize, outSize):
        super(LinearModelDummy, self).__init__()
        self.inSize = inSize
        self.outSize = outSize
        self.fc = nn.Linear(inSize[0]*inSize[1]*2, outSize[0]*outSize[1]*2, bias=0)
        
    def forward(self, x):
        x = x.view(1, self.inSize[0]*self.inSize[1]*2)
        x = self.fc(x)
        x = x.view(1, 2, self.outSize[0]*self.outSize[1])
        return x
    
class SimpleLinearModel(nn.Module):
    """
    Model consisting only of a fully connected layer.
    """
    def __init__(self, inSize, outSize):
        super(SimpleLinearModel, self).__init__()
        self.fc = nn.Linear(inSize, outSize)
        
    def forward(self, x):
        x = self.fc(x)
        return x
    
class LSTMTagger(nn.Module):
    """
    Simple LSTM model.
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    
def pixelCorrespondence(t):
    """
    Returns a dictionary. Keys are positions of the output matrix. Values are
    the pixel in the input matrix it corresponds to.
    Function only valid if t.sameInSahpe and t.sameOutShape
    """
    pixelsColoredAllSamples = []
    # In which positions does each color appear?
    for s in t.trainSamples:
        pixelsColored = [[] for i in range(10)]
        m = s.inMatrix.m
        for i,j in np.ndindex(t.inShape):
            pixelsColored[m[i,j]].append((i,j))
        pixelsColoredAllSamples.append(pixelsColored)
    # For each pixel in output matrix, find correspondent pixel in input matrix
    pixelMap = {}
    for i,j in np.ndindex(t.outShape):
        candidates = set()
        for s in range(t.nTrain):
            m = t.trainSamples[s].outMatrix.m
            if len(candidates) == 0:
                candidates = set(pixelsColoredAllSamples[s][m[i,j]])
            else:
                candidates = set(pixelsColoredAllSamples[s][m[i,j]]) & candidates
            if len(candidates) == 0:
                return {}
        pixelMap[(i,j)] = next(iter(candidates))
    
    return pixelMap

###############################################################################
# %% Utils

def identityM(matrix):
    """
    Function that, given Matrix, returns its corresponding numpy.ndarray m
    """
    if isinstance(matrix, np.ndarray):
        return matrix.copy()
    else:
        return matrix.m.copy()

def correctFixedColors(inMatrix, x, fixedColors, onlyChangedInColors):
    """
    Given an input matrix (inMatrix), an output matrix (x) and a set of colors
    that should not change between the input and the output (fixedColors),
    this function returns a copy of x, but correcting the pixels that 
    shouldn't have changed back into the original, unchanged color.
    
    inMatrix and x are required to have the same shape.
    """
    m = x.copy()
    for i,j in np.ndindex(m.shape):
        if inMatrix[i,j] in fixedColors:
            m[i,j] = inMatrix[i,j]
        if m[i,j] in onlyChangedInColors:
            m[i,j] = inMatrix[i,j]
    return m
                
def incorrectPixels(m1, m2):
    """
    Returns the number of incorrect pixels (0 is best).
    """
    if m1.shape != m2.shape:
        return 1000
    return np.sum(m1!=m2)

def deBackgroundizeMatrix(m, color):
    """
    Given a matrix m and a color, this function returns a matrix whose elements
    are 0 or 1, depending on whether the corresponding pixel is of the given
    color or not.
    """
    return np.uint8(m == color)

def relDicts(colors):
    """
    Given a list of colors (numbers from 0 to 9, no repetitions allowed), this
    function returns two dictionaries giving the relationships between the
    color and its index in the list.
    It's just a way to map the colors to list(range(nColors)).
    """
    rel = {}
    for i in range(len(colors)):
        rel[i] = colors[i]
    invRel = {v: k for k,v in rel.items()}
    for i in range(len(colors)):
        rel[i] = [colors[i]]
    return rel, invRel

def dummify(x, nChannels, rel=None):
    """
    Given a matrix and a relationship given by relDicts, this function returns
    a nColors x shape(x) matrix consisting only of ones and zeros. For each
    channel (corresponding to a color), each element will be 1 if in the 
    original matrix x that pixel is of the corresponding color.
    If rel is not specified, it is expected that the values of x range from
    0 to nChannels-1.
    """
    img = np.full((nChannels, x.shape[0], x.shape[1]), 0, dtype=np.uint8)
    if rel==None:
        for i in range(nChannels):
            img[i] = x==i
    else:
        for i in range(len(rel)):
            img[i] = np.isin(x,rel[i])
    return img

def dummifyColor(x, color):
    """
    Given a matrix x and a color, this function returns a 2-by-shape(x) matrix
    of ones and zeros. In one channel, the elements will be 1 if the pixel is
    of the given color. In the other channel, they will be 1 otherwise.
    """
    img = np.full((2, x.shape[0], x.shape[1]), 0, dtype=np.uint8)
    img[0] = x!=color
    img[1] = x==color
    return img

def updateBestFunction(t, f, bestScore, bestFunction, checkPerfect=False, prevScore=None):
    """
    Given a task t, a partial function f, a best score and a best function, 
    this function executes f to all the matrices in t.trainSamples. If the
    resulting score is lower than bestScore, then it returns f and the new
    best score. Otherwise, it returns bestFunction again.
    If the parameter checkPerfect is set to True, it returns f if the
    transformation is perfect, this is, if all the pixels that are changed
    (cannot be zero) are changed correctly. 
    """
    fun = copy.deepcopy(f)
    score = 0
    if checkPerfect:
        changedPixels = 0
    for sample in t.trainSamples:
        pred = fun(sample.inMatrix)
        score += incorrectPixels(sample.outMatrix.m, pred)
        if checkPerfect:
            changedPixels += incorrectPixels(pred, sample.inMatrix.m)
    if score < bestScore:
        bestScore = score
        bestFunction = fun
    if checkPerfect:
        if changedPixels != 0 and prevScore - changedPixels == score: 
            isPerfect = True
        else:
            isPerfect = False
        return fun, score, isPerfect
    return bestFunction, bestScore

# %% Symmetrize

# if t.lrSymmetric or t.udSymmetric or t.d1Symmetric:
# if len(t.changingColors) == 1:
def symmetrize(matrix, axis, color=None, outColor=None, refColor=None):
    """
    Given a matrix and a color, this function tries to turn pixels of that
    given color into some other one in order to make the matrix symmetric.
    "axis" is a list or set specifying the symmetry axis (lr, ud, d1 or d2).
    """
    # Left-Right
    def LRSymmetrize(m):
        width = m.shape[1] - 1
        for i in range(m.shape[0]):
            for j in range(int(m.shape[1] / 2)):
                if m[i,j] != m[i,width-j]:
                    if color==None:
                        if m[i,j]==refColor and m[i,width-j]!=refColor:
                            m[i,width-j] = outColor
                        elif m[i,j]!=refColor and m[i,width-j]==refColor:
                            m[i,j] = outColor
                    else:
                        if m[i,j] == color:
                            m[i,j] = m[i,width-j]
                        elif m[i,width-j]==color:
                            m[i,width-j] = m[i,j]
        return m
    
    # Up-Down
    def UDSymmetrize(m):
        height = m.shape[0] - 1
        for i in range(int(m.shape[0] / 2)):
            for j in range(m.shape[1]):
                if m[i,j] != m[height-i,j]:
                    if color==None:
                        if m[i,j]==refColor and m[height-i,j]!=refColor:
                            m[height-i,j] = outColor
                        elif m[i,j]!=refColor and m[height-i,j]==refColor:
                            m[i,j] = outColor
                    else:
                        if m[i,j] == color:
                            m[i,j] = m[height-i,j]
                        elif m[height-i,j]==color:
                            m[height-i,j] = m[i,j]
        return m

    # Main diagonal
    def D1Symmetrize(m):
        for i,j in np.ndindex(m.shape):
            if m[i,j] != m[j,i]:
                if color==None:
                    if m[i,j]==refColor and m[j,i]!=refColor:
                        m[j,i] = outColor
                    elif m[i,j]!=refColor and m[j,i]==refColor:
                        m[i,j] = outColor
                else:
                    if m[i,j] == color:
                        m[i,j] = m[j,i]
                    elif m[j,i]==color:
                        m[j,i] = m[i,j]
        return m
    
    def D2Symmetrize(matrix):
        for i,j in np.ndindex(m.shape):
            if m[i,j] != m[m.shape[0]-j-1, m.shape[1]-i-1]:
                if color==None:
                    if m[i,j]==refColor and m[m.shape[0]-j-1, m.shape[1]-i-1]!=refColor:
                        m[m.shape[0]-j-1, m.shape[1]-i-1] = outColor
                    elif m[i,j]!=refColor and m[m.shape[0]-j-1, m.shape[1]-i-1]==refColor:
                        m[i,j] = outColor
                else:
                    if m[i,j] == color:
                        m[i,j] = m[m.shape[0]-j-1, m.shape[1]-i-1]
                    elif m[m.shape[0]-j-1, m.shape[1]-i-1]==color:
                        m[m.shape[0]-j-1, m.shape[1]-i-1] = m[i,j]
        return m
    
    m = matrix.m.copy()
    while True:
        prevMatrix = m.copy()
        if "lr" in axis:
            m = LRSymmetrize(m)
        if "ud" in axis:
            m = UDSymmetrize(m)
        if "d1" in axis:
            m = D1Symmetrize(m)
        if "d2" in axis:
            m = D2Symmetrize(m)
        if np.array_equal(prevMatrix, m):
            break
            
    return m

# %% Color symmetric pixels (task 653)

def colorSymmetricPixels(matrix, inColor, outColor, axis, includeAxis=False):
    """
    This function finds the pixels of color inColor that are symmetric
    according to the given axis in the input matrix, and colors them with the
    color given by outColor.
    Axis can be "lr", "ud", "d1" or "d2".
    """
    m = matrix.m.copy()
    if axis=="lr":
        for i,j in np.ndindex((m.shape[0], int(m.shape[1]/2))):
            if m[i,j]==inColor and m[i,m.shape[1]-1-j]==inColor:
                m[i,j] = outColor
                m[i,m.shape[1]-1-j] = outColor
        if includeAxis and ((m.shape[1]%2)==1):
            j = int(m.shape[1]/2)
            for i in range(m.shape[0]):
                if m[i,j]==inColor:
                    m[i,j] = outColor
    if axis=="ud":
        for i,j in np.ndindex((int(m.shape[0]/2), m.shape[1])):
            if m[i,j]==inColor and m[m.shape[0]-1-i,j]==inColor:
                m[i,j] = outColor
                m[m.shape[0]-1-i,j] = outColor
        if includeAxis and ((m.shape[0]%2)==1):
            i = int(m.shape[0]/2)
            for j in range(m.shape[1]):
                if m[i,j]==inColor:
                    m[i,j] = outColor
    if axis=="d1":
        for i in range(m.shape[0]):
            for j in range(i):
                if m[i,j]==inColor and m[j,i]==inColor:
                    m[i,j] = outColor
                    m[j,i] = outColor
        if includeAxis:
            for i in range(m.shape[0]):
                if m[i,i]==inColor:
                    m[i,i] = outColor
    if axis=="d2":
        for i in range(m.shape[0]):
            for j in range(m.shape[0]-i-1):
                if m[i,j]==inColor and m[m.shape[1]-j-1,m.shape[0]-i-1]==inColor:
                    m[i,j] = outColor
                    m[m.shape[1]-j-1,m.shape[0]-i-1] = outColor
        if includeAxis:
            for i in range(m.shape[0]):
                if m[i, m.shape[0]-i-1]==inColor:
                   m[i, m.shape[0]-i-1] = outColor
                
    return m

def getBestColorSymmetricPixels(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "colorSymmetricPixels" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    
    for cic in t.commonChangedInColors:
        for coc in t.commonChangedOutColors:
            f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                        axis="lr", includeAxis=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                        axis="lr", includeAxis=False)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                        axis="ud", includeAxis=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                        axis="ud", includeAxis=False)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            if all([s.inMatrix.shape[0]==s.inMatrix.shape[1] for s in t.trainSamples+t.testSamples]):
                f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                            axis="d1", includeAxis=True)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                            axis="d1", includeAxis=False)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                            axis="d2", includeAxis=True)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                            axis="d2", includeAxis=False)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                
    return bestFunction

# %% Train and predict models  

def trainCNNDummyColor(t, k, pad):
    """
    This function trains a CNN with only one convolution of filter k and with
    padding values equal to pad.
    The training samples will have two channels: the background color and any
    other color. The training loop loops through all the non-background colors
    of each sample, treating them independently.
    This is useful for tasks like number 3.
    """
    model = OneConvModel(2, k, pad)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    for e in range(50): # numEpochs            
        optimizer.zero_grad()
        loss = 0.0
        for s in t.trainSamples:
            for c in s.colors:
                if c != t.backgroundColor:
                    x = dummifyColor(s.inMatrix.m, c)
                    x = torch.tensor(x).unsqueeze(0).float()
                    y = deBackgroundizeMatrix(s.outMatrix.m, c)
                    y = torch.tensor(y).unsqueeze(0).long()
                    y_pred = model(x)
                    loss += criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    return model

@torch.no_grad()
def predictCNNDummyColor(matrix, model):
    """
    Predict function for a model trained using trainCNNDummyColor.
    """
    m = matrix.m.copy()
    pred = np.ones(m.shape, dtype=np.uint8) * matrix.backgroundColor
    for c in matrix.colors:
        if c != matrix.backgroundColor:
            x = dummifyColor(m, c)
            x = torch.tensor(x).unsqueeze(0).float()
            x = model(x).argmax(1).squeeze(0).numpy()
            for i,j in np.ndindex(m.shape):
                if x[i,j] != 0:
                    pred[i,j] = c
    return pred

def trainCNN(t, commonColors, nChannels, k=5, pad=0):
    """
    This function trains a CNN model with kernel k and padding value pad.
    It is required that all the training samples have the same number of colors
    (adding the colors in the input and in the output).
    It is also required that the output matrix has always the same shape as the
    input matrix.
    The colors are tried to be order in a specific way: first the colors that
    are common to every sample (commonColors), and then the others.
    """
    model = OneConvModel(nChannels, k, pad)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    #losses = np.zeros(100)
    for e in range(100):
        optimizer.zero_grad()
        loss = 0.0
        for s in t.trainSamples:
            sColors = commonColors.copy()
            for c in s.colors:
                if c not in sColors:
                    sColors.append(c)
            rel, invRel = relDicts(sColors)
            x = dummify(s.inMatrix.m, nChannels, rel)
            x = torch.tensor(x).unsqueeze(0).float()
            y = s.outMatrix.m.copy()
            for i,j in np.ndindex(y.shape):
                y[i,j] = invRel[y[i,j]]
            y = torch.tensor(y).unsqueeze(0).long()
            y_pred = model(x)
            loss += criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        #losses[e] = loss
    return model#, losses

@torch.no_grad()
def predictCNN(matrix, model, commonColors, nChannels):
    """
    Predict function for a model trained using trainCNN.
    """
    m = matrix.m.copy()
    pred = np.zeros(m.shape, dtype=np.uint8)
    sColors = commonColors.copy()
    for c in matrix.colors:
        if c not in sColors:
            sColors.append(c)
    rel, invRel = relDicts(sColors)
    if len(sColors) > nChannels:
        return m
    x = dummify(m, nChannels, rel)
    x = torch.tensor(x).unsqueeze(0).float()
    x = model(x).argmax(1).squeeze(0).numpy()
    for i,j in np.ndindex(m.shape):
        if x[i,j] not in rel.keys():
            pred[i,j] = x[i,j]
        else:
            pred[i,j] = rel[x[i,j]][0]
    return pred

def getBestCNN(t):
    """
    This function returns the best CNN with only one convolution, after trying
    different kernel sizes and padding values.
    There are as many channels as total colors or the minimum number of
    channels that is necessary.
    """
    kernel = [3,5,7]
    pad = [0,-1]    
    bestScore = 100000
    for k, p in product(kernel, pad):
        cc = list(range(10))
        model = trainCNN(t, commonColors=cc, nChannels=10, k=k, pad=p)
        score = sum([incorrectPixels(predictCNN(t.trainSamples[s].inMatrix, model, cc, 10), \
                                     t.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
        if score < bestScore:
            bestScore=score
            ret = partial(predictCNN, model=model, commonColors=cc, nChannels=10)
            if score==0:
                return ret
    return ret

def getBestSameNSampleColorsCNN(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "predictCNN" and works best for the training samples, after
    training a CNN using the training Samples for training.
    """
    kernel = [3,5,7]
    pad = [0,-1]    
    bestScore = 100000
    for k, p in product(kernel, pad):
        cc = list(t.commonSampleColors)
        nc = t.trainSamples[0].nColors
        model = trainCNN(t, commonColors=cc, nChannels=nc, k=k, pad=p)
        score = sum([incorrectPixels(predictCNN(t.trainSamples[s].inMatrix, model, cc, nc), \
                                     t.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
        if score < bestScore:
            bestScore=score
            ret = partial(predictCNN, model=model, commonColors=cc, nChannels=nc)
            if score==0:
                return ret
            
    return ret

# %% CNN learning the output
    
def getNeighbourColors(m, i, j, border=0):
    """
    Given a matrix m and a position i,j, this function returns a list of the
    values of the neighbours of (i,j).
    """
    x = []
    y = m[i-1,j] if i>0 else border
    x.append(y)
    y = m[i,j-1] if j>0 else border
    x.append(y)
    y = m[i+1,j] if i<m.shape[0]-1 else border
    x.append(y)
    y = m[i,j+1] if j<m.shape[1]-1 else border
    x.append(y)
    return x

def getDNeighbourColors(m, i, j, kernel=3, border=0):
    """
    Given a matrix m and a position i,j, this function returns a list of the
    values of the diagonal neighbours of (i,j).
    """
    x = []
    y = m[i-1,j-1] if (i>0 and j>0) else border
    x.append(y)
    y = m[i+1,j-1] if (i<m.shape[0]-1 and j>0) else border
    x.append(y)
    y = m[i-1,j+1] if (i>0 and j<m.shape[1]-1) else border
    x.append(y)
    y = m[i+1,j+1] if (i<m.shape[0]-1 and j<m.shape[1]-1) else border
    x.append(y)
    
    if kernel==5:
        y = m[i-2,j-2] if (i>1 and j>1) else border
        x.append(y)
        y = m[i-1,j-2] if (i>0 and j>1) else border
        x.append(y)
        y = m[i,j-2] if j>1 else border
        x.append(y)
        y = m[i+1,j-2] if (i<m.shape[0]-1 and j>1) else border
        x.append(y)
        y = m[i+2,j-2] if (i<m.shape[0]-2 and j>1) else border
        x.append(y)
        y = m[i+2,j-1] if (i<m.shape[0]-2 and j>0) else border
        x.append(y)
        y = m[i+2,j] if i<m.shape[0]-2 else border
        x.append(y)
        y = m[i+2,j+1] if (i<m.shape[0]-2 and j<m.shape[1]-1) else border
        x.append(y)
        y = m[i+2,j+2] if (i<m.shape[0]-2 and j<m.shape[1]-2) else border
        x.append(y)
        y = m[i+1,j+2] if (i<m.shape[0]-1 and j<m.shape[1]-2) else border
        x.append(y)
        y = m[i,j+2] if j<m.shape[1]-2 else border
        x.append(y)
        y = m[i-1,j+2] if (i>0 and j<m.shape[1]-2) else border
        x.append(y)
        y = m[i-2,j+2] if (i>1 and j<m.shape[1]-2) else border
        x.append(y)
        y = m[i-2,j+1] if (i>1 and j<m.shape[1]-1) else border
        x.append(y)
        y = m[i-2,j] if i>1 else border
        x.append(y)
        y = m[i-2,j-1] if (i>1 and j>0) else border
        x.append(y)
    return x

def getAllNeighbourColors(m, i, j, kernel=3, border=0):
    """
    This function returns a list the neighbour colors of the pixel i,j in the
    matrix m.
    """
    return getNeighbourColors(m,i,j,border) + getDNeighbourColors(m,i,j,kernel,border)

def colorNeighbours(mIn, mOut ,i, j):
    """
    Given matrices mIn and mOut, and coordinates i and j, this function colors
    the neighbours of i and j in mIn with the colors of the neighbours of i and
    j in mOut.
    """
    if i>0:
        mIn[i-1,j] = mOut[i-1,j]
    if j>0:
        mIn[i,j-1] = mOut[i,j-1]
    if i<mIn.shape[0]-1:
        mIn[i+1,j] = mOut[i+1,j]
    if j<mIn.shape[1]-1:
        mIn[i,j+1] = mOut[i,j+1]
        
def colorDNeighbours(mIn, mOut, i, j):
    """
    Given matrices mIn and mOut, and coordinates i and j, this function colors
    the neighbours of i and j in mIn with the colors of the neighbours of i and
    j in mOut. It includes diagonal neighbours.
    """
    colorNeighbours(mIn, mOut ,i, j)
    if i>0 and j>0:
        mIn[i-1,j-1] = mOut[i-1,j-1]
    if i<mIn.shape[0]-1 and j>0:
        mIn[i+1,j-1] = mOut[i+1,j-1]
    if i>0 and j<mIn.shape[1]-1:
        mIn[i-1,j+1] = mOut[i-1,j+1]
    if i<mIn.shape[0]-1 and j<mIn.shape[1]-1:
        mIn[i+1,j+1] = mOut[i+1,j+1]
     
def evolve(t, kernel=3, border=0, includeRotations=False):
    """
    Given a task t, this function returns a list of dictionaries. Each
    dictionary contains as keys lists of numbers representing colors of 
    neighbouring pixels. The value is the color of the pixel that is surrounded
    by these neighbours. A key only exist if the value can be unique.
    There are 4 dictionaries returned: For the first one, only the direct
    neighbours are considered. For the second one, also the diagonal neighbours
    are considered. For the third one, a kernel of 5 is considered (this is,
    24 neighbours), and for the fourth one is of kernel 3 and also considers
    the background color.
    """
    def evolveInputMatrices(mIn, mOut, changeCIC=False):
        reference = [m.copy() for m in mIn]
        for m in range(len(mIn)):
            if changeCIC:
                for i,j in np.ndindex(mIn[m].shape):
                    if mIn[m][i,j] not in set.union(fixedColors, changedOutColors): 
                        colorDNeighbours(mIn[m], mOut[m], i, j)
                        break
            else:
                for i,j in np.ndindex(mIn[m].shape):
                    if referenceIsFixed and reference[m][i,j] in fixedColors:
                        colorDNeighbours(mIn[m], mOut[m], i, j)
                    elif reference[m][i,j] in changedOutColors:
                        colorDNeighbours(mIn[m], mOut[m], i, j)
                    
    nColors = t.trainSamples[0].nColors
    
    if not t.allEqual(t.sampleColors):
        sampleRel = []
        sampleInvRel = []
        commonColors = t.orderedColors
        for s in t.trainSamples:
            colors = commonColors.copy()
            for i,j in np.ndindex(s.inMatrix.shape):
                if s.inMatrix.m[i,j] not in colors:
                    colors.append(s.inMatrix.m[i,j])
                    if len(colors) == nColors:
                        break
            if len(colors) != nColors:
                for i,j in np.ndindex(s.outMatrix.shape):
                    if s.outMatrix.m[i,j] not in colors:
                        colors.append(s.outMatrix.m[i,j])
                        if len(colors) == nColors:
                            break
            rel, invRel = relDicts(colors)
            sampleRel.append(rel)
            sampleInvRel.append(invRel)
            
        fixedColors = set()
        for c in t.fixedColors:
            fixedColors.add(sampleInvRel[0][c])
        changedOutColors = set()
        for c in t.commonChangedOutColors:
            changedOutColors.add(sampleInvRel[0][c])
        for c in range(len(commonColors), nColors):
            changedOutColors.add(c)
    else:
        fixedColors = t.fixedColors
        changedOutColors = t.commonChangedOutColors
        
    referenceIsFixed = t.trainSamples[0].inMatrix.nColors == len(fixedColors)+1
    
    outMatrices = [s.outMatrix.m.copy() for s in t.trainSamples]
    referenceOutput = [s.inMatrix.m.copy() for s in t.trainSamples]
    
    if includeRotations:
        for i in range(1,4):
            for m in range(t.nTrain):
                outMatrices.append(np.rot90(outMatrices[m].copy(), i))
                referenceOutput.append(np.rot90(referenceOutput[m].copy(), i))
                    
    
    if not t.allEqual(t.sampleColors):
        for m in range(len(outMatrices)):
            for i,j in np.ndindex(outMatrices[m].shape):
                outMatrices[m][i,j] = sampleInvRel[m%t.nTrain][outMatrices[m][i,j]]
                referenceOutput[m][i,j] = sampleInvRel[m%t.nTrain][referenceOutput[m][i,j]]
    
    colorFromNeighboursK2 = {}
    colorFromNeighboursK3 = {}
    colorFromNeighboursK5 = {}
    for i in range(10):
        referenceInput = [m.copy() for m in referenceOutput]
        evolveInputMatrices(referenceOutput, outMatrices)
        if np.all([np.array_equal(referenceInput[m], referenceOutput[m]) for m in range(len(referenceInput))]):
            evolveInputMatrices(referenceOutput, outMatrices, True)
        for m in range(len(outMatrices)):
            for i,j in np.ndindex(referenceInput[m].shape):
                if referenceInput[m][i,j] != referenceOutput[m][i,j]:
                    neighbourColors = tuple(getNeighbourColors(referenceInput[m],i,j,border))
                    colorFromNeighboursK2[neighbourColors] = referenceOutput[m][i,j]
                    neighbourColors = tuple(getAllNeighbourColors(referenceInput[m],i,j,3,border))
                    colorFromNeighboursK3[neighbourColors] = referenceOutput[m][i,j]
                    neighbourColors = tuple(getAllNeighbourColors(referenceInput[m],i,j,5,border))
                    colorFromNeighboursK5[neighbourColors] = referenceOutput[m][i,j]
       
    colorFromNeighboursK2 = {k:v for k,v in colorFromNeighboursK2.items() if \
                             not all([x not in set.union(changedOutColors, fixedColors) for x in k])}
    colorFromNeighboursK3 = {k:v for k,v in colorFromNeighboursK3.items() if \
                             not all([x not in set.union(changedOutColors, fixedColors) for x in k])}
    colorFromNeighboursK5 = {k:v for k,v in colorFromNeighboursK5.items() if \
                             not all([x not in set.union(changedOutColors, fixedColors) for x in k])}
                
    colorfromNeighboursK3Background = {}
    for m in outMatrices:
        for i,j in np.ndindex(m.shape):
            if m[i,j] in t.commonChangedInColors:
                neighbourColors = tuple(getAllNeighbourColors(m,i,j,3,border))
                colorfromNeighboursK3Background[neighbourColors] = m[i,j]
        
    
    return [colorFromNeighboursK2, colorFromNeighboursK3,\
            colorFromNeighboursK5, colorfromNeighboursK3Background]

def applyEvolve(matrix, cfn, nColors, changedOutColors=set(), fixedColors=set(),\
                changedInColors=set(), referenceIsFixed=False, commonColors=set(),\
                kernel=None, border=0, nIterations=1000):
    """
    Given a matrix and the "colors from neighbours" (cfn) list of dictionaries
    returned by the function evolve, this function colors the pixels of the
    matrix according to the rules given by cfn.
    """    
    
    def colorPixel(m,newM,i,j):
        if newM[i,j] not in cic and colorAroundCIC==False:
            return
        if kernel==None:
            tup3 = tuple(getAllNeighbourColors(m,i,j,3,border))
            tup5 = tuple(getAllNeighbourColors(m,i,j,5,border))
            #tup = getMostSimilarTuple(tup)
            if tup3 in cfn[1].keys():
                if tup3 in cfn[3].keys():
                    if tup5 in cfn[2].keys():
                        newM[i,j] = cfn[2][tup5]
                else:
                    newM[i,j] = cfn[1][tup3] 
            elif tup5 in cfn[2].keys():
                newM[i,j] = cfn[2][tup5]
        elif kernel==2:
            tup2 = tuple(getNeighbourColors(m,i,j,border))
            if tup2 in cfn[0].keys():
                newM[i,j] = cfn[0][tup2]
        elif kernel==3:
            tup3 = tuple(getAllNeighbourColors(m,i,j,3,border))
            if tup3 in cfn[1].keys():
                newM[i,j] = cfn[1][tup3]
        elif kernel==5:
            tup5 = tuple(getAllNeighbourColors(m,i,j,5,border))
            if tup5 in cfn[2].keys():
                newM[i,j] = cfn[2][tup5]
        
    def colorPixelsAround(m,newM,i,j):
        if i>0:
            colorPixel(m,newM,i-1,j)
        if j>0:
            colorPixel(m,newM,i,j-1)
        if i<m.shape[0]-1:
            colorPixel(m,newM,i+1,j)
        if j<m.shape[1]-1:
            colorPixel(m,newM,i,j+1)
        if i>0 and j>0:
            colorPixel(m,newM,i-1,j-1)
        if i<m.shape[0]-1 and j>0:
            colorPixel(m,newM,i+1,j-1)
        if i>0 and j<m.shape[1]-1:
            colorPixel(m,newM,i-1,j+1)
        if i<m.shape[0]-1 and j<m.shape[1]-1:
            colorPixel(m,newM,i+1,j+1)
    
    m = matrix.m.copy()
    
    if len(commonColors) > 0:
        colors = list(commonColors.copy())
        for i,j in np.ndindex(m.shape):
            if m[i,j] not in colors:
                colors.append(m[i,j])
        rel, invRel = relDicts(colors)
        
        for i,j in np.ndindex(m.shape):
            m[i,j] = invRel[m[i,j]]
            
        fc = set()
        for c in fixedColors:
            fc.add(invRel[c])        
        coc = set()
        for c in changedOutColors:
            coc.add(invRel[c])
        for c in range(len(commonColors), nColors):
            coc.add(c)
        cic = set()
        for c in changedInColors:
            cic.add(invRel[c])
    else:
        fc = fixedColors
        coc = changedOutColors
        cic = changedInColors
    
    it = 0
    colorAroundCIC=False
    while it<nIterations:
        it += 1
        newM = m.copy()
        #seen = np.zeros(m.shape, dtype=np.bool)
        if colorAroundCIC:
            for i,j in np.ndindex(m.shape):
                colorPixelsAround(m,newM,i,j)
            colorAroundCIC=False
            m=newM.copy()
            continue
        for i,j in np.ndindex(m.shape):
            if referenceIsFixed and m[i,j] in fixedColors:
                colorPixelsAround(m,newM,i,j)
            elif m[i,j] in coc:
                colorPixelsAround(m,newM,i,j)
        if np.array_equal(newM,m):
            if it==1:
                colorAroundCIC=True
            else:
                break
        m = newM.copy()
        
    if len(commonColors) > 0:
        for i,j in np.ndindex(m.shape):
            if m[i,j] in rel.keys():
                m[i,j] = rel[m[i,j]][0]
            else:
                m[i,j] = rel[0][0] # Patch for bug in task 22
        
    return m
    
def getBestEvolve(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "evolve" and works best for the training samples.
    """
    nColors = t.trainSamples[0].nColors
    fc = t.fixedColors
    cic = t.commonChangedInColors
    coc = t.commonChangedOutColors
    refIsFixed = t.trainSamples[0].inMatrix.nColors == len(fc)+1
    
    bestScore = 1000
    bestFunction = None
    
    cfn = evolve(t)
    if t.allEqual(t.sampleColors):
        f = partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                    fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                    kernel=None, border=0)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
        f =  partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                     fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                     kernel=5, border=0)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction

    else:
        f = partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                    fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                    kernel=None, border=0, commonColors=t.orderedColors)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction    
        
        f =  partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                     fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                     kernel=5, border=0, commonColors=t.orderedColors)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    cfn = evolve(t, includeRotations=True)
    if t.allEqual(t.sampleColors):
        f = partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                    fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                    kernel=None, border=0)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction    
        
        f =  partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                     fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                     kernel=5, border=0)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction

    else:
        f = partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                    fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                    kernel=None, border=0, commonColors=t.orderedColors)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction    
        
        f =  partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                     fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                     kernel=5, border=0, commonColors=t.orderedColors)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    return bestFunction


# Good examples: 790,749,748,703,679,629,605,585,575,573,457,344,322,
#                283,236,231,201,198,59,23

class EvolvingLine():
    """
    The object of this class include information on how to draw an evolving 
    line.
    ...
    Attributes
    ----------
    color: int
        The color of the line
    direction: str
        The direction of the line. It can be one of 4: 'l', 'r', 'u', 'd'.
    position: tupe (int, int)
        The current position of the line. It marks where to draw the next
        pixel.
    cic: set
        Short for "changedInColors". Colors of the input matrix that can be
        modified.
    colorRules: dict
        Keys are colors. Values are strings encoding what the line should do
        when it encounters this color. The values can be 'r' (turn right),
        'l' (turn left), 'stop', 'skip' (skip over the pixel and continue), 
        'split' (generate two evolvingLines, one turning right and the other
        one left) or 'convert' (change the color of the line)
    dealWith: dict
        Same as colorRules, but including borders. Borders are considered as
        colors number 10 (left), 11 (right), 12 (top) and 13 (bottom).
    fixedDirection: bool
        True if the direction cannot be modified. False if it is modified when
        turning.
    stepSize: int
        Number of pixels to color at every step. If not specified, it is 1.
    turning: bool
        True if the line is currently turning. False otherwise.
    
    Methods
    -------
    draw(self, m, direction=None):
        Draw the evolving line in the matrix m.
    """
    def __init__(self, color, direction, position, cic, source=None, \
                 colorRules=None, stepSize=None, fixedDirection=True, turning=False,\
                 turnAfterNSteps=[None, None], stepIncrease=None, \
                 alternateStepIncrease=False):
        """
        cic = changedInColors
        """
        self.source = source # Shape
        self.color = color
        self.direction = direction
        self.position = position
        self.cic = cic
        self.colorRules = colorRules
        self.fixedDirection = fixedDirection
        self.dealWith = {}
        self.stepSize = stepSize
        self.turning = turning
        # left=10, right=11, top=12, bot=13
        for color in range(14): # 10 colors + 4 borders
            self.dealWith[color] = 'stop'
        if type(colorRules)==str:
            for i in range(10):
                if i not in cic:
                    self.dealWith[i] = colorRules
        elif colorRules!=None:
            for cr in colorRules:
                self.dealWith[cr[0]] = cr[1]    
        self.dealWith[self.color] = "skip"
        self.maxSteps = turnAfterNSteps.copy() # [number of maximum steps, direction]
        self.stepIncrease = stepIncrease
        self.step = -1
        self.alternateStepIncrease = alternateStepIncrease
        self.increaseStep = False
        
    def draw(self, m, direction=None):
        # If we reached the maximum number of steps, turn
        if self.maxSteps[0]!=None:
            if self.maxSteps[0]==self.step:
                if self.maxSteps[1]=="stop":
                    return
                
                #self.turning=True
                
                if self.direction=='u' and self.maxSteps[1]=='r' or\
                self.direction=='d' and self.maxSteps[1]=='l':
                    direction = 'r'
                    if not self.fixedDirection:
                        self.direction = 'r'
                        
                elif self.direction=='u' and self.maxSteps[1]=='l' or\
                self.direction=='d' and self.maxSteps[1]=='r':
                    direction = 'l'
                    if not self.fixedDirection:
                        self.direction = 'l'   
                        
                elif self.direction=='r' and self.maxSteps[1]=='l' or\
                self.direction=='l' and self.maxSteps[1]=='r':
                    direction = 'u'
                    if not self.fixedDirection:
                        self.direction = 'u'
                        
                elif self.direction=='l' and self.maxSteps[1]=='l' or\
                self.direction=='r' and self.maxSteps[1]=='r':
                    direction = 'd'
                    if not self.fixedDirection:
                        self.direction = 'd' 
                        
                if self.stepIncrease!=None:
                    if self.increaseStep:
                        self.maxSteps[0]+=self.stepIncrease
                        if self.alternateStepIncrease:
                            self.increaseStep=False
                    else:
                        self.increaseStep=True
                        
                self.step = -1
                    
        self.step += 1
        
        if direction==None:
            direction=self.direction
                    
        # Left
        if direction=='l':
            if self.position[1]==0:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(10, m)
                return
            newColor = m[self.position[0], self.position[1]-1]
            if newColor in self.cic:
                if self.turning:
                    self.turning=False
                m[self.position[0], self.position[1]-1] = self.color
                self.position[1] -= 1
                self.draw(m)
            else:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(newColor, m)
    
        # Right
        if direction=='r':
            if self.position[1]==m.shape[1]-1:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(11, m)
                return
            newColor = m[self.position[0], self.position[1]+1]
            if newColor in self.cic:
                if self.turning:
                    self.turning=False
                m[self.position[0], self.position[1]+1] = self.color
                self.position[1] += 1
                self.draw(m)
            else:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(newColor, m)
                
        # Up
        if direction=='u':
            if self.position[0]==0:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(12, m)
                return
            newColor = m[self.position[0]-1, self.position[1]]
            if newColor in self.cic:
                if self.turning:
                    self.turning=False
                m[self.position[0]-1, self.position[1]] = self.color
                self.position[0] -= 1
                self.draw(m)
            else:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(newColor, m)
        
        # Down
        if direction=='d':
            if self.position[0]==m.shape[0]-1:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(13, m)
                return
            newColor = m[self.position[0]+1, self.position[1]]
            if newColor in self.cic:
                if self.turning:
                    self.turning=False
                m[self.position[0]+1, self.position[1]] = self.color
                self.position[0] += 1
                self.draw(m)
            else:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(newColor, m)
        
    def dealWithColor(self, color, m):
        if self.dealWith[color] == "stop":
            return
        
        if self.dealWith[color] == "convert":
            if self.direction=='l':
                if self.position[1]!=0:
                    self.color = color
                    self.position[1]-=1
                    self.draw(m)
                else:
                    return
            if self.direction=='r':
                if self.position[1]!=m.shape[1]-1:
                    self.color = color
                    self.position[1]+=1
                    self.draw(m)
                else:
                    return
            if self.direction=='u':
                if self.position[0]!=0:
                    self.color = color
                    self.position[0]-=1
                    self.draw(m)
                else:
                    return
            if self.direction=='d':
                if self.position[0]!=m.shape[0]-1:
                    self.color = color
                    self.position[0]+=1
                    self.draw(m)
                else:
                    return
            
        if self.dealWith[color] == "split":
            if self.direction=='l' or self.direction=='r':
                if self.position[0]!=0:
                    l1 = EvolvingLine(self.color, self.direction, self.position.copy(), self.cic,\
                                      colorRules=self.colorRules, fixedDirection=self.fixedDirection, \
                                      turning=True)
                    if self.fixedDirection==False:
                        l1.direction='u'
                    l1.draw(m, direction='u')
                if self.position[0]!=m.shape[0]-1:
                    l2 = EvolvingLine(self.color, self.direction, self.position.copy(), self.cic,\
                                      colorRules=self.colorRules, fixedDirection=self.fixedDirection, \
                                      turning=True)
                    if self.fixedDirection==False:
                        l2.direction='d'
                    l2.draw(m, direction='d')
            if self.direction=='u' or self.direction=='d':
                if self.position[1]!=0:
                    l1 = EvolvingLine(self.color, self.direction, self.position.copy(), self.cic,\
                                      colorRules=self.colorRules, fixedDirection=self.fixedDirection, \
                                      turning=True)
                    if self.fixedDirection==False:
                        l1.direction='l'
                    l1.draw(m, direction='l')
                if self.position[1]!=m.shape[1]-1:
                    l2 = EvolvingLine(self.color, self.direction, self.position.copy(), self.cic,\
                                      colorRules=self.colorRules, fixedDirection=self.fixedDirection, \
                                      turning=True)
                    if self.fixedDirection==False:
                        l2.direction='r'
                    l2.draw(m, direction='r')
                    
        if self.dealWith[color] == "skip":
            if self.direction=='l':
                if self.position[1]!=0:
                    self.position[1]-=1
                    self.draw(m)
                else:
                    return
            if self.direction=='r':
                if self.position[1]!=m.shape[1]-1:
                    self.position[1]+=1
                    self.draw(m)
                else:
                    return
            if self.direction=='u':
                if self.position[0]!=0:
                    self.position[0]-=1
                    self.draw(m)
                else:
                    return
            if self.direction=='d':
                if self.position[0]!=m.shape[0]-1:
                    self.position[0]+=1
                    self.draw(m)
                else:
                    return
                    
        # Left
        if self.dealWith[color] == 'l':
            if self.direction=='u':
                if self.position[1]!=0:
                    if not self.fixedDirection:
                        self.direction = 'l'
                    self.draw(m, direction='l')
                return
            if self.direction=='d':
                if self.position[1]!=m.shape[1]-1:
                    if not self.fixedDirection:
                        self.direction = 'r'
                    self.draw(m, direction='r')
                return
            if self.direction=='l':
                if self.position[0]!=m.shape[0]-1:
                    if not self.fixedDirection:
                        self.direction = 'd'
                    self.draw(m, direction='d')
                return
            if self.direction=='r':
                if self.position[0]!=0:
                    if not self.fixedDirection:
                        self.direction = 'u'
                    self.draw(m, direction='u')
                return
            
        # Right
        if self.dealWith[color] == 'r':
            if self.direction=='u':
                if self.position[1]!=m.shape[1]-1:
                    if not self.fixedDirection:
                        self.direction = 'r'
                    self.draw(m, direction='r')
                return
            if self.direction=='d':
                if self.position[1]!=0:
                    if not self.fixedDirection:
                        self.direction = 'l'
                    self.draw(m, direction='l')
                return
            if self.direction=='l':
                if self.position[0]!=0:
                    if not self.fixedDirection:
                        self.direction = 'u'
                    self.draw(m, direction='u')
                return
            if self.direction=='r':
                if self.position[0]!=m.shape[0]-1:
                    if not self.fixedDirection:
                        self.direction = 'd'
                    self.draw(m, direction='d')
                return            
        
def detectEvolvingLineSources(t):
    """
    Given a Task t, this function detects the sources in which EvolvingLines
    start. Currently it only supports pixels. It returns a set of pairs 
    (color (int), direction (str)).
    """
    sources = set()
    if len(t.commonChangedOutColors)==1:
        coc = next(iter(t.commonChangedOutColors))
    else:
        coc = None
    possibleSourceColors = set.intersection(t.commonChangedOutColors, t.commonInColors)
    if len(possibleSourceColors) == 0:
        possibleSourceColors = set.intersection(t.almostCommonColors, t.unchangedColors)
    if len(possibleSourceColors) != 0:
        firstIt = True
        for sample in t.trainSamples:
            sampleSources = set()
            for color in possibleSourceColors:
                if coc==None:
                    targetColor=color
                else:
                    targetColor=coc
                for shape in sample.inMatrix.shapes:
                    if shape.color==color and shape.nPixels==1:                        
                        # First special case: Corners
                        if shape.position==(0,0):
                            if sample.outMatrix.m[1][0]==targetColor and sample.outMatrix.m[0][1]==targetColor:
                                sampleSources.add((color, "away"))
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                            elif sample.outMatrix.m[1][0]==targetColor:
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                            elif sample.outMatrix.m[0][1]==targetColor:
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                        elif shape.position==(0,sample.inMatrix.shape[1]-1):
                            if sample.outMatrix.m[1][sample.outMatrix.shape[1]-1]==targetColor and sample.outMatrix.m[0][sample.outMatrix.shape[1]-2]==targetColor:
                                sampleSources.add((color, "away"))
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                            elif sample.outMatrix.m[1][sample.outMatrix.shape[1]-1]==targetColor:
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                            elif sample.outMatrix.m[0][sample.outMatrix.shape[1]-2]==targetColor:
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                        elif shape.position==(sample.inMatrix.shape[0]-1,0):
                            if sample.outMatrix.m[sample.outMatrix.shape[0]-2][0]==targetColor and sample.outMatrix.m[sample.outMatrix.shape[0]-1][1]==targetColor:
                                sampleSources.add((color, "away"))
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                            elif sample.outMatrix.m[sample.outMatrix.shape[0]-2][0]==targetColor:
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                            elif sample.outMatrix.m[sample.outMatrix.shape[0]-1][1]==targetColor:
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                        elif shape.position==(sample.inMatrix.shape[0]-1,sample.inMatrix.shape[1]-1):
                            if sample.outMatrix.m[sample.outMatrix.shape[0]-2][sample.outMatrix.shape[1]-1]==targetColor and sample.outMatrix.m[sample.outMatrix.shape[0]-1][sample.outMatrix.shape[1]-2]==targetColor:
                                sampleSources.add((color, "away"))
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                            elif sample.outMatrix.m[sample.outMatrix.shape[0]-2][sample.outMatrix.shape[1]-1]==targetColor:
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                            elif sample.outMatrix.m[sample.outMatrix.shape[0]-1][sample.outMatrix.shape[1]-2]==targetColor:
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                        
                        # Second special case: Border but not corner
                        elif shape.position[0]== 0:
                            if sample.outMatrix.m[1,shape.position[1]]==targetColor:
                                sampleSources.add((color,"away"))
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                            if sample.outMatrix.m[0,shape.position[1]-1]==targetColor:
                                sampleSources.add((color, 'l'))
                            if sample.outMatrix.m[0,shape.position[1]+1]==targetColor:
                                sampleSources.add((color, 'r'))
                        elif shape.position[0]== sample.inMatrix.shape[0]-1:
                            if sample.outMatrix.m[sample.inMatrix.shape[0]-2,shape.position[1]]==targetColor:
                                sampleSources.add((color,"away"))
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                            if sample.outMatrix.m[sample.inMatrix.shape[0]-1,shape.position[1]-1]==targetColor:
                                sampleSources.add((color, 'l'))
                            if sample.outMatrix.m[sample.inMatrix.shape[0]-1,shape.position[1]+1]==targetColor:
                                sampleSources.add((color, 'r'))
                        elif shape.position[1]== 0:
                            if sample.outMatrix.m[shape.position[0],1]==targetColor:
                                sampleSources.add((color,"away"))
                                sampleSources.add((color,'r'))
                                sampleSources.add((color, 'l'))
                            if sample.outMatrix.m[shape.position[0]-1,0]==targetColor:
                                sampleSources.add((color, 'u'))
                            if sample.outMatrix.m[shape.position[0]+1,0]==targetColor:
                                sampleSources.add((color, 'd'))
                        elif shape.position[1]== sample.inMatrix.shape[1]-1:
                            if sample.outMatrix.m[shape.position[0],sample.inMatrix.shape[1]-2]==targetColor:
                                sampleSources.add((color,"away"))
                                sampleSources.add((color,'r'))
                                sampleSources.add((color, 'l'))
                            if sample.outMatrix.m[shape.position[0]-1,sample.inMatrix.shape[1]-1]==targetColor:
                                sampleSources.add((color, 'u'))
                            if sample.outMatrix.m[shape.position[0]+1,sample.inMatrix.shape[1]-1]==targetColor:
                                sampleSources.add((color, 'd'))
                                
                        # Third case: Not border
                        else:
                            if sample.outMatrix.m[shape.position[0]+1, shape.position[1]]==targetColor:
                                sampleSources.add((color, 'd'))
                            if sample.outMatrix.m[shape.position[0]-1, shape.position[1]]==targetColor:
                                sampleSources.add((color, 'u'))
                            if sample.outMatrix.m[shape.position[0], shape.position[1]+1]==targetColor:
                                sampleSources.add((color, 'r'))
                            if sample.outMatrix.m[shape.position[0], shape.position[1]-1]==targetColor:
                                sampleSources.add((color, 'l'))
            if firstIt:
                sources = sampleSources
                firstIt = False
            else:
                sources = set.intersection(sources, sampleSources) 
                
    return sources

def getBestEvolvingLines(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "drawEvolvingLines" and works best for the training samples.
    """
    if any([s.inMatrix.shape[0]==1 or s.inMatrix.shape[1]==1 for s in t.trainSamples]):
        return partial(identityM)
    
    sources = detectEvolvingLineSources(t)
    
    fixedColorsList = list(t.fixedColors2)
    cic=t.commonChangedInColors
    #cic = [color for color in list(range(10)) if color not in fixedColorsList]
    if len(t.commonChangedOutColors)==1:
        coc = next(iter(t.commonChangedOutColors))
    else:
        coc = None
    
    bestScore = 1000
    bestFunction = partial(identityM)
    
    mergeColors = t.commonOutColors-t.totalInColors
    if len(mergeColors) == 1:
        mergeColor = next(iter(mergeColors))
    else:
        mergeColor = None
    
    for actions in combinations_with_replacement(["stop", 'l', 'r', "split", "skip"],\
                                                 len(t.fixedColors2)):
        rules = []
        for c in range(len(fixedColorsList)):
            rules.append([fixedColorsList[c], actions[c]])
            
        f = partial(drawEvolvingLines, sources=sources, rules=rules, cic=cic, \
                    fixedDirection=True, coc=coc, mergeColor=mergeColor)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        f = partial(drawEvolvingLines, sources=sources, rules=rules, cic=cic, \
                    fixedDirection=False, coc=coc, mergeColor=mergeColor)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if coc!=None:
            f = partial(drawEvolvingLines, sources=sources, rules=rules, cic=cic, \
                        fixedDirection=True, coc=coc, mergeColor=mergeColor)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            f = partial(drawEvolvingLines, sources=sources, rules=rules, cic=cic, \
                        fixedDirection=False, coc=coc, mergeColor=mergeColor)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    f = partial(drawEvolvingLines, sources=sources, rules="convert", cic=cic, \
                    fixedDirection=False, coc=coc, mergeColor=mergeColor)  
    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        
    for asi in [True, False]:
        for stepIncrease in [None, 1, 2]:
            for steps in [1, 2, 3, 4]:
                for direction in ['r', 'l']:
                    f = partial(drawEvolvingLines, sources=sources, rules="stop", cic=cic, \
                    fixedDirection=False, alternateStepIncrease=asi, \
                    stepIncrease=stepIncrease, turnAfterNSteps=[steps, direction], mergeColor=mergeColor) 
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            
    return bestFunction

def mergeMatrices(matrices, backgroundColor, mergeColor=None):
    """
    All matrices are required to have the same shape.
    """
    result = np.zeros(matrices[0].shape, dtype=np.uint8)
    if mergeColor==None:
        for i,j in np.ndindex(matrices[0].shape):
            done=False
            for m in matrices:
                if m[i,j]!=backgroundColor:
                    result[i,j] = m[i,j]
                    done=True
                    break
            if not done:
                result[i,j] = backgroundColor
    else:
        for i,j in np.ndindex(matrices[0].shape):
            colors = set()
            for m in matrices:
                if m[i,j]!=backgroundColor:
                    colors.add(m[i,j])
            if len(colors)==0:
                result[i,j] = backgroundColor
            elif len(colors)==1:
                result[i,j] = next(iter(colors))
            else:
                result[i,j] = mergeColor
    return result
        
def drawEvolvingLines(matrix, sources, rules, cic, fixedDirection, coc=None, \
                      stepIncrease=None, alternateStepIncrease=False, \
                      turnAfterNSteps=[None, None], mergeColor=None):
    """
    Given a set of sources, this function draws the evolving lines starting
    at these sources following the given rules in the given matrix.
    """
    if len(sources)==0:
        return matrix.m.copy()
    fd = fixedDirection
    matrices = []
    for source in sources:
        newM = matrix.m.copy()
        for i,j in np.ndindex(matrix.shape):
            if matrix.m[i,j]==source[0]:
                if source[1]=="away":
                    if i==0:
                        if coc==None:
                            line = EvolvingLine(source[0], 'd', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                        else:
                            line = EvolvingLine(coc, 'd', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                    elif i==matrix.m.shape[0]-1:
                        if coc==None:
                            line = EvolvingLine(source[0], 'u', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                        else:
                            line = EvolvingLine(coc, 'u', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                    elif j==0:
                        if coc==None:
                            line = EvolvingLine(source[0], 'r', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                        else:
                            line = EvolvingLine(coc, 'r', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                    elif j==matrix.m.shape[1]-1:
                        if coc==None:
                            line = EvolvingLine(source[0], 'l', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                        else:
                            line = EvolvingLine(coc, 'l', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                    else:
                        return matrix.m.copy()
                else:
                    if coc==None:
                        line = EvolvingLine(source[0], source[1], [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                            stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                            turnAfterNSteps=turnAfterNSteps)
                    else:
                        line = EvolvingLine(coc, source[1], [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                            stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                            turnAfterNSteps=turnAfterNSteps)
                line.draw(newM)
        matrices.append(newM)
    m = mergeMatrices(matrices, next(iter(cic)), mergeColor)
    return m

# %% Crossed coordinates
    
def paintCrossedCoordinates(matrix, refColor, outColor, fixedColors=set()):
    """
    Given a Matrix, this function returns a matrix (numpy.ndarray) by coloring
    the crossed coordinates that have a given refColor in the input matrix
    with the color outColor.
    """
    m = matrix.m.copy()
    xCoord = set()
    yCoord = set()
    
    for i,j in np.ndindex(matrix.shape):
        if m[i,j]==refColor:
            xCoord.add(i)
            yCoord.add(j)
    
    for i in xCoord:
        for j in yCoord:
            if matrix.m[i,j] not in fixedColors:
                m[i,j] = outColor
    
    return m

# %% Linear Models

# If input always has the same shape and output always has the same shape
# And there is always the same number of colors in each sample    
def trainLinearModel(t, commonColors, nChannels):
    """
    This function trains a linear model.
    It is required that all the training samples have the same number of colors
    (adding the colors in the input and in the output).
    It is also required that all the input matrices have the same shape, and
    all the output matrices have the same shape.
    The colors are tried to be order in a specific way: first the colors that
    are common to every sample (commonColors), and then the others.
    """
    model = LinearModel(t.inShape, t.outShape, nChannels)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for e in range(100):
        optimizer.zero_grad()
        loss = 0.0
        for s in t.trainSamples:
            sColors = commonColors.copy()
            for c in s.colors:
                if c not in sColors:
                    sColors.append(c)
            rel, invRel = relDicts(sColors)
            x = dummify(s.inMatrix.m, nChannels, rel)
            x = torch.tensor(x).unsqueeze(0).float()
            y = s.outMatrix.m.copy()
            for i,j in np.ndindex(y.shape):
                y[i,j] = invRel[y[i,j]]
            y = torch.tensor(y).unsqueeze(0).view(1,-1).long()
            y_pred = model(x)
            loss += criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    return model

@torch.no_grad()
def predictLinearModel(matrix, model, commonColors, nChannels, outShape):
    """
    Predict function for a model trained using trainLinearModel.
    """
    m = matrix.m.copy()
    pred = np.zeros(outShape, dtype=np.uint8)
    sColors = commonColors.copy()
    for c in matrix.colors:
        if c not in sColors:
            sColors.append(c)
    rel, invRel = relDicts(sColors)
    if len(sColors) > nChannels:
        return
    x = dummify(m, nChannels, rel)
    x = torch.tensor(x).unsqueeze(0).float()
    x = model(x).argmax(1).squeeze(0).view(outShape).numpy()
    for i,j in np.ndindex(outShape):
        if x[i,j] not in rel.keys():
            pred[i,j] = x[i,j]
        else:
            pred[i,j] = rel[x[i,j]][0]
    return pred

def trainLinearDummyModel(t):
    """
    This function trains a linear model.
    The training samples will have two channels: the background color and any
    other color. The training loop loops through all the non-background colors
    of each sample, treating them independently.
    """
    model = LinearModelDummy(t.inShape, t.outShape)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for e in range(100):
        optimizer.zero_grad()
        loss = 0.0
        for s in t.trainSamples:
            for c in s.colors:
                if c != t.backgroundColor:
                    x = dummifyColor(s.inMatrix.m, c)
                    x = torch.tensor(x).unsqueeze(0).float()
                    y = deBackgroundizeMatrix(s.outMatrix.m, c)
                    y = torch.tensor(y).unsqueeze(0).long()
                    y = y.view(1, -1)
                    y_pred = model(x)
                    loss += criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    return model
    
@torch.no_grad()
def predictLinearDummyModel(matrix, model, outShape, backgroundColor):
    """
    Predict function for a model trained using trainLinearDummyModel.
    """
    m = matrix.m.copy()
    pred = np.zeros(outShape, dtype=np.uint8)
    for c in matrix.colors:
        if c != backgroundColor:
            x = dummifyColor(m, c)
            x = torch.tensor(x).unsqueeze(0).float()
            x = model(x).argmax(1).squeeze().view(outShape).numpy()
            for i,j in np.ndindex(outShape):
                if x[i,j] != 0:
                    pred[i,j] = c
    return pred

def trainLinearModelShapeColor(t):
    """
    For trainLinearModelShapeColor we need to have the same shapes in the input
    and in the output, and in the exact same positions. The training loop loops
    through all the shapes of the task, and its aim is to predict the final
    color of each shape.
    The features of the linear model are:
        - One feature per color in the task. Value of 1 if the shape has that
        color, 0 otherwise.
        - Several features representing the number of pixels of the shape.
        Only one of these features can be equal to 1, the rest will be equal
        to 0.
        - 5 features to encode the number of holes of the shape (0,1,2,3 or 4)
        - Feature encoding whether the shape is a square or not.
        - Feature encoding whether the shape is a rectangle or not.
        - Feature encoding whether the shape touches the border or not.
    """
    inColors = set.union(*t.changedInColors+t.changedOutColors) - t.unchangedColors
    colors = list(inColors) + list(set.union(*t.changedInColors+t.changedOutColors) - inColors)
    rel, invRel = relDicts(list(colors))
    shapePixelNumbers = t.shapePixelNumbers
    _,nPixelsRel = relDicts(shapePixelNumbers)
    # inFeatures: [colors that change], [number of pixels]+1, [number of holes] (0-4),
    # isSquare, isRectangle, isBorder
    nInFeatures = len(inColors) + len(shapePixelNumbers) + 1 + 5 + 3
    model = SimpleLinearModel(nInFeatures, len(colors))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    num_epochs = 80
    trainShapes = []
    for s in t.trainSamples:
        for shapeI in range(s.inMatrix.nShapes):
            trainShapes.append((s.inMatrix.shapes[shapeI],\
                                s.outMatrix.shapes[shapeI].color))
        for e in range(num_epochs):
            optimizer.zero_grad()
            loss = 0.0
            for s,label in trainShapes:
                inFeatures = torch.zeros(nInFeatures)
                if s.color in inColors:
                    inFeatures[invRel[s.color]] = 1
                    inFeatures[len(inColors)+nPixelsRel[s.nPixels]] = 1
                    inFeatures[len(inColors)+len(shapePixelNumbers)+1+min(s.nHoles, 4)] = 1
                    inFeatures[nInFeatures-1] = int(s.isSquare)
                    inFeatures[nInFeatures-2] = int(s.isRectangle)
                    inFeatures[nInFeatures-1] = s.isBorder
                    #inFeatures[nInFeatures-4] = s.nHoles
                    #inFeatures[t.nColors+5] = s.position[0].item()
                    #inFeatures[t.nColors+6] = s.position[1].item()
                    y = torch.tensor(invRel[label]).unsqueeze(0).long()
                    x = inFeatures.unsqueeze(0).float()
                    y_pred = model(x)
                    loss += criterion(y_pred, y)
            if loss == 0:
                continue
            loss.backward()
            optimizer.step()
            for p in model.parameters():
                p.data.clamp_(min=0.05, max=1)
    return model

@torch.no_grad()
def predictLinearModelShapeColor(matrix, model, colors, unchangedColors, shapePixelNumbers):
    """
    Predict function for a model trained using trainLinearModelShapeColor.
    """
    inColors = colors - unchangedColors
    colors = list(inColors) + list(colors - inColors)
    rel, invRel = relDicts(list(colors))
    _,nPixelsRel = relDicts(shapePixelNumbers)
    nInFeatures = len(inColors) + len(shapePixelNumbers) + 1 + 5 + 3
    pred = matrix.m.copy()
    for shape in matrix.shapes:
        if shape.color in inColors:
            inFeatures = torch.zeros(nInFeatures)
            inFeatures[invRel[shape.color]] = 1
            if shape.nPixels not in nPixelsRel.keys():
                inFeatures[len(inColors)+len(shapePixelNumbers)] = 1
            else:
                inFeatures[len(inColors)+nPixelsRel[shape.nPixels]] = 1
            inFeatures[len(inColors)+len(shapePixelNumbers)+1+min(shape.nHoles, 4)] = 1
            inFeatures[nInFeatures-1] = int(shape.isSquare)
            inFeatures[nInFeatures-2] = int(shape.isRectangle)
            inFeatures[nInFeatures-3] = shape.isBorder
            #inFeatures[nInFeatures-4] = shape.nHoles
            #inFeatures[nColors+5] = shape.position[0].item()
            #inFeatures[nColors+6] = shape.position[1].item()
            x = inFeatures.unsqueeze(0).float()
            y = model(x).squeeze().argmax().item()
            pred = changeColorShapes(pred, [shape], rel[y][0])
    return pred

# %% LSTM
def prepare_sequence(seq, to_ix):
    """
    Utility function for LSTM.
    """
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def trainLSTM(t, inColors, colors, inRel, outRel, reverse, order):
    """
    This function tries to train a model that colors shapes according to a
    sequence.
    """
    EMBEDDING_DIM = 10
    HIDDEN_DIM = 10
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(inColors), len(colors))
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 150
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = 0.0
        for s in t.trainSamples:
            inShapes = [shape for shape in s.inMatrix.shapes if shape.color in inColors]
            inSeq = sorted(inShapes, key=lambda x: (x.position[order[0]], x.position[order[1]]), reverse=reverse)
            inSeq = [shape.color for shape in inSeq]
            outShapes = [shape for shape in s.outMatrix.shapes if shape.color in colors]
            targetSeq = sorted(outShapes, key=lambda x: (x.position[order[0]], x.position[order[1]]), reverse=reverse)
            targetSeq = [shape.color for shape in targetSeq]
            inSeq = prepare_sequence(inSeq, inRel)
            targetSeq = prepare_sequence(targetSeq, outRel)
            tag_scores = model(inSeq)
            loss += loss_function(tag_scores, targetSeq)
        loss.backward()
        optimizer.step()
    return model
    
@torch.no_grad()
def predictLSTM(matrix, model, inColors, colors, inRel, rel, reverse, order):
    """
    Predict function for a model trained using trainLSTM.
    """
    m = matrix.m.copy()
    inShapes = [shape for shape in matrix.shapes if shape.color in inColors]
    if len(inShapes)==0:
        return m
    sortedShapes = sorted(inShapes, key=lambda x: (x.position[order[0]], x.position[order[1]]), reverse=reverse)
    inSeq = [shape.color for shape in sortedShapes]
    inSeq = prepare_sequence(inSeq, inRel)
    pred = model(inSeq).argmax(1).numpy()
    for shapeI in range(len(sortedShapes)):
        m = changeColorShapes(m, [sortedShapes[shapeI]], rel[pred[shapeI]][0])
    return m
             
def getBestLSTM(t):
    """
    This function tries to find out which one is the best-fitting LSTM model
    for the task t. The LSTM models try to change the color of shapes to fit
    sequences. Examples are tasks 175, 331, 459 or 594.
    4 LSTM models are trained, considering models that order shapes by X
    coordinage, models that order them by Y coordinate, and considering both
    directions of the sequence (normal and reverse).
    """
    colors = set.union(*t.changedInColors+t.changedOutColors)
    inColors = colors - t.unchangedColors
    if len(inColors) == 0:
        return partial(identityM)
    _,inRel = relDicts(list(inColors))
    colors = list(inColors) + list(colors - inColors)
    rel, outRel = relDicts(colors)
    
    for s in t.trainSamples:
        inShapes = [shape for shape in s.inMatrix.shapes if shape.color in inColors]
        outShapes = [shape for shape in s.outMatrix.shapes if shape.color in colors]
        if len(inShapes) != len(outShapes) or len(inShapes) == 0:
            return partial(identityM)
            
    reverse = [True, False]
    order = [(0,1), (1,0)]    
    bestScore = 1000
    for r, o in product(reverse, order):        
        model = trainLSTM(t, inColors=inColors, colors=colors, inRel=inRel,\
                          outRel=outRel, reverse=r, order=o)
        
        score = 0
        for s in t.trainSamples:
            m = predictLSTM(s.inMatrix, model, inColors, colors, inRel, rel, r, o)
            score += incorrectPixels(m, s.outMatrix.m)
        if score < bestScore:
            bestScore=score
            ret = partial(predictLSTM, model=model, inColors=inColors,\
                          colors=colors, inRel=inRel, rel=rel, reverse=r, order=o) 
            if bestScore==0:
                return ret
    return ret

# %% Other utility functions

def insertShape(matrix, shape):
    """
    Given a matrix (numpy.ndarray) and a Shape, this function returns the
    same matrix but with the shape inserted.
    """
    m = matrix.copy()
    shapeM = shape.m.copy()
    for i,j in np.ndindex(shape.shape):
        if shapeM[i,j] != 255:
            if shape.position[0]+i<matrix.shape[0] and shape.position[1]+j<matrix.shape[1]\
                    and shape.position[0]+i >= 0 and shape.position[1]+j >= 0:
                m[tuple(map(operator.add, (i,j), shape.position))] = shapeM[i,j]
    return m

def deleteShape(matrix, shape, backgroundColor):
    """
    Given a matrix (numpy.ndarray) and a Shape, this function substitutes
    the shape by the background color of the matrix.
    """
    m = matrix.copy()
    for c in shape.pixels:
        m[tuple(map(operator.add, c, shape.position))] = backgroundColor
    return m

def symmetrizeSubmatrix(matrix, ud=False, lr=False, rotation=False, newColor=None, subShape=None):
    """
    Given a Matrix, make the non-background part symmetric
    """
    m = matrix.m.copy()
    bC = matrix.backgroundColor
    if np.all(m == bC):
        return m
    x1, x2, y1, y2 = 0, m.shape[0]-1, 0, m.shape[1]-1
    while x1 <= x2 and np.all(m[x1,:] == bC):
        x1 += 1
    while x2 >= x1 and np.all(m[x2,:] == bC):
        x2 -= 1
    while y1 <= y2 and np.all(m[:,y1] == bC):
        y1 += 1
    while y2 >= y1 and np.all(m[:,y2] == bC):
        y2 -= 1
    subMat = m[x1:x2+1,y1:y2+1].copy()
    
    if subShape == None:
        symList = []
        if ud:
            symList.append(np.flipud(subMat.copy()))
            if lr:
                symList.append(np.fliplr(subMat.copy()))
                symList.append(np.fliplr(np.flipud(subMat.copy())))
            elif lr:
                symList.append(np.fliplr(subMat.copy()))
        elif rotation:
            for x in range(1,4):
                symList.append(np.rot90(subMat.copy(),x))
        for newSym in symList:
            score, bestScore = 0, 0
            bestX, bestY = 0, 0
            for i, j in np.ndindex((m.shape[0]-newSym.shape[0]+1,m.shape[1]-newSym.shape[1]+1)):
                score = np.count_nonzero(np.logical_or(m[i:i+newSym.shape[0],j:j+newSym.shape[1]] == newSym, newSym == bC))
                if score > bestScore:
                    bestX, bestY = i, j
                    bestScore = score
            for i, j in np.ndindex(newSym.shape):
                if newSym[i,j] != bC:
                    if newColor == None:
                        m[bestX+i, bestY+j] = newSym[i,j]
                    else:
                        if m[bestX+i,bestY+j] == bC:
                           m[bestX+i, bestY+j] = newColor 
    else:
        found = False
        for x in range(subMat.shape[0]-subShape.shape[0]+1):
            for y in range(subMat.shape[1]-subShape.shape[1]+1):
                #if np.all(m[x1+x:x1+x+subShape.shape[0],y1+y:y1+y+subShape.shape[1]]==subShape.m):
                if np.all(np.equal(m[x1+x:x1+x+subShape.shape[0],y1+y:y1+y+subShape.shape[1]] == bC,subShape.m == 255)):
                    found = True   
                    break
            if found:
                break
        if not found:
            return m
        if ud and lr:
            if 2*x+x1+subShape.shape[0] > m.shape[0] or 2*y+y1+subShape.shape[0]> m.shape[1]:
                return m
            for i in range(subMat.shape[0]):
                for j in range(subMat.shape[1]):
                    if subMat[i][j] != bC:
                        m[2*x+x1+subShape.shape[0]-i-1,y1+j] = subMat[i,j]
                        m[x1+i,2*y+y1+subShape.shape[0]-j-1] = subMat[i,j]
                        m[2*x+x1+subShape.shape[0]-i-1,2*y+y1+subShape.shape[0]-j-1] = subMat[i,j]
        elif rotation:
            if x1+y+x+subShape.shape[0] > m.shape[0] or y1+x+y+subShape.shape[1] > m.shape[1]\
                or y1+y-x+subMat.shape[0] >= m.shape[0] or x1+x-y+subMat.shape[1] >= m.shape[1]\
                or x1+2*x+subShape.shape[0] > m.shape[0] or y1+2*y+subShape.shape[0] > m.shape[1]:
                return m
            for i in range(subMat.shape[0]):
                for j in range(subMat.shape[1]):
                    if subMat[i,j] != bC:
                        m[x1+x+subShape.shape[0]+y-j-1,y1+y-x+i] = subMat[i,j]
                        m[x1+x-y+j,y1+y+subShape.shape[0]+x-i-1] = subMat[i,j]
                        m[x1+2*x+subShape.shape[0]-i-1,y1+2*y+subShape.shape[0]-j-1] = subMat[i,j]        
    return m

def getBestSymmetrizeSubmatrix(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "symmetrizeSubmatrix" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    rotation, lr, ud = False, False, False
    croppedSamples = [cropAllBackground(s.outMatrix) for s in t.trainSamples]
    if all(np.all(np.flipud(m)==m) for m in croppedSamples):
        lr = True
    if all(np.all(np.fliplr(m)==m) for m in croppedSamples):    
        ud = True    
    if all(m.shape[0]==m.shape[1] and np.all(np.rot90(m)==m) for m in croppedSamples):
        rotation = True
    for sh in t.commonInDShapes:
        bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeSubmatrix,\
                                                        lr=lr,ud=ud,rotation=rotation,subShape=sh), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeSubmatrix,lr=lr,\
                                                            ud=ud,rotation=rotation), bestScore, bestFunction)
    return bestFunction

def colorMap(matrix, cMap):
    """
    cMap is a dict of color changes. Each input color can map to one and only
    one output color. Only valid if t.sameIOShapes.
    """
    m = matrix.m.copy()
    for i,j in np.ndindex(m.shape):
        if m[i,j] in cMap.keys(): # Otherwise, it means m[i,j] unchanged
            m[i,j] = cMap[matrix.m[i,j]]
    return m

def revertColorOrder(matrix):
    m = matrix.m.copy()
    colors = [color for color,count in sorted(matrix.colorCount.items(), key=lambda item:item[1])]
    colorDict = {}
    for i in range(len(colors)):
        colorDict[colors[i]] = colors[len(colors)-i-1]
    for i,j in np.ndindex(m.shape):
        m[i,j] = colorDict[m[i,j]]
    return m

def changeColorShapes(matrix, shapes, color):
    """
    Given a matrix (numpy.ndarray), a list of Shapes (they are expected to
    be present in the matrix) and a color, this function returns the same
    matrix, but with the shapes of the list having the given color.
    """
    if len(shapes) == 0:
        return matrix
    m = matrix.copy()
    if color not in list(range(10)):
        return m
    for s in shapes:
        for c in s.pixels:
            m[tuple(map(operator.add, c, s.position))] = color
    return m

def changeShapes(m, inColor, outColor, bigOrSmall=None, isBorder=None):
    """
    Given a Matrix, this function changes the Shapes of the matrix
    that have color inColor to having the color outColor, if they satisfy the
    given conditions bigOrSmall (is the shape the smallest/biggest one?) and
    isBorder.
    """
    return changeColorShapes(m.m.copy(), m.getShapes(inColor, bigOrSmall, isBorder), outColor)

def paintShapesInHalf(matrix, shapeColor, color, half, diagonal=False, middle=None):
    """
    Half can be 'u', 'd', 'l' or 'r'.
    """
    m = matrix.m.copy()
    if diagonal:
        shapesToPaint = [shape for shape in matrix.dShapes if shape.color==shapeColor]
    else:
        shapesToPaint = [shape for shape in matrix.shapes if shape.color==shapeColor]
    
    for shape in shapesToPaint:
        if (shape.shape[0]%2)==0 or middle!=None:
            if middle==True:
                iLimit=int((shape.shape[0]+1)/2)
            else:
                iLimit=int(shape.shape[0]/2)
            if half=='u':
                for i,j in np.ndindex((iLimit, shape.shape[1])):
                    if shape.m[i,j]==shape.color:
                        m[shape.position[0]+i, shape.position[1]+j] = color
            if half=='d':
                for i,j in np.ndindex((iLimit, shape.shape[1])):
                    if shape.m[shape.shape[0]-1-i,j]==shape.color:
                        m[shape.position[0]+shape.shape[0]-1-i, shape.position[1]+j] = color
        if (shape.shape[1]%2)==0 or middle!=None:
            if middle==True:
                jLimit=int((shape.shape[1]+1)/2)
            else:
                jLimit=int(shape.shape[1]/2)
            if half=='l':
                for i,j in np.ndindex((shape.shape[0], jLimit)):
                    if shape.m[i,j]==shape.color:
                        m[shape.position[0]+i, shape.position[1]+j] = color
            if half=='r':
                for i,j in np.ndindex((shape.shape[0], jLimit)):
                    if shape.m[i,shape.shape[1]-1-j]==shape.color:
                        m[shape.position[0]+i, shape.position[1]+shape.shape[1]-1-j] = color
            
    return m

def getBestPaintShapesInHalf(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "paintShapesInHalf" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    for half in ['u', 'd', 'l', 'r']:
        for cic in t.commonChangedInColors:
            for coc in t.commonChangedOutColors:
                for middle, diagonal in product([None, True, False], [True, False]):
                    f = partial(paintShapesInHalf, shapeColor=cic, color=coc,\
                                half=half, diagonal=diagonal, middle=middle)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
    return bestFunction

# %% Paint shapes from border color

def paintShapeFromBorderColor(matrix, shapeColors, fixedColors, diagonals=False):
    m = matrix.m.copy()
    shapesToChange = [shape for shape in matrix.shapes if shape.color in shapeColors]
    
    for shape in shapesToChange:
        neighbourColors = Counter()
        for i,j in np.ndindex(shape.shape):
            if shape.m[i,j]!=255:
                if diagonals:
                    neighbourColors += Counter(getAllNeighbourColors(matrix.m,\
                                               shape.position[0]+i, shape.position[1]+j,\
                                               kernel=3, border=-1))
                else:
                    neighbourColors += Counter(getNeighbourColors(matrix.m,\
                                               shape.position[0]+i, shape.position[1]+j,\
                                               border=-1))
        for color in fixedColors|shapeColors|set([-1]):
            neighbourColors.pop(color, None)
        if len(neighbourColors)>0:
            newColor = max(neighbourColors.items(), key=operator.itemgetter(1))[0]
            m = changeColorShapes(m, [shape], newColor)
    return m

def getBestPaintShapeFromBorderColor(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "paintShapeFromBorderColor" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    shapeColors = t.commonChangedInColors
    fixedColors = t.fixedColors
    for diagonals in [True, False]:
        f = partial(paintShapeFromBorderColor, shapeColors=shapeColors,\
                    fixedColors=fixedColors, diagonals=diagonals)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
    return bestFunction

# %% Things with features
def isFixedShape(shape, fixedShapeFeatures):
    return shape.hasFeatures(fixedShapeFeatures)

def hasFeatures(candidate, reference):
    if all([i==False for i in reference]):
        return False
    for i in range(len(reference)):
        if reference[i] and not candidate[i]:
            return False
    return True

# %% Change Colors with features

def getClosestFixedShapeColor(shape, fixedShapes):
    def getDistance(x1, x2):
        x, y = sorted((x1, x2))
        if x[0] <= x[1] < y[0] and all( y[0] <= y[1] for y in (x1,x2)):
            return y[0] - x[1]
        return 0
    
    color = 0
    minDistance = 1000
    for fs in fixedShapes:
        xDist = getDistance([fs.position[0],fs.position[0]+fs.shape[0]-1], \
                            [shape.position[0],shape.position[0]+shape.shape[0]-1])
        yDist = getDistance([fs.position[1],fs.position[1]+fs.shape[1]-1], \
                            [shape.position[1],shape.position[1]+shape.shape[1]-1])
        
        if xDist+yDist < minDistance:
            minDistance = xDist+yDist
            color = fs.color
    return color

def getShapeFeaturesForColorChange(t, fixedShapeFeatures=None, fixedColors=None,\
                                   predict=False):  
    shapeFeatures = []
    
    if predict:
        matrices = [t]
    else:
        matrices = [s.inMatrix for s in t.trainSamples]
                  
    for m in range(len(matrices)):
        # Smallest and biggest shapes:
        biggestShape = 0
        smallestShape = 1000
        for shape in matrices[m].shapes:
            if shape.color not in fixedColors:
                if shape.nPixels>biggestShape:
                    biggestShape=shape.nPixels
                if shape.nPixels<smallestShape:
                    smallestShape=shape.nPixels
                    
        # Closest fixed shape
        if predict:
            fixedShapes = []
            for shape in matrices[0].shapes:
                if shape.hasFeatures(fixedShapeFeatures):
                    fixedShapes.append(shape)
        else:
            fixedShapes = t.trainSamples[m].fixedShapes
        
        for shape in matrices[m].shapes:
            shFeatures = []
            for c in range(10):
                shFeatures.append(shape.color==c)
            shFeatures.append(shape.isBorder)
            shFeatures.append(not shape.isBorder)
            shFeatures.append(shape.lrSymmetric)
            shFeatures.append(shape.udSymmetric)
            shFeatures.append(shape.d1Symmetric)
            shFeatures.append(shape.d2Symmetric)
            shFeatures.append(shape.isSquare)
            shFeatures.append(shape.isRectangle)
            for nPix in range(1,30):
                shFeatures.append(shape.nPixels==nPix)
            for nPix in range(1,6):
                shFeatures.append(shape.nPixels>nPix)
            for nPix in range(2,7):
                shFeatures.append(shape.nPixels<nPix)
            shFeatures.append((shape.nPixels%2)==0)
            shFeatures.append((shape.nPixels%2)==1)
            for h in range(5):
                shFeatures.append(shape.nHoles==h)
            shFeatures.append(shape.nPixels==biggestShape)
            shFeatures.append(shape.nPixels==smallestShape)
            #shFeatures.append(self.isUniqueShape(sh))
            #shFeatures.append(not self.isUniqueShape(sh))
            closestFixedShapeColor = getClosestFixedShapeColor(shape, fixedShapes)
            for c in range(10):
                shFeatures.append(closestFixedShapeColor==c)
                
            shapeFeatures.append(shFeatures)
            
    return shapeFeatures

def getColorChangesWithFeatures(t):
    shapeFeatures = getShapeFeaturesForColorChange(t, fixedColors=t.fixedColors)
    colorChangesWithFeatures = {}
    nFeatures = len(shapeFeatures[0])
    trueList = []
    falseList = []
    for i in range(nFeatures):
        trueList.append(True)
        falseList.append(False)
    for c in t.totalOutColors:
        colorChangesWithFeatures[c] = trueList
    changeCounter = Counter() # How many shapes change to color c?
    # First, initialise the colorChangesWithFeatures. For every outColor, we
    # detect which features are True for all the shapes that change to that color.
    shapeCounter = 0            
    for sample in t.trainSamples:
        for shape in sample.inMatrix.shapes:
            shapeChanges = False
            for i,j in np.ndindex(shape.shape):
                if shape.m[i,j]!=255:
                    if sample.outMatrix.m[shape.position[0]+i,shape.position[1]+j]!=shape.m[i,j]:
                        color = sample.outMatrix.m[shape.position[0]+i,shape.position[1]+j]
                        shapeChanges=True
                        break
            if shapeChanges:
                changeCounter[color] += 1
                colorChangesWithFeatures[color] = \
                [colorChangesWithFeatures[color][x] and shapeFeatures[shapeCounter][x]\
                 for x in range(nFeatures)]
            shapeCounter += 1
    # Now, there might be more True values than necessary in a certain entry
    # of colorChangesWithFeatures. Therefore, we try to determine the minimum
    # number of necessary True features.
    for c in t.totalOutColors:
        if colorChangesWithFeatures[c] == trueList:
            continue
        trueIndices = [i for i, x in enumerate(colorChangesWithFeatures[c]) if x]
        # First, check if only one feature is enough
        goodIndices = []
        for index in trueIndices:
            trueCount = 0
            featureList = falseList.copy()
            featureList[index] = True
            for sf in shapeFeatures:
                if hasFeatures(sf, featureList):
                    trueCount += 1
            # If the true count matches the number of changed shapes, we're done!
            if trueCount == changeCounter[c]:
                goodIndices.append(index)
        if len(goodIndices) > 0:
            featureList = falseList.copy()
            for index in goodIndices:
                featureList[index] = True
            colorChangesWithFeatures[c] = featureList
        # If we're not done, then check with combinations of 2 features
        else:
            for i,j in combinations(trueIndices, 2):
                trueCount = 0
                featureList = falseList.copy()
                featureList[i] = True
                featureList[j] = True
                for sf in shapeFeatures:
                    if hasFeatures(sf, featureList):
                        trueCount += 1
                # If the true count matches the number of changed shapes, we're done!
                if trueCount == changeCounter[c]:
                    colorChangesWithFeatures[c] = featureList
                    break   
                    
    return colorChangesWithFeatures

def changeShapesWithFeatures(matrix, ccwf, fixedColors, fixedShapeFeatures):
    """
    ccwp stands for 'color change with properties'. It's a dictionary. Its keys
    are integers encoding the color of the output shape, and its values are the
    properties that the input shape has to satisfy in order to execute the
    color change.
    """
    featureList = getShapeFeaturesForColorChange(matrix, fixedColors=fixedColors,\
                                                 fixedShapeFeatures=fixedShapeFeatures,\
                                                 predict=True)
    m = matrix.m.copy()
    sortedCcwf = {k: v for k, v in sorted(ccwf.items(), key=lambda item: sum(item[1]))}
    for color in sortedCcwf.keys():
        for sh in range(len(matrix.shapes)):
            if (matrix.shapes[sh].color in fixedColors):# or \
            #(matrix.shapes[sh].hasFeatures(fixedShapeFeatures)):
                continue
            if hasFeatures(featureList[sh], ccwf[color]):
                m = changeColorShapes(m, [matrix.shapes[sh]], color)
                #break
    return m


# %% Change pixels with features

def pixelRecolor(t):
    """
    if t.sameIOShapes
    """
    Input = [s.inMatrix.m for s in t.trainSamples]
    Output = [s.outMatrix.m for s in t.trainSamples]
        
    Best_Dict = -1
    Best_Q1 = -1
    Best_Q2 = -1
    Best_v = -1
    
    # v ranges from 0 to 3. This gives an extra flexibility of measuring distance from any of the 4 corners
    Pairs = []
    for t in range(15):
        for Q1 in range(1,8):
            for Q2 in range(1,8):
                if Q1+Q2 == t:
                    Pairs.append((Q1,Q2))
                    
    for Q1, Q2 in Pairs:
        for v in range(4):
            if Best_Dict != -1:
                continue
            possible = True
            Dict = {}
            
            for x, y in zip(Input, Output):
                n = len(x)
                k = len(x[0])
                for i in range(n):
                    for j in range(k):
                        if v == 0 or v ==2:
                            p1 = i%Q1
                        else:
                            p1 = (n-1-i)%Q1
                        if v == 0 or v ==3:
                            p2 = j%Q2
                        else :
                            p2 = (k-1-j)%Q2
                        color1 = x[i][j]
                        color2 = y[i][j]
                        if color1 != color2:
                            rule = (p1, p2, color1)
                            if rule not in Dict:
                                Dict[rule] = color2
                            elif Dict[rule] != color2:
                                possible = False
            if possible:
                
                # Let's see if we actually solve the problem
                for x, y in zip(Input, Output):
                    n = len(x)
                    k = len(x[0])
                    for i in range(n):
                        for j in range(k):
                            if v == 0 or v ==2:
                                p1 = i%Q1
                            else:
                                p1 = (n-1-i)%Q1
                            if v == 0 or v ==3:
                                p2 = j%Q2
                            else :
                                p2 = (k-1-j)%Q2
                           
                            color1 = x[i][j]
                            rule = (p1,p2,color1)
                            
                            if rule in Dict:
                                color2 = 0 + Dict[rule]
                            else:
                                color2 = 0 + y[i][j]
                            if color2 != y[i][j]:
                                possible = False 
                if possible:
                    Best_Dict = Dict
                    Best_Q1 = Q1
                    Best_Q2 = Q2
                    Best_v = v
        
    if Best_Dict == -1:
        return [-1]#meaning that we didn't find a rule that works for the traning cases
    else:
        return [Best_Dict, Best_v, Best_Q1, Best_Q2]
    
def executePixelRecolor(matrix, Best_Dict, Best_v, Best_Q1, Best_Q2):
    m = np.zeros(matrix.shape, dtype = np.uint8)
    for i,j in np.ndindex(matrix.shape):
        if Best_v == 0 or Best_v ==2:
            p1 = i%Best_Q1
        else:
            p1 = (matrix.shape[0]-1-i)%Best_Q1
        if Best_v == 0 or Best_v ==3:
            p2 = j%Best_Q2
        else :
            p2 = (matrix.shape[1]-1-j)%Best_Q2
       
        color1 = matrix.m[i,j]
        rule = (p1, p2, color1)
        if (p1, p2, color1) in Best_Dict:
            m[i][j] = 0 + Best_Dict[rule]
        else:
            m[i][j] = 0 + color1
 
    return m

def doRulesWithReference(m, reference, rules):
    for i,j in np.ndindex(m.shape):
        y = (m[i,j], reference[i,j])
        if y in rules.keys():
            m[i,j] = rules[y]
    return m    

def doPixelMod2Row(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesRow = np.ones(m.shape[1], dtype=np.uint8)
    for i in range(m.shape[0]):
        if i%2 == 0:
            reference[i,:] = onesRow.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod3Row(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesRow = np.ones(m.shape[1], dtype=np.uint8)
    twosRow = np.full(m.shape[1], 2, dtype=np.uint8)
    for i in range(m.shape[0]):
        if i%3 == 0:
            reference[i,:] = onesRow.copy()
        elif i%3 == 1:
            reference[i,:] = twosRow.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod2RowReverse(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesRow = np.ones(m.shape[1], dtype=np.uint8)
    for i in range(m.shape[0]):
        if i%2 == 0:
            reference[m.shape[0]-i-1,:] = onesRow.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod3RowReverse(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesRow = np.ones(m.shape[1], dtype=np.uint8)
    twosRow = np.full(m.shape[1], 2, dtype=np.uint8)
    for i in range(m.shape[0]):
        if i%3 == 0:
            reference[m.shape[0]-i-1,:] = onesRow.copy()
        elif i%3 == 1:
            reference[m.shape[0]-i-1,:] = twosRow.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod2Col(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesCol = np.ones(m.shape[0], dtype=np.uint8)
    for j in range(m.shape[1]):
        if j%2 == 0:
            reference[:,j] = onesCol.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod3Col(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesCol = np.ones(m.shape[0], dtype=np.uint8)
    twosCol = np.full(m.shape[0], 2, dtype=np.uint8)
    for j in range(m.shape[1]):
        if j%3 == 0:
            reference[:,j] = onesCol.copy()
        elif j%3 == 1:
            reference[:,j] = twosCol.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod2ColReverse(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesCol = np.ones(m.shape[0], dtype=np.uint8)
    for j in range(m.shape[1]):
        if j%2 == 0:
            reference[:,m.shape[1]-j-1] = onesCol.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod3ColReverse(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesCol = np.ones(m.shape[0], dtype=np.uint8)
    twosCol = np.full(m.shape[0], 2, dtype=np.uint8)
    for j in range(m.shape[1]):
        if j%3 == 0:
            reference[:,m.shape[1]-j-1] = onesCol.copy()
        elif j%3 == 1:
            reference[:,m.shape[1]-j-1] = twosCol.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod2Alternate(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        reference[i,j] = (i+j)%2
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod3Alternate(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        reference[i,j] = (i+j)%3
    m = doRulesWithReference(m, reference, rules)
    return m

def getPixelChangeCriteria(t):
    # Row
    # Mod 2
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesRow = np.ones(sample.inMatrix.shape[1], dtype=np.uint8)
        for i in range(sample.inMatrix.shape[0]):
            if i%2 == 0:
                reference[i,:] = onesRow.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod2Row, rules=x)
    # Mod 3
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesRow = np.ones(sample.inMatrix.shape[1], dtype=np.uint8)
        twosRow = np.full(sample.inMatrix.shape[1], 2, dtype=np.uint8)
        for i in range(sample.inMatrix.shape[0]):
            if i%3 == 0:
                reference[i,:] = onesRow.copy()
            elif i%3 == 1:
                reference[i,:] = twosRow.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod3Row, rules=x)
    
    # Row Reverse
    # Mod 2
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesRow = np.ones(sample.inMatrix.shape[1], dtype=np.uint8)
        for i in range(sample.inMatrix.shape[0]):
            if i%2 == 0:
                reference[sample.inMatrix.shape[0]-i-1,:] = onesRow.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod2RowReverse, rules=x)
    # Mod 3
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesRow = np.ones(sample.inMatrix.shape[1], dtype=np.uint8)
        twosRow = np.full(sample.inMatrix.shape[1], 2, dtype=np.uint8)
        for i in range(sample.inMatrix.shape[0]):
            if i%3 == 0:
                reference[sample.inMatrix.shape[0]-i-1,:] = onesRow.copy()
            elif i%3 == 1:
                reference[sample.inMatrix.shape[0]-i-1,:] = twosRow.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod3RowReverse, rules=x)

    # Col
    # Mod 2
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesCol = np.ones(sample.inMatrix.shape[0], dtype=np.uint8)
        for j in range(sample.inMatrix.shape[1]):
            if j%2 == 0:
                reference[:,j] = onesCol.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod2Col, rules=x)
    # Mod 3
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesCol = np.ones(sample.inMatrix.shape[0], dtype=np.uint8)
        twosCol = np.full(sample.inMatrix.shape[0], 2, dtype=np.uint8)
        for j in range(sample.inMatrix.shape[1]):
            if j%3 == 0:
                reference[:,j] = onesCol.copy()
            elif j%3 == 1:
                reference[:,j] = twosCol.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod3Col, rules=x)
    
    # Col Reverse
    # Mod 2
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesCol = np.ones(sample.inMatrix.shape[0], dtype=np.uint8)
        for j in range(sample.inMatrix.shape[1]):
            if j%2 == 0:
                reference[:,sample.inMatrix.shape[1]-j-1] = onesCol.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod2ColReverse, rules=x)
    # Mod 3
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesCol = np.ones(sample.inMatrix.shape[0], dtype=np.uint8)
        twosCol = np.full(sample.inMatrix.shape[0], 2, dtype=np.uint8)
        for j in range(sample.inMatrix.shape[1]):
            if j%3 == 0:
                reference[:,sample.inMatrix.shape[1]-j-1] = onesCol.copy()
            elif j%3 == 1:
                reference[:,sample.inMatrix.shape[1]-j-1] = twosCol.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod3ColReverse, rules=x)
    
    # Alternate
    # Mod2
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        for i,j in np.ndindex(sample.inMatrix.shape):
            reference[i,j] = (i+j)%2
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod2Alternate, rules=x)
    # Mod3
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        for i,j in np.ndindex(sample.inMatrix.shape):
            reference[i,j] = (i+j)%3
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod3Alternate, rules=x)
    
    return 0    

# %% Surround Shape

def surroundShape(matrix, shape, color, fixedColors, nSteps = None, forceFull=False, \
                  stepIsShape=False, stepIsNHoles=False):

    """
    Given a matrix (numpy.ndarray), a Shape and a color, this function
    surrounds the given Shape with the given color.
    """
    
    m = matrix.copy()
    shapeMatrix = shape.m.copy()
    
    if nSteps==None:
        if stepIsShape:
            nSteps = int(shape.shape[0]/2)
        elif stepIsNHoles:
            nSteps = shape.nHoles
        else:
            nSteps = 15
    
    step = 0
    
    while step<nSteps:
        step += 1
        if forceFull:
            if shape.position[0]-step<0 or shape.position[0]+shape.shape[0]+step>matrix.shape[0] or\
            shape.position[1]-step<0 or shape.position[1]+shape.shape[1]+step>matrix.shape[1]:
                step -= 1
                break
            
            done = False
            for i in range(shape.position[0]-step, shape.position[0]+shape.shape[0]+step):
                if matrix[i, shape.position[1]-step] in fixedColors:
                    step -= 1
                    done = True
                    break
                if matrix[i, shape.position[1]+shape.shape[1]+step-1] in fixedColors:
                    step -= 1
                    done = True
                    break
            if done:
                break
            for j in range(shape.position[1]-step, shape.position[1]+shape.shape[1]+step):
                if matrix[shape.position[0]-step, j] in fixedColors:
                    step -= 1
                    done = True
                    break
                if matrix[shape.position[0]+shape.shape[0]+step-1, j] in fixedColors:
                    step -= 1
                    done = True
                    break
            if done:
                break
        
        row = np.full(shapeMatrix.shape[1], -1, dtype=np.uint8)
        col = np.full(shapeMatrix.shape[0]+2, -1, dtype=np.uint8)
        newM = shapeMatrix.copy() 
        newM = np.vstack([row,newM,row])
        newM = np.column_stack([col,newM,col])
        
        for i in range(newM.shape[0]):
            for j in range(newM.shape[1]):
                if newM[i,j] != 255:
                    newM[i, j-1] = color
                    break
            for j in reversed(range(newM.shape[1])):
                if newM[i,j] != 255:
                    newM[i, j+1] = color
                    break
                    
        for j in range(newM.shape[1]):
            for i in range(newM.shape[0]):
                if newM[i,j] != 255:
                    newM[i-1, j] = color
                    break
            for i in reversed(range(newM.shape[0])):
                if newM[i,j] != 255:
                    newM[i+1, j] = color
                    break
                    
        shapeMatrix = newM.copy()
                    
    for i,j in np.ndindex(shapeMatrix.shape):
        if shape.position[0]-step+i<0 or shape.position[0]-step+i>=matrix.shape[0] or \
        shape.position[1]-step+j<0 or shape.position[1]-step+j>=matrix.shape[1]:
            continue
        if shapeMatrix[i,j] != 255:
            m[shape.position[0]-step+i, shape.position[1]-step+j] = shapeMatrix[i,j]
        
    return m
    
def surroundAllShapes(matrix, shapeColor, surroundColor, fixedColors, nSteps=None,\
                      forceFull=False, stepIsShape=False, stepIsNHoles=False):
    """
    Given a Matrix, a shapeColor and a surroundColor, this function surrounds
    all the Shapes of color shapeColor with the given surroundColor.
    """
    m = matrix.m.copy()
    shapesToSurround = [s for s in matrix.shapes if s.color == shapeColor]
    if stepIsShape:
        shapesToSurround = [s for s in shapesToSurround if s.isSquare]
    for s in shapesToSurround:
        m = surroundShape(m, s, surroundColor, fixedColors, nSteps=nSteps,\
                          forceFull=forceFull, stepIsShape=stepIsShape,\
                          stepIsNHoles=stepIsNHoles)
    return m

def getBestSurroundShapes(t):    
    """
    Given a Task t, this function returns the partial function that uses the
    function "surroundShapes" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    
    for fc in t.fixedColors:
        for coc in t.commonChangedOutColors:
            f = partial(surroundAllShapes, shapeColor=fc, surroundColor=coc, \
                        fixedColors=t.fixedColors, forceFull=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            
            f = partial(surroundAllShapes, shapeColor=fc, surroundColor=coc, \
                            fixedColors=t.fixedColors, stepIsShape=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            
            f = partial(surroundAllShapes, shapeColor=fc, surroundColor=coc, \
                            fixedColors=t.fixedColors, forceFull=True, stepIsShape=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            
            f = partial(surroundAllShapes, shapeColor=fc, surroundColor=coc, \
                            fixedColors=t.fixedColors, forceFull=False, stepIsNHoles=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            
            f = partial(surroundAllShapes, shapeColor=fc, surroundColor=coc, \
                            fixedColors=t.fixedColors, forceFull=True, stepIsNHoles=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            
            for nSteps in range(1,4):
                f = partial(surroundAllShapes, shapeColor=fc, surroundColor=coc, \
                            fixedColors=t.fixedColors, nSteps=nSteps)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                
                f = partial(surroundAllShapes, shapeColor=fc, surroundColor=coc, \
                            fixedColors=t.fixedColors, nSteps=nSteps, forceFull=True)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
            
    return bestFunction

# %% Extend Color

def extendColor(matrix, direction, cic, fixedColors, color=None, sourceColor=None,\
                deleteExtensionColors=set(), deleteIfBorder=False, \
                breakAtFixedColor=False, mergeColor=None):
    """
    Given a Matrix matrix and a direction, this function extends all the pixels
    of the given color in the given direction direction can be any of:
        'hv', 'h', 'v', 'u', 'd', 'l', 'r', 'all', 'diag', 'd1', 'd2'
    If no color is specified, then all the colors but the changedInColors (cic)
    and the fixedColors are extended.
    """
    
    if sourceColor==None:
        sourceColor=color
        
    matrices = []
    
    # Vertical
    if direction=='all' or direction=='hv' or direction=='v' or direction=='u':
        for j in range(matrix.shape[1]):
            colorCells=False
            start = matrix.shape[0]-1
            for i in reversed(range(matrix.shape[0])):
                if color==None and sourceColor==None:
                    if matrix.m[i,j] not in (fixedColors|cic):
                        sourceColor = matrix.m[i,j]
                if matrix.m[i,j]==sourceColor:
                    m = matrix.m.copy()
                    colorCells=True
                    start = i
                if colorCells:
                    if matrix.m[i,j] in cic:
                        if color==None:
                            m[i,j] = sourceColor
                        else:
                            m[i,j] = color
                    elif m[i,j]!=sourceColor and breakAtFixedColor=="any":
                        sourceColor = m[i,j]
                    elif matrix.m[i,j] in fixedColors and breakAtFixedColor:
                        break
                if colorCells and ((matrix.m[i,j] in deleteExtensionColors) or \
                                   i==0 and deleteIfBorder):
                    currentI = i
                    for i in range(currentI, start):
                        m[i,j] = matrix.m[i,j]
                    break
            if color==None:
                sourceColor=None
            if colorCells:
                matrices.append(m)
    if direction=='all' or direction=='hv' or direction=='v' or direction=='d':
        for j in range(matrix.shape[1]):
            colorCells=False
            start = 0
            for i in range(matrix.shape[0]):
                if color==None and sourceColor==None:
                    if matrix.m[i,j] not in (fixedColors|cic):
                        sourceColor = matrix.m[i,j]
                if matrix.m[i,j]==sourceColor:
                    m = matrix.m.copy()
                    colorCells=True
                    start = i
                if colorCells and matrix.m[i,j] in fixedColors and breakAtFixedColor:
                    break
                if colorCells:
                    if matrix.m[i,j] in cic:
                        if color==None:
                            m[i,j] = sourceColor
                        else:
                            m[i,j] = color
                    elif m[i,j]!=sourceColor and breakAtFixedColor=="any":
                        sourceColor = m[i,j]
                    elif matrix.m[i,j] in fixedColors and breakAtFixedColor:
                        break
                        
                if colorCells and ((matrix.m[i,j] in deleteExtensionColors) or \
                                   i==m.shape[0]-1 and deleteIfBorder):
                    currentI = i+1
                    for i in reversed(range(start, currentI)):
                        m[i,j] = matrix.m[i,j]
                    break
            if color==None:
                sourceColor=None
            if colorCells:
                matrices.append(m)
             
    # Horizontal
    if direction=='all' or direction=='hv' or direction=='h' or direction=='l':
        for i in range(matrix.shape[0]):
            colorCells=False
            start = matrix.shape[1]-1
            for j in reversed(range(matrix.shape[1])):  
                if color==None and sourceColor==None:
                    if matrix.m[i,j] not in (fixedColors|cic):
                        sourceColor = matrix.m[i,j]
                if matrix.m[i,j]==sourceColor:
                    m = matrix.m.copy()
                    colorCells=True
                    start = j
                if colorCells:
                    if matrix.m[i,j] in cic:
                        if color==None:
                            m[i,j] = sourceColor
                        else:
                            m[i,j] = color
                    elif m[i,j]!=sourceColor and breakAtFixedColor=="any":
                        sourceColor = m[i,j]
                    elif matrix.m[i,j] in fixedColors and breakAtFixedColor:
                        break
                if colorCells and ((matrix.m[i,j] in deleteExtensionColors) or \
                                   j==0 and deleteIfBorder):
                    currentJ = j
                    for j in range(currentJ, start):
                        m[i,j] = matrix.m[i,j]
                    break
            if color==None:
                sourceColor=None
            if colorCells:
                matrices.append(m)
    if direction=='all' or direction=='hv' or direction=='h' or direction=='r':
        for i in range(matrix.shape[0]):
            colorCells=False
            start = 0
            for j in range(matrix.shape[1]):
                if color==None and sourceColor==None:
                    if matrix.m[i,j] not in (fixedColors|cic):
                        sourceColor = matrix.m[i,j]
                if matrix.m[i,j]==sourceColor:
                    m = matrix.m.copy()
                    colorCells=True
                    start = j
                if colorCells: 
                    if matrix.m[i,j] in cic:
                        if color==None:
                            m[i,j] = sourceColor
                        else:
                            m[i,j] = color
                    elif m[i,j]!=sourceColor and breakAtFixedColor=="any":
                        sourceColor = m[i,j]
                    elif matrix.m[i,j] in fixedColors and breakAtFixedColor:
                        break
                if colorCells and ((matrix.m[i,j] in deleteExtensionColors) or \
                                   j==m.shape[1]-1 and deleteIfBorder):
                    currentJ = j+1
                    for j in reversed(range(start, currentJ)):
                        m[i,j] = matrix.m[i,j]
                    break
            if color==None:
                sourceColor=None
            if colorCells:
                matrices.append(m)
              
    if len(matrices)>0:
        m = mergeMatrices(matrices, next(iter(cic)), mergeColor=mergeColor)
    else:
        m = matrix.m.copy()
                
    # Diagonal
    if direction=='all' or  direction=='diag' or direction=='d1' or direction=='d2':
        if direction=='diag' or direction=='all':
            directions = ['d1', 'd2']
        else:
            directions = [direction]
        for direction in directions:
            for transpose in [True, False]:
                if transpose and direction=='d1':
                    matrix.m = np.rot90(matrix.m, 2).T
                    m = np.rot90(m, 2).T
                if transpose and direction=='d2':
                    matrix.m = matrix.m.T
                    m = m.T
                if direction=='d2':
                    matrix.m = np.fliplr(matrix.m)
                    m = np.fliplr(m)
                for i in range(-matrix.shape[0]+1, matrix.shape[1]):
                    diag = np.diagonal(matrix.m, i)
                    colorCells=False
                    for j in range(len(diag)):
                        if color==None and sourceColor==None:
                            if i<=0:
                                if matrix.m[-i+j,j] not in (fixedColors|cic):
                                    sourceColor = matrix.m[-i+j,j]
                            else:
                                if matrix.m[j,i+j] not in (fixedColors|cic):
                                    sourceColor = matrix.m[j,i+j]
                        if i<=0:
                            if matrix.m[-i+j,j]==sourceColor:
                                colorCells=True
                            if colorCells:
                                if matrix.m[-i+j,j] in cic:
                                    if color==None:
                                        m[-i+j,j] = sourceColor
                                    else:
                                        m[-i+j,j] = color
                                elif matrix.m[-i+j,j]!=sourceColor and breakAtFixedColor=="any":
                                    sourceColor = m[-i+j,j]
                                elif matrix.m[-i+j,j] in fixedColors and breakAtFixedColor:
                                    break
                            if colorCells and ((matrix.m[-i+j,j] in deleteExtensionColors) or \
                                               j==len(diag)-1 and deleteIfBorder):
                                for j in range(len(diag)):
                                    m[-i+j,j] = matrix.m[-i+j,j]
                                break
                        else:
                            if matrix.m[j,i+j]==sourceColor:
                                colorCells=True
                            if colorCells:
                                if matrix.m[j,i+j] in cic:
                                    if color==None:
                                        m[j,i+j] = sourceColor
                                    else:
                                        m[j,i+j] = color
                                elif matrix.m[j,i+j]!=sourceColor and breakAtFixedColor=="any":
                                    sourceColor = m[j,i+j]
                                elif matrix.m[j,i+j] in fixedColors and breakAtFixedColor:
                                    break
                            if colorCells and ((matrix.m[j,i+j] in deleteExtensionColors) or \
                                               j==len(diag)-1 and deleteIfBorder):
                                for j in range(len(diag)):
                                    m[j,i+j] = matrix.m[j,i+j]
                                break
                    if color==None:
                        sourceColor=None
                if direction=='d2':
                    matrix.m = np.fliplr(matrix.m)
                    m = np.fliplr(m)
                if transpose and direction=='d2':
                    matrix.m = matrix.m.T
                    m = m.T
                if transpose and direction=='d1':
                    matrix.m = np.rot90(matrix.m, 2).T
                    m = np.rot90(m, 2).T

    return m    

def getBestExtendColor(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "extendColor" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    
    mergeColors = t.commonOutColors-t.totalInColors
    if len(mergeColors) == 1:
        mergeColor = next(iter(mergeColors))
    else:
        mergeColor = None
    
    cic = t.commonChangedInColors
    if len(cic)==0:
        return bestFunction
    fixedColors = t.fixedColors
    for d in ['r', 'l', 'h', 'u', 'd', 'v', 'hv', 'd1', 'd2', 'diag', 'all']:
        for dib,bafc in product([True, False], [True, False, "any"]):
            f = partial(extendColor, direction=d, cic=cic, fixedColors=fixedColors,\
                        deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            f = partial(extendColor, direction=d, cic=cic, fixedColors=fixedColors,\
                        deleteExtensionColors=fixedColors,\
                        deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            for coc in t.commonChangedOutColors:    
                f = partial(extendColor, color=coc, direction=d, cic=cic, fixedColors=fixedColors,\
                            deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                f = partial(extendColor, color=coc, direction=d, cic=cic, fixedColors=fixedColors,\
                            deleteExtensionColors=fixedColors,\
                            deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                for fc in t.fixedColors:
                    f = partial(extendColor, color=coc, direction=d, cic=cic, sourceColor=fc, \
                                fixedColors=fixedColors,\
                                deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
                    f = partial(extendColor, color=coc, direction=d, cic=cic, sourceColor=fc, \
                                fixedColors=fixedColors, deleteExtensionColors=fixedColors,\
                                deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
                
    return bestFunction

# %% Fill rectangleInside
def fillRectangleInside(matrix, rectangleColor, fillColor):
    m = matrix.m.copy()
    for shape in matrix.shapes:
        if shape.isRectangle and shape.color==rectangleColor:
            if shape.shape[0] > 2 and shape.shape[1] > 2:
                rect = np.full((shape.shape[0]-2, shape.shape[1]-2), fillColor, dtype=np.uint8)
                m[shape.position[0]+1:shape.position[0]+shape.shape[0]-1,\
                  shape.position[1]+1:shape.position[1]+shape.shape[1]-1] = rect
    return m

# %% Color longest line
def colorLongestLines(matrix, cic, coc, direction):
    """
    cic stands for "changedInColor"
    coc stands for "changedOutColor"
    direction can be one of 4 strings: 'v', 'h', 'hv', 'd' (vertical,
    horizontal, diagonal)
    It is assumed t.sameIOShapes
    """    
    m = matrix.m.copy()
    
    longest=0
    positions = set()
    if direction=='h':
        for i in range(m.shape[0]):
            count = 0
            for j in range(m.shape[1]):
                if m[i,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add((i,j))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add((i,m.shape[1]-1))
        for pos in positions:
            for j in range(pos[1]-longest, pos[1]):
                m[pos[0],j] = coc
        return m                
        
    elif direction=='v':
        for j in range(m.shape[1]):
            count = 0
            for i in range(m.shape[0]):
                if m[i,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add((i,j))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add((m.shape[0]-1,j))
        for pos in positions:
            for i in range(pos[0]-longest, pos[0]):
                m[i,pos[1]] = coc
        return m 
                        
    elif direction=='hv':
        longestH = 0
        longestV = 0
        positionsH = set()
        positionsV = set()
        for i in range(m.shape[0]):
            count = 0
            for j in range(m.shape[1]):
                if m[i,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longestH:
                        if count > longestH:
                            positionsH = set()
                        longestH = count
                        positionsH.add((i,j))
                    count = 0
            if count >= longestH:
                if count > longestH:
                    positionsH = set()
                longestH = count
                positionsH.add((i,m.shape[1]-1))
        for j in range(m.shape[1]):
            count = 0
            for i in range(m.shape[0]):
                if m[i,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longestV:
                        if count > longestV:
                            positionsV = set()
                        longestV = count
                        positionsV.add((i,j))
                    count = 0
            if count >= longestV:
                if count > longestV:
                    positionsV = set()
                longestV = count
                positionsV.add((m.shape[0]-1,j))
        for pos in positionsH:
            for j in range(pos[1]-longestH, pos[1]):
                m[pos[0],j] = coc
        for pos in positionsV:
            for i in range(pos[0]-longestV, pos[0]):
                m[i,pos[1]] = coc
        return m
    
    elif direction=='d':
        # Direction of main diagonal
        for i in reversed(range(m.shape[0])):
            count = 0
            jLimit = min(m.shape[1], m.shape[0]-i)
            for j in range(jLimit):
                if m[i+j,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add(((i+j-1,j-1), 'main'))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add(((i+jLimit-1,jLimit-1), 'main'))
        for j in range(1, m.shape[1]):
            count = 0
            iLimit = min(m.shape[0], m.shape[1]-j)
            for i in range(iLimit):
                if m[i,j+i]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add(((i-1,j+i-1), 'main'))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add(((iLimit-1,j+iLimit-1), 'main'))
                
        # Direction of counterdiagonal
        for i in range(m.shape[0]):
            count = 0
            jLimit = min(m.shape[1], i+1)
            for j in range(jLimit):
                if m[i-j,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add(((i-j+1, j-1), 'counter'))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add(((i-jLimit+1,jLimit-1), 'counter'))
        for j in range(m.shape[1]):
            count = 0
            iLimit = min(m.shape[0], m.shape[1]-j)
            for i in range(iLimit):
                if m[m.shape[0]-i-1,j+i]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add(((m.shape[0]-i,j+i-1), 'counter'))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add(((m.shape[0]-iLimit,j+iLimit-1), 'counter'))
        
        # Draw the lines
        for pos in positions:
            if pos[1]=='main':
                for x in range(longest):
                    m[pos[0][0]-x, pos[0][1]-x] = coc
            else:
                for x in range(longest):
                    m[pos[0][0]+x, pos[0][1]-x] = coc
        return m
    return m
    
# %% Move shapes    

def moveShape(matrix, shape, background, direction, until = -1, nSteps = 100, \
              keepOriginal=False):
    """
    'direction' can be l, r, u, d, ul, ur, dl, dr
    (left, right, up, down, horizontal, vertical, diagonal1, diagonal2)
    'until' can be a color or -1, which will be interpreted as border
    If 'until'==-2, then move until the shape encounters anything
    """
    m = matrix.copy()
    if not keepOriginal:
        m = changeColorShapes(m, [shape], background)
    s = copy.deepcopy(shape)
    if nSteps=="shapeX":
        nSteps = shape.shape[0]
    if nSteps=="shapeY":
        nSteps = shape.shape[1]
    step = 0
    while True and step != nSteps:
        step += 1
        for c in s.pixels:
            pos = (s.position[0]+c[0], s.position[1]+c[1])
            if direction == "l":
                newPos = (pos[0], pos[1]-1)
            if direction == "r":
                newPos = (pos[0], pos[1]+1)
            if direction == "u":
                newPos = (pos[0]-1, pos[1])
            if direction == "d":
                newPos = (pos[0]+1, pos[1])
            if direction == "ul":
                newPos = (pos[0]-1, pos[1]-1)
            if direction == "ur":
                newPos = (pos[0]-1, pos[1]+1)
            if direction == "dl":
                newPos = (pos[0]+1, pos[1]-1)
            if direction == "dr":
                newPos = (pos[0]+1, pos[1]+1)
                
            if newPos[0] not in range(m.shape[0]) or \
            newPos[1] not in range(m.shape[1]):
                if until != -1 and until != -2:
                    return matrix.copy()
                else:
                    return insertShape(m, s)
            if until == -2 and m[newPos] != background:
                return insertShape(m, s)
            if m[newPos] == until:
                return insertShape(m, s)
            
        if direction == "l":
            s.position = (s.position[0], s.position[1]-1)
        if direction == "r":
            s.position = (s.position[0], s.position[1]+1)
        if direction == "u":
            s.position = (s.position[0]-1, s.position[1])
        if direction == "d":
            s.position = (s.position[0]+1, s.position[1])
        if direction == "ul":
            s.position = (s.position[0]-1, s.position[1]-1)
        if direction == "ur":
            s.position = (s.position[0]-1, s.position[1]+1)
        if direction == "dl":
            s.position = (s.position[0]+1, s.position[1]-1)
        if direction == "dr":
            s.position = (s.position[0]+1, s.position[1]+1)
      
    return insertShape(m, s) 
    
def moveAllShapes(matrix, background, direction, until, nSteps=100, color=None, \
                  keepOriginal=False):
    """
    direction can be l, r, u, d, ul, ur, dl, dr, h, v, d1, d2, all, any
    """
    if color==None or color=="multiColor":
        shapesToMove = matrix.multicolorShapes
    elif color=="diagonalMultiColor":
        shapesToMove=matrix.multicolorDShapes
    elif color=="singleColor":
        shapesToMove = [s for s in matrix.shapes if s.color!=background]
    elif color=="diagonalSingleColor":
        shapesToMove = [s for s in matrix.dShapes if s.color!=background]
    else:
        shapesToMove = [s for s in matrix.shapes if s.color in color]
    if direction == 'l':
        shapesToMove.sort(key=lambda x: x.position[1])
    if direction == 'r':
        shapesToMove.sort(key=lambda x: x.position[1]+x.shape[1], reverse=True)
    if direction == 'u':
        shapesToMove.sort(key=lambda x: x.position[0])  
    if direction == 'd':
        shapesToMove.sort(key=lambda x: x.position[0]+x.shape[0], reverse=True)
    m = matrix.m.copy()
    if len(shapesToMove) > 15:
        return m
    for s in shapesToMove:
        newMatrix = m.copy()
        if direction == "any":
            for d in ['l', 'r', 'u', 'd', 'ul', 'ur', 'dl', 'dr']:
                newMatrix = moveShape(m, s, background, d, until, keepOriginal=keepOriginal)
                if not np.all(newMatrix == m):
                    return newMatrix
                    break
        else:
            m = moveShape(m, s, background, direction, until, nSteps, keepOriginal=keepOriginal)
    return m
    
def moveShapeToClosest(matrix, shape, background, until=None, diagonals=False, restore=True):
    """
    Given a matrix (numpy.ndarray) and a Shape, this function moves the
    given shape until the closest shape with the color given by "until".
    """
    m = matrix.copy()
    s = copy.deepcopy(shape)
    m = deleteShape(m, shape, background)
    if until==None:
        if hasattr(shape, "color"):
            until=shape.color
        else:
            return matrix
    if until not in m:
        return matrix
    nSteps = 0
    while True:
        for c in s.pixels:
            pixelPos = tuple(map(operator.add, c, s.position))
            if nSteps <= pixelPos[0] and m[pixelPos[0]-nSteps, pixelPos[1]] == until:
                while nSteps>=0 and m[pixelPos[0]-nSteps, pixelPos[1]]!=background:
                    nSteps-=1
                s.position = (s.position[0]-nSteps, s.position[1])
                return insertShape(m, s)
            if pixelPos[0]+nSteps < m.shape[0] and m[pixelPos[0]+nSteps, pixelPos[1]] == until:
                while nSteps>=0 and m[pixelPos[0]+nSteps, pixelPos[1]]!=background:
                    nSteps-=1
                s.position = (s.position[0]+nSteps, s.position[1])
                return insertShape(m, s)
            if nSteps <= pixelPos[1] and m[pixelPos[0], pixelPos[1]-nSteps] == until:
                while nSteps>=0 and m[pixelPos[0], pixelPos[1]-nSteps]!=background:
                    nSteps-=1
                s.position = (s.position[0], s.position[1]-nSteps)
                return insertShape(m, s)
            if pixelPos[1]+nSteps < m.shape[1] and m[pixelPos[0], pixelPos[1]+nSteps] == until:
                while nSteps>=0 and m[pixelPos[0], pixelPos[1]+nSteps]!=background:
                    nSteps-=1
                s.position = (s.position[0], s.position[1]+nSteps)
                return insertShape(m, s)
            if diagonals:
                if nSteps <= pixelPos[0] and nSteps <= pixelPos[1] and \
                m[pixelPos[0]-nSteps, pixelPos[1]-nSteps] == until:
                    s.position = (s.position[0]-nSteps+1, s.position[1]-nSteps+1)
                    return insertShape(m, s)
                if nSteps <= pixelPos[0] and pixelPos[1]+nSteps < m.shape[1] and \
                m[pixelPos[0]-nSteps, pixelPos[1]+nSteps] == until:
                    s.position = (s.position[0]-nSteps+1, s.position[1]+nSteps-1)
                    return insertShape(m, s)
                if pixelPos[0]+nSteps < m.shape[0] and nSteps <= pixelPos[1] and \
                m[pixelPos[0]+nSteps, pixelPos[1]-nSteps] == until:
                    s.position = (s.position[0]+nSteps-1, s.position[1]-nSteps+1)
                    return insertShape(m, s)
                if pixelPos[0]+nSteps < m.shape[0] and pixelPos[1]+nSteps < m.shape[1] and \
                m[pixelPos[0]+nSteps, pixelPos[1]+nSteps] == until:
                    s.position = (s.position[0]+nSteps-1, s.position[1]+nSteps-1)
                    return insertShape(m, s)
        nSteps += 1
        if nSteps > m.shape[0] and nSteps > m.shape[1]:
            if restore:
                return matrix
            else:
                return m
        
def moveAllShapesToClosest(matrix, background, colorsToMove=None, until=None, \
                           diagonals=False, restore=True, fixedShapeFeatures=None):
    """
    This function moves all the shapes with color "colorsToMove" until the
    closest shape with color "until".
    """
    m = matrix.m.copy()
    if len(matrix.shapes) > 25:
        return m
    fixedShapes = []
    if until == None:
        colorsToMove = []
        for shape in matrix.shapes:
            if hasFeatures(shape.boolFeatures, fixedShapeFeatures):
                fixedShapes.append(shape)
                colorsToMove.append(shape.color)
    elif colorsToMove==None:
        colorsToMove = matrix.colors - set([background, until])
    else:
        colorsToMove = [colorsToMove]
    for ctm in colorsToMove:
        for shape in matrix.shapes:
            if shape not in fixedShapes:
                if shape.color == ctm:
                    m = moveShapeToClosest(m, shape, background, until, diagonals, restore)
    return m

def getBestMoveShapes(t, candidate):
    """
    This functions tries to find, for a given task t, the best way to move
    shapes.
    """
    directions = ['l', 'r', 'u', 'd', 'ul', 'ur', 'dl', 'dr', 'any']
    bestScore = 1000
    bestFunction = partial(identityM)
    
    doSingleColor = all([len(s.inMatrix.shapes) < 50 for s in t.trainSamples])
    doDSingleColor = all([len(s.inMatrix.dShapes) < 15 for s in t.trainSamples])
    doMulticolor = all([len(s.inMatrix.multicolorShapes) < 15 for s in t.trainSamples])
        
    # Move all shapes in a specific direction, until a non-background thing is touched
    for d in directions:
        if doSingleColor:
            f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                        direction=d, color="singleColor")
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
        if doDSingleColor:
            f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                        direction=d, color="diagonalSingleColor")
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
        if doMulticolor:
            f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                        direction=d, color="multiColor")
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
        f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                    direction=d, color="diagonalMultiColor")
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                    direction=d, color="diagonalMultiColor", nSteps="shapeX")
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                    direction=d, color="diagonalMultiColor", nSteps="shapeY")
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        if doDSingleColor:
            f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                        direction=d, color="diagonalSingleColor", nSteps="shapeX")
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                        direction=d, color="diagonalSingleColor", nSteps="shapeY")
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
        
    if doSingleColor:
        colorsToChange = list(t.colors - t.fixedColors - set({t.backgroundColor}))
        ctc = [[c] for c in colorsToChange] + [colorsToChange] # Also all colors
        for c in ctc:
            for d in directions:
                moveUntil = colorsToChange + [-1] + [-2] #Border, any
                for u in moveUntil:
                    f = partial(moveAllShapes, color=c, background=t.backgroundColor,\
                                    direction=d, until=u)
                    bestFunction, bestScore, isPerfect = updateBestFunction(t, f, bestScore, bestFunction, \
                                                                            checkPerfect=True, prevScore=candidate.score)
                    if isPerfect:
                        return bestFunction
                    if bestScore==0:
                        return bestFunction
                for nSteps in range(1, 5):
                    f = partial(moveAllShapes, color=c, background=t.backgroundColor,\
                                direction=d, until=-2, nSteps=nSteps)
                    bestFunction, bestScore, isPerfect = updateBestFunction(t, f, bestScore, bestFunction,\
                                                                            checkPerfect=True, prevScore=candidate.score)
                    if isPerfect:
                        return bestFunction
                    if bestScore==0:
                        return bestFunction
                    f = partial(moveAllShapes, color=c, background=t.backgroundColor,\
                                direction=d, until=-2, nSteps=nSteps, keepOriginal=True)
                    bestFunction, bestScore, isPerfect = updateBestFunction(t, f, bestScore, bestFunction,\
                                                                            checkPerfect=True, prevScore=candidate.score)
                    if isPerfect:
                        return bestFunction
                    if bestScore==0:
                        return bestFunction
                
    
        if t.backgroundColor != -1 and hasattr(t, 'fixedColors'):
            colorsToMove = t.almostCommonColors - set([t.backgroundColor]) - t.fixedColors
            for ctm in colorsToMove:
                for uc in t.unchangedColors:
                    f = partial(moveAllShapesToClosest, colorsToMove=ctm,\
                                     background=t.backgroundColor, until=uc)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
                    
                    f = partial(moveAllShapesToClosest, colorsToMove=ctm,\
                                background=t.backgroundColor, until=uc, diagonals=True)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
                
        if all([len(sample.fixedShapes)>0 for sample in t.trainSamples]):
            f = partial(moveAllShapesToClosest, background=t.backgroundColor,\
                        fixedShapeFeatures = t.fixedShapeFeatures)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
        
    return bestFunction

# %% Complete rectangles
def completeRectangles(matrix, sourceColor, newColor):
    """
    It is assumed that the background is clear.
    """
    m = matrix.m.copy()
    for s in matrix.multicolorDShapes:
        if hasattr(s, 'color') and s.color==sourceColor:
            newShape = copy.deepcopy(s)
            newShape.m[newShape.m==255] = newColor
            m = insertShape(m, newShape)
    return m

# %% Delete shapes
# Like that this only solves task 96. It's clearly not general enough.
def deletePixels(matrix, diagonals=False):
    """
    Given a matrix, this functions deletes all the pixels. This means that all
    the dShapes consisting of only 1 pixel are converted to the color that
    surrounds most of that pixel.
    """
    m = matrix.m.copy()
    if m.shape[0]==1 and m.shape[1]==1:
        return m
    
    if diagonals:
        shapes = matrix.dShapes
    else:
        shapes = matrix.shapes
    for s in shapes:
        if s.nPixels==1:
            surrColors = Counter()
            if s.position[0]>0:
                surrColors[m[s.position[0]-1, s.position[1]]] += 1
            if s.position[1]<m.shape[1]-1:
                surrColors[m[s.position[0], s.position[1]+1]] += 1
            if s.position[0]<m.shape[0]-1:
                surrColors[m[s.position[0]+1, s.position[1]]] += 1
            if s.position[1]>0:
                surrColors[m[s.position[0], s.position[1]-1]] += 1
            if len(set(surrColors.values()))==1:
                if s.position[0]>0 and s.position[1]>0:
                    surrColors[m[s.position[0]-1, s.position[1]-1]] += 1
                if s.position[0]>0 and s.position[1]<m.shape[1]-1:
                    surrColors[m[s.position[0]-1, s.position[1]+1]] += 1
                if s.position[0]<m.shape[0]-1 and s.position[1]<m.shape[1]-1:
                    surrColors[m[s.position[0]+1, s.position[1]+1]] += 1
                if s.position[0]<m.shape[0]-1 and s.position[1]>0:
                    surrColors[m[s.position[0]+1, s.position[1]-1]] += 1
            
            m[s.position[0],s.position[1]] = max(surrColors.items(), key=operator.itemgetter(1))[0]
    return m

# %% Connect Pixels

def connectPixels(matrix, pixelColor=None, connColor=None, fixedColors=set(),\
                  allowedChanges={}, lineExclusive=False, diagonal=False):
    """
    Given a matrix, this function connects all the pixels that have the same
    color. This means that, for example, all the pixels between two red pixels
    will also be red.
    If "pixelColor" is specified, then only the pixels with the specified color
    will be connected.
    Ìf "connColor" is specified, the color used to connect the pixels will be
    the one given by this parameter.
    If there are any colors in the "fixedColors" set, then it is made sure that
    they remain unchanged.
    "allowedChanges" is a dictionary determining which color changes are
    allowed. It's exclusive with the options unchangedColors and connColor.
    """
    m = matrix.copy()
    # Row
    for i in range(m.shape[0]):
        lowLimit = 0
        while lowLimit < m.shape[1] and matrix[i, lowLimit] != pixelColor:
            lowLimit += 1
        lowLimit += 1
        upLimit = m.shape[1]-1
        while upLimit > lowLimit and matrix[i, upLimit] != pixelColor:
            upLimit -= 1
        if upLimit > lowLimit:
            if lineExclusive:
                for j in range(lowLimit, upLimit):
                    if matrix[i,j] == pixelColor:
                        lowLimit = upLimit
                        break
            for j in range(lowLimit, upLimit):
                if connColor != None:
                    if matrix[i,j] != pixelColor and matrix[i,j] not in fixedColors:
                        m[i,j] = connColor
                else:
                    if matrix[i,j] in allowedChanges.keys():
                        m[i,j] = allowedChanges[matrix[i,j]]
       
    # Column             
    for j in range(m.shape[1]):
        lowLimit = 0
        while lowLimit < m.shape[0] and matrix[lowLimit, j] != pixelColor:
            lowLimit += 1
        lowLimit += 1
        upLimit = m.shape[0]-1
        while upLimit > lowLimit and matrix[upLimit, j] != pixelColor:
            upLimit -= 1
        if upLimit > lowLimit:
            if lineExclusive:
                for i in range(lowLimit, upLimit):
                    if matrix[i,j] == pixelColor:
                        lowLimit = upLimit
                        break
            for i in range(lowLimit, upLimit):
                if connColor != None:
                    if matrix[i,j] != pixelColor and matrix[i,j] not in fixedColors:
                        m[i,j] = connColor
                else:
                    if matrix[i,j] in allowedChanges.keys():
                        m[i,j] = allowedChanges[matrix[i,j]]
    
    # Diagonal         
    if diagonal:
        for d in [1, 2]:
            if d==2:
                matrix = np.fliplr(matrix)
                m = np.fliplr(m)
            for i in range(-matrix.shape[0]-1, matrix.shape[1]):
                diag = np.diagonal(matrix, i)
                lowLimit = 0
                while lowLimit < len(diag) and diag[lowLimit] != pixelColor:
                    lowLimit += 1
                lowLimit += 1
                upLimit = len(diag)-1
                while upLimit > lowLimit and diag[upLimit] != pixelColor:
                    upLimit -= 1
                if upLimit > lowLimit:
                    if lineExclusive:
                        for j in range(lowLimit, upLimit):
                            if i<=0:
                                if matrix[-i+j,j] == pixelColor:
                                    lowLimit = upLimit
                                    break
                            else:
                                if matrix[j,i+j] == pixelColor:
                                    lowLimit = upLimit
                                    break
                    for j in range(lowLimit, upLimit):
                        if i<=0:
                            if connColor != None:
                                if matrix[-i+j,j] != pixelColor and matrix[-i+j,j] not in fixedColors:
                                    m[-i+j,j] = connColor
                            else:
                                if matrix[-i+j,j] in allowedChanges.keys():
                                    m[-i+j,j] = allowedChanges[matrix[-i+j,j]]
                        else:
                            if connColor != None:
                                if matrix[j,i+j] != pixelColor and matrix[j,i+j] not in fixedColors:
                                    m[j,i+j] = connColor
                            else:
                                if matrix[j,i+j] in allowedChanges.keys():
                                    m[j,i+j] = allowedChanges[matrix[j,i+j]]
            if d==2:
                matrix = np.fliplr(matrix)
                m = np.fliplr(m)

    return m

def connectAnyPixels(matrix, pixelColor=None, connColor=None, fixedColors=set(),\
                     allowedChanges={}, lineExclusive=False, diagonal=False):
    """
    Given a Matrix, this function draws a line connecting two pixels of the
    same color (different from the background color). The color of the line
    is the same as the color of the pixels, unless specified by "connColor".
    """
    
    m = matrix.m.copy()
    if pixelColor==None:
        if connColor==None:
            for c in matrix.colors - set([matrix.backgroundColor]):
                m = connectPixels(m, c, c, lineExclusive=lineExclusive, diagonal=diagonal)
            return m
        else:
            for c in matrix.colors - set([matrix.backgroundColor]):
                m = connectPixels(m, c, connColor, lineExclusive=lineExclusive, diagonal=diagonal)
            return m
    else:
        if len(allowedChanges)>0:
            m = connectPixels(m, pixelColor, allowedChanges=allowedChanges,\
                              lineExclusive=lineExclusive, diagonal=diagonal)
        else:
            m = connectPixels(m, pixelColor, connColor, fixedColors, lineExclusive=lineExclusive,\
                              diagonal=diagonal)
    return m

def rotate(matrix, angle):
    """
    Angle can be 90, 180, 270
    """
    assert angle in [90, 180, 270], "Invalid rotation angle"
    if isinstance(matrix, np.ndarray):
        m = matrix.copy()
    else:
        m = matrix.m.copy()
    return np.rot90(m, int(angle/90))    
    
def mirror(matrix, axis):
    """
    Axis can be lr, up, d1, d2
    """
    if isinstance(matrix, np.ndarray):
        m = matrix.copy()
    else:
        m = matrix.m.copy()
    assert axis in ["lr", "ud", "d1", "d2"], "Invalid mirror axis"
    if axis == "lr":
        return np.fliplr(m)
    if axis == "ud":
        return np.flipud(m)
    if axis == "d1":
        return m.T
    if axis == "d2":
        return m[::-1,::-1].T

def flipShape(matrix, shape, axis, background):
    # Axis can be lr, ud
    m = matrix.copy()
    smallM = np.ones((shape.shape[0], shape.shape[1]), dtype=np.uint8) * background
    for c in shape.pixels:
        smallM[c] = shape.color
    if axis == "lr":
        smallM = np.fliplr(smallM)
    if axis == "ud":
        smallM = np.flipud(smallM)
    for i,j in np.ndindex(smallM.shape):
        m[shape.position[0]+i, shape.position[1]+j] = smallM[i,j]
    return m

def flipAllShapes(matrix, axis, color, background, byColor=False, diagonal=False):#, multicolor=False):
    m = matrix.m.copy()
    if byColor:
        shapesToMirror = [s for s in matrix.shapesByColor if s.color in color]
    #elif multicolor:
    #    if diagonal:
    #        shapesToMirror = [s for s in matrix.multicolorDShapes]
    #    else:
    #        shapesToMirror = [s for s in matrix.multicolorShapes]
    else:
        if diagonal:
            shapesToMirror = [s for s in matrix.dShapes if s.color in color]
        else:
            shapesToMirror = [s for s in matrix.shapes if s.color in color]
    for s in shapesToMirror:
        m = flipShape(m, s, axis, background)
    return m

def getBestFlipAllShapes(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "flipAllShapes" and works best for the training samples.
    """
    bestFunction = partial(identityM)
    if t.backgroundColor == -1:
        return bestFunction
    bestScore = 1000
    colors = set([0,1,2,3,4,5,6,7,8,9]) - set([t.backgroundColor])
    for d in ["lr", "ud"]:
        #for multicolor in [True, False]:
        for diagonal in [True, False]:
            bestFunction, bestScore = updateBestFunction(t, partial(flipAllShapes, axis=d, color=colors,\
                                    background=t.backgroundColor, diagonal=diagonal), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(flipAllShapes, axis=d, color=colors,\
                                background=t.backgroundColor, byColor=True), bestScore, bestFunction)
    return bestFunction

def mapPixels(matrix, pixelMap, outShape):
    """
    Given a Matrix as input, this function maps each pixel of that matrix
    to an outputMatrix, given by outShape.
    The dictionary pixelMap determines which pixel in the input matrix maps
    to each pixel in the output matrix.
    """
    inMatrix = matrix.m.copy()
    m = np.zeros(outShape, dtype=np.uint8)
    for i,j in np.ndindex(outShape):
        m[i,j] = inMatrix[pixelMap[i,j]]
    return m

def switchColors(matrix, color1=None, color2=None):
    """
    This function switches the color1 and the color2 in the matrix.
    If color1 and color2 are not specified, then the matrix is expected to only
    have 2 colors, and they will be switched.
    """
    if type(matrix) == np.ndarray:
        m = matrix.copy()
    else:
        m = matrix.m.copy()
    if color1==None or color2==None:
        color1 = m[0,0]
        for i,j in np.ndindex(m.shape):
            if m[i,j]!=color1:
                color2 = m[i,j]
                break
    for i,j in np.ndindex(m.shape):
        if m[i,j]==color1:
            m[i,j] = color2
        else:
            m[i,j] = color1        
    return m

# %% Follow row/col patterns
def identifyColor(m, pixelPos, c2c, rowStep=None, colStep=None):
    """
    Utility function for followPattern.
    """
    if colStep!=None and rowStep!=None:
        i = 0
        while i+pixelPos[0] < m.shape[0]:
            j = 0
            while j+pixelPos[1] < m.shape[1]:
                if m[pixelPos[0]+i, pixelPos[1]+j] != c2c:
                    return m[pixelPos[0]+i, pixelPos[1]+j]
                j += colStep
            i += rowStep
        return c2c
    
def identifyColStep(m, c2c):
    """
    Utility function for followPattern.
    """
    colStep = 1
    while colStep < int(m.shape[1]/2)+1:
        isGood = True
        for j in range(colStep):
            for i in range(m.shape[0]):
                block = 0
                colors = set()
                while j+block < m.shape[1]:
                    colors.add(m[i,j+block])
                    block += colStep
                if c2c in colors:
                    if len(colors) > 2:
                        isGood = False
                        break
                else:
                    if len(colors) > 1:
                        isGood = False
                        break
            if not isGood:
                break  
        if isGood:
            return colStep 
        colStep+=1 
    return m.shape[1]

def identifyRowStep(m, c2c):
    """
    Utility function for followPattern.
    """
    rowStep = 1
    while rowStep < int(m.shape[0]/2)+1:
        isGood = True
        for i in range(rowStep):
            for j in range(m.shape[1]):
                block = 0
                colors = set()
                while i+block < m.shape[0]:
                    colors.add(m[i+block,j])
                    block += rowStep
                if c2c in colors:
                    if len(colors) > 2:
                        isGood = False
                        break
                else:
                    if len(colors) > 1:
                        isGood = False
                        break
            if not isGood:
                break  
        if isGood:
            return rowStep 
        rowStep+=1  
    return m.shape[0]            

def followPattern(matrix, rc, colorToChange=None, rowStep=None, colStep=None):
    """
    Given a Matrix, this function turns it into a matrix that follows a
    pattern. This will be made row-wise, column-wise or both, depending on the
    parameter "rc". "rc" can be "row", "column" or "both".
    'colorToChange' is the number corresponding to the only color that changes,
    if any.
    'rowStep' and 'colStep' are only to be given if the rowStep/colStep is the
    same for every train sample.
    """  
    m = matrix.m.copy()
            
    if colorToChange!=None:
        if rc=="col":
            rowStep=m.shape[0]
            if colStep==None:
                colStep=identifyColStep(m, colorToChange)
        if rc=="row":
            colStep=m.shape[1]
            if rowStep==None:
                rowStep=identifyRowStep(m, colorToChange)
        if rc=="both":
            if colStep==None and rowStep==None:
                colStep=identifyColStep(m, colorToChange)
                rowStep=identifyRowStep(m, colorToChange) 
            elif rowStep==None:
                rowStep=m.shape[0]
            elif colStep==None:
                colStep=m.shape[1]                       
        for i,j in np.ndindex((rowStep, colStep)):
            color = identifyColor(m, (i,j), colorToChange, rowStep, colStep)
            k = 0
            while i+k < m.shape[0]:
                l = 0
                while j+l < m.shape[1]:
                    m[i+k, j+l] = color
                    l += colStep
                k += rowStep
            
    return m

# %% Fill the blank
def fillTheBlankParameters(t):
    matrices = []
    for s in t.trainSamples:
        m = s.inMatrix.m.copy()
        blank = s.blankToFill
        m[blank.position[0]:blank.position[0]+blank.shape[0],\
          blank.position[1]:blank.position[1]+blank.shape[1]] = s.outMatrix.m.copy()
        matrices.append(Matrix(m))
        
    x = []
    x.append(all([m.lrSymmetric for m in matrices]))
    x.append(all([m.udSymmetric for m in matrices]))
    x.append(all([m.d1Symmetric for m in matrices]))
    x.append(all([m.d2Symmetric for m in matrices]))
    return x

def fillTheBlank(matrix, params):
    m = matrix.m.copy()
    if len(matrix.blanks) == 0:
        return m
    blank = matrix.blanks[0]
    color = blank.color
    pred = np.zeros(blank.shape, dtype=np.uint8)
    
    # lr
    if params[0]:
        for i,j in np.ndindex(blank.shape):
            if m[blank.position[0]+i, m.shape[1]-1-(blank.position[1]+j)] != color:
                pred[i,j] = m[blank.position[0]+i, m.shape[1]-1-(blank.position[1]+j)]
    # ud
    if params[1]:
        for i,j in np.ndindex(blank.shape):
            if m[m.shape[0]-1-(blank.position[0]+i), blank.position[1]+j] != color:
                pred[i,j] = m[m.shape[0]-1-(blank.position[0]+i), blank.position[1]+j]
    # d1
    if params[2] and m.shape[0]==m.shape[1]:
        for i,j in np.ndindex(blank.shape):
            if m[blank.position[1]+j, blank.position[0]+i] != color:
                pred[i,j] = m[blank.position[1]+j, blank.position[0]+i]
    # d2 (persymmetric matrix)
    if params[3] and m.shape[0]==m.shape[1]:
        for i,j in np.ndindex(blank.shape):
            if m[m.shape[1]-1-(blank.position[1]+j), m.shape[0]-1-(blank.position[0]+i)] != color:
                pred[i,j] = m[m.shape[1]-1-(blank.position[1]+j), m.shape[0]-1-(blank.position[0]+i)]
    
    return pred
    
# %% Operations with more than one matrix

# All the matrices need to have the same shape

def pixelwiseAnd(matrices, falseColor, targetColor=None, trueColor=None):
    """
    This function returns the result of executing the pixelwise "and" operation
    in a list of matrices.
    
    Parameters
    ----------
    matrices: list
        A list of numpy.ndarrays of the same shape
    falseColor: int
        The color of the pixel in the output matrix if the "and" operation is
        false.
    targetColor: int
        The color to be targeted by the "and" operation. For example, if
        targetColor is red, then the "and" operation will be true for a pixel
        if all that pixel is red in all of the input matrices.
        If targetColor is None, then the "and" operation will return true if
        the pixel has the same color in all the matrices, and false otherwise.
    trueColor: int
        The color of the pixel in the output matrix if the "and" operation is
        true.
        If trueColor is none, the output color if the "and" operation is true
        will be the color of the evaluated pixel.
    """
    m = np.zeros(matrices[0].shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        if targetColor == None:
            if all([x[i,j] == matrices[0][i,j] for x in matrices]):
                if trueColor == None:
                    m[i,j] = matrices[0][i,j]
                else:
                    m[i,j] = trueColor
            else:
                m[i,j] = falseColor
        else:
            if all([x[i,j] == targetColor for x in matrices]):
                if trueColor == None:
                    m[i,j] = matrices[0][i,j]
                else:
                    m[i,j] = trueColor
            else:
                m[i,j] = falseColor
    return m

"""
def pixelwiseOr(matrices, falseColor, targetColor=None, trueColor=None, \
                trueValues=None):
    See pixelwiseAnd.
    trueValues is a list with as many elements as matrices.
    m = np.zeros(matrices[0].shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        if targetColor == None:
            isFalse = True
            for x in matrices:
                if x[i,j] != falseColor:
                    isFalse = False
                    if trueColor == None:
                        m[i,j] = x[i,j]
                    else:
                        m[i,j] = trueColor
                    break
            if isFalse:
                m[i,j] = falseColor
        else:
            if any([x[i,j] == targetColor for x in matrices]):
                if trueColor == None:
                    m[i,j] = targetColor
                else:
                    m[i,j] = trueColor
            else:
                m[i,j] = falseColor
    return m
"""

def pixelwiseOr(matrices, falseColor, targetColor=None, trueColor=None, \
                trueValues=None):
    """
    See pixelwiseAnd.
    trueValues is a list with as many elements as matrices.
    """
    m = np.zeros(matrices[0].shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        if targetColor == None:
            trueCount = 0
            index = 0
            for x in matrices:
                if x[i,j] != falseColor:
                    trueCount += 1
                    trueIndex = index
                index += 1
            if trueCount==0:
                m[i,j] = falseColor
            else:
                if trueColor!=None:
                    m[i,j] = trueColor
                elif trueValues!=None:
                    if trueCount==1:
                        m[i,j] = trueValues[trueIndex]
                    else:
                        m[i,j] = matrices[trueIndex][i,j]
                else:
                    m[i,j] = matrices[trueIndex][i,j]
        else:
            if any([x[i,j] == targetColor for x in matrices]):
                if trueColor == None:
                    m[i,j] = targetColor
                else:
                    m[i,j] = trueColor
            else:
                m[i,j] = falseColor
    return m

def pixelwiseXor(m1, m2, falseColor, targetColor=None, trueColor=None, \
                 firstTrue=None, secondTrue=None):
    """
    See pixelwiseAnd. The difference is that the Xor operation only makes sense
    with two input matrices.
    """
    m = np.zeros(m1.shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        if targetColor == None:
            if (m1[i,j] == falseColor) != (m2[i,j] == falseColor):
                if trueColor == None:
                    if firstTrue == None:
                        if m1[i,j] != falseColor:
                            m[i,j] = m1[i,j]
                        else:
                            m[i,j] = m2[i,j]
                    else:
                        if m1[i,j] != falseColor:
                            m[i,j] = firstTrue
                        else:
                            m[i,j] = secondTrue
                else:
                    m[i,j] = trueColor     
            else:
                m[i,j] = falseColor
        else:
            if (m1[i,j] == targetColor) != (m2[i,j] == targetColor):
                if trueColor == None:
                    if firstTrue == None:
                        if m1[i,j] != falseColor:
                            m[i,j] = m1[i,j]
                        else:
                            m[i,j] = m2[i,j]
                    else:
                        if m1[i,j] != falseColor:
                            m[i,j] = firstTrue
                        else:
                            m[i,j] = secondTrue
                else:
                    m[i,j] = trueColor     
            else:
                m[i,j] = falseColor
    return m

# %% Downsize and Minimize
    
def getDownsizeFactors(matrix):
    """
    Still unused
    """
    xDivisors = set()
    for x in range(1, matrix.shape[0]):
        if (matrix.shape[0]%x)==0:
            xDivisors.add(x)
    yDivisors = set()
    for y in range(1, matrix.shape[1]):
        if (matrix.shape[1]%y)==0:
            yDivisors.add(y)
    
    downsizeFactors = set()
    for x,y in product(xDivisors, yDivisors):
        downsizeFactors.add((x,y))
 
    return downsizeFactors

def downsize(matrix, newShape, falseColor=None):
    """
    Given a matrix and a shape, this function returns a new matrix with the
    given shape. The elements of the return matrix are given by the colors of 
    each of the submatrices. Each submatrix is only allowed to have the
    background color and at most another one (that will define the output
    color of the corresponding pixel).
    """
    if falseColor==None:
        falseColor = matrix.backgroundColor
    if (matrix.shape[0]%newShape[0])!=0 or (matrix.shape[1]%newShape[1])!=0:
        return matrix.m.copy()
    xBlock = int(matrix.shape[0]/newShape[0])
    yBlock = int(matrix.shape[1]/newShape[1])
    m = np.full(newShape, matrix.backgroundColor, dtype=np.uint8)
    for i,j in np.ndindex(newShape[0], newShape[1]):
        color = -1
        for x,y in np.ndindex(xBlock, yBlock):
            if matrix.m[i*xBlock+x, j*yBlock+y] not in [matrix.backgroundColor, color]:
                if color==-1:
                    color = matrix.m[i*xBlock+x, j*yBlock+y]
                else:
                    return matrix.m.copy()
        if color==-1:
            m[i,j] = falseColor
        else:
            m[i,j] = color
    return m

def minimize(matrix):
    """
    Given a matrix, this function returns the matrix resulting from the
    following operations:
        If two consecutive rows are equal, delete one of them
        If two consecutive columns are equal, delete one of them
    """
    m = matrix.m.copy()
    x = 1
    for i in range(1, matrix.shape[0]):
        if np.array_equal(m[x,:],m[x-1,:]):
            m = np.delete(m, (x), axis=0)
        else:
            x+=1
    x = 1
    for i in range(1, matrix.shape[1]):
        if np.array_equal(m[:,x],m[:,x-1]):
            m = np.delete(m, (x), axis=1)
        else:
            x+=1
    return m
            
        

# %% Operations to extend matrices
    
def extendMatrix(matrix, color, position="tl", xShape=None, yShape=None, isSquare=False, goodDimension=None):
    """
    Given a matrix, xShape(>matrix.shape[0]), yShape(>matrix.shape[1]) and a color,
    this function extends the matrix using the dimensions given by xShape and
    yShape by coloring the extra pixels with the given color.
    If xShape or yShape are not given, then they will be equal to matrix.shape[0]
    of matrix.shape[1], respectively.
    The position of the input matrix in the output matrix can be given by
    specifying "tl", "tr", "bl" or "br" (top-left/top-right/bot-left/bot-right).
    The default is top-left.
    """
    if isSquare:
        if goodDimension=='x':
            xShape = matrix.shape[0]
            yShape = matrix.shape[0]
        if goodDimension=='y':
            xShape = matrix.shape[1]
            yShape = matrix.shape[1]
    if xShape==None:
        xShape = matrix.shape[0]
    if yShape==None:
        yShape = matrix.shape[1]
    m = np.full((xShape, yShape), color, dtype=np.uint8)
    if position=="tl":
        m[0:matrix.shape[0], 0:matrix.shape[1]] = matrix.m.copy()
    elif position=="tr":
        m[0:matrix.shape[0], yShape-matrix.shape[1]:yShape] = matrix.m.copy()
    elif position=="bl":
        m[xShape-matrix.shape[0]:xShape, 0:matrix.shape[1]] = matrix.m.copy()
    else:
        m[xShape-matrix.shape[0]:xShape, yShape-matrix.shape[1]:yShape] = matrix.m.copy()
    
    return m

def getBestExtendMatrix(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "extendMatrix" and works best for the training samples.
    """
    if t.backgroundColor==-1:
        totalColorCount = Counter()
        for s in t.trainSamples:
            totalColorCount += s.inMatrix.colorCount
        background = max(totalColorCount.items(), key=operator.itemgetter(1))[0]
    else:
        background=t.backgroundColor
    bestScore = 1000
    bestFunction = partial(identityM)
    
    # Define xShape and yShape:
    xShape = None
    yShape = None
    isSquare=False
    goodDimension=None
    
    # If the outShape is given, easy
    if t.sameOutShape:
        xShape=t.outShape[0]
        yShape=t.outShape[1]
    # If the output xShape is always the same and the yShape keeps constant, pass the common xShape
    elif len(set([s.outMatrix.shape[0] for s in t.trainSamples]))==1:
        if all([s.outMatrix.shape[1]==s.inMatrix.shape[1] for s in t.trainSamples]):
            xShape=t.trainSamples[0].outMatrix.shape[0]  
    # If the output yShape is always the same and the xShape keeps constant, pass the common yShape
    elif len(set([s.outMatrix.shape[1] for s in t.trainSamples]))==1:
        if all([s.outMatrix.shape[0]==s.inMatrix.shape[0] for s in t.trainSamples]):
            yShape=t.trainSamples[0].outMatrix.shape[1] 
    # If the matrix is always squared, and one dimension (x or y) is fixed, do this:
    elif all([s.outMatrix.shape[0]==s.outMatrix.shape[1] for s in t.trainSamples]):
        isSquare=True
        if all([s.outMatrix.shape[1]==s.inMatrix.shape[1] for s in t.trainSamples]):
            goodDimension='y'     
        elif all([s.outMatrix.shape[0]==s.inMatrix.shape[0] for s in t.trainSamples]):
            goodDimension='x'      
    
    for position in ["tl", "tr", "bl", "br"]:
        f = partial(extendMatrix, color=background, xShape=xShape, yShape=yShape,\
                    position=position, isSquare=isSquare, goodDimension=goodDimension)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    return bestFunction
    
def getFactor(matrix, factor):
    """
    Given a Task.inShapeFactor (that can be a string), this function
    returns its corresponding tuple for the given matrix.
    """
    if factor == "squared":
        f = (matrix.shape[0], matrix.shape[1])
    elif factor == "xSquared":
        f = (matrix.shape[0], 1)
    elif factor == "ySquared":
        f = (1, matrix.shape[1])
    elif factor == "nColors":
        f = (matrix.nColors, matrix.nColors)
    elif factor == "nColors-1":
        f = (matrix.nColors-1, matrix.nColors-1)
    else:
        f = factor
    return f

def multiplyPixels(matrix, factor):
    """
    Factor is a 2-dimensional tuple.
    The output matrix has shape matrix.shape*factor. Each pixel of the input
    matrix is expanded by factor.
    """
    factor = getFactor(matrix, factor)
    m = np.zeros(tuple(s * f for s, f in zip(matrix.shape, factor)), dtype=np.uint8)
    for i,j in np.ndindex(matrix.m.shape):
        for k,l in np.ndindex(factor):
            m[i*factor[0]+k, j*factor[1]+l] = matrix.m[i,j]
    return m

def multiplyMatrix(matrix, factor):
    """
    Copy the matrix "matrix" into every submatrix of the output, which has
    shape matrix.shape * factor.
    """
    factor = getFactor(matrix, factor)
    m = np.zeros(tuple(s * f for s, f in zip(matrix.shape, factor)), dtype=np.uint8)
    for i,j in np.ndindex(factor):
        m[i*matrix.shape[0]:(i+1)*matrix.shape[0], j*matrix.shape[1]:(j+1)*matrix.shape[1]] = matrix.m.copy()
    return m

def matrixTopLeft(matrix, factor, background=0):
    """
    Copy the matrix into the top left corner of the multiplied matrix
    """
    factor = getFactor(matrix, factor)
    m = np.full(tuple(s * f for s, f in zip(matrix.shape, factor)), background, dtype=np.uint8)
    m[0:matrix.shape[0], 0:matrix.shape[1]] = matrix.m.copy()
    return m
    
def matrixBotRight(matrix, factor, background=0):
    """
    Copy the matrix into the bottom right corner of the multiplied matrix
    """
    factor = getFactor(matrix, factor)
    m = np.full(tuple(s * f for s, f in zip(matrix.shape, factor)), background, dtype=np.uint8)
    m[(factor[0]-1)*matrix.shape[0]:factor[0]*matrix.shape[0], \
      (factor[1]-1)*matrix.shape[1]:factor[1]*matrix.shape[1]]
    return m

def getBestMosaic(t):
    """
    Given a task t, this function tries to find the best way to generate a
    mosaic, given that the output shape is always bigger than the input shape
    with a shape factor that makes sense.
    A mosaic is a matrix that takes an input matrix as reference, and then
    copies it many times. The copies can include rotations or mirrorings.
    """
    factor = t.inShapeFactor
    ops = []
    ops.append(partial(identityM))
    ops.append(partial(mirror, axis="lr"))
    ops.append(partial(mirror, axis="ud"))
    ops.append(partial(rotate, angle=180))
    if t.inMatricesSquared:
        ops.append(partial(mirror, axis="d1"))
        ops.append(partial(mirror, axis="d2"))
        ops.append(partial(rotate, angle=90))
        ops.append(partial(rotate, angle=270))
    bestOps = []
    for i in range(factor[0]):
        bestOps.append([])
        for j in range(factor[1]):
            bestScore = 1000
            bestOp = partial(identityM)
            for op in ops:
                score = 0
                for s in t.trainSamples:
                    inM = s.inMatrix.m.copy()
                    outM = s.outMatrix.m[i*inM.shape[0]:(i+1)*inM.shape[0], j*inM.shape[1]:(j+1)*inM.shape[1]]
                    score += incorrectPixels(op(inM),outM)
                if score < bestScore:
                    bestScore = score
                    bestOp = op
                    if score==0:
                        break
            bestOps[i].append(bestOp)
    return bestOps

def generateMosaic(matrix, ops, factor):
    """
    Generates a mosaic from the given matrix using the operations given in the
    list ops. The output matrix has shape matrix.shape*factor.
    """
    m = np.zeros(tuple(s * f for s, f in zip(matrix.shape, factor)), dtype=np.uint8)
    for i in range(factor[0]):
        for j in range(factor[1]):
            m[i*matrix.shape[0]:(i+1)*matrix.shape[0], j*matrix.shape[1]:(j+1)*matrix.shape[1]] = \
            ops[i][j](matrix)
    return m

# Only if the factor is squared
def getBestMultiplyMatrix(t, falseColor): 
    """
    Given a Task t, this function returns the partial function that uses the
    function "multiplyMatrix" and works best for the training samples.
    """
    def getFullMatrix(matrix, color):
        return np.full(matrix.shape, color, dtype=np.uint8)
    # Possible operations on the matrix
    ops = []
    ops.append(partial(identityM))
    ops.append(partial(mirror, axis="lr"))
    ops.append(partial(mirror, axis="ud"))
    ops.append(partial(rotate, angle=180))
    if t.inMatricesSquared:
        ops.append(partial(mirror, axis="d1"))
        ops.append(partial(mirror, axis="d2"))
        ops.append(partial(rotate, angle=90))
        ops.append(partial(rotate, angle=270))
    if all([n==2 for n in t.nInColors]):
        ops.append(partial(switchColors))
    
    # Conditions
    def trueCondition(matrix, pixel):
        return True
    def maxColor(matrix, pixel):
        x = [k for k, v in sorted(matrix.colorCount.items(), key=lambda item: item[1], reverse=True)]
        if len(x)<2 or matrix.colorCount[x[0]]!=matrix.colorCount[x[1]]:
            return pixel==max(matrix.colorCount, key=matrix.colorCount.get)
        else:
            return False
    def minColor(matrix,pixel):
        x = [k for k, v in sorted(matrix.colorCount.items(), key=lambda item: item[1])]
        if len(x)<2 or matrix.colorCount[x[-1]]!=matrix.colorCount[x[-2]]:
            return pixel==min(matrix.colorCount, key=matrix.colorCount.get)
        else:
            return False
    def isColor(matrix, pixel, color):
        return pixel==color
    def nonZero(matrix, pixel):
        return pixel!=0
    def zero(matrix, pixel):
        return pixel==0
    conditions = []
    conditions.append(partial(trueCondition))
    conditions.append(partial(maxColor))
    conditions.append(partial(minColor))
    conditions.append(partial(nonZero))
    conditions.append(partial(zero))
    for c in t.colors:
        conditions.append(partial(isColor, color=c))

    bestScore = 1000
    for op, cond in product(ops, conditions):
        score = 0
        for s in t.trainSamples:
            factor = getFactor(s.inMatrix, t.inShapeFactor)
            for i,j in np.ndindex(factor):
                inM = s.inMatrix.m.copy()
                outM = s.outMatrix.m[i*inM.shape[0]:(i+1)*inM.shape[0], j*inM.shape[1]:(j+1)*inM.shape[1]]
                if cond(s.inMatrix, inM[i,j]):
                    score += incorrectPixels(op(inM),outM)
                else:
                    score += incorrectPixels(getFullMatrix(inM, falseColor), outM)
        if score < bestScore:
            bestScore = score
            opCond = (op, cond)
            if score==0:
                return opCond
    return opCond

def doBestMultiplyMatrix(matrix, opCond, falseColor):
    factor = matrix.shape
    m = np.full(tuple(s * f for s, f in zip(matrix.shape, factor)), falseColor, dtype=np.uint8)
    for i,j in np.ndindex(factor):
        if opCond[1](matrix, matrix.m[i,j]):
            m[i*matrix.shape[0]:(i+1)*matrix.shape[0], j*matrix.shape[1]:(j+1)*matrix.shape[1]] = \
            opCond[0](matrix)
    return m

# %% Multiply pixels

def multiplyPixelsAndAnd(matrix, factor, falseColor):
    """
    This function basically is the same as executing the functions
    multiplyPixels, multiplyMatrix, and executing pixelwiseAnd with these two
    matrices as inputs
    """
    factor = getFactor(matrix, factor)
    m = matrix.m.copy()
    multipliedM = multiplyPixels(matrix, factor)
    for i,j in np.ndindex(factor):
        newM = multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]]
        multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]] = pixelwiseAnd([m, newM], falseColor)
    return multipliedM

def multiplyPixelsAndOr(matrix, factor, falseColor):
    """
    This function basically is the same as executing the functions
    multiplyPixels, multiplyMatrix, and executing pixelwiseOr with these two
    matrices as inputs
    """
    factor = getFactor(matrix, factor)
    m = matrix.m.copy()
    multipliedM = multiplyPixels(matrix, factor)
    for i,j in np.ndindex(factor):
        newM = multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]]
        multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]] = pixelwiseOr([m, newM], falseColor)
    return multipliedM

def multiplyPixelsAndXor(matrix, factor, falseColor):
    """
    This function basically is the same as executing the functions
    multiplyPixels, multiplyMatrix, and executing pixelwiseXor with these two
    matrices as inputs
    """
    factor = getFactor(matrix, factor)
    m = matrix.m.copy()
    multipliedM = multiplyPixels(matrix, factor)
    for i,j in np.ndindex(factor):
        newM = multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]]
        multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]] = pixelwiseXor(m, newM, falseColor)
    return multipliedM

# %% Operations considering all submatrices of task with outShapeFactor
    
def getSubmatrices(m, factor):
    """
    Given a matrix m and a factor, this function returns a list of all the
    submatrices with shape determined by the factor.
    """
    matrices = []
    nRows = int(m.shape[0] / factor[0])
    nCols = int(m.shape[1] / factor[1])
    for i,j in np.ndindex(factor):
        matrices.append(m[i*nRows:(i+1)*nRows, j*nCols:(j+1)*nCols])
    return matrices

def outputIsSubmatrix(t, isGrid=False):
    """
    Given a task t that has outShapeFactor, this function returns true if any
    of the submatrices is equal to the output matrix for every sample.
    """
    for sample in t.trainSamples:
        if isGrid:
            matrices = [c[0].m for c in sample.inMatrix.grid.cellList]
        else:
            matrices = getSubmatrices(sample.inMatrix.m, sample.outShapeFactor)
        anyIsSubmatrix = False
        for m in matrices:
            if np.array_equal(m, sample.outMatrix.m):
                anyIsSubmatrix = True
                break
        if not anyIsSubmatrix:
            return False
    return True

def selectSubmatrixWithMaxColor(matrix, color, outShapeFactor=None, isGrid=False):
    """
    Given a matrix, this function returns the submatrix with most appearances
    of the color given. If the matrix is not a grid, an outShapeFactor must be
    specified.
    """
    if isGrid:
        matrices = [c[0].m for c in matrix.grid.cellList]
    else:
        matrices = getSubmatrices(matrix.m, outShapeFactor)
        
    maxCount = 0
    matricesWithProperty = 0
    bestMatrix = None
    for mat in matrices:
        m = Matrix(mat)
        if color in m.colors:
            if m.colorCount[color]>maxCount:
                bestMatrix = mat.copy()
                maxCount = m.colorCount[color]
                matricesWithProperty = 1
            if m.colorCount[color]==maxCount:
                matricesWithProperty += 1
    if matricesWithProperty!=1:
        return matrix.m.copy()
    else:
        return bestMatrix
    
def selectSubmatrixWithMinColor(matrix, color, outShapeFactor=None, isGrid=False):
    """
    Given a matrix, this function returns the submatrix with least appearances
    of the color given. If the matrix is not a grid, an outShapeFactor must be
    specified.
    """
    if isGrid:
        matrices = [c[0].m for c in matrix.grid.cellList]
    else:
        matrices = getSubmatrices(matrix.m, outShapeFactor)
        
    minCount = 1000
    matricesWithProperty = 0
    bestMatrix = None
    for mat in matrices:
        m = Matrix(mat)
        if color in m.colors:
            if m.colorCount[color]<minCount:
                bestMatrix = mat.copy()
                minCount = m.colorCount[color]
                matricesWithProperty = 1
            elif m.colorCount[color]==minCount:
                matricesWithProperty += 1
    if matricesWithProperty!=1:
        return matrix.m.copy()
    else:
        return bestMatrix
    
def selectSubmatrixWithMostColors(matrix, outShapeFactor=None, isGrid=False):
    """
    Given a matrix, this function returns the submatrix with the most number of
    colors. If the matrix is not a grid, an outShapeFactor must be specified.
    """
    if isGrid:
        matrices = [c[0].m for c in matrix.grid.cellList]
    else:
        matrices = getSubmatrices(matrix.m, outShapeFactor)
        
    maxNColors = 0
    matricesWithProperty = 0
    bestMatrix = None
    for mat in matrices:
        m = Matrix(mat)
        if len(m.colorCount)>maxNColors:
            bestMatrix = mat.copy()
            maxNColors = len(m.colorCount)
            matricesWithProperty = 1
        elif len(m.colorCount)==maxNColors:
            matricesWithProperty += 1
    if matricesWithProperty!=1:
        return matrix.m.copy()
    else:
        return bestMatrix
    
def selectSubmatrixWithLeastColors(matrix, outShapeFactor=None, isGrid=False):
    """
    Given a matrix, this function returns the submatrix with the least number
    of colors. If the matrix is not a grid, an outShapeFactor must be
    specified.
    """
    if isGrid:
        matrices = [c[0].m for c in matrix.grid.cellList]
    else:
        matrices = getSubmatrices(matrix.m, outShapeFactor)
        
    minNColors = 1000
    matricesWithProperty = 0
    bestMatrix = None
    for mat in matrices:
        m = Matrix(mat)
        if len(m.colorCount)<minNColors:
            bestMatrix = mat.copy()
            minNColors = len(m.colorCount)
            matricesWithProperty = 1
        elif len(m.colorCount)==minNColors:
            matricesWithProperty += 1
    if matricesWithProperty!=1:
        return matrix.m.copy()
    else:
        return bestMatrix
        
def getBestSubmatrixPosition(t, outShapeFactor=None, isGrid=False):
    """
    Given a task t, and assuming that all the input matrices have the same
    shape and all the ouptut matrices have the same shape too, this function
    tries to check whether the output matrix is just the submatrix in a given
    position. If that's the case, it returns the position. Otherwise, it
    returns 0.
    """
    iteration = 0
    possiblePositions = []
    for sample in t.trainSamples:
        if isGrid:
            matrices = [c[0].m for c in sample.inMatrix.grid.cellList]
        else:
            matrices = getSubmatrices(sample.inMatrix.m, outShapeFactor)
            
        possiblePositions.append(set())
        for m in range(len(matrices)):
            if np.array_equal(matrices[m], sample.outMatrix.m):
                possiblePositions[iteration].add(m)
        
        iteration += 1
    positions = set.intersection(*possiblePositions)
    if len(positions)==1:
        return next(iter(positions))
    else:
        return 0
                
def selectSubmatrixInPosition(matrix, position, outShapeFactor=None, isGrid=False):
    """
    Given a matrix and a position, this function returns the submatrix that
    appears in the given position (submatrices are either defined by
    outShapeFactor or by the shape of the grid cells).
    """
    if isGrid:
        matrices = [c[0].m for c in matrix.grid.cellList]
    else:
        matrices = getSubmatrices(matrix.m, outShapeFactor)
        
    return matrices[position].copy()

def maxColorFromCell(matrix):
    """
    Only to be called if matrix.isGrid.
    Given a matrix with a grid, this function returns a matrix with the same
    shape as the grid. Every pixel of the matrix will be colored with the 
    color that appears the most in the corresponding cell of the grid.
    """
    m = np.zeros(matrix.grid.shape, dtype=np.uint8)
    for i,j  in np.ndindex(matrix.grid.shape):
        color = max(matrix.grid.cells[i][j][0].colorCount.items(), key=operator.itemgetter(1))[0]
        m[i,j] = color
    return m

def colorAppearingXTimes(matrix, times):
    m = np.zeros(matrix.grid.shape, dtype=np.uint8)
    for i,j in np.ndindex(matrix.grid.shape):
        for k,v in matrix.grid.cells[i][j][0].colorCount.items():
            if v==times:
                m[i,j] = k
    return m
        
def pixelwiseAndInSubmatrices(matrix, factor, falseColor, targetColor=None, trueColor=None):
    matrices = getSubmatrices(matrix.m.copy(), factor)
    return pixelwiseAnd(matrices, falseColor, targetColor, trueColor)

def pixelwiseOrInSubmatrices(matrix, factor, falseColor, targetColor=None, trueColor=None, \
                             trueValues=None):
    matrices = getSubmatrices(matrix.m.copy(), factor)
    return pixelwiseOr(matrices, falseColor, targetColor, trueColor, trueValues)

def pixelwiseXorInSubmatrices(matrix, factor, falseColor, targetColor=None, trueColor=None, \
                              firstTrue=None, secondTrue=None):
    matrices = getSubmatrices(matrix.m.copy(), factor)
    return pixelwiseXor(matrices[0], matrices[1], falseColor, targetColor, trueColor, firstTrue, secondTrue)

# %% Operations considering all submatrices of a grid

def pixelwiseAndInGridSubmatrices(matrix, falseColor, targetColor=None, trueColor=None):
    matrices = [c[0].m for c in matrix.grid.cellList]
    return pixelwiseAnd(matrices, falseColor, targetColor, trueColor)

def pixelwiseOrInGridSubmatrices(matrix, falseColor, targetColor=None, trueColor=None, \
                                 trueValues=None):
    matrices = [c[0].m for c in matrix.grid.cellList]
    return pixelwiseOr(matrices, falseColor, targetColor, trueColor, trueValues)

def pixelwiseXorInGridSubmatrices(matrix, falseColor, targetColor=None, trueColor=None, \
                                  firstTrue=None, secondTrue=None):
    m1 = matrix.grid.cellList[0][0].m.copy()
    m2 = matrix.grid.cellList[1][0].m.copy()
    return pixelwiseXor(m1, m2, falseColor, targetColor, trueColor, firstTrue, secondTrue)

# %% crop all shapes
    
def cropAllShapes(matrix, background, diagonal=False):
    if diagonal:
        shapes = [shape for shape in matrix.dShapes if shape.color!=background]
    else:
        shapes = [shape for shape in matrix.shapes if shape.color!=background]
    shapes = sorted(shapes, key=lambda x: x.nPixels, reverse=True)
    
    if len(shapes)==0:
        return matrix.m.copy()
    
    m = shapes[0].m.copy()
    for i,j in np.ndindex(m.shape):
        if m[i,j]==255:
            m[i,j] = background
    
    outMatrix = Matrix(m)
    if diagonal:
        outShapes = [shape for shape in outMatrix.dShapes if shape.color==background]
    else:
        outShapes = [shape for shape in outMatrix.shapes if shape.color==background]
    
    for s in outShapes:
        if s.color==background:
            for shape in shapes:
                if shape.hasSameShape(s):
                    m = changeColorShapes(m, [s], shape.color)
                    break
    return m

def getLayerDict(t):
    """
    Only to be called when there is a potential bijective dictionary between submatrices of the input and output. This
    function returns such dictionary. 
    """
    shList = []
    for s in t.trainSamples:
        shList += [sh.shape for sh in s.outMatrix.shapes]
    sc = Counter(shList)
    shape = sc.most_common(1)[0][0]
    layerDict = dict()
    for s in t.trainSamples:
        inM = s.inMatrix.m
        outM = s.outMatrix.m
        xR = inM.shape[0]//shape[0]
        yR = inM.shape[1]//shape[1]
        for x in range(xR):
            for y in range(yR):
                if inM[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]].tobytes() in layerDict:
                    if np.all(layerDict[inM[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]].tobytes()] !=\
                                                outM[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]]):
                        return None, dict()
                layerDict[inM[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]].tobytes()]\
                                    = outM[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]] 
    return shape, layerDict

def subMatToLayer(matrix, shapeAndDict):
    """
    Transforms each submatrix according to the given dictionary shapeAndDict. If the input submatrix is not 
    in the dictionary, returns the original matrix. 
    """
    shape = shapeAndDict[0]
    layerDict = shapeAndDict[1]
    m = matrix.m.copy()
    if shape is None:
        return m
    xR = m.shape[0]//shape[0]
    yR = m.shape[1]//shape[1]
    for x in range(xR):
        for y in range(yR):
            if m[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]].tobytes() in layerDict:
                m[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]] =\
                        layerDict[m[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]].tobytes()].copy()
            else:
                return m
    return m
    
def getBestFitToFrame(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "fitToFrame" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    crop = False
    if t.outSmallerThanIn:
        crop = True
    for s in [True, False]:
        bestFunction, bestScore = updateBestFunction(t, partial(fitToFrame, crop=crop, includeFrame=True, scale=s), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(fitToFrame, crop=crop, includeFrame=False, scale=s), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(fitToFrame, crop=crop, includeFrame=True, colorMatch=True), bestScore, bestFunction)
    return bestFunction

def fitToFrame(matrix, crop=False, scale=False, includeFrame=True, colorMatch=False):
    """
    To be called only if the task has a partial or full frame. Attempts to fit or adjust
    a shape or list of shapes inside the frame.
    """
    m = matrix.m.copy()
    if len(matrix.partialFrames) != 1:
        return m
    frame = matrix.partialFrames[0]
    found = False
    maux = Matrix(deleteShape(m, frame, matrix.backgroundColor))
    for sh in maux.multicolorDShapes:
        if sh != frame and sh.shape[0]>1 and sh.shape[1]>1:
            m = deleteShape(m, sh, matrix.backgroundColor)
            found = True
            break
    if not found or sh.shape[0] > frame.shape[0] or sh.shape[1] > frame.shape[1]:
        return m
    if scale:
        r = min((frame.shape[0]-2)//sh.shape[0], (frame.shape[1]-2)//sh.shape[1])
        newSh = copy.deepcopy(sh)
        newSh.m = np.repeat(np.repeat(sh.m, r, axis=1), r, axis=0)
        newSh.shape = newSh.m.shape
        newSh.pixels = set([(i,j) for i,j in np.ndindex(newSh.m.shape) if newSh.m[i,j]!=255])
    else:
        newSh = copy.deepcopy(sh)
    newSh.position = (frame.position[0] + (frame.shape[0]-newSh.shape[0])//2, frame.position[1] + (frame.shape[1]-newSh.shape[1])//2)
    if colorMatch:
        if len(set(sh.m[:,0]).intersection(matrix.m[frame.position[0]:frame.position[0]+\
               frame.shape[0],frame.position[1]])-set([255])) == 0:
            newSh.m = np.fliplr(newSh.m)
    m = insertShape(m, newSh)
    if crop:
        bC = matrix.backgroundColor
        if np.all(m == bC):
            return m
        x1, x2, y1, y2 = 0, m.shape[0]-1, 0, m.shape[1]-1
        while x1 <= x2 and np.all(m[x1,:] == bC):
            x1 += 1
        while x2 >= x1 and np.all(m[x2,:] == bC):
            x2 -= 1
        while y1 <= y2 and np.all(m[:,y1] == bC):
            y1 += 1
        while y2 >= y1 and np.all(m[:,y2] == bC):
            y2 -= 1
        if includeFrame:
            return(m[x1:x2+1,y1:y2+1])
        elif x1+1<x2 and y1+1<y2:
            return(m[x1+1:x2,y1+1:y2])
    return m
    
def getBestCountColors(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "countColors" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    outShape = [None]
    sliced = [False]
    if t.sameOutShape:
        outShape.append(t.trainSamples[0].outMatrix.shape)
    if t.outSmallerThanIn or not t.sameIOShapes:
        sliced += [True]
    if t.sameIOShapes:
        outShape=['inShape']
    for byShape in [True,False]:
        for outCol in [t.backgroundColor, t.trainSamples[0].outMatrix.backgroundColor, 0]:
            for rotate in range(-1,2):
                for flip in [True, False]:
                    for ignore in ['max', 'min', None]:
                        for sl in sliced:
                            for oSh in outShape:
                                bestFunction, bestScore = updateBestFunction(t, partial(countColors,\
                                            rotate=rotate,outBackgroundColor=outCol, flip=flip, sliced=sl,\
                                            ignore=ignore, outShape=oSh, byShape=byShape), bestScore, bestFunction)
                                bestFunction, bestScore = updateBestFunction(t, partial(countColors,\
                                            rotate=rotate,outBackgroundColor=outCol, flip=flip, sliced=sl,\
                                            ignore=ignore, outShape=oSh, byShape=byShape, sortByColor=1), bestScore, bestFunction)
    return bestFunction

def countColors(matrix, outBackgroundColor=-1, outShape=None,ignoreBackground=True,\
                ignore=False, sliced=False, rotate=0, flip=False, byShape=False, sortByColor=False):
    """
    Counts the colors of a given input matrix as pixels. The parameters flip, rotate and sliced determine the layout of the
    output pixels. The argument sortByColor sorts the colors by their number, otherwise they are sorted by occurrence. The 
    parameter ignore allows to ignore either the most or least common colors.
    """
    if byShape:
        cc = Counter([sh.color for sh in matrix.shapes if (sh.color != matrix.backgroundColor or not ignoreBackground)])
        cc = sorted(cc.items(), key=lambda x: x[1], reverse=True)
    else:
        cc = matrix.colorCount
        #add line to sort geographically
        cc = sorted(cc.items(), key=lambda x: x[1], reverse=True)
    if sortByColor:
        cc = sorted(cc, key=lambda x: x[0])
    if outBackgroundColor == -1:
        bC = matrix.backgroundColor
    else:
        bC = outBackgroundColor
    if ignoreBackground:
        cc = [c for c in cc if c[0] != bC]
    if ignore == 'max':
        cc = [cc[i] for i in range(1,len(cc))]
    elif ignore == 'min':
        cc = [c for c in cc if c[1] == max([c[1] for c in cc]) ]
    if len(cc) == 0:
        return matrix.m.copy()
    if outShape == None:
        m = np.full((max([c[1] for c in cc]),len(cc)), fill_value=bC)
        for j in range(len(cc)):
            for i in range(cc[j][1]):
                if i >= m.shape[0] or j >= m.shape[1]:
                    break
                m[i,j] = cc[j][0]
    else:
        if outShape == 'inShape':
            m = np.full(matrix.shape, fill_value=bC)
            if m.shape[0] > m.shape[1]:
                m = np.rot90(m)
            for j in range(len(cc)):
                for i in range(cc[j][1]):
                    if i >= m.shape[0] or j >= m.shape[1]:
                        break
                    m[i,j] = cc[j][0]
        else:
            m = np.full(outShape, fill_value=bC)
            cc = [c[0] for c in cc for j in range(c[1])]
            i = 0
            while i < m.shape[0]:
                j = 0
                while j < m.shape[1]:
                    if i*m.shape[1]+j >= len(cc):
                        break
                    m[i,j] = cc[i*m.shape[1]+j]
                    j += 1 
                i += 1 
    if sliced:
        m = [m[0,:]]
    m = np.rot90(m, rotate)
    if flip:
        m = np.flipud(m)
    if outShape == 'inShape' and m.shape !=  matrix.shape:
        m = np.rot90(m)
    return m

def getBestCountShapes(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "countShapes" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    if all([len(s.inMatrix.shapes)>15 for s in t.trainSamples]):
            return bestFunction
    if t.sameIOShapes:
        oSh = 'inShape'
    elif t.sameOutShape:
        oSh = t.trainSamples[0].outMatrix.shape
    else:
        oSh = None
        
    for outC in set([None]).union(set.intersection(*t.outColors)):
        for inC in set([-1]).union(set.intersection(*t.inColors)):
            for sh in [None] + t.commonInShapes:
                for lay in ['h','d']:
                    for skip in [True, False]:
                        bestFunction, bestScore = updateBestFunction(t, partial(countShapes,color=inC,\
                                        outShape=oSh, lay=lay, outColor=outC, shape=sh, skip=skip), bestScore, bestFunction)
    return bestFunction

def countShapes(matrix, color=-1, shape=None, outColor=None, outShape=None, lay='d', skip=False):
    """
    This function returns the count of shapes of a given color or shape, and lays that many pixels. 
    The pixels layout is specified by the arguments outShape, skip and lay. 
    """
    if color < 0:
        shc = [sh for sh in matrix.shapes]
    else:
        shc = [sh for sh in matrix.shapes if sh.color == color]
    if shape != None:
        shc = [sh for sh in shc if sh == shape]
    if outShape == None:
        m = np.full((len(shc),len(shc)), fill_value=matrix.backgroundColor)
    elif outShape == 'inShape':
        m = np.full(matrix.shape, fill_value=matrix.backgroundColor)
    else:
        m = np.full(outShape, fill_value=matrix.backgroundColor)
    if lay == 'd':
        for d in range(min(len(shc), m.shape[0], m.shape[1])):
            if outColor == None:
                m[d,d] = shc[d].color
            else:
                m[d,d] = outColor
    elif lay == 'h':
        shc = [sh.color for sh in shc]
        if skip:
            shc = [[c,matrix.backgroundColor] for c in shc]
            shc = [c for p in shc for c in p]
        i = 0
        while i < m.shape[0]:
            j = 0
            while j < m.shape[1]:
                if i*m.shape[1]+j >= len(shc):
                    break
                if outColor == None:
                    m[i,j] = shc[i*m.shape[1]+j]
                elif shc[i*m.shape[1] + j] != matrix.backgroundColor:
                    m[i,j] = outColor
                j += 1           
            i += 1 
    return m 

def getBestSymmetrizeAllShapes(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "symmetrizeAllShapes" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    for cc in set.intersection(*t.inColors).union(set([-1])):
        bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeAllShapes, targetColor = cc), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeAllShapes, targetColor = cc, byColor=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeAllShapes, targetColor = cc, context=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeAllShapes, targetColor = cc, context=True, byColor=True), bestScore, bestFunction)

    return bestFunction

def symmetrizeAllShapes(matrix, diagonal=True, multicolor=True, targetColor=-1,\
                        context=False, lr = True, ud = True, byColor=False):
    """
    Symmetrize all shapes of a given type with respect to specified axes lr or ud. If tagetColor is specified, 
    then only shapes of that color will be symmetrized. 
    """
    m = matrix.m.copy()
    bC = matrix.backgroundColor
    if byColor:
        shList = [sh for sh in matrix.shapesByColor if (sh.shape[0]<m.shape[0] and sh.shape[1]<m.shape[1])]
    else:    
        if not multicolor: 
            if diagonal:   
                shList = [sh for sh in matrix.dShapes]
            else:   
                shList = [sh for sh in matrix.shapes]
        else:
            if diagonal: 
                shList = [sh for sh in matrix.multicolorDShapes]
            else:
                shList = [sh for sh in matrix.multicolorShapes]
    if targetColor > -1:
        shList = [sh for sh in shList if hasattr(sh, 'color') and sh.color == targetColor]
    for sh in shList:
        if context:
            shM = m[sh.position[0]:sh.position[0]+sh.shape[0], sh.position[1]:sh.position[1]+sh.shape[1]]
        else:
            shM = sh.m.copy()
        if lr:
            shMlr = np.fliplr(shM)
            for i,j in np.ndindex(sh.shape):
                if shM[i,j] == bC or shM[i,j] == 255:
                    shM[i,j] = shMlr[i,j]
        if ud:
            shMud = np.flipud(shM)
            for i,j in np.ndindex(sh.shape):
                if shM[i,j] == bC or shM[i,j] == 255:
                    shM[i,j] = shMud[i,j]
        if context:
            m[sh.position[0]:sh.position[0]+sh.shape[0], sh.position[1]:sh.position[1]+sh.shape[1]] = shM
        else:
            newInsert = copy.deepcopy(sh)
            newInsert.m = shM
            m = insertShape(m, newInsert)
    return m

def paintGridLikeBackground(matrix):
    """
    Ignores the grid by paiting it in the background color (most repeated color or second if first coincides with grid color).
    In some cases cells have more than one color but it is still worth ignoring the grid. 
    """
    m = matrix.m.copy()
    bC = max(matrix.colorCount,key=matrix.colorCount.get)
    if matrix.isGrid:
        m[m==matrix.grid.color] = bC
    elif matrix.isAsymmetricGrid:
        m[m==matrix.asymmetricGrid.color] = bC
    return m  

def downsizeMode(matrix, newShape, falseColor=None):
    """
    Given a matrix and a shape, this function returns a new matrix with the
    given shape. The elements of the return matrix are given by the colors of 
    each of the submatrices. If a submatrix has more than one color the mode is
    chosen.
    """
    if (matrix.shape[0]%newShape[0])!=0 or (matrix.shape[1]%newShape[1])!=0:
        return matrix.m.copy()
    xBlock = int(matrix.shape[0]/newShape[0])
    yBlock = int(matrix.shape[1]/newShape[1])
    m = np.full(newShape, matrix.backgroundColor, dtype=np.uint8)
    for i,j in np.ndindex(newShape[0], newShape[1]):
        colorCount = Counter(matrix.m[i*xBlock: (i+1)*xBlock, j*yBlock: (j+1)*yBlock].flatten())
        m[i,j] = max(colorCount, key=colorCount.get)
    return m    

def getBestColorByPixels(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "colorByPixels" and works best for the training samples.
    """
    delPix = False
    if isDeleteTask(t) or t.outSmallerThanIn:
        delPix = True
    bestScore = 1000
    bestFunction = partial(identityM)
    bestFunction, bestScore = updateBestFunction(t, partial(colorByPixels, deletePixels=delPix), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(colorByPixels, colorMap=True, deletePixels=delPix), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(colorByPixels, oneColor=True, deletePixels=delPix), bestScore, bestFunction)
    return bestFunction

def colorByPixels(matrix, colorMap=False, oneColor=False, deletePixels=False):
    """
    Attempts to find color changes dictated by pixels. Be it pixels determine the color of the closest shape,\
    be it adjacent pixels determine a color map. 
    """
    m = matrix.m.copy()
    shList = [sh for sh in matrix.shapes if (sh.color != matrix.backgroundColor) and len(sh.pixels)>1]
    pixList = [sh for sh in matrix.dShapes if (sh.color != matrix.backgroundColor) and len(sh.pixels)==1]
    ogPixList = [p for p in pixList]
    if len(shList)==0 or len(pixList)==0 or len(pixList)>15:
        return m
    if colorMap:
        cMap = dict()
        seenP = []
        for p1 in pixList:
            for p2 in pixList:
                if abs(p1.position[1]-p2.position[1])==1 and p1.position[0]==p2.position[0] and p1.color not in cMap.keys():
                    cMap[p1.color] = p2.color
                    seenP.append(p1.position)                   
        if deletePixels:
            for pix in pixList:
                m[pix.position[0], pix.position[1]] = matrix.backgroundColor
        for i,j in np.ndindex(m.shape):
            if m[i,j] in cMap.keys() and (i,j) not in seenP:
                m[i,j] = cMap[m[i,j]]     
    else:
        if oneColor:
            cc = Counter([sh.color for sh in pixList])
            newC = max(cc, key=cc.get)
            if deletePixels:
                m[m==newC]=matrix.backgroundColor
            m[m!=matrix.backgroundColor]=newC
        else:
            if len(pixList) < len(shList):
                return m
            c = 0
            nSh = len(shList)
            while c < nSh:
                minD, newD = 1000, 1000
                bestSh, bestPix = None, None
                for pix in pixList:
                    for i in range(len(pixList)):
                        for sh in shList:
                            newD = min(np.linalg.norm(np.subtract(pix.position,np.add(p, sh.position))) for p in sh.pixels) 
                            if newD < minD:
                                minD = newD
                                bestSh = sh
                                bestPix = pix
                if bestSh != None:
                    for i,j in np.ndindex(bestSh.shape):
                        if bestSh.m[i,j] != 255:
                            m[bestSh.position[0]+i, bestSh.position[1]+j]=bestPix.color
                    c += 1
                    shList.remove(bestSh)
                    pixList.remove(bestPix)
    if deletePixels:
        for pix in ogPixList:
            m[pix.position] = matrix.backgroundColor
    return m  

def isDeleteTask(t):
    """
    Check if the task involves replacing pixels by background color, and thus deleting content. 
    """
    if hasattr(t, 'colorChanges') and t.backgroundColor in [c[1] for c in t.colorChanges]:
        return True
    return False

def getBestDeleteShapes(t, multicolor=False, diagonal=True):
    """
    Given a Task t, this function returns the partial function that uses the
    function "deleteShapes" and works best for the training samples.
    """
    attrs = set(['LaSh','SmSh','MoCl','MoCo','PiXl'])
    bestScore = 1000
    bestFunction = partial(identityM)
    for attr in attrs:
        bestFunction, bestScore = updateBestFunction(t, partial(deleteShapes, diagonal=diagonal, multicolor=multicolor,attributes=set([attr])), bestScore, bestFunction)
    return bestFunction

def getDeleteAttributes(t, diagonal=True):
    """
    Given a Task t that involves deleting shapes, this function returns the identifying attributes of the
    deleted shapes. 
    """
    bC = max(0, t.backgroundColor)
    if diagonal:
        if t.nCommonInOutDShapes == 0:
            return set()
        attrs = set.union(*[s.inMatrix.getShapeAttributes(backgroundColor=bC,\
                    singleColor=True, diagonals=True)[s.inMatrix.dShapes.index(sh[0])]\
                    for s in t.trainSamples for sh in s.commonDShapes])
        nonAttrs = set()
        c = 0
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=True, diagonals=True)
            for shi in range(len(s.inMatrix.shapes)):
                if any(s.inMatrix.dShapes[shi] == sh2[0] for sh2 in s.commonDShapes) or s.inMatrix.dShapes[shi].color == bC:
                    continue
                else:
                    if c == 0:
                        nonAttrs = shAttrs[shi]
                        c += 1
                    else:
                        nonAttrs = nonAttrs.intersection(shAttrs[shi])
                        c += 1
    else:
        if t.nCommonInOutShapes == 0:
            return set()
        attrs = set.union(*[s.inMatrix.getShapeAttributes(backgroundColor=bC,\
                    singleColor=True, diagonals=False)[s.inMatrix.shapes.index(sh[0])]\
                    for s in t.trainSamples for sh in s.commonShapes])
        nonAttrs = set()
        c = 0
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=True, diagonals=False)
            for shi in range(len(s.inMatrix.shapes)):
                if any(s.inMatrix.shapes[shi] == sh2[0] for sh2 in s.commonShapes) or s.inMatrix.shapes[shi].color == bC:
                    continue
                else:
                    if c == 0:
                        nonAttrs = shAttrs[shi]
                        c += 1
                    else:
                        nonAttrs = nonAttrs.intersection(shAttrs[shi])
                        c += 1
    return set(nonAttrs - attrs)

def deleteShapes(matrix, attributes, diagonal, multicolor):
    """
    Deletes the shapes of matrix that have specified attributes. 
    """
    m = matrix.m.copy()
    if not multicolor: 
        if diagonal:   
            shList = [sh for sh in matrix.dShapes]
        else:   
            shList = [sh for sh in matrix.shapes]
    else:
        if diagonal: 
            shList = [sh for sh in matrix.multicolorDShapes]
        else:
            shList = [sh for sh in matrix.multicolorShapes]
    attrList = matrix.getShapeAttributes(matrix.backgroundColor, not multicolor, diagonal)
    for shi in range(len(shList)):
        if len(attrList[shi].intersection(attributes)) > 0:
            m = deleteShape(m, shList[shi], matrix.backgroundColor)
    return m

def getBestArrangeShapes(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "arrangeShapes" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    if hasattr(t, 'outShape'):
        bestFunction, bestScore = updateBestFunction(t, partial(arrangeShapes, outShape=t.outShape,\
                                                            diagonal=True, multicolor=False), bestScore, bestFunction)
    elif t.outIsInMulticolorShapeSize:
        bestFunction, bestScore = updateBestFunction(t, partial(arrangeShapes, diagonal=True,\
                                                                multicolor=True,outShape='LaSh'), bestScore, bestFunction)
    else:
        bestFunction, bestScore = updateBestFunction(t, partial(arrangeShapes,shByColor=True,outShape='LaSh'), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(arrangeShapes,shByColor=True, fullFrames=True,outShape='LaSh'), bestScore, bestFunction)
    return bestFunction

def arrangeShapes (matrix, outShape = None, multicolor=True, diagonal=True, shByColor=False,\
                   fullFrames=False, outDummyMatrix=None, outDummyColor=0):
    """
    Given an output size outShape, attempt to arrange and fit all the shapes of a given typs inside it. Alternatively,
    a template matrix outDummyMatrix can be passed, in which case attempts to reproduce its pattern.
    """
    def completeFrames(shape,rotate=False,fill=False):
        'version of symmetrize shape intended for frame-like shapes' 
        m = shape.m.copy()
        if m.shape[0]>m.shape[1]:   
            newm = np.full((m.shape[0], m.shape[0]), fill_value=255)
            sideL = m.shape[0]
        else:
            newm = np.full((m.shape[1], m.shape[1]), fill_value=255)
            sideL = m.shape[1]
        for i,j in np.ndindex(m.shape):
            if m[i,j] != 255:
                if newm[i,j] == 255:
                    newm[i,j] = m[i,j]
                if newm[sideL - j - 1,i] == 255:
                    newm[sideL - j - 1,i] = m[i,j]
                if newm[sideL - i - 1,sideL - j - 1] == 255:
                    newm[sideL - i - 1,sideL - j - 1] = m[i,j]
                if newm[j,sideL - i - 1] == 255 and m[i,j]:
                    newm[j,sideL - i - 1] = m[i,j]
        newSh = copy.deepcopy(shape)
        newSh.m = newm
        newSh.shape = newm.shape
        return newSh
    
    def tessellateShapes (mat, shL, n, bC, rotation=False):
        m = mat.copy()
        arrFound = False
        rot = 1
        """
        Attempts to tessellate matrix mat with background color bC shapes is list sh.
        """
        if rotation:
            rot = 4
            if len(shL[n].pixels)==1:
                rot = 1
            if len(shL[n].pixels)==2:
                rot = 2
        for x in range(rot):
            sh = copy.deepcopy(shL[n])
            sh.m = np.rot90(sh.m,x).copy()
            sh.shape = sh.m.shape
            if mat.shape[0] < sh.shape[0] or mat.shape[1] < sh.shape[1]:
                continue
            for i, j in np.ndindex(tuple(mat.shape[k] - sh.shape[k] + 1 for k in (0,1))):
                if np.all(np.logical_or(m[i: i+sh.shape[0], j: j+sh.shape[1]] == bC, sh.m == 255)):
                    for k, l in np.ndindex(sh.shape):
                        if sh.m[k,l] != 255:
                            m[i+k,j+l] = sh.m[k,l]
                    if n == len(shL) - 1:
                        return m, True
                    m, arrFound = tessellateShapes(m, shL, n+1, bC, rotation)
                    if arrFound:
                        return m, True
                    if not arrFound:
                        for k, l in np.ndindex(sh.shape):
                            if sh.m[k,l] != 255:
                                m[i+k,j+l] = bC
        return m, False
                                            
    if shByColor:
        shList = [sh for sh in matrix.shapesByColor]
        if fullFrames:
            shList = [completeFrames(sh) for sh in matrix.shapesByColor]
    else:
        if not multicolor: 
            if diagonal:   
                shList = [sh for sh in matrix.dShapes if sh.color != matrix.backgroundColor]
            else:   
                shList = [sh for sh in matrix.shapes if sh.color != matrix.backgroundColor]
        else:
            if diagonal: 
                shList = [sh for sh in matrix.multicolorDShapes]
            else:
                shList = [sh for sh in matrix.multicolorShapes]
    if len(shList) < 2 or len(shList)>7:
        return matrix.m.copy()
    if outDummyMatrix is None:
        shList.sort(key=lambda x: x.shape[0]*x.shape[1], reverse=True)
        if outShape == 'LaSh':
            outShape = shList[0].shape    
        if outShape == None:
            outShape = matrix.shape
        if all((sh.shape[0]<=outShape[0] and sh.shape[1]<=outShape[1]) for sh in shList) and\
                                sum(len(sh.pixels) for sh in shList) <= outShape[0]*outShape[1]:
            m, tessellate = tessellateShapes(np.full(outShape, fill_value=matrix.backgroundColor),shList,\
                                             0,matrix.backgroundColor)
            if tessellate:
                return m
            m, tessellate = tessellateShapes(np.full(outShape, fill_value=matrix.backgroundColor),shList,\
                                             0,matrix.backgroundColor,rotation=True)
            if tessellate:
                return m
    else:
        m = np.full(outDummyMatrix.shape,fill_value=outDummyColor)
        shC = Counter([sh.shape for sh in shList])
        pSh = shC.most_common(1)[0][0]
        if pSh[0]>m.shape[0] or pSh[1]>m.shape[1] or m.shape[0]%pSh[0] != 0 or m.shape[1]%pSh[1] != 0:
            return matrix.m.copy()
        for i, j in np.ndindex(m.shape[0]//pSh[0], m.shape[1]//pSh[1]):
            for k, l in np.ndindex(matrix.shape[0]-pSh[0]+1, matrix.shape[1]-pSh[1]+1):
                if np.all((matrix.m[k:k+pSh[0],l:l+pSh[1]] == outDummyColor)==(outDummyMatrix[i*pSh[0]:(i+1)*pSh[0],j*pSh[1]:(j+1)*pSh[1]]==0)):
                    m[i*pSh[0]:(i+1)*pSh[0],j*pSh[1]:(j+1)*pSh[1]] = matrix.m[k:k+pSh[0],l:l+pSh[1]]
                    break
        return m
    return matrix.m.copy()

def getBestLayShapes(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "layShapes" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    outShape = None
    if all([len(s.inMatrix.shapes) > 15 for s in t.trainSamples]):
        return bestFunction
    if t.sameIOShapes:
        outShape = 'inShape'
    elif hasattr(t, 'outShape'):
        outShape = t.outShape
    if outShape != None:
        for sortBy in ['grid','lr','ud','smallToLarge',set.intersection(*t.inColors)]:
            for reverse in [False, True]:
                for overlap in [(0,0), (1,1), (-1,-1)]:
                    for multicolor in [True, False]:
                        for direction in [(1,0), (0,1), (1,1)]:
                            bestFunction, bestScore = updateBestFunction(t, partial(layShapes, firstPos=(0,0), diagonal=True, multicolor=multicolor,\
                                                            outShape=outShape, overlap=overlap, direction=direction, sortBy=sortBy, reverse=reverse), bestScore, bestFunction)
                        bestFunction, bestScore = updateBestFunction(t, partial(layShapes, firstPos=(1,0), direction=(-1,0), diagonal=True, multicolor=multicolor,\
                                                            outShape=outShape, overlap=overlap, sortBy=sortBy, reverse=reverse), bestScore, bestFunction)
                        bestFunction, bestScore = updateBestFunction(t, partial(layShapes, firstPos=(0,1), direction=(0,-1), diagonal=True, multicolor=multicolor,\
                                                            outShape=outShape, overlap=overlap, sortBy=sortBy, reverse=reverse), bestScore, bestFunction)
                        bestFunction, bestScore = updateBestFunction(t, partial(layShapes, firstPos=(1,1), direction=(-1,-1), diagonal=True, multicolor=multicolor,\
                                                            outShape=outShape, overlap=overlap, sortBy=sortBy, reverse=reverse), bestScore, bestFunction)
    elif t.outSmallerThanIn:
         bestFunction, bestScore = updateBestFunction(t, partial(layShapes,completeRect=True,diagonal=True,\
                                         direction=(0,0),sortBy='smallToLarge', reverse=True, outShape='LaSh',multicolor=False), bestScore, bestFunction)
    return bestFunction

def layShapes(matrix, firstPos=(0,0), direction=(0,1), overlap=(0,0), outShape='inShape', multicolor=True,\
              diagonal=True, sortBy='lrud', completeRect=False, reverse=True):
    """
    Moves all shapes and lays them in an appropriate way in a matrix of size outShape. 
    The first shape is position at firstPos, and the succesive shapes are layed in the given direction,
    with possible overlap. Shapes can be sorted before by position, size or color. 
    """
    def completeRectangles(shape):
        """
        Version of complete rectangles shape intended for frame-like shapes.
        """
        newSh = copy.deepcopy(shape)
        newSh.m = np.full(shape.shape, fill_value=shape.color)
        newSh.shape = newSh.m.shape
        return newSh
    m = matrix.m.copy()
    if not multicolor: 
        if diagonal:   
            shList = [sh for sh in matrix.dShapes if sh.color != matrix.backgroundColor]
        else:   
            shList = [sh for sh in matrix.shapes if sh.color != matrix.backgroundColor]
    else:
        if diagonal: 
            shList = [sh for sh in matrix.multicolorDShapes]
        else:
            shList = [sh for sh in matrix.multicolorShapes]
    if completeRect and (not multicolor):
        shList = [completeRectangles(sh) for sh in shList]
    if sortBy == 'smallToLarge':
        shList.sort(key=lambda x: len(x.pixels), reverse=reverse)
    elif sortBy == 'lr':
        shList.sort(key=lambda x: x.position[1], reverse=reverse)
    elif sortBy == 'ud':
        shList.sort(key=lambda x: x.position[0], reverse=reverse)
    elif sortBy == 'grid':
        shList.sort(key=lambda x: x.position[0])
        newList = []
        shList.sort(key=lambda x: x.position[0])
        gridD = int(len(shList)**(1/2))
        for i in range(gridD):
            newList += sorted(shList[i*gridD: (i + 1)*gridD], key=lambda x: x.position[1])
        shList = [sh for sh in newList]
    elif type(sortBy) == int:
        shList.sort(key=lambda x: x.colorCount[sortBy])
    if len(shList) == 0:
        return m
    if outShape == 'inShape':
        m = np.full(matrix.shape, fill_value=matrix.backgroundColor)
    elif outShape == 'LaSh' and sortBy == 'smallToLarge' and reverse:
        m = np.full(shList[0].m.shape, fill_value=matrix.backgroundColor)
    else:
        m = np.full(outShape, fill_value=matrix.backgroundColor)
    shList = [sh for sh in shList if (sh.shape[0] <= m.shape[0] and sh.shape[1] <= m.shape[1])]
    startPos = (firstPos[0]*(m.shape[0]), firstPos[1]*(m.shape[1]))
    (currentX, currentY) = startPos
    for sh in shList:
        if currentX + sh.shape[0]*direction[0] > m.shape[0] or currentX + sh.shape[0]*direction[0] < 0:
            (currentX, currentY) = (startPos[0], currentY + sh.shape[1] - overlap[1])
            if currentY > m.shape[1] or currentY < 0:
                return matrix.m.copy()
        if currentY + sh.shape[1]*direction[1] > m.shape[1] or currentY + sh.shape[1]*direction[1] < 0:
            (currentX, currentY) = (currentX + sh.shape[0] - overlap[0], startPos[1])
            if currentX > m.shape[0] or currentX < 0:
                return matrix.m.copy()
        newInsert = copy.deepcopy(sh)
        newInsert.position = (currentX, currentY)
        if direction[0] < 0:
            newInsert.position = (newInsert.position[0] - sh.shape[0], newInsert.position[1])
        if direction[1] < 0:
            newInsert.position = (newInsert.position[0], newInsert.position[1] - sh.shape[1])
        m = insertShape(m, newInsert)
        currentX, currentY = (currentX + (sh.shape[0]- overlap[0])*direction[0] , currentY + (sh.shape[1]- overlap[1])*direction[1])
    return m

def getBestAlignShapes(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "alignShapes" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    if t.sameIOShapes:
        for cc in set.intersection(*t.inColors):
            bestFunction, bestScore = updateBestFunction(t, partial(alignShapes, refColor=cc), bestScore, bestFunction)
    elif t.outSmallerThanIn:
        bestFunction, bestScore = updateBestFunction(t, partial(alignShapes, compress=True, crop=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(alignShapes, compress=False, crop=True), bestScore, bestFunction)
    return bestFunction

def alignShapes(matrix, compress=True, diagonal=True, multicolor=False, refColor=None, crop=False):
    """
    Attempts to align the shapes in matrix. If refColor is passed, then the shapes of that color remain unmoved.
    The arguments compress and crop, are called if the output of the task is smaller than the input. 
    """
    m = matrix.m.copy()
    if not multicolor: 
        if diagonal:   
            shList = [sh for sh in matrix.dShapes if sh.color != matrix.backgroundColor]
        else:   
            shList = [sh for sh in matrix.shapes if sh.color != matrix.backgroundColor]
    else:
        if diagonal: 
            shList = [sh for sh in matrix.multicolorDShapes]
        else:
            shList = [sh for sh in matrix.multicolorShapes]
    if len(shList) < 2:
        return m
    shList.sort(key=lambda x: x.position[1])
    arrFound = False
    if all(shList[i].position[1]+shList[i].shape[1] <= shList[i+1].position[1] for i in range(len(shList)-1)):
        arrFound = (0,1)
    if arrFound == False:
        shList.sort(key=lambda x: x.position[0])
        if all(shList[i].position[0]+shList[i].shape[0] <= shList[i+1].position[0] for i in range(len(shList)-1)):
           arrFound = (1,0)
    if arrFound == False:
        return m  
    for sh in shList:
        m = deleteShape(m, sh, matrix.backgroundColor)
    (currentX, currentY) = (0, 0)
    if refColor != None and not crop:
        refPos = -1
        for sh in shList:
            if sh.color == refColor:
                refPos = sh.position
                break
        if refPos == -1:
            return matrix.m.copy()
        for sh in shList:
            newInsert = copy.deepcopy(sh)
            if arrFound == (0,1):
                newInsert.position = (refPos[0], sh.position[1])
            elif arrFound == (1,0):
                newInsert.position = (sh.position[0], refPos[1])
            m = insertShape(m, newInsert)
    else:
        for sh in shList:
            newInsert = copy.deepcopy(sh)
            if compress:
                newInsert.position = (currentX, currentY)
                (currentX, currentY) = (currentX + sh.shape[0]*arrFound[0], currentY + sh.shape[1]*arrFound[1])
            else:
                newInsert.position = (0,0)
            m = insertShape(m, newInsert)
        
    if crop:
        bC = matrix.backgroundColor
        if np.all(m == bC):
            return m
        x1, x2, y1, y2 = 0, m.shape[0]-1, 0, m.shape[1]-1
        while x1 <= x2 and np.all(m[x1,:] == bC):
            x1 += 1
        while x2 >= x1 and np.all(m[x2,:] == bC):
            x2 -= 1
        while y1 <= y2 and np.all(m[:,y1] == bC):
            y1 += 1
        while y2 >= y1 and np.all(m[:,y2] == bC):
            y2 -= 1
        return(m[x1:x2+1,y1:y2+1])
    else:
        return m

def isReplicateTask(t):
    """
    Identify if a given task involves the action of replicating shapes. In this case, return a list of 
    booleans indicating the types of shapes involved. 
    """
    if all(any(sh[2] > 1 for sh in s.commonMulticolorDShapes) for s in t.trainSamples):
        return [True, True, True]
    elif all(any(sh[2] > 1 for sh in s.commonMulticolorShapes) for s in t.trainSamples):
        return [True, True, False]
    elif all(any(sh[2] > 1 for sh in s.commonShapes) for s in t.trainSamples):
        return [True, False, False]
    elif all(any(sh[2] > 1 for sh in s.commonDShapes) for s in t.trainSamples):
        return [True, False, True]
    return [False]

def getBestReplicateShapes(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "replicateShapes" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    isReplicateParams = isReplicateTask(t)
    deleteOriginal = False
    multicolor = True
    diagonal = True
    if isReplicateParams[0]:
        multicolor = isReplicateParams[1]
        diagonal = isReplicateParams[2]
    if isDeleteTask(t):
        deleteOriginal = True
    
    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, diagonal=diagonal, multicolor=multicolor,deleteOriginal=deleteOriginal,\
                                                            anchorType='subframe'), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, diagonal=diagonal, multicolor=multicolor,deleteOriginal=deleteOriginal,\
                                                            anchorType='subframe', allCombs=False,attributes=set(['MoCl'])), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, diagonal=diagonal, multicolor=multicolor,deleteOriginal=deleteOriginal,\
                                                            anchorType='subframe', allCombs=False,scale=True,attributes=set(['MoCl'])), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, diagonal=diagonal, multicolor=multicolor,deleteOriginal=deleteOriginal,\
                                                            anchorType='subframe', allCombs=True,attributes=set(['MoCl'])), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, diagonal=diagonal, multicolor=multicolor,deleteOriginal=deleteOriginal,\
                                                            anchorType='subframe', allCombs=True,scale=True,attributes=set(['MoCl'])), bestScore, bestFunction)
    
    """
    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes,diagonal=diagonal, multicolor=False, anchorType='subframe', allCombs=False,\
                                                                adoptAnchorColor=True), bestScore, bestFunction)
    if isReplicateParams[0]:
        if bestScore == 0:
            return bestFunction
        for attributes in [set(['MoCl'])]:
            cC = Counter([cc[0] for cc in t.colorChanges])
            cc = max(cC, key=cC.get)
            bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=diagonal, multicolor=multicolor, anchorType='all', anchorColor=cc,\
                                    mirror=None, rotate=0, allCombs=True, scale=False, deleteOriginal=deleteOriginal), bestScore, bestFunction)
            bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=diagonal, multicolor=multicolor, anchorType='all', anchorColor=cc,\
                                    mirror=None, rotate=0, allCombs=True, scale=False, deleteOriginal=deleteOriginal), bestScore, bestFunction)
            if bestScore == 0:
                return bestFunction
            for mirror in [None, 'lr', 'ud']:
                for rotate in range(0, 4):
                    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=diagonal, multicolor=multicolor, anchorType='all', anchorColor=cc,\
                                    mirror=mirror, rotate=rotate, allCombs=False, scale=False, deleteOriginal=deleteOriginal), bestScore, bestFunction)
                    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=diagonal, multicolor=multicolor, anchorType='all', anchorColor=cc,\
                                    mirror=mirror, rotate=rotate, allCombs=False, scale=True, deleteOriginal=deleteOriginal), bestScore, bestFunction)
                    if bestScore == 0:      
                        return bestFunction
    """
                                            
    cC = Counter([cc[0] for cc in t.colorChanges])
    cc = max(cC, key=cC.get)
    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, diagonal=False, multicolor=multicolor, anchorType='all', anchorColor=cc,\
                        allCombs=False, scale=False, deleteOriginal=deleteOriginal,perfectFit=True), bestScore, bestFunction)
    for attributes in [set(['UnCo'])]:    
        bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=True, multicolor=False, anchorType='all', anchorColor=cc,\
                        allCombs=True, scale=False, deleteOriginal=deleteOriginal), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=True, multicolor=False, anchorType='all', anchorColor=cc,\
                        allCombs=False, scale=False, deleteOriginal=deleteOriginal, perfectFit=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=True, multicolor=False, anchorType='all', anchorColor=cc,\
                        allCombs=False, scale=True, deleteOriginal=deleteOriginal, perfectFit=False), bestScore, bestFunction)
   
    """
    if t.hasPartialFrame:
        for attributes in [set(['IsRef'])]:    
            bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=True, multicolor=False, anchorType='all', anchorColor=cc,\
                            allCombs=True, scale=False, deleteOriginal=deleteOriginal), bestScore, bestFunction)
            bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=True, multicolor=False, anchorType='all', anchorColor=cc,\
                            allCombs=False, scale=False, deleteOriginal=deleteOriginal, perfectFit=True), bestScore, bestFunction)
    """        
    return bestFunction

def replicateShapes(matrix, attributes=None, diagonal=False, multicolor=True, anchorType=None, anchorColor=0,\
                    mirror=None, rotate=0, allCombs=False, scale=False, deleteOriginal=False, perfectFit=False,
                    adoptAnchorColor=False):
    """
    Replicates shapes at target locations. The arguments diagonal, multicolor, attributes determine the shapes to be
    replicated. The arguments mirror, rotate, allCombs and scale modify the shape before it is replicated. 
    The arguments anchorType, anchorColor and perfectFit determine the places where the shapes should be replicated.
    The options adoptAnchorColor, modify the shape once it has been replicated, either by changing its color. The option
    deleteOriginal, deletes the original shape after replicating it. 
    """
    m = matrix.m.copy()
    #first find the shape or shapes to replicate
    if diagonal:
        if multicolor:
            shList = matrix.multicolorDShapes
        else:
            shList = matrix.dShapes
    else:
        if multicolor:
            shList = matrix.multicolorShapes
        else:
            shList = matrix.shapes
    if attributes != None:
        repList = []
        attrList = matrix.getShapeAttributes(backgroundColor=matrix.backgroundColor,\
                                             singleColor=not multicolor, diagonals=diagonal)
        for shi in range(len(shList)):
            if len(attrList[shi].intersection(attributes)) == 1:
                repList.append(shList[shi])
        if len(repList) == 0:
            return m
    else:
        if multicolor:
            repList = [sh for sh in shList if (len(sh.pixels)>1 and not sh.isSquare)]
        else:
            repList = [sh for sh in shList if (sh.color != matrix.backgroundColor and not sh.isSquare)]
    delList = [sh for sh in repList]
    if len(repList) > 10:
        return m
    #apply transformations to replicating shapes
    if allCombs:
        newList = []
        for repShape in repList:
            for r in range(0,4):
                mr, mrM = np.rot90(repShape.m.copy(), r), np.rot90(repShape.m[::-1,::].copy(), r)
                newRep, newRepM = copy.deepcopy(repShape), copy.deepcopy(repShape)
                newRep.m, newRepM.m = mr, mrM
                newRep.shape, newRepM.shape = mr.shape, mrM.shape
                newList.append(newRep)
                newList.append(newRepM)
        repList = [sh for sh in newList]           
    elif mirror == 'lr' and len(repList) == 1:
        newRep = copy.deepcopy(repList[0])
        newRep.m = repList[0].m[::,::-1]
        repList = [newRep]
    elif mirror == 'ud' and len(repList) == 1:
        newRep = copy.deepcopy(repList[0])
        newRep.m = repList[0].m[::-1,::]
        repList = [newRep]
    elif rotate > 0 and len(repList) == 1:
        newRep = copy.deepcopy(repList[0])
        newRep.m = np.rot90(repList[0].m,rotate)
        newRep.shape = newRep.m.shape
        repList = [newRep]
    if scale == True:
        newRepList=[]
        for repShape in repList:
            for sc in range(4,0,-1):
                newRep = copy.deepcopy(repShape)
                newRep.m = np.repeat(np.repeat(repShape.m, sc, axis=1), sc, axis=0)
                newRep.shape = newRep.m.shape
                newRep.pixels = set([(i,j) for i,j in np.ndindex(newRep.m.shape) if newRep.m[i,j]!=255])
                newRepList.append(newRep)
        repList = [sh for sh in newRepList]
    repList.sort(key=lambda x: len(x.pixels), reverse=True)
    if anchorType == 'subframe' and scale:
        repList.sort(key=lambda x: len(x.pixels))
    #then find places to replicate
    if anchorType == 'all':
        seenM = np.zeros(m.shape, dtype=int)
        for repSh in repList:
            if np.all(np.logical_or(repSh.m==255,repSh.m==anchorColor)):
                continue
            for j in range(matrix.shape[1] - repSh.shape[1]+1):
                for i in range(matrix.shape[0] - repSh.shape[0]+1):
                    if np.all(np.logical_or(m[i:i+repSh.shape[0],j:j+repSh.shape[1]]==anchorColor,repSh.m==255))\
                                                    and np.all(seenM[i:i+repSh.shape[0],j:j+repSh.shape[1]]==0):
                        if perfectFit:
                            surrPixList = set([(i+p[0]+1, j+p[1]) for p in repSh.pixels]+[(i+p[0], j+p[1]+1) for p in repSh.pixels]\
                                               +[(i+p[0]-1, j+p[1]) for p in repSh.pixels]+[(i+p[0], j+p[1]-1) for p in repSh.pixels])
                            surrPixList = surrPixList - set([(i+p[0], j+p[1]) for p in repSh.pixels])
                            surrPixList = set([p for p in surrPixList if (p[0]>=0 and p[1]>=0 and p[0]<m.shape[0] and p[1]<m.shape[1])])
                            if len(set([m[p[0],p[1]] for p in surrPixList]))==1 and m[list(surrPixList)[0]]!=anchorColor: 
                                newInsert = copy.deepcopy(repSh)
                                newInsert.position = (i, j)
                                m = insertShape(m, newInsert)
                                for l, k in np.ndindex(repSh.shape):
                                    seenM[i+l,j+k]=1
                        else:
                            newInsert = copy.deepcopy(repSh)
                            newInsert.position = (i, j)
                            m = insertShape(m, newInsert)
                            for l, k in np.ndindex(repSh.shape):
                                seenM[i+l,j+k]=1
    elif anchorType == 'subframe':
        seenM = np.zeros(matrix.shape)
        if attributes != None:
            for repSh in delList:
                for i,j in np.ndindex(repSh.shape):
                    seenM[i+repSh.position[0],j+repSh.position[1]]=1
        for sh2 in shList:
            score, bestScore= 0, 0
            bestSh = None
            for repSh in repList:
                if adoptAnchorColor:
                    if sh2.isSubshape(repSh,sameColor=False,rotation=False,mirror=False) and len(sh2.pixels)<len(repSh.pixels):
                        for x in range((repSh.shape[0]-sh2.shape[0])+1):
                            for y in range((repSh.shape[1]-sh2.shape[1])+1):
                                mAux = m[max(sh2.position[0]-x, 0):min(sh2.position[0]-x+repSh.shape[0], m.shape[0]), max(sh2.position[1]-y, 0):min(sh2.position[1]-y+repSh.shape[1], m.shape[1])]
                                shAux = repSh.m[max(0, x-sh2.position[0]):min(repSh.shape[0],m.shape[0]+x-sh2.position[0]),max(0, y-sh2.position[1]):min(repSh.shape[1],m.shape[1]+y-sh2.position[1])]
                                seenAux = seenM[max(sh2.position[0]-x, 0):min(sh2.position[0]-x+repSh.shape[0], m.shape[0]), max(sh2.position[1]-y, 0):min(sh2.position[1]-y+repSh.shape[1], m.shape[1])]
                                if np.all(np.logical_or(shAux!=255, mAux == matrix.backgroundColor)) and np.all(seenAux==0):
                                    score = np.count_nonzero(mAux!=matrix.backgroundColor)
                                    if score > bestScore:
                                        bestScore = score
                                        bestX, bestY = sh2.position[0]-x, sh2.position[1]-y
                                        bestSh = copy.deepcopy(repSh)
                else:
                    if sh2.isSubshape(repSh,sameColor=True,rotation=False,mirror=False) and len(sh2.pixels)<len(repSh.pixels):
                        for x in range((repSh.shape[0]-sh2.shape[0])+1):
                            for y in range((repSh.shape[1]-sh2.shape[1])+1):
                                mAux = m[max(sh2.position[0]-x, 0):min(sh2.position[0]-x+repSh.shape[0], m.shape[0]), max(sh2.position[1]-y, 0):min(sh2.position[1]-y+repSh.shape[1], m.shape[1])]
                                shAux = repSh.m[max(0, x-sh2.position[0]):min(repSh.shape[0],m.shape[0]+x-sh2.position[0]),max(0, y-sh2.position[1]):min(repSh.shape[1],m.shape[1]+y-sh2.position[1])]
                                seenAux = seenM[max(sh2.position[0]-x, 0):min(sh2.position[0]-x+repSh.shape[0], m.shape[0]), max(sh2.position[1]-y, 0):min(sh2.position[1]-y+repSh.shape[1], m.shape[1])]
                                if np.all(np.logical_or(mAux==shAux, mAux == matrix.backgroundColor)) and np.all(seenAux==0):
                                    score = np.count_nonzero(mAux==shAux)
                                    if score > bestScore:
                                        bestScore = score
                                        bestX, bestY = sh2.position[0]-x, sh2.position[1]-y
                                        bestSh = copy.deepcopy(repSh)
            if bestSh != None:
                for i,j in np.ndindex(bestSh.shape):
                    if i+bestX>=0 and i+bestX<seenM.shape[0] and j+bestY>=0 and j+bestY<seenM.shape[1]:
                        seenM[i+bestX,j+bestY]=1
                newInsert = copy.deepcopy(bestSh)
                if adoptAnchorColor:
                    newInsert.m[newInsert.m!=255]=sh2.color
                newInsert.position = (bestX, bestY)
                newInsert.shape = newInsert.m.shape
                m = insertShape(m, newInsert)        
    if deleteOriginal:
        for sh in delList:
            m = deleteShape(m, sh, matrix.backgroundColor)
    return(m)
    
def getBestReplicateOneShape(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "replicateOneShape" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    if t.outSmallerThanIn or (not t.sameIOShapes):
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='pixelwise', reshape=True,\
                                                                multicolor=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='horizontal', reshape=True,\
                                                                multicolor=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='vertical', reshape=True,\
                                                                multicolor=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='pixelwise', reshape=True,\
                                                                multicolor=False, paintLikePix=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='horizontal', reshape=True,\
                                                                multicolor=False, paintLikePix=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='vertical', reshape=True,\
                                                                multicolor=False, paintLikePix=True), bestScore, bestFunction)
    deleteO = [False]
    deleteA = [False]
    if isDeleteTask(t):
        deleteO += [True]
        deleteA += [True]
    if t.sameIOShapes:
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='pixelwise', multicolor=False,\
                                                                paintLikePix=True), bestScore, bestFunction)
        for deleteOriginal in deleteO:
            for deleteAnchor in deleteA:
                bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, deleteOriginal=deleteOriginal,\
                                                                deleteAnchor=deleteAnchor), bestScore, bestFunction)
    return bestFunction

def replicateOneShape(matrix, diagonal=True, multicolor=True, deleteOriginal=False,\
                      deleteAnchor=False, lay=False, reshape=False,overlap=0, paintLikePix=False):
    """
    To be called when the task involved replicating one shape. The target positions are specified by pixels. Pixels may
    specify a color change with paitLikePix, or can be deleted after with deleteAnchor. The arguments reshape and lay modify
    the output by moving the shapes or changing the output shape. 
    """
    m = matrix.m.copy()
    #first find the shape or shapes to replicate
    if diagonal:
        if multicolor:
            shList = [sh for sh in matrix.multicolorDShapes if len(sh.pixels)>1]
        else:
            shList = [sh for sh in matrix.dShapes if (len(sh.pixels)>1 and sh.color != matrix.backgroundColor)]
    else:
        if multicolor:
            shList = [sh for sh in matrix.multicolorShapes if len(sh.pixels)>1]
        else:
            shList = [sh for sh in matrix.shapes if (len(sh.pixels)>1 and sh.color != matrix.backgroundColor)]
    pixList = matrix.isolatedPixels#[pix for pix in matrix.dShapes if len(pix.pixels)==1]
    if len(shList) != 1 or len(pixList) == 0:
        return m
    repSh = shList[0]
    if lay != False:
        if lay == 'pixelwise':
            if len(pixList) < 2:
                return m
            if len(set([p.position[0] for p in pixList])) == 1:
                pixList.sort(key=lambda x: x.position[1])
                steps = [(0, pixList[i].position[1] - pixList[i-1].position[1] - 1) for i in range(1,len(pixList))]
                direction = (0, (pixList[1].position[1] - pixList[0].position[1] - 1)//abs(pixList[1].position[1] - pixList[0].position[1] - 1))
            elif len(set([p.position[1] for p in pixList])) == 1:
                pixList.sort(key=lambda x: x.position[0])
                steps = [(pixList[i].position[0] - pixList[i-1].position[0] - 1, 0) for i in range(1,len(pixList))]
                direction = ((pixList[1].position[0] - pixList[0].position[0] - 1)//abs(pixList[1].position[0] - pixList[0].position[0] - 1), 0)
            else:
                return m
            if paintLikePix and repSh.color == pixList[-1].color:
                    steps = steps[::-1]
                    steps = [(-st[0], -st[1]) for st in steps]
                    direction = (-direction[0], -direction[1])
                    pixList = pixList[::-1]
            if reshape:
                m = np.full((repSh.shape[0]*(1 + (len(pixList)-1)*abs(direction[0])), repSh.shape[1]*(1 + (len(pixList)-1)*abs(direction[1]))),\
                            fill_value = matrix.backgroundColor)
                for i in range(len(pixList)):
                    newInsert = copy.deepcopy(repSh)
                    newInsert.position = (i*repSh.shape[0]*direction[0], i*repSh.shape[1]*direction[1])
                    if paintLikePix:
                        newInsert.m[repSh.m == repSh.color] = pixList[i].color
                    m = insertShape(m, newInsert)
                deleteOriginal = False
            else:
                pos = repSh.position
                for (p,i) in zip(steps, [j for j in range(1,len(pixList))]):
                    pos = (pos[0] + direction[0]*repSh.shape[0] + p[0], pos[1] + direction[1]*repSh.shape[1] + p[1])
                    newInsert = copy.deepcopy(repSh)
                    newInsert.position = pos
                    if paintLikePix:
                        newInsert.m[repSh.m == repSh.color] = pixList[i].color
                    m = insertShape(m, newInsert)
        elif lay == 'horizontal': 
            m = np.full((repSh.shape[0], len(pixList)*repSh.shape[1]), fill_value = matrix.backgroundColor)
            deleteOriginal = False
            for i in range(len(pixList)):
                newInsert = copy.deepcopy(repSh)
                newInsert.position = (0, i*repSh.shape[1])
                m = insertShape(m, newInsert)
        elif lay == 'vertical': 
            m = np.full((len(pixList)*repSh.shape[0], repSh.shape[1]), fill_value = matrix.backgroundColor)
            deleteOriginal = False
            for i in range(len(pixList)):
                newInsert = copy.deepcopy(repSh)
                newInsert.position = (i*repSh.shape[0], 0)
                m = insertShape(m, newInsert)
    else:
        for pix in pixList:
            if (pix.position[0] >= repSh.position[0]) and (pix.position[1] >= repSh.position[1]) \
                and (pix.position[0] < repSh.position[0]+repSh.shape[0]) and (pix.position[1] < repSh.position[1]+repSh.shape[1]):
                continue
            newInsert = copy.deepcopy(repSh)
            if pix.m[0,0] in repSh.m:
                newInsert = copy.deepcopy(repSh)
                for i, j in np.ndindex(repSh.shape):
                    if repSh.m[i,j] == pix.m[0,0]:
                        newInsert.position = (pix.position[0]-i, pix.position[1]-j)
                        break
            else:
                newInsert.position = (pix.position[0] - (repSh.shape[0]-1)//2, pix.position[1] - (repSh.shape[1]-1)//2)
            m = insertShape(m, newInsert)
            if deleteAnchor:
                m = deleteShape(m, pix, matrix.backgroundColor)
    if deleteOriginal:
        m = deleteShape(m, repSh, matrix.backgroundColor)
    return m

def getBestMoveToPanel(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "moveToPanel" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    for fit in [True, False]:
        for igPan in [True, False]:
            for uniq in [True, False]:
                bestFunction, bestScore = updateBestFunction(t, partial(moveToPanel,fit=fit,\
                                                ignorePanel=igPan), bestScore, bestFunction)
    return bestFunction

def moveToPanel(matrix, diagonal=True, fit=False, ignorePanel=False, cropPanel=True, uniq=True):
    """
    Moves shapes and places them inside a larger square shape panel. The arguments ignorePanel and cropPanel
    modify the output in the cases where the output matrix has a different shape. 
    """
    m = matrix.m.copy()
    shList = [sh for sh in matrix.multicolorDShapes if len(sh.pixels)>1]
    if len(shList) < 2 or len(shList) > 8:
        return m
    shList.sort(key=lambda x: x.shape[0]*x.shape[1],reverse=True)
    panel = shList[0]
    shList = shList[1:]
    if fit and hasattr(panel, 'color'):
        pC = panel.color
        for sh in shList:
            found = False
            for x in range(4):
                rotSh = np.rot90(sh.m, x).copy()
                if panel.shape[0]-rotSh.shape[0]+1<0 or panel.shape[1]-rotSh.shape[1]+1<0:
                    continue
                for i, j in np.ndindex(panel.shape[0]-rotSh.shape[0]+1,panel.shape[1]-rotSh.shape[1]+1):
                    if np.all((rotSh==pC)==(panel.m[i:i+rotSh.shape[0],j:j+rotSh.shape[1]]==255)):
                        newInsert = copy.deepcopy(sh)
                        newInsert.m = rotSh
                        newInsert.shape = rotSh.shape
                        newInsert.position = (panel.position[0]+i, panel.position[1]+j)
                        m = insertShape(m, newInsert)
                        found = True
                        break
                if found:
                    break
    else:
        pixList = [pix for pix in matrix.dShapes if len(pix.pixels)==1]
        pixList = [pix for pix in pixList if all(pix.position[i]>=panel.position[i]\
                                        and pix.position[i]<panel.position[i]+panel.shape[i] for i in [0,1])]
        if len(pixList)==0:
            return m
        newInsList = []
        for pix in pixList:
            for sh in shList:
                if pix.m[0,0] in sh.m:
                    newInsert = copy.deepcopy(sh)
                    if sh.nColors == 1:
                        newInsert.position = (pix.position[0] - (sh.shape[0]-1)//2, pix.position[1] - (sh.shape[1]-1)//2)
                    else:
                        for i, j in np.ndindex(sh.shape):
                            if sh.m[i,j] == pix.m[0,0]:
                                newInsert.position = (pix.position[0]-i, pix.position[1]-j) 
                    newInsList.append(newInsert)
                    if uniq:
                        break
            if ignorePanel:
                m = deleteShape(m, panel, matrix.backgroundColor)
            for sh in newInsList:
                m = insertShape(m, sh)
    if cropPanel:
        return m[panel.position[0]:panel.position[0]+panel.shape[0],\
                 panel.position[1]:panel.position[1]+panel.shape[1]]
    return m
        
def printShapes(matrices, base=0, backgroundColor=0):
    """
    This function returns the result of printing one matrices on the other.
    The matrices are shape matrices and may contain 255.
    """
    if base == 1:
        matrices = matrices[::-1]
    m = np.zeros(matrices[0].shape, dtype=int)
    for i,j in np.ndindex(m.shape):
        if matrices[0][i,j] != 255 and matrices[0][i,j] != backgroundColor:
            if matrices[1][i,j] != 255:
                m[i,j] = matrices[1][i,j]
            else:
                m[i,j] = matrices[0][i,j]
        else:
            m[i,j] = backgroundColor
    return m 

def multiplyMatrices(matrices, outShape=None, background=0, base=0, color=0):
    """
    Copy m1 matrix into every pixel of m2 if it is not background. Output has shape
    m1.shape*m2.shape. base and color are arguments to swap m1 and m2 and to choose
    color appropriately
    """
    if base == 1:
        matrices = matrices[::-1]
    m1 = matrices[0].copy()
    m2 = matrices[1].copy()
    s1 = m1.shape
    s2 = m2.shape
    m = np.full((m1.shape[0]*m2.shape[0],m1.shape[1]*m2.shape[1]), fill_value=background)
    if color == 0:
        for i,j in np.ndindex(s1):
            if m1[i,j] != background:
                m[i*s2[0]:(i+1)*s2[0], j*s2[1]:(j+1)*s2[1]] = m2
    else:
       for i,j in np.ndindex(s1):
            if m1[i,j] != background:
                for k,l in np.ndindex(m2.shape):
                    if m2[k,l] != background:
                        m[i*s2[0]+k, j*s2[1]+l] = m1[i,j] 
    return m   

def overlapSubmatrices(matrix, colorHierarchy, shapeFactor=None):
    """
    This function returns the result of overlapping all submatrices of a given
    shape factor pixelswise with a given color hierarchy. Includes option to overlap
    all grid cells.     
    """
    if shapeFactor == None:
       submat = [t[0].m for t in matrix.grid.cellList]

    else:
        matrix = matrix.m
        sF = tuple(sin // sfact for sin, sfact in zip(matrix.shape, shapeFactor))
        submat = [matrix[sF[0]*i:sF[0]*(i+1),sF[1]*j:sF[1]*(j+1)] for i,j in np.ndindex(shapeFactor)]
    return overlapMatrices(submat, colorHierarchy)

def overlapMatrices(matrices, colorHierarchy):
    """
    Overlaps matrices of a given shape according to the color hierarchy
    """
    m = np.zeros(matrices[0].shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        m[i,j] = colorHierarchy[max([colorHierarchy.index(x[i,j]) for x in matrices])]
    return m

def overlapShapes(matrix, diagonal=True, multicolor=True, byColor=False, hierarchy=[0,1,2,3,4,5,6,7,8,9]):
    """
    Overlaps shapes of a given type of the same size. If there is a color conflict, a color is chosen 
    acording to the given hierarchy. 
    """
    if not multicolor: 
        if diagonal:   
            shList = [sh for sh in matrix.dShapes]
        else:   
            shList = [sh for sh in matrix.shapes]
    else:
        if diagonal: 
            shList = [sh for sh in matrix.multicolorDShapes]
        else:
            shList = [sh for sh in matrix.multicolorShapes]
    if byColor:
        shList = [sh for sh in matrix.shapesByColor]
    shList = [sh for sh in shList if sh.isRectangle]
    if len(set([sh.shape for sh in shList])) != 1:
        return matrix.m.copy()
    return overlapMatrices([sh.m for sh in shList],hierarchy)

def getCropAttributes(t, diagonal, multicolor, sameColor=True):
    """
    If the task involved cropping a shape, this function returns the unique identifying arguments of that shape. 
    """
    bC = max(0, t.backgroundColor)
    if diagonal and not multicolor:
        if t.nCommonInOutDShapes == 0:
            return set()
        attrs = set.intersection(*[s.inMatrix.getShapeAttributes(backgroundColor=bC,\
                    singleColor=True, diagonals=True)[s.inMatrix.dShapes.index(s.commonDShapes[0][0])] for s in t.trainSamples])
        nonAttrs = set()
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=True, diagonals=True)
            for shi in range(len(s.inMatrix.dShapes)):
                if s.inMatrix.dShapes[shi] == s.commonDShapes[0][0]:
                    continue
                else:
                    nonAttrs = nonAttrs.union(shAttrs[shi])  
    if not diagonal and not multicolor:
        if t.nCommonInOutShapes == 0:
            return set()
        attrs = set.intersection(*[s.inMatrix.getShapeAttributes(backgroundColor=bC,\
                    singleColor=True, diagonals=False)[s.inMatrix.shapes.index(s.commonShapes[0][0])] for s in t.trainSamples])
        nonAttrs = set()
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=True, diagonals=False)
            for shi in range(len(s.inMatrix.shapes)):
                if s.inMatrix.shapes[shi] == s.commonShapes[0][0]:
                    continue
                else:
                    nonAttrs = nonAttrs.union(shAttrs[shi]) 
    if not diagonal and multicolor:
        if not t.outIsInMulticolorShapeSize:
            return set()                               
        attrs = set()
        nonAttrs = set()
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=False, diagonals=False)
            crop = False
            for shi in range(len(s.inMatrix.multicolorShapes)):
                if s.inMatrix.multicolorShapes[shi].shape == s.outMatrix.shape and\
                np.all(np.logical_or(s.inMatrix.multicolorShapes[shi].m == s.outMatrix.m, s.inMatrix.multicolorShapes[shi].m==255)):
                    crop = True
                    if len(attrs) == 0:
                        attrs = shAttrs[shi]
                    attrs = attrs.intersection(shAttrs[shi])
                else:
                    nonAttrs = nonAttrs.union(shAttrs[shi])    
        if not crop:
                return set()
    if diagonal and multicolor:
        if not t.outIsInMulticolorDShapeSize:
            return set()                               
        attrs = set()
        nonAttrs = set()
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=False, diagonals=True)
            crop = False
            for shi in range(len(s.inMatrix.multicolorDShapes)):
                if s.inMatrix.multicolorDShapes[shi].shape == s.outMatrix.shape and\
                np.all(np.logical_or(s.inMatrix.multicolorDShapes[shi].m == s.outMatrix.m, s.inMatrix.multicolorDShapes[shi].m==255)):
                    crop = True
                    if len(attrs) == 0:
                        attrs = shAttrs[shi]
                    attrs = attrs.intersection(shAttrs[shi])
                else:
                    nonAttrs = nonAttrs.union(shAttrs[shi])    
        if not crop:
                return set()
    return(attrs - nonAttrs)
        
def getBestCropShape(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "cropShape" and works best for the training samples.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    bC = max(0, t.backgroundColor)
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,True, False),\
                                                           backgroundColor=bC, singleColor=True, diagonals=True), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,False, False),\
                                                           backgroundColor=bC, singleColor=True, diagonals=False), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,True, True),\
                                                           backgroundColor=bC, singleColor=False, diagonals=True), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,False, True),\
                                                           backgroundColor=bC, singleColor=False, diagonals=False), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    for attr in ['LaSh', 'MoCo', 'MoCl', 'UnSh', 'UnSi']:
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=bC, singleColor=True, diagonals=True), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=bC, singleColor=True, diagonals=False), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=bC, singleColor=False, diagonals=True), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=bC, singleColor=True, diagonals=False), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    return bestFunction
    
def cropShape(matrix, attributes, backgroundColor=0, singleColor=True, diagonals=True, context=False):
    """
    This function crops the shape out of a matrix with the maximum common attributes. If the argument context is passed,
    the pixels lying inside the shape matrix are included. 
    """
    if singleColor: 
        if diagonals:   
            shapeList = [sh for sh in matrix.dShapes]
        else:   
            shapeList = [sh for sh in matrix.shapes]
    else:
        if diagonals: 
            shapeList = [sh for sh in matrix.multicolorDShapes]
        else:
            shapeList = [sh for sh in matrix.multicolorShapes]
    bestShapes = []
    score = 0
    attrList = matrix.getShapeAttributes(backgroundColor, singleColor, diagonals)
    for i in range(len(shapeList)):
        shscore = len(attributes.intersection(attrList[i]))
        if shscore > score:
            score = shscore
            bestShapes = [i]
        elif shscore == score:
            bestShapes += [i]
    if len(bestShapes) == 0:
        return matrix
    if context:
        bestShape = shapeList[bestShapes[0]]
        m = matrix.m[bestShape.position[0]:bestShape.position[0]+bestShape.shape[0], bestShape.position[1]:bestShape.position[1]+bestShape.shape[1]]
        return m.copy() 
    else:
        bestShape = shapeList[bestShapes[0]].m.copy()
        bestShape[bestShape==255]=backgroundColor
    return bestShape
    
def getBestCropReference(t):
    """
    Given a Task t, this function returns the partial function that uses the
    function "cropReference" and works best for the training samples.
    """
    bestFunction = partial(identityM)
    bestScore = 1000
    for sh in t.commonInShapes:
        bestFunction, bestScore = updateBestFunction(t, partial(cropShapeReference, refShape=sh,\
                                refType='subshape', multicolor=True, diagonal=False), bestScore, bestFunction)
    if len(t.commonInShapes) == 1:
        bestFunction, bestScore = updateBestFunction(t, partial(cropShapeReference, refShape=t.commonInShapes[0],\
                                refType='mark', multicolor=False, diagonal=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(cropShapeReference, refShape=t.commonInShapes[0],\
                                refType='mark', multicolor=False, diagonal=False), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(cropShapeReference,\
                                refType='pixels', maxOrMin='min',multicolor=True, diagonal=False), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(cropShapeReference,\
                                refType='pixels', maxOrMin='max',multicolor=True, diagonal=False), bestScore, bestFunction)
    #add referenced by frames
    return bestFunction

def cropShapeReference(matrix, refShape=None, refType='subshape', maxOrMin='max', sameColor=True, multicolor=True, diagonal=False):
    """
    Given a reference type refType, this function returns a referenced shape of the given type. 
    """
    if multicolor:
        if diagonal:
            shList = matrix.multicolorDShapes
        else:
            shList = matrix.multicolorShapes
    else:
        if diagonal:
            shList = matrix.dShapes
        else:
            shList = matrix.shapes
    if refType == 'subshape':
        bestSh, bestScore = None, 0
        for sh in shList:
            if hasattr(sh, 'subshapes') and refShape in sh.subshapes:
                score = np.count_nonzero([refShape == sh2 for sh2 in sh.subshapes])
                if score > bestScore:
                    bestSh = sh
                    bestScore = score
        if bestSh == None:
            return matrix.m.copy()
        return bestSh.m
    elif refType == 'pixels':
        if maxOrMin == 'max':
            bestSh, bestScore = None, 0
        else:
            bestSh, bestScore = None, 1000
        for sh in shList:
            if hasattr(sh, 'subshapes'):
                score = len([p for p in sh.subshapes if len(p.pixels) == 1])
                if maxOrMin == 'max' and score > bestScore:
                    bestSh = sh
                    bestScore = score
                if maxOrMin == 'min' and score < bestScore:
                    bestSh = sh
                    bestScore = score
        if bestSh == None:
            return matrix.m.copy()
        return bestSh.m              
    elif refType == 'frame':
        return matrix.m.copy()
    elif refType == 'mark':
        foundRef = False
        for sh in shList:
            if sh == refShape:
                refShape = sh
                foundRef = True
                break
        if not foundRef:
            return matrix.m
        #otherwise return closest to reference
        bestShape = None
        dist = 1000
        refPos = refShape.position
        refPixels = [(p[0]+refPos[0], p[1]+refPos[1]) for p in refShape.pixels]
        for sh in shList:
            if sh == refShape or sh.color == matrix.backgroundColor:
                continue
            for p2 in [(p[0]+sh.position[0], p[1]+sh.position[1]) for p in sh.pixels]:
                if min(abs((p[0]-p2[0]))+abs((p[1]-p2[1])) for p in refPixels) < dist:
                    bestShape = sh
                    dist = min(abs((p[0]-p2[0])+(p[1]-p2[1])) for p in refPixels)
        if bestShape == None:
            return matrix.m.copy()
        bestShape=bestShape.m
        bestShape[bestShape==255]=matrix.backgroundColor
        return bestShape         

def cropAllBackground(matrix):
    """
    Deletes external rows and columns as long as they are colored fully by the background color. 
    """
    m = matrix.m.copy()
    bC = matrix.backgroundColor
    if np.all(m == bC):
        return m
    x1, x2, y1, y2 = 0, m.shape[0]-1, 0, m.shape[1]-1
    while x1 <= x2 and np.all(m[x1,:] == bC):
        x1 += 1
    while x2 >= x1 and np.all(m[x2,:] == bC):
        x2 -= 1
    while y1 <= y2 and np.all(m[:,y1] == bC):
        y1 += 1
    while y2 >= y1 and np.all(m[:,y2] == bC):
        y2 -= 1
    return(m[x1:x2+1,y1:y2+1])

def cropOnlyMulticolorShape(matrix, diagonals=False):
    """
    This function is supposed to be called if there is one and only one 
    multicolor shape in all the input samples. This function just returns it.
    """
    if diagonals:
        m = matrix.multicolorDShapes[0].m.copy()
        m[m==255] = matrix.multicolorDShapes[0].background
    else:
        m = matrix.multicolorShapes[0].m.copy()
        m[m==255] = matrix.multicolorShapes[0].background
    return m

def cropFullFrame(matrix, includeBorder=True, bigOrSmall = None):
    m = matrix.m.copy()
    if bigOrSmall == None and len(matrix.fullFrames) != 1:
        return m
    if bigOrSmall == "small":
        frame = matrix.fullFrames[-1]
    else:
        frame = matrix.fullFrames[0]
    if includeBorder:
        return m[frame.position[0]:frame.position[0]+frame.shape[0], \
                 frame.position[1]:frame.position[1]+frame.shape[1]]
    else:
        return m[frame.position[0]+1:frame.position[0]+frame.shape[0]-1, \
                 frame.position[1]+1:frame.position[1]+frame.shape[1]-1]
        
def cropPartialFrame(matrix, includeBorder=True, bigOrSmall = None):
    """
    Crop the unique partial frame of a matrix. Options to include the border of the frame,
    or to choose the frame according to its size. 
    """
    m = matrix.m.copy()
    if len(matrix.partialFrames) == 0:
        return m
    if bigOrSmall == "small":
        frame = matrix.partialFrames[-1]
    else:
        frame = matrix.partialFrames[0]
    if includeBorder:
        return m[frame.position[0]:frame.position[0]+frame.shape[0], \
                 frame.position[1]:frame.position[1]+frame.shape[1]]
    else:
        return m[frame.position[0]+1:frame.position[0]+frame.shape[0]-1, \
                 frame.position[1]+1:frame.position[1]+frame.shape[1]-1]
        
def getBestTwoShapeFunction(t):
    """
    Inputs a task with two input shapes and tries a series of operations that make sense.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    multicolor = t.twoShapeTask[1]
    diagonal = t.twoShapeTask[2]
    typ = t.twoShapeTask[-1]
    cropAfter = [0] 
    if t.outSmallerThanIn:
        cropAfter += [1,2]
    #try possible operations
    for crop in cropAfter:
        for flip in [True,False]:
            if t.twoShapeTask[3]:
                #pixelwise and/or
                for c in permutations(t.totalOutColors,2):
                    bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(pixelwiseAnd, falseColor=c[0],\
                                targetColor=None,trueColor=c[1]), diagonal=diagonal,typ=typ, multicolor=multicolor, crop=crop, flip=flip), bestScore, bestFunction)
                    bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(pixelwiseOr, falseColor=c[0],\
                                targetColor=None,trueColor=c[1]), diagonal=diagonal,typ=typ, multicolor=multicolor, crop=crop, flip=flip), bestScore, bestFunction)
                #print shapes            
                for base in [0,1]:                
                    for bC in t.commonInColors:          
                        bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(printShapes, base=base,\
                                backgroundColor=bC),diagonal=diagonal,typ=typ, multicolor=multicolor, crop=crop, flip=flip), bestScore, bestFunction)
                #overlap matrices
                if typ > 1:
                    bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(overlapMatrices,\
                                colorHierarchy=[0,1,2,3,4,5,6,7,8,9]),diagonal=diagonal,typ=typ, multicolor=multicolor, crop=crop, flip=flip), bestScore, bestFunction)
            else:
                for c in permutations(t.totalOutColors,2):
                    for target in [None]:
                        bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(pixelwiseAnd, falseColor=c[0],\
                                    targetColor=target,trueColor=c[1]), diagonal=diagonal, multicolor=multicolor,typ=typ, crop=crop, flip=flip, downsizeToSmallest=True), bestScore, bestFunction)
                        bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(pixelwiseOr, falseColor=c[0],\
                                    targetColor=target,trueColor=c[1]), diagonal=diagonal, multicolor=multicolor,typ=typ, crop=crop, flip=flip, downsizeToSmallest=True), bestScore, bestFunction)
                        bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(pixelwiseAnd, falseColor=c[0],\
                                    targetColor=target,trueColor=c[1]), diagonal=diagonal, multicolor=multicolor,typ=typ, crop=crop, flip=flip, scaleToLargest=True), bestScore, bestFunction)
                        bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(pixelwiseOr, falseColor=c[0],\
                                    targetColor=target,trueColor=c[1]), diagonal=diagonal, multicolor=multicolor,typ=typ, crop=crop, flip=flip, scaleToLargest=True), bestScore, bestFunction)  
                for base in [0,1]:
                    for bC in t.commonInColors:
                        bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(printShapes, base=base,\
                                backgroundColor=bC),diagonal=diagonal, multicolor=multicolor, crop=crop,typ=typ, flip=flip, downsizeToSmallest=True), bestScore, bestFunction)
                        bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(printShapes, base=base,\
                                backgroundColor=bC),diagonal=diagonal, multicolor=multicolor, crop=crop,typ=typ, flip=flip, scaleToLargest=True), bestScore, bestFunction)
            #multiply matrices
            for c in [0,1]:
                for b in [0,1]:
                    bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(multiplyMatrices, base=b,\
                                background=0, color=c), typ=typ, crop=crop, flip=flip), bestScore, bestFunction)
                    bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(multiplyMatrices, base=b,\
                                background=0, color=c), typ=typ, crop=crop, flip=flip, downsizeToSmallest=True), bestScore, bestFunction)
               
    return bestFunction

def twoShapeFun(matrix, f=partial(identityM), typ=1, diagonal=True, multicolor=True,\
                flip=False, rotate=False, downsizeToSmallest=False, scaleToLargest=False, crop=False):
    """
    Apply function f to the shapes of matrix. By default, this function should only be called when matrix has two shapes.
    The arguments downsizeToSmallest of scaleToLargest match the size of the shapes if possible. The argument crop returns
    the result of the operation without context. The arguments flip and rotate modify the shapes before the operations are 
    performed. 
    """
    def cropAllBackgroundM(m, background):
        if np.all(m == background):
            return m
        x1, x2, y1, y2 = 0, m.shape[0]-1, 0, m.shape[1]-1
        while x1 <= x2 and np.all(m[x1,:] == background):
            x1 += 1
        while x2 >= x1 and np.all(m[x2,:] == background):
            x2 -= 1
        while y1 <= y2 and np.all(m[:,y1] == background):
            y1 += 1
        while y2 >= y1 and np.all(m[:,y2] == background):
            y2 -= 1
        return(m[x1:x2+1,y1:y2+1])
                                            
    def minimizeM(m):
        x = 1
        for i in range(1, m.shape[0]):
            if np.array_equal(m[x,:],m[x-1,:]):
                m = np.delete(m, (x), axis=0)
            else:
                x+=1
        x = 1
        for i in range(1, m.shape[1]):
            if np.array_equal(m[:,x],m[:,x-1]):
                m = np.delete(m, (x), axis=1)
            else:
                x+=1
        return m
                                            
    def scaleM(m, s):
        sc = min(s[0]//m.shape[0], s[1]//m.shape[1])
        sm = np.repeat(np.repeat(m, sc, axis=1), sc, axis=0)
        return sm
                                            
    m = matrix.m.copy()
    if typ == 1:
        if multicolor:
            if diagonal: 
                shList = [sh for sh in matrix.multicolorDShapes]
            else:
                shList = [sh for sh in matrix.multicolorShapes]
        else: 
            if diagonal:   
                shList = [sh for sh in matrix.dShapes]
            else:   
                shList = [sh for sh in matrix.shapes]
        posList = [sh.position for sh in shList]
        mList = [sh.m.copy() for sh in shList]
    elif typ == 2:
        if not hasattr(matrix,'grid'):
            return m
        mList = [c[0].m for c in matrix.grid.cellList]
        posList = [c[1] for c in matrix.grid.cellList]
    elif typ == 3:
        if matrix.shape[0] == 2*matrix.shape[1]:
            mList = [matrix.m[:matrix.shape[0]//2,:].copy(), matrix.m[matrix.shape[0]//2:,:].copy()]
            posList = [(0,0),(matrix.shape[1],0)]
        elif matrix.shape[1] == 2*matrix.shape[0]:
            mList = [matrix.m[:,:matrix.shape[1]//2].copy(), matrix.m[:,matrix.shape[1]//2:].copy()]
            posList = [(0,0),(0,matrix.shape[0])]
        else:
            return m
        mList.sort(key=lambda x: len(np.unique(x))) 
    if len(mList)  != 2:
        return m    
    if flip:
        mList[1] = np.fliplr(mList[1])
    #sort list largest first
    if mList[0].shape[0]*mList[0].shape[1]<mList[1].shape[0]*mList[1].shape[1]:
        mList = mList[::-1]
        posList = posList[::-1]
    if downsizeToSmallest:
        mList[0] = minimizeM(mList[0])
    if scaleToLargest:
        mList[1] = scaleM(mList[1], mList[0].shape)
    if mList[0].shape != mList[1].shape and f.func.__name__ != 'multiplyMatrices':
        return m
    if  f.func.__name__ == 'multiplyMatrices':
        return f(mList)
    elif crop == 1:
        return(f(mList))
    elif crop == 2:
        return cropAllBackgroundM(f(mList),matrix.backgroundColor)
    else:
        if scaleToLargest:
            pos = posList[0]
            sh = mList[0]
            m[pos[0]:pos[0] + sh.shape[0], pos[1]:pos[1] + sh.shape[1]] = f(mList)                
        elif downsizeToSmallest:
            pos = posList[1]
            sh = mList[1]
            m[pos[0]:pos[0] + sh.shape[0], pos[1]:pos[1] + sh.shape[1]] = f(mList)   
        return m
    
###############################################################################
###############################################################################
# %% Main function: getPossibleOperations
def getPossibleOperations(t, c):
    """
    Given a Task t and a Candidate c, this function returns a list of all the
    possible operations that make sense applying to the input matrices of c.
    The elements of the list to be returned are partial functions, whose input
    is a Matrix and whose output is a numpy.ndarray (2-dim matrix).
    """ 
    candTask = c.t
    x = [] # List to be returned
    
    ###########################################################################
    # Fill the blanks
    if t.fillTheBlank:
        params = fillTheBlankParameters(t)
        x.append(partial(fillTheBlank, params=params))
        
    # switchColors
    if all([n==2 for n in candTask.nInColors]):
        x.append(partial(switchColors))
        
    # downsize
    if candTask.sameOutShape:
        outShape = candTask.outShape
        x.append(partial(downsize, newShape=outShape))
        x.append(partial(downsizeMode,newShape=outShape))
        if t.backgroundColor!=-1:
            x.append(partial(downsize, newShape=outShape, falseColor=t.backgroundColor))
        #if candTask.sameOutDummyMatrix and candTask.backgroundColor != -1:
        #    x.append(partial(arrangeShapes,outDummyMatrix=candTask.trainSamples[0].outMatrix.dummyMatrix,\
        #                     outDummyColor=candTask.trainSamples[0].outMatrix.backgroundColor))
    x.append(getBestCountColors(candTask))  
    x.append(getBestCountShapes(candTask))  
    ###########################################################################
    # sameIOShapes
    if candTask.sameIOShapes:
        
        #######################################################################
        
        # ColorMap
        ncc = len(candTask.colorChanges)
        if len(set([cc[0] for cc in candTask.colorChanges])) == ncc and ncc != 0:
            x.append(partial(colorMap, cMap=dict(candTask.colorChanges)))
            
        x.append(partial(revertColorOrder))
            
        # Symmetrize
        if all([len(x)==1 for x in candTask.changedInColors]):
            color = next(iter(candTask.changedInColors[0]))
            axis = []
            if candTask.lrSymmetric:
                axis.append("lr")
            if candTask.udSymmetric:
                axis.append("ud")
            if all([sample.inMatrix.shape[0]==sample.inMatrix.shape[1] for sample in candTask.testSamples]):
                if candTask.d1Symmetric:
                    axis.append("d1")
                if candTask.d2Symmetric:
                    axis.append("d2")
            x.append(partial(symmetrize, axis=axis, color=color))
            if candTask.totalOutColors==1:
                for fc in candTask.fixedColors:
                    x.append(partial(symmetrize, axis=axis, refColor=fc,\
                                     outColor=next(iter(candTask.totalOutColors))))
    
        # Color symmetric pixels
        x.append(getBestColorSymmetricPixels(candTask))
    
        # Complete rectangles
        if candTask.backgroundColor!=-1 and len(candTask.fixedColors)==1 and \
        len(candTask.colorChanges)==1:
            sc = next(iter(candTask.fixedColors))
            nc = next(iter(candTask.colorChanges))[1]
            x.append(partial(completeRectangles, sourceColor=sc, newColor=nc))
        
        x.append(partial(deletePixels, diagonals=True))
        x.append(partial(deletePixels, diagonals=False))
        
        #######################################################################
        # For LinearShapeModel we need to have the same shapes in the input
        # and in the output, and in the exact same positions.
        # This model predicts the color of the shape in the output.
        
        if candTask.onlyShapeColorChanges:
            ccwf = getColorChangesWithFeatures(candTask)
            fsf = candTask.fixedShapeFeatures
            x.append(partial(changeShapesWithFeatures, ccwf=ccwf, fixedColors=candTask.fixedColors,\
                             fixedShapeFeatures=fsf))
                
            if all(["getBestLSTM" not in str(op.func) for op in c.ops]):        
                x.append(getBestLSTM(candTask))
            
            # Other deterministic functions that change the color of shapes.
            for cc in candTask.commonColorChanges:
                for border, bs in product([True, False, None], ["big", "small", None]):
                    x.append(partial(changeShapes, inColor=cc[0], outColor=cc[1],\
                                     bigOrSmall=bs, isBorder=border))
                    
            if isReplicateTask(candTask)[0]:
                x.append(getBestReplicateShapes(candTask))
            x.append(getBestColorByPixels(candTask))
            
            return x
        
        #######################################################################
        # Complete row/col patterns
        colStep=None
        rowStep=None
        if candTask.followsRowPattern:
            if candTask.allEqual(candTask.rowPatterns):
                rowStep=candTask.rowPatterns[0]
        if candTask.followsColPattern:
            if candTask.allEqual(candTask.colPatterns):
                colStep=candTask.colPatterns[0]
        if candTask.allEqual(candTask.changedInColors) and len(candTask.changedInColors[0])==1:
            c2c = next(iter(candTask.changedInColors[0]))
        else:
            c2c=None
                
        if candTask.followsRowPattern and candTask.followsColPattern:
            x.append(partial(followPattern, rc="both", colorToChange=c2c,\
                             rowStep=rowStep, colStep=colStep))
        elif candTask.followsRowPattern:
            x.append(partial(followPattern, rc="row", colorToChange=c2c,\
                             rowStep=rowStep, colStep=colStep))
        elif candTask.followsColPattern:
            x.append(partial(followPattern, rc="col", colorToChange=c2c,\
                             rowStep=rowStep, colStep=colStep))

        #######################################################################
        # CNNs
        
        #x.append(getBestCNN(candTask))
        #if candTask.sameNSampleColors and all(["predictCNN" not in str(op.func) for op in c.ops]):
        #    x.append(getBestSameNSampleColorsCNN(candTask))

        """
        if t.backgroundColor != -1:
            model = trainCNNDummyColor(candTask, 5, -1)
            x.append(partial(predictCNNDummyColor, model=model))
            model = trainCNNDummyColor(candTask, 3, 0)
            x.append(partial(predictCNNDummyColor, model=model))
            #model = trainOneConvModelDummyColor(candTask, 7, -1)
            #x.append(partial(predictConvModelDummyColor, model=model))
        """
            
        #cc = list(t.commonSampleColors)
        #model = trainCNNDummyCommonColors(t, cc, 3, -1)
        #x.append(partial(predictCNNDummyCommonColors, model=model,\
        #                commonColors=cc))
        
        #######################################################################
        # Transformations if the color count is always the same:
        # Rotations, Mirroring, Move Shapes, Mirror Shapes, ...
        if candTask.sameColorCount:
            for axis in ["lr", "ud"]:
                x.append(partial(mirror, axis = axis))
            # You can only mirror d1/d2 or rotate if the matrix is squared.
            if candTask.inMatricesSquared:
                for axis in ["d1", "d2"]:
                    x.append(partial(mirror, axis = axis))
                for angle in [90, 180, 270]:
                    x.append(partial(rotate, angle = angle))
                
                                                         
            # Mirror shapes
            x.append(getBestFlipAllShapes(candTask))
                                
        #######################################################################
        # Other sameIOShapes functions
        # Move shapes
        x.append(getBestMoveShapes(candTask, candidate=c))
        
        pr = pixelRecolor(candTask)
        if len(pr)!=1:
            x.append(partial(executePixelRecolor, Best_Dict=pr[0], Best_v=pr[1], Best_Q1=pr[2], Best_Q2=pr[3]))
        
        fun = getPixelChangeCriteria(candTask)
        if fun != 0:
            x.append(fun)
            
        # extendColor
        x.append(getBestExtendColor(candTask))
        
        # surround shapes
        x.append(getBestSurroundShapes(candTask))
        
        # Paint shapes in half
        x.append(getBestPaintShapesInHalf(candTask))
        
        # Paint shapes from border color
        x.append(getBestPaintShapeFromBorderColor(candTask))
            
        # fillRectangleInside
        for cic in candTask.commonChangedInColors:
            for coc in candTask.commonChangedOutColors:
                x.append(partial(fillRectangleInside, rectangleColor=cic, fillColor=coc))
        
        # Color longest lines
        if len(candTask.colorChanges)==1:
            change = next(iter(candTask.colorChanges))
            x.append(partial(colorLongestLines, cic=change[0], coc=change[1], direction='h'))
            x.append(partial(colorLongestLines, cic=change[0], coc=change[1], direction='v'))
            x.append(partial(colorLongestLines, cic=change[0], coc=change[1], direction='hv'))
            x.append(partial(colorLongestLines, cic=change[0], coc=change[1], direction='d'))

        # Connect Pixels
        x.append(partial(connectAnyPixels))
        x.append(partial(connectAnyPixels, diagonal=True))
        if all([len(x)==1 for x in candTask.changedInColors]):
            x.append(partial(connectAnyPixels, connColor=next(iter(candTask.changedOutColors[0]))))
            x.append(partial(connectAnyPixels, connColor=next(iter(candTask.changedOutColors[0])), \
                             diagonal=True))

        fc = candTask.fixedColors
        #if hasattr(t, "fixedColors"):
        #    tfc = candTask.fixedColors
        #else:
        #    tfc = set()
        for pc in candTask.colors - candTask.commonChangedInColors:
            for cc in candTask.commonChangedOutColors:
                x.append(partial(connectAnyPixels, pixelColor=pc, \
                                 connColor=cc, fixedColors=fc))
                x.append(partial(connectAnyPixels, pixelColor=pc, \
                                 connColor=cc, fixedColors=fc, diagonal=True))
        for pc in candTask.colors - candTask.commonChangedInColors:
            x.append(partial(connectAnyPixels, pixelColor=pc, allowedChanges=dict(candTask.colorChanges)))
            x.append(partial(connectAnyPixels, pixelColor=pc, allowedChanges=dict(candTask.colorChanges), \
                             diagonal=True))
            x.append(partial(connectAnyPixels, pixelColor=pc, allowedChanges=dict(candTask.colorChanges),\
                             lineExclusive=True))
            x.append(partial(connectAnyPixels, pixelColor=pc, allowedChanges=dict(candTask.colorChanges),\
                             lineExclusive=True, diagonal=True))
                
        for cc in candTask.commonColorChanges:
            for cc in candTask.commonColorChanges:
                for border, bs in product([True, False, None], ["big", "small", None]):
                    x.append(partial(changeShapes, inColor=cc[0], outColor=cc[1],\
                                     bigOrSmall=bs, isBorder=border))
        
        if all(len(set([sh.shape for sh in s.outMatrix.shapes]))==1 for s in candTask.trainSamples):
            x.append(partial(subMatToLayer,shapeAndDict=getLayerDict(candTask)))
        #replicate/symmterize/other shape related tasks
        #x.append(getBestAlignShapes(candTask))
        x.append(getBestSymmetrizeSubmatrix(candTask))
        x.append(partial(replicateShapes,diagonal=True, multicolor=True, allCombs=False,anchorType='subframe', scale=False))
        x.append(partial(replicateShapes,diagonal=True, multicolor=True, allCombs=True,anchorType='subframe', scale=False, deleteOriginal=True))
        if isReplicateTask(candTask)[0]:
            x.append(getBestReplicateShapes(candTask))
        #x.append(getBestReplicateOneShape(candTask))
        x.append(getBestSymmetrizeAllShapes(candTask))
        x.append(getBestColorByPixels(candTask))
        
        #if len(candTask.colorChanges) == 1:
        #    x.append(partial(replicateShapes,diagonal=True, multicolor=False, allCombs=True,\
        #                     anchorColor = list(candTask.colorChanges)[0][0], anchorType='all', attributes=set(['UnCo'])))

        #delete shapes
        if isDeleteTask(candTask) and all(t.backgroundColor == c[1] for c in candTask.colorChanges):
            x.append(getBestDeleteShapes(candTask, True, True))
                    
    ###########################################################################
    # Cases in which the input has always the same shape, and the output too
    if candTask.sameInShape and candTask.sameOutShape and \
    all(candTask.trainSamples[0].inMatrix.shape == s.inMatrix.shape for s in candTask.testSamples):
            
        if candTask.sameNSampleColors:
            #Cases where output colors are a subset of input colors
            if candTask.sameNInColors and all(s.inHasOutColors for s in candTask.trainSamples):
                if hasattr(candTask, 'outShapeFactor') or (hasattr(candTask,\
                              'gridCellIsOutputShape') and candTask.gridCellIsOutputShape):
                    ch = dict(sum([Counter(s.outMatrix.colorCount) for s in candTask.trainSamples],Counter()))
                    ch = sorted(ch, key=ch.get)
                    if candTask.backgroundColor in ch:
                        ch.remove(candTask.backgroundColor)
                    ch = list(set([0,1,2,3,4,5,6,7,8,9]).difference(set(ch))) + ch
                    if hasattr(candTask, 'outShapeFactor'):
                        x.append(partial(overlapSubmatrices, colorHierarchy=ch, shapeFactor=candTask.outShapeFactor))
                    else:
                        x.append(partial(overlapSubmatrices, colorHierarchy=ch))
        
        pixelMap = pixelCorrespondence(candTask)
        if len(pixelMap) != 0:
            x.append(partial(mapPixels, pixelMap=pixelMap, outShape=candTask.outShape))
    
    ###########################################################################
    # Evolve
    if candTask.sameIOShapes and all([len(x)==1 for x in candTask.changedInColors]) and\
    len(candTask.commonChangedInColors)==1 and candTask.sameNSampleColors:
        x.append(getBestEvolve(candTask))
        #cfn = evolve(candTask)
        #x.append(getBestEvolve(candTask, cfn))
    #    x.append(partial(applyEvolve, cfn=cfn, nColors=candTask.trainSamples[0].nColors,\
    #                     kernel=5, nIterations=1))
    
    if candTask.sameIOShapes and all([len(x)==1 for x in candTask.changedInColors]) and\
    len(candTask.commonChangedInColors)==1:
        x.append(getBestEvolvingLines(candTask))
        
    ###########################################################################
    # Other cases
    
    if candTask.sameIOShapes and len(candTask.fixedColors)!=0:
        for color in candTask.fixedColors:
            for outColor in candTask.commonOutColors:
                x.append(partial(paintCrossedCoordinates, refColor=color,\
                                 outColor=outColor, fixedColors=candTask.fixedColors))
    
    if candTask.inSmallerThanOut and t.inSmallerThanOut:
        x.append(getBestExtendMatrix(candTask))
        
    if hasattr(candTask, 'inShapeFactor'):
        x.append(partial(multiplyPixels, factor=candTask.inShapeFactor))
        x.append(partial(multiplyMatrix, factor=candTask.inShapeFactor))
        
        for c in candTask.commonSampleColors:
            x.append(partial(multiplyPixelsAndAnd, factor=candTask.inShapeFactor,\
                             falseColor=c))
            x.append(partial(multiplyPixelsAndOr, factor=candTask.inShapeFactor,\
                             falseColor=c))
            x.append(partial(multiplyPixelsAndXor, factor=candTask.inShapeFactor,\
                             falseColor=c))
            
        if type(candTask.inShapeFactor)==tuple:
            ops = getBestMosaic(candTask)
            x.append(partial(generateMosaic, ops=ops, factor=candTask.inShapeFactor))
            
        if all([s.inMatrix.shape[0]**2 == s.outMatrix.shape[0] and \
                s.inMatrix.shape[1]**2 == s.outMatrix.shape[1] for s in candTask.trainSamples]):
            totalColorCount = Counter()
            for sample in t.trainSamples:
                for color in sample.outMatrix.colorCount.keys():
                    totalColorCount[color] += sample.outMatrix.colorCount[color]
            falseColor = max(totalColorCount.items(), key=operator.itemgetter(1))[0]
            opCond = getBestMultiplyMatrix(candTask, falseColor)
            x.append(partial(doBestMultiplyMatrix, opCond=opCond, falseColor=falseColor))
            
            
    if hasattr(candTask, 'outShapeFactor'):
        if outputIsSubmatrix(candTask):
            for color in range(10):
                x.append(partial(selectSubmatrixWithMaxColor, color=color, outShapeFactor=candTask.outShapeFactor))
                x.append(partial(selectSubmatrixWithMinColor, color=color, outShapeFactor=candTask.outShapeFactor))
            x.append(partial(selectSubmatrixWithMostColors, outShapeFactor=candTask.outShapeFactor))
            x.append(partial(selectSubmatrixWithLeastColors, outShapeFactor=candTask.outShapeFactor))
            position = getBestSubmatrixPosition(candTask, outShapeFactor=candTask.outShapeFactor)
            x.append(partial(selectSubmatrixInPosition, position=position, outShapeFactor=candTask.outShapeFactor))
        
        # Pixelwise And
        for c in candTask.commonOutColors:
            x.append(partial(pixelwiseAndInSubmatrices, factor=candTask.outShapeFactor,\
                             falseColor=c))
        if len(candTask.totalOutColors) == 2:
            for target in candTask.totalInColors:
                for c in permutations(candTask.totalOutColors, 2):
                    x.append(partial(pixelwiseAndInSubmatrices, \
                                     factor=candTask.outShapeFactor, falseColor=c[0],\
                                     targetColor=target, trueColor=c[1]))
        
        # Pixelwise Or
        for c in candTask.commonOutColors:
            x.append(partial(pixelwiseOrInSubmatrices, factor=candTask.outShapeFactor,\
                             falseColor=c))
        if len(candTask.totalOutColors) == 2:
            for target in candTask.totalInColors:
                for c in permutations(candTask.totalOutColors, 2):
                    x.append(partial(pixelwiseOrInSubmatrices, \
                                     factor=candTask.outShapeFactor, falseColor=c[0],\
                                     targetColor=target, trueColor=c[1]))
        if candTask.backgroundColor!=-1:
            colors = candTask.commonOutColors - candTask.commonInColors
            if candTask.outShapeFactor[0]*candTask.outShapeFactor[1]==len(colors):
                for c in permutations(colors, len(colors)):
                    x.append(partial(pixelwiseOrInSubmatrices, factor=candTask.outShapeFactor,\
                                     falseColor=candTask.backgroundColor,\
                                     trueValues = c))
        
        # Pixelwise Xor
        if candTask.outShapeFactor in [(2,1), (1,2)]:
            for c in candTask.commonOutColors:
                x.append(partial(pixelwiseXorInSubmatrices, factor=candTask.outShapeFactor,\
                                 falseColor=c))
            if len(candTask.commonOutColors - candTask.commonInColors)==2 and\
            candTask.backgroundColor!=-1:
                colors = candTask.commonOutColors - candTask.commonInColors
                for c in permutations(colors, 2):
                    x.append(partial(pixelwiseXorInSubmatrices, falseColor=candTask.backgroundColor,\
                                     firstTrue=c[0], secondTrue=c[1]))
            if len(candTask.totalOutColors) == 2:
                for target in candTask.totalInColors:
                    for c in permutations(candTask.totalOutColors, 2):
                        x.append(partial(pixelwiseXorInSubmatrices, \
                                         factor=candTask.outShapeFactor, falseColor=c[0],\
                                         targetColor=target, trueColor=c[1]))
    
    if hasattr(candTask, 'gridCellIsOutputShape') and candTask.gridCellIsOutputShape:
        if outputIsSubmatrix(candTask, isGrid=True):
            for color in range(10):
                x.append(partial(selectSubmatrixWithMaxColor, color=color, isGrid=True))
                x.append(partial(selectSubmatrixWithMinColor, color=color, isGrid=True))
            x.append(partial(selectSubmatrixWithMostColors, isGrid=True))
            x.append(partial(selectSubmatrixWithLeastColors, isGrid=True))
            position = getBestSubmatrixPosition(candTask, isGrid=True)
            x.append(partial(selectSubmatrixInPosition, position=position, isGrid=True))
        
        # Pixelwise And
        for c in candTask.commonOutColors:
            x.append(partial(pixelwiseAndInGridSubmatrices, falseColor=c))
        if len(candTask.totalOutColors) == 2:
            for target in candTask.totalInColors:
                for c in permutations(candTask.totalOutColors, 2):
                    x.append(partial(pixelwiseAndInGridSubmatrices, falseColor=c[0],\
                                     targetColor=target, trueColor=c[1]))
                        
        # Pixelwise Or
        for c in candTask.commonOutColors:
            x.append(partial(pixelwiseOrInGridSubmatrices, falseColor=c))
        if len(candTask.totalOutColors) == 2:
            for target in candTask.totalInColors:
                for c in permutations(candTask.totalOutColors, 2):
                    x.append(partial(pixelwiseOrInGridSubmatrices, falseColor=c[0],\
                                     targetColor=target, trueColor=c[1]))
        if candTask.backgroundColor!=-1:
            colors = candTask.commonOutColors - candTask.commonInColors
            if candTask.trainSamples[0].inMatrix.grid.nCells==len(colors):
                for c in permutations(colors, len(colors)):
                    x.append(partial(pixelwiseOrInGridSubmatrices,\
                                     falseColor=candTask.backgroundColor,\
                                     trueValues = c))
        
        # Pixelwise Xor
        if all([s.inMatrix.grid.nCells == 2 for s in candTask.trainSamples]) \
        and all([s.inMatrix.grid.nCells == 2 for s in candTask.testSamples]):
            for c in candTask.commonOutColors:
                x.append(partial(pixelwiseXorInGridSubmatrices, falseColor=c))
            if len(candTask.commonOutColors - candTask.commonInColors)==2 and\
            candTask.backgroundColor!=-1:
                colors = candTask.commonOutColors - candTask.commonInColors
                for c in permutations(colors, 2):
                    x.append(partial(pixelwiseXorInGridSubmatrices, falseColor=candTask.backgroundColor,\
                                     firstTrue=c[0], secondTrue=c[1]))
            if len(candTask.totalOutColors) == 2:
                for target in candTask.totalInColors:
                    for c in permutations(candTask.totalOutColors, 2):
                        x.append(partial(pixelwiseXorInGridSubmatrices, falseColor=c[0],\
                                         targetColor=target, trueColor=c[1]))
       
    if candTask.inputIsGrid:
        if all([s.inMatrix.grid.shape==s.outMatrix.shape for s in candTask.trainSamples]):
            for times in range(1, 6):
                x.append(partial(colorAppearingXTimes, times=times))
            x.append(partial(maxColorFromCell))
               
    x.append(getBestLayShapes(candTask)) 
    #x.append(getBestReplicateOneShape(candTask))
    #tasks with two shapes
    if candTask.twoShapeTask[0]:
        x.append(getBestTwoShapeFunction(t))
    # Cropshape
    #x.append(partial(colorByPixels))
    #x.append(partial(colorByPixels, colorMap=True))
    #x.append(partial(colorByPixels, oneColor=True))
    x.append(partial(overlapShapes))
    if candTask.outSmallerThanIn:
        #x.append(getBestAlignShapes(candTask))
        #x.append(partial(deleteShapes, attributes = getDeleteAttributes(candTask, diagonal = False), diagonal = False, multicolor=False))
        #x.append(partial(replicateShapes, allCombs=True, scale=False,attributes=set(['MoCl']),anchorType='subframe',deleteOriginal=True))
        #x.append(partial(replicateShapes, allCombs=False, scale=True,attributes=set(['MoCl']),anchorType='subframe',deleteOriginal=True))
        #x.append(getBestArrangeShapes(candTask))
        if candTask.backgroundColor!=-1:
            x.append(partial(cropAllShapes, background=candTask.backgroundColor, diagonal=True))
            x.append(partial(cropAllShapes, background=candTask.backgroundColor, diagonal=False))
        
        bestCrop = getBestCropShape(candTask)
        if 'attributes' in bestCrop.keywords.keys():
            for attr in bestCrop.keywords['attributes']:
                if isinstance(attr, str) and attr[:2] == 'mo' and len(bestCrop.keywords['attributes']) > 1:
                    continue
                newCrop = copy.deepcopy(bestCrop)
                newCrop.keywords['attributes'] = set([attr])
                x.append(newCrop)
                
        x.append(getBestCropReference(candTask))  
    
        for attrs in [set(['LaSh']),set(['UnCo'])]:
            x.append(partial(cropShape, attributes=attrs, backgroundColor=max(0,candTask.backgroundColor),\
                                 singleColor=True, diagonals=True)) 
            x.append(partial(cropShape, attributes=attrs, backgroundColor=max(0,candTask.backgroundColor),\
                                 singleColor=True, diagonals=True, context=True)) 
        if candTask.outIsInMulticolorShapeSize:
            x.append(getBestMoveToPanel(candTask))
    if all([len(s.inMatrix.multicolorShapes)==1 for s in candTask.trainSamples+candTask.testSamples]):
        x.append(partial(cropOnlyMulticolorShape, diagonals=False))
    if all([len(s.inMatrix.multicolorDShapes)==1 for s in candTask.trainSamples+candTask.testSamples]):
        x.append(partial(cropOnlyMulticolorShape, diagonals=True))
    if all([len(sample.inMatrix.fullFrames)==1 for sample in candTask.trainSamples+candTask.testSamples]):
        x.append(partial(cropFullFrame))
        x.append(partial(cropFullFrame, includeBorder=False))
    if all([len(sample.inMatrix.fullFrames)>1 for sample in candTask.trainSamples+candTask.testSamples]):
        x.append(partial(cropFullFrame, bigOrSmall="big"))
        x.append(partial(cropFullFrame, bigOrSmall="small"))
        x.append(partial(cropFullFrame, bigOrSmall="big", includeBorder=False))
        x.append(partial(cropFullFrame, bigOrSmall="small", includeBorder=False))
    
    #more frames
    if candTask.hasPartialFrame:
        x.append(getBestFitToFrame(candTask))
        if candTask.outSmallerThanIn:
            x.append(partial(cropPartialFrame, includeBorder=False))
            x.append(partial(cropPartialFrame, includeBorder=True))
    
    if candTask.sameIOShapes:        
        if candTask.sameNSampleColors and all(["predictCNN" not in str(op.func) for op in c.ops]):
            x.append(getBestSameNSampleColorsCNN(candTask))
    
    # startOps

    x.append(partial(paintGridLikeBackground))
    x.append(partial(cropAllBackground))
    
    # minimize
    if not candTask.sameIOShapes:
        x.append(partial(minimize))
    
    return x

###############################################################################
###############################################################################
# Submission Setup
    
class Candidate():
    """
    Objects of the class Candidate store the information about a possible
    candidate for the solution.

    ...
    Attributes
    ----------
    ops: list
        A list containing the operations to be performed to the input matrix
        in order to get to the solution. The elements of the list are partial
        functions (from functools.partial).
    score: int
        The score of the candidate. The score is defined as the sum of the
        number incorrect pixels when applying ops to the input matrices of the
        train samples of the task.
    tasks: list
        A list containing the tasks (in its original format) after performing
        each of the operations in ops, starting from the original inputs.
    t: Task
        The Task object corresponding to the current status of the task.
        This is, the status after applying all the operations of ops to the
        input matrices of the task.
    seen: bool
        True if we have already checked and executed the possible operations
        of the candidate. False otherwise.
    """
    def __init__(self, ops, tasks, score=1000, predictions=np.zeros((2,2))):
        self.ops = ops
        self.score = score
        self.tasks = tasks
        self.t = None
        self.predictions = predictions
        self.seen = False

    def __lt__(self, other):
        """
        A candidate is better than another one if its score is lower.
        """
        if self.score == other.score:
            return len(self.ops) < len(other.ops)
        return self.score < other.score

    def generateTask(self):
        """
        Assign to the attribute t the Task object corresponding to the
        current task status.
        """
        self.t = Task(self.tasks[-1], 'dummyIndex', submission=True)

class Best3Candidates():
    """
    An object of this class stores the three best candidates of a task.

    ...
    Attributes
    ----------
    candidates: list
        A list of three elements, each one of them being an object of the class
        Candidate.
    """
    def __init__(self, Candidate1, Candidate2, Candidate3):
        self.candidates = [Candidate1, Candidate2, Candidate3]

    def maxCandidate(self):
        """
        Returns the index of the candidate with highest score.
        """
        x = 0
        if self.candidates[1] > self.candidates[0]:
            x = 1
        if self.candidates[2] > self.candidates[x]:
            x = 2
        return x

    def addCandidate(self, c):
        """
        Given a candidate c, this function substitutes c with the worst
        candidate in self.candidates only if it's a better candidate (its score
        is lower).
        """
        if all([self.candidates[i].score < c.score for i in range(3)]):
            return
        
        for i in range(3):
            if all([np.array_equal(self.candidates[i].predictions[x], c.predictions[x]) \
                    for x in range(len(c.predictions))]):
                return
        iMaxCand = self.maxCandidate()
        for i in range(3):
            if c < self.candidates[iMaxCand]:
                c.generateTask()
                self.candidates[iMaxCand] = c
                break

    def allPerfect(self):
        return all([c.score==0 for c in self.candidates])

    def getOrderedIndices(self):
        """
        Returns a list of 3 indices (from 0 to 2) with the candidates ordered
        from best to worst.
        """
        orderedList = [0]
        if self.candidates[1] < self.candidates[0]:
            orderedList.insert(0, 1)
        else:
            orderedList.append(1)
        if self.candidates[2] < self.candidates[orderedList[0]]:
            orderedList.insert(0, 2)
        elif self.candidates[2] < self.candidates[orderedList[1]]:
            orderedList.insert(1, 2)
        else:
            orderedList.append(2)
        return orderedList
    
# Separate task by shapes
class TaskSeparatedByShapes():
    """
    An object of this class stores a Task that has been separated by Shapes, 
    as well as the necessary information to restore it into the original the 
    matrices according to the original inputs.
    ...
    Attributes
    ----------
    originalTask: dict
        The original task in its original format.
    separatedTask: Task
        The task separated by shapes, according to the criteria defined in
        needsSeparationByShapes.
    nShapes: dict
        Stores the number of Shapes used to separate each train and test
        sample.
    background: int
        The background color of the original task, necessary for the restoring
        step.
    mergeColor: int
        Color to use when merging matrices if there is a conflict.
    """
    def __init__(self, task, background, diagonal=False):
        self.originalTask = task
        self.separatedTask = None
        self.nShapes = {'train': [], 'test': []}
        self.background = background
        self.mergeColor = None

    def getRange(self, trainOrTest, index):
        """
        Returns the range of matrices in the separated task to be merged in
        order to obtain the final trainOrTest matrix on the given index.
        """
        i, position = 0, 0
        while i < index:
            position += self.nShapes[trainOrTest][i]
            i += 1
        return (position, position+self.nShapes[trainOrTest][index])
            
def needsSeparationByShapes(t):
    """
    This function checks whether the Task t needs to be separated by Shapes.
    If that's the case, it returns an object of the class TaskSeparatedByShapes, 
    and otherwise it returns False.
    A Task can be separated by Shapes if all the Samples can. A Sample can be
    separated by Shapes if it has a background color, and the same number of
    Shapes in the input and in the output, in a way that a Shape in the input
    clearly corresponds to a Shape in the output. For the task to need
    separation by Shapes, at least one of the Samples has to have two separable
    Shapes.
    Separability by Shapes is checked for single color, multicolor and diagonal
    Shapes.
    """
    def getOverlap(inShape, inPos, outShape, outPos):
        x1a, y1a, x1b, y1b = inPos[0], inPos[1], outPos[0], outPos[1]
        x2a, y2a = inPos[0]+inShape[0]-1, inPos[1]+inShape[1]-1
        x2b, y2b = outPos[0]+outShape[0]-1, outPos[1]+outShape[1]-1
        if x1a<=x1b:
            if x2a<=x1b:
                return 0
            x = x2a-x1b+1
        elif x1b<=x1a:
            if x2b<=x1a:
                return 0
            x = x2b-x1a+1
        if y1a<=y1b:
            if y2a<=y1b:
                return 0
            y = y2a-y1b+1
        elif y1b<=y1a:
            if y2b<=y1a:
                return 0
            y = y2b-y1a+1

        return x*y
    
    def generateNewTask(inShapes, outShapes, testShapes):
        # Assign every input shape to the output shape with maximum overlap
        separatedTask = TaskSeparatedByShapes(t.task.copy(), t.backgroundColor)
        task = {'train': [], 'test': []}
        for s in range(t.nTrain):
            seenIndices = set()
            for inShape in inShapes[s]:
                shapeIndex = 0
                maxOverlap = 0
                bestIndex = -1
                for outShape in outShapes[s]:
                    overlap = getOverlap(inShape.shape, inShape.position, outShape.shape, outShape.position)
                    if overlap > maxOverlap:
                        maxOverlap = overlap
                        bestIndex = shapeIndex
                    shapeIndex += 1
                if bestIndex!=-1 and bestIndex not in seenIndices:
                    seenIndices.add(bestIndex)
                    # Generate the new input and output matrices
                    inM = np.full(t.trainSamples[s].inMatrix.shape, t.backgroundColor ,dtype=np.uint8)
                    outM = inM.copy()
                    inM = insertShape(inM, inShape)
                    outM = insertShape(outM, outShapes[s][bestIndex])
                    task['train'].append({'input': inM.tolist(), 'output': outM.tolist()})
            # If we haven't dealt with all the shapes successfully, then return
            if len(seenIndices) != len(inShapes[s]):
                return False
            # Record the number of new samples generated by sample s
            separatedTask.nShapes['train'].append(len(inShapes[s]))
        for s in range(t.nTest):
            for testShape in testShapes[s]:
                inM = np.full(t.testSamples[s].inMatrix.shape, t.backgroundColor ,dtype=np.uint8)
                inM = insertShape(inM, testShape)
                if t.submission:
                    task['test'].append({'input': inM.tolist()})
                else:
                    task['test'].append({'input': inM.tolist(), 'output': t.testSamples[s].outMatrix.m.tolist()})
            # Record the number of new samples generated by sample s
            separatedTask.nShapes['test'].append(len(testShapes[s]))
                
        
        # Complete and return the TaskSeparatedByShapes object
        separatedTask.separatedTask = task.copy()
        return separatedTask
        

    # I need to have a background color to generate the new task object
    if t.backgroundColor==-1 or not t.sameIOShapes:
        return False
    # Only consider tasks without small matrices
    if any([s.inMatrix.shape[0]*s.inMatrix.shape[1]<43 for s in t.trainSamples+t.testSamples]):
        return False

    mergeColors = t.commonOutColors - t.totalInColors
    if len(mergeColors) == 1:
        mergeColor = next(iter(mergeColors))
    else:
        mergeColor = None
    
    # First, consider normal shapes (not background, not diagonal, not multicolor) (Task 84 as example)
    inShapes = [[shape for shape in s.inMatrix.shapes if shape.color!=t.backgroundColor] for s in t.trainSamples]
    outShapes = [[shape for shape in s.outMatrix.shapes if shape.color!=t.backgroundColor] for s in t.trainSamples]
    testShapes = [[shape for shape in s.inMatrix.shapes if shape.color!=t.backgroundColor] for s in t.testSamples]
    if all([len(inShapes[s])<=7 and len(inShapes[s])==len(outShapes[s]) for s in range(t.nTrain)]):
        newTask = generateNewTask(inShapes, outShapes, testShapes)
        if newTask != False:
            if len(mergeColors) == 1:
                newTask.mergeColor = mergeColor
            return newTask
        
    # Now, consider diagonal shapes (Task 681 as example)
    inShapes = [[shape for shape in s.inMatrix.dShapes if shape.color!=t.backgroundColor] for s in t.trainSamples]
    outShapes = [[shape for shape in s.outMatrix.dShapes if shape.color!=t.backgroundColor] for s in t.trainSamples]
    testShapes = [[shape for shape in s.inMatrix.dShapes if shape.color!=t.backgroundColor] for s in t.testSamples]
    if all([len(inShapes[s])<=5 and len(inShapes[s])==len(outShapes[s]) for s in range(t.nTrain)]):
        newTask = generateNewTask(inShapes, outShapes, testShapes)
        if newTask != False:
            if len(mergeColors) == 1:
                newTask.mergeColor = mergeColor
            return newTask
    
    # Now, multicolor non-diagonal shapes (Task 611 as example)
    inShapes = [[shape for shape in s.inMatrix.multicolorShapes] for s in t.trainSamples]
    outShapes = [[shape for shape in s.outMatrix.multicolorShapes] for s in t.trainSamples]
    testShapes = [[shape for shape in s.inMatrix.multicolorShapes] for s in t.testSamples]
    if all([len(inShapes[s])<=7 and len(inShapes[s])==len(outShapes[s]) for s in range(t.nTrain)]):
        newTask = generateNewTask(inShapes, outShapes, testShapes)
        if newTask != False:
            if len(mergeColors) == 1:
                newTask.mergeColor = mergeColor
            return newTask
    
    # Finally, multicolor diagonal (Task 610 as example)
    inShapes = [[shape for shape in s.inMatrix.multicolorDShapes] for s in t.trainSamples]
    outShapes = [[shape for shape in s.outMatrix.multicolorDShapes] for s in t.trainSamples]
    testShapes = [[shape for shape in s.inMatrix.multicolorDShapes] for s in t.testSamples]
    if all([len(inShapes[s])<=5 and len(inShapes[s])==len(outShapes[s]) for s in range(t.nTrain)]):
        newTask = generateNewTask(inShapes, outShapes, testShapes)
        if newTask != False:
            if len(mergeColors) == 1:
                newTask.mergeColor = mergeColor
            return newTask

    return False

# Separate task by colors
class TaskSeparatedByColors():
    """
    An object of this class stores a Task that has been separated by colors, 
    as well as the necessary information to restore it into the original the 
    matrices according to the original inputs.
    ...
    Attributes
    ----------
    originalTask: dict
        The original task in its original format.
    separatedTask: Task
        The task separated by shapes, according to the criteria defined in
        needsSeparationByShapes.
    commonColors: set
        The colors that appear in every sample.
    extraColors: dict
        The number of colors that appear in every sample and are different from
        the commonColors.
    """
    def __init__(self, task):
        self.originalTask = task
        self.separatedTask = None
        self.commonColors = None
        self.extraColors = {'train': [], 'test': []}

    def getRange(self, trainOrTest, index):
        """
        Returns the range of matrices in the separated task to be merged in
        order to obtain the final trainOrTest matrix on the given index.
        """
        i, position = 0, 0
        while i < index:
            position += len(self.extraColors[trainOrTest][i])
            i += 1
        return (position, position+len(self.extraColors[trainOrTest][index]))


def needsSeparationByColors(t):
    """
    This function checks whether the Task t needs to be separated by colors.
    If that's the case, it returns an object of the class TaskSeparatedByColors, 
    and otherwise it returns False.
    A Task can be separated by colors the number of colors in each Sample is
    different. If that's the case, every Sample will be converted into as many
    samples as non-common colors it has (common color refers to color that 
    appears in every Sample).
    """
    def generateMatrix(matrix, colorsToKeep, backgroundColor):
        m = matrix.copy()
        for i,j in np.ndindex(matrix.shape):
            if m[i,j] not in colorsToKeep:
                m[i,j] = backgroundColor

        return m

    def generateNewTask(commonColors, backgroundColor):
        # Assign every input shape to the output shape with maximum overlap
        separatedTask = TaskSeparatedByColors(t.task.copy())
        task = {'train': [], 'test': []}
        for s in range(t.nTrain):
            separatedTask.extraColors['train'].append([])
            colorsToConsider = (t.trainSamples[s].inMatrix.colors | t.trainSamples[s].outMatrix.colors)\
                                - commonColors
            if len(colorsToConsider)==0:
                return False
            for color in colorsToConsider:
                separatedTask.extraColors['train'][s].append(color)
                inM = generateMatrix(t.trainSamples[s].inMatrix.m, commonColors|set([color]), backgroundColor)
                outM = generateMatrix(t.trainSamples[s].outMatrix.m, commonColors|set([color]), backgroundColor)
                task['train'].append({'input': inM.tolist(), 'output': outM.tolist()})

        for s in range(t.nTest):
            separatedTask.extraColors['test'].append([])
            if t.submission:
                colorsToConsider = t.testSamples[s].inMatrix.colors - commonColors
                if len(colorsToConsider)==0:
                    return False
                for color in colorsToConsider:
                    separatedTask.extraColors['test'][s].append(color)
                    inM = generateMatrix(t.testSamples[s].inMatrix.m, commonColors|set([color]), backgroundColor)
                    task['test'].append({'input': inM.tolist()})
            else:
                colorsToConsider = (t.testSamples[s].inMatrix.colors | t.testSamples[s].outMatrix.colors)\
                                    - commonColors
                if len(colorsToConsider)==0:
                    return False
                for color in colorsToConsider:
                    separatedTask.extraColors['test'][s].append(color)
                    inM = generateMatrix(t.testSamples[s].inMatrix.m, commonColors|set([color]), backgroundColor)
                    outM = generateMatrix(t.testSamples[s].outMatrix.m, commonColors|set([color]), backgroundColor)
                    task['test'].append({'input': inM.tolist(), 'output': t.testSamples[s].outMatrix.m.tolist()})

        # Complete and return the TaskSeparatedByShapes object
        separatedTask.separatedTask = task.copy()
        return separatedTask


    # I need to have a background color to generate the new task object
    if t.backgroundColor==-1 or not t.sameIOShapes:
        return False
    # Only consider tasks without small matrices
    if any([s.inMatrix.shape[0]*s.inMatrix.shape[1]<43 for s in t.trainSamples+t.testSamples]):
        return False

    commonColors = t.commonInColors | t.commonOutColors

    if all([sample.nColors == len(commonColors) for sample in t.trainSamples]):
        return False
    if any([sample.nColors < len(commonColors) for sample in t.trainSamples]):
        return False

    newTask = generateNewTask(commonColors, t.backgroundColor)

    return newTask

# Crop task if necessary
def getCroppingPosition(matrix):
    """
    Function to be used only if the the Task is to be cropped. If that's the
    case, given a matrix, this function returns the cropping position.
    """
    bC = matrix.backgroundColor
    x, xMax, y, yMax = 0, matrix.m.shape[0]-1, 0, matrix.m.shape[1]-1
    while x <= xMax and np.all(matrix.m[x,:] == bC):
        x += 1
    while y <= yMax and np.all(matrix.m[:,y] == bC):
        y += 1
    return [x,y]
    
def needsCropping(t):
    """
    This function checks whether a Task needs to be cropped. A Task needs to be
    cropped if any of the background is considered irrelevant, this is, if a
    part of the background is not modified.
    """
    # Only to be used if t.sameIOShapes
    for sample in t.trainSamples:
        if sample.inMatrix.backgroundColor != sample.outMatrix.backgroundColor:
            return False
        if getCroppingPosition(sample.inMatrix) != getCroppingPosition(sample.outMatrix):
            return False
        inMatrix = cropAllBackground(sample.inMatrix)
        outMatrix = cropAllBackground(sample.outMatrix)
        if inMatrix.shape!=outMatrix.shape or sample.inMatrix.shape==inMatrix.shape:
            return False
    return True

def cropTask(t, task):
    """
    This function crops the background of all the matrices of the Task t. The
    result of cropping the matrices is performed in-place in for the given
    task, and the positions and background colors of every matrix are returned
    in order to recover them in the revert process.
    """
    positions = {"train": [], "test": []}
    backgrounds = {"train": [], "test": []}
    for s in range(t.nTrain):
        task["train"][s]["input"] = cropAllBackground(t.trainSamples[s].inMatrix).tolist()
        task["train"][s]["output"] = cropAllBackground(t.trainSamples[s].outMatrix).tolist()
        backgrounds["train"].append(t.trainSamples[s].inMatrix.backgroundColor)
        positions["train"].append(getCroppingPosition(t.trainSamples[s].inMatrix))
    for s in range(t.nTest):
        task["test"][s]["input"] = cropAllBackground(t.testSamples[s].inMatrix).tolist()
        backgrounds["test"].append(t.testSamples[s].inMatrix.backgroundColor)
        positions["test"].append(getCroppingPosition(t.testSamples[s].inMatrix))
        if not t.submission:
            task["test"][s]["output"] = cropAllBackground(t.testSamples[s].outMatrix).tolist()
    return positions, backgrounds

def recoverCroppedMatrix(matrix, outShape, position, backgroundColor):
    """
    Function to revert the cropping of the matrices that had been cropped with
    the function cropTask.
    """
    m = np.full(outShape, backgroundColor, dtype=np.uint8)
    m[position[0]:position[0]+matrix.shape[0], position[1]:position[1]+matrix.shape[1]] = matrix.copy()
    return m
    
def needsRecoloring(t):
    """
    This method determines whether the task t needs recoloring or not.
    It needs recoloring if every color in an output matrix appears either
    in the input or in every output matrix.
    Otherwise a recoloring doesn't make sense.
    If this function returns True, then orderTaskColors should be executed
    as the first part of the preprocessing of t.
    """
    for sample in t.trainSamples:
        for color in sample.outMatrix.colors:
            if (color not in sample.inMatrix.colors) and (color not in t.commonOutColors):
                return False
    return True

def orderTaskColors(t):
    """
    Given a task t, this function generates a new task (as a dictionary) by
    recoloring all the matrices in a specific way.
    The goal of this function is to impose that if two different colors
    represent the exact same thing in two different samples, then they have the
    same color in both of the samples.
    Right now, the criterium to order colors is:
        1. Common colors ordered according to Task.orderColors
        2. Colors that appear both in the input and the output
        3. Colors that only appear in the input
        4. Colors that only appear in the output
    In steps 2-4, if there is more that one color satisfying that condition, 
    the ordering will happen according to the colorCount.
    """
    def orderColors(trainOrTest):
        if trainOrTest=="train":
            samples = t.trainSamples
        else:
            samples = t.testSamples
        for sample in samples:
            sampleColors = t.orderedColors.copy()
            sortedColors = [k for k, v in sorted(sample.inMatrix.colorCount.items(), key=lambda item: item[1])]
            for c in sortedColors:
                if c not in sampleColors:
                    sampleColors.append(c)
            if trainOrTest=="train" or t.submission==False:
                sortedColors = [k for k, v in sorted(sample.outMatrix.colorCount.items(), key=lambda item: item[1])]
                for c in sortedColors:
                    if c not in sampleColors:
                        sampleColors.append(c)
                    
            rel, invRel = relDicts(sampleColors)
            if trainOrTest=="train":
                trainRels.append(rel)
                trainInvRels.append(invRel)
            else:
                testRels.append(rel)
                testInvRels.append(invRel)
                
            inMatrix = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
            for c in sample.inMatrix.colors:
                inMatrix[sample.inMatrix.m==c] = invRel[c]
            if trainOrTest=='train' or t.submission==False:
                outMatrix = np.zeros(sample.outMatrix.shape, dtype=np.uint8)
                for c in sample.outMatrix.colors:
                    outMatrix[sample.outMatrix.m==c] = invRel[c]
                if trainOrTest=='train':
                    task['train'].append({'input': inMatrix.tolist(), 'output': outMatrix.tolist()})
                else:
                    task['test'].append({'input': inMatrix.tolist(), 'output': outMatrix.tolist()})
            else:
                task['test'].append({'input': inMatrix.tolist()})
        
    task = {'train': [], 'test': []}
    trainRels = []
    trainInvRels = []
    testRels = []
    testInvRels = []
    
    orderColors("train")
    orderColors("test")
    
    return task, trainRels, trainInvRels, testRels, testInvRels

def recoverOriginalColors(matrix, rel):
    """
    Given a matrix, this function is intended to recover the original colors
    before being modified in the orderTaskColors function.
    rel is supposed to be either one of the trainRels or testRels outputs of
    that function.
    """
    m = matrix.copy()
    for i,j in np.ndindex(matrix.shape):
        if matrix[i,j] in rel.keys(): # TODO Task 162 fails. Delete this when fixed
            m[i,j] = rel[matrix[i,j]][0]
    return m

def ignoreGrid(t, task, inMatrix=True, outMatrix=True):
    """
    Given a Task t with a grid and its corresponding task dictionary, this
    function modifies task in-place by considering each cell of the grid as a
    pixel in the new matrix. For doing this, all the cells in the grid need to
    have only one color.
    """
    for s in range(t.nTrain):
        if inMatrix:
            m = np.zeros(t.trainSamples[s].inMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.trainSamples[s].inMatrix.grid.cells[i][j][0].colors))
            task["train"][s]["input"] = m.tolist()
        if outMatrix:
            m = np.zeros(t.trainSamples[s].outMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.trainSamples[s].outMatrix.grid.cells[i][j][0].colors))
            task["train"][s]["output"] = m.tolist()
    for s in range(t.nTest):
        if inMatrix:
            m = np.zeros(t.testSamples[s].inMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.testSamples[s].inMatrix.grid.cells[i][j][0].colors))
            task["test"][s]["input"] = m.tolist()
        if outMatrix and not t.submission:
            m = np.zeros(t.testSamples[s].outMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.testSamples[s].outMatrix.grid.cells[i][j][0].colors))
            task["test"][s]["output"] = m.tolist()

def recoverGrid(t, x, s):
    """
    Given a matrix x, this function recovers the grid that had been removed
    with ignoreGrid.
    """
    realX = t.testSamples[s].inMatrix.m.copy()
    cells = t.testSamples[s].inMatrix.grid.cells
    for cellI in range(len(cells)):
        for cellJ in range(len(cells[0])):
            cellShape = cells[cellI][cellJ][0].shape
            position = cells[cellI][cellJ][1]
            for k,l in np.ndindex(cellShape):
                realX[position[0]+k, position[1]+l] = x[cellI,cellJ]
    return realX

def ignoreAsymmetricGrid(t, task):
    """
    Given a Task t with a grid and its corresponding task dictionary, this
    function modifies task in-place by considering each cell of the asymmetric
    grid as a pixel in the new matrix. For doing this, all the cells in the
    grid need to have only one color.
    """
    for s in range(t.nTrain):
        m = np.zeros(t.trainSamples[s].inMatrix.asymmetricGrid.shape, dtype=np.uint8)
        for i,j in np.ndindex(m.shape):
            m[i,j] = next(iter(t.trainSamples[s].inMatrix.asymmetricGrid.cells[i][j][0].colors))
        task["train"][s]["input"] = m.tolist()
        m = np.zeros(t.trainSamples[s].outMatrix.asymmetricGrid.shape, dtype=np.uint8)
        for i,j in np.ndindex(m.shape):
            m[i,j] = next(iter(t.trainSamples[s].outMatrix.asymmetricGrid.cells[i][j][0].colors))
        task["train"][s]["output"] = m.tolist()
    for s in range(t.nTest):
        m = np.zeros(t.testSamples[s].inMatrix.asymmetricGrid.shape, dtype=np.uint8)
        for i,j in np.ndindex(m.shape):
            m[i,j] = next(iter(t.testSamples[s].inMatrix.asymmetricGrid.cells[i][j][0].colors))
        task["test"][s]["input"] = m.tolist()
        if not t.submission:
            m = np.zeros(t.testSamples[s].outMatrix.asymmetricGrid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.testSamples[s].outMatrix.asymmetricGrid.cells[i][j][0].colors))
            task["test"][s]["output"] = m.tolist()

def recoverAsymmetricGrid(t, x, s):
    """
    Given a matrix x, this function recovers the asymmetric grid that had been
    removed with ignoreAsymmetricGrid.
    """
    realX = t.testSamples[s].inMatrix.m.copy()
    cells = t.testSamples[s].inMatrix.asymmetricGrid.cells
    for cellI in range(len(cells)):
        for cellJ in range(len(cells[0])):
            cellShape = cells[cellI][cellJ][0].shape
            position = cells[cellI][cellJ][1]
            for k,l in np.ndindex(cellShape):
                realX[position[0]+k, position[1]+l] = x[cellI,cellJ]
    return realX

def rotateTaskWithOneBorder(t, task):
    """
    Given a Task t, this function determines whether any of the Samples needs
    have its Matrices rotated and, if that's the case, the task with the
    rotated Matrices is returned. Otherwise, it returns false.
    Matrices will be rotated if all the Samples contain a fixed Frontier that
    is in one of the borders of the Matrix. If that's the case, it will be made
    sure that the Frontier will be vertical and along the first column.
    """
    rotTask = copy.deepcopy(task)
    rotations = {'train': [], 'test': []}
    for s in range(t.nTrain):
        border = t.trainSamples[s].commonFullBorders[0]
        if border.direction=='h' and border.position==0:
            rotations['train'].append(1)
            rotTask['train'][s]['input'] = np.rot90(t.trainSamples[s].inMatrix.m, 1).tolist()
            rotTask['train'][s]['output'] = np.rot90(t.trainSamples[s].outMatrix.m, 1).tolist()
        elif border.direction=='v' and border.position==t.trainSamples[s].inMatrix.shape[1]-1:
            rotations['train'].append(2)
            rotTask['train'][s]['input'] = np.rot90(t.trainSamples[s].inMatrix.m, 2).tolist()
            rotTask['train'][s]['output'] = np.rot90(t.trainSamples[s].outMatrix.m, 2).tolist()
        elif border.direction=='h' and border.position==t.trainSamples[s].inMatrix.shape[0]-1:
            rotations['train'].append(3)
            rotTask['train'][s]['input'] = np.rot90(t.trainSamples[s].inMatrix.m, 3).tolist()
            rotTask['train'][s]['output'] = np.rot90(t.trainSamples[s].outMatrix.m, 3).tolist()
        else:
            rotations['train'].append(0)
    
    for s in range(t.nTest):
        if t.submission:
            hasBorder=False
            for border in t.testSamples[s].inMatrix.fullBorders:
                if border.color!=t.testSamples[s].inMatrix.backgroundColor:
                    if border.direction=='h' and border.position==0:
                        rotations['test'].append(1)
                        rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 1).tolist()
                    elif border.direction=='v' and border.position==t.testSamples[s].inMatrix.shape[1]-1:
                        rotations['test'].append(2)
                        rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 2).tolist()
                    elif border.direction=='h' and border.position==t.testSamples[s].inMatrix.shape[0]-1:
                        rotations['test'].append(3)
                        rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 3).tolist()
                    else:
                        rotations['test'].append(0)
                    hasBorder=True
                    break
            if not hasBorder:
                return False
        else:
            border = t.testSamples[s].commonFullBorders[0]
            if border.direction=='h' and border.position==0:
                rotations['test'].append(1)
                rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 1).tolist()
                rotTask['test'][s]['output'] = np.rot90(t.testSamples[s].outMatrix.m, 1).tolist()
            elif border.direction=='v' and border.position==t.testSamples[s].inMatrix.shape[1]-1:
                rotations['test'].append(2)
                rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 2).tolist()
                rotTask['test'][s]['output'] = np.rot90(t.testSamples[s].outMatrix.m, 2).tolist()
            elif border.direction=='h' and border.position==t.testSamples[s].inMatrix.shape[0]-1:
                rotations['test'].append(3)
                rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 3).tolist()
                rotTask['test'][s]['output'] = np.rot90(t.testSamples[s].outMatrix.m, 3).tolist()
            else:
                rotations['test'].append(0)
        
    return rotTask, rotations

def rotateHVTask(t, task):
    """
    Given a Task t, this function determines whether any of the Samples needs
    have its Matrices rotated and, if that's the case, the task with the
    rotated Matrices is returned. Otherwise, it returns false.
    Matrices will be rotated if all the Samples are either "Horizontal" or
    "Vertical". In that case, it will be made sure that all the Matrices of the
    rotated task are "Horizontal".
    """
    rotTask = copy.deepcopy(task)
    rotations = {'train': [], 'test': []}
    
    for s in range(t.nTrain):
        if t.trainSamples[s].isVertical:
            rotations['train'].append(1)
            rotTask['train'][s]['input'] = np.rot90(t.trainSamples[s].inMatrix.m, 1).tolist()
            rotTask['train'][s]['output'] = np.rot90(t.trainSamples[s].outMatrix.m, 1).tolist()
        else:
            rotations['train'].append(0)
    
    for s in range(t.nTest):
        if t.submission:
            if t.testSamples[s].inMatrix.isHorizontal:
                rotations['test'].append(0)
            elif t.testSamples[s].inMatrix.isVertical:
                rotations['test'].append(1)
                rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 1).tolist()
            else:
                return False, False
        else:
            if t.testSamples[s].isHorizontal:
                rotations['test'].append(0)
            elif t.testSamples[s].isVertical:
                rotations['test'].append(1)
                rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 1).tolist()
                rotTask['test'][s]['output'] = np.rot90(t.testSamples[s].outMatrix.m, 1).tolist()
            else:
                return False, False
            
    return rotTask, rotations

def recoverRotations(matrix, trainOrTest, s, rotations):
    """
    Revert the rotation executed the given matrix during the preprocessing.
    """
    if rotations[trainOrTest][s] == 1:
        m = np.rot90(matrix, 3)
    elif rotations[trainOrTest][s] == 2:
        m = np.rot90(matrix, 2)
    elif rotations[trainOrTest][s] == 3:
        m = np.rot90(matrix, 1)
    else:
        m = matrix.copy()        
    return m

def tryOperations(t, c, cTask, b3c, firstIt=False):
    """
    Given a Task t and a Candidate c, this function applies all the
    operations that make sense to the input matrices of c. After a certain
    operation is performed to all the input matrices, a new candidate is
    generated from the resulting output matrices. If the score of the candidate
    improves the score of any of the 3 best candidates, it will be saved in the
    variable b3c, which is an object of the class Best3Candidates.
    """
    if c.seen or c.score==0 or b3c.allPerfect():
        return
    startOps = ("switchColors", "cropShape", "cropAllBackground", "minimize", \
                "maxColorFromCell", "deleteShapes", "replicateShapes","colorByPixels",\
                "paintGridLikeBackground")
    repeatIfPerfect = ("extendColor", "moveAllShapes")
    
    # Get the operations for the given candidate and try all of them.
    possibleOps = getPossibleOperations(t, c)
    possibleOps = [x for x in possibleOps if x is not None]
    for op in possibleOps:
        for s in range(t.nTrain):
            cTask["train"][s]["input"] = op(c.t.trainSamples[s].inMatrix).tolist()
            if c.t.sameIOShapes and len(c.t.fixedColors) != 0:
                cTask["train"][s]["input"] = correctFixedColors(\
                     c.t.trainSamples[s].inMatrix.m,\
                     np.array(cTask["train"][s]["input"]),\
                     c.t.fixedColors, c.t.commonOnlyChangedInColors).tolist()
        newPredictions = []
        for s in range(t.nTest):
            newOutput = op(c.t.testSamples[s].inMatrix)
            newPredictions.append(newOutput)
            cTask["test"][s]["input"] = newOutput.tolist()
            if c.t.sameIOShapes and len(c.t.fixedColors) != 0:
                cTask["test"][s]["input"] = correctFixedColors(\
                     c.t.testSamples[s].inMatrix.m,\
                     np.array(cTask["test"][s]["input"]),\
                     c.t.fixedColors, c.t.commonOnlyChangedInColors).tolist()
        cScore = sum([incorrectPixels(np.array(cTask["train"][s]["input"]), \
                                            t.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
        changedPixels = sum([incorrectPixels(c.t.trainSamples[s].inMatrix.m, \
                                                  np.array(cTask["train"][s]["input"])) for s in range(t.nTrain)])
        
        # Generate a new Candidate after applying the operation, and update
        # the best 3 candidates.
        newCandidate = Candidate(c.ops+[op], c.tasks+[copy.deepcopy(cTask)], cScore,\
                                 predictions=newPredictions)
        b3c.addCandidate(newCandidate)
        if firstIt and str(op)[28:60].startswith(startOps):
            if all([np.array_equal(np.array(cTask["train"][s]["input"]), \
                                   t.trainSamples[s].inMatrix.m) for s in range(t.nTrain)]):
                continue
            newCandidate.generateTask()
            tryOperations(t, newCandidate, cTask, b3c)
        elif str(op)[28:60].startswith(repeatIfPerfect) and c.score - changedPixels == cScore and changedPixels != 0:
            newCandidate.generateTask()
            tryOperations(t, newCandidate, cTask, b3c)
    c.seen = True
            
def getPredictionsFromTask(originalT, task):
    """
    Given a task in its dict format, and originalT, an object of the class Task
    storing information of the original task, this function returns the
    predictions and the corresponding Best3Candidates object computed for
    the task.
    """
    # Preprocessing
    taskNeedsRecoloring = needsRecoloring(originalT)
    if taskNeedsRecoloring:
        task, trainRels, trainInvRels, testRels, testInvRels = orderTaskColors(originalT)
        t = Task(task, task_id, submission=True)
    else:
        t = originalT

    cTask = copy.deepcopy(task)

    if t.sameIOShapes:
        taskNeedsCropping = needsCropping(t)
    else:
        taskNeedsCropping = False
    if taskNeedsCropping:
        cropPositions, backgrounds = cropTask(t, cTask)
        t2 = Task(cTask, task_id, submission=True, backgrounds=backgrounds)
    elif t.hasUnchangedGrid:
        if t.gridCellsHaveOneColor:
            ignoreGrid(t, cTask) # This modifies cTask, ignoring the grid
            t2 = Task(cTask, task_id, submission=True)
        elif t.outGridCellsHaveOneColor:
            ignoreGrid(t, cTask, inMatrix=False)
            t2 = Task(cTask, task_id, submission=True)
        else:
            t2 = t
    elif t.hasUnchangedAsymmetricGrid and t.assymmetricGridCellsHaveOneColor:
        ignoreAsymmetricGrid(t, cTask)
        t2 = Task(cTask, task_id, submission=True)
    else:
        t2 = t
        
    if t2.sameIOShapes:
        hasRotated = False
        if t2.hasOneFullBorder:
            hasRotated, rotateParams = rotateTaskWithOneBorder(t2, cTask)
        elif t2.requiresHVRotation:
            hasRotated, rotateParams = rotateHVTask(t2, cTask)
        if hasRotated!=False:
            cTask = hasRotated.copy()
            t2 = Task(cTask, task_id, submission=True)

     # Generate the three candidates with best possible score
    
    cScore = sum([incorrectPixels(np.array(cTask["train"][s]["input"]), \
                                         t2.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
    dummyPredictions = [sample.inMatrix.m for sample in t2.testSamples]
    c = Candidate([], [task], score=cScore, predictions=dummyPredictions)
    c.t = t2
    b3c = Best3Candidates(c, c, c)

    prevScore = sum([c.score for c in b3c.candidates])
    firstIt = True
    while True:
        copyB3C = copy.deepcopy(b3c)
        for c in copyB3C.candidates:
            if c.score == 0:
                continue
            tryOperations(t2, c, cTask, b3c, firstIt)
            if firstIt:
                firstIt = False
                break
        score = sum([c.score for c in b3c.candidates])
        if score >= prevScore:
            break
        else:
            prevScore = score
            
    taskPredictions = []
    
    # Once the best 3 candidates have been found, make the predictions and
    # revert the preprocessing.
    for s in range(t.nTest):
        taskPredictions.append([])
        for c in b3c.candidates:
            x = t2.testSamples[s].inMatrix.m.copy()
            for opI in range(len(c.ops)):
                newX = c.ops[opI](Matrix(x))
                if t2.sameIOShapes and len(t2.fixedColors) != 0:
                    x = correctFixedColors(x, newX, t2.fixedColors, t2.commonOnlyChangedInColors)
                else:
                    x = newX.copy()
            if t2.sameIOShapes and hasRotated!=False:
                x = recoverRotations(x, "test", s, rotateParams)
            if taskNeedsCropping:
                x = recoverCroppedMatrix(x, originalT.testSamples[s].inMatrix.shape, \
                                         cropPositions["test"][s], t.testSamples[s].inMatrix.backgroundColor)
            elif t.hasUnchangedGrid and (t.gridCellsHaveOneColor or t.outGridCellsHaveOneColor):
                x = recoverGrid(t, x, s)
            elif t.hasUnchangedAsymmetricGrid and t.assymmetricGridCellsHaveOneColor:
                x = recoverAsymmetricGrid(t, x, s)
            if taskNeedsRecoloring:
                x = recoverOriginalColors(x, testRels[s])
            
            taskPredictions[s].append(x)
            
    return taskPredictions, b3c
        
###############################################################################
# %% Main Loop and submission

submission = pd.DataFrame(columns=['output'])
submission.index.name = 'output_id'
    
# flag = False
for task_id, task in tqdm(data.items()):
    # if flag:
    #     pass
    # elif task_id == "7b6016b9":
    #     flag = True
    #     with open("temp_submission_until_7b6016b9.pkl", "rb") as f:
    #         submission = pickle.load(f)
    #     print("*************** CONTINUE RUN FROM 7b6016b9 *****************")
    # else:
    #     continue
        
    bestScores = []
    print("task_id", task_id)
                    
    originalT = Task(task, task_id, submission=True)
        
    predictions, b3c = getPredictionsFromTask(originalT, task.copy())
    
    if any([c.score==0 for c in b3c.candidates]):
        perfectScore = True
    
    separationByShapes = needsSeparationByShapes(originalT)
    separationByColors = needsSeparationByColors(originalT)

    if separationByShapes != False:
        separatedT = Task(separationByShapes.separatedTask, task_id, submission=True)
        sepPredictions, sepB3c = getPredictionsFromTask(separatedT, separationByShapes.separatedTask.copy())

        if any([c.score==0 for c in sepB3c.candidates]):
            perfectScore = True
        
        mergedPredictions = []
        for s in range(originalT.nTest):
            mergedPredictions.append([])
            matrixRange = separationByShapes.getRange("test", s)
            matrices = [[sepPredictions[i][cand] for i in range(matrixRange[0], matrixRange[1])] \
                         for cand in range(3)]
            for cand in range(3):
                pred = mergeMatrices(matrices[cand], originalT.backgroundColor, separationByShapes.mergeColor)
                mergedPredictions[s].append(pred)
        
        finalPredictions = []
        for s in range(originalT.nTest):
            finalPredictions.append([[], [], []])
        
        b3cIndices = b3c.getOrderedIndices()
        sepB3cIndices = sepB3c.getOrderedIndices()

        b3cIndex, sepB3cIndex = 0, 0
        i = 0
        if b3c.candidates[b3cIndices[0]].score==0:
            bestScores.append(0)
            for s in range(originalT.nTest):
                finalPredictions[s][0] = predictions[s][b3cIndices[0]]
            i += 1
        if sepB3c.candidates[sepB3cIndices[0]].score==0:
            bestScores.append(0)
            for s in range(originalT.nTest):
                finalPredictions[s][i] = mergedPredictions[s][sepB3cIndices[0]]
            i += 1
        while i < 3:
            if b3c.candidates[b3cIndices[b3cIndex]] < sepB3c.candidates[sepB3cIndices[sepB3cIndex]]:
                bestScores.append(b3c.candidates[b3cIndices[b3cIndex]].score)
                for s in range(originalT.nTest):
                    finalPredictions[s][i] = predictions[s][b3cIndices[b3cIndex]]
                b3cIndex += 1
            else:
                bestScores.append(sepB3c.candidates[sepB3cIndices[sepB3cIndex]].score)
                for s in range(originalT.nTest):
                    finalPredictions[s][i] = mergedPredictions[s][sepB3cIndices[sepB3cIndex]]
                sepB3cIndex += 1
            i += 1

    elif separationByColors != False:
        separatedT = Task(separationByColors.separatedTask, task_id, submission=True)
        sepPredictions, sepB3c = getPredictionsFromTask(separatedT, separationByColors.separatedTask.copy())

        if any([c.score==0 for c in sepB3c.candidates]):
            perfectScore = True
        
        mergedPredictions = []
        for s in range(originalT.nTest):
            mergedPredictions.append([])
            matrixRange = separationByColors.getRange("test", s)
            matrices = [[sepPredictions[i][cand] for i in range(matrixRange[0], matrixRange[1])] \
                         for cand in range(3)]
            for cand in range(3):
                pred = mergeMatrices(matrices[cand], originalT.backgroundColor)
                mergedPredictions[s].append(pred)

        finalPredictions = []
        for s in range(originalT.nTest):
            finalPredictions.append([[], [], []])
        
        b3cIndices = b3c.getOrderedIndices()
        sepB3cIndices = sepB3c.getOrderedIndices()

        b3cIndex, sepB3cIndex = 0, 0
        i = 0
        if b3c.candidates[b3cIndices[0]].score==0:
            bestScores.append(0)
            for s in range(originalT.nTest):
                finalPredictions[s][0] = predictions[s][b3cIndices[0]]
            i += 1
        if sepB3c.candidates[sepB3cIndices[0]].score==0:
            bestScores.append(0)
            for s in range(originalT.nTest):
                finalPredictions[s][i] = mergedPredictions[s][sepB3cIndices[0]]
            i += 1
        while i < 3:
            if b3c.candidates[b3cIndices[b3cIndex]] < sepB3c.candidates[sepB3cIndices[sepB3cIndex]]:
                bestScores.append(b3c.candidates[b3cIndices[b3cIndex]].score)
                for s in range(originalT.nTest):
                    finalPredictions[s][i] = predictions[s][b3cIndices[b3cIndex]]
                b3cIndex += 1
            else:
                bestScores.append(sepB3c.candidates[sepB3cIndices[sepB3cIndex]].score)
                for s in range(originalT.nTest):
                    finalPredictions[s][i] = mergedPredictions[s][sepB3cIndices[sepB3cIndex]]
                sepB3cIndex += 1
            i += 1
    else:
        for c in b3c.candidates:
            bestScores.append(c.score)
        finalPredictions = predictions

    pred = []
    for pair_id in range(originalT.nTest):
        for i in range(len(finalPredictions[pair_id])):
            pred.append(flattener(finalPredictions[pair_id][i].astype(int).tolist()))
    predictions = pred

    if len(predictions) == 0:
        pred = '|0| |0| |0|'
    elif len(predictions) == 1:
        pred = predictions[0] + ' ' + predictions[0] + ' ' + predictions[0]
    elif len(predictions) == 2:
        pred =  predictions[0] + ' ' + predictions[1] + ' ' + predictions[0]
    elif len(predictions) == 3:
        pred = predictions[0] + ' ' + predictions[1] + ' ' + predictions[2]
    
    for pair_id in range(originalT.nTest):
        submission.loc[task_id+'_'+str(pair_id), 'output'] = pred
    
    with open("temp_submission.pkl", "wb") as f:
        pickle.dump(submission, f)


# In[143]:


import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
import math
import inspect
import collections
from os.path import join as path_join
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.nn import Conv2d
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
import random
from time import time
from random import randint
import matplotlib.pyplot as plt
from matplotlib import colors
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from time import sleep
import copy
import gc
from pdb import set_trace as st
import timeit
import itertools
from numpy.lib.stride_tricks import as_strided
from scipy.spatial import distance
from collections import defaultdict
import warnings
from skimage import measure
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
device = "cuda" if torch.cuda.is_available() else "cpu"


# In[144]:


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(0)    


# In[145]:


from sklearn.neighbors import KNeighborsClassifier


# 
# # helper funcs

# In[146]:


def is_symmetry_lr(inp):
    return np.array(inp).tolist() == np.fliplr(inp).tolist()
def is_symmetry_ud(inp):
    return np.array(inp).tolist() == np.flipud(inp).tolist()

def input_output_shape_is_same(task):
    return all([np.array(el['input']).shape == np.array(el['output']).shape for el in task['train']])

def sliding_window_search(large, small):
    m1, n1 = np.array(large).shape
    m2, n2 = np.array(small).shape
    for m in range(m1 - m2 + 1):
        for n in range(n1 - n2 + 1):
            if np.array(small).tolist() == np.array(large)[m:m+m2, n:n+n2].tolist():
                return True, m, n
    return False, -1, -1

def matrix_use_color(m):
    return list(set(list(itertools.chain.from_iterable(m))))

def is_square(task):
    return all([np.array(el['input']).shape[0] == np.array(el['input']).shape[1] for el in task['train']])

def inouts_flip(task_train):
    for inout in task_train:
        inout['input'] = np.flip(inout['input'])
    return task_train

def inouts_flipud(task_train):
    for inout in task_train:
        inout['input'] = np.flipud(inout['input'])
    return task_train

def inouts_fliplr(task_train):
    for inout in task_train:
        inout['input'] = np.fliplr(inout['input'])
    return task_train

def match_rate(mat1, mat2):
    m1 = np.array(mat1)
    m2 = np.array(mat2)
    if m1.shape != m2.shape:
        return -1
    v1 = list(itertools.chain.from_iterable(mat1))
    v2 = list(itertools.chain.from_iterable(mat2))
    score1 = np.sum(m1==m2) / len(v1)
    score2 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return (score1 + score2) / 2

def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

def getDefaultPred(inp):
    pred_1 = flattener(inp)
    # for the second guess, change all 0s to 5s
    inp = [[5 if i==0 else i for i in j] for j in inp]
    pred_2 = flattener(inp)
    # for the last gues, change everything to 0
    inp = [[0 for i in j] for j in inp]
    pred_3 = flattener(inp)
    # concatenate and add to the submission output
    pred = pred_1 + ' ' + pred_2 + ' ' + pred_3 + ' ' 
    return pred

def preds_to_str(preds_list, idx):
    pred_strs = []
#     st()
    for i in range(len(preds_list[0])):
        pred_str = ''
        for j, preds in enumerate(reversed(preds_list)):
            if j == 3:
                break
            pred_str += flattener(np.array(preds[i]).tolist()) + ' '
        pred_strs.append(pred_str)
    return pred_strs

def preds_to_str_only0(preds0, idx):
    preds = []
    for i in range(len(preds0)):
        pred0 = flattener(np.array(preds0[i]).tolist())
        pred1 = flattener(np.array([[0]]).tolist())
        preds.append(pred0 + ' ' + pred1 + ' ' + pred1 + ' ')
    return preds

def get_not_use_num(matrix1, matrix2):
    v1 = list(itertools.chain.from_iterable(matrix1))
    v2 = list(itertools.chain.from_iterable(matrix2))
    for j in range(1,10):
        if (j not in v1) & (j not in v2):
            return j
    return 1


# # test NN

# In[147]:


def test_nn(train, test):
    task={'train':train, 'test':test}

    train_dataset = ArcDataset(task, mode="train", augment=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=device_collate)
    valid_dataset = ArcDataset(task, mode="test", augment=False)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=device_collate)

    net = Task330Net().to(device)
    criterion = hinge_loss
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time()
    tmp_train_loss = 1

    for param in net.named_parameters():
        print(f"{param[0]:>15} {list(param[1].shape)}")
    count = 0
    for epoch in range(5000):
        train_loss = valid_loss = 0.0
        train_loss_denom = valid_loss_denom = 0

        ####################
        # train
        ####################
        net.train()
        for i, (feature, target) in enumerate(train_dataloader):
            outputs = net(feature)
            loss = criterion(outputs, target)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record
            train_loss += loss.item()
            train_loss_denom += feature.shape[0]

        train_loss /= train_loss_denom

        ####################
        # eval
        ####################
#         net.eval()
#         with torch.no_grad():
#             for i, (feature, target) in enumerate(valid_dataloader):
#                 feature = feature.to(device)
#                 target = target.to(device)

#                 outputs = net(feature)
#                 loss = criterion(outputs, target)

#                 # record
#                 valid_loss += loss.item()
#                 valid_loss_denom += feature.shape[0]

#         valid_loss /= valid_loss_denom
    
        if epoch%100==0:
#             print(f"epoch {epoch:4d}  |  train_loss: {train_loss:5.6f}  valid_loss: {valid_loss:5.6f}  |  time: {time()-t0:7.1f} sec")
            print(f"epoch {epoch:4d}  |  train_loss: {train_loss:5.6f}  |  time: {time()-t0:7.1f} sec")
            if tmp_train_loss <= train_loss:
                count += 1
            if count >= 4:
                break
            tmp_train_loss = train_loss

#             if best_loss > valid_loss:
#                 best_loss = valid_loss
#                 filename = f"./work/trained_weight/{MODEL_NAME}_epoch{epoch:03d}_loss{valid_loss:.3f}.pth"
#                 torch.save(net.state_dict(), filename)

    return check(task, lambda x: task_train330(x, net))

class ArcDataset(torch.utils.data.Dataset):
    def __init__(self, task=None, mode="train", augment=False):
        if task is not None:
            assert mode in ["train", "test"]
            self.mode = mode
            self.task = task[mode]
        self.augment = augment

    def __len__(self):
        return len(self.task)
    
    def __getitem__(self, index):
        t = self.task[index]
        t_in = torch.tensor(t["input"])
        t_out = torch.tensor(t["output"])
        t_in, t_out = self.preprocess(t_in, t_out)
        return t_in, t_out
    
    def preprocess(self, t_in, t_out):
        if self.augment:
            t_in, t_out = self._random_rotate(t_in, t_out)
        t_in = self._one_hot_encode(t_in)
        t_out = self._one_hot_encode(t_out)
        return t_in, t_out
    
    def _one_hot_encode(self, x):
        return torch.eye(10)[x].permute(2, 0, 1)
    
    def _random_rotate(self, t_in, t_out):
        t_in_shape = t_in.shape
        t_out_shape = t_out.shape
        t_in = t_in.reshape(-1, *t_in_shape[-2:])
        t_out = t_out.reshape(-1, *t_out_shape[-2:])
        r = randint(0, 7)
        if r%2 == 0:
            t_in = t_in.permute(0, 2, 1)
            t_out = t_out.permute(0, 2, 1)
        r //= 2
        if r%2 == 0:
            t_in = t_in[:, :, torch.arange(t_in.shape[-1]-1, -1, -1)]
            t_out = t_out[:, :, torch.arange(t_out.shape[-1]-1, -1, -1)]
        r //= 2
        if r%2 == 0:
            t_in = t_in[:, torch.arange(t_in.shape[-2]-1, -1, -1), :]
            t_out = t_out[:, torch.arange(t_out.shape[-2]-1, -1, -1), :]
        t_in = t_in.reshape(*t_in_shape[:-2], *t_in.shape[-2:])
        t_out = t_out.reshape(*t_out_shape[:-2], *t_out.shape[-2:])
        return t_in, t_out
    
def device_collate(batch):
    return tuple(map(lambda x: torch.stack(x).to(device), zip(*batch)))

def hinge_loss(y_pred, y_true):
    loss = y_pred.clone()
    loss[y_true>0.5] = 1-loss[y_true>0.5]
    loss[loss<0] = 0
    return loss.sum(0).mean()

class Task330Net(nn.Module):
    def __init__(self):
        super(Task330Net, self).__init__()
        siz = 16
        self.conv1 = nn.Conv2d(10, siz, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout2d(p=0.1)
        self.conv1x1 = nn.Conv2d(siz, 10, 1)

    def forward(self, x):
        x2 = self.conv1(x)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.conv1x1(x2)
        x = x + x2  # skip connection
        return x
    
def check(task, pred_func):
    preds = []
    for i, t in enumerate(task["test"]):
        t_in = np.array(t["input"])
        preds.append(pred_func(t_in))
    return preds

def task_train330(x, net):
    def one_hot_decode(x):
        return x.argmax(0)
    net.eval()
    with torch.no_grad():
        x = torch.tensor(x).to(device)
        y_dummy = x.clone()
        dataset = ArcDataset(augment=False)
        x = dataset.preprocess(x, y_dummy)[0].unsqueeze(0)
        y = net(x).detach()
        y = one_hot_decode(y.squeeze(0))
    y = y.to("cpu").numpy()
    return y


# # CA

# In[148]:


class CAModel(nn.Module):
    def __init__(self, num_states):
        super(CAModel, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(num_states, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_states, kernel_size=1)
        )
        
    def forward(self, x, steps=1):
        for _ in range(steps):
            x = self.transition(torch.softmax(x, dim=1))
        return x

def solve_task(task, max_steps=10):
    model = CAModel(10).to(device)
    num_epochs = 100
    criterion = nn.CrossEntropyLoss()
    losses = np.zeros((max_steps - 1) * num_epochs)

    for num_steps in range(1, max_steps):
        optimizer = torch.optim.Adam(model.parameters(), lr=(0.1 / (num_steps * 2)))
        
        for e in range(num_epochs):
            optimizer.zero_grad()
            loss = 0.0

            for sample in task:
                # predict output from input
#                 st()
                x = torch.from_numpy(inp2img(sample["input"])).unsqueeze(0).float().to(device)
                y = torch.tensor(sample["output"]).long().unsqueeze(0).to(device)
                y_pred = model(x, num_steps)
                loss += criterion(y_pred, y)
                
                # predit output from output
                # enforces stability after solution is reached
                y_in = torch.from_numpy(inp2img(sample["output"])).unsqueeze(0).float().to(device)
                y_pred = model(y_in, 1) 
                loss += criterion(y_pred, y)

            loss.backward()
            optimizer.step()
            losses[(num_steps - 1) * num_epochs + e] = loss.item()
    return model, num_steps, losses
                
@torch.no_grad()
def predict(model, task):
    predictions = []
    for sample in task:
        x = torch.from_numpy(inp2img(sample["input"])).unsqueeze(0).float().to(device)
        pred = model(x, 100).argmax(1).squeeze().cpu().numpy()
        predictions.append(pred)
    return predictions


# # model

# In[149]:


def inp2img(inp):
    inp = np.array(inp)
    img = np.full((10, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    for i in range(10):
        img[i] = (inp==i)
    return img

class TaskSolver:
    def train(self, task_train, n_epoch=30, preprocess_funcs=[],final=False):
        """basic pytorch train loop"""
        self.net = Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2)

        criterion = CrossEntropyLoss()
        optimizer = Adam(self.net.parameters(), lr = 0.1)
        for epoch in range(n_epoch):
            for sample in task_train:
                inputs = copy.deepcopy(sample['input'])
                for preprocess_func in preprocess_funcs:
                    inputs = preprocess_func(inputs)
                inputs = FloatTensor(inp2img(inputs)).unsqueeze(dim=0)
                labels = LongTensor(sample['output']).unsqueeze(dim=0)
                optimizer.zero_grad()
                outputs = self.net(inputs)
#                 import pdb; pdb.set_trace()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
#                 st()
        return self

    def predict(self, task_test, preprocess_funcs=[], success_map={}, idx='searching', final_score_map={}, final=False):
        predictions = []
        with torch.no_grad():
            for i, in_out in enumerate(task_test):
                inputs = copy.deepcopy(in_out['input'])
                # input変更
                for preprocess_func in preprocess_funcs:
                    inputs = preprocess_func(inputs)
                inputs = FloatTensor(inp2img(inputs)).unsqueeze(dim=0)
                outputs = self.net(inputs)
                pred =  outputs.squeeze(dim=0).cpu().numpy().argmax(0)
                if ('output' in in_out) & (idx != 'searching') & (not final):
                    similarity = match_rate(in_out['output'], pred)
                    if (idx not in final_score_map) or (final_score_map.get(idx, 101) < similarity):
                        final_score_map[idx] = similarity
                predictions.append(pred)

        return predictions


# # post process

# In[150]:


def post_processes(preds, origin_task):
    processed_preds = []
    for pred in preds:
        all_input_same_colors, all_output_same_colors, all_input_and_output_same_colors, each_input_and_output_same_colors = search_inout_used_colors(origin_task['train'])
        colors_are_black_and_one_flag, only_used_color_num = colors_are_black_and_one(origin_task['train'][0]['output'])
        if all_output_same_colors & colors_are_black_and_one_flag:
            pred = np.where(pred != 0, only_used_color_num, pred)
        processed_preds.append(pred)
    return processed_preds


# In[151]:


def colors_are_black_and_one(matrix):
    unique_nums = np.unique(matrix).tolist()
    if 0 not in unique_nums:
        return False, None
        
    unique_nums.remove(0)
    if len(unique_nums) == 1:
        return True, unique_nums[0]
    else:
        return False, None

def search_inout_used_colors(task_train):
    all_input_same_colors = True
    all_output_same_colors = True
    all_input_and_output_same_colors = True
    each_input_and_output_same_colors = True
    input_unique_nums=[]
    for i, inout in enumerate(task_train):
        if each_input_and_output_same_colors:
            if np.unique(inout['input']).tolist() != np.unique(inout['output']).tolist():
                each_input_and_output_same_colors = False
                all_input_and_output_same_colors = False
        if i == 0:
            input_unique_nums = np.unique(inout['input']).tolist()
            output_unique_nums = np.unique(inout['output']).tolist()
            continue

        if input_unique_nums != np.unique(inout['input']).tolist():
            all_input_same_colors = False
            all_input_and_output_same_colors = False
        if output_unique_nums != np.unique(inout['output']).tolist():
            all_output_same_colors = False
            all_input_and_output_same_colors = False
    return all_input_same_colors, all_output_same_colors, all_input_and_output_same_colors, each_input_and_output_same_colors            


# # mirror aug

# In[152]:


def fliplr_aug(train):
    return mirror_aug(train, np.fliplr)

def flipud_aug(train):
    return mirror_aug(train, np.flipud)

def flip_aug(train):
    return mirror_aug(train, np.flip)

def transpose_aug(train):
    return mirror_aug(train, np.transpose)

def mirror_aug(train, aug_func):
    inouts = []
    for j, inout_origin in enumerate(train):
        inout = copy.deepcopy(inout_origin)
        same_flag = False
        inout['input'] = aug_func(inout['input']).tolist()
        inout['output'] = aug_func(inout['output']).tolist()
        if inout['input'] != inout_origin['input'].tolist():
            for io in inouts:
                if io['input'] == inout['input']:
                    same_flag = True
                    break
            if not same_flag:
                for inout_other in train:
                    if inout_other['input'].tolist() == inout['input']:
                        same_flag = True
                        break
            if not same_flag:
                inouts.append({'input': inout['input'], 'output': inout['output']})
    return inouts


# # color aug

# In[153]:


def has_duplicates(seq):
    return len(seq) != len(set(seq))

def color_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_color_augument(task['train'])

def one_train_color_augument(train, verbose=False):
    two_colors = []
    for pair in combinations([0,1,2,3,4,5,6,7,8,9], 2):
        two_colors.append(pair)
    color_pairs_list = []
    for i in range(1, 6):
        for color_pairs in combinations(two_colors, i):
            if has_duplicates(list(itertools.chain.from_iterable(list(color_pairs)))):
                continue
            color_pairs_list.append(color_pairs)
    inouts = []
    for color_pairs in color_pairs_list:
        for j, inout_origin in enumerate(train):
            inout = copy.deepcopy(inout_origin)
            same_flag = False
            inout['input'] = np.array(inout['input'])
            inout['output'] = np.array(inout['output'])
            tmp_inp = copy.deepcopy(inout['input'])
            tmp_out = copy.deepcopy(inout['output'])

            for color_pair in color_pairs:
                inout['input'] = np.where(tmp_inp == color_pair[0], color_pair[1], inout['input'])
                inout['input'] = np.where(tmp_inp == color_pair[1], color_pair[0], inout['input'])
                inout['output'] = np.where(tmp_out == color_pair[0], color_pair[1], inout['output'])
                inout['output'] = np.where(tmp_out == color_pair[1], color_pair[0], inout['output'])

            inout['input'] = inout['input'].tolist()
            inout['output'] = inout['output'].tolist()
            if inout['input'] != tmp_inp.tolist():
                for io in inouts:
                    if io['input'] == inout['input']:
                        same_flag = True
                        break
                if not same_flag:
                    for inout_other in train:
                        if inout_other['input'].tolist() == inout['input']:
                            same_flag = True
                            break
                if not same_flag:
                    inouts.append({'input': inout['input'], 'output': inout['output']})
    if verbose:
        print(len(inouts))

    return inouts


def super_heavy_color_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_super_heavy_color_augument(task['train'])

def one_train_super_heavy_color_augument(train, verbose=False):
    two_colors = []
    for pair in combinations([1,2,3,4,5,6,7,8,9], 2):
        two_colors.append(pair)
    color_pairs_list = []
    for i in [2, 3, 4]:
        for color_pairs in combinations(two_colors, i):
            if has_duplicates(list(itertools.chain.from_iterable(list(color_pairs)))):
                continue
            color_pairs_list.append(color_pairs)
    inouts = []
    for color_pairs in color_pairs_list:
        for j, inout_origin in enumerate(train):
            inout = copy.deepcopy(inout_origin)
            same_flag = False
            inout['input'] = np.array(inout['input'])
            inout['output'] = np.array(inout['output'])
            tmp_inp = copy.deepcopy(inout['input'])
            tmp_out = copy.deepcopy(inout['output'])

            for color_pair in color_pairs:
                inout['input'] = np.where(tmp_inp == color_pair[0], color_pair[1], inout['input'])
                inout['input'] = np.where(tmp_inp == color_pair[1], color_pair[0], inout['input'])
                inout['output'] = np.where(tmp_out == color_pair[0], color_pair[1], inout['output'])
                inout['output'] = np.where(tmp_out == color_pair[1], color_pair[0], inout['output'])

            inout['input'] = inout['input'].tolist()
            inout['output'] = inout['output'].tolist()
            if inout['input'] != tmp_inp.tolist():
                for io in inouts:
                    if io['input'] == inout['input']:
                        same_flag = True
                        break
                if not same_flag:
                    for inout_other in train:
                        if inout_other['input'].tolist() == inout['input']:
                            same_flag = True
                            break
                if not same_flag:
                    inouts.append({'input': inout['input'], 'output': inout['output']})
    if verbose:
        print(len(inouts))

    return inouts


def heavy_color_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_heavy_color_augument(task['train'])

def one_train_heavy_color_augument(train, verbose=False):
    two_colors = []
    for pair in combinations([1,2,3,4,5,6,7,8,9], 2):
        two_colors.append(pair)
    color_pairs_list = []
    for i in [3, 4]:
        for color_pairs in combinations(two_colors, i):
            if has_duplicates(list(itertools.chain.from_iterable(list(color_pairs)))):
                continue
            color_pairs_list.append(color_pairs)

    inouts = []
    for color_pairs in color_pairs_list:
        for j, inout_origin in enumerate(train):
            inout = copy.deepcopy(inout_origin)
            same_flag = False
            inout['input'] = np.array(inout['input'])
            inout['output'] = np.array(inout['output'])
            tmp_inp = copy.deepcopy(inout['input'])
            tmp_out = copy.deepcopy(inout['output'])

            for color_pair in color_pairs:
                inout['input'] = np.where(tmp_inp == color_pair[0], color_pair[1], inout['input'])
                inout['input'] = np.where(tmp_inp == color_pair[1], color_pair[0], inout['input'])
                inout['output'] = np.where(tmp_out == color_pair[0], color_pair[1], inout['output'])
                inout['output'] = np.where(tmp_out == color_pair[1], color_pair[0], inout['output'])

            inout['input'] = inout['input'].tolist()
            inout['output'] = inout['output'].tolist()
            if inout['input'] != tmp_inp.tolist():
                for io in inouts:
                    if io['input'] == inout['input']:
                        same_flag = True
                        break
                if not same_flag:
                    for inout_other in train:
                        if inout_other['input'].tolist() == inout['input']:
                            same_flag = True
                            break
                if not same_flag:
                    inouts.append({'input': inout['input'], 'output': inout['output']})
    if verbose:
        print(len(inouts))

    return inouts


def medium_heavy_color_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_medium_heavy_color_augument(task['train'])

def one_train_medium_heavy_color_augument(train, verbose=False):
    two_colors = []
    for pair in combinations([1,2,3,4,5,6,7,8,9], 2):
        two_colors.append(pair)
    color_pairs_list = []
    for color_pairs in combinations(two_colors, 2):
        if has_duplicates(list(itertools.chain.from_iterable(list(color_pairs)))):
            continue
        color_pairs_list.append(color_pairs)
    inouts = []
    for color_pairs in color_pairs_list:
        for j, inout_origin in enumerate(train):
            inout = copy.deepcopy(inout_origin)
            same_flag = False
            inout['input'] = np.array(inout['input'])
            inout['output'] = np.array(inout['output'])
            tmp_inp = copy.deepcopy(inout['input'])
            tmp_out = copy.deepcopy(inout['output'])

            for color_pair in color_pairs:
                inout['input'] = np.where(tmp_inp == color_pair[0], color_pair[1], inout['input'])
                inout['input'] = np.where(tmp_inp == color_pair[1], color_pair[0], inout['input'])
                inout['output'] = np.where(tmp_out == color_pair[0], color_pair[1], inout['output'])
                inout['output'] = np.where(tmp_out == color_pair[1], color_pair[0], inout['output'])

            inout['input'] = inout['input'].tolist()
            inout['output'] = inout['output'].tolist()
            if inout['input'] != tmp_inp.tolist():
                for io in inouts:
                    if io['input'] == inout['input']:
                        same_flag = True
                        break
                if not same_flag:
                    for inout_other in train:
                        if inout_other['input'].tolist() == inout['input']:
                            same_flag = True
                            break
                if not same_flag:
                    inouts.append({'input': inout['input'], 'output': inout['output']})
    if verbose:
        print(len(inouts))

    return inouts


def medium_color_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_medium_color_augument(task['train'])

def one_train_medium_color_augument(train, verbose=False):
    color_pairs_list = [
        [[2, 3], [4, 5], [6, 7], [8, 9]],
        [[1, 3], [4, 6], [8, 7], [2, 9]],
        [[6, 3], [2, 5], [4, 7], [1, 9]],
        [[2, 4], [7, 5], [6, 9], [8, 1]],
        [[1, 4], [5, 6], [8, 3], [2, 7]],
        [[7, 3], [6, 1], [8, 4], [5, 9]],
    ]
    inouts = []
    for color_pairs in color_pairs_list:
        if has_duplicates(list(itertools.chain.from_iterable(list(color_pairs)))):
            continue
        for j, inout_origin in enumerate(train):
            inout = copy.deepcopy(inout_origin)
            same_flag = False
            inout['input'] = np.array(inout['input'])
            inout['output'] = np.array(inout['output'])
            tmp_inp = copy.deepcopy(inout['input'])
            tmp_out = copy.deepcopy(inout['output'])

            for color_pair in color_pairs:
                inout['input'] = np.where(tmp_inp == color_pair[0], color_pair[1], inout['input'])
                inout['input'] = np.where(tmp_inp == color_pair[1], color_pair[0], inout['input'])
                inout['output'] = np.where(tmp_out == color_pair[0], color_pair[1], inout['output'])
                inout['output'] = np.where(tmp_out == color_pair[1], color_pair[0], inout['output'])

            inout['input'] = inout['input'].tolist()
            inout['output'] = inout['output'].tolist()
            if inout['input'] != tmp_inp.tolist():
                for io in inouts:
                    if io['input'] == inout['input']:
                        same_flag = True
                        break
                if not same_flag:
                    for inout_other in train:
                        if inout_other['input'].tolist() == inout['input']:
                            same_flag = True
                            break
                if not same_flag:
                    inouts.append({'input': inout['input'], 'output': inout['output']})
    if verbose:
        print(len(inouts))

    return inouts


def medium_light_color_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_medium_light_color_augument(task['train'])

def one_train_medium_light_color_augument(train, verbose=False):
    color_pairs_list = [
        [[2, 3], [4, 5], [6, 7], [8, 9]],
        [[1, 3], [4, 6], [8, 7], [2, 9]],
        [[6, 3], [2, 5], [4, 7], [1, 9]],
    ]
    inouts = []
    for color_pairs in color_pairs_list:
        if has_duplicates(list(itertools.chain.from_iterable(list(color_pairs)))):
            continue
        for j, inout_origin in enumerate(train):
            inout = copy.deepcopy(inout_origin)
            same_flag = False
            inout['input'] = np.array(inout['input'])
            inout['output'] = np.array(inout['output'])
            tmp_inp = copy.deepcopy(inout['input'])
            tmp_out = copy.deepcopy(inout['output'])

            for color_pair in color_pairs:
                inout['input'] = np.where(tmp_inp == color_pair[0], color_pair[1], inout['input'])
                inout['input'] = np.where(tmp_inp == color_pair[1], color_pair[0], inout['input'])
                inout['output'] = np.where(tmp_out == color_pair[0], color_pair[1], inout['output'])
                inout['output'] = np.where(tmp_out == color_pair[1], color_pair[0], inout['output'])

            inout['input'] = inout['input'].tolist()
            inout['output'] = inout['output'].tolist()
            if inout['input'] != tmp_inp.tolist():
                for io in inouts:
                    if io['input'] == inout['input']:
                        same_flag = True
                        break
                if not same_flag:
                    for inout_other in train:
                        if inout_other['input'].tolist() == inout['input']:
                            same_flag = True
                            break
                if not same_flag:
                    inouts.append({'input': inout['input'], 'output': inout['output']})
    if verbose:
        print(len(inouts))

    return inouts


def light_color_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_light_color_augument(task['train'])

def one_train_light_color_augument(train, verbose=False):
    color_pairs_list = [
        [[2, 3], [4, 5], [6, 7], [8, 9]],
    ]
    inouts = []
    for color_pairs in color_pairs_list:
        if has_duplicates(list(itertools.chain.from_iterable(list(color_pairs)))):
            continue
        for j, inout_origin in enumerate(train):
            inout = copy.deepcopy(inout_origin)
            same_flag = False
            inout['input'] = np.array(inout['input'])
            inout['output'] = np.array(inout['output'])
            tmp_inp = copy.deepcopy(inout['input'])
            tmp_out = copy.deepcopy(inout['output'])

            for color_pair in color_pairs:
                inout['input'] = np.where(tmp_inp == color_pair[0], color_pair[1], inout['input'])
                inout['input'] = np.where(tmp_inp == color_pair[1], color_pair[0], inout['input'])
                inout['output'] = np.where(tmp_out == color_pair[0], color_pair[1], inout['output'])
                inout['output'] = np.where(tmp_out == color_pair[1], color_pair[0], inout['output'])

            inout['input'] = inout['input'].tolist()
            inout['output'] = inout['output'].tolist()
            if inout['input'] != tmp_inp.tolist():
                for io in inouts:
                    if io['input'] == inout['input']:
                        same_flag = True
                        break
                if not same_flag:
                    for inout_other in train:
                        if inout_other['input'].tolist() == inout['input']:
                            same_flag = True
                            break
                if not same_flag:
                    inouts.append({'input': inout['input'], 'output': inout['output']})
    if verbose:
        print(len(inouts))

    return inouts

def mirror_augument(tasks):
    for idx, task in tasks.iteritems():
        task['train'] += one_train_mirror_augument(task['train'])

def one_train_mirror_augument(train):
    aug_func_petterns = [np.transpose, mirror_x, mirror_y]
    inouts = []
    for i in range(len(aug_func_petterns)):
        for func_combi in combinations(aug_func_petterns, i+1):
            for j, inout_origin in enumerate(train):
                inout = copy.deepcopy(inout_origin)
                same_flag = False
                for func in list(func_combi):
                    inout['input'] = func(inout['input']).tolist()
                    inout['output'] = func(inout['output']).tolist()
                if inout['input'] != inout_origin['input']:
                    for io in inouts:
                        if io['input'] == inout['input']:
                            same_flag = True
                            break
                    if not same_flag:
                        for inout_other in train:
                            if inout_other['input'].tolist() == inout['input']:
                                same_flag = True
                                break
                    if not same_flag:
                        inouts.append({'input': inout['input'], 'output': inout['output']})

    return inouts


# # get_mode_preds

# In[154]:


def get_mode_preds(preds_list):
    preds = []
    origin_shape_map = {}
    each_question_preds = defaultdict(list)
    for i in range(len(preds_list[0])):
        origin_shape_map[i] = np.array(preds_list[0][i]).shape
        for preds in preds_list:
            each_question_preds[i].append(np.array(preds[i]).reshape(-1))
    mode_preds = []
    
    for i in range(len(preds_list[0])):
        ans = []
        for j in range(len(each_question_preds[i][0])):
            ms = [m[j] for m in each_question_preds[i]]
            ans.append(np.argmax(np.bincount(ms)))
        mode_preds.append(np.array(ans).reshape(origin_shape_map[i]).tolist())
    return mode_preds


# # final score update

# In[155]:


def final_score_update(test_tasks, preds_list, final_score_map, idx, success_map):
    print(f'{idx}, 順番: mode, (aug CA), aug func0')
    for i in range(len(preds_list[0])):
        pred_str = ''
        for j, preds in enumerate(reversed(preds_list)):
            pred = np.array(preds[i])
            if test_tasks[i]['output'] == pred.tolist():
                success_map[f'{idx}_{i}'] = True            
            similarity = match_rate(pred, test_tasks[i]['output'])
            print(f'similarity: {similarity}')
            if (idx not in final_score_map) or (final_score_map.get(idx, 101) < similarity):
                final_score_map[idx] = similarity


# # final_train_and_predict

# In[156]:


each_preds = defaultdict(lambda: defaultdict(list))
ca_skip = False
def final_train_and_predict(task_train, task_train2, task_train_aug, task_test, task_test2, idx, success_map, final_score_map, final=False, promising=False, origin_task=None):
    funcs0 = []
    funcs1 = []
    preds_list = []
    mode_preds_list = []
    if not final:
#         st()
        ts = TaskSolver()
        ts.train(task_train, preprocess_funcs=funcs0)
        preds = ts.predict(task_test, preprocess_funcs=funcs0, success_map=success_map, idx=idx, final_score_map=final_score_map)
        return preds 
    
    # 別のNNに通す
#     if promising:
#         train = copy.deepcopy(task_train_aug)
#         test = copy.deepcopy(task_test)
#         for func in funcs0:
#             for inout in train:
#                 inout['input'] = func(inout['input'])
#             for inout in test:
#                 inout['input'] = func(inout['input'])
#         preds = test_nn(train, test)
#         preds = post_processes(preds, origin_task)
#         each_preds[idx]['another NN'] = preds
#         preds_list.append(preds)
#         mode_preds_list.append(preds)
#         preds = preds_to_str_only0(preds, idx)

    # not aug, funcs0
#     ts = TaskSolver()
#     ts.train(task_train, preprocess_funcs=funcs0, final=final)
#     preds3 = ts.predict(task_test, preprocess_funcs=funcs0, success_map=success_map, idx=idx, final_score_map=final_score_map, final=final)
#     preds3 = post_processes(preds3, origin_task)
#     preds_list.append(preds3)
#     mode_preds_list.append(preds3)
#     each_preds[idx]['not aug, normal NN, funcs0'] = preds3
#     st()
    
    # not aug, funcs1
#     ts = TaskSolver()
#     ts.train(task_train2, preprocess_funcs=funcs1, final=final)
#     preds4 = ts.predict(task_test2, preprocess_funcs=funcs1, success_map=success_map, idx=idx, final_score_map=final_score_map, final=final)
#     preds4 = post_processes(preds4, origin_task)
#     each_preds[idx]['not aug, normal NN, funcs1'] = preds4
#     preds_list.append(preds4)
#     mode_preds_list.append(preds4)

    # not aug, func0
    ts = TaskSolver()
    ts.train(task_train, preprocess_funcs=funcs1, final=final)
    preds4 = ts.predict(task_test, preprocess_funcs=funcs1, success_map=success_map, idx=idx, final_score_map=final_score_map, final=final)
    preds4 = post_processes(preds4, origin_task)
    each_preds[idx]['not aug, normal NN, funcs1'] = preds4
    preds_list.append(preds4)
    mode_preds_list.append(preds4)    
    
    # aug, funcs0
    ts = TaskSolver()
    ts.train(task_train_aug, preprocess_funcs=funcs0, final=final)
    preds2 = ts.predict(task_test, preprocess_funcs=funcs0, success_map=success_map, idx=idx, final_score_map=final_score_map, final=final)
    preds2 = post_processes(preds2, origin_task)
    preds_list.append(preds2)
    mode_preds_list.append(preds2)
    mode_preds_list.append(preds2) # weight2
    each_preds[idx]['aug, normal NN, funcs0'] = preds2

    if (len(task_train_aug) < 200) & (not ca_skip):
#     if False:
#         print('CA1 start')
        # not aug CA
        # TODO: 消すか検証する
#         train_changed = []
#         test_changed = []
#         for inout in task_train:
#             io = copy.deepcopy(inout)
#             for f in funcs0:
#                 io['input'] = f(io['input'])
#             train_changed.append(io)
#         for inout in task_test:
#             io = copy.deepcopy(inout)
#             for f in funcs0:
#                 io['input'] = f(io['input'])
#             test_changed.append(io)
#         model, _, _ = solve_task(train_changed)
#         preds0 = predict(model, test_changed)
#         preds0 = post_processes(preds0, origin_task)
#         each_preds[idx]['not aug CA'] = preds0
#         preds_list.append(preds0)
#         preds_list.append(preds0)

        # aug, CA
        print('CA1 start')
#         train_changed = []
#         test_changed = []
#         for inout in task_train_aug:
#             io = copy.deepcopy(inout)
#             for f in funcs0:
#                 io['input'] = f(io['input'])
#             train_changed.append(io)
#         for inout in task_test:
#             io = copy.deepcopy(inout)
#             for f in funcs0:
#                 io['input'] = f(io['input'])
#             test_changed.append(io)
        model, _, _ = solve_task(task_train_aug)
        preds1 = predict(model, task_test)
        preds1 = post_processes(preds1, origin_task)
        preds_list.append(preds1)
        mode_preds_list.append(preds1)
        mode_preds_list.append(preds1) # weight2
        each_preds[idx]['aug CA'] = preds1

    preds_mode = get_mode_preds(mode_preds_list)
    each_preds[idx]['mode matrix'] = preds_mode
    preds_list.append(preds_mode)
    if 'output' in task_test[0]:
#         st()
        final_score_update(task_test, preds_list, final_score_map, idx, success_map)

    preds = preds_to_str(preds_list, idx)
#     st()

    return preds


# # apply_aug

# In[157]:


def apply_mirror_aug(train, preprocess_best_score, idx, use_transpose_flag, promising_map):
    use_mirror_augs = []
    for aug_func in [flipud_aug, fliplr_aug, flip_aug]:
        inouts = aug_func(train)
#         st()
        similarity = get_similarity(train+inouts, [], 'searching_mirror_aug_'+idx, {})
        print(f'{aug_func}: ------> {similarity}')
        if similarity > 0.99:
            promising_map[idx] = True
        if (similarity > preprocess_best_score) or ((similarity==1) & (preprocess_best_score != 1)):
            use_mirror_augs.append(aug_func)
    if use_transpose_flag:
        print(transpose_aug)
        inouts = transpose_aug(train)
        similarity = get_similarity(train+inouts, [], 'searching_transpose_aug_'+idx, {})
        print(similarity, preprocess_best_score)
        if (similarity > preprocess_best_score) or ((similarity==1) & (preprocess_best_score != 1)):
            use_mirror_augs.append(transpose_aug)
        if similarity > 0.99:
            promising_map[idx] = True
    return use_mirror_augs

def apply_transpose_aug(train):
    for inout_origin in train:
        inout = copy.deepcopy(inout_origin)
        m, n = np.array(inout['input']).shape
        if m != n:
            return False
    return True

def apply_color_aug(train, preprocess_best_score, best_aug_score_map, idx, promising_map):
    best_aug_score_map[idx] = 0
    use_inouts = []
    use_aug_func = return_arg
    skip_heavy_flag = False
    heavy_funcs = [one_train_medium_heavy_color_augument, one_train_heavy_color_augument, one_train_super_heavy_color_augument]
#     for aug_func in [one_train_light_color_augument, one_train_medium_light_color_augument, one_train_medium_color_augument, one_train_medium_heavy_color_augument, one_train_heavy_color_augument, one_train_super_heavy_color_augument]:
    for aug_func in [one_train_light_color_augument, one_train_medium_light_color_augument, one_train_medium_color_augument, one_train_medium_heavy_color_augument, one_train_heavy_color_augument]:
#     for aug_func in [one_train_medium_heavy_color_augument, one_train_light_color_augument, one_train_medium_light_color_augument, one_train_medium_color_augument]:
        if aug_func in heavy_funcs:
            if skip_heavy_flag == True:
                continue
            if (best_aug_score_map[idx] < 0.997) or (best_aug_score_map[idx] < preprocess_best_score+0.04):
                skip_heavy_flag = True
                continue
        inouts = aug_func(train)
        scores = []
        # 重い場合時間がかかるので学習は一度にする
        if aug_func in heavy_funcs:
            ts = TaskSolver()
            tmp_train = train + inouts
            if len(tmp_train) < 10:
                continue
            val_train, tmp_train = tmp_train[:3], tmp_train[3:]
            ts.train(tmp_train, preprocess_funcs=[])
            for i in range(3):
                preds = ts.predict([val_train[i]], preprocess_funcs=[])
                similarity = match_rate(preds[0], val_train[i]['output'])
                scores.append(similarity)
        # 軽い場合はなるべくpreと条件を揃えたいので都度学習
        else:
            for i in range(3):
                similarity = train_and_evaluate(train+inouts, [], seed=i, idx='searching_aug', success_map={})
                scores.append(similarity)
        score = np.mean(scores)
        print(f'{aug_func}: ------> {score}')
        if score > 0.9999:
            promising_map[idx] = True
            return use_inouts, aug_func
        if score < 0.8:
            return use_inouts, use_aug_func
        if (score > best_aug_score_map[idx]) & (score > preprocess_best_score):
            best_aug_score_map[idx] = score
            use_inouts = inouts
            use_aug_func = aug_func
        # 常に更新、かつ直前で0.99以上じゃない場合、重い処理をスキップ
        if score < best_aug_score_map[idx]:
            skip_heavy_flag = True
        if (aug_func == one_train_medium_heavy_color_augument) & ((score < 0.997)):
            skip_heavy_flag = True

    if best_aug_score_map[idx] > 0.98:
        promising_map[idx] = True
    return use_inouts, use_aug_func


# # train_and_evaluate

# In[158]:


def train_and_evaluate(train, func_combi, seed, idx, success_map, search_func=False):
    ts = TaskSolver()
    tmp_train = copy.deepcopy(train)
    val_train = [tmp_train.pop(seed % len(tmp_train))]
    ts.train(tmp_train, preprocess_funcs=func_combi)
    preds = ts.predict(val_train, preprocess_funcs=func_combi) # idxを渡すとsimilarityがprintされる
    return match_rate(preds[0], val_train[0]['output'])


# # add fill closed area

# In[159]:


def add_fill_closed_area(train):
    inp_origin = train[0]['input']
    out = train[0]['output']
    apply_flag = False
    for func in [np.array, np.flip, np.fliplr, np.flipud]:
        inp = np.array(inp_origin.copy())
        inp = func(inp)
        if len(set([ele for ele in np.array(out).reshape(-1) - np.array(inp).reshape(-1) if ele != 0])) == 1:
            fill_color = [ele for ele in np.array(out).reshape(-1) - np.array(inp).reshape(-1) if ele != 0][0]
            apply_flag = True
            break
    if not apply_flag:
        return [inouts_array]
    best_score = 0
    best_enclose_color = 0

    for enclose_color in range(1,10):
        inp_copy = inp.copy()
        if enclose_color == fill_color:
            continue
        H, W = inp_copy.shape
        Dy = [0, -1, 0, 1]
        Dx = [1, 0, -1, 0]
        arr_padded = np.pad(inp_copy, ((1,1),(1,1)), "constant", constant_values=0)
        searched = np.zeros(arr_padded.shape, dtype=bool)
        searched[0, 0] = True
        q = [(0, 0)]
        while q:
            y, x = q.pop()
            for dy, dx in zip(Dy, Dx):
                y_, x_ = y+dy, x+dx
                if not 0 <= y_ < H+2 or not 0 <= x_ < W+2:
                    continue
                if not searched[y_][x_] and arr_padded[y_][x_]==0:
                    q.append((y_, x_))
                    searched[y_, x_] = True
        res = searched[1:-1, 1:-1]
        res |= inp_copy==enclose_color
        inp_copy[~res]=fill_color
        similarity = match_rate(inp_copy, out)
        if similarity > best_score:
            best_score = similarity
            best_enclose_color = enclose_color
    def fill_closed_area(task_train):
        for inout in task_train:
            inp = inout['input']
            inp = np.array(inp)
            H, W = inp.shape
            Dy = [0, -1, 0, 1]
            Dx = [1, 0, -1, 0]
            arr_padded = np.pad(inp, ((1,1),(1,1)), "constant", constant_values=0)
            searched = np.zeros(arr_padded.shape, dtype=bool)
            searched[0, 0] = True
            q = [(0, 0)]
            while q:
                y, x = q.pop()
                for dy, dx in zip(Dy, Dx):
                    y_, x_ = y+dy, x+dx
                    if not 0 <= y_ < H+2 or not 0 <= x_ < W+2:
                        continue
                    if not searched[y_][x_] and arr_padded[y_][x_]==0:
                        q.append((y_, x_))
                        searched[y_, x_] = True
            res = searched[1:-1, 1:-1]
            res |= inp==best_enclose_color

            inp[~res] = fill_color
    #         st()
            inout['input'] = inp
        return task_train

    return [inouts_array, fill_closed_area]


# In[160]:


def add_train0_double(task_train, m1,n1,m2,n2):
    if not((m2 >= m1) & (n2 >= n1) & (m2 % m1 == 0) & (n2 % n1 == 0)):
        return []
    for inout in task_train:
        m1_, n1_ = np.array(inout['input']).shape
        m2_, n2_ = np.array(inout['output']).shape
        if (m2 / m1 != m2_ / m1_) or (n2 / n1 != n2_ / n1_):
            return []    
    def train0_double(task_train):
        for inout in task_train:
            x = inout['input']
            x = np.array(x)
            m, n = m2//m1, n2//n1
            x_upsampled = x.repeat(m, axis=0).repeat(n, axis=1)
            x_tiled = np.tile(x, (m, n))
            y = x_upsampled & x_tiled
            inout['input'] = y    
        return task_train
    return [train0_double]


# # add double func

# In[161]:


# 前提：same_shape_inputs, same_shape_outputs
def add_gdc_double_funcs_with_same_shape_inputs(task_train, m1, n1, m2, n2):
    if (m1==m2) & (n1==n2):
        return []
    m_gdc = math.gcd(m1, m2)
    n_gdc = math.gcd(n1, n2)
    if ((m_gdc==1) or (n_gdc==1)):
        return []
    if not ((m2 >= m1) & (n2 >= n1)):
        return []
    transpose_funcs = [np.array]
    if m1 == n1:
        transpose_funcs.append(np.transpose)
    transpose_flip_map = {}
    flip_cols_map = {}
    flip_rows_map = {}
    inp_mn_map = {}
    for m in range(m2 // m_gdc):
        for n in range(n2 // n_gdc):
            transpose_flip_map[str(m)+','+str(n)] = [np.array] # 初期値
            flip_cols_map[str(m)+','+str(n)] = []
            flip_rows_map[str(m)+','+str(n)] = []
            inp_mn_map[str(m)+','+str(n)] = [0,0]
            correct_flag = False
            pickup_output = np.array(task_train[0]['output'])[m*m_gdc:(m+1)*m_gdc, n*n_gdc:(n+1)*n_gdc]
            best_score = 0
            for transpose_func in transpose_funcs:
                if correct_flag:
                    break
                for inp_m in range(m1 // m_gdc):
                    if correct_flag:
                        break
                    for inp_n in range(n1 // n_gdc):
                        if correct_flag:
                            break
                        inp_copy = np.array(copy.deepcopy(task_train[0]['input']))
                        inp_copy = inp_copy[inp_m*m_gdc:(inp_m+1)*m_gdc, inp_n*n_gdc:(inp_n+1)*n_gdc]
                        inp_copy = transpose_func(inp_copy)
                        for flip_func in [np.flip, np.flipud, np.fliplr, np.array]:
                            if correct_flag:
                                break
                            inp_copy_copy = copy.deepcopy(inp_copy)
                            inp_copy_copy = flip_func(inp_copy_copy)
                            if pickup_output.tolist() == inp_copy_copy.tolist():
                                correct_flag = True
                                transpose_flip_map[str(m)+','+str(n)] = [transpose_func, flip_func]
                                inp_mn_map[str(m)+','+str(n)] = [inp_n, inp_m]
                                flip_cols_map[str(m)+','+str(n)] = []
                                flip_rows_map[str(m)+','+str(n)] = []
                                break
                            similarity = match_rate(pickup_output, inp_copy_copy)
                            if best_score < similarity:
                                best_score = similarity
                                transpose_flip_map[str(m)+','+str(n)] = [transpose_func, flip_func]
                                inp_mn_map[str(m)+','+str(n)] = [inp_n, inp_m]
                                flip_cols_map[str(m)+','+str(n)] = []
                                flip_rows_map[str(m)+','+str(n)] = []
#                                 st()
                        for i in range(m_gdc+1):
                            if correct_flag:
                                break
                            for change_rows in combinations(range(m_gdc), i):
                                if correct_flag:
                                    break
                                change_rows = list(change_rows)
                                for j in range(n_gdc+1):
                                    if correct_flag:
                                        break
                                    for change_cols in combinations(range(n_gdc), j):
                                        change_cols = list(change_cols)
                                        inp_copy_copy = copy.deepcopy(inp_copy)
                                        inp_copy_copy[change_rows, :] = np.fliplr(inp_copy_copy[change_rows, :])
                                        inp_copy_copy[:, change_cols] = np.flipud(inp_copy_copy[:, change_cols])
                                        if pickup_output.tolist() == inp_copy_copy.tolist():
                                            correct_flag = True
                                            transpose_flip_map[str(m)+','+str(n)] = [transpose_func, flip_func]
                                            inp_mn_map[str(m)+','+str(n)] = [inp_n, inp_m]
                                            flip_cols_map[str(m)+','+str(n)] = change_cols
                                            flip_rows_map[str(m)+','+str(n)] = change_rows  
                                            break
                                        
                                        similarity = match_rate(pickup_output, inp_copy_copy)
                                        if best_score < similarity:
                                            best_score = similarity
                                            transpose_flip_map[str(m)+','+str(n)] = [transpose_func, flip_func]
                                            inp_mn_map[str(m)+','+str(n)] = [inp_n, inp_m]
                                            flip_cols_map[str(m)+','+str(n)] = change_cols
                                            flip_rows_map[str(m)+','+str(n)] = change_rows

    def double(task_train):
        for inout in task_train:
            inp = inout['input']
            ans = np.zeros((m2, n2)).astype(int)
            for coordinate, transpose_funcs in transpose_flip_map.items():
                m, n = coordinate.split(',')
                m, n = int(m), int(n)
                inp_copy = np.array(copy.deepcopy(inp))
                inp_n, inp_m = inp_mn_map[coordinate]
                inp_copy = inp_copy[inp_m*m_gdc:(inp_m+1)*m_gdc, inp_n*n_gdc:(inp_n+1)*n_gdc]
                for transpose_func in transpose_funcs:
                    inp_copy = transpose_func(inp_copy)
                change_cols = flip_cols_map[coordinate]
                change_rows = flip_rows_map[coordinate]
                inp_copy[:, change_cols] = np.flipud(inp_copy[:, change_cols])
                inp_copy[change_rows, :] = np.fliplr(inp_copy[change_rows, :])
                ans[m*m_gdc:(m+1)*m_gdc, n*n_gdc:(n+1)*n_gdc] = inp_copy
            inout['input'] = ans
        return task_train

    return [double]


# In[162]:


# each shapeが違って割合が一緒の場合
def add_gcd_double_funcs(task_train, m1, n1, m2, n2):
    if (m2==m1) & (n2==n1):
        return []
    m = math.gcd(m1, m2)
    n = math.gcd(n1, n2)    
    if not ((m2 >= m1) & (n2 >= n1) & (m2 % m1 == 0) & (n2 % n1 == 0)):
        return []
    for inout in task_train:
        m1_, n1_ = np.array(inout['input']).shape
        m2_, n2_ = np.array(inout['output']).shape
        if (m2 / m1 != m2_ / m1_) or (n2 / n1 != n2_ / n1_):
            return []

    transpose_funcs = [np.array]
    if m1 == n1:
        transpose_funcs.append(np.transpose)
    transpose_flip_map = {}
    flip_cols_map = {}
    flip_rows_map = {}
    for m in range(m2 // m1):
        for n in range(n2 // n1):
            transpose_flip_map[str(m)+','+str(n)] = [np.array]
            correct_flag = False
            pickup_output = np.array(task_train[0]['output'])[m*m1:(m+1)*m1, n*n1:(n+1)*n1]
            best_score = 0
            for flip_func in [np.flip, np.flipud, np.fliplr, np.array]:
                for transpose_func in transpose_funcs:
                    if correct_flag:
                        break
                    inp_copy = copy.deepcopy(task_train[0]['input'])
                    inp_copy = transpose_func(inp_copy)
                    inp_copy = flip_func(inp_copy)
                    if pickup_output.tolist() == inp_copy.tolist():
                        correct_flag = True
                        transpose_flip_map[str(m)+','+str(n)] = [flip_func, transpose_func]
                        flip_cols_map[str(m)+','+str(n)] = []
                        flip_rows_map[str(m)+','+str(n)] = []
                    similarity = match_rate(pickup_output, inp_copy)
                    if best_score < similarity:
                        best_score = similarity
                        transpose_flip_map[str(m)+','+str(n)] = [flip_func, transpose_func]
                        flip_cols_map[str(m)+','+str(n)] = []
                        flip_rows_map[str(m)+','+str(n)] = []
    def double(task_train):
        for inout in task_train:
            inp = inout['input']
            inp_m, inp_n = np.array(inp).shape
            m, n = m2//m1 - 1, n2 // n1 - 1
            ans = np.zeros((inp_m*(int(m)+1), inp_n*(int(n)+1))).astype(int)
            for coordinate, transpose_funcs in transpose_flip_map.items():
                m, n = coordinate.split(',')
                m, n = int(m), int(n)
                inp_copy = copy.deepcopy(inp)
                for transpose_func in transpose_funcs:
                    inp_copy = transpose_func(inp_copy)

                ans[m*inp_m:(m+1)*inp_m, n*inp_n:(n+1)*inp_n] = inp_copy
            inout['input'] = ans
        return task_train

    return [double]


# In[163]:


# all_output_shape are same
# outputsの形は違うけど倍率が同じ場合は別の関数で
def add_double_funcs_with_same_shape_inputs(train, m1, n1, m2, n2):
    inp = np.array(train[0]['input'].copy())
    out = np.array(train[0]['output'].copy())
    tmp_ans = np.zeros(out.shape)
    ans = np.zeros(out.shape)
    best_ans_ms = []
    best_ans_ns = []
    best_m_flips= []
    best_n_flips = []
    if (m2==m1) & (n2==n1):
        return []
    # 縦横どちらかでもoutputが小さい場合は別関数で
    if not ((m2 >= m1) & (n2 >= n1)):
        return []
    for inout in train:
        m1_, n1_ = np.array(inout['input']).shape
        m2_, n2_ = np.array(inout['output']).shape
        if (m1 != m1_) or (n1 != n1_) or (m2 != m2_) or (n2 != n2_):
            return []

    for ans_m in range(m2):
        o = out[ans_m:ans_m+1, :n1]
        best_score = 0
        for inp_m in range(m1):
            i = inp[inp_m:inp_m+1, :]
#             st()
            for flip in [np.array, np.fliplr]:
                similarity = match_rate(flip(i), flip(o))
                if best_score < similarity:
                    best_score = similarity
                    best_ans_m = inp_m
                    best_flip = flip

        best_ans_ms.append(best_ans_m)
        best_m_flips.append(best_flip)

    for i, (flip, m) in enumerate(zip(best_m_flips, best_ans_ms)):
        tmp_ans[i:i+1, :n1] = flip(inp[m:m+1, :])
    for ans_n in range(n2):
        o = out[:, ans_n:ans_n+1]
        best_score = 0
        for inp_n in range(n1):
            i = tmp_ans[:, inp_n:inp_n+1]
            for flip in [np.array, np.fliplr]:
                similarity = match_rate(flip(i), flip(o))
                if best_score < similarity:
                    best_score = similarity
                    best_ans_n = inp_n
                    best_flip = flip
        best_ans_ns.append(best_ans_n)    
        best_n_flips.append(best_flip)
    def double(task_train):
        for inout in task_train:
            inp = inout['input']
            inp = np.array(inp)
            tmp_ans = np.zeros(out.shape)
            ans = np.zeros(out.shape)        
            for i, (flip, m) in enumerate(zip(best_m_flips, best_ans_ms)):
                tmp_ans[i:i+1, :n1] = flip(inp[m:m+1, :])

            for i, (flip, n) in enumerate(zip(best_n_flips, best_ans_ns)):
                ans[:, i:i+1] = flip(tmp_ans[:, n:n+1])
            inout['input'] = ans
        return task_train
    
    return [double]


# In[164]:


def get_period_length_vertical(arr):
    arr = np.array(arr)
    H, W = arr.shape
    period = 1
    while True:
        cycled = np.pad(arr[:period, :], ((0,H-period),(0,0)), 'wrap')
        if (cycled==arr).all():
            return period
        period += 1
        
def add_train2_double_vertical(task_train, m1, n1, m2, n2):
    if not((n1 == n2) & (m2 > m1)):
        return []
    
    def train2_double(task_train):
        for inout in task_train:
            inp = inout['input']
            inp = np.array(inp)
            period = get_period_length_vertical(inp)
            y = inp[:period, :]
            y = np.pad(y, ((0,m2-period),(0,0)), 'wrap')
            inout['input'] = y
        return task_train
    return [train2_double]


# In[165]:


def get_period_length_horizontal(arr):
    arr = np.array(arr)
    H, W = arr.shape
    period = 1
    while True:
        cycled = np.pad(arr[:, :period], ((0,0),(0,W-period)), 'wrap')
        if (cycled==arr).all():
            return period
        period += 1
        
def add_train2_double_horizontal(task_train, m1, n1, m2, n2):
    if not((m1 == m2) & (n2 > n1)):
        return []
    
    def train2_double(task_train):
        for inout in task_train:
            inp = inout['input']
            inp = np.array(inp)
            period = get_period_length_horizontal(inp)
            y = inp[:, :period]
            y = np.pad(y, ((0,0),(0,n2-period)), 'wrap')
            inout['input'] = y
        return task_train    
    return [train2_double]


# # add crop

# In[166]:


def crop_width(task_train_origin):
    for inout in task_train_origin:
        inout['input'] = np.array(inout['input'])
    try:
        task_train = copy.deepcopy(task_train_origin)
        for inout in task_train:
            inp = inout['input']
            max_width = 0
            max_width_color = 0
            for c in [color for color in np.unique(inp) if color != 0]:
                coords = np.argwhere(inp==c)
                x_min, y_min = coords.min(axis=0)
                x_max, y_max = coords.max(axis=0)        
                if max_width < y_max - y_min:
                    max_width = y_max - y_min
                    max_width_color = c
            coords = np.argwhere(inp==max_width_color)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)        
            inout['input'] = inp[x_min:x_max+1, y_min:y_max+1]
        return task_train
    except:
        return task_train_origin

def add_crop_width(task_train):
    for inout in task_train:
        try:
            inp = np.array(inout['input'])
            out = np.array(inout['output'])
            max_width = 0
            max_width_color = 0
            for c in [color for color in np.unique(inp) if color != 0]:
                coords = np.argwhere(inp==c)
                x_min, y_min = coords.min(axis=0)
                x_max, y_max = coords.max(axis=0)   
                if max_width < y_max - y_min:
                    max_width = y_max - y_min
                    max_width_color = c
            coords = np.argwhere(inp==max_width_color)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)               
            if (inp[x_min:x_max+1, y_min:y_max+1].shape != out.shape) & (inp[x_min:x_max+1, y_min:y_max+1].shape != out.T.shape):
                return []   
        except:
            return []

    return [crop_height]
def crop_height(task_train_origin):
    for inout in task_train_origin:
        inout['input'] = np.array(inout['input'])
    try:
        task_train = copy.deepcopy(task_train_origin)
        for inout in task_train:
            inp = inout['input']
            max_height = 0
            max_height_color = 0
            for c in [color for color in np.unique(inp) if color != 0]:
                coords = np.argwhere(inp==c)
                x_min, y_min = coords.min(axis=0)
                x_max, y_max = coords.max(axis=0)        
                if max_height < x_max - x_min:
                    max_height = x_max - x_min
                    max_height_color = c
            coords = np.argwhere(inp==max_height_color)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)        
            inout['input'] = inp[x_min:x_max+1, y_min:y_max+1]
        return task_train
    except:
        return task_train_origin

def add_crop_height(task_train):
    for inout in task_train:
        try:
            inp = np.array(inout['input'])
            out = np.array(inout['output'])
            max_height = 0
            max_height_color = 0
            for c in [color for color in np.unique(inp) if color != 0]:
                coords = np.argwhere(inp==c)
                x_min, y_min = coords.min(axis=0)
                x_max, y_max = coords.max(axis=0)   
                if max_height < x_max - x_min:
                    max_height = x_max - x_min
                    max_height_color = c
            coords = np.argwhere(inp==max_height_color)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)               
            if (inp[x_min:x_max+1, y_min:y_max+1].shape != out.shape) & (inp[x_min:x_max+1, y_min:y_max+1].shape != out.T.shape):
                return []   
        except:
            return []

    return [crop_height]
def crop_max(task_train_origin):
    for inout in task_train_origin:
        inout['input'] = np.array(inout['input'])
    task_train = copy.deepcopy(task_train_origin)
    try:
        for inout in task_train:
            a = inout['input']
            b = np.bincount(a.flatten(),minlength=10)
            b[0] = 255
            c = np.argsort(b)[-2]
            coords = np.argwhere(a==c)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            inout['input'] = a[x_min:x_max+1, y_min:y_max+1]
    except:
        return task_train_origin
    return task_train
    
def crop_min(task_train_origin):
    for inout in task_train_origin:
        inout['input'] = np.array(inout['input'])
    task_train = copy.deepcopy(task_train_origin)
    try:
        for inout in task_train:
            a = inout['input']
            b = np.bincount(a.flatten(),minlength=10)
            c = int(np.where(b==np.min(b[np.nonzero(b)]))[0])
            coords = np.argwhere(a==c)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            inout['input'] = a[x_min:x_max+1, y_min:y_max+1]
    except:
        return task_train_origin
    return task_train

def add_crop_max(task_train):
    for inout in task_train:
        try:
            inp = np.array(inout['input'])
            out = np.array(inout['output'])
            bin_c = np.bincount(inp.flatten(), minlength=10)

            bin_c[0] = 255
            c = np.argsort(bin_c)[-2]
            coords = np.argwhere(inp==c)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            inp = inp[x_min:x_max+1, y_min:y_max+1]  
            if (inp.shape != out.shape) & (inp.T.shape != out.shape):
                return []
        except:
            return []
    return [crop_max]

def add_crop_min(task_train):
    for inout in task_train:
        try: 
            inp = np.array(inout['input'])
            out = np.array(inout['output'])
            bin_c = np.bincount(inp.flatten(), minlength=10)
            c = int(np.where(bin_c==np.min(bin_c[np.nonzero(bin_c)]))[0])
            coords = np.argwhere(inp==c)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            inp = inp[x_min:x_max+1, y_min:y_max+1]
            if (inp.shape != out.shape) & (inp.T.shape != out.shape):
                return []
        except:
            return []
    return [crop_min]    


# # all_inputs_same_shape and all_outputs_same_shape

# In[167]:


def all_inputs_same_shape_and_all_outputs_same_shape(task_train):
    m1, n1 = np.array(task_train[0]['input']).shape
    m2, n2 = np.array(task_train[0]['output']).shape
    all_inputs_same_shape = True
    all_outputs_same_shape = True
    for inout in task_train:
        m1_, n1_ = np.array(inout['input']).shape
        m2_, n2_ = np.array(inout['output']).shape
        if (m1_ != m1) or (n1_ != n1):
            all_inputs_same_shape = False
        if (m2_ != m2) or (n2_ != n2):
            all_outputs_same_shape = False
    return all_inputs_same_shape, all_outputs_same_shape


# # add size change funcs

# In[168]:


def add_size_change_funcs(task_train, task_n):
    size_change_funcs = [inouts_array]
    m1, n1 = np.array(task_train[0]['input']).shape
    m2, n2 = np.array(task_train[0]['output']).shape
    all_inputs_same_shape = True
    all_outputs_same_shape = True
    for inout in task_train:
        m1_, n1_ = np.array(inout['input']).shape
        m2_, n2_ = np.array(inout['output']).shape
        if (m1_ != m1) or (n1_ != n1):
            all_inputs_same_shape = False       
        if (m2_ != m2) or (n2_ != n2):
            all_outputs_same_shape = False
    if m1 == n1 == m2 == n2:
        return size_change_funcs    

    # grid
    size_change_funcs += add_grid_funcs(m1, n1, m2, n2)
    
    # div
    if (m1 >= m2*2) or (n1 > n2*2):
        size_change_funcs += add_div_funcs(task_train, m1, n1, m2, n2)
    else:
        size_change_funcs += add_div_funcs2(task_train, m1, n1, m2, n2, vertically=True)
        size_change_funcs += add_div_funcs2(task_train, m1, n1, m2, n2, vertically=False)
    if (m1 > m2) & (n1 > n2) & (m1 < 20) & (n1 < 20):
        size_change_funcs += add_object_detect2(task_train)
        
    # double
    if all_inputs_same_shape & all_outputs_same_shape:
        size_change_funcs += add_train2_double_horizontal(task_train, m1, n1, m2, n2)
        size_change_funcs += add_train2_double_vertical(task_train, m1, n1, m2, n2)
        size_change_funcs += add_gdc_double_funcs_with_same_shape_inputs(task_train, m1, n1, m2, n2)
        size_change_funcs += add_recolor(task_train, task_n)
    else:
        size_change_funcs += add_gcd_double_funcs(task_train, m1, n1, m2, n2)
        
    size_change_funcs += add_train0_double(task_train, m1, n1, m2, n2)
    if (m1 >= m2) & (n1 >= n2):
        size_change_funcs += add_crop_max(task_train)
        size_change_funcs += add_crop_min(task_train)
        size_change_funcs += add_crop_height(task_train)
        size_change_funcs += add_crop_width(task_train)
        size_change_funcs += add_crop_by_line(task_train)
    # 他にもたくさん足す
    return size_change_funcs


# In[169]:


# for interface...
def return_arg(inp):
    return inp

def inouts_transpose(task_train):
    for inout in task_train:
        inout['input'] = np.transpose(inout['input'])
    return task_train

def inouts_array(task_train):
    for inout in task_train:
        inout['input'] = np.array(inout['input'])
    return task_train

def add_transpose(task_train):
    m1, n1 = np.array(task_train[0]['input']).shape
    m2, n2 = np.array(task_train[0]['output']).shape
    if (m1==n2) & (n1==m2):
        return [inouts_array, inouts_transpose]
    else:
        return [inouts_array]

def add_grid_funcs(m1, n1, m2, n2):
    grid_funcs = []
    if (m1 <= m2) & (n1 <= n2):
        return grid_funcs
    if (m2%m1 == 0) & (n2%n1 == 0):
        def griding(task_train):
            for inout in enumerate(task_train):
                inp = copy.deepcopy(inout['input'])
                m_grid = m2 // m1
                n_grid = n2 // n1
                inp = np.array(inp)
                m, n = inp.shape
                ans_tmp = np.zeros((m*m_grid, n), dtype='int')
                for i in range(m):
                    for j in range(m_grid):
                        ans_tmp[i*m_grid+j, :] = inp[i, :]
                ans = copy.deepcopy(ans_tmp)
                for stack_n in range(n_grid-1):
                    ans = np.hstack([ans, ans_tmp])
                for i in range(n):
                    for j in range(n_grid):
                        ans[:, i*n_grid+j] = ans_tmp[:, i]
                inout['input'] = ans
            return task_train
        grid_funcs.append(griding)

    if (m1 != n1) & (m2%n1 == 0) & (n2%m1 == 0):
        def transpose_griding(task_train):
            for inout in task_train:
                inp = copy.deepcopy(inp_o)
                m_grid = m2 // n1
                n_grid = n2 // m1
                inp = np.transpose(inp)
                m, n = inp.shape
                ans_tmp = np.zeros((m*m_grid, n), dtype='int')
                for i in range(m):
                    for j in range(m_grid):
                        ans_tmp[i*m_grid+j, :] = inp[i, :]
                ans = copy.deepcopy(ans_tmp)
                for stack_n in range(n_grid-1):
                    ans = np.hstack([ans, ans_tmp])
                for i in range(n):
                    for j in range(n_grid):
                        ans[:, i*n_grid+j] = ans_tmp[:, i]
                inout['input'] = ans
            return task_train
        grid_funcs.append(transpose_griding)
    return grid_funcs

def div_two_inputs(inp, long):
    inp = np.array(inp)
    m, n = inp.shape
    if m == n:
        return inp, inp
    horizontal = False
    # 縦長にする
    if n > m:
        horizontal = True
        inp = inp.T
        m, n = inp.shape

    a = inp[:long, :]
    b = inp[m-long:, :]
    # 元が横長だったら戻す
    if horizontal:
        a = a.T
        b = b.T
    return a, b

def add_div_funcs(train, m1, n1, m2, n2):
    for inout in train:
        m1_0, n1_0 = np.array(inout['input']).shape
        m2_0, n2_0 = np.array(inout['output']).shape
        if (m1_0 != m1) or (n1_0 != n1) or (m2_0 != m2) or (n2_0 != n2):
            return []
    if (m1 == n1) or (np.min([m1, n1]) != np.min([m2, n2])) or(np.max([m1,n1]) <= np.max([m2,n2])):
        return []
    long = np.max([m2,n2])
    def div_and(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs(inp, long)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            inout['input'] = a_align_num&b_align_num
        return task_train

    def div_or(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs(inp, long)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            inout['input'] = a_align_num|b_align_num
        return task_train

    def div_xor(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs(inp, long)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            inout['input'] = a_align_num^b_align_num
        return task_train
    def div_not_and(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs(inp, long)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            c = a_align_num&b_align_num
            inout['input'] = np.where(c == 0, not_use_num, 0)
        return task_train

    def div_not_or(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs(inp, long)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            c = a_align_num|b_align_num
            inout['input'] = np.where(c == 0, not_use_num, 0)
        return task_train

    def div_not_xor(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs(inp, long)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            c = a_align_num^b_align_num
            inout['input'] = np.where(c == 0, not_use_num, 0)
        return task_train

    return [div_and, div_or, div_xor, div_not_and, div_not_or, div_not_xor]


# In[170]:


def div_two_inputs2(inp, belt_length, vertically):
    inp = np.array(inp)
    if vertically:
        inp = inp.T
    m, n = inp.shape
    after_n = (n-belt_length) // 2

    a = inp[:, :after_n]
    b = inp[:, n-after_n:]
    if vertically:
        a, b = a.T, b.T
    return a, b

# 真ん中のベルトで分けるタイプ。ベルトの長さは各inputで一定の場合のみ
def add_div_funcs2(train, m1, n1, m2, n2, vertically):
    if vertically:
        if (n1 != n2) or (m1 < m2*2):
            return []
        belt_length = m1 - m2*2
    else:
        if (m1 != m2) or (n1 < n2*2):
            return []
        belt_length = n1 - n2*2

    for inout in train:
        m1_0, n1_0 = np.array(inout['input']).shape
        m2_0, n2_0 = np.array(inout['output']).shape
        if vertically:
            if (n1_0 != n2_0) or (m1_0 != m2_0*2 + belt_length):
                return []
        else:
            if (m1_0 != m2_0) or (n1_0 != n2_0*2 + belt_length):
                return []

    def div_and(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs2(inp, belt_length, vertically)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            inout['input'] = a_align_num&b_align_num
        return task_train

    def div_or(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs2(inp, belt_length, vertically)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            inout['input'] = a_align_num|b_align_num
        return task_train

    def div_xor(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs2(inp, belt_length, vertically)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            inout['input'] = a_align_num^b_align_num
        return task_train

    def div_not_and(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs2(inp, belt_length, vertically)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            c = a_align_num&b_align_num
            inout['input'] = np.where(c == 0, not_use_num, 0)
        return task_train

    def div_not_or(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs2(inp, belt_length, vertically)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            c = a_align_num|b_align_num
            inout['input'] = np.where(c == 0, not_use_num, 0)
        return task_train

    def div_not_xor(task_train):
        for inout in task_train:
            inp = inout['input']
            a, b = div_two_inputs2(inp, belt_length, vertically)
            not_use_num = get_not_use_num(a, b)
            a_align_num = np.where(a==0, 0, not_use_num)
            b_align_num = np.where(b==0, 0, not_use_num)
            c = a_align_num^b_align_num
            inout['input'] = np.where(c == 0, not_use_num, 0)
        return task_train

    return [div_and, div_or, div_xor, div_not_and, div_not_or, div_not_xor]


# # add patch funcs

# In[171]:


patch_skip = False
def add_patch_funcs(task_train, idx):
    if patch_skip:
        return [inouts_array]
    
    if correct_task_train(task_train):
        return [inouts_array]
    
    if idx in ['15696249', 'e345f17b']:
        return [inouts_array]
    inp, out = np.array(task_train[-1]["input"]), np.array(task_train[-1]["output"])
    for inout in task_train:
        if np.array(inout["input"]).shape != inp.shape:
            return [inouts_array]

    flip_funcs = [np.array, np.flip, np.flipud, np.fliplr]
    transpose_funcs = [np.array, np.transpose]
    best_score = match_rate(inp, out)
    best_feat = None
    for flip_func in flip_funcs:
        for transpose_func in transpose_funcs:
            inp_copy = copy.deepcopy(inp)
            inp_copy = flip_func(inp_copy)
            inp_copy = transpose_func(inp_copy)
            pred, feat = call_pred_train(inp_copy, out, patch_image)
            similarity = match_rate(out, pred)
            if best_score < similarity:
                best_score = similarity
                best_flip_func = flip_func
                best_transpose_func = transpose_func
                best_feat = feat

    def this_time_patch_image(task_train):
        for inout in task_train:
            inp = inout['input']
            if (best_feat is not None) & (best_feat != {}):
                inp = best_flip_func(inp)
                inp = best_transpose_func(inp)
    #             print(best_feat)
                pred = call_pred_test(inp, patch_image, best_feat)
    #             if np.array(pred).shape != task['test'][0]''
                if pred.shape != np.array(inp).shape:
                    inout['input'] = np.array(inp)
                    continue
                inout['input'] = pred
            else:
                inout['input'] = np.array(inp)
        return task_train
    
    return [this_time_patch_image, inouts_array]


# In[172]:


def in_out_diff(t_in, t_out):
    x_in, y_in = t_in.shape
    x_out, y_out = t_out.shape
    diff = np.zeros((max(x_in, x_out), max(y_in, y_out)))
    diff[:x_in, :y_in] -= t_in
    diff[:x_out, :y_out] += t_out
    return diff

def check_symmetric(a):
    try:
        sym = 1
        if np.array_equal(a, a.T):
            sym *= 2 #Check main diagonal symmetric (top left to bottom right)
        if np.array_equal(a, np.flip(a).T):
            sym *= 3 #Check antidiagonal symmetric (top right to bottom left)
        if np.array_equal(a, np.flipud(a)):
            sym *= 5 # Check horizontal symmetric of array
        if np.array_equal(a, np.fliplr(a)):
            sym *= 7 # Check vertical symmetric of array
        return sym
    except:
        return 0
    
def bbox(a):
    try:
        r = np.any(a, axis=1)
        c = np.any(a, axis=0)
        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    except:
        return 0,a.shape[0],0,a.shape[1]

def cmask(t_in):
    cmin = 999
    cm = 0
    for c in range(10):
        t = t_in.copy().astype('int8')
        t[t==c],t[t>0],t[t<0]=-1,0,1
        b = bbox(t)
        a = (b[1]-b[0])*(b[3]-b[2])
        s = (t[b[0]:b[1],b[2]:b[3]]).sum()
        if a>2 and a<cmin and s==a:
            cmin=a
            cm=c
    return cm

def mask_rect(a):
    r,c = a.shape
    m = a.copy().astype('uint8')
    for i in range(r-1):
        for j in range(c-1):
            if m[i,j]==m[i+1,j]==m[i,j+1]==m[i+1,j+1]>=1:m[i,j]=2
            if m[i,j]==m[i+1,j]==1 and m[i,j-1]==2:m[i,j]=2
            if m[i,j]==m[i,j+1]==1 and m[i-1,j]==2:m[i,j]=2
            if m[i,j]==1 and m[i-1,j]==m[i,j-1]==2:m[i,j]=2
    m[m==1]=0
    return (m==2)
    
def call_pred_train(t_in, t_out, pred_func):
    try:
        feat = {}
        feat['s_out'] = t_out.shape
        if t_out.shape==t_in.shape:
            diff = in_out_diff(t_in,t_out)
            feat['diff'] = diff
            feat['cm'] = t_in[diff!=0].max()
        else:
            feat['diff'] = (t_in.shape[0]-t_out.shape[0],t_in.shape[1]-t_out.shape[1])
            feat['cm'] = cmask(t_in)
        feat['sym'] = check_symmetric(t_out)
        args = inspect.getargspec(pred_func).args
        if len(args)==1:
            return pred_func(t_in), feat
        elif len(args)==2:
            t_pred = pred_func(t_in,feat[args[1]])    
        elif len(args)==3:
            t_pred = pred_func(t_in,feat[args[1]],feat[args[2]])
        feat['sizeok'] = len(t_out)==len(t_pred)
        t_pred = np.resize(t_pred,t_out.shape)
        acc = (t_pred==t_out).sum()/t_out.size
        return t_pred, feat
    except:
        return t_in, {}

def call_pred_test(t_in, pred_func, feat):
    args = inspect.getargspec(pred_func).args
    if len(args)==1:
        return pred_func(t_in)
    elif len(args)==2:
        t_pred = pred_func(t_in,feat[args[1]]) 
    elif len(args)==3:
        t_pred = pred_func(t_in,feat[args[1]],feat[args[2]])
    return t_pred

num2color = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]
color2num = {c: n for n, c in enumerate(num2color)}

def get_tile(img ,mask):
    try:
        m,n = img.shape
        a = img.copy().astype('int8')
        a[mask] = -1
        r=c=0
        for x in range(n):
            if np.count_nonzero(a[0:m,x]<0):continue
            for r in range(2,m):
                if 2*r<m and (a[0:r,x]==a[r:2*r,x]).all():break
            if r<m:break
            else: r=0
        for y in range(m):
            if np.count_nonzero(a[y,0:n]<0):continue
            for c in range(2,n):
                if 2*c<n and (a[y,0:c]==a[y,c:2*c]).all():break
            if c<n:break
            else: c=0
        if c>0:
            for x in range(n-c):
                if np.count_nonzero(a[:,x]<0)==0:
                    a[:,x+c]=a[:,x]
                elif np.count_nonzero(a[:,x+c]<0)==0:
                    a[:,x]=a[:,x+c]
        if r>0:
            for y in range(m-r):
                if np.count_nonzero(a[y,:]<0)==0:
                    a[y+r,:]=a[y,:]
                elif np.count_nonzero(a[y+r,:]<0)==0:
                    a[y,:]=a[y+r,:]
        return a[r:2*r,c:2*c]
    except:
        return a[0:1,0:1]
    
def patch_image(t_in,s_out,cm=0):
    t_in = np.array(t_in)
    try:
        t = t_in.copy()
        ty,tx=t.shape
        if cm>0:
            m = mask_rect(t==cm)
        else:
            m = (t==cm)   
        tile = get_tile(t ,m)
        if tile.size>2 and s_out==t.shape:
            rt = np.tile(tile,(1+ty//tile.shape[0],1+tx//tile.shape[1]))[0:ty,0:tx]
            if (rt[~m]==t[~m]).all():
                return rt
        for i in range(6):
            m = (t==cm)
            t -= cm
            if tx==ty:
                a = np.maximum(t,t.T)
                if (a[~m]==t[~m]).all():t=a.copy()
                a = np.maximum(t,np.flip(t).T)
                if (a[~m]==t[~m]).all():t=a.copy()
            a = np.maximum(t,np.flipud(t))
            if (a[~m]==t[~m]).all():t=a.copy()
            a = np.maximum(t,np.fliplr(t))
            if (a[~m]==t[~m]).all():t=a.copy()
            t += cm
            m = (t==cm)
            lms = measure.label(m.astype('uint8'))
            for l in range(1,lms.max()+1):
                lm = np.argwhere(lms==l)
                lm = np.argwhere(lms==l)
                x_min = max(0,lm[:,1].min()-1)
                x_max = min(lm[:,1].max()+2,t.shape[0])
                y_min = max(0,lm[:,0].min()-1)
                y_max = min(lm[:,0].max()+2,t.shape[1])
                gap = t[y_min:y_max,x_min:x_max]
                sy,sx=gap.shape
                if i==1:
                    sy//=2
                    y_max=y_min+sx
                gap = t[y_min:y_max,x_min:x_max]
                sy,sx=gap.shape
                allst = as_strided(t, shape=(ty,tx,sy,sx),strides=2*t.strides)    
                allst = allst.reshape(-1,sy,sx)
                allst = np.array([a for a in allst if np.count_nonzero(a==cm)==0])
                gm = (gap!=cm)
                for a in allst:
                    if sx==sy:
                        fpd = a.T
                        fad = np.flip(a).T
                        if i==1:gm[sy-1,0]=gm[0,sx-1]=False
                        if (fpd[gm]==gap[gm]).all():
                            gm = (gap!=cm)
                            np.putmask(gap,~gm,fpd)
                            t[y_min:y_max,x_min:x_max] = gap
                            break
                        if i==1:gm[0,0]=gm[sy-1,sx-1]=False
                        if (fad[gm]==gap[gm]).all():
                            gm = (gap!=cm)
                            np.putmask(gap,~gm,fad)
                            t[y_min:y_max,x_min:x_max] = gap
                            break 
                    fud = np.flipud(a)
                    flr = np.fliplr(a)
                    if i==1:gm[sy-1,0]=gm[0,sx-1]=gm[0,0]=gm[sy-1,sx-1]=False
                    if (a[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,a)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
                    elif (fud[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,fud)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
                    elif (flr[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,flr)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
        if s_out==t.shape:
            return t
        else:
            m = (t_in==cm)
            return np.resize(t[m],crop_min(m).shape)
    except:
        return t_in
    


# # add train004 growth

# In[173]:


def add_train4_growth(shaped_train):
    if correct_task_train(shaped_train):
        return [inouts_array]
    x = shaped_train[0]['input'].copy()
    out = shaped_train[0]['output'].copy()
    x = np.array(x)
    def get_base_pattern(arr, w, h):
        # find maximum number of unique color tiles in 3x3 field
        H, W = arr.shape
        arr_onehot = 1<<arr
        arr_bool = arr.astype(bool).astype(np.int32)
        counts = np.zeros(arr.shape, dtype=np.int32)
        colors = np.zeros(arr.shape, dtype=np.int32)
        for y in range(H-2):
            for x in range(W-2):
                counts[y, x] = arr_bool[y:y+2, x:x+2].sum()
                colors[y, x] = np.bitwise_or.reduce(arr_onehot[y:y+2, x:x+2].reshape(-1))
        n_colors = np.zeros(arr.shape, dtype=np.int32)
        for c in range(1, 10):
            n_colors += colors>>c & 1
        counts[n_colors>=2] = 0
        res_y, res_x = np.unravel_index(np.argmax(counts), counts.shape)
        pattern = arr[res_y:res_y+h, res_x:res_x+w].astype(bool).astype(np.int32)
        return (res_y, res_x), pattern
    repeat_num = 10
    best_wh = (2, 2)
    best_score = 0
    correct_flag = False
    for w in range(2,7):
        if correct_flag:
            break
        for h in range(2,7):
            (base_y, base_x), pattern = get_base_pattern(x, w, h)
#             p(pattern)
            try:
                pad_size = repeat_num * np.max([w, h])
                x_padded = np.pad(x, ((pad_size,pad_size),(pad_size,pad_size)), "constant", constant_values=0)
                base_y += pad_size
                base_x += pad_size
                y = x_padded.copy()
                for dy in [-(h+1), 0, h+1]:
                    for dx in [-(w+1), 0, w+1]:
                        y_, x_ = base_y+dy, base_x+dx
                        if dy==dx==0:
                            continue
                        count = np.bincount(x_padded[y_:y_+h+1, x_:x_+w+1].reshape(-1))
                        if count[0]==9:
                            continue
                        count[0] = 0
                        color = count.argmax()
                        for i in range(1, repeat_num):
                            y[base_y+dy*i:base_y+dy*i+h, base_x+dx*i:base_x+dx*i+w] = color * pattern
                y = y[pad_size:-pad_size, pad_size:-pad_size]
                score = match_rate(y, out)
                if best_score < score:
                    best_score = score
                    best_wh = (w, h)
                    if score == 1:
                        correct_flag = True
                        break
            except:
                pass
    def train4_growth(task_train):
        for inout in task_train:
            inp = inout['input']
            x = np.array(inp)
            try:
                w, h = best_wh
                (base_y, base_x), pattern = get_base_pattern(x, w, h)
                pad_size = repeat_num * np.max([w, h])
                x_padded = np.pad(x, ((pad_size,pad_size),(pad_size,pad_size)), "constant", constant_values=0)
                base_y += pad_size
                base_x += pad_size
                y = x_padded.copy()
                for dy in [-(h+1), 0, h+1]:
                    for dx in [-(w+1), 0, w+1]:
                        y_, x_ = base_y+dy, base_x+dx
                        if dy==dx==0:
                            continue
                        count = np.bincount(x_padded[y_:y_+h+1, x_:x_+w+1].reshape(-1))
                        if count[0]==9:
                            continue
                        count[0] = 0
                        color = count.argmax()
                        for i in range(1, repeat_num):
                            y[base_y+dy*i:base_y+dy*i+h, base_x+dx*i:base_x+dx*i+w] = color * pattern
                inout['input'] = y[pad_size:-pad_size, pad_size:-pad_size]
            except:
                inout['input'] = x
        return task_train
        
    return [inouts_array, train4_growth]


# # add change color funcs

# In[174]:


def add_change_color_funcs(task_train):
    if correct_task_train(task_train):
        return [inouts_array]
    in_use_colors, out_use_colors, color_changed = about_color(task_train)
    if not color_changed:
        return [inouts_array]
    inout_map = {}
    for in_color in in_use_colors:
        for out_color in out_use_colors:
            scores = []
            best_score = 0
            for inout in task_train:
                inp = inout['input'].copy()
                out = inout['output'].copy()
                in_vec = list(itertools.chain.from_iterable(inp))
                out_vec = list(itertools.chain.from_iterable(out))
                if (in_color not in in_vec) or (out_color not in out_vec):
                    continue
                inp = np.where(np.array(inp) == in_color, out_color, inp)
                scores.append(match_rate(inp, out))
            if np.mean(scores) > best_score:
                best_score = np.mean(scores)
                inout_map[in_color] = out_color
    def change_color(task_train):
        for inout in task_train:
            inp_origin = inout['input']
            inp = np.array(inp_origin.copy())
            vec = list(itertools.chain.from_iterable(inp_origin))
            for in_color, out_color in inout_map.items():
                if in_color in vec:
                    inp = np.where(np.array(inp_origin) == in_color, out_color, inp)
            inout['input'] = inp
        return task_train
    
    return [inouts_array, change_color]


# In[175]:


def about_color(task_train):
    in_colors = []
    out_colors = []
    color_changed = False
    for inout in task_train:
        in_vec = list(itertools.chain.from_iterable(inout['input']))
        in_colors += list(set(in_vec))
        out_vec = list(itertools.chain.from_iterable(inout['output']))
        out_colors += list(set(out_vec))
        if set(in_vec) != set(out_vec):
            color_changed = True
    return list(set(in_colors)), list(set(out_colors)), color_changed

def about_color_for_test(task_test):
    in_colors = []
    out_colors = []
    color_changed = False
    for inout in task_test:
        in_vec = list(itertools.chain.from_iterable(inout['input']))
        in_colors += list(set(in_vec))
    return list(set(in_colors))

def about_color_for_task(task):
    in_colors = []
    out_colors = []
    color_changed = False
    for inout in task['train']:
        in_vec = list(itertools.chain.from_iterable(inout['input']))
        in_colors += list(set(in_vec))
    for inout in task['test']:
        in_vec = list(itertools.chain.from_iterable(inout['input']))
        in_colors += list(set(in_vec))
    return list(set(in_colors))


# # add task train6

# In[176]:


def add_task_train6(task_train):
    if correct_task_train(task_train):
        return [inouts_array]
    use_same_color = True
    for inout in task_train:
        in_vec = list(itertools.chain.from_iterable(inout['input']))
        in_use_colors = list(set(in_vec))
        in_use_colors.remove(0) if 0 in in_use_colors else 0
        out_vec = list(itertools.chain.from_iterable(inout['output']))
        out_use_colors = list(set(out_vec))
        out_use_colors.remove(0) if 0 in out_use_colors else 0
        if sorted(in_use_colors) != sorted(out_use_colors):
            use_same_color = False
    if use_same_color:    
        return [inouts_array, task_train6]
    else:
        return [inouts_array]
    
def task_train6(task_train):
    for inout in task_train:
        x = inout['input']
        x = np.array(x)
        H, W = x.shape
        vec = list(itertools.chain.from_iterable(x))
        use_colors = list(set(vec))
        use_colors.remove(0) if 0 in use_colors else 0
        colors = [0] * len(use_colors)
        for yy in range(H):
            for xx in range(W):
                color = x[yy, xx]
                if color != 0:
                    colors[(yy+xx)%len(use_colors)] = color
        y = x.copy()
        for yy in range(H):
            for xx in range(W):
                y[yy, xx] = colors[(yy+xx)%len(use_colors)]
        inout['input'] = y
    return task_train        


# # add move object

# In[177]:


move_skip = False
def add_move_object(task_train):
    start = time()
    if move_skip:
        return [inouts_array]
    if correct_task_train(task_train):
        return [inouts_array]
    inp = np.array(task_train[0]['input'])
    out = np.array(task_train[0]['output'])
    in_use_colors, _, _ = about_color(task_train)
    in_use_colors = [c for c in in_use_colors if c != 0]
    best_score = match_rate(inp, out)
    best_goal_color = 0
    best_move_color = 0
    best_change_color = 0
    best_move_num = 0
    best_direction = [0, 0]
    best_correction = 0
    Dy = [0, 1, 0, -1]
    Dx = [1, 0, -1, 0]
    should_change = False
    for use_color_n, goal_color in enumerate(in_use_colors):
        for move_color in in_use_colors:
            if (time() - start > 60*60) & (len(in_use_colors) / 2 > use_color_n):
                return [inouts_array]

            goal_idx_set = set(tuple(idx) for idx in np.array(np.where(inp==goal_color)).T)
            move_idx_list = [tuple(idx) for idx in np.array(np.where(inp==move_color)).T]
            for dy, dx in zip(Dy, Dx):
                for move_num in range(1, 40):
                    obj_idx = set((idx[0]+dy*move_num, idx[1]+dx*move_num) for idx in move_idx_list)
                    if obj_idx & goal_idx_set:
                        for correction in [-2, -1, 0, 1, 2]:
                            for change_color in range(10):
                                inp_copy = copy.deepcopy(inp)
                                for idx in obj_idx:
                                    idx = (idx[0]+(dy*correction), idx[1]+(dx*correction))
                                    if (idx[0] < 0) or (idx[1] < 0) or (inp_copy.shape[0] <= idx[0]) or (inp_copy.shape[1] <= idx[1]):
                                        break
                                    inp_copy[idx] = change_color
                                for origin_move_pad_color in range(10):
                                    inp_copy2 = copy.deepcopy(inp_copy)
                                    for move_idx in move_idx_list:
                                        inp_copy2[move_idx] = origin_move_pad_color
                                    score = match_rate(inp_copy2, out)
                                    if best_score < score:
                                        should_change = True
                                        best_score = score
                                        best_goal_color = goal_color
                                        best_move_color = move_color
                                        best_move_num = move_num
                                        best_direction = [dy, dx]
                                        best_correction = correction
                                        best_change_color = change_color
                                        best_origin_move_pad_color = origin_move_pad_color

    def move_object(task_train_origin):
        for inout in task_train_origin:
            inout['input'] = np.array(inout['input'])
        if not should_change:
            return task_train_origin
        task_train = copy.deepcopy(task_train_origin)
        for i, inout in enumerate(task_train):
            finished = False
            inp = np.array(inout['input'])
            directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
            for direction in directions:
                if finished:
                    break
                for move_num in range(1, 50):
                    if finished:
                        break
                    goal_idx_set = set(tuple(idx) for idx in np.array(np.where(inp==best_goal_color)).T)
                    move_idx_list = [tuple(idx) for idx in np.array(np.where(inp==best_move_color)).T]
                    obj_idx = set((idx[0] + direction[0]*move_num, idx[1] + direction[1]*move_num) for idx in move_idx_list)
                    if obj_idx & goal_idx_set:
                        for idx in obj_idx:
                            idx = (idx[0]+(direction[0]*best_correction), idx[1]+(direction[1]*best_correction))
                            if (idx[0] < 0) or (idx[1] < 0) or (inp.shape[0] <= idx[0]) or (inp.shape[1] <= idx[1]):
                                continue
                            inp[idx] = best_change_color
                        for move_idx in move_idx_list:
                            inp[move_idx] = best_origin_move_pad_color
                        task_train[i]['input'] = inp
                        finished = True
        # if recursion:
        #     for i in range(5):
        #         if 'output' in task_train[0]:
        #             if correct_task_train(task_train):
        #                 return task_train
        #         funcs = add_move_object(task_train, False)
        #         for func in funcs:
        #             task_train = func(task_train)
        return task_train
    return [move_object, inouts_array]


# # correct task

# In[178]:


def correct_task_train(task_train):
    correct = True
    for inout in task_train:
        if np.array(inout['input']).tolist() != np.array(inout['output']).tolist():
            correct = False
    return correct


# # check_p

# In[179]:


def check_p(task, pred_func):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(3, n, figsize=(4*n,12), dpi=50)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fnum = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
        t_pred, feat = call_pred_train(t_in, t_out, pred_func)
        plot_one(axs[0,fnum],t_in,f'train-{i} input')
        plot_one(axs[1,fnum],t_out,f'train-{i} output')
        plot_one(axs[2,fnum],t_pred,f'train-{i} pred')
        fnum += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
        t_pred = call_pred_test(t_in, pred_func, feat)        
        plot_one(axs[0,fnum],t_in,f'test-{i} input')
        plot_one(axs[1,fnum],t_out,f'test-{i} output')
        plot_one(axs[2,fnum],t_pred,f'test-{i} pred')
#         t_pred = np.resize(t_pred,t_out.shape)
        fnum += 1
    plt.show()
    return 1


# # recolor

# In[180]:


def Defensive_Copy(A): 
    n = len(A)
    k = len(A[0])
    L = np.zeros((n,k), dtype = int)
    for i in range(n):
        for j in range(k):
            L[i,j] = 0 + A[i][j]
    return L.tolist()

def Create(task, task_id = 0):
    n = len(task['train'])
    Input = [Defensive_Copy(task['train'][i]['input']) for i in range(n)]
    Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]
    Input.append(Defensive_Copy(task['test'][task_id]['input']))
    return Input, Output

def add_recolor(task_train, task_n):
    return [inouts_array]
    m1, n1 = np.array(task_train[0]['input']).shape
    m2, n2 = np.array(task_train[0]['output']).shape
    all_inputs_same_shape = True
    all_outputs_same_shape = True
    for inout in task_train:
        m1_, n1_ = np.array(inout['input']).shape
        m2_, n2_ = np.array(inout['output']).shape
        if (m1_ != m1) or (n1_ != n1):
            all_inputs_same_shape = False
        if (m2_ != m2) or (n2_ != n2):
            all_outputs_same_shape = False
    if (not all_inputs_same_shape) or (not all_outputs_same_shape):
        return [inouts_array]

    inputs = []
    outputs = []
    for inout in task_train:
        inputs.append(copy.deepcopy(inout['input']))
        outputs.append(copy.deepcopy(inout['output']))

    N = len(inputs)
    x0 = inputs[0]
    y0 = outputs[0]
    n = len(x0)
    k = len(x0[0])
    a = len(y0)
    b = len(y0[0])

    List1 = {}
    List2 = {}
    for i in range(n):
        for j in range(k):
            seq = []
            for x in inputs:
                seq.append(x[i][j])
            List1[(i,j)] = seq

    for p in range(a):
        for q in range(b):
            seq1 = []
            for y in outputs:
                seq1.append(y[p][q])

            places = []
            for key in List1:
                if List1[key] == seq1:
                    places.append(key)

            List2[(p,q)] = places
            if len(places) == 0:
                return [inouts_array]

    def recolor(task_train):
        for inout in task_train:
            inout['input'] = np.array(inout['input'])
        answer = np.zeros((a,b), dtype = int)
        for inout_n, inout in enumerate(task_train):
            for p in range(a):
                for q in range(b):
                    palette = [0,0,0,0,0,0,0,0,0,0]
                    for i, j in List2[(p,q)]:
                        color = inout['input'][i][j]
                        palette[color]+=1
                    answer[p,q] =  np.argmax(palette)

            task_train[inout_n]['input'] = np.array(answer)
        return task_train

    return [inouts_array, recolor]


# # get_similarity

# In[181]:


def get_similarity(train, func_combi, idx, search_func=True):
    similarities = []
    for seed in range(3):
        similarity = train_and_evaluate(train, func_combi, seed, idx, {}, search_func=search_func)
        similarities.append(similarity)
    return np.mean(similarities)


# # add kneighbors

# In[182]:


def comparematrixes(a,b):
    out=0;
    for i in range(min(len(a),len(b))):
        for j in range(min(len(a[0]),len(b[0]))):
            if a[i][j]==b[i][j]:
                out+=1
    out/=len(a)*len(a[0]);
    return 1-out;
def add_kneighbors(task_train):
    all_inputs_same_shape, all_outputs_same_shape = all_inputs_same_shape_and_all_outputs_same_shape(task_train)
    if (not all_inputs_same_shape) or (not all_outputs_same_shape):
#         return [inouts_array]
        pass
    ines=[];
    outes=[];
    for i in range(len(task_train)):
        vx=task_train[i]["input"].copy();
        vi=task_train[i]["output"].copy();
        if (len(vx) > 10) or (len(vi) > 10):
            return [inouts_array]
        for k1 in range(min(len(vx),len(vi))):
            for k2 in range(min(len(vx[0]),len(vi[0]))):
                dtm=[];
                for k3 in range(-2,2+1,1):
                    for k4 in range(-2,2+1,1):
                        if(k1+k3<len(vx) and k1+k3>=0 and k2+k4<len(vx[0]) and k2+k4>=0 and k1+k3<len(vi) and k1+k3>=0 and k2+k4<len(vi[0]) and k2+k4>=0):
                            td=[0,0,0,0,0,0,0,0,0,0,0];
                            if (vx[k1+k3][k2+k4] > 10) or (vi[k1+k3][k2+k4]):
                                return [inouts_array]
                            td[vx[k1+k3][k2+k4]]=1
                            dtm+=td.copy();
                            td=[0,0,0,0,0,0,0,0,0,0,0];
                            td[vi[k1+k3][k2+k4]]=1;
                            dtm+=td.copy();
                        else:
                            dtm+=[0,0,0,0,0,0,0,0,0,0,0];
                            dtm+=[0,0,0,0,0,0,0,0,0,0,0];
                ines.append(dtm);
                if(len(vi)>k1 and len(vi[0])>k2 and k1>=0 and k2>=0):
                    outes.append(vi[k1][k2]);
                else:
                    outes.append(0);
    knn = KNeighborsClassifier(n_neighbors = 1);
    ines=json.loads(json.dumps(ines));
    knn.fit(ines,outes);
    outs=[]
    def kneighbors(task_train_origin):
        for inout in task_train_origin:
            inout['input'] = np.array(inout['input'])
        task_train = copy.deepcopy(task_train_origin)
        for i in range(len(task_train)):
            thisdone=False;
            vx=task_train[i]["input"].copy();
            vi=task_train[i]["input"].copy();
            for U in range(20):
                for k1 in range(len(vx)):
                    for k2 in range(len(vx[0])):
                        dtm=[];
                        for k3 in range(-2,2+1,1):
                            for k4 in range(-2,2+1,1):
                                if(k1+k3<len(vx) and k1+k3>=0 and k2+k4<len(vx[0]) and k2+k4>=0 and k1+k3<len(vi) and k1+k3>=0 and k2+k4<len(vi[0]) and k2+k4>=0):
                                    td = [0,0,0,0,0,0,0,0,0,0,0];
                                    td[vx[k1+k3][k2+k4]]=1
                                    dtm+=td.copy();
                                    td = [0,0,0,0,0,0,0,0,0,0,0];
                                    td[vi[k1+k3][k2+k4]]=1;
                                    dtm+=td.copy();
                                else:
                                    dtm+=[0,0,0,0,0,0,0,0,0,0,0];
                                    dtm+=[0,0,0,0,0,0,0,0,0,0,0];
                        vi[k1][k2]=int(knn.predict([dtm])[0]);
                vx=vi.copy();
            task_train[i]['input'] = vx
        return task_train
    return [inouts_array, kneighbors]


# # ARC solver

# In[183]:


class ARC_solver:
    def __init__(self):
        self.identified_objects = []
        self.io_inx = [] # the original index of the identified objects (io)
        self.io_height = [] # height of io
        self.io_width = [] # width of io
        self.io_pixel_count = [] # count of non-background pixels
        self.io_size = [] # overall grid size
        self.io_unique_colors = [] # number of unique colors
        self.io_main_color = [] # the dominating color

    def reset(self):
        self.identified_objects = []
        self.io_inx = []
        self.io_height = []
        self.io_width = []
        self.io_pixel_count = []
        self.io_size = []
        self.io_unique_colors = []
        self.io_main_color = []

    def get_background(self, image):
        # if image contains 0
        if 0 in image:
            background = 0
        # else use the most frequent pixel color
        else:
            unique_colors, counts = np.unique(image, return_counts = True)
            background = unique_colors[np.argmax(counts)]
        return background

    def check_pairs(self, inx_pairs, this_pair, return_inx = False):
        # check if this_pair is in inx_pairs
        match = []
        for pair in inx_pairs:
            if pair[0] == this_pair[0] and pair[1] == this_pair[1]:
                match.append(True)
            else:
                match.append(False)
        if return_inx:
            return any(match), np.where(match)
        else:
            return any(match)

    def check_neighbors(self, all_pairs, this_pair, objectness, this_object):
        # all_pairs: an array of index pairs for all nonzero/colored pixels
        # this_pair: the index pair whose neighbors will be checked
        # objectness: an array with the shape of original image, storage for how much objectness has been identified
        # this_object: the current object we are looking at
        row_inx = this_pair[0]
        col_inx = this_pair[1]
        objectness[row_inx, col_inx] = this_object
        # find if any neighboring pixels contain color
        if self.check_pairs(all_pairs, [row_inx-1, col_inx-1]): # up-left
            objectness[row_inx-1, col_inx-1] = this_object
        if self.check_pairs(all_pairs, [row_inx-1, col_inx]): # up
            objectness[row_inx-1, col_inx] = this_object
        if self.check_pairs(all_pairs, [row_inx-1, col_inx+1]): # up-right
            objectness[row_inx-1, col_inx+1] = this_object
        if self.check_pairs(all_pairs, [row_inx, col_inx-1]): # left
            objectness[row_inx, col_inx-1] = this_object
        if self.check_pairs(all_pairs, [row_inx, col_inx+1]): # right
            objectness[row_inx, col_inx+1] = this_object
        if self.check_pairs(all_pairs, [row_inx+1, col_inx-1]): # down-left
            objectness[row_inx+1, col_inx-1] = this_object
        if self.check_pairs(all_pairs, [row_inx+1, col_inx]): # down
            objectness[row_inx+1, col_inx] = this_object
        if self.check_pairs(all_pairs, [row_inx+1, col_inx+1]): # down-right
            objectness[row_inx+1, col_inx+1] = this_object
        return objectness

    def identify_object_by_color(self, true_image, background = 0):
        # identify obeject by the color only
        unique_colors = np.unique(true_image)
        for i, color in enumerate(unique_colors):
            image = np.copy(true_image) # make a copy from original first
            if color == background:
                continue
            image[image != color] = background
            inx = np.where(image == color)
            obj = image[np.min(inx[0]):np.max(inx[0])+1, np.min(inx[1]):np.max(inx[1])+1]
            # append the object attributes
            self.identified_objects.append(obj)
            self.io_inx.append(inx)
            self.io_height.append(obj.shape[0])
            self.io_width.append(obj.shape[1])
            self.io_pixel_count.append(obj[obj != background].shape[0])
            self.io_size.append(obj.size)
            nc, c = np.unique(obj, return_counts = True)
            self.io_unique_colors.append(nc)
            self.io_main_color.append(nc[np.argmax(c)])

    def identify_object_by_isolation(self, image, background = 0):
        # identify all objects by physical isolation on the given image
        all_pairs = np.array(np.where(image != background)).T
        objectness = np.zeros(image.shape)
        this_object = 1
        while len(all_pairs) >= 1:
            init_pair = all_pairs[0] # start with the first pair
            objectness = self.check_neighbors(all_pairs, init_pair, objectness, this_object)
            # get a list of index pairs whose neghbors haven't been checked
            unchecked_pairs = np.array(np.where(objectness == this_object)).T
            checked_pairs = np.zeros((0,2))
            # check all the index pairs in the expanding unchecked_pairs untill all have been checked
            while len(unchecked_pairs) != 0:
                this_pair = unchecked_pairs[0]
                objectness = self.check_neighbors(all_pairs, this_pair, objectness, this_object)
                # append the checked_pairs
                checked_pairs = np.vstack((checked_pairs, this_pair))
                # get all index pairs for the currently identified object
                current_object_pairs = np.array(np.where(objectness == this_object)).T
                # delete the checked pairs from current object pairs
                checked_inx = []
                for pair in checked_pairs:
                    _, inx = self.check_pairs(current_object_pairs, pair, return_inx = True)
                    checked_inx.append(inx[0][0])
                unchecked_pairs = np.delete(current_object_pairs, checked_inx, axis = 0)

            # store this object to identified_objects
            current_object_pairs = np.array(np.where(objectness == this_object)).T
            cop = current_object_pairs.T
            obj = image[np.min(cop[0]):np.max(cop[0])+1, np.min(cop[1]):np.max(cop[1])+1]
            # delete the current object pairs from all_pairs
            cop_inx = []
            for pair in current_object_pairs:
                _, this_cop_inx = self.check_pairs(all_pairs, pair, return_inx = True)
                cop_inx.append(this_cop_inx[0][0])
            all_pairs = np.delete(all_pairs, cop_inx, axis = 0)
            # append the object attribute
            # p(obj)
            if np.array(obj).shape[0] * np.array(obj).shape[0] >= 3:
                self.identified_objects.append(obj)
            self.io_inx.append(inx)
            self.io_height.append(obj.shape[0])
            self.io_width.append(obj.shape[1])
            self.io_pixel_count.append(obj[obj != background].shape[0])
            self.io_size.append(obj.size)
            nc, c = np.unique(obj, return_counts = True)
            self.io_unique_colors.append(nc)
            self.io_main_color.append(nc[np.argmax(c)])
            # start identifying a new object
            this_object += 1
        return objectness

    def identify_object_by_color_isolation(self, true_image, background = 0):
        # identify objects first by color then by physical isolation
        unique_colors = np.unique(true_image)
        for i, color in enumerate(unique_colors):
            image = np.copy(true_image) # make a copy from the original first
            if color == background:
                continue
            # identify objects by isolation in this color only
            image[image != color] = background
            self.identify_object_by_isolation(image, background = background)
            

    def sort(self, objs, inp):
        xs = []
        ys = []
        for i, o in enumerate(objs):
            _, m, n = sliding_window_search(inp, o)
            xs.append(m)
            ys.append(n)

        ans = [[[]],[[]],[[]],[[]]]
        left = np.array(ys).argsort()[0:2] # 1,3
        right = np.array(ys).argsort()[2:4] # 1,3
        if xs[left[0]] <= xs[left[1]]:
            ans[0] = objs[left[0]]
            ans[2] = objs[left[1]]
        else:
            ans[2] = objs[left[0]]
            ans[0] = objs[left[1]]    
        if xs[right[0]] <= xs[right[1]]:
            ans[1] = objs[right[0]]
            ans[3] = objs[right[1]]
        else:
            ans[3] = objs[right[0]]
            ans[1] = objs[right[1]]
        return ans        
            
    def merge(self, objects, belt, use_color):
#         ans = objects
        ans=[[[]],[[]],[[]],[[]]]
        for o in objects:
            o = np.array(o)
            max_total = 0
            for x in [0,1]:
                for y in [0,1]:
                    if max_total < o[x:x+len(o)-1, y:y+len(o[0])-1].sum():
                        max_total = o[x:x+len(o)-1, y:y+len(o[0])-1].sum()
                        max_xy = (x, y)
            if max_xy == (0,0):
                ans[3] = o
            elif max_xy == (0,1):
                ans[2] = o
            elif max_xy == (1,0):
                ans[1] = o
            else:
                ans[0] = o

        if belt == 0:
            belt_list = [[use_color]]*len(ans[0])
            u=np.hstack([ans[0], ans[1]])
            u
            s=np.hstack([ans[2], ans[3]])
            return np.vstack([u,s])
        else:
            belt_list = [[use_color]*belt]*len(ans[0])

            u=np.hstack([ans[0], belt_list, ans[1]])
            s=np.hstack([ans[2], belt_list, ans[3]])
            belt_list = [[use_color]*len(s[0])]*belt
            return np.vstack([u,belt_list,s])



# # add block merge

# In[184]:


def divide_block_and_merge(task_train_origin, objn):
    for inout in task_train_origin:
        inout['input'] = np.array(inout['input'])
    task_train = copy.deepcopy(task_train_origin)
    for i, inout in enumerate(task_train):    
        arc = ARC_solver()
        inp = inout['input']
        inp = np.array(inp)
        use_color = list(set(list(itertools.chain.from_iterable(inp))))
        if len(use_color) != 2:
            return task_train_origin
        
        try:
            inp_o = copy.deepcopy(inp)
            inp = np.where(inp_o==use_color[0], use_color[1], inp)
            inp = np.where(inp_o==use_color[1], use_color[0], inp)
            background = arc.get_background(inp)
            arc.identify_object_by_isolation(inp, background)
            if len(arc.identified_objects) == 4:
                arc.identified_objects = arc.sort(arc.identified_objects, inp)
                out = np.array(arc.identified_objects[objn])
                out_o = copy.deepcopy(out)
                out = np.where(out_o==use_color[0], use_color[1], out)
                out = np.where(out_o==use_color[1], use_color[0], out)
                task_train[i]['input'] = out
        except:
            return task_train_origin
    return task_train

def divide_block_and_merge1(task_train_origin):
    return divide_block_and_merge(task_train_origin, 1)

def divide_block_and_merge2(task_train_origin):
    return divide_block_and_merge(task_train_origin, 2)

def divide_block_and_merge3(task_train_origin):
    return divide_block_and_merge(task_train_origin, 3)

def add_block_merge(task_train):
    arc = ARC_solver()
    if len(task_train) > 2:
        task_n = 2
    else:
        task_n = 0
    inp = task_train[task_n]['input']
    inp = np.array(inp)
    use_color = list(set(list(itertools.chain.from_iterable(inp))))
    if len(use_color) != 2:
        return []
    inp_o = copy.deepcopy(inp)
    inp = np.where(inp_o==use_color[0], use_color[1], inp)
    inp = np.where(inp_o==use_color[1], use_color[0], inp)
    background = arc.get_background(inp)
    arc.identify_object_by_isolation(inp, background)
    if len(arc.identified_objects) == 4:
        try:
            arc.identified_objects = arc.sort(arc.identified_objects, inp)
#             for i in arc.identified_objects:
#                 p(i)
            for i in range(4):
                out = np.array(arc.identified_objects[i])
                out_o = copy.deepcopy(out)
                out = np.where(out_o==use_color[0], use_color[1], out)
                out = np.where(out_o==use_color[1], use_color[0], out)    

                if out.tolist() == task_train[task_n]['output']:
                    return [divide_block_and_merge1, divide_block_and_merge2, divide_block_and_merge3]
        except:
            return []
    return []


# # add object detect2

# In[185]:


def select_by_ele(objects, ele):
    if ele == 'height':
        max_height = 0
        for obj in objects:
            if len(obj) > max_height:
                selected = obj
                max_height = len(obj)
    if ele == 'width':
        max_width = 0
        for obj in objects:
            if len(obj[0]) > max_width:
                selected = obj
                max_width = len(obj[0])
    if ele == 'area':
        max_area = 0
        for obj in objects:
            if len(obj) * len(obj[0]) > max_area:
                selected = obj
                max_area = len(obj) * len(obj[0])
        
    return selected

def add_object_detect2(task_train):
    for select_ele in ['height', 'width', 'area']:
        sucess = True
        for inout in task_train:
            arc = ARC_solver()
            inp = copy.deepcopy(inout['input'])
            inp = np.array(inp)
            background = arc.get_background(inp)
            arc.identify_object_by_isolation(inp, background)
            obj = select_by_ele(arc.identified_objects, select_ele)
            if (obj.shape != np.array(inout['output']).shape) & (obj.shape != np.array(inout['output']).T.shape):
                sucess = False
        if sucess:
            def object_detect2(task_train_origin):
                for inout in task_train_origin:
                    inout['input'] = np.array(inout['input'])
                task_train = copy.deepcopy(task_train_origin)
                for i, inout in enumerate(task_train):   
                    try:
                        arc = ARC_solver()
                        inp = copy.deepcopy(inout['input'])
                        inp = np.array(inp)
                        background = arc.get_background(inp)
                        arc.identify_object_by_isolation(inp, background)
                        obj = select_by_ele(arc.identified_objects, select_ele)
                        task_train[i]['input'] = obj
                    except:
                        return task_train_origin
                return task_train
                    
            return [object_detect2]
    return []


# # add crop_by_line

# In[186]:


def add_crop_by_line(task_train):
    success = True
    for i, inout in enumerate(task_train):
        inp = np.array(copy.deepcopy(inout['input']))
        use_color = matrix_use_color(inp)
        max_area = 0
        max_enclosure_color = 0
        include_line = False
        uses = [0,0,0,0]
        found = False
        use_max_x = 0
        use_max_y = 0
        use_min_x = 0
        use_min_y = 0
        for color in use_color:
            idx = [idx.tolist() for idx in np.array(np.where(inp==color)).T]

            max_x = 0
            max_y = 0
            min_x = 100
            min_y = 100
            for i in idx:
                if i[0] < min_x:
                    min_x = i[0]
                if i[1] < min_y:
                    min_y = i[1]
                if i[0] > max_x:
                    max_x = i[0]
                if i[1] > max_y:
                    max_y = i[1]

            enclosure_flag = True
            for x in range(min_x, max_x+1):
                if (inp[x][min_y] != color) or (inp[x][max_y] != color):
                    enclosure_flag = False
            for y in range(min_y, max_y+1):
                if (inp[min_x][y] != color) or (inp[max_x][y] != color):
                    enclosure_flag = False
            for x in range(min_x+1, max_x):
                for y in range(min_y+1, max_y):
                    if inp[x][y] == color:
                        enclosure_flag = False
            if enclosure_flag & (max_x > 0) & (max_x - min_x > 1):
                area = (max_x-min_x)*(max_y-min_y)
                if max_area < area:
                    max_area = area
                    max_enclosure_color = color
                    found = True
                    use_max_x = max_x
                    use_max_y = max_y
                    use_min_x = min_x
                    use_min_y = min_y
        if not found:
            return []
        if i == 0:
            if np.array(inout['output']).shape == (use_max_x-use_min_x-1, use_max_y-use_min_y-1):
                include_line = False
            elif np.array(inout['output']).shape == (use_max_x-use_min_x+1, use_max_y-use_min_y+1):
                include_line = True
            else:
                success = False
        else:
            if (not include_line) & (np.array(inout['output']).shape == (use_max_x-use_min_x-1, use_max_y-use_min_y-1)) or ((include_line) & (np.array(inout['output']).shape == (use_max_x-use_min_x+1, use_max_y-use_min_y+1))):
                pass
            else:
                success = False

        if not success:
            break

    if success:
        def crop_by_max_enclosure_color(task_train_origin):
            for inout in task_train_origin:
                inout['input'] = np.array(inout['input'])
            task_train = copy.deepcopy(task_train_origin)
            for task_n, inout in enumerate(task_train):
                inp = np.array(copy.deepcopy(inout['input']))
                use_color = matrix_use_color(inp)
                max_area = 0
                max_enclosure_color = 0
                include_line = False
                uses = [0,0,0,0]
                found = False
                use_max_x = 0
                use_max_y = 0
                use_min_x = 0
                use_min_y = 0
                for color in use_color:
                    idx = [idx.tolist() for idx in np.array(np.where(inp==color)).T]

                    max_x = 0
                    max_y = 0
                    min_x = 100
                    min_y = 100
                    for i in idx:
                        if i[0] < min_x:
                            min_x = i[0]
                        if i[1] < min_y:
                            min_y = i[1]
                        if i[0] > max_x:
                            max_x = i[0]
                        if i[1] > max_y:
                            max_y = i[1]
                    enclosure_flag = True
                    for x in range(min_x, max_x+1):
                        if (inp[x][min_y] != color) or (inp[x][max_y] != color):
                            enclosure_flag = False
                    for y in range(min_y, max_y+1):
                        if (inp[min_x][y] != color) or (inp[max_x][y] != color):
                            enclosure_flag = False
                    for x in range(min_x+1, max_x):
                        for y in range(min_y+1, max_y):
                            if inp[x][y] == color:
                                enclosure_flag = False
                    if enclosure_flag & (max_x > 0) & (max_x - min_x > 1):
                        area = (max_x-min_x)*(max_y-min_y)
                        if max_area < area:
                            max_area = area
                            max_enclosure_color = color
                            found = True
                            use_max_x = max_x
                            use_max_y = max_y
                            use_min_x = min_x
                            use_min_y = min_y                
                if include_line:
                    out = inp[use_min_x:use_max_x+1, use_min_y:use_max_y+1]
                else:
                    out = inp[use_min_x+1:use_max_x, use_min_y+1:use_max_y]
                task_train[task_n]['input'] = out
            return task_train
                
        return [crop_by_max_enclosure_color]
    return []


# # add back_to_black

# In[187]:


def back_to_black(task_train_origin):
    for inout in task_train_origin:
        inout['input'] = np.array(inout['input'])
    task_train = copy.deepcopy(task_train_origin)
    for task_n, inout in enumerate(task_train):
        inp = inout['input']
        inp_o = copy.deepcopy(inp)
        i = list(itertools.chain.from_iterable(inp))
        most_use_color = collections.Counter(i).most_common()[0][0]
        inp = np.where(inp_o==most_use_color, 0, inp)
        inp = np.where(inp_o==0, most_use_color, inp)
        task_train[task_n]['input'] = inp
    return task_train

def add_back_to_black_funcs(task_train):
    change_back = True
    for inout in task_train:
        i = list(itertools.chain.from_iterable(inout['input']))
        if collections.Counter(i).most_common()[0][0] == 0:
            change_back = False
    
    if change_back:
        return [back_to_black, inouts_array]
    else:
        return [inouts_array]


# In[189]:


class ARC_solver:
    def __init__(self):
        self.identified_objects = []
        self.io_inx = [] # the original index of the identified objects (io)
        self.io_height = [] # height of io
        self.io_width = [] # width of io
        self.io_pixel_count = [] # count of non-background pixels
        self.io_size = [] # overall grid size
        self.io_unique_colors = [] # number of unique colors
        self.io_main_color = [] # the dominating color

    def reset(self):
        self.identified_objects = []
        self.io_inx = []
        self.io_height = []
        self.io_width = []
        self.io_pixel_count = []
        self.io_size = []
        self.io_unique_colors = []
        self.io_main_color = []

    def get_background(self, image):
        # if image contains 0
        if 0 in image:
            background = 0
        # else use the most frequent pixel color
        else:
            unique_colors, counts = np.unique(image, return_counts = True)
            background = unique_colors[np.argmax(counts)]
        return background

    def check_pairs(self, inx_pairs, this_pair, return_inx = False):
        # check if this_pair is in inx_pairs
        match = []
        for pair in inx_pairs:
            if pair[0] == this_pair[0] and pair[1] == this_pair[1]:
                match.append(True)
            else:
                match.append(False)
        if return_inx:
            return any(match), np.where(match)
        else:
            return any(match)

    def check_neighbors(self, all_pairs, this_pair, objectness, this_object):
        # all_pairs: an array of index pairs for all nonzero/colored pixels
        # this_pair: the index pair whose neighbors will be checked
        # objectness: an array with the shape of original image, storage for how much objectness has been identified
        # this_object: the current object we are looking at
        row_inx = this_pair[0]
        col_inx = this_pair[1]
        objectness[row_inx, col_inx] = this_object
        # find if any neighboring pixels contain color
        if self.check_pairs(all_pairs, [row_inx-1, col_inx-1]): # up-left
            objectness[row_inx-1, col_inx-1] = this_object
        if self.check_pairs(all_pairs, [row_inx-1, col_inx]): # up
            objectness[row_inx-1, col_inx] = this_object
        if self.check_pairs(all_pairs, [row_inx-1, col_inx+1]): # up-right
            objectness[row_inx-1, col_inx+1] = this_object
        if self.check_pairs(all_pairs, [row_inx, col_inx-1]): # left
            objectness[row_inx, col_inx-1] = this_object
        if self.check_pairs(all_pairs, [row_inx, col_inx+1]): # right
            objectness[row_inx, col_inx+1] = this_object
        if self.check_pairs(all_pairs, [row_inx+1, col_inx-1]): # down-left
            objectness[row_inx+1, col_inx-1] = this_object
        if self.check_pairs(all_pairs, [row_inx+1, col_inx]): # down
            objectness[row_inx+1, col_inx] = this_object
        if self.check_pairs(all_pairs, [row_inx+1, col_inx+1]): # down-right
            objectness[row_inx+1, col_inx+1] = this_object
        return objectness

    def identify_object_by_color(self, true_image, background = 0):
        # identify obeject by the color only
        unique_colors = np.unique(true_image)
        for i, color in enumerate(unique_colors):
            image = np.copy(true_image) # make a copy from original first
            if color == background:
                continue
            image[image != color] = background
            inx = np.where(image == color)
            obj = image[np.min(inx[0]):np.max(inx[0])+1, np.min(inx[1]):np.max(inx[1])+1]
            # append the object attributes
            self.identified_objects.append(obj)
            self.io_inx.append(inx)
            self.io_height.append(obj.shape[0])
            self.io_width.append(obj.shape[1])
            self.io_pixel_count.append(obj[obj != background].shape[0])
            self.io_size.append(obj.size)
            nc, c = np.unique(obj, return_counts = True)
            self.io_unique_colors.append(nc)
            self.io_main_color.append(nc[np.argmax(c)])

    def identify_object_by_isolation(self, image, background = 0):
        # identify all objects by physical isolation on the given image
        all_pairs = np.array(np.where(image != background)).T
        objectness = np.zeros(image.shape)
        this_object = 1
        while len(all_pairs) >= 1:
            init_pair = all_pairs[0] # start with the first pair
            objectness = self.check_neighbors(all_pairs, init_pair, objectness, this_object)
            # get a list of index pairs whose neghbors haven't been checked
            unchecked_pairs = np.array(np.where(objectness == this_object)).T
            checked_pairs = np.zeros((0,2))
            # check all the index pairs in the expanding unchecked_pairs untill all have been checked
            while len(unchecked_pairs) != 0:
                this_pair = unchecked_pairs[0]
                objectness = self.check_neighbors(all_pairs, this_pair, objectness, this_object)
                # append the checked_pairs
                checked_pairs = np.vstack((checked_pairs, this_pair))
                # get all index pairs for the currently identified object
                current_object_pairs = np.array(np.where(objectness == this_object)).T
                # delete the checked pairs from current object pairs
                checked_inx = []
                for pair in checked_pairs:
                    _, inx = self.check_pairs(current_object_pairs, pair, return_inx = True)
                    checked_inx.append(inx[0][0])
                unchecked_pairs = np.delete(current_object_pairs, checked_inx, axis = 0)

            # store this object to identified_objects
            current_object_pairs = np.array(np.where(objectness == this_object)).T
            cop = current_object_pairs.T
            obj = image[np.min(cop[0]):np.max(cop[0])+1, np.min(cop[1]):np.max(cop[1])+1]
            # delete the current object pairs from all_pairs
            cop_inx = []
            for pair in current_object_pairs:
                _, this_cop_inx = self.check_pairs(all_pairs, pair, return_inx = True)
                cop_inx.append(this_cop_inx[0][0])
            all_pairs = np.delete(all_pairs, cop_inx, axis = 0)
            # append the object attribute
            # p(obj)
            self.identified_objects.append(obj)
            self.io_inx.append(inx)
            self.io_height.append(obj.shape[0])
            self.io_width.append(obj.shape[1])
            self.io_pixel_count.append(obj[obj != background].shape[0])
            self.io_size.append(obj.size)
            nc, c = np.unique(obj, return_counts = True)
            self.io_unique_colors.append(nc)
            self.io_main_color.append(nc[np.argmax(c)])
            # start identifying a new object
            this_object += 1
        return objectness

    def identify_object_by_color_isolation(self, true_image, background = 0):
        # identify objects first by color then by physical isolation
        unique_colors = np.unique(true_image)
        for i, color in enumerate(unique_colors):
            image = np.copy(true_image) # make a copy from the original first
            if color == background:
                continue
            # identify objects by isolation in this color only
            image[image != color] = background
            self.identify_object_by_isolation(image, background = background)
            

    def sort(self, objs, inp):
        xs = []
        ys = []
        for i, o in enumerate(objs):
            _, m, n = sliding_window_search(inp, o)
            xs.append(m)
            ys.append(n)

        ans = [[[]],[[]],[[]],[[]]]
        left = np.array(ys).argsort()[0:2] # 1,3
        right = np.array(ys).argsort()[2:4] # 1,3
        if xs[left[0]] <= xs[left[1]]:
            ans[0] = objs[left[0]]
            ans[2] = objs[left[1]]
        else:
            ans[2] = objs[left[0]]
            ans[0] = objs[left[1]]    
        if xs[right[0]] <= xs[right[1]]:
            ans[1] = objs[right[0]]
            ans[3] = objs[right[1]]
        else:
            ans[3] = objs[right[0]]
            ans[1] = objs[right[1]]        
        return ans        
            
    def merge(self, objects, belt, use_color):
        ans=objects
        ans=[[[]],[[]],[[]],[[]]]
        for o in objects:
            o = np.array(o)
            max_total = 0
            for x in [0,1]:
                for y in [0,1]:
                    if max_total < o[x:x+len(o)-1, y:y+len(o[0])-1].sum():
                        max_total = o[x:x+len(o)-1, y:y+len(o[0])-1].sum()
                        max_xy = (x, y)
            if max_xy == (0,0):
                ans[3] = o
            elif max_xy == (0,1):
                ans[2] = o
            elif max_xy == (1,0):
                ans[1] = o
            else:
                ans[0] = o

        if belt == 0:
            belt_list = [[use_color]]*len(ans[0])
            u=np.hstack([ans[0], ans[1]])
            u
            s=np.hstack([ans[2], ans[3]])
            return np.vstack([u,s])
        else:
            belt_list = [[use_color]*belt]*len(ans[0])

            u=np.hstack([ans[0], belt_list, ans[1]])
            s=np.hstack([ans[2], belt_list, ans[3]])
            belt_list = [[use_color]*len(s[0])]*belt
            return np.vstack([u,belt_list,s])


# In[ ]:





# In[190]:


def about_color_for_test(task_test):
    in_colors = []
    out_colors = []
    color_changed = False
    for inout in task_test:
        in_vec = list(itertools.chain.from_iterable(inout['input']))
        in_colors += list(set(in_vec))
    return list(set(in_colors))

def all_inputs_same_shape_and_all_outputs_same_shape(task_train):
    m1, n1 = np.array(task_train[0]['input']).shape
    m2, n2 = np.array(task_train[0]['output']).shape
    all_inputs_same_shape = True
    all_outputs_same_shape = True
    for inout in task_train:
        m1_, n1_ = np.array(inout['input']).shape
        m2_, n2_ = np.array(inout['output']).shape
        if (m1_ != m1) or (n1_ != n1):
            all_inputs_same_shape = False
        if (m2_ != m2) or (n2_ != n2):
            all_outputs_same_shape = False
    return all_inputs_same_shape, all_outputs_same_shape



def change_color_for_p(inp_origin, in_use):
    inp = copy.deepcopy(inp_origin)
    color_map = {}
    use_color_num = len(in_use)
    out_colors = range(use_color_num)
    for i,o in zip(sorted(in_use), sorted(out_colors)):
        color_map[i] = o

    for i, o in color_map.items():
        inp = np.where(np.array(inp_origin) == i, o, inp)
    return inp.tolist()


# In[191]:


def all_inouts_same_shape(task_train):
    for inout in task_train:
        if np.array(inout['input']).shape != np.array(inout['output']).shape:
            return False
    return True


# In[192]:


def update_by_teacher_row_n(inp_o, back=False, n=0):
    try:
        one_color_rows = []
        not_one_color_rows = []
        inp = copy.deepcopy(inp_o)
        for row in inp:
            if len(set(row)) != 1:
                not_one_color_rows.append(row)
            else:
                one_color_rows.append(row)
        c = collections.Counter(np.array(inp).flatten().tolist())
        back_color = c.most_common()[0][0]
        for row_n, row in enumerate(inp):
            if len(set(row)) == 1:
                continue
            tea = copy.deepcopy(not_one_color_rows[n])
            success = True
            for ele_n, ele in enumerate(row):
                if (ele != back_color) & (tea[ele_n] != ele):
                    success = False
            if success:
                inp[row_n] = tea
            else:
                inp[row_n] = [back_color] * len(row)

        return np.array(inp).tolist()
    except:
        return [[0]]    


# In[193]:


def perfect_same(big, small):
    for x in range(len(big[0])-len(small[0])+1):
        for y in range(len(big)-len(small)+1):
            if np.array(big)[y:y+len(small), x:x+len(small[0])].tolist() == small.tolist():
                return True
    small = np.flipud(small)
    for x in range(len(big[0])-len(small[0])+1):
        for y in range(len(big)-len(small)+1):
            if np.array(big)[y:y+len(small), x:x+len(small[0])].tolist() == small.tolist():
                return True    
    small = np.flip(small)
    for x in range(len(big[0])-len(small[0])+1):
        for y in range(len(big)-len(small)+1):
            if np.array(big)[y:y+len(small), x:x+len(small[0])].tolist() == small.tolist():
                return True    
    return False            


# In[194]:


from collections import Counter
def connect_two_point(inp_o, fill=False):
    try:
        counter = Counter(np.array(inp_o).flatten().tolist())
        back = counter.most_common()[0][0]
        inp = copy.deepcopy(np.array(inp_o))
        for row_n, row in enumerate(inp_o):
            start = -1
            for ele_n, ele in enumerate(row):
                if ele != back:
                    if start == -1:
                        start = ele_n
                    else:
                        end = ele_n
                        back_pos = (start + end) // 2
                        for i in range(back_pos - start - 1):
                            inp[row_n, start+1+i] = row[start]
                            inp[row_n, end-1-i] = row[end]
                        if ((end - start) % 2 == 1) & fill:
                            i += 1
                            inp[row_n, start+1+i] = row[start]
                            inp[row_n, end-1-i] = row[end]                          
                        start = ele_n

        for row_n, row in enumerate(np.transpose(inp_o)):
            start = -1
            for ele_n, ele in enumerate(row):
                if ele != back:
                    if start == -1:
                        start = ele_n
                    else:
                        end = ele_n
                        back_pos = (start + end) // 2
                        for i in range(back_pos - start - 1):
                            inp[start+1+i, row_n] = row[start]
                            inp[end-1-i, row_n] = row[end]
                        if ((end - start) % 2 == 1) & fill:
                            i += 1
                            inp[start+1+i, row_n] = row[start]
                            inp[end-1-i, row_n] = row[end]                        
                        start = ele_n
        return inp.tolist()
    except:
        return [[0]]
        


# In[196]:


from lightgbm import LGBMClassifier
import pdb

def preds_to_str(preds_list, idx=''):
    pred_strs = []
#     st()
    for i in range(len(preds_list[0])):
        pred_str = ''
        for j, preds in enumerate(reversed(preds_list)):
            if j == 3:
                break
            pred_str += flattener(np.array(preds[i]).tolist()) + ' '
        pred_strs.append(pred_str)
    return pred_strs

data_path = Path('../input/abstraction-and-reasoning-challenge/')
data_path = Path('data/')

test_path = data_path / 'test'
# https://www.kaggle.com/inversion/abstraction-and-reasoning-starter-notebook
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

def get_moore_neighbours(color, cur_row, cur_col, nrows, ncols):

    if cur_row<=0: top = -1
    else: top = color[cur_row-1][cur_col]
        
    if cur_row>=nrows-1: bottom = -1
    else: bottom = color[cur_row+1][cur_col]
        
    if cur_col<=0: left = -1
    else: left = color[cur_row][cur_col-1]
        
    if cur_col>=ncols-1: right = -1
    else: right = color[cur_row][cur_col+1]
        
    return top, bottom, left, right

def get_tl_tr(color, cur_row, cur_col, nrows, ncols):
        
    if cur_row==0:
        top_left = -1
        top_right = -1
    else:
        if cur_col==0: top_left=-1
        else: top_left = color[cur_row-1][cur_col-1]
        if cur_col==ncols-1: top_right=-1
        else: top_right = color[cur_row-1][cur_col+1]   
        
    return top_left, top_right

def features(task, mode='train'):
    cur_idx = 0
    num_train_pairs = len(task[mode])
    total_inputs = sum([len(task[mode][i]['input'])*len(task[mode][i]['input'][0]) for i in range(num_train_pairs)])
    feat = np.zeros((total_inputs,nfeat))
    target = np.zeros((total_inputs,), dtype=np.int)
    
    global local_neighb
    for task_num in range(num_train_pairs):
        input_color = np.array(task[mode][task_num]['input'])
        target_color = task[mode][task_num]['output']
        nrows, ncols = len(task[mode][task_num]['input']), len(task[mode][task_num]['input'][0])

        target_rows, target_cols = len(task[mode][task_num]['output']), len(task[mode][task_num]['output'][0])
        
        if (target_rows!=nrows) or (target_cols!=ncols):
            print('Number of input rows:',nrows,'cols:',ncols)
            print('Number of target rows:',target_rows,'cols:',target_cols)
            not_valid=1
            return None, None, 1

        for i in range(nrows):
            for j in range(ncols):
                feat[cur_idx,0] = i
                feat[cur_idx,1] = j
                feat[cur_idx,2] = input_color[i][j]
                feat[cur_idx,3:7] = get_moore_neighbours(input_color, i, j, nrows, ncols)
                feat[cur_idx,7:9] = get_tl_tr(input_color, i, j, nrows, ncols)
                feat[cur_idx,9] = len(np.unique(input_color[i,:]))
                feat[cur_idx,10] = len(np.unique(input_color[:,j]))
                feat[cur_idx,11] = (i+j)
                feat[cur_idx,12] = len(np.unique(input_color[i-local_neighb:i+local_neighb,
                                                             j-local_neighb:j+local_neighb]))
        
                target[cur_idx] = target_color[i][j]
                cur_idx += 1
            
    return feat, target, 0

all_task_ids = sorted(os.listdir(test_path))
lgb_range = [24]

nfeat = 13
local_neighb = 5
valid_scores = {}
for task_n, task_id in enumerate(all_task_ids):
    if task_n not in lgb_range:
        continue
    task_file = str(test_path / task_id)
    with open(task_file, 'r') as f:
        task = json.load(f)

    feat, target, not_valid = features(task)
    if not_valid:
        not_valid = 0
        continue

    nrows, ncols = len(task['train'][-1]['input']
                       ), len(task['train'][-1]['input'][0])
    # use the last train sample for validation
    val_idx = len(feat) - nrows*ncols

    train_feat = feat[:val_idx]
    val_feat = feat[val_idx:, :]

    train_target = target[:val_idx]
    val_target = target[val_idx:]

    #     check if validation set has a new color
    #     if so make the mapping color independant
    if len(set(val_target) - set(train_target)):
        continue

    lgb = LGBMClassifier(n_estimators=50, n_jobs=-1)
    lgb.fit(feat, target,
            verbose=-1)

#     training on input pairs is done.
#     test predictions begins here

    num_test_pairs = len(task['test'])
    for task_num in range(num_test_pairs):
        cur_idx = 0
        input_color = np.array(task['test'][task_num]['input'])
        nrows, ncols = len(task['test'][task_num]['input']), len(
            task['test'][task_num]['input'][0])
        feat = np.zeros((nrows*ncols, nfeat))
        unique_col = {col: i for i, col in enumerate(
            sorted(np.unique(input_color)))}

        for i in range(nrows):
            for j in range(ncols):
                feat[cur_idx, 0] = i
                feat[cur_idx, 1] = j
                feat[cur_idx, 2] = input_color[i][j]
                feat[cur_idx, 3:7] = get_moore_neighbours(
                    input_color, i, j, nrows, ncols)
                feat[cur_idx, 7:9] = get_tl_tr(
                    input_color, i, j, nrows, ncols)
                feat[cur_idx, 9] = len(np.unique(input_color[i, :]))
                feat[cur_idx, 10] = len(np.unique(input_color[:, j]))
                feat[cur_idx, 11] = (i+j)
                feat[cur_idx, 12] = len(np.unique(input_color[i-local_neighb:i+local_neighb,
                                                              j-local_neighb:j+local_neighb]))

                cur_idx += 1

        preds = lgb.predict(feat).reshape(nrows, ncols)
        preds = preds.astype(int).tolist()
        submission.loc[submission.index == f'{task_id[:-5]}_{task_num}', 'output'] = flattener(preds) 


# In[1]:


def about_color(task_train):
    in_colors = []
    out_colors = []
    color_changed = False
    for inout in task_train:
        in_vec = list(itertools.chain.from_iterable(inout['input']))
        in_colors += list(set(in_vec))
        out_vec = list(itertools.chain.from_iterable(inout['output']))
        out_colors += list(set(out_vec))
        if set(in_vec) != set(out_vec):
            color_changed = True
    return list(set(in_colors)), list(set(out_colors)), color_changed
def paint_rolling2(inp, back_color, pos):
    i = 1
    while True:
        if i % 4 == 1:
            inp[pos[0]:pos[0]+i, pos[1]] = back_color
            pos = [pos[0]+i, pos[1]]
        elif i % 4 == 2:
            if pos[1]-i+1 < 0:
                inp[pos[0], :pos[1]+1] = back_color
            else:
                inp[pos[0], pos[1]-i+1:pos[1]+1] = back_color
            pos = [pos[0], pos[1]-i]
        elif i % 4 == 3:
            inp[pos[0]-i+1:pos[0]+1, pos[1]] = back_color
            pos = [pos[0]-i, pos[1]]
        elif i % 4 == 0:
            inp[pos[0], pos[1]:pos[1]+i] = back_color
            pos = [pos[0], pos[1]+i]
        i += 1
        if (pos[0]<0) or (pos[1] < 0) or (pos[0] >= inp.shape[0]) or (pos[1] >= inp.shape[1]) or (i > 100):
#             inp[:, -2] = back_color
            # inp[0, :] = back_color
            return inp
def paint_each_and_vstack2(inp):
    try:
        for i in range(len(inp)):
            if len(set(inp[i])) != 1:
                a = np.array(inp[:i])
                b = np.array(inp[i:])
        for i in range(len(inp)):
            if len(set(inp[i])) == 1:
                back_color = inp[i][0]
                break
        use_color = list(set(list(itertools.chain.from_iterable(a)))-set([back_color]))[0]
        pos = tuple(nd[0] for nd in np.where(a == use_color))
        pos = [pos[0]+1, pos[1]+1]
        a = [[use_color]*a.shape[1]]*a.shape[0]

        use_color = list(set(list(itertools.chain.from_iterable(b)))-set([back_color]))[0]
        b = [[use_color]*b.shape[1]]*b.shape[0]

        mat = np.vstack([a,b])
        ud_flag = False
        if np.array(a).shape[0] > np.array(b).shape[0]:
            mat = np.flipud(mat)
            ud_flag = True

        mat = paint_rolling2(mat, back_color, pos)

        if ud_flag:
            mat = np.flipud(mat)

        return mat
    except:
        return inp 

def add_stack4(task):
    skip = False
    success = False
    if task['test'][0]['input'] != np.transpose(task['test'][0]['input']).tolist():
        return False
    for inout in task['train']:
        inp_min_n_shape = np.array(inout['input']).shape
        out_min_n_shape = np.array(inout['output']).shape
        if (inp_min_n_shape[0] * 2 - 1 != out_min_n_shape[0]) or (inp_min_n_shape[1] * 2 - 1 != out_min_n_shape[1]):
            return False
        print(3)
        inp = inout['input']
        out = inout['output']
        if (np.flip(stack4(np.flip(inp))).tolist() == out) or (np.array(stack4(inp)).tolist() == out):
            return True
    return False


def rebuild_by_identified_objects(objs, background, x, pattern):
    try:
        size_map = {}
        for i, o in enumerate(objs):
            size_map[i] = len(np.array(o).flatten())
        size_map = sorted(size_map.items(), key=lambda x:x[1])
        out = copy.deepcopy(objs[size_map[2][0]])
        out_color = out[1][1]
        ele = np.array(objs[size_map[pattern[0]][0]])
        ele = np.where(ele==background, out_color, ele)
        cood = objs[size_map[pattern[1]][0]]
        for row_n, row in enumerate(cood):
            for col_n, r in enumerate(row):
                if r != background:
                    out[row_n*len(ele):(row_n+1)*len(ele), col_n*len(ele[0]):(col_n+1)*len(ele[0])] = ele
        for i in range((x-len(out[0]))//2):
            out = np.insert(out, 0, background,axis=0)
            out = np.insert(out, 0, background,axis=1)
            out = np.insert(out, len(out[0]), background,axis=1)
            out = np.insert(out, len(out), background,axis=0)
        return out
    except:
        return [[0]]

def recolor_by_origin_placement(inp_o, obj, background):
    inp = np.array(copy.deepcopy(inp_o))
    coods = []
    obj_coods = []
    x=0
    for i in range(len(obj)):
        y=0
        x+=1
        for j in range(len(obj[0])):
            y+=1
            if np.all(inp[x+i*len(obj):x+(i+1)*len(obj), y+j*len(obj[0]):y+(j+1)*len(obj[0])] == obj):
                coods.append([x+i*len(obj),y+j*len(obj[0])])
                obj_coods.append([i,j])
    inp = np.where(inp_o == background, obj[0][0], inp)
    inp = np.where(inp_o == obj[0][0], background, inp)
    print(coods)
    for c in obj_coods:
        obj[c[0]][c[1]] = background
    for c in coods:
        inp[c[0]:c[0]+len(obj), c[1]:c[1]+len(obj[0])] = obj
    return inp
def paint_rolling4(inp, back_color, pos):
    i = 1
    while True:
        if i % 4 == 1:
            inp[pos[0]:pos[0]+i, pos[1]] = back_color
            pos = [pos[0]+i, pos[1]]
        elif i % 4 == 2:
            if pos[1]-i+1 < 0:
                inp[pos[0], :pos[1]+1] = back_color
            else:
                inp[pos[0], pos[1]-i+1:pos[1]+1] = back_color
            pos = [pos[0], pos[1]-i]
        elif i % 4 == 3:
            inp[pos[0]-i+1:pos[0]+1, pos[1]] = back_color
            pos = [pos[0]-i, pos[1]]
        elif i % 4 == 0:
            inp[pos[0], pos[1]:pos[1]+i] = back_color
            pos = [pos[0], pos[1]+i]
        i += 1
        if (pos[0]<0) or (pos[1] < 0) or (pos[0] >= inp.shape[0]) or (pos[1] >= inp.shape[1]) or (i > 100):
            inp[:, -2:] = back_color
            # inp[0, :] = back_color
            return inp

def paint_each_and_vstack4(inp):
    try:
        for i in range(len(inp)):
            if len(set(inp[i])) != 1:
                a = np.array(inp[:i])
                b = np.array(inp[i:])
        for i in range(len(inp)):
            if len(set(inp[i])) == 1:
                back_color = inp[i][0]
                break
        use_color = list(set(list(itertools.chain.from_iterable(a)))-set([back_color]))[0]
        pos = tuple(nd[0] for nd in np.where(a == use_color))
        pos = [pos[0]+1, pos[1]+1]
        a = [[use_color]*a.shape[1]]*a.shape[0]

        use_color = list(set(list(itertools.chain.from_iterable(b)))-set([back_color]))[0]
        b = [[use_color]*b.shape[1]]*b.shape[0]

        mat = np.vstack([a,b])
        ud_flag = False
        if np.array(a).shape[0] > np.array(b).shape[0]:
            mat = np.flipud(mat)
            ud_flag = True

        mat = paint_rolling4(mat, back_color, pos)

        if ud_flag:
            mat = np.flipud(mat)

        return mat
    except:
        return inp

def paint_rolling3(inp, back_color, pos):
    i = 1
    while True:
        if i % 4 == 1:
            inp[pos[0]:pos[0]+i, pos[1]] = back_color
            pos = [pos[0]+i, pos[1]]
        elif i % 4 == 2:
            if pos[1]-i+1 < 0:
                inp[pos[0], :pos[1]+1] = back_color
            else:
                inp[pos[0], pos[1]-i+1:pos[1]+1] = back_color
            pos = [pos[0], pos[1]-i]
        elif i % 4 == 3:
            inp[pos[0]-i+1:pos[0]+1, pos[1]] = back_color
            pos = [pos[0]-i, pos[1]]
        elif i % 4 == 0:
            inp[pos[0], pos[1]:pos[1]+i] = back_color
            pos = [pos[0], pos[1]+i]
        i += 1
        if (pos[0]<0) or (pos[1] < 0) or (pos[0] >= inp.shape[0]) or (pos[1] >= inp.shape[1]) or (i > 100):
            inp[:, -2] = back_color
            # inp[0, :] = back_color
            return inp

def paint_each_and_vstack3(inp):
    try:
        for i in range(len(inp)):
            if len(set(inp[i])) != 1:
                a = np.array(inp[:i])
                b = np.array(inp[i:])
        for i in range(len(inp)):
            if len(set(inp[i])) == 1:
                back_color = inp[i][0]
                break
        use_color = list(set(list(itertools.chain.from_iterable(a)))-set([back_color]))[0]
        pos = tuple(nd[0] for nd in np.where(a == use_color))
        pos = [pos[0]+1, pos[1]+1]
        a = [[use_color]*a.shape[1]]*a.shape[0]

        use_color = list(set(list(itertools.chain.from_iterable(b)))-set([back_color]))[0]
        b = [[use_color]*b.shape[1]]*b.shape[0]

        mat = np.vstack([a,b])
        ud_flag = False
        if np.array(a).shape[0] > np.array(b).shape[0]:
            mat = np.flipud(mat)
            ud_flag = True

        mat = paint_rolling3(mat, back_color, pos)

        if ud_flag:
            mat = np.flipud(mat)

        return mat
    except:
        return inp

def stack4(inp_o):
    try:
        inp = np.array(copy.deepcopy(inp_o))
#         inp = np.where(inp==inp.T, inp, inp[-1][-1])
        a = inp
        b = np.fliplr(inp)
        c = np.flipud(inp)
        d = np.flip(inp)
        e = np.hstack([a,b[:, 1:]])
        f = np.hstack([c,d[:, 1:]])
        return np.vstack([e, f[1:, :]])
    except:
        return inp_o                

def copy_by_belt_and_change_color(inp_o, change, to_back, to_belt=False, reverse=False, mirror=True):
    try:
        inp = copy.deepcopy(inp_o)
        belt = inp[0][0]
        one_color_col_colors = []
        for col in np.transpose(inp).tolist():
            if len(set(col)) == 1:
                test_has_one_color_col = True
                one_color_col_colors.append(col[0])
        one_color_col_colors = list(set(one_color_col_colors))
        back = inp[0][0]
        if len(set(np.array(inp)[:, 0])) == 1:
            back = inp[0][0]
        elif len(set(np.array(inp)[:, -1])) == 1:
            back = inp[-1][-1]
        if one_color_col_colors[0] == back:
            belt = one_color_col_colors[1]
        else:
            belt = one_color_col_colors[0]

        belt_xs = []
        for ele_n, ele in enumerate(inp[0]):
            if ele == belt:
                belt_xs.append(ele_n)
        change_left = False
        change_right = True
        if np.array(inp)[:, :belt_xs[0]].flatten().tolist().count(back) / len(np.array(inp)[:, :belt_xs[0]].flatten().tolist()) > np.array(inp)[:, belt_xs[-1]+1:].flatten().tolist().count(back) / len(np.array(inp)[:, belt_xs[1]+1:].flatten().tolist()):
            change_left = True
            change_right = False

        range_x = np.min([belt_xs[0], len(inp[0])-belt_xs[-1]-1])
        inp = np.array(inp)
        use_colors = list(set(inp.flatten().tolist()) - set([back, belt]))
        if len(use_colors) == 0:
            use_color = belt
        else:
            use_color = use_colors[0]

        for x in range(range_x):
            for y in range(len(inp)):
                a, b = inp[y, belt_xs[0]-x-1], inp[y, belt_xs[-1]+x+1]
                if (a != back) & (b != back):
                    if a == b:
                        if to_back:
                            inp[y, belt_xs[-1]+x+1] = back
                            inp[y, belt_xs[0]-x-1] = back
                        elif change:
                            if a == belt:
                                inp[y, belt_xs[-1]+x+1] = use_color
                                inp[y, belt_xs[0]-x-1] = use_color
                            else:
                                inp[y, belt_xs[-1]+x+1] = belt
                                inp[y, belt_xs[0]-x-1] = belt
                    else:
                        if to_belt:
                            inp[y, belt_xs[-1]+x+1] = belt
                            inp[y, belt_xs[0]-x-1] = belt
                        elif reverse:
                            inp[y, belt_xs[-1]+x+1] = a
                            inp[y, belt_xs[0]-x-1] = b
                        else:
                            if a == belt:
                                inp[y, belt_xs[-1]+x+1] = use_color
                            else:
                                inp[y, belt_xs[-1]+x+1] = belt
                            if b == belt:
                                inp[y, belt_xs[0]-x-1] = use_color
                            else:
                                inp[y, belt_xs[0]-x-1] = belt
                elif (a != back):
                    if a == belt:
                        inp[y, belt_xs[-1]+x+1] = use_color
                    else:
                        inp[y, belt_xs[-1]+x+1] = belt
                elif inp[y, belt_xs[-1]+x+1] != back:
                    if b == belt:
                        inp[y, belt_xs[0]-x-1] = use_color
                    else:
                        inp[y, belt_xs[0]-x-1] = belt
        if not mirror:
            if change_left:
                inp[:, belt_xs[0]-range_x:belt_xs[0]] = np.fliplr(inp[:, belt_xs[0]-range_x:belt_xs[0]])
            else:
                inp[:, belt_xs[1]+1:belt_xs[1]+1+range_x] = np.fliplr(inp[:, belt_xs[1]+1:belt_xs[1]+1+range_x])

        return inp
    except:
        return [[0]]
def add_recolor_by_origin_placement(task):
    inp = task['test'][0]['input']
    inp_train = task['train'][0]['input']
    if (len(inp) != len(inp[0])) or (len(inp_train) != len(inp_train[0])):
        return False

    use_color = list(set(list(itertools.chain.from_iterable(inp))))
    if len(use_color) != 2:
        return False
    for inout in task['train']:
        success = False

        i_test = np.array(inout['input'])
        arc = ARC_solver()
        background = arc.get_background(i_test)
        arc.identify_object_by_isolation(i_test, background)

        a = recolor_by_origin_placement(i_test, arc.identified_objects[0], background)
        if np.array(a).tolist() == inout['output']:
            success = True
            break
    return success

def add_rebuild_by_identified_objects(task):
    use_colors = list(set(np.array(task['test'][0]['input']).flatten().tolist()))
    if len(use_colors) != 4:
        return False
    inp = task['train'][-2]['input']
    out = task['train'][-2]['output']
    if (len(inp[0]) != len(out[0])):
        return False

    success = False
    for test_n, inout in enumerate(task['train']):
        i_test = np.array(inout['input'])
        arc = ARC_solver()
        background = arc.get_background(i_test)
        arc.identify_object_by_isolation(i_test, background)

        a = rebuild_by_identified_objects(arc.identified_objects, background, len(i_test[0]), [0,1])
        b = rebuild_by_identified_objects(arc.identified_objects, background, len(i_test[0]), [1,0])
        if (np.array(a).tolist() == inout['output']) or (np.array(b).tolist() == inout['output']):
            success = True
            break
    return success

def add_copy_by_belt_and_change_color(task):
    skip = False
    inp = task['test'][0]['input']
    for n, row in enumerate(inp):
        if len(set(row)) == 1:
            skip = True
    if skip:
        return False
    unique_one_color_col_ns = []
    for n, col in enumerate(np.transpose(inp)):
        if len(set(col)) == 1:
            unique_one_color_col_ns.append(col[0])
    if len(set(unique_one_color_col_ns)) != 2:
        return False
    success = False
    for test_n, inout in enumerate(task['train']):
        i_test = np.transpose(inout['input']).tolist()
        a = np.transpose(copy_by_belt_and_change_color(i_test,True,False, mirror=False)).tolist()
        b = np.transpose(copy_by_belt_and_change_color(i_test,True,False, reverse=True)).tolist()
        c = np.transpose(copy_by_belt_and_change_color(i_test,True,False, to_belt=True)).tolist()
        if (a == inout['output']) or (b == inout['output']) or (c == inout['output']):
            success = True
            break
    if not success:
        return False
    return True

def add_paint_each_and_vstack(task):
    inp = copy.deepcopy(task['train'][-1]['input'])
    in_use, out_use, color_changed = about_color(copy.deepcopy(task['train'][-1:]))
    if len(in_use) != 3:
        return False
    v=list(itertools.chain.from_iterable(inp))
    if (len(v) - v.count(in_use[0]) != 2) & (len(v) - v.count(in_use[1]) != 2) & (len(v) - v.count(in_use[2]) != 2):
        return False
    if np.array(paint_each_and_vstack2(inp)).tolist() != task['train'][-1]['output']:
        return False
    return True

def connect_two_point2(inp_o, fill=False):
    try:
        counter = Counter(np.array(inp_o).flatten().tolist())
        back = counter.most_common()[0][0]
        inp = copy.deepcopy(np.array(inp_o))

        for row_n, row in enumerate(np.transpose(inp_o)):
            start = -1
            for ele_n, ele in enumerate(row):
                if ele != back:
                    if start == -1:
                        start = ele_n
                    else:
                        end = ele_n
                        back_pos = (start + end) // 2
                        for i in range(back_pos - start - 1):
                            inp[start+1+i, row_n] = row[start]
                            inp[end-1-i, row_n] = row[end]
                        if ((end - start) % 2 == 1) & fill:
                            i += 1
                            inp[start+1+i, row_n] = row[start]
                            inp[end-1-i, row_n] = row[end]                        
                        start = ele_n
        return inp.tolist()
    except:
        return [[0]]  
def add_connect_two_point2(task):
    success = False
    for inout in task['train']:
        if (np.array(connect_two_point2(inout['input'], fill=True)).tolist() == inout['output']) or (np.transpose(connect_two_point2(np.transpose(inout['input']), fill=True)).tolist() == inout['output']):
            success = True
    return success
def stack4_2(inp_o):
    try:
        inp = np.array(copy.deepcopy(inp_o))
        inp[-1][-2] = inp[-1][-1]
        inp[-2][-1] = inp[-1][-1]
        a = inp
        b = np.fliplr(inp)
        c = np.flipud(inp)
        d = np.flip(inp)
        e = np.hstack([a,b[:, 1:]])
        f = np.hstack([c,d[:, 1:]])
        return np.vstack([e, f[1:, :]])
    except:
        return inp_o

def add_several_funcs(task):
    try:
        if (len(task['test'][0]['input']) % 16 == 0) & (len(task['test'][0]['input'][0]) % 16 == 0):
            return None, False
        use_flag = add_recolor_by_origin_placement(task)
        if use_flag:
            return recolor_by_origin_placement, True
        use_flag = add_rebuild_by_identified_objects(task)
        if use_flag:
            return rebuild_by_identified_objects, True
        use_flag = add_copy_by_belt_and_change_color(task)
        if use_flag:
            return copy_by_belt_and_change_color, True
        use_flag = add_paint_each_and_vstack(task)
        if use_flag:
            return paint_each_and_vstack3, True
        use_flag = add_stack4(task)
        if use_flag:
            return stack4, True
        use_flag = add_connect_two_point2(task)
        if use_flag:
            return connect_two_point2, True
        if add_block_merge(task['train']):
                return divide_block_and_merge3, True
        return None, False
    except:
        return None, False

def apply_several_func(task, func):
    try:
        if func == recolor_by_origin_placement:
            for test_n, inout in enumerate(task['test']):
                i_test = np.array(inout['input'])
                arc = ARC_solver()
                background = arc.get_background(i_test)
                arc.identify_object_by_isolation(i_test, background)

                preds = []
                a = recolor_by_origin_placement(i_test, arc.identified_objects[0], background)
                b = [[0]]
                c = [[0]]
                return [a,b,c]
        elif func == rebuild_by_identified_objects:
            for test_n, inout in enumerate(task['test']):
                i_test = np.array(inout['input'])
        #         p(i_test)
                arc = ARC_solver()
                background = arc.get_background(i_test)
                arc.identify_object_by_isolation(i_test, background)

                preds = []
                a = rebuild_by_identified_objects(arc.identified_objects, background, len(i_test[0]), [0,1])
                b = rebuild_by_identified_objects(arc.identified_objects, background, len(i_test[0]), [1,0])
                c = [[0]]
                return [a,b,c]
        elif func == copy_by_belt_and_change_color:
            for test_n, inout in enumerate(task['test']):
                i_test = inout['input']
                preds = []
                a = copy_by_belt_and_change_color(i_test,True,False, mirror=False)
                b = copy_by_belt_and_change_color(i_test,True,False, reverse=True)
                c = copy_by_belt_and_change_color(i_test,True,False, to_belt=True)
                return [a,b,c]
        elif func == paint_each_and_vstack3:
            for test_n, inout in enumerate(task['test']):
                i_test = inout['input']
                a=paint_each_and_vstack3(np.flip(i_test))
                b=paint_each_and_vstack3(i_test)
                c=paint_each_and_vstack4(i_test)
                return [a,b,c]
        elif func == stack4:
            for test_n, inout in enumerate(task['test']):
                i_test = inout['input']
                if i_test[0][0] == i_test[0][1]:
                    a=stack4_2(np.flip(i_test))
                    b=stack4(np.flip(i_test))
                    c=stack4(i_test)
                else:
                    a=stack4_2(i_test)
                    b=stack4(i_test)
                    c=stack4(np.flip(i_test))
                return [a,b,c]
        elif func == connect_two_point2:
            for test_n, inout in enumerate(task['test']):
                i_test = inout['input']
                preds = []
                a = connect_two_point2(inout['input'], fill=False)
                b = connect_two_point2(inout['input'], fill=True)
                c = np.transpose(connect_two_point2(np.transpose(inout['input']), fill=True)).tolist()
                return [a, b, c]
        elif func == divide_block_and_merge3:
            t1=divide_block_and_merge3(task['test'], 1)
            t2=divide_block_and_merge3(task['test'], 2)
            t3=divide_block_and_merge3(task['test'], 3)
            return [t1[0]['input'], t2[0]['input'], t3[0]['input']]

    except:
        return []
    
def add_block_merge(task_train):
    try:
        arc = ARC_solver()
        inout = task_train[-1]
        inp = copy.deepcopy(inout['input'])
        inp = np.array(inp)
        use_color = list(set(list(itertools.chain.from_iterable(inp))))
        if len(use_color) != 2:
            return False
        inp_o = copy.deepcopy(inp)
        inp = np.where(inp_o==use_color[0], use_color[1], inp)
        inp = np.where(inp_o==use_color[1], use_color[0], inp)
        background = arc.get_background(inp)
        arc.identify_object_by_isolation(inp, background)
        if len(arc.identified_objects) != 4:
            return False

#             arc.identified_objects = arc.sort(arc.identified_objects, inp)
#             for i in arc.identified_objects:
#                 p(i)
#         out = arc.merge(arc.identified_objects, 1, use_color[1])
        for i in range(4):
            out = arc.identified_objects[i]
            out_o = copy.deepcopy(out)
            out = np.where(out_o==use_color[0], use_color[1], out)
            out = np.where(out_o==use_color[1], use_color[0], out)
            if out.tolist() == inout['output']:
                return True
    except:
        pass
    return False    
def divide_block_and_merge3(task_train_origin, obj_numb):
    for inout in task_train_origin:
        inout['input'] = np.array(inout['input'])
    task_train = copy.deepcopy(task_train_origin)
    for i, inout in enumerate(task_train):    
        arc = ARC_solver()
        inp = inout['input']
        inp = np.array(inp)
        use_color = list(set(list(itertools.chain.from_iterable(inp))))
        if len(use_color) != 2:
            return task_train_origin
#         try:
        inp_o = copy.deepcopy(inp)
        inp = np.where(inp_o==use_color[0], use_color[1], inp)
        inp = np.where(inp_o==use_color[1], use_color[0], inp)
        background = arc.get_background(inp)
        arc.identify_object_by_isolation(inp, background)
        if len(arc.identified_objects) == 4:
            arc.identified_objects = arc.sort(arc.identified_objects, inp)
            out = arc.identified_objects[obj_numb]
            out_o = copy.deepcopy(out)
            out = np.where(out_o==use_color[0], use_color[1], out)
            out = np.where(out_o==use_color[1], use_color[0], out)
            task_train[i]['input'] = out
#         except:
#             return task_train_origin
    return task_train
def main(idx, task, env='dev'):
    several_f = False
    func_combi_map = defaultdict(list)
    result = pd.Series()
    preprocess_best_score_map = {}
    best_aug_score_map = {}
    success_list = []
    final_score_map = {}
    pre_final_score_map = {}
    promising_map = defaultdict(bool)
    time_map = {}
    skip = False
    back_to_black = False
    origin_back_color = 1
    preprocess_best_score = 0
    best_func_combi = []
    task_n = 0
    correct_only_preprocess_flag = False
    use_several_func = False
    start = time()
    print('--------------')
    print(f'{task_n}：{idx}')
    flip_funcs = [inouts_array, inouts_flip, inouts_flipud, inouts_fliplr]
    back_to_black_funcs = add_back_to_black_funcs(copy.deepcopy(task['train']))
    func, use_several_func_flag = add_several_funcs(task)
    if (len(task['test'][0]['input']) % 16 != 0) & (len(task['test'][0]['input'][0]) % 16 != 0) & (not use_several_func_flag):
        skip = True

    for back_to_black_func in back_to_black_funcs:
        if use_several_func_flag:
            outputs = apply_several_func(task, func)
            if len(outputs) != 0:
                use_several_func = True
                break
            else:
                use_several_func_flag = False
        if correct_only_preprocess_flag or use_several_func or skip:
            break
        train_copy0 = back_to_black_func(copy.deepcopy(task['train']))
        size_change_funcs = add_size_change_funcs(train_copy0, task_n)

        for size_change_func in size_change_funcs:
            if correct_only_preprocess_flag or use_several_func:
                break
            shaped_train = size_change_func(copy.deepcopy(train_copy0))
            # print(type(shaped_train))
            transpose_funcs = add_transpose(shaped_train)
            for transpose_func in transpose_funcs:
                if correct_only_preprocess_flag or use_several_func:
                    break
                shaped_train1 = transpose_func(copy.deepcopy(shaped_train))
#                     if size_change_func == divide_block_and_merge1:
#                         st()

                shape_different_flag = False
#                     print(size_change_funcs)
                for shaped_inout in shaped_train1:
                    if shaped_inout['input'].shape != np.array(shaped_inout['output']).shape:
                        shape_different_flag = True
                        break
                if shape_different_flag:
                    break

                train4_funcs = add_train4_growth(shaped_train1)
                for train4_func in train4_funcs:
                    if correct_only_preprocess_flag:
                        break
                    shaped_train2 = train4_func(copy.deepcopy(shaped_train1))
                    # print(type(shaped_train2))
                    fill_closed_area_funcs = add_fill_closed_area(shaped_train2.copy())
                    for fill_closed_area_func in fill_closed_area_funcs:
                        if correct_only_preprocess_flag:
                            break
                        shaped_train3 = fill_closed_area_func(copy.deepcopy(shaped_train2))
                        # print(type(shaped_train3))
                        for flip_func_num, flip_func in enumerate(flip_funcs):
                            if correct_only_preprocess_flag:
                                break
                            shaped_train4 = flip_func(copy.deepcopy(shaped_train3))
                            patch_funcs = add_patch_funcs(shaped_train4, idx)
                            for patch_func in patch_funcs:
                                if correct_only_preprocess_flag:
                                    break
                                shaped_train5 = patch_func(copy.deepcopy(shaped_train4))
                                task_train6_funcs = add_task_train6(shaped_train5)
                                for train6_funcs in task_train6_funcs:
                                    if correct_only_preprocess_flag:
                                        break
                                    shaped_train6 = train6_funcs(copy.deepcopy(shaped_train5))
                                    move_object_funcs = add_move_object(shaped_train6)
                                    for move_object_func in move_object_funcs:
                                        if correct_only_preprocess_flag:
                                            break
                                        shaped_train7 = move_object_func(copy.deepcopy(shaped_train6))
                                        recolor_funcs = add_recolor(shaped_train7, task_n)
                                        for recolor_func in recolor_funcs:
                                            if correct_only_preprocess_flag:
                                                break
                                            shaped_train8 = recolor_func(copy.deepcopy(shaped_train7))
                                            kneighbor_funcs = add_kneighbors(shaped_train8)
                                            for kneighbor_func in kneighbor_funcs:
                                                if correct_only_preprocess_flag:
                                                    break
                                                shaped_train9 = kneighbor_func(copy.deepcopy(shaped_train8))

                                                change_color_funcs = add_change_color_funcs(shaped_train9)
                                                for change_color_func in change_color_funcs:
                                                    if correct_only_preprocess_flag:
                                                        break
                                                    shaped_train10 = change_color_func(copy.deepcopy(shaped_train9))
                                                    second_shape_different_flag = False
                                                    shaped_train_copy = shaped_train10
                                                    func_combi = [func for func in [back_to_black_func, size_change_func, patch_func, flip_func, transpose_func, train4_func, fill_closed_area_func, train6_funcs, move_object_func, recolor_func, kneighbor_func, change_color_func] if func != inouts_array]
                                                    func_combi += [inouts_array] if len(func_combi) == 0 else []
                                                    for train_num, in_out in enumerate(copy.deepcopy(shaped_train_copy)):
                                                        if in_out['input'].shape != np.array(in_out['output']).shape:
                                                            second_shape_different_flag = True
                                                            break
                                                        # st()
                                                        if in_out['input'].tolist() == in_out['output']:
            #                                                 print(func_combi)
                                                            correct_only_preprocess_flag = True
            #                                                 st()
                                                            for idx_minus_num in [1, 2]:
                                                                another_in_out = shaped_train_copy[train_num - idx_minus_num]
                                                                if another_in_out['input'].tolist() != another_in_out['output']:
                                                                    correct_only_preprocess_flag = False
                                                            if correct_only_preprocess_flag:
                                                                func_combi_map[idx].append(func_combi)
                                                                preprocess_best_score = 0.999
                                                    if second_shape_different_flag or correct_only_preprocess_flag:
                                                        continue
                #                                     st()
                                                    similarity = get_similarity(shaped_train_copy, [], idx)
                    #                                 print(func_combi)
                    #                                 print(similarity)
                                                    if similarity > preprocess_best_score:
                                                        func_combi_map[idx].append(func_combi)
                                                        preprocess_best_score = similarity
                                                        best_func_combi = func_combi
                                                        preprocess_best_score_map[idx] = preprocess_best_score

    if use_several_func:
        result[f'{idx}_0'] = ''
        several_f = True
        for out in outputs:
            result[f'{idx}_0'] += flattener(np.array(out).tolist()) + ' '

        success_list.append(task_n)
    elif correct_only_preprocess_flag:
        # TODO: 一回目はこれでやるとして、2回目以降を考える
        print('↓correct_only_preprocess!↓')
        print(f'idx: {idx}, func: {func_combi_map[idx]}')
        success_list.append(task_n)

        preds0, preds1, preds2 = [], [], []
        if divide_block_and_merge1 in func_combi_map[idx][0]:
            funcs0 = [divide_block_and_merge1]
            funcs1 = [divide_block_and_merge2]
            funcs2 = [divide_block_and_merge3]
        else:
            funcs0 = func_combi_map[idx][-1 % len(func_combi_map[idx])]
            funcs1 = func_combi_map[idx][-2 % len(func_combi_map[idx])]
            funcs2 = func_combi_map[idx][-3 % len(func_combi_map[idx])]
#             task_test = copy.deepcopy(task['test'])
#             for f in funcs0:
#                 task_test = f(task_test)
#             st()
        success = False
        final_score_map[idx] = 0
        for i, _ in enumerate(task['test']):
            result[f'{idx}_{i}'] = ''
        for funcs in [funcs0, funcs1, funcs2]:
            task_test = copy.deepcopy(task['test'])
            for func in funcs:
                task_test = func(task_test)
            for i, sample in enumerate(task_test):
                if 'output' in sample:
                    if sample['input'].tolist() == sample['output']:
                        preprocess_best_score_map[idx] = 1.0
                        final_score_map[idx] = 1.0
                pred = flattener(sample['input'].tolist())
                result[f'{idx}_{i}'] += pred + ' '

    elif (len(func_combi_map[idx]) > 0) or (input_output_shape_is_same(task)):
        task_train = copy.deepcopy(task['train'])
        task_test = copy.deepcopy(task['test'])
        if len(func_combi_map[idx]) == 0:
            func_combi_map[idx].append([inouts_array])
        for func in func_combi_map[idx][-1]:
            task_train = func(task_train)
            task_test = func(task_test)

        task_train2 = copy.deepcopy(task['train'])
        task_test2 = copy.deepcopy(task['test'])
        funcs2 = func_combi_map[idx][-2 % len(func_combi_map[idx])]
        for func in funcs2:
            task_train2 = func(task_train2)
            task_test2 = func(task_test2)
        task_train_aug = copy.deepcopy(task_train)
        print(f'preprocess_best_score: {preprocess_best_score}, funcs: {func_combi_map[idx]}')
        if preprocess_best_score > 0.99:
            promising_map[idx] = True
        if preprocess_best_score > 0.7:
            if 'output' in task_test[0]:
                pre_preds = final_train_and_predict(task_train, task_train2, task_train_aug, task_test, task_test2, idx=idx, success_map={}, final_score_map=pre_final_score_map, origin_task=task)
            use_transpose_flag = apply_transpose_aug(task_train)
            color_inouts, color_aug_func = apply_color_aug(task_train, preprocess_best_score, best_aug_score_map, idx, promising_map)
            print(f'color_aug_func: {color_aug_func}')
            mirror_aug_funcs = apply_mirror_aug(task_train, preprocess_best_score, idx, use_transpose_flag, promising_map)
            print(f'mirror_aug_funcs: {mirror_aug_funcs}')
            # こちらも一応試したい
            # mirror_augs = [flipud_aug, fliplr_aug, flip_aug, transpose_aug]
            task_train_aug = task_train + color_inouts
            for mirror_aug_func in mirror_aug_funcs:
                mirror_inouts = mirror_aug_func(task_train)
                task_train_aug += mirror_inouts
#             st()
        print(f'final_train_length： {len(task_train_aug)}')
        preds = final_train_and_predict(task_train, task_train2, task_train_aug, task_test, task_test2, idx=idx, success_map={}, final_score_map=final_score_map, final=True, promising=promising_map[idx], origin_task=task)
        for i, pred in enumerate(preds):
            result[f'{idx}_{i}'] = pred

    else:
        task_test = copy.deepcopy(task['test'])
        inputs = [el['input'] for el in task_test]
        for i, inp in enumerate(inputs):
            result[f'{idx}_{i}'] = getDefaultPred(inp)
    t = time() - start
    print(f'{round(t)}秒')
    time_map[idx] = t
    if (task_n in success_list):
        print(f'-------------------------------------------------------------------------------------success!! idx: {idx}')
    if env == 'production':
        os.system(f'echo {idx}: best_score_with_preprocess: {preprocess_best_score_map.get(idx, "different shapes...")}')
    return result, func_combi_map, success_list, preprocess_best_score_map, final_score_map, best_aug_score_map, pre_final_score_map, time_map, several_f

ca_skip = True
move_skip = True
patch_skip = True
update_tasks = []

private = '00576224' not in data.keys()
if private:
    for idx, task in data.items():
        preds, func_combi_map, success_list, preprocess_best_score_map, final_score_map, best_aug_score_map, pre_final_score_map, time_map, several_f = main(idx, task, 'production')
        if len(success_list) != 0:
            update_tasks.append(i)
            for idx, pred_str in preds.items():
                submission.loc[submission.index == idx, 'output'] = pred_str


# In[ ]:


submission.to_csv('submission.csv')


# In[ ]:





# In[ ]:





# In[ ]:




