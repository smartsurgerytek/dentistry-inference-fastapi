from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, find_contours
from skimage.measure import approximate_polygon
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from src.allocation.domain.pa_dental_segmentation.utils import *
import yaml



