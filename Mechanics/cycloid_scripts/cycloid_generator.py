import numpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from shapely.geometry.polygon import LinearRing
import time
import ezdxf

from pysvg.filter import *
from pysvg.gradient import *
from pysvg.linking import *
from pysvg.script import *
from pysvg.shape import *
from pysvg.structure import *
from pysvg.style import *
from pysvg.text import *
from pysvg.builders import *
from pysvg.parser import parse

from copy import deepcopy

blank = [[0,0]]
mm_to_inch = 1/25.4
def cos(angle):
    return np.cos(np.radians(angle))

def sin(angle):
    return np.sin(np.radians(angle))

def offset(amount, points):
    poly_line = LinearRing(points)
    poly_line_offset = poly_line.parallel_offset(amount, side="left", resolution=16, 
                                            join_style=1, mitre_limit=1)
    return poly_line_offset.coords

global clicked
def on_click(event):
    global clicked
    if event.button is MouseButton.LEFT:
        clicked=True

def setup_plot(amount):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(xlim=(-amount,amount), ylim=(-amount,amount))
    plt.axis('off')

    plt.ion()
    plt.show()

    plt.connect('button_press_event', on_click)
    return ax

def end_plot():
    plt.ioff()
    plt.show(block=True)

def numpyCoordndarry_to_python(verts) -> list[list[float, float]]:
    '''
    converts the numpy array of two coordinates array to python builtins
    '''
    return list( [float(num) for num in vert] for vert in verts )

pin_radius = 2.5
pin_circle_radius = 50 
number_of_pins = 10

# the circumference of the rolling circle needs to be exactly equal to the pitch of the pins
# rolling circle circumference = circumference of pin circle / number of pins
rolling_circle_radius = pin_circle_radius / number_of_pins 
reduction_ratio = number_of_pins - 1 # reduction ratio
cycloid_base_radius = reduction_ratio * rolling_circle_radius # base circle diameter of cycloidal disk

contraction = 2

axes = setup_plot((pin_circle_radius+4*pin_radius))

cycloid_base = plt.Circle((0,0), cycloid_base_radius, fill=False, linestyle='--', lw=2)
axes.add_patch(cycloid_base)

rolling_circle = plt.Circle((0,0), rolling_circle_radius, fill=False, lw=2)
axes.add_patch(rolling_circle)

rolling_circle_line = plt.Line2D((0,1),(0,0), lw=2, color='red')
# axes.add_line(rolling_circle_line)

# polygon to hold the main epicycloid
epicycloid_points = []
epicycloid = plt.Polygon(blank, fill=False, closed=False, color='red', lw=2)
axes.add_patch(epicycloid)

for angle in range(0,361):
    # rotate rolling circle round the center of the cycloid
    x =  (cycloid_base_radius + rolling_circle_radius) * cos(angle)
    y =  (cycloid_base_radius + rolling_circle_radius) * sin(angle)
    rolling_circle.center = (x, y)
    
    point_x = x + (rolling_circle_radius - contraction) * cos(number_of_pins*angle)
    point_y = y + (rolling_circle_radius - contraction) * sin(number_of_pins*angle)

    rolling_circle_line.set_xdata((x,point_x))
    rolling_circle_line.set_ydata((y,point_y))
    
    epicycloid_points.append([point_x,point_y])
    epicycloid.set_xy(epicycloid_points)

    plt.pause(0.0001)

# draw pins
#TODO:
# base_circle = plt.Circle((0, 0), pin_circle_radius)
# axes.add_patch(base_circle)
pin_verts_lists = []
for pin_angle in np.linspace(0,360,num=number_of_pins+1):
    pincircle = plt.Circle((pin_circle_radius*cos(pin_angle)+rolling_circle_radius - contraction, 
                            pin_circle_radius*sin(pin_angle))
                           ,pin_radius)

    axes.add_patch(pincircle)
    # pin_verts.extend(numpyCoordndarry_to_python(pincircle.get_verts()))
    pin_verts_lists.append(pincircle.get_verts())

# polygon to hold the offset epicycloid
offset_epicycloid = plt.Polygon(offset(pin_radius,epicycloid_points), fill=False, closed=True)  #CHANGED: I changed closed to True

axes.add_patch(offset_epicycloid)

### My code
# get coordinates of cycloid
cycloid_verts = offset_epicycloid.get_verts()
# cycloid_verts = numpyCoordndarry_to_python(offset_epicycloid.get_verts())

def coords_to_dxf(version: str, verts: list[list[float, float]], file_name: str) -> None:
    '''
    converting list of coordinates of cycloid to dxf file

    :param verts: the list of coordinates to get exported
    :param file_name: the dxf filename to be exported
    '''
    doc = ezdxf.new(version) # create a new DXF drawing in R2010 fromat 

    msp = doc.modelspace() # add new entities to the modelspace
    msp.add_polyline2d(verts) # add a LINE entity

    doc.saveas(file_name)

def coord_to_dxf2(version: str, verts: list[list[float, float]], file_name: str) -> None:
    '''
    converts verts to dxf file

    :param verts: the list of coordinates to get exported
    :param file_name: the dxf filename to be exported
    '''
    doc = ezdxf.new(version)
    doc.units = ezdxf.units.MM

    msp = doc.modelspace()
    prev_coord = verts[0]
    for vert in verts:
        msp.add_line(prev_coord, vert)
        prev_coord = vert

    doc.saveas(file_name)

def coords_to_dxf2(version: str, verts_list: list[list[list[float, float]]], file_name: str) -> None:
    '''
    converts verts to dxf file

    :param verts: the list of coordinates to get exported
    :param file_name: the dxf filename to be exported
    '''
    doc = ezdxf.new(version)
    doc.units = ezdxf.units.MM

    msp = doc.modelspace()

    for verts in verts_list:
        prev_coord = verts[0]
        for vert in verts:
            msp.add_line(prev_coord, vert)
            prev_coord = vert

    doc.saveas(file_name)


coord_to_dxf2('R2010', cycloid_verts, '../cycloid.dxf')
coords_to_dxf2('R2010', pin_verts_lists, '../pins.dxf')

# coords_to_dxf(verts, '../cycloid2.dxf')
#TODO: add the pins coordinates to the dxf file

# versions = ['R12', 'R2000', 'R2004', 'R2007', 'R2010', 'R2013', 'R2018']

# for version in versions:
#     coords_to_dxf(version, verts, f"{version}.dxf")



def coords_to_svg(verts: list[list[float, float]], file_name: str) -> None:
    '''
    converts set of coords to svg file
    '''

    oh = ShapeBuilder()
    s = Svg('test')
    
    prev_coord = verts[0]
    for coord in verts:
        s.addElement(oh.createLine(prev_coord[0], prev_coord[1], coord[0], coord[1]))
        prev_coord = coord

    s.save(file_name)

# coords_to_svg(verts, "../cycloid.svg")


end_plot()

