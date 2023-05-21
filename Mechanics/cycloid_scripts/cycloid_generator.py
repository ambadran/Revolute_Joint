from __future__ import annotations
from dataclasses import dataclass
import math
import shapely
import ezdxf
from copy import deepcopy

def sin(angle: float) -> float:
    '''
    math.sin but takes in degrees
    '''
    return math.sin(math.radians(angle))

def cos(angle: float) -> float:
    '''
    math.cos but takes in degrees
    '''
    return math.cos(math.radians(angle))


@dataclass
class Coordinate:
    x: int | float
    y: int | float
    z: Optional[int | float] = None
    d: Optional[int] = None

    def __eq__(self, other: Coordinate) -> bool:
        '''
        defines equality between coordinates
        '''
        return (self.x == other.x and self.y == other.y and self.z == other.z)

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        z = "" if self.z is None else f"Z{self.z}"
        return f"(X{self.x}Y{self.y}{z})"

    def __getitem__(self, index) -> float:
        '''
        Enable slicing of coordinate objects
        '''
        if index == 0:
            return self.x

        elif index == 1:
            return self.y

        # elif self.z:
        #     if index == 2:
        #         return self.z

        elif index == 2:
            return self.z

        else:
            raise IndexError("Only X, Y, Z values available")

    def __len__(self) -> int:
        '''
        return whether it's x,y or x,y,z
        '''
        if self.z is None:
            return 2
        else:
            return 3

    def __round__(self, accuracy: int) -> None:
        '''
        Rounds x, y and z values to 'accuracy' decimal place
        '''
        self.x = round(self.x, accuracy)
        self.y = round(self.y, accuracy)
        if self.z:
            self.z = round(self.z, accuracy)

    @classmethod
    def get_min_max(cls, coordinates_list: list[Coordinate]) -> tuple[Coordinate, Coordinate]:
        '''
        finds the Min, Max of X and Y coordinates from input list of coordinates

        :param coordinates: list of coordinates [(x, y), ..]
        :return: ((x_min, y_min), (x_max, y_max))
        '''

        if len(coordinates_list) == 0:
            raise ValueError('An empty list is passed')

        if len(set([type(i) for i in coordinates_list])) != 1:
            raise ValueError('All values in coordinates_list argument must be of type Coordinate')

        if type(coordinates_list[0]) != Coordinate:
            raise ValueError('All values in coordinates_list argument must be of type Coordinate')

        x_min, y_min = coordinates_list[0].x, coordinates_list[0].y
        x_max, y_max = x_min, y_min

        for coordinate in coordinates_list:

            new_x, new_y = coordinate.x, coordinate.y

            if new_x < x_min:
                x_min = new_x

            elif new_x > x_max:
                x_max = new_x

            if new_y < y_min:
                y_min = new_y

            elif new_y > y_max:
                y_max = new_y

        return Coordinate(x_min, y_min), Coordinate(x_max, y_max)

@dataclass
class Edge:
    start: Coordinate 
    end: Coordinate
    thickness: float

    @property
    def delta_x(self) -> float:
        '''
        :return delta x
        '''
        return round(self.end.x - self.start.x, 5)

    @property
    def delta_y(self) -> float:
        '''
        :return: delta y
        '''
        return round(self.end.y - self.start.y, 5)
    
    @property
    def delta_z(self) -> float:
        '''
        :return: delta z
        '''
        return round(self.end.z - self.start.z, 5)

    @property
    def gradient(self) -> float:
        '''
        Assumes the edge as a linear equation
        m = (y2-y1)/(x2-x1)

        :returns: gradient of the edge
        '''
        try:
            return round(self.delta_y / self.delta_x, 3)
        except ZeroDivisionError:
            return Infinity() 

    @property
    def y_intercept(self) -> float:
        '''
        Assumes the edge as linear equation
        y=mx+c
        c=y-mx
taking y as y1 and x as x1 

        :return: y intercept of the edge
        '''
        return self.start.y - self.gradient*self.start.x

    @property
    def absolute_length(self) -> float:
        '''
        return absolute length
        '''
        if self.start.z == None or self.end.z == None:
            return round(math.sqrt(round((self.delta_x)**2 + (self.delta_y)**2, 5)), 6)

        else:
            return round(math.sqrt(round((self.delta_x)**2 + (self.delta_y)**2 + (self.delta_z)**2, 5)), 6)

    @property
    def midpoint(self) -> Coordinate:
        '''
        :returns the midpoint coordinate of an edge
        '''
        return Coordinate(round((self.start.x + self.end.x)/2, 6), round((self.start.y + self.end.y)/2, 6))

    def anticlockwise_successors(self, edge_list) -> list[Edge]:
        '''
        :returns: a list of the right most edge to the left most edge relative to self
        '''
        ### Step 1: Put inverted OG edge into the edge list
        inverted_og = self.reversed()
        include_inverted_og = True
        if inverted_og not in edge_list:
            include_inverted_og = False
            edge_list.append(inverted_og)

        ### Step 2: Get all edges in there correct Quadrant
        quadrants = [[], [], [], []]
        for edge in edge_list:

            if edge.delta_x >= 0 and edge.delta_y >= 0:
                quadrants[0].append(edge)

            elif edge.delta_x > 0 and edge.delta_y < 0:
                quadrants[3].append(edge)

            elif edge.delta_x < 0 and edge.delta_y > 0:
                quadrants[1].append(edge)

            elif edge.delta_x <= 0 and edge.delta_y <= 0:
                quadrants[2].append(edge)

            else:
                raise ValueError("WTF?!??!")

        ### Step 3: Get each Quadrant list in order
        sorted_quadrants = [[], [], [], []]
        for ind, quadrant in enumerate(quadrants):
            sorted_quadrants[ind] = sorted(quadrant, key= lambda x: x.gradient)

        ### Step 4: Split the list with the inverted OG into 'prelist' and 'post list'
        if inverted_og.delta_x >= 0 and inverted_og.delta_y >= 0:
            og_index = sorted_quadrants[0].index(inverted_og)

            pre_list = sorted_quadrants[0][:og_index]
            post_list = sorted_quadrants[0][og_index+1:]

            quadrant_order = [1, 2, 3]

        elif inverted_og.delta_x > 0 and inverted_og.delta_y < 0:
            og_index = sorted_quadrants[3].index(inverted_og)

            pre_list = sorted_quadrants[3][:og_index]
            post_list = sorted_quadrants[3][og_index+1:]

            quadrant_order = [0, 1, 2]

        elif inverted_og.delta_x < 0 and inverted_og.delta_y > 0:
            og_index = sorted_quadrants[1].index(inverted_og)

            pre_list = sorted_quadrants[1][:og_index]
            post_list = sorted_quadrants[1][og_index+1:]

            quadrant_order = [2, 3, 0]

        elif inverted_og.delta_x <= 0 and inverted_og.delta_y <= 0:
            og_index = sorted_quadrants[2].index(inverted_og)

            pre_list = sorted_quadrants[2][:og_index]
            post_list = sorted_quadrants[2][og_index+1:]

            quadrant_order = [3, 0, 1]

        if include_inverted_og:
            pre_list.append(inverted_og)


        ### Step 5: Creating the final list :)
        final_list = []

        final_list.extend(post_list)
        final_list.extend(sorted_quadrants[quadrant_order[0]])
        final_list.extend(sorted_quadrants[quadrant_order[1]])
        final_list.extend(sorted_quadrants[quadrant_order[2]])
        final_list.extend(pre_list)

        debug = False
        if debug:
            print()
            print(edge_list, 'initial edge_list', len(edge_list), 'edges')
            print()
            print(quadrants[0], 'Q1')
            print(quadrants[1], 'Q2')
            print(quadrants[2], 'Q3')
            print(quadrants[3], 'Q4')
            print()
            print(post_list, 'post list')
            print(sorted_quadrants[quadrant_order[0]], f'quadrant{quadrant_order[0]}')
            print(sorted_quadrants[quadrant_order[1]], f'quadrant{quadrant_order[1]}')
            print(sorted_quadrants[quadrant_order[2]], f'quadrant{quadrant_order[2]}')
            print()



        return final_list

    def reversed(self) -> Edge:
        '''
        return the reversed Edge
        '''
        return Edge(self.end, self.start, self.thickness)

    @classmethod
    def visualize_edges(cls, edges: list[Edge], hide_turtle=True, speed=0, x_offset=20, y_offset=20, line_width=3, multiplier=8, terminate=False) -> None:
        '''
        visualizes the sequence of edges in a list 
        '''
        skk = turtle.Turtle()
        turtle.width(line_width)
        turtle.speed(speed)
        if hide_turtle:
            turtle.hideturtle()
        else:
            turtle.showturtle()

        colors = ['black', 'red', 'blue', 'light blue', 'green', 'brown', 'dark green', 'orange', 'gray', 'indigo']
        color = random.choice(colors)
        while color in Graph.used_colors:
            color = random.choice(colors)

        turtle.pencolor(color)

        turtle.up()

        turtle.setpos((edges[0].start.x - x_offset) * multiplier, (edges[0].start.y - y_offset) * multiplier)
        for edge in edges:
            turtle.down()
            turtle.setpos((edge.start.x - x_offset) * multiplier, (edge.start.y - y_offset) * multiplier)
            turtle.setpos((edge.end.x - x_offset) * multiplier, (edge.end.y - y_offset) * multiplier)
            turtle.up()

        Graph.used_colors.add(color)
        if len(Graph.used_colors) == len(colors):
            print('\n\n!!!!!!!!!! COLORS RESET !!!!!!!!!!!!!!!!!\n\n')
            Graph.used_colors = set()

        if terminate:
            turtle.done()

    def intersection(self, other: Edge) -> Coordinate:
        '''
        :param self: the first edge 
        :param other: the second edge 

        returns the coordinate of intersection between self and other
        '''
        if type(self.gradient) != Infinity() and type(other) != Infinity():
            if (prev_gradient - gradient) != 0:
                x = round((y_intercept - prev_y_intercept) / (prev_gradient - gradient), 3)
                y = round(gradient * x + y_intercept, 3)
                return Coordinate(x, y)
            else:
                return None

        elif type(self.gradient) == Infinity() and type(other) != Infinity():
            x = self.start.x
            y = round(other.gradient*x + other.y_intercept, 3)
            return Coordinate(x, y)

        elif type(self.gradient) != Infinity() and type(other) == Infinity():
            x = other.start.x
            y = round(self.gradient*x + self.y_intercept, 3)
            return Coordinate(x, y)

        elif type(self.gradient) == Infinity() and type(other) == Infinity():
            # it's either infinite intersections if same line or no intersection if different lines
            return None

    def is_same_direction(self, other: Edge) -> bool:
        '''
        :param self: edge1
        :param other: edge2

        :returns: whether the two edges are on point to the same direction or not
        '''
        if self.gradient != other.gradient:
            return False

        if self.gradient != Infinity():
            return (self.delta_x > 0 and other.delta_x > 0) or (self.delta_x < 0 and other.delta_x < 0)

        else:
            return (self.delta_y > 0 and other.delta_y > 0) or (self.delta_y < 0 and other.delta_y < 0)
           
    def pointing_away_from_coord(self, coordinate: Coordinate) -> bool:
        '''
        if the coordinate exists before the receprical line intersecting the midpoint of the edge, 
        it's behind the edge if it exists after the receprical line intersecting the midpoint of the edge it's after

        :returns: whether the direction of the edge is pointing away from the coordinate or towards the coordinate behind it
        '''
        l1 = Edge(self.start, coordinate, None).absolute_length
        l2 = Edge(self.end, coordinate, None).absolute_length

        if l1 < l2:
            return True

        elif l1 > l2:
            return False

        else:
            raise ValueError("coordinate exactly on top in between, couldn't determine whether it's pointing away or to")

    def __eq__(self, other) -> bool:
        '''
        Equality Definition
        '''
        return (self.start == other.start and self.end == other.end and self.thickness == other.thickness)

    def __hash__(self):
        '''
        hashing the Edge object
        '''
        return hash(str(self))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        '''
        string representation of the edge
        '''
        return f"{self.start} -> {self.end} : thickness{self.thickness}"

class CycloidalDrive:
    '''
    class to generate, visualize, export the cycloidal drive shape itself and other parts needed to create a cyloidal gear reduction system
    '''
    MINIMUM_WALL_LENGTH = 3  # mm
    def __init__(self, pin_radius: float, pin_base_radius: float, pin_count: int, contraction: int, 
                 pin_cycloid_tolerance: float, roller_pin_count: int, roller_pin_radius: float,  roller_pin_tolerance: float,
                 bearing_outer: float, bearing_inner: float, bearing_outer_tolerance: float, bearing_inner_tolerance: float,  
                 layer_thickness: float):
        """
        constructor for cycloidal instance

        :param pin_radius: radius of pins the cycloid will revolve around
        :param pin_base_radius: how far each pin is from center of pin base circle
        :param pin_count: number of pins equally distributed along the circumference of the pin base circle
        :param contraction:
        """
        ### User defined inputs
        self.pin_radius = pin_radius
        self.pin_base_radius = pin_base_radius
        self.pin_count = pin_count

        self.contraction = contraction
        self.pin_cycloid_tolerance = pin_cycloid_tolerance

        self.roller_pin_count = roller_pin_count
        self.roller_pin_radius = roller_pin_radius
        self.roller_pin_tolerance = roller_pin_tolerance

        self.bearing_outer = bearing_outer  # diameter
        self.bearing_inner = bearing_inner  # diamter
        self.bearing_outer_tolerance = bearing_outer_tolerance  #diameter
        self.bearing_inner_tolerance = bearing_inner_tolerance  #diameter

        self.layer_thickness = layer_thickness
        
        ### automatically defined variables
        self.rolling_circle_radius = round(self.pin_base_radius/self.pin_count, 6)
        self.reduction_ratio = pin_count - 1
        self.cycloid_base_circle_radius = round(self.rolling_circle_radius*self.reduction_ratio, 6)
        self.eccentricity = Edge(self.pin_base_center, self.cycloid_center, None).absolute_length
        self.roller_pin_hole_radius = round((self.roller_pin_radius*2 + 2*self.eccentricity)/2, 5)

    @property
    def pin_base_center(self) -> Coordinate:
        '''
        :return: coordinate of pin base center
        '''
        # x_coord = (self.pin_base_radius + self.pin_cycloid_tolerance)*cos(90) + self.rolling_circle_radius - self.contraction
        # y_coord = (self.pin_base_radius + self.pin_cycloid_tolerance)*sin(0)
        # which is
        x_coord = self.rolling_circle_radius - self.contraction
        y_coord = 0

        return Coordinate(x_coord, y_coord, 0)

    @property
    def pin_coordinates(self) ->list[Coordinate]:
        '''
        :return : list of coorinates of center of pins 
        '''
        pin_coordinates = []
        for pin_angle in range(0, 361, 360//self.pin_count):
            x_coord = (self.pin_base_radius + self.pin_cycloid_tolerance)*cos(pin_angle) + self.rolling_circle_radius - self.contraction
            y_coord = (self.pin_base_radius + self.pin_cycloid_tolerance)*sin(pin_angle)

            pin_coordinates.append(Coordinate(x_coord, y_coord, 0))

        return pin_coordinates

    @property
    def cycloid_center(self) -> Coordinate:
        '''

        '''
        return Coordinate(0, 0, 0)

    @property
    def cycloid_coordinates(self) -> list[Coordinate]:
        '''
        :return: list of coordinates of the cycloid
        '''
        self.rolling_circle_coords = []
        self.rolling_circle_lines = []
        coordinates = []
        for angle in range(0, 361):

            rolling_cir_x = (self.cycloid_base_circle_radius + self.rolling_circle_radius) * cos(angle)
            rolling_cir_y = (self.cycloid_base_circle_radius + self.rolling_circle_radius) * sin(angle)

            rolling_cir_coord = Coordinate(rolling_cir_x, rolling_cir_y, 0)

            self.rolling_circle_coords.append(rolling_cir_coord)

            end_x = rolling_cir_x + (self.rolling_circle_radius - self.contraction) * cos(self.pin_count*angle)
            end_y = rolling_cir_y + (self.rolling_circle_radius - self.contraction) * sin(self.pin_count*angle)

            rolling_circle_line_end = Coordinate(end_x, end_y, 0)

            self.rolling_circle_lines.append(Edge(rolling_cir_coord, rolling_circle_line_end, 0))

            coordinates.append(rolling_circle_line_end)

        poly_line = shapely.LinearRing(deepcopy(coordinates))
        poly_line_offset = poly_line.parallel_offset(self.pin_radius, side='left', resolution=16, join_style=1, mitre_limit=1)
        
        offseted_coordinates = poly_line_offset.coords
        offseted_coordinates = [Coordinate(coord[0], coord[1], 0) for coord in offseted_coordinates]

        return offseted_coordinates

    @property
    def roller_pin_holes_centers(self) -> list[Coordinate]:
        '''
        :return: list of center of the roller pin circles
        '''
        # Find the circumference which the roller pin hole circles will lie on the cycloid itself
        # aka distance from center of cycloid to center of each roller pin hole circle
        # it will be the midpoint of distance {roller_pin_hole_possible_area} 
        # between ( bearing_outer+bearing_outer_tolerance )
        # and closes point from cycloid coordinates to center of cycloid
        cycloid_center = self.cycloid_center
        cycloid_coordinates = self.cycloid_coordinates

        min_distance = Edge(cycloid_center, cycloid_coordinates[0], None).absolute_length
        for coord in cycloid_coordinates:
            new_distance = Edge(cycloid_center, coord, None).absolute_length
            if new_distance < min_distance:
                min_distance = new_distance

        # the roller_pin_hole_possible_area must be greater than 
        # {2* ( roller_pin_hole_radius + (least possible 3dprinted wall thickness) )}
        roller_pin_hole_available_length = min_distance - round((self.bearing_outer + self.bearing_outer_tolerance)/2, 5) 
        roller_pin_hole_needed_length = round(2*self.roller_pin_hole_radius, 5) + 2*CycloidalDrive.MINIMUM_WALL_LENGTH
        if roller_pin_hole_available_length < roller_pin_hole_needed_length:
            raise ValueError('length needed to for roller pin holes is smaller than available length in cycloid\nMust make cycloid bigger or make bearing outer diamter smaller')

        roller_pin_holes_centers_at_circle_radius = round((self.bearing_outer + self.bearing_outer_tolerance)/2, 5) + round(roller_pin_hole_available_length/2, 5)

        # find evenly distributed angles to put the center in according to self.roller_pin_count
        roller_pin_hole_centers = []
        for angle in range(0, 361, 360//self.roller_pin_count):
            x = roller_pin_holes_centers_at_circle_radius*cos(angle)
            y = roller_pin_holes_centers_at_circle_radius*sin(angle)

            roller_pin_hole_centers.append(Coordinate(x, y, 0))

        return roller_pin_hole_centers
       
    @property
    def roller_pin_centers(self) -> list[Coordinate]:
        '''
        return list of coordinates of center of the roller pin in the 3rd layer of roller pin output shaft
        '''
        roller_pin_centers = self.roller_pin_holes_centers
        for coord in roller_pin_centers:
            coord.x += self.eccentricity

        return roller_pin_centers

    def export_cyloid_dxf(self, version, file_name) -> None:
        '''
        exports dxf file for cycloid
        '''
        doc = ezdxf.new(version)
        doc.units = ezdxf.units.MM

        msp = doc.modelspace()
        verts = list(self.cycloid_coordinates)
        # verts = self.coordinates
        prev_coord = verts[0]
        for vert in verts:
            msp.add_line(prev_coord, vert)
            prev_coord = vert

        doc.saveas(file_name)

    def export_pins_dxf(self, version, file_name) -> None:
        '''
        exports pins as a dxf file
        '''
        doc = ezdxf.new(version)
        doc.units = ezdxf.units.MM

        msp = doc.modelspace()

        coords_list = self.pin_coordinates
        for coord in coords_list:
            msp.add_circle(coord, self.pin_radius)

        doc.saveas(file_name)

    def export_everything(self, version, file_name) -> None:
        '''
        exports cycloid and pins
        '''
        doc = ezdxf.new(version)
        doc.units = ezdxf.units.MM

        msp = doc.modelspace()

        ### Layer 1: Base Pins
        # Center hole for ball bearing that will encompass the eccentric shaft
        center = self.pin_base_center
        radius = round((self.bearing_outer + self.bearing_outer_tolerance)/2, 5)
        msp.add_circle(center, radius)

        # Eccentric Shaft
        center = self.pin_base_center
        radius = round((self.bearing_inner + self.bearing_inner_tolerance)/2, 5)
        msp.add_circle(center, radius)

        # The pins
        coords_list = self.pin_coordinates
        for coord in coords_list:
            #NOTE Although the pins are part of the pin base object, its existence is in the same height as the layer2: the cycloid
            coord.z += self.layer_thickness
            msp.add_circle(coord, self.pin_radius)
        
        ### Layer 2: The Cycloid
        # Center hole for ball bearing that will encompass the eccentric shaft
        center = self.cycloid_center
        center.z += self.layer_thickness
        radius = round((self.bearing_outer + self.bearing_outer_tolerance)/2, 5)
        msp.add_circle(center, radius)

        # Eccentric Shaft
        center = self.cycloid_center
        center.z += self.layer_thickness
        radius = round((self.bearing_inner + self.bearing_inner_tolerance)/2, 5)
        msp.add_circle(center, radius)

        # The Cycloid itself
        verts = self.cycloid_coordinates
        for vert in verts:
            vert.z += self.layer_thickness  # putting it one layer above
        prev_coord = verts[0]
        for vert in verts:
            msp.add_line(prev_coord, vert)
            prev_coord = vert

        # Roller Pin Holes
        for center in self.roller_pin_holes_centers:
            center.z += self.layer_thickness
            msp.add_circle(center, self.roller_pin_hole_radius)

        ### Layer 3: Roller Pin Output Shaft
        # Center hole for ball bearing that will encompass the eccentric shaft
        center = self.pin_base_center
        center.z += self.layer_thickness*2
        radius = round((self.bearing_outer + self.bearing_outer_tolerance)/2, 5)
        msp.add_circle(center, radius)

        # Eccentric Shaft
        center = self.pin_base_center
        center.z += self.layer_thickness*2
        radius = round((self.bearing_inner + self.bearing_inner_tolerance)/2, 5)
        msp.add_circle(center, radius)

        # The Roller Pins
        radius = self.roller_pin_radius + self.roller_pin_tolerance
        for center in self.roller_pin_centers:
            center.z += self.layer_thickness*2
            msp.add_circle(center, radius)

        # Output Screw holes 
        #TODO:

        doc.saveas(file_name)

if __name__ == '__main__':

    ### User defined inputs in mm
    # Layer 1: Pin Base
    pin_count = 10  # reduction ratio = pin_count -1
    pin_radius = 2.5
    pin_base_radius = 50  # The circumference at which the pins are placed on

    # Layer 2: Cycloid
    contraction = 2
    pin_cycloid_tolerance = 0.5

    # Layer 3: Roller Pin Output Shaft
    roller_pin_count = 4
    roller_pin_radius = 5
    roller_pin_tolerance = -0.2  # radius-wise, will be added to roller pins NOT roller pin holes

    # Layer 4: Cover

    # Inter-Layer Object: Eccentric Shaft
    bearing_outer = 32  # diameter
    bearing_inner = 20  # diameter
    bearing_outer_tolerance = 0.2  # Less Plastic by making extra diameter for center hole of ball bearing in each layer
    bearing_inner_tolerance = 0.15  # Extra plastic to the diameter of the eccentric shaft

    # Others
    layer_thickness = 7  # thickness of each layer

    cycloid = CycloidalDrive(pin_radius, pin_base_radius, pin_count, contraction, pin_cycloid_tolerance,
                             roller_pin_count, roller_pin_radius, roller_pin_tolerance,
                             bearing_outer, bearing_inner, bearing_outer_tolerance, bearing_inner_tolerance,
                             layer_thickness)

    cycloid.export_everything('R2010', '../cycloidal_drive_sketches.dxf')



            






