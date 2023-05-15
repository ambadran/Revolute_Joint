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

        elif self.z:
            if index == 2:
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
        :return delta y
        '''
        return round(self.end.y - self.start.y, 5)
    
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
        return round(math.sqrt(round((self.delta_x)**2 + (self.delta_y)**2, 5)), 6)

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


class Graph:
    '''
    Graph Data structure
    
    It is made out of vertices (Coordinates) 
    and Edges which represents connections between vertices

    #NOTE: each connection is made out of TWO edges assigned to each vertix
        one edge assigned to vertix1 which is 1->2
        second edge assigned to vertix2 which is 2->1

    The Underlying Datastructures:
    1- Dictionary of key vertices(Coordinate) and value list of Edges GOING OUT from the vertex key

    '''
    # class variables
    used_colors = set()
    TINY_EDGE_OFFSET = 0.2
    CURVE_THRESHOLD_LENGTH = 0.8

    # Debuggin switches
    DEBUG_APPLY_OFFSET = False
    DEBUG_FILTER_TINY_EDGES = False
    DEBUG_ORDERED_EDGES = False
    DEBUG_TO_SINGLY_LINKEDLIST = False

    def __init__(self, vertices: list[Coordinate] = []):

        # Dictionary to relate each vertices to its edges
        self.vertex_edges: dict[Coordinate: list[Edge]] = {vertex: [] for vertex in vertices}

        # Dictionary to relate each vertices to the vertices it is attached to in the other end
        self.vertex_vertices: dict[Coordinate: list[Coordinate]] = {vertex: [] for vertex in vertices}

        # Dictionary to relate each vertices to the ComponentPad it's on (ONLY if there is one)
        self.vertex_componentpad: dict[Coordinate: Optional[ComponentPad]] = {vertex: None for vertex in vertices}

    @property
    def vertex_count(self) -> int:
        '''
        return number of vertices in this graph
        '''
        return len(self.vertex_edges)

    @property
    def edge_count(self) -> int:
        '''
        return number of edges in this graph
        '''
        return sum(len(edges) for edges in self.vertex_edges.values())

    def add_vertex(self, vertex: Coordinate) -> None:
        '''
        adds the new vertex to the underlying data structures of the graph

        :param vertex: the new vertex to be added to our graph
        '''
        if vertex not in self.vertex_edges:
            self.vertex_edges[vertex] = []
            self.vertex_vertices[vertex] = []

    def add_edge(self, edge: Edge) -> None:
        '''
        adds the new edge to the underlying data structures of the graph

        :param edge: the new edge to be added to the graph
        '''
        # checking if it's an edge
        if abs(edge.delta_x) == 0 and abs(edge.delta_y) == 0:
            raise ValueError("not an edge, it's a point")

        # Adding only if it's not there, ensure not duplicates
        if edge not in self.vertex_edges[edge.start]:
            self.vertex_edges[edge.start].append(edge)
        if edge.reversed() not in self.vertex_edges[edge.end]:
            self.vertex_edges[edge.end].append(edge.reversed())

        if edge.end not in self.vertex_vertices[edge.start]:
            self.vertex_vertices[edge.start].append(edge.end)
        if edge.start not in self.vertex_vertices[edge.end]:
            self.vertex_vertices[edge.end].append(edge.start)

    def ordered_edges(self, ignore_non_tree=False) -> list[Ezdge]:
        '''
        DP algorithm to order edges,
        #NOTE: HIGHLY DEPENDENT ON 'Edge.anticlockwise_successors()'

        :return: an ordered list of how to traverse the trace in a continual manner
        '''
        ordered_edges = []

        visited = set()
        next_edge = list(self.vertex_edges.values())[0][0]
        while len(visited) < self.edge_count:

            # print()
            # print(next_edge, 'start')
            ordered_edges.append(next_edge)
            
            next_v = next_edge.end

            if len(self.vertex_edges[next_v]) == 1:
                # dead end must return
                next_edge = self.vertex_edges[next_v][0]
                visited.add(next_edge)
                # print(next_edge, 'DEADEND')

            else:
                successors = next_edge.anticlockwise_successors(self.vertex_edges[next_v])
                # print(successors, 'successors')
                for edge in successors:
                    # print('potential edge', edge, edge not in visited, edge.reversed() != next_edge)
                    if edge not in visited and edge.reversed() != next_edge:
                        # print('yes')
                        next_edge = edge
                        visited.add(next_edge)
                        break

                    else:
                        pass
                        # print('no')
                else:
                    ### This is now an non-tree graph.
                    ### Backtracking !!!

                    # Edge.visualize_edges(ordered_edges, speed=1, terminate=True)
                    if ignore_non_tree:
                        # print(len(visited), self.edge_count, 'lskjdflksdjflksjdf')
                        return ordered_edges
                    else:
                        raise ValueError("Use Graph.ordered_edges_non_tree() for non-tree graphs")
                
        # Edge.visualize_edges(ordered_edges, speed=1)
        return ordered_edges

    @property
    def ordered_edges_non_tree(self) -> list[Edge]:
        '''
        DP algorithm to order edges,
        #NOTE: HIGHLY DEPENDENT ON 'Edge.anticlockwise_successors()'

        :return: an ordered list of how to traverse the trace in a continual manner
        '''
        ordered_edges = []

        visited = set()
        next_edge = list(self.vertex_edges.values())[0][0]
        first_edge = deepcopy(next_edge)
        backtracking_frontier = Queue()
        while len(visited) < self.edge_count:

            # print()
            # print(next_edge, 'start')
            ordered_edges.append(next_edge)
            if next_edge not in visited:
                backtracking_frontier.push(next_edge)
            
            next_v = next_edge.end

            if len(self.vertex_edges[next_v]) == 1:
                # dead end must return
                next_edge = self.vertex_edges[next_v][0]
                visited.add(next_edge)
                # print(next_edge, 'DEADEND')

            else:
                successors = next_edge.anticlockwise_successors(self.vertex_edges[next_v])
                # print(successors, 'successors')
                for edge in successors:
                    # print('potential edge', edge, edge not in visited, edge.reversed() != next_edge)
                    if edge not in visited and edge.reversed() != next_edge:
                        # print('yes')
                        next_edge = edge
                        visited.add(next_edge)
                        break

                    # else:
                        # pass
                        # print('no')
                else:
                    ### This is now an non-tree graph.
                    ### Backtracking !!!
                    if backtracking_frontier.empty:
                        # print(next_edge, 'Frontier Emptied!!!!')
                        if next_edge == first_edge:
                            next_edge = next_edge.reversed()
                            # print(f'Going the other side :) {next_edge}')
                        else:
                            raise ValueError("No Solution")

                    else:
                        next_edge = backtracking_frontier.pop()
                        # print(f'Backtracking to {next_edge}')

        if Graph.DEBUG_ORDERED_EDGES:
            Edge.visualize_edges(ordered_edges, speed=1, hide_turtle=False, line_width=2, terminate=True)

        return ordered_edges

    @property
    def ordered_edges_non_tree_diff_lists(self) -> list[list[Edge]]:
        '''
        exactly like ordered_edges_non_tree but return list of list of different continious edge orders
        '''
        #TODO:
        pass

    def __contains__(self, vertex: Coordinate) -> bool:
        '''
        checks if given vertex is already added to the graph or not

        :param vertex: vertex to check if it's in the graph or not
        :return: whether given vertex is in the graph or not
        '''
        if type(vertex) != Coordinate:
            raise ValueError("can only use 'in' operator '__contains__' for vertex objects (Coordinate objects)")

        return vertex in list(self.vertex_vertices.keys())

    def visualize(self, hide_turtle=True, x_offset=20, y_offset=20, speed = 0, line_width=3, multiplier=8, terminate=False) -> None:
        '''
        Uses Python Turtle graphs to draw the graph
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

        for edges in self.vertex_edges.values():
            if edges != []:
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

    @classmethod
    def visualize_graphs(cls, graph_list: list[Graph]) -> None:
        '''
        calls self.visualize for a bunch of graphs
        
        :param graph_list: list of graphs to visualize
        '''
        for graph in graph_list[:-1]:
            graph.visualize()
        graph_list[-1].visualize(terminate=True)

    def __str__(self) -> str:
        '''
        string representation of the graph
        '''
        desc = ""
        for vertex, vertices in self.vertex_vertices.items():
            desc += f"{vertex} -> {[str(vertex) for vertex in vertices]}\n"  # could of just written vertices as it is 
                                                                            # I want str() not repr()
        desc += '\n'

        return desc

    def seperate(self) -> list[Graph]:
        '''
        Uses a DP algorithm to seperate one big graph into list of graphs which contain one continious trace

        :return: list of continious trace graph
        '''
        seperated_graphs = []

        visited = set()
        # print(self)
        
        for vertex in list(self.vertex_edges.keys()):

            if vertex not in visited:
                # print(vertex, 'added new')
                new_graph = Graph([vertex])
                visited.add(vertex)

                for other_vertex in self.vertex_vertices[vertex]:
                    # print(other_vertex, 'added form vertex_vertices')
                    new_graph.add_vertex(other_vertex)
                    visited.add(other_vertex)

                for edge in self.vertex_edges[vertex]:
                    # print(edge, 'added from.vertex_edges')
                    new_graph.add_edge(edge)

                seperated_graphs.append(new_graph)
                # print()

            else:
                # finding the graph that has this vertex
                found = False
                graphs_ind_to_join = []  # if a vertex is found in more than one graph, then join
                for ind, graph in enumerate(seperated_graphs):
                    if vertex in graph:
                        graphs_ind_to_join.append(ind)
                        found = True

                if found == False:
                    raise ValueError("HOW THE HELL??!?!")

                # print(vertex, 'found at graph of ind', graphs_ind_to_join)

                # Create new joined graph from many graphs with same vertex
                if len(graphs_ind_to_join) > 1:
                    # found more than one graph with same vertex, creating joined graph
                    joined_graph = Graph()
                    for graph_ind in graphs_ind_to_join:
                        joined_graph = Graph.join(joined_graph, seperated_graphs[graph_ind])

                    # print(f'Joining graphs of inds: {graphs_ind_to_join}')

                    # Remove the seperate graphs with same vertex and put the newly created joined graph
                    new_seperated_graphs = [joined_graph]
                    wanted_ind = 0
                    for ind, graph in enumerate(seperated_graphs):
                        if ind not in graphs_ind_to_join:
                            new_seperated_graphs.append(graph)

                    seperated_graphs = deepcopy(new_seperated_graphs)

                else:
                    wanted_ind = graphs_ind_to_join[0]

                for other_vertex2 in self.vertex_vertices[vertex]:
                    # print(other_vertex2, 'added form vertex_vertices')
                    visited.add(other_vertex2)
                    seperated_graphs[wanted_ind].add_vertex(other_vertex2)

                for edge in self.vertex_edges[vertex]:
                    # print(edge, 'added from.vertex_edges')
                    seperated_graphs[wanted_ind].add_edge(edge)
                # print()
        
        # Removing duplicate edges
        #TODO
        # for graph_ind, graph in enumerate(seperated_graphs):
        #     for vertex in graph.vertex_edges:
        #         seperated_graphs[graph_ind].vertex_edges[vertex] = list(set(graph.vertex_edges[vertex]))


        return seperated_graphs

    def apply_offsets(self, extra_offset=0, terminate_after=False) -> Graph:
        '''
        for debugging
        # trace_graphs_seperated_unoffseted[0].apply_offsets(terminate_after=True)

        The graph to execute .apply_offsets() to is a graph of continious lines of ZERO thickness,
        This function will create a new graph from the old one with the thickness applied :)

        :return: Graph with thickness applied. The edges are of thickness 0
        '''
        if Graph.DEBUG_APPLY_OFFSET:
            print('\nNEW CALL!!!!!!!!!!!!!!!')

        new_graph = Graph()

        ordered_edges = self.ordered_edges()

        ### PRE-ITERATION: get y=mx+c of edge of ind=-1
        # Getting gradient and y_intercept of last edge in cycle
        last_edge = ordered_edges[-1]
        gradient = last_edge.gradient
        abs_offset = round(last_edge.thickness/2 + extra_offset, 3)
        if last_edge.gradient != Infinity():
            alpha = round(math.atan(last_edge.gradient), 3)
            theta = round(math.pi/2 - alpha, 3)
            y_offset = round(abs_offset / math.sin(theta), 3)

            if last_edge.delta_x > 0:
                prev_y_intercept = round(last_edge.y_intercept - y_offset, 3)

            elif last_edge.delta_x < 0:
                prev_y_intercept = round(last_edge.y_intercept + y_offset, 3)

        else:
            if last_edge.delta_y > 0:
                vertical_line_offset_x = last_edge.start.x - abs_offset

            elif last_edge.delta_y < 0:
                vertical_line_offset_x = last_edge.start.x + abs_offset

        prev_gradient = gradient

        ordered_edges.append(ordered_edges[0])  # to force it to do one more iteration at the end
        for ind, edge in enumerate(ordered_edges):

            ### 1- Getting the linear equation of the offseted line
            # The Gradient is ofcoarse the same as the gradient of the original line since they're parallel
            # to get the y-intercept however I devised the following algorithm :)
            gradient = edge.gradient
            abs_offset = round(edge.thickness/2 + extra_offset, 3)

            if gradient != Infinity():
                alpha = round(math.atan(edge.gradient), 3)
                theta = round(math.pi/2 - alpha, 3)
                y_offset = round(abs_offset / math.sin(theta), 3)

                if edge.delta_x > 0:
                    y_intercept = round(edge.y_intercept - y_offset, 3)

                elif edge.delta_x < 0:
                    y_intercept = round(edge.y_intercept + y_offset, 3)

            else:
                if edge.delta_y > 0:
                    vertical_line_offset_x = edge.start.x + abs_offset

                elif edge.delta_y < 0:
                    vertical_line_offset_x = edge.start.x - abs_offset

            # offseted line equation is now 
            # y = gradient * x + y_intercept    :)

            ### Parrallel opposite direction edges are essentially deadends, point of return
            # They are dealt with differently in the else statement
            if ind == 0:
                # This is because the last item in the list is the first one as I just added it before the main loop
                same_dir_ind = ind - 1  
            else:
                same_dir_ind = ind
            if not(gradient == prev_gradient and not edge.is_same_direction(ordered_edges[same_dir_ind-1])):

                ### 2- Getting intersection between previous edge and current edge to get 'current_vertex'
                # Solving simultaneous equations :)
                if gradient == Infinity() and prev_gradient != Infinity():
                    x = round(vertical_line_offset_x, 3)
                    y = round(prev_gradient * x + prev_y_intercept, 3)

                elif prev_gradient == Infinity() and gradient != Infinity():
                    if ordered_edges[same_dir_ind-1].delta_y > 0:
                        x = edge.start.x + abs_offset 
                    else:
                        x = edge.start.x - abs_offset 

                    y = round(gradient * x + y_intercept, 3)

                elif prev_gradient == Infinity() and gradient == Infinity():
                    if edge.delta_y > 0:
                        x = edge.start.x + abs_offset 
                    else:
                        x = edge.start.x - abs_offset 

                    y = edge.start.y

                elif prev_gradient == 0 and gradient == 0:
                    x = edge.start.x

                    if edge.delta_x > 0:
                        y = edge.start.y - abs_offset 
                    else:
                        y = edge.start.y + abs_offset

                elif prev_gradient == gradient:
                    if ind != 0:
                        x = current_edge.end.x
                        y = current_edge.end.y
                    else:
                        raise ValueError('This case is not implemented yet')
                        pass #TODO:

                else:  # both gradient different and not infinity and not 0
                    x = round((y_intercept - prev_y_intercept) / (prev_gradient - gradient), 3)
                    y = round(gradient * x + y_intercept, 3)


                # Adding the vertex to the graph
                current_vertex = Coordinate(x, y)
                new_graph.add_vertex(current_vertex)

                ### 3- Creating and adding a new edge between previous vertex and current vertex
                if ind != 0:  # ONLY FOR ITERATIONS OF INDEX>1
                    current_edge = Edge(prev_vertex, current_vertex, LASER_BEAM_THICKNESS)
                    new_graph.add_edge(current_edge)

                if Graph.DEBUG_APPLY_OFFSET:
                    print('\nNew Iteration:')
                    print(f'Current edge index: {ind}')
                    print(f'Current edge : {edge}')
                    print()

                    if ind != 0:
                        print(f'Previous offseted Vertex: {prev_vertex}')
                    else:
                        print(f'No prev offseted vertex for first iterations')
                    if prev_gradient != Infinity():
                        print(f'Previous offseted edge linear equations:\ny = {prev_gradient}*x + {prev_y_intercept}')
                    else:
                        print(f'prev_gradient = {prev_gradient}')
                    print()

                    print(f'Current offseted Vertex: {current_vertex}')
                    if gradient != Infinity():
                        print(f'Current offseted edge linear equations: y = {gradient}*x + {y_intercept}')
                    else:
                        print(f'gradient = {gradient}')
                    print()

                    if ind != 0:
                        print(f'Newly Created Edge: {current_edge}')
                    else:
                        print(f'No edge to be created for first iteration')
                    print()

                    print()

                ### Setting variables for next iteration
                if gradient != Infinity():
                    prev_y_intercept = y_intercept
                prev_gradient = gradient
                prev_vertex = current_vertex

            else:  
                # lines are parallel and must join them with semi-circle

                #NOTE: This is a TEMPORARY SOLUTION, will connect them with a straight 
                # line for now, should connect them with semi-circle

                if gradient == Infinity():
                    ### 2&3- Create and add the two connecting vertices of the deadend to the graph
                    # Find intersection b/w:
                    # current offseted edge and inverse line
                    # previous offseted edge and inverse line
                    if edge.delta_y > 0:
                        x1 = edge.start.x - abs_offset 
                        x2 = edge.start.x + abs_offset
                    else:
                        x1 = edge.start.x + abs_offset 
                        x2 = edge.start.x - abs_offset

                    y1 = edge.start.y
                    y2 = y1

                elif gradient == 0:
                    ### 2&3- Create and add the two connecting vertices of the deadend to the graph
                    # Find intersection b/w:
                    # current offseted edge and inverse line
                    # previous offseted edge and inverse line
                    x1 = edge.start.x
                    x2 = x1

                    if edge.delta_x > 0:
                        y1 = edge.start.y + abs_offset 
                        y2 = edge.start.y - abs_offset
                    else:
                        y1 = edge.start.y - abs_offset
                        y2 = edge.start.y + abs_offset

                else:
                    ### 2- Getting invserse linear equation (for next step)
                    inverse_gradient = round((-1)/gradient, 3)
                    inverse_y_intercept = round(edge.start.y - inverse_gradient*edge.start.x, 3)

                    ### 3- Create and add the two connecting vertices of the deadend to the graph
                    # Find intersection b/w:
                    # current offseted edge and inverse line
                    # previous offseted edge and inverse line
                    x1 = round((inverse_y_intercept - prev_y_intercept) / (prev_gradient - inverse_gradient), 3)
                    y1 = round(inverse_gradient*x1 + inverse_y_intercept, 3)

                    x2 = round((inverse_y_intercept - y_intercept) / (gradient - inverse_gradient), 3)
                    y2 = round(inverse_gradient*x2 + inverse_y_intercept, 3)

                vertex1 = Coordinate(x1, y1)
                new_graph.add_vertex(vertex1)

                vertex2 = Coordinate(x2, y2)

                semicircle_vertices = Coordinate.generate_semicircle_coordinates(vertex1, vertex2, edge.delta_x, edge.delta_y)

                for semicircle_vertex in semicircle_vertices:
                    new_graph.add_vertex(semicircle_vertex)

                new_graph.add_vertex(vertex2)

                ### 4- Adding the new edges to the graph
                # b/w:
                # previous vertex and V1
                # V1 and V2
                if ind != 0:
                    edge1 = Edge(prev_vertex, vertex1, LASER_BEAM_THICKNESS)
                    new_graph.add_edge(edge1)

                prev_semicircle_vertex = vertex1
                for ind, semicircle_vertex in enumerate(semicircle_vertices):
                    try:
                        new_graph.add_edge(Edge(prev_semicircle_vertex, semicircle_vertex, LASER_BEAM_THICKNESS))
                    except ValueError:
                        raise ValueError("BIG PROBLEMMMMMM, two identical coordinates generated, maybe the max point may be not")
                    prev_semicircle_vertex = semicircle_vertex

                # edge2 = Edge(vertex1, vertex2, LASER_BEAM_THICKNESS)  # Straight line connection
                edge2 = Edge(prev_semicircle_vertex, vertex2, LASER_BEAM_THICKNESS)
                new_graph.add_edge(edge2)

                if Graph.DEBUG_APPLY_OFFSET:
                    print('\nNew Iteration:')
                    print('PARALLEL EDGE DETECTED!!!')
                    print(f'm=infinity -> {gradient==Infinity()}, m=0 -> {gradient == 0}')
                    print()

                    print(f'Current edge index: {ind}')
                    print(f'Current edge : {edge}')
                    print()

                    # if ind != 0:
                    # in case of first edge is a parallel line, all the circle edges will be added and it will have a high index
                    try:
                        print(f'Previous offseted Vertex: {prev_vertex}')
                    except Exception:
                        print(f'No prev offseted vertex for first iterations')
                    if gradient != Infinity():
                        print(f'Previous offseted edge linear equations:\ny = {prev_gradient}*x + {prev_y_intercept}')
                    print()

                    if gradient != Infinity() and gradient != 0:
                        print(f'Inverse linear equation:\ny = {inverse_gradient}*x + {inverse_y_intercept}')
                    else:
                        print(f'prev_gradient = gradient = {gradient} = {prev_gradient}')
                    print()

                    if gradient != Infinity():
                        print(f'Current offseted edge linear equations: y = {gradient}*x + {y_intercept}')
                    print()

                    print(f'vertex1: {vertex1}')
                    print(f'vertex2: {vertex2}')
                    # if ind != 0:
                    # in case of first edge is a parallel line, all the circle edges will be added and it will have a high index
                    try:
                        print(f'edge1 (prev to inverse): {edge1}')
                    except Exception:
                        print('No edge1 for first iteration')
                    # print(f'edge2 (inverse to current): {edge2}')  # it's replaced with a semicircle now
                    print()

                    print()
               
                ### 5- Setting previous variable for next iteration
                prev_vertex = vertex2
                prev_gradient = gradient
                if gradient != Infinity():
                    prev_y_intercept = y_intercept

        if Graph.DEBUG_APPLY_OFFSET:
            Edge.visualize_edges(self.ordered_edges(), hide_turtle=False, x_offset=25, y_offset=25, multiplier=10, speed=0)
            new_graph.visualize(speed=0, line_width=1, x_offset=25, y_offset=25, multiplier = 10, terminate=terminate_after)

        return new_graph

    def to_coordinate(self) -> list[Coordinate]:
        '''

        '''
        return []

    def remove_vertex(self, vertex: Coordinate) -> None:
        '''
        Specific sequence to undo add_vertex and resolve all the edges connected to it

        :param vertex: vertex to be removed from the graph
        '''
        if vertex not in self.vertex_vertices:
            raise ValueError("vertex is already not there!")

        # Case 1: vertex is a deadline
        if (len(self.vertex_vertices[vertex]) == 1):
            # Step 1: Remove the key entirely
            previous_vertex = self.vertex_vertices.pop(vertex)[0]
            self.vertex_edges.pop(vertex)

            # Step 2: Remove the vertex and edge values from other key vertices
            self.vertex_vertices[previous_vertex].remove(vertex)

            # Step 3: Remove the edges in the previous vertex that has anything to do with the to be deleted vertex
            edges_to_be_removed = []
            for edge in self.vertex_edges[previous_vertex]:
                if edge.start == vertex or edge.end == vertex:
                    edges_to_be_removed.append(edge)

            for edge in edges_to_be_removed:
                self.vertex_edges[previous_vertex].remove(edge)

        # Case 2: vertex is a link
        else:
            raise ValueError('not implemented yet')

    def remove_edge(self, edge: Edge) -> None:
        '''
        #TODO: THIS FUNCTION CAN ONLY BE APPLIED BEFORE EXECUTING .seperate()
        :param edge: edge to be removed from graph
        '''
        if edge.start not in self.vertex_vertices or edge.end not in self.vertex_vertices:
            # raise ValueError(f"{edge} is not in graph")
            return None

        if edge.start not in self.vertex_vertices[edge.end] or edge.end not in self.vertex_vertices[edge.start]:
            raise ValueError("Edge not properly implemented in graph")

        if len(self.vertex_vertices[edge.start]) == 1:
            # deadend at starting point of edge
            self.remove_vertex(edge.start)
            return None

        elif len(self.vertex_vertices[edge.end]) == 1:
            # deadend at ending point of edge
            self.remove_vertex(edge.end)
            return None

        # Step 1: identify previous verticies
        prev_vertex_list = deepcopy(self.vertex_vertices[edge.start])
        prev_vertex_list.remove(edge.end)

        # Step 2: identify after vertex
        after_vertex = edge.end

        # Step 3: Delete before vertex if there is only one of them, no other edge is attached to it, it's floating point now
        if len(prev_vertex_list) == 1:
            # prev_vertex = prev_vertex_list[0]  # not needed anywhere
            self.vertex_vertices.pop(edge.start)
            self.vertex_edges.pop(edge.start)

        # Step 4: Replace every existence of edge.start in each prev_vertex with after_vertex
        for prev_vertex in prev_vertex_list:
            # Replacing Vertices
            for ind1, prev_prev_vertex in enumerate(self.vertex_vertices[prev_vertex]):
                if prev_prev_vertex == edge.start:
                    self.vertex_vertices[prev_vertex][ind1] = after_vertex

            # Replacing Edges
            for ind2, prev_prev_edge in enumerate(self.vertex_edges[prev_vertex]):
                if prev_prev_edge.start == edge.start:
                    self.vertex_edges[prev_vertex][ind2].start = after_vertex

                if prev_prev_edge.end == edge.start:
                    self.vertex_edges[prev_vertex][ind2].end = after_vertex

        # Step 5: Replace every existence of edge.start in after_vertex with all the vertices/edges of prev_vertex
        # Deleting Vertices
        if self.vertex_vertices[after_vertex].count(edge.start) <= 1:  # only 0 or 1 count is allowed
            if edge.start in self.vertex_vertices[after_vertex]:  # checking if it's there or not
                self.vertex_vertices[after_vertex].remove(edge.start)
        else:
            raise ValueError('Multiple occurance of a vertex found!')

        # Deleting Edges
        edges_to_be_removed = []
        for prev_edge in self.vertex_edges[after_vertex]:
            if prev_edge.start == edge.start or prev_edge.end == edge.start:
                edges_to_be_removed.append(prev_edge)

        for edge_tobe_del in edges_to_be_removed:
            self.vertex_edges[after_vertex].remove(edge_tobe_del)

        for prev_vertex in prev_vertex_list:
            # Adding Vertices
            if prev_vertex not in self.vertex_vertices[after_vertex]:
                self.vertex_vertices[after_vertex].append(prev_vertex)

            # Adding Edges
            edge1 = Edge(prev_vertex, after_vertex, edge.thickness)
            if edge1 not in self.vertex_edges[after_vertex]:
                self.vertex_edges[after_vertex].append(edge1)

            edge2 = edge1.reversed()
            if edge2 not in self.vertex_edges[after_vertex]:
                self.vertex_edges[after_vertex].append(edge2)

    def filter_tiny_edges(self) -> None:
        '''
        Removes the stupid small infuriating edges that mess up with everything
        Affects self Graph
        '''
        edges_to_be_removed = []
        for vertex in self.vertex_vertices.keys():
            for edge in self.vertex_edges[vertex]:
                if abs(edge.delta_x) <= Graph.TINY_EDGE_OFFSET and abs(edge.delta_y) <= Graph.TINY_EDGE_OFFSET:
                    edges_to_be_removed.append(edge)
        
        for edge in edges_to_be_removed:
            self.remove_edge(edge)

        if Graph.DEBUG_FILTER_TINY_EDGES:
            print("Removed Edges:")
            for edge in edges_to_be_removed:
                print(edge)
            print()

    @classmethod
    def join(cls, *graphs: Graph) -> Graph:
        '''
        joins the input graphs

        :graphs: undefined number of graphs
        :return: one graph joined from the all the graphs given from *graphs attribute
        '''
        # Mode checking
        list_mode = False
        if len(graphs) == 1 and type(graphs[0]) == list:
            list_mode = True

        elif len(set([type(graph) for graph in graphs])) != 1:
            raise ValueError("All Arguments MUST be of type Graph OR a one list argument of all graphs")

        joined_graph = Graph()

        if list_mode:
            graphs = graphs[0]

        for graph in graphs:
            joined_graph.vertex_vertices.update(graph.vertex_vertices)
            joined_graph.vertex_edges.update(graph.vertex_edges)

        return joined_graph



        
#TODO: the ultimate goal is have list of trace variables of datatype graph, each has is a continious trace
# i edited the __str__ func of Graph to display the funcs I defined as single_dir_...
#       I have a graph that has all the vertices(it's a Coordinate) pointing to one or more vertices 
#       without pointing back (as the original functions does)
# I have traces ready in this 'traces' variables
# The ONLY thing left is the order. now they are not ordered, meaning- it's like
                                            # coord2 -> coord1
                                            # coord4 -> coord3
                                            # coord1 -> NOTHING
                                            # coord3 -> coord2
                                    # but i have to order it
                                            # coord4 -> coord3
                                            # coord3 -> coord2
                                            # coord2 -> coord1
                                            # coord1 -> NOTHING

# after that i can easily extract each trace by finding the coord that points to nothing:
        # I know that this is the start of a new trace and the end of a previous trace 

    def to_singly_linkedlist(self, terminate_after=False) -> Node:
        '''
        Converts Graph to singly linked list

        #IMP: ONLY WORKS FOR GRAPHS THAT DOESN'T HAVE BRANCHS
            every vertices point to the next and previous vertex ONLY, creating a loop (or not)

        :return: Head Node of the linked list that has all the values of the graph
        '''
        # Checking if graph can be converted
        for vertex in self.vertex_vertices.keys():
            if len(self.vertex_vertices[vertex]) != 2:
                raise ValueError("every vertices must point to two other vertices only")

            if len(self.vertex_edges[vertex]) != 2:
                raise ValueError("every vertices must point to two other edges only")

        #TODO: didn't test if the vertices list values of the current vertex has in their vertices list the current vertex value

        # Converting!
        visited = set()

        first_vertex = list(self.vertex_vertices.keys())[0]
        visited.add(first_vertex)
        next_node = Node(first_vertex, None)  # the last node, doesn't have any heads

        while True:
            for vertex in self.vertex_vertices[next_node.vertex]:
                if vertex not in visited:
                    next_node = Node(vertex, next_node)
                    visited.add(vertex)
                    break
            else:
                if first_vertex in self.vertex_vertices[next_node.vertex]:
                    # it's a loop
                    # adding the last link which couldn't be added normally as it's in visited
                    next_node = Node(first_vertex, next_node)

                break

        if next_node.last_node.vertex == first_vertex:
            next_node.pre_last_node.parent = next_node  # joining the node to end as it's a loop

        # testing everything is ok, all vertices available in the linkedlist as the graph
        node_vertex_set = set(next_node.to_list())
        graph_vertex_set = set(self.vertex_vertices.keys())
        if node_vertex_set != graph_vertex_set:
            # print(next_node.to_list(), len(next_node.to_list()), len(node_vertex_set))
            # print()
            # print(self.vertex_vertices.keys(), len(self.vertex_vertices), len(graph_vertex_set))
            print(node_vertex_set.difference(graph_vertex_set), 'node - graph')
            print()
            print(graph_vertex_set.difference(node_vertex_set), 'graph - node')

            raise ValueError("HOW THE FUCK DID IT PASS THE FIRST TEST AND NOT PASS THIS ONE?!?!?!")

        if Graph.DEBUG_TO_SINGLY_LINKEDLIST:
            next_node.visualize(terminate=terminate_after)

        return next_node

class Cycloid:
    '''
    class to generate, visualize, export the cycloid shape itself and other parts needed to create a cyloidal gear reduction system
    '''
    def __init__(self, pin_radius: float, pin_base_radius: float, pin_count: int, contraction: int, pin_cycloid_tolerance: int):
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
        # self.roller_pin_count  #TODO:
        # self.roller_pin_radius  #TODO:
        
        ### automatically defined variables
        self.rolling_circle_radius = round(self.pin_base_radius/self.pin_count, 6)
        self.reduction_ratio = pin_count - 1
        self.cycloid_base_circle_radius = round(self.rolling_circle_radius*self.reduction_ratio, 6)
        # self.eccentricity  #TODO:
        # self.cycloid_roller_pin_hole_radius = self.roller_pin_radius + 2 * self.eccentricity  #TODO:
        # self.cycloid_center_roller_pin_distance  #TODO:

    @property
    def coordinates(self) -> list[Coordinate]:
        '''
        :return: list of coordinates of the cycloid
        '''
        self.rolling_circle_coords = []
        self.rolling_circle_lines = []
        coordinates = []
        for angle in range(0, 361):

            rolling_cir_x = (self.cycloid_base_circle_radius + self.rolling_circle_radius) * cos(angle)
            rolling_cir_y = (self.cycloid_base_circle_radius + self.rolling_circle_radius) * sin(angle)

            rolling_cir_coord = Coordinate(rolling_cir_x, rolling_cir_y)

            self.rolling_circle_coords.append(rolling_cir_coord)

            end_x = rolling_cir_x + (self.rolling_circle_radius - self.contraction) * cos(self.pin_count*angle)
            end_y = rolling_cir_y + (self.rolling_circle_radius - self.contraction) * sin(self.pin_count*angle)

            rolling_circle_line_end = Coordinate(end_x, end_y)

            self.rolling_circle_lines.append(Edge(rolling_cir_coord, rolling_circle_line_end, 0))

            coordinates.append(rolling_circle_line_end)

        # coordinates_graph = Graph(coordinates)
        # offseted_coordinates_graph = coordinates_graph.apply_offsets()
        # offseted_coordinates_graph.visualize()

        # return coordinates

        poly_line = shapely.LinearRing(deepcopy(coordinates))
        poly_line_offset = poly_line.parallel_offset(self.pin_radius, side='left', resolution=16, join_style=1, mitre_limit=1)
        
        offseted_coordinates = poly_line_offset.coords

        return offseted_coordinates

    @property
    def pin_coordinates(self) ->list[Coordinate]:
        '''
        :return : list of coorinates of center of pins 
        '''
        pin_coordinates = []
        for pin_angle in range(0, 361, 360//self.pin_count):
            x_coord = (self.pin_base_radius + self.pin_cycloid_tolerance)*cos(pin_angle) + self.rolling_circle_radius - self.contraction
            y_coord = (self.pin_base_radius + self.pin_cycloid_tolerance)*sin(pin_angle)

            pin_coordinates.append(Coordinate(x_coord, y_coord))

        return pin_coordinates

    @property
    def cycloid_base_circle_coordinate(self) -> Coordinate:
        '''

        '''
        pass  #TODO:

    @property
    def pin_base_coordinate(self) -> Coordinate:
        '''

        '''
        pass  #TODO:

    @property
    def roller_pin_coordinates(self) -> list[Coordinate]:
        '''

        '''
        pass  #TODO:

    def export_cyloid_dxf(self, version, file_name) -> None:
        '''
        exports dxf file for cycloid
        '''
        doc = ezdxf.new(version)
        doc.units = ezdxf.units.MM

        msp = doc.modelspace()
        verts = list(self.coordinates)
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
            msp.add_circle(coord, pin_radius)

        doc.saveas(file_name)

    def export_everything(self, version, file_name) -> None:
        '''
        exports cycloid and pins
        '''
        doc = ezdxf.new(version)
        doc.units = ezdxf.units.MM

        msp = doc.modelspace()

        # creating the cycloid
        verts = list(self.coordinates)
        prev_coord = verts[0]
        for vert in verts:
            msp.add_line(prev_coord, vert)
            prev_coord = vert

        # creating the pins
        coords_list = self.pin_coordinates
        for coord in coords_list:
            msp.add_circle(coord, pin_radius)

        doc.saveas(file_name)

if __name__ == '__main__':

    # User defined inputs in mm
    pin_radius = 2.5
    pin_base_radius = 50
    pin_count = 10  # reduction ratio = pin_count -1
    contraction = 2
    pin_cycloid_tolerance = 0.2

    cycloid = Cycloid(pin_radius, pin_base_radius, pin_count, contraction, pin_cycloid_tolerance)

    # cycloid.export_cyloid_dxf('R2010', '../cycloid.dxf')
    # cycloid.export_pins_dxf('R2010', '../pins.dxf')

    cycloid.export_everything('R2010', '../cycloid_and_pins.dxf')



            






