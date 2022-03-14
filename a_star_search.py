import sys
from heapq import heappop, heappush

# A-Star algorithm implementation for finding paths through 2D terrain map.
# This implementation of the A-star algorithm is based on the wikipedia article for A-star.
# source
# https://en.wikipedia.org/wiki/A*_search_algorithm

# Use of a priority queue ADT to select the next best minimum costing waypoint
# (similar to Dijkstra except adding a heuristic distance as well) The heuristic
# distance can be added to the cost to travel to a waypoint to get a total cost of
# visiting that waypoint.

# Total weight of a waypoint is the cost to get there plus the distance remain line of
# sight (LOS). In this case we are going to be using the taxi-cab method instead of calculate euclidean distance.
# We can use the function f(n) = g(n) + h(n) to calculate the best score and choose the waypoint to travel to next

"""A STAR ALGORITHM HIGH LEVEL STEPS

STEP 1: Create data structures
STEP 2: Initialize weight/score maps
STEP 3: process next optimal waypoint in PQ

    STEP 3.1: exit condition for algorithm when goal is achieved
    STEP 3.2: process all valid moves to get neighboring waypoints
    STEP 3.3: for each near waypoint calculate g, h, and f scores

        STEP 3.3.1: calculate g cost: the cost to move from start to waypoint
        STEP 3.3.2: calculate h cost: the cost to move from waypoint to goal
        STEP 3.3.3: calculate h cost: the cost to move from waypoint to goal
        STEP 3.3.4: if g score is the best then update to waypoint values map
            STEP 3.3.4.1: add waypoint to PQ for processing if it's not in there
"""

IMPASSIBLE = 'x'


def neighbors(waypoint, moves, map, world):
    """nearest waypoints

    find valid waypoints that the agent is allowed to move to

    Args
        waypoint: (x,y)
        moves: array of (x,y) moves
        map: 2d array of coordinates
        world: world map of characters/terrain

    Returns
        array of valid (x,y) coords
    """

    # STEP 1: set up a data structure to store nearest valid move in range of moves
    valid_coordinates = []

    # STEP 2: perform each move and if the move is within range of the map
    for move in moves:
        possible_move = (waypoint[0] + move[0],
                         waypoint[1] + move[1])  # conduct a move
        if possible_move in map:  # make sure the possible move is on the map
            # ensure waypoint is passible terrain
            if world[waypoint[1] + move[1]][waypoint[0] + move[0]] != IMPASSIBLE:
                valid_coordinates.append(possible_move)  # add a valid waypoint
    return valid_coordinates


def movement_cost(waypoint, costs, world):
    """calculate cost of movement

    cost to move over a type of terrain

    Args
        waypoint: (x,y)
        costs: map of terrain to cost values
        world: 2d map of terrain types

    Returns
        int value of cost
    """
    x = waypoint[0]  # index 1 of tuple
    y = waypoint[1]  # index 2 of tuple
    terrain_type = world[y][x]  # reverse x, y for proper index
    return costs[terrain_type]  # return int value of terrain cost


def heuristic(current_position, goal):
    """calculate heuristic

    cost to hypothetically move from current position to the goal
    add together x dist and y dist from current postion to goal position in any direction

    Args
        current_position: (x1,y1)
        goal: (x2,y2)

    Returns
        int value of cost
    """
    x1, y1 = current_position
    x2, y2 = goal
    return abs(x1 - x2) + abs(y1 - y2)  # abs value, negative not valid


def build_path(parents_map, waypoint):
    """build a procedure of moves to make to get from start to goal

    backtrack through hashmap of {waypoint: previous position} and build an array of moves in order
    this links positions together by their previous position or parent position

    Args
        previous_map: map of points with
        goal: (x2,y2)

    Returns
        array of moves to travel through the path
    """
    moves_path = []
    while waypoint in parents_map:
        x1 = waypoint[0]
        y1 = waypoint[1]
        waypoint = parents_map[waypoint]  # parent is now waypoint
        x2 = waypoint[0]
        y2 = waypoint[1]
        x = x1 - x2  # x child - x parent
        y = y1 - y2  # y child - y parent
        moves_path.append((x, y))  # add move to path
    return moves_path[::-1]


def a_star_search(world, start, goal, costs, moves, heuristic):
    """find a series of moves to get from the start to goal

    greedy search smartly for the most efficient path to the goal

    Args
        world: 2D plain of coordinates
        start:(x,y) start waypoint
        goal: (x,y) end waypoint
        costs: hash map of costs per terrain item
        moves: array of (x,y) offsets

    Returns
        array of valid moves that follow the most efficient path
    """

    # STEP 1: Create data structures
    waypoint_priority_queue = []  # set of waypoints to process
    heappush(waypoint_priority_queue, (start))  # min heap DS or PQ ADT
    parents = {}  # parent relationship
    g_scores = {}  # score from start -> waypoint
    f_scores = {}  # score start -> waypoint -> end

    # STEP 2: Initialize weight/score maps
    map = []
    for y in range(len(world)):  # y: indices of outer array
        for x in range(len(world[0])):  # x: indices of inner array
            g_scores[(x, y)] = sys.maxsize  # init g score
            f_scores[(x, y)] = sys.maxsize  # init f score
            map.append((x, y))  # init grid of world
    g_scores[start] = 0  # init starting g value -> 0
    # f_score[start] = heuristic(start)

    # STEP 3: process next optimal waypoint in PQ
    while waypoint_priority_queue:
        # process next optimal waypoint
        current_waypoint = heappop(waypoint_priority_queue)

        # STEP 3.1: exit condition for algorithm when goal is achieved
        if current_waypoint == goal:
            return build_path(parents, current_waypoint)

        # STEP 3.2: process all valid moves to get neighboring waypoints
        neighboring_waypoints = neighbors(current_waypoint, moves, map, world)

        # STEP 3.3: for each near waypoint calculate g, h, and f scores
        for waypoint in neighboring_waypoints:
            # STEP 3.3.1: calculate g cost: the cost to move from start to waypoint
            cost_g = g_scores[current_waypoint] + \
                movement_cost(waypoint, costs, world)
            # STEP 3.3.2: calculate h cost: the cost to move from waypoint to goal
            cost_h = heuristic(waypoint, goal)
            # STEP 3.3.3: calculate f cost: total cost f(x) = g(x) + h(x)
            cost_f = cost_g + cost_h

            # STEP 3.3.4: if g score is the best then update to waypoint values map
            # if the calculated g score is the best
            if cost_g < g_scores[waypoint]:
                # {(waypoint): (parent)}
                parents[waypoint] = current_waypoint
                g_scores[waypoint] = cost_g  # {(point): g value}
                f_scores[waypoint] = cost_f  # {(point): f value}

                # STEP 3.3.4.1: add waypoint to PQ for processing if it's not in there
                if waypoint not in waypoint_priority_queue:
                    heappush(waypoint_priority_queue, waypoint)


def pretty_print_solution(world, path, start):
    vectors = {
        (0, -1): "^",
        (1, 0): ">",
        (0, 1): "v",
        (-1, 0): "<"
    }

    # initialize x and y values to start position
    x = start[0]
    y = start[1]

    # set start arrow based on first move
    for vector in vectors:
        if vector == path[1]:
            world[y][x] = vectors[vector]

    # find and replace characters in the map
    last_move = len(path)-1
    count = 0
    for move in path:
        x = x + move[0]
        y = y + move[1]
        if count == last_move:
            world[y][x] = "G"
        else:
            world[y][x] = vectors[move]
        count += 1

    # pretty print the array
    for y in world:
        print(*y, sep=' ')


# tests
cardinal_moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
costs = {'.': 1, '*': 3, '#': 5, '~': 7}
test_world = [
    ['.', '*', '*', '*', '*', '*', '*'],
    ['.', '*', '*', '*', '*', '*', '*'],
    ['.', '*', '*', '*', '*', '*', '*'],
    ['.', '.', '.', '.', '.', '.', '.'],
    ['*', '*', '*', '*', '*', '*', '.'],
    ['*', '*', '*', '*', '*', '*', '.'],
    ['*', '*', '*', '*', '*', '*', '.'],
]

test_path = a_star_search(test_world, (0, 0), (6, 6),
                          costs, cardinal_moves, heuristic)

expected = [(0, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 0),
            (1, 0), (1, 0), (1, 0), (0, 1), (0, 1), (0, 1)]

if expected == test_path:
    print("\n...test passed")
else:
    print("\n...failed, expected:")
    print(expected)
    print("got: ")
print(test_path)

full_world = [
    ['.', '.', '.', '.', '.', '*', '*', '*', '*', '*', '*', '*', '*', '*',
        '*', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '*', '*', '*', '*', '*', '*', '*',
     '*', '*', '.', '.', 'x', 'x', 'x', 'x', 'x', 'x', 'x', '.', '.'],
    ['.', '.', '.', '.', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*',
     '*', '*', '*', 'x', 'x', 'x', '#', '#', '#', 'x', 'x', '#', '#'],
    ['.', '.', '.', '.', '#', 'x', 'x', 'x', '*', '*', '*', '*', '~', '~',
     '*', '*', '*', '*', '*', '.', '.', '#', '#', 'x', 'x', '#', '.'],
    ['.', '.', '.', '#', '#', 'x', 'x', '*', '*', '.', '.', '~', '~', '~',
     '~', '*', '*', '*', '.', '.', '.', '#', 'x', 'x', 'x', '#', '.'],
    ['.', '#', '#', '#', 'x', 'x', '#', '#', '.', '.', '.', '.', '~', '~',
     '~', '~', '~', '.', '.', '.', '.', '.', '#', 'x', '#', '.', '.'],
    ['.', '#', '#', 'x', 'x', '#', '#', '.', '.', '.', '.', '#', 'x', 'x',
     'x', '~', '~', '~', '.', '.', '.', '.', '.', '#', '.', '.', '.'],
    ['.', '.', '#', '#', '#', '#', '#', '.', '.', '.', '.', '.', '.', '#',
     'x', 'x', 'x', '~', '~', '~', '.', '.', '#', '#', '#', '.', '.'],
    ['.', '.', '.', '#', '#', '#', '.', '.', '.', '.', '.', '.', '#', '#',
     'x', 'x', '.', '~', '~', '.', '.', '#', '#', '#', '.', '.', '.'],
    ['.', '.', '.', '~', '~', '~', '.', '.', '#', '#', '#', 'x', 'x', 'x',
     'x', '.', '.', '.', '~', '.', '#', '#', '#', '.', '.', '.', '.'],
    ['.', '.', '~', '~', '~', '~', '~', '.', '#', '#', 'x', 'x', 'x', '#',
     '.', '.', '.', '.', '.', '#', 'x', 'x', 'x', '#', '.', '.', '.'],
    ['.', '~', '~', '~', '~', '~', '.', '.', '#', 'x', 'x', '#', '.', '.',
     '.', '.', '~', '~', '.', '.', '#', 'x', 'x', '#', '.', '.', '.'],
    ['~', '~', '~', '~', '~', '.', '.', '#', '#', 'x', 'x', '#', '.', '~',
     '~', '~', '~', '.', '.', '.', '#', 'x', '#', '.', '.', '.', '.'],
    ['.', '~', '~', '~', '~', '.', '.', '#', '*', '*', '#', '.', '.', '.',
     '.', '~', '~', '~', '~', '.', '.', '#', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', 'x', '.', '.', '*', '*', '*', '*', '#', '#', '#',
     '#', '.', '~', '~', '~', '.', '.', '#', 'x', '#', '.', '.', '.'],
    ['.', '.', '.', 'x', 'x', 'x', '*', '*', '*', '*', '*', '*', 'x', 'x',
     'x', '#', '#', '.', '~', '.', '#', 'x', 'x', '#', '.', '.', '.'],
    ['.', '.', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*',
     'x', 'x', 'x', '.', '.', 'x', 'x', 'x', '.', '.', '.', '.', '.'],
    ['.', '.', '.', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*', '*',
     '*', '*', 'x', 'x', 'x', 'x', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', 'x', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*',
     '.', '.', '.', '#', '#', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', 'x', 'x', 'x', '*', '*', '*', '*', '*', '*', '.',
     '.', '.', '.', '.', '.', '.', '.', '.', '.', '~', '~', '~', '~'],
    ['.', '.', '#', '#', '#', '#', 'x', 'x', '*', '*', '*', '*', '*', '.',
     'x', '.', '.', '.', '.', '.', '~', '~', '~', '~', '~', '~', '~'],
    ['.', '.', '.', '.', '#', '#', '#', 'x', 'x', 'x', '*', '*', 'x', 'x',
     '.', '.', '.', '.', '.', '.', '~', '~', '~', '~', '~', '~', '~'],
    ['.', '.', '.', '.', '.', '.', '#', '#', '#', 'x', 'x', 'x', 'x', '.',
     '.', '.', '.', '#', '#', '.', '.', '~', '~', '~', '~', '~', '~'],
    ['.', '#', '#', '.', '.', '#', '#', '#', '#', '#', '.', '.', '.', '.',
     '.', '#', '#', 'x', 'x', '#', '#', '.', '~', '~', '~', '~', '~'],
    ['#', 'x', '#', '#', '#', '#', '.', '.', '.', '.', '.', 'x', 'x', 'x',
     '#', '#', 'x', 'x', '.', 'x', 'x', '#', '#', '~', '~', '~', '~'],
    ['#', 'x', 'x', 'x', '#', '.', '.', '.', '.', '.', '#', '#', 'x', 'x',
     'x', 'x', '#', '#', '#', '#', 'x', 'x', 'x', '~', '~', '~', '~'],
    ['#', '#', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '#', '#', '#', '#', '#', '.', '.', '.', '.', '#', '#', '#', '.', '.', '.']]

world_path = a_star_search(
    full_world, (0, 0), (26, 26), costs, cardinal_moves, heuristic)
print("\nworld path: ", world_path)
# count = 0
# for move in world_path:
#     print("move ", count, ":", move)
#     count += 1

pretty_print_solution(test_world, test_path, (0, 0))
pretty_print_solution(full_world, world_path, (0, 0))
