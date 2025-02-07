# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from collections import deque
from game import Directions
from typing import List
#from searchAgents import manhattanHeuristic
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    start=problem.getStartState()
    if problem.isGoalState(start):
        return []
    visited=set()
    result=[]
    def dfs(current: tuple[int, int],path: list=[]):
        visited.add(current)
        if problem.isGoalState(current):
            result.extend(path)
            return True
        for next,move,cost in problem.getSuccessors(current):
            if next in visited:
                continue
            path.append(move)
            if dfs(next,path):
                return True
            path.pop()
        return False
    dfs(start)
    return result
    

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []
    visited = set()
    fringe = deque()
    parent = {}
    fringe.append(start)
    visited.add(start)

    goal_state = None
    while fringe:
        state = fringe.popleft()
        if problem.isGoalState(state):
            goal_state = state
            break
        for next_state, action, cost in problem.getSuccessors(state):
            if next_state not in visited:
                visited.add(next_state)
                parent[next_state] = (state, action)
                fringe.append(next_state)
    
    if goal_state is None:
        return []
    
    # 回溯构造路径
    path = []
    while goal_state != start:
        state, action = parent[goal_state]
        path.append(action)
        goal_state = state
    path.reverse()
    return path
        
def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []
    frontier = PriorityQueue()
    frontier.push((start, 0), 0)
    bestCost = {start: 0}  # 记录状态最佳累计代价
    parent = {}
    goal_state = None
    while not frontier.isEmpty():
        state, cost = frontier.pop()
        # 若当前累计代价大于已知最佳代价，则跳过
        if cost > bestCost[state]:
            continue

        if problem.isGoalState(state):
            goal_state = state
            break

        for nextState, action, stepCost in problem.getSuccessors(state):
            newCost = cost + stepCost
            # 只有发现更低代价时更新，不进行相等时的替换
            if newCost < bestCost.get(nextState, float('inf')):
                bestCost[nextState] = newCost
                frontier.push((nextState, newCost), newCost)
                parent[nextState] = (state, action)
    path = []
    if goal_state is None:
        return path
    while goal_state != start:
        state, action = parent[goal_state]
        path.append(action)
        goal_state = state
    path.reverse()
    return path
def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []
    frontier = PriorityQueue()
    # 初始优先级为启发值
    frontier.push((start, 0), heuristic(start, problem))
    bestCost = {start: 0}
    parent = {}
    goal_state = None

    while not frontier.isEmpty():
        state, cost = frontier.pop()
        if problem.isGoalState(state):
            goal_state = state
            break

        for nextState, action, stepCost in problem.getSuccessors(state):
            newCost = cost + stepCost
            if newCost < bestCost.get(nextState, float('inf')):
                bestCost[nextState] = newCost
                priority = newCost + heuristic(nextState, problem)
                frontier.push((nextState, newCost), priority)
                parent[nextState] = (state, action)

    path = []
    if goal_state is None:
        return path
    while goal_state != start:
        state, action = parent[goal_state]
        path.append(action)
        goal_state = state
    path.reverse()
    return path

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
