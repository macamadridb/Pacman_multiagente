# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Comenzamos con la puntuacion del estado sucesor
        Score = successorGameState.getScore()
        severePenalty = float('-inf') # Penalizacion severa por cercania a sus fantasmas
        dangerRadius = 3 # Radio de peligro alrededor de los fantasmas

        #Calculamos la distancia a la comida mas cercana
        def evaluateFood(newFood, newPos):
            foodList = newFood.asList()
            totalScore = 0
            for food in foodList:
                distance = manhattanDistance(newPos, food)
                totalScore += 1.0 / distance
            return totalScore

        #Calcular la distancia a los fantasmas
        def evaluateGhosts(newGhostStates, newPos, dangerRadius, newScaredTimes):
            for i, ghost in enumerate(newGhostStates):
                ghostPosition = ghost.getPosition()
                distance = manhattanDistance(newPos, ghostPosition)

                if distance > 0  and distance < dangerRadius:
                    if newScaredTimes[i] == 0: #Fantasma no asustado
                        return severePenalty
                    else:
                        return 10
            return 0
       

        finalScore = Score + evaluateFood(newFood, newPos) + evaluateGhosts(newGhostStates, newPos, dangerRadius, newScaredTimes)
        return finalScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"  
         # Verifica si es un estado terminal
        def terminalTest(gameState, depth):
            return gameState.isWin() or gameState.isLose() or depth == self.depth

        # Funcion que determina el valor maximo para Pac-Man
        def maxValue(currState, depth):
            actions = currState.getLegalActions(0)
            if terminalTest(currState, depth) or len(actions) == 0:
                return self.evaluationFunction(currState)

            v = float('-inf')  # Valor inicial bajo

            for action in actions:
                newState = currState.generateSuccessor(0, action)
                v = max(v, minValue(newState, depth, 1))
            return v

        # Funcion que determina el valor minimo para los fantasmas
        def minValue(currState, depth, agent):
            actions = currState.getLegalActions(agent)
            if terminalTest(currState, depth) or len(actions) == 0:
                return self.evaluationFunction(currState)

            v = float('inf')  # Valor inicial alto

            for action in actions:
                newState = currState.generateSuccessor(agent, action)
                if agent == currState.getNumAgents() - 1:  # Ultimo fantasma
                    v = min(v, maxValue(newState, depth + 1))  # Retorna a maxValue y aumenta la profundidad
                else:
                    v = min(v, minValue(newState, depth, agent + 1))  # Continuar con los fantasmas
            return v

        # Encuentra la mejor accion para Pac-man en el estado inicial
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            score = minValue(successorState, 0, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Devuelve la accion minimax usando self.depth y self.evaluationFunction.
        """

        # Verifica si es un estado terminal
        def terminalTest(gameState, depth, actions):
            return gameState.isWin() or gameState.isLose() or len(actions) == 0 or depth == self.depth

        # Determina si es turno de Pacman o de un fantasma
        def fixCall(currState, newState, depth, agent, a, b):
            numAgents = currState.getNumAgents()
            if agent + 1 == numAgents:
                # Si el agente actual es el ultimo, llama a maxValue
                return maxValue(newState, depth + 1, 0, a, b)
            else:
                # Si no es el ultimo, llama a minValue
                return minValue(newState, depth, agent + 1, a, b)

        # Funcion que calcula el valor maximo para Pacman
        def maxValue(currState, depth, agent, a, b):
            actions = currState.getLegalActions(agent)
            if terminalTest(currState, depth, actions):
                # Si es un estado terminal, devuelve la evaluacion
                return self.evaluationFunction(currState), None

            v = float('-inf')  # Valor inicial bajo
            vAction = None      # Almacena la mejor accion

            for action in actions:
                # Genera el nuevo estado para cada accion
                newState = currState.generateSuccessor(agent, action)
                # Llama a minValue para evaluar el estado generado
                newValue, _ = minValue(newState, depth, 1, a, b)
                if newValue > v:
                    v = newValue
                    vAction = action

                # Poda si el valor es mayor que beta
                if v > b:
                    return v, vAction
                # Actualiza alpha
                a = max(a, v)

            return v, vAction

        # Funcion que calcula el valor minimo para los fantasmas
        def minValue(currState, depth, agent, a, b):
            actions = currState.getLegalActions(agent)
            if terminalTest(currState, depth, actions):
                # Si es un estado terminal, devuelve la evaluacion
                return self.evaluationFunction(currState), None

            v = float('inf')  # Valor inicial alto
            vAction = None     # Almacena la mejor accion

            for action in actions:
                # Genera el nuevo estado para cada accion
                newState = currState.generateSuccessor(agent, action)
                # Llama a fixCall para determinar el siguiente agente
                newValue, _ = fixCall(currState, newState, depth, agent, a, b)
                if newValue < v:
                    v = newValue
                    vAction = action

                # Poda si el valor es menor que alpha
                if v < a:
                    return v, vAction
                # Actualiza beta
                b = min(b, v)

            return v, vAction

        # Valores iniciales para alpha y beta
        alpha = float('-inf')
        beta = float('inf')
        # Llama a maxValue para iniciar el algoritmo y obtener la mejor accion
        _, action = maxValue(gameState, 0, 0, alpha, beta)
        return action
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
  ##COPIE Y PEGUE EL CODE DE MINIMAX PORQUE DABA ERROR SI NO HABIA CODIGO AQUI##
         # Verifica si es un estado terminal
        def terminalTest(gameState, depth):
            return gameState.isWin() or gameState.isLose() or depth == self.depth

        # Funcion que determina el valor maximo para Pac-Man
        def maxValue(currState, depth):
            actions = currState.getLegalActions(0)
            if terminalTest(currState, depth) or len(actions) == 0:
                return self.evaluationFunction(currState)

            v = float('-inf')  # Valor inicial bajo

            for action in actions:
                newState = currState.generateSuccessor(0, action)
                v = max(v, minValue(newState, depth, 1))
            return v

        # Funcion que determina el valor minimo para los fantasmas
        def minValue(currState, depth, agent):
            actions = currState.getLegalActions(agent)
            if terminalTest(currState, depth) or len(actions) == 0:
                return self.evaluationFunction(currState)

            v = float('inf')  # Valor inicial alto

            for action in actions:
                newState = currState.generateSuccessor(agent, action)
                if agent == currState.getNumAgents() - 1:  # Ultimo fantasma
                    v = min(v, maxValue(newState, depth + 1))  # Retorna a maxValue y aumenta la profundidad
                else:
                    v = min(v, minValue(newState, depth, agent + 1))  # Continuar con los fantasmas
            return v

        # Encuentra la mejor accion para Pac-man en el estado inicial
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            score = minValue(successorState, 0, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <Funcion de evaluacion para Pacman. Prioriza la comida cercana y evita fantasmas.>
    """
    "*** YOUR CODE HERE ***"
    # Informacion basica del estado actual
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()

    # Puntuacion inicial basada en el puntaje actual del juego
    score = currentGameState.getScore()

    # Incentivo por comida cercana (basado en la comida mas accesible)
    foodList = foodGrid.asList()
    if foodList:
        nearestFoodDist = min(manhattanDistance(pacmanPos, food) for food in foodList)
        score += 10 / nearestFoodDist  # Incentivo para la comida mas proxima

    # Penalizacion y/o incentivo basado en la proximidad a fantasmas
    for ghost in ghostStates:
        ghostDistance = manhattanDistance(pacmanPos, ghost.getPosition())
        if ghost.scaredTimer > 0:  # Si el fantasma esta asustado
            if ghostDistance > 0:
                score += 20 / ghostDistance  # Incentivo moderado por comer fantasmas asustados
        else:  # Si el fantasma no esta asustado
            if ghostDistance < 2:
                score -= 100  # Penalizacion alta si el fantasma esta en la misma casilla o adyacente
            else:
                score -= 10 / ghostDistance  # Penalizacion moderada si el fantasma esta cerca

    # Penalizacion por cantidad de comida restante (para incentivar la rapidez)
    score -= len(foodList) * 4

    return score
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

