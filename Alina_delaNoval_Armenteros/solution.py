"""
SmartPlayer for HEX using MC-RAVE with incremental Union-Find and MAST sampling

- MC-RAVE: Monte Carlo Tree Search with AMAF statistics
- Incremental Union-Find: fast win detection via virtual nodes
- MAST: learns move quality from playouts using Boltzmann sampling
"""

from player import Player
from board import HexBoard
import time, math, random, copy
from collections import deque

# Search parameters
TIME_LIMIT    = 4.5
EXPLORATION_C = 0.0  # Pure exploitation (RAVE handles exploration)
RAVE_BIAS     = 0.00913  # Balance between UCT and AMAF statistics
FPU           = 0.35  # First-Play Urgency for untried moves
MAST_TEMPERATURE = 0.3  # Lower tau increases MAST influence in rollout
LGR_RANDOMNESS = 0.2  # Exploration probability: ignore LGR even when available

# Hexagonal adjacency directions (even-r offset coordinate system)
_DIRS_EVEN = ((-1,-1),(-1,0),(0,-1),(0,1),(1,-1),(1,0))
_DIRS_ODD  = ((-1, 0),(-1,1),(0,-1),(0,1),(1, 0),(1,1))

def get_neighbors(row, col, size):
    """Returns valid neighbors of a hexagonal cell."""
    dirs = _DIRS_EVEN if row % 2 == 0 else _DIRS_ODD
    return [(row+dr, col+dc) for dr,dc in dirs
            if 0 <= row+dr < size and 0 <= col+dc < size]

class HexUnionFind:
    """Fast win detection using Union-Find with virtual nodes.
    
    Virtual nodes connect to board edges, so finding a path between
    opposite edges becomes a simple find() operation.
    """
    __slots__ = ('parent','rank','size','VL','VR','VT','VB')

    def __init__(self, size):
        n = size*size
        self.parent = list(range(n+4))
        self.rank   = [0]*(n+4)
        self.size   = size
        self.VL=n; self.VR=n+1; self.VT=n+2; self.VB=n+3

    def _idx(self, r, c): return r*self.size+c

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra,rb = self.find(a), self.find(b)
        if ra==rb: return
        if self.rank[ra]<self.rank[rb]: ra,rb=rb,ra
        self.parent[rb]=ra
        if self.rank[ra]==self.rank[rb]: self.rank[ra]+=1

    def place(self, r, c, player, grid):
        """Union cell with board edges and same-colored neighbors."""
        i = self._idx(r,c)
        if player==1:
            if c==0:             self.union(i, self.VL)
            if c==self.size-1:   self.union(i, self.VR)
        else:
            if r==0:             self.union(i, self.VT)
            if r==self.size-1:   self.union(i, self.VB)
        for nr,nc in get_neighbors(r,c,self.size):
            if grid[nr][nc]==player:
                self.union(i, self._idx(nr,nc))

    def p1_wins(self): return self.find(self.VL)==self.find(self.VR)
    def p2_wins(self): return self.find(self.VT)==self.find(self.VB)


def _build_uf_from_grid(grid, size):
    """Build Union-Find structures from current board state for both players."""
    uf1,uf2 = HexUnionFind(size), HexUnionFind(size)
    for r in range(size):
        for c in range(size):
            p = grid[r][c]
            if p==1: uf1.place(r,c,1,grid)
            elif p==2: uf2.place(r,c,2,grid)
    return uf1, uf2

def check_win(grid, player_id, size):
    """BFS-based win check for immediate winning/blocking moves."""
    visited,queue = set(),deque()
    if player_id==1:
        for r in range(size):
            if grid[r][0]==1: queue.append((r,0)); visited.add((r,0))
        goal = lambda r,c: c==size-1
    else:
        for c in range(size):
            if grid[0][c]==2: queue.append((0,c)); visited.add((0,c))
        goal = lambda r,c: r==size-1
    while queue:
        r,c = queue.popleft()
        if goal(r,c): return True
        for nr,nc in get_neighbors(r,c,size):
            if (nr,nc) not in visited and grid[nr][nc]==player_id:
                queue.append((nr,nc)); visited.add((nr,nc))
    return False


def _find_decisive_moves(grid, size, player_id):
    """Find winning move and blocking move using Union-Find.
    
    Tests each empty cell to detect if placement wins or blocks opponent.
    Returns immediately when decisive move found.
    """
    opp_id = 3 - player_id
    uf_own = HexUnionFind(size)
    uf_opp = HexUnionFind(size)
    
    for r in range(size):
        for c in range(size):
            p = grid[r][c]
            if p == player_id:
                uf_own.place(r, c, player_id, grid)
            elif p == opp_id:
                uf_opp.place(r, c, opp_id, grid)
    for r in range(size):
        for c in range(size):
            if grid[r][c] != 0:
                continue
            
            parent_own_save = uf_own.parent[:]
            rank_own_save = uf_own.rank[:]
            uf_own.place(r, c, player_id, grid)
            
            if (player_id == 1 and uf_own.p1_wins()) or (player_id == 2 and uf_own.p2_wins()):
                return (r, c), None
            
            uf_own.parent = parent_own_save
            uf_own.rank = rank_own_save
            
            parent_opp_save = uf_opp.parent[:]
            rank_opp_save = uf_opp.rank[:]
            uf_opp.place(r, c, opp_id, grid)
            
            if (opp_id == 1 and uf_opp.p1_wins()) or (opp_id == 2 and uf_opp.p2_wins()):
                return None, (r, c)
            
            uf_opp.parent = parent_opp_save
            uf_opp.rank = rank_opp_save
    
    return None, None

class MCTSNode:
    """Node in MCTS tree with MC-RAVE (AMAF) statistics and incremental Union-Find."""
    __slots__ = ('move','player','parent','children',
                 'visits','wins','visits_amaf','wins_amaf',
                 'untried_moves','empty','uf1','uf2')

    def __init__(self, move, player, parent, legal_moves, grid=None, size=None,
                 parent_uf1=None, parent_uf2=None):
        self.move          = move
        self.player        = player
        self.parent        = parent
        self.children      = {}
        self.visits        = 0
        self.wins          = 0.0
        self.visits_amaf   = 0
        self.wins_amaf     = 0.0
        self.empty         = list(legal_moves)
        self.untried_moves = list(legal_moves)
        random.shuffle(self.untried_moves)
        
        if parent_uf1 is not None and move is not None:
            self.uf1 = copy.deepcopy(parent_uf1)
            self.uf2 = copy.deepcopy(parent_uf2)
            r, c = move
            if player == 1:
                self.uf1.place(r, c, 1, grid)
            else:
                self.uf2.place(r, c, 2, grid)
        elif grid is not None and size is not None:
            self.uf1, self.uf2 = _build_uf_from_grid(grid, size)
        else:
            self.uf1 = None
            self.uf2 = None

    def rave_value(self):
        if self.visits==0:
            if self.visits_amaf>0:
                return self.wins_amaf/self.visits_amaf + FPU
            return float('inf')
        q_uct = self.wins/self.visits
        explore = 0.0
        if EXPLORATION_C>0.0 and self.parent and self.parent.visits>0:
            explore = EXPLORATION_C*math.sqrt(math.log(self.parent.visits)/self.visits)
        if self.visits_amaf>0:
            q_amaf = self.wins_amaf/self.visits_amaf
            n,nt   = self.visits, self.visits_amaf
            beta   = nt/(n+nt+4.0*n*nt*RAVE_BIAS*RAVE_BIAS)
            q      = (1.0-beta)*q_uct + beta*q_amaf
        else:
            q = q_uct
        return q + explore

    def is_fully_expanded(self): return len(self.untried_moves)==0
    def best_child_rave(self):
        return max(self.children.values(), key=lambda c: c.rave_value())
    def most_visited_child(self):
        return max(self.children.values(), key=lambda c: c.visits)

def _fast_rollout(grid, size, start_player, empty_list):
    """Fast random playout using Union-Find for win detection."""
    sim_grid = [row[:] for row in grid]
    uf1, uf2 = _build_uf_from_grid(sim_grid, size)
    moves = list(empty_list)
    random.shuffle(moves)
    moves_by_player = {1:[],2:[]}
    player = start_player
    for r,c in moves:
        sim_grid[r][c] = player
        moves_by_player[player].append((r,c))
        if player==1:
            uf1.place(r,c,1,sim_grid)
            if uf1.p1_wins():
                return 1, moves_by_player
        else:
            uf2.place(r,c,2,sim_grid)
            if uf2.p2_wins():
                return 2, moves_by_player
        player = 3-player
    return (1 if uf1.p1_wins() else 2), moves_by_player

class MCTSTree:
    """MC-RAVE search engine with MAST sampling: learns from playout statistics."""

    def __init__(self, player_id, grid, size):
        self.player_id = player_id
        self.size      = size
        self.root_grid = [row[:] for row in grid]
        legal = [(r,c) for r in range(size) for c in range(size) if grid[r][c]==0]
        self.root = MCTSNode(move=None, player=None, parent=None, legal_moves=legal,
                             grid=grid, size=size)
        self.root.visits = 1
        
        # MAST statistics indexed by (player, move) to avoid cross-player leakage.
        self.mast_wins   = {}  # (player, move) -> wins
        self.mast_total  = {}  # (player, move) -> total trials
        self.mast_tau    = MAST_TEMPERATURE
        self.lgr         = {1: {}, 2: {}}  # lgr[player][opponent_last_move] = reply_move

    def search(self, time_limit):
        deadline = time.time()+time_limit
        iters = 0
        while time.time()<deadline:
            self._run_iteration()
            iters += 1
        return iters

    def best_move(self):
        if not self.root.children: return None
        return self.root.most_visited_child().move

    def _mast_weight(self, player, move):
        """Boltzmann weight for a (player, move) pair."""
        key = (player, move)
        total = self.mast_total.get(key, 0)
        wins = self.mast_wins.get(key, 0)
        q = (wins / total) if total > 0 else 0.5
        return math.exp(q / self.mast_tau)

    def _mast_pick(self, legal_moves, player):
        """Pick one legal move for the current player using MAST weights."""
        if not legal_moves:
            return None
        weights = [self._mast_weight(player, mv) for mv in legal_moves]
        total_w = sum(weights)
        if total_w <= 0:
            return random.choice(legal_moves)
        r_val = random.random() * total_w
        cumsum = 0.0
        for idx, w in enumerate(weights):
            cumsum += w
            if r_val < cumsum:
                return legal_moves[idx]
        return legal_moves[-1]

    def _bridge_response(self, sim_grid, size, last_move, player, legal):
        """Detect threatened local bridge carriers and return a repair move if any."""
        if last_move is None:
            return None
        r, c = last_move
        neighbors = get_neighbors(r, c, size)

        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1:]:
                r1, c1 = n1
                r2, c2 = n2

                if sim_grid[r1][c1] != player or sim_grid[r2][c2] != player:
                    continue

                neigh1 = set(get_neighbors(r1, c1, size))
                neigh2 = set(get_neighbors(r2, c2, size))
                carriers = (neigh1 & neigh2) - {(r, c)}

                for ro, co in carriers:
                    if sim_grid[ro][co] == 0 and (ro, co) in legal:
                        return (ro, co)
        return None

    def rollout(self, sim_grid, size, start_player, empty_list,
                node_uf1, node_uf2):
        """Turn-by-turn playout with player-specific MAST and Union-Find."""
        uf1 = copy.deepcopy(node_uf1)
        uf2 = copy.deepcopy(node_uf2)
        legal = set(empty_list)
        
        moves_by_player = {1:[], 2:[]}
        player = start_player
        last_move = None
        
        while legal:
            mv = None

            # Priority 1: bridge response.
            mv = self._bridge_response(sim_grid, size, last_move, player, legal)

            # Priority 2: stochastic LGR reply to opponent's latest move.
            if mv is None and last_move is not None:
                cand = self.lgr[player].get(last_move)
                if cand is not None and cand in legal:
                    if random.random() > LGR_RANDOMNESS:
                        mv = cand

            # Priority 3: player-specific MAST.
            if mv is None:
                mv = self._mast_pick(list(legal), player)

            r, c = mv
            legal.discard(mv)

            sim_grid[r][c] = player
            moves_by_player[player].append(mv)
            
            if player == 1:
                uf1.place(r, c, 1, sim_grid)
                if uf1.p1_wins():
                    return 1, moves_by_player
            else:
                uf2.place(r, c, 2, sim_grid)
                if uf2.p2_wins():
                    return 2, moves_by_player
            
            last_move = mv
            player = 3 - player
        
        return (1 if uf1.p1_wins() else 2), moves_by_player

    def _run_iteration(self):
        # Selection: traverse best path until unfully-expanded node
        node           = self.root
        sim_grid       = [row[:] for row in self.root_grid]
        current_player = self.player_id
        while node.is_fully_expanded() and node.children:
            node = node.best_child_rave()
            sim_grid[node.move[0]][node.move[1]] = node.player
            current_player = 3-node.player

        # Expansion: add new child node
        if node.untried_moves:
            idx  = random.randrange(len(node.untried_moves))
            move = node.untried_moves[idx]
            node.untried_moves[idx] = node.untried_moves[-1]
            node.untried_moves.pop()
            r,c = move
            sim_grid[r][c] = current_player
            legal_child = [m for m in node.empty if m != move]
            child = MCTSNode(move=move, player=current_player,
                             parent=node, legal_moves=legal_child,
                             grid=sim_grid, size=self.size,
                             parent_uf1=node.uf1, parent_uf2=node.uf2)
            node.children[move] = child
            node           = child
            current_player = 3-current_player

        # Simulation: fast playout
        winner, moves_by_player = self.rollout(
            sim_grid, self.size, current_player, node.empty,
            node.uf1, node.uf2
        )

        # Backpropagation: update statistics along path
        self._backpropagate(node, winner, moves_by_player)

    def _backpropagate(self, leaf, winner, moves_by_player):
        current = leaf
        while current is not None:
            current.visits += 1
            if current.player is not None and current.player==winner:
                current.wins += 1.0
            if current.children:
                next_player = (self.player_id if current.player is None
                               else 3-current.player)
                for move in moves_by_player.get(next_player,()):
                    if move in current.children:
                        child = current.children[move]
                        child.visits_amaf += 1
                        if child.player==winner:
                            child.wins_amaf += 1.0
            current = current.parent
        
        # Update MAST statistics after backpropagation completes
        for player_id, moves in moves_by_player.items():
            for move in moves:
                key = (player_id, move)
                self.mast_total[key] = self.mast_total.get(key, 0) + 1
                if player_id == winner:
                    self.mast_wins[key] = self.mast_wins.get(key, 0) + 1

        # Update LGR table from winning replies.
        seq1 = moves_by_player[1]
        seq2 = moves_by_player[2]
        if winner == 1:
            offset = 1 if self.player_id == 1 else 0
            skip = 1 if len(seq1) > len(seq2) else 0
            for i in range(len(seq2) - skip):
                if i + offset < len(seq1):
                    self.lgr[1][seq2[i]] = seq1[i + offset]
        else:
            offset = 1 if self.player_id == 2 else 0
            skip = 1 if len(seq2) > len(seq1) else 0
            for i in range(len(seq1) - skip):
                if i + offset < len(seq2):
                    self.lgr[2][seq1[i]] = seq2[i + offset]


class SmartPlayer(Player):
    """Autonomous HEX player using MC-RAVE + MAST algorithm.
    
    Strategy: 
    1. Center opening
    2. Winning move detection
    3. Block opponent threat
    4. MCTS search with MAST sampling
    """
    def __init__(self, player_id):
        super().__init__(player_id)

    def play(self, board):
        size   = board.size
        grid   = board.board
        legal  = [(r,c) for r in range(size) for c in range(size) if grid[r][c]==0]
        
        if not legal:
            raise ValueError("No moves available")
        if len(legal) == 1:
            return legal[0]
        if len(legal) == size*size:
            return (size//2, size//2)
        
        win_move, block_move = _find_decisive_moves(grid, size, self.player_id)
        
        if win_move is not None:
            return win_move
        if block_move is not None:
            return block_move
        
        mcts = MCTSTree(player_id=self.player_id, grid=grid, size=size)
        mcts.search(time_limit=TIME_LIMIT)
        best = mcts.best_move()
        return best if best is not None else random.choice(legal)
