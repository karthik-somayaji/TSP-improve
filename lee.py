from collections import defaultdict, Counter
import random
#import render
import argparse
import re

import heapq

class Lee:
    def __init__(self, n, m):
        self.n, self.m = n, m
        self.paths = {}
        self.via_cost = 10


    def add_path(self, nm, path):
        self.paths[nm] = path

    def show(self):

        h_edges = defaultdict(set)
        v_edges = defaultdict(set)

        for nm, path in self.paths.items():
            for p0, p1 in zip(path[:-1], path[1:]):
                i0, i1 = min(p0[0], p1[0]), max(p0[0], p1[0])
                j0, j1 = min(p0[1], p1[1]), max(p0[1], p1[1])
                k0, k1 = min(p0[2], p1[2]), max(p0[2], p1[2])
                if k0 != k1: # via
                    assert k0 + 1 == k1
                elif i0 == i1: # h_edge
                    assert j0 + 1 == j1
                    h_edges[((i0,j0),(i1,j1))].add(nm)
                elif j0 == j1: # v_edge
                    assert i0 + 1 == i1
                    v_edges[((i0,j0),(i1,j1))].add(nm)
                else:
                    assert False, (p0, p1)

        def h_edge_on( i, j):
            return h_edges[((i,j),(i,j+1))]

        def v_edge_on( i, j):
            return v_edges[((i,j),(i+1,j))]

        #render.show(self.n, self.m, h_edge_on, v_edge_on)


    def path2str(self, path):
        s = []
        for u, v in zip(path[:-1], path[1:]):
            if u[2] != v[2]: # via
                assert u[0] == v[0] and u[1] == v[1]
                if u[2]+1 == v[2]: # up
                    s.append('+')
                elif u[2]-1 == v[2]: # down
                    s.append('-')
                else:
                    assert False
            elif u[0] == v[0]: # horizontal
                if u[1]+1 == v[1]: # right
                    s.append('R')
                elif u[1]-1 == v[1]: # left
                    s.append('L')
                else:
                    assert False
            elif u[1] == v[1]: # vertical
                if u[0]+1 == v[0]: # down
                    s.append('D')
                elif u[0]-1 == v[0]: # up
                    s.append('U')
                else:
                    assert False
            else:
                assert False

        return ''.join(s)

    def _astar(self, nm, src, tgt, obstacles=None, heuristic=(lambda v: 0)):

        if obstacles is None:
            obstacles_s = set()
        else:
            obstacles_s = obstacles
        
        def adjacent_states(u):
            i, j, k = u

            if k == 0: # vertical
                next_states = [(i-1,j,0),(i+1,j,0),(i,j,1)]

            elif k == 1: # horizontal
                next_states = [(i,j-1,1),(i,j+1,1),(i,j,0)]
            else:
                assert False

            for ii, jj, kk in next_states:
                if 0 <= ii < self.n and 0 <= jj < self.m and 0 <= kk < 2 and (ii, jj) not in obstacles_s:
                    yield (ii, jj, kk), (self.via_cost if kk != k else 1)


        src0 = src[0], src[1], 0
        src1 = src[0], src[1], 1

        tgt0 = tgt[0], tgt[1], 0
        tgt1 = tgt[0], tgt[1], 1

        dist = {src0 : 0, src1 : 0}

        came_from = {src0 : None, src1 : None}

        q = [(0, src0), (0, src1)]

        heapq.heapify(q)

        while q:
            _, u = heapq.heappop(q)

            if u == tgt0 or u == tgt1:
                path = [u]
                while (u := came_from[u]) is not None:
                    path.append(u)
                path.reverse()
                return path

            for v, weight in adjacent_states(u):
                alt = dist[u] + weight
                if v not in dist or alt < dist[v]:
                    dist[v] = alt
                    priority = alt + heuristic(v)
                    heapq.heappush(q, (priority, v))
                    came_from[v] = u

        return None


    def dijkstra(self, nm, src, tgt, obstacles=None):
        return self._astar(nm, src, tgt, obstacles=obstacles)


    def astar(self, nm, src, tgt, obstacles=None):
        def heuristic(v):
            delta_i = abs(tgt[0] - v[0])
            delta_j = abs(tgt[1] - v[1])

            res = delta_i + delta_j
            
            if v[2] == 0 and delta_i != 0 or \
               v[2] == 1 and delta_j != 0:
                res += 10

            return res

        return self._astar(nm, src, tgt, obstacles=obstacles, heuristic=heuristic)

    def route_all(self, lst, alg='bfs', check=False):

        fn = self.astar if alg == 'astar' else self.dijkstra

        all_ok = True

        print("="*80)

        obstacles = set()

        for _, src, tgt in lst:
            obstacles.add(src)
            obstacles.add(tgt)

        for nm, src, tgt in lst:
            obstacles.remove(src)
            obstacles.remove(tgt)

            path_l = fn( nm, src, tgt, obstacles=obstacles)

            if path_l is None:
                obstacles.add(src)
                obstacles.add(tgt)
                all_ok = False
            else:
                obstacles.update([(tup[0], tup[1]) for tup in path_l])
                self.add_path(nm, path_l)

            print(nm, src, tgt, self.path2str(path_l) if path_l is not None else None)

        return all_ok

    def total_wire_length(self):
        s = 0
        for _, path in self.paths.items():
            ss = 0
            assert len(path) >= 1
            i0, j0, k0 = path[0]
            for i1, j1, k1 in path[1:]:
                ss += self.via_cost if k0 != k1 else 1
                i0, j0, k0 = i1, j1, k1 
            s += ss
        return s
        

def main(n, m, lst, num_trials, alg='bfs', check=False):
    count = 0
    histo = Counter()
    for _ in range(num_trials):
        samp = random.sample(lst, len(lst))
        #ind = [3,5,1,2,4,0]
        #samp = [lst[i] for i in ind]
        a = Lee(n, m)
        ok = a.route_all(samp, alg=alg, check=check)
        if ok:
            print(f'Routed all {len(lst)} nets. Wire Length = {a.total_wire_length()}')
            count += 1
            histo[a.total_wire_length()] += 1
        else:
            print(f'Only routed {len(a.paths)} of {len(lst)} nets.')

        a.show()
    print(f'Successfull routed {count} of {num_trials} times.')
    print(f'Wirelength histogram:', list(sorted(histo.items())))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Lee Router")
    parser.add_argument("-m", "--model", type=str, default="ten_nets_8x8")
    parser.add_argument("-n", "--num_trials", type=int, default=100)
    parser.add_argument("-a", "--alg", type=str, default='bfs')
    parser.add_argument("-c", "--check", action='store_true')

    args = parser.parse_args()

    if args.model == "two_nets_10x10":
        main(10, 10, [("a", (3,2), (7,6)), ("b", (6,4), (2,8))], num_trials=args.num_trials, alg=args.alg)

        """
  01234567
 +--------
0|  1  8  
1|     5
2|12 6  78
3|2   79
4|      6
5|3   54
6| 4    9a
7|  3  a
"""

    elif args.model == "ten_nets_8x8":

        main(8, 8, [
        ("1", (2, 0), (0, 2)),
        ("2", (3, 0), (2, 1)),
        ("3", (5, 0), (7, 2)),
        ("4", (6, 1), (5, 5)),
        ("5", (5, 4), (1, 5)),
        ("6", (2, 3), (4, 6)),
        ("7", (3, 4), (2, 6)),
        ("8", (0, 5), (2, 7)),
        ("9", (3, 5), (6, 6)),
        ("a", (7, 5), (6, 7)),
        ], num_trials=args.num_trials, alg=args.alg)


    elif args.model == "ten_nets_16x16":
        main(16, 16, [
        ("1", (4, 0), (0, 4)),
        ("2", (6, 0), (4, 2)),
        ("3", (10, 0), (14, 4)),
        ("4", (12, 2), (10, 10)),
        ("5", (10, 8), (2, 10)),
        ("6", (4, 6), (8, 12)),
        ("7", (6, 8), (4, 12)),
        ("8", (0, 10), (4, 14)),
        ("9", (6, 10), (12, 12)),
        ("a", (14, 10), (12, 14)),
        ], num_trials=args.num_trials, alg=args.alg)


    elif args.model == "ten_nets_24x24":
        main(24, 24, [
        ("1", (6, 0), (0, 6)),
        ("2", (9, 0), (6, 3)),
        ("3", (15, 0), (21, 6)),
        ("4", (18, 3), (15, 15)),
        ("5", (15, 12), (3, 15)),
        ("6", (6, 9), (12, 18)),
        ("7", (9, 12), (6, 18)),
        ("8", (0, 15), (6, 21)),
        ("9", (9, 15), (18, 18)),
        ("a", (21, 15), (18, 21))
        ], num_trials=args.num_trials, alg=args.alg)

    elif args.model == "river_8x8":
        main(8, 8, [
        ("0", (7, 0), (0, 7)),
        ("1", (7, 1), (1, 7)),
        ("2", (7, 2), (2, 7)),
        ("3", (7, 3), (3, 7)),
        ("4", (7, 4), (4, 7)),
        ("5", (7, 5), (5, 7)),
        ("6", (7, 6), (6, 7)),
        ], num_trials=args.num_trials, alg=args.alg)

    elif args.model == "synthetic_4x20":
        main(4, 20, [
        ("a", (1, 0), (3, 2)),
        ("A", (2, 1), (0, 3)),
        ("b", (1, 4), (3, 6)),
        ("B", (2, 5), (0, 7)),
        ("c", (1, 8), (3, 10)),
        ("C", (2, 9), (0, 11)),
        ("d", (1, 12), (3, 14)),
        ("D", (2, 13), (0, 15)),
        ("e", (1, 16), (3, 18)),
        ("E", (2, 17), (0, 19)),
        ], num_trials=args.num_trials, alg=args.alg, check=args.check)

    elif args.model == "synthetic_4x16":
        main(4, 16, [
        ("a", (1, 0), (3, 2)),
        ("A", (2, 1), (0, 3)),
        ("b", (1, 4), (3, 6)),
        ("B", (2, 5), (0, 7)),
        ("c", (1, 8), (3, 10)),
        ("C", (2, 9), (0, 11)),
        ("d", (1, 12), (3, 14)),
        ("D", (2, 13), (0, 15)),
        ], num_trials=args.num_trials, alg=args.alg, check=args.check)

    elif args.model == "synthetic_4x12":
        main(4, 12, [
        ("a", (1, 0), (3, 2)),
        ("A", (2, 1), (0, 3)),
        ("b", (1, 4), (3, 6)),
        ("B", (2, 5), (0, 7)),
        ("c", (1, 8), (3, 10)),
        ("C", (2, 9), (0, 11)),
        ], num_trials=args.num_trials, alg=args.alg, check=args.check)

    elif args.model == "synthetic_4x8":
        main(4, 12, [
        ("a", (1, 0), (3, 2)),
        ("A", (2, 1), (0, 3)),
        ("b", (1, 4), (3, 6)),
        ("B", (2, 5), (0, 7)),
        ], num_trials=args.num_trials, alg=args.alg, check=args.check)

    else:
        assert False, f"Unknown model: {args.model}"