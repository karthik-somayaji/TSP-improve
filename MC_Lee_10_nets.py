from collections import defaultdict, Counter
import random
import render
import argparse
import re
import statistics
import numpy as np

import heapq

class Lee:
    def __init__(self, n, m):
        self.n, self.m = n, m
        self.paths = {}
        self.via_cost = 10

        self.sum_of_counts = 0


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

        render.show(self.n, self.m, h_edge_on, v_edge_on)


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

        count = 0

        while q:
            count += 1
            _, u = heapq.heappop(q)

            if u == tgt0 or u == tgt1:
                path = [u]
                while (u := came_from[u]) is not None:
                    path.append(u)
                path.reverse()

                self.sum_of_counts += count

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
            
            # k=1 is horizontal
            if v[2] == 1 and delta_i != 0 or \
               v[2] == 0 and delta_j != 0:
                res += self.via_cost

            # causes failures
            #res += random.randrange(0, 11)

            return res

        return self._astar(nm, src, tgt, obstacles=obstacles, heuristic=heuristic)

    def route_all(self, lst, alg='astar', check=False):

        fn = self.astar if alg == 'astar' else self.dijkstra

        all_ok = True

        #print("="*80)

        obstacles = set()

        for _, src, tgt in lst:
            obstacles.add(src)
            obstacles.add(tgt)

        for nm, src, tgt in lst:
            obstacles.remove(src)
            obstacles.remove(tgt)

            path_l = fn( nm, src, tgt, obstacles=obstacles)
            if check:
                path_l_ref = self.dijkstra( nm, src, tgt, obstacles=obstacles)
                assert (path_l is None) == (path_l_ref is None)

                if path_l is not None:
                    pl, pl_ref = self.path_length(path_l), self.path_length(path_l_ref)
                    #print(f'Checking path lengths: {alg} {pl} dijkstra {pl_ref}')
                    assert pl == pl_ref

            if path_l is None:
                obstacles.add(src)
                obstacles.add(tgt)
                all_ok = False
            else:
                obstacles.update([(tup[0], tup[1]) for tup in path_l])
                self.add_path(nm, path_l)

            #print(nm, src, tgt, self.path2str(path_l) if path_l is not None else None)

        return self.total_wire_length() if all_ok else 1000

    def path_length(self, path):
        ss = 0
        assert len(path) >= 1
        i0, j0, k0 = path[0]
        for i1, j1, k1 in path[1:]:
            ss += self.via_cost if k0 != k1 else 1
            i0, j0, k0 = i1, j1, k1 
        return ss

    def total_wire_length(self):
        return sum(self.path_length(path) for _, path in self.paths.items())
        

def determine_order(nets):
    counts = []

    for net0, src0, tgt0 in nets:
        bbox = min(src0[0], tgt0[0]), min(src0[1], tgt0[1]), max(src0[0], tgt0[0]), max(src0[1], tgt0[1]), 

        count = 0

        for net1, src1, tgt1 in nets:
            if net0 == net1:
                continue

            if bbox[0] < src1[0] < bbox[2] and bbox[1] < src1[1] < bbox[3]:
                count += 1

            if bbox[0] < tgt1[0] < bbox[2] and bbox[1] < tgt1[1] < bbox[3]:
                count += 1

        counts.append(count)

    ordering = list(sorted(zip(counts,nets)))
    #print(ordering)

    return [b for _, b in ordering]

def main(n, m, lst, num_trials, alg='astar', check=False, order=False):
    count = 0
    histo = Counter()
    wirelength_arr = []

    def route(samp):
        nonlocal count

        a = Lee(n, m)

        length = a.route_all(samp, alg=alg, check=check)
        return length

    if order:
        samp = determine_order(lst)
        route(samp)
        if count == 1:
            print(f'Successfully routed using the ordering heuristic.')
        else:
            print(f'Routing failed when using the ordering heuristic.')
    else:
        for i in range(num_trials):
            samp = random.sample(lst, len(lst))
            length = route(samp)
            wirelength_arr.append(length)
    return wirelength_arr


def test_total_wire_length():
    a = Lee(4, 4)

    a.paths['0'] = [(0,0,0), (0,1,0), (0,1,1), (1,1,1)]
    a.via_cost = 10
    assert a.total_wire_length() == 12

    a.via_cost = 0
    assert a.total_wire_length() == 2


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Lee Router")
    parser.add_argument("-m", "--model", type=str, default="ten_nets_8x8")
    parser.add_argument("-n", "--num_trials", type=int, default=50)
    parser.add_argument("-a", "--alg", type=str, default='astar')
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-c", "--check", action='store_true')
    parser.add_argument("-o", "--order", action='store_true')

    args = parser.parse_args()

    nets = [15, 22, 29, 50]
    print(nets)
    nets_mean_break_pts = []

    lst1 = [
        ("0", (2, 0), (0, 2)),
        ("1", (3, 0), (2, 1)),
        ("2", (5, 0), (7, 2)),
        ("3", (6, 1), (5, 5)),
        ("4", (5, 4), (1, 5)),
        ("5", (2, 3), (4, 6)),
        ("6", (3, 4), (2, 6)),
        ("7", (0, 5), (2, 7)),
        ("8", (3, 5), (6, 6)),
        ("9", (7, 5), (6, 7)),
        ]

    for net in nets:
        mult = (net + 7)//8
        lst = [(x, (a*mult, b*mult), (c*mult, d*mult)) for (x,(a,b), (c,d)) in lst1]
        print('Analysing Lee_', str(net) )
        mean_break_pts = []
        prob = "Lee_" + str(net)
        num = 10
        for seed in range(5):
            random.seed(seed+8)
            arr = main(net, net, lst, num_trials=args.num_trials, alg=args.alg, check=args.check, order=args.order)
            print(arr)
            break_point = args.num_trials if np.all(np.array(arr)==args.num_trials) else np.argmin(arr)
            mean_break_pts.append(break_point)
            print(break_point)
        nets_mean_break_pts.append(statistics.mean(mean_break_pts)) 
        
    nets_mean_break_pts = np.array(nets_mean_break_pts)
    print(nets_mean_break_pts)
    np.savetxt('saved_files/MC_10nets_all_nets.txt', nets_mean_break_pts )
