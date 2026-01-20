from torch.utils.data import Dataset
import torch
import pickle
import os


class MVPDTSP(object):

    NAME = 'mvpdtsp'  # Multi-Vehicle Pickup and Delivery TSP

    def __init__(self, p_size, num_vehicles=2, init_val_met='p2d', with_assert=False, capacity=None, use_makespan=False):

        self.size = p_size  # number of pickup+delivery nodes
        self.num_vehicles = num_vehicles
        self.num_depots = num_vehicles
        self.num_pairs = p_size // 2
        self.pickup_start = self.num_depots
        self.delivery_start = self.num_depots + self.num_pairs
        self.do_assert = with_assert
        self.init_val_met = init_val_met
        self.capacity = capacity
        self.use_makespan = use_makespan
        self.state = 'eval'
        print(
            f'MVPDTSP with {self.size} nodes and {self.num_vehicles} vehicles.',
            ' Do assert:', with_assert,
            ' Use makespan:', use_makespan,
        )

    def input_feature_encoding(self, batch):
        return batch['coordinates']

    def get_pickup_indices(self, device=None):
        return torch.arange(self.pickup_start, self.pickup_start + self.num_pairs, device=device)

    def get_delivery_indices(self, device=None):
        return torch.arange(self.delivery_start, self.delivery_start + self.num_pairs, device=device)

    def map_action_to_pickup(self, action_removal):
        return action_removal + self.pickup_start

    def pickup_to_delivery(self, pickup_index):
        return pickup_index + self.num_pairs

    def get_visited_order_map(self, visited_time):
        bs, gs = visited_time.size()
        order = visited_time % gs
        mask_order = order.view(bs, gs, 1) > order.view(bs, 1, gs)
        return mask_order

    def get_vehicle_id(self, rec):
        bs, total_nodes = rec.size()
        vehicle_id = torch.full((bs, total_nodes), -1, device=rec.device, dtype=torch.long)
        arange = torch.arange(bs, device=rec.device)

        for v in range(self.num_depots):
            pre = torch.full((bs,), v, device=rec.device, dtype=torch.long)
            active = torch.ones((bs,), device=rec.device, dtype=torch.bool)

            for _ in range(total_nodes):
                current = rec[arange, pre]
                new = (vehicle_id[arange, current] < 0) & active
                vehicle_id[arange, current] = torch.where(
                    new, torch.full_like(vehicle_id[arange, current], v), vehicle_id[arange, current]
                )
                pre = current
                active = active & (current != v)
                if not active.any():
                    break

        return vehicle_id

    def get_vehicle_id_and_order(self, rec):
        bs, total_nodes = rec.size()
        vehicle_id = torch.full((bs, total_nodes), -1, device=rec.device, dtype=torch.long)
        order = torch.full((bs, total_nodes), -1, device=rec.device, dtype=torch.long)
        arange = torch.arange(bs, device=rec.device)

        for v in range(self.num_depots):
            vehicle_id[arange, v] = v
            order[arange, v] = 0
            pre = torch.full((bs,), v, device=rec.device, dtype=torch.long)
            active = torch.ones((bs,), device=rec.device, dtype=torch.bool)
            step = torch.zeros((bs,), device=rec.device, dtype=torch.long)

            for _ in range(total_nodes):
                current = rec[arange, pre]
                new = (vehicle_id[arange, current] < 0) & active
                step = torch.where(active, step + 1, step)
                order[arange, current] = torch.where(new, step, order[arange, current])
                vehicle_id[arange, current] = torch.where(
                    new, torch.full_like(vehicle_id[arange, current], v), vehicle_id[arange, current]
                )
                pre = current
                active = active & (current != v)
                if not active.any():
                    break

        return vehicle_id, order

    def remove_pair_from_rec(self, rec, pair_index):
        rec_removed = rec.clone()
        argsort = rec_removed.argsort()
        pre_pairfirst = argsort.gather(1, pair_index)
        post_pairfirst = rec_removed.gather(1, pair_index)
        rec_removed.scatter_(1, pre_pairfirst, post_pairfirst)
        rec_removed.scatter_(1, pair_index, pair_index)

        argsort = rec_removed.argsort()
        pre_pairsecond = argsort.gather(1, pair_index + self.num_pairs)
        post_pairsecond = rec_removed.gather(1, pair_index + self.num_pairs)
        rec_removed.scatter_(1, pre_pairsecond, post_pairsecond)

        return rec_removed

    def get_real_mask(self, selected_node, visited_order_map):

        bs, gs, _ = visited_order_map.size()
        selected_node = selected_node.view(-1)

        mask = visited_order_map.clone()
        mask[torch.arange(bs), selected_node] = True
        mask[torch.arange(bs), self.pickup_to_delivery(selected_node)] = True
        mask[torch.arange(bs), :, selected_node] = True
        mask[torch.arange(bs), :, self.pickup_to_delivery(selected_node)] = True

        return mask

    def get_initial_solutions(self, batch, val_m=1):

        batch_size = batch['coordinates'].size(0)
        total_nodes = self.num_depots + self.size

        def build_solution(methods):

            rec = torch.zeros(batch_size, total_nodes).long()
            pickup_indices = torch.arange(self.num_pairs)

            for b in range(batch_size):
                # assign pickup-delivery pairs to vehicles
                if methods == 'random':
                    perm = pickup_indices[torch.randperm(self.num_pairs)]
                elif methods == 'greedy':
                    perm = pickup_indices
                else:
                    raise NotImplementedError()

                vehicle_routes = [[] for _ in range(self.num_vehicles)]
                for idx, p in enumerate(perm):
                    v = idx % self.num_vehicles
                    pickup = self.pickup_start + p
                    delivery = self.delivery_start + p
                    vehicle_routes[v].extend([pickup.item(), delivery.item()])

                # if greedy, reorder within each vehicle by nearest neighbor on pickups
                if methods == 'greedy':
                    coords = batch['coordinates'][b].cpu()
                    for v in range(self.num_vehicles):
                        route = vehicle_routes[v]
                        if not route:
                            continue
                        depot_idx = v
                        remaining = route.copy()
                        ordered = []
                        current = depot_idx
                        while remaining:
                            # pick next pickup (even index in remaining list) closest to current
                            pickup_candidates = remaining[0::2]
                            pickup_coords = coords[pickup_candidates]
                            current_coord = coords[current]
                            dists = ((pickup_coords - current_coord) ** 2).sum(-1)
                            p_idx = int(torch.argmin(dists).item())
                            pickup = pickup_candidates[p_idx]
                            delivery = pickup + self.num_pairs
                            ordered.extend([pickup, delivery])
                            remaining.remove(pickup)
                            remaining.remove(delivery)
                            current = delivery
                        vehicle_routes[v] = ordered

                # build disjoint cycles
                for v in range(self.num_vehicles):
                    depot_idx = v
                    route = vehicle_routes[v]
                    if len(route) == 0:
                        rec[b, depot_idx] = depot_idx
                        continue
                    rec[b, depot_idx] = route[0]
                    for i in range(len(route) - 1):
                        rec[b, route[i]] = route[i + 1]
                    rec[b, route[-1]] = depot_idx

            return rec

        return build_solution(self.init_val_met).expand(batch_size, total_nodes).clone()

    def step(self, batch, rec, exchange, pre_bsf, action_record):

        bs, gs = rec.size()
        pre_bsf = pre_bsf.view(bs, -1)

        cur_vec = action_record.pop(0) * 0.
        cur_vec[torch.arange(bs), exchange[:, 0]] = 1.
        action_record.append(cur_vec)

        selected = exchange[:, 0].view(bs, 1)
        selected = self.map_action_to_pickup(selected)
        first = exchange[:, 1].view(bs, 1)
        second = exchange[:, 2].view(bs, 1)

        next_state = self.insert_star(rec, selected, first, second)

        new_obj = self.get_costs(batch, next_state)

        now_bsf = torch.min(torch.cat((new_obj[:, None], pre_bsf[:, -1, None]), -1), -1)[0]

        reward = pre_bsf[:, -1] - now_bsf

        return next_state, reward, torch.cat((new_obj[:, None], now_bsf[:, None]), -1), action_record

    def insert_star(self, solution, pair_index, first, second):

        rec = solution.clone()
        bs, gs = rec.size()

        # fix connection for pairing node
        argsort = rec.argsort()
        pre_pairfirst = argsort.gather(1, pair_index)
        post_pairfirst = rec.gather(1, pair_index)
        rec.scatter_(1, pre_pairfirst, post_pairfirst)
        rec.scatter_(1, pair_index, pair_index)

        argsort = rec.argsort()

        pre_pairsecond = argsort.gather(1, pair_index + self.num_pairs)
        post_pairsecond = rec.gather(1, pair_index + self.num_pairs)

        rec.scatter_(1, pre_pairsecond, post_pairsecond)

        # fix connection for pairing node
        post_second = rec.gather(1, second)
        rec.scatter_(1, second, pair_index + self.num_pairs)
        rec.scatter_(1, pair_index + self.num_pairs, post_second)

        post_first = rec.gather(1, first)
        rec.scatter_(1, first, pair_index)
        rec.scatter_(1, pair_index, post_first)

        return rec

    def check_feasibility(self, rec):

        total_nodes = self.num_depots + self.size

        assert (
            (torch.arange(total_nodes, out=rec.new())).view(1, -1).expand_as(rec)
            == rec.sort(1)[0]
        ).all(), (
            (
                (torch.arange(total_nodes, out=rec.new())).view(1, -1).expand_as(rec)
                == rec.sort(1)[0]
            ),
            "not visiting all nodes",
            rec,
        )

        # calculate visited time per vehicle
        bs = rec.size(0)
        visited_time = torch.zeros((bs, total_nodes), device=rec.device)
        vehicle_id = torch.full((bs, total_nodes), -1, device=rec.device, dtype=torch.long)

        arange = torch.arange(bs, device=rec.device)

        for v in range(self.num_depots):
            pre = torch.full((bs,), v, device=rec.device, dtype=torch.long)
            order = torch.zeros((bs,), device=rec.device, dtype=torch.long)
            active = torch.ones((bs,), device=rec.device, dtype=torch.bool)

            for _ in range(total_nodes):
                current = rec[arange, pre]
                new = (vehicle_id[arange, current] < 0) & active
                order = torch.where(active, order + 1, order)
                visited_time[arange, current] = torch.where(
                    new, v * total_nodes + order, visited_time[arange, current]
                )
                vehicle_id[arange, current] = torch.where(
                    new, torch.full_like(vehicle_id[arange, current], v), vehicle_id[arange, current]
                )
                pre = current
                active = active & (current != v)
                if not active.any():
                    break

        pickup_indices = self.get_pickup_indices(device=rec.device)
        delivery_indices = self.get_delivery_indices(device=rec.device)

        pickup_vehicle = vehicle_id[:, pickup_indices]
        delivery_vehicle = vehicle_id[:, delivery_indices]
        assert (pickup_vehicle == delivery_vehicle).all(), "pickup/delivery in different vehicles"

        pickup_order = visited_time[:, pickup_indices]
        delivery_order = visited_time[:, delivery_indices]
        assert (pickup_order < delivery_order).all(), "deliverying without pick-up"

    def get_swap_mask(self, selected_node, visited_order_map, top2=None, rec=None):
        if rec is None:
            return self.get_real_mask(selected_node, visited_order_map)

        selected_node = selected_node.view(-1, 1)
        rec_removed = self.remove_pair_from_rec(rec, selected_node)
        vehicle_id, order = self.get_vehicle_id_and_order(rec_removed)
        bs, gs = order.size()

        mask_order = order.view(bs, gs, 1) > order.view(bs, 1, gs)
        invalid_order = (order.view(bs, gs, 1) < 0) | (order.view(bs, 1, gs) < 0)
        mask_vehicle = vehicle_id.view(bs, gs, 1) != vehicle_id.view(bs, 1, gs)
        invalid_vehicle = (vehicle_id.view(bs, gs, 1) < 0) | (vehicle_id.view(bs, 1, gs) < 0)
        mask = mask_order | invalid_order | mask_vehicle | invalid_vehicle

        selected_node = selected_node.view(-1)
        delivery_node = self.pickup_to_delivery(selected_node)
        arange = torch.arange(bs, device=rec.device)
        mask[arange, selected_node] = True
        mask[arange, delivery_node] = True
        mask[arange, :, selected_node] = True
        mask[arange, :, delivery_node] = True

        return mask

    def _get_route_lengths(self, batch, rec):

        coords = batch['coordinates']
        batch_size, total_nodes, _ = coords.size()
        arange = torch.arange(batch_size, device=rec.device)

        route_lengths = torch.zeros(batch_size, self.num_vehicles, device=rec.device)
        for v in range(self.num_vehicles):
            cur = torch.full((batch_size,), v, device=rec.device, dtype=torch.long)
            active = torch.ones(batch_size, device=rec.device, dtype=torch.bool)

            for _ in range(total_nodes):
                nxt = rec[arange, cur]
                cur_coord = coords[arange, cur]
                nxt_coord = coords[arange, nxt]
                step_len = (cur_coord - nxt_coord).norm(p=2, dim=1)
                route_lengths[:, v] += step_len * active.float()
                cur = nxt
                active = active & (cur != v)
                if not active.any():
                    break

        return route_lengths

    def compute_cost_components(self, batch, rec):

        # check feasibility
        if self.do_assert:
            self.check_feasibility(rec)

        route_lengths = self._get_route_lengths(batch, rec)
        distance = route_lengths.sum(1)
        makespan = route_lengths.max(1)[0]
        return distance, makespan

    def get_costs(self, batch, rec):

        distance, makespan = self.compute_cost_components(batch, rec)
        if self.use_makespan:
            return distance + makespan
        return distance

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MVPDPDataset(*args, **kwargs)


class MVPDPDataset(Dataset):
    def __init__(
        self,
        filename=None,
        size=20,
        num_samples=10000,
        offset=0,
        distribution=None,
        num_vehicles=2,
    ):

        super(MVPDPDataset, self).__init__()

        self.data = []
        self.size = size
        self.num_vehicles = num_vehicles

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl', 'file name error'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [self.make_instance(args) for args in data[offset:offset + num_samples]]

        else:
            self.data = [
                {
                    'loc': torch.FloatTensor(self.size, 2).uniform_(0, 1),
                    'depot': torch.FloatTensor(2).uniform_(0, 1),
                }
                for _ in range(num_samples)
            ]

        self.N = len(self.data)

        # calculate distance matrix
        for i, instance in enumerate(self.data):
            depot = instance['depot'].reshape(1, 2).repeat(self.num_vehicles, 1)
            self.data[i]['coordinates'] = torch.cat((depot, instance['loc']), dim=0)
            del self.data[i]['depot']
            del self.data[i]['loc']
        print(f'{self.N} instances initialized.')

    def make_instance(self, args):
        depot, loc, *args = args
        grid_size = 1
        if len(args) > 0:
            depot_types, customer_types, grid_size = args
        return {
            'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
            'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        }

    def calculate_distance(self, data):
        N_data = data.shape[0]
        dists = torch.zeros((N_data, N_data), dtype=torch.float)
        d1 = -2 * torch.mm(data, data.T)
        d2 = torch.sum(torch.pow(data, 2), dim=1)
        d3 = torch.sum(torch.pow(data, 2), dim=1).reshape(1, -1).T
        dists = d1 + d2 + d3
        dists[dists < 0] = 0
        return torch.sqrt(dists)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]
