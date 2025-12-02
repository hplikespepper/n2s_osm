from torch.utils.data import Dataset
import torch
import pickle
import os
import numpy as np

class PDTSP_2V(object):
    """
    Pickup and Delivery TSP with 2 Vehicles (PDTSP_2V)
    
    Two vehicles share the same depot. Pickup-delivery pairs are divided based on 
    the y-coordinate of pickup locations:
    - Vehicle 1: serves pickup nodes with y >= 0.5 (upper half)
    - Vehicle 2: serves pickup nodes with y < 0.5 (lower half)
    
    Both vehicles can move freely across the entire map, but each vehicle can only
    pick up items from its assigned region (though delivery can be anywhere).
    
    Objective: minimize total tour length of both vehicles combined.
    """

    NAME = 'pdtsp_2v'
    
    def __init__(self, p_size, init_val_met='greedy', with_assert=False):
        
        self.size = p_size          # the number of customer nodes (excluding depot)
        self.do_assert = with_assert
        self.init_val_met = init_val_met
        self.state = 'eval'
        print(f'PDTSP_2V with {self.size} nodes (2 vehicles).', 
              ' Do assert:', with_assert)
    
    def input_feature_encoding(self, batch):
        """Return node coordinates for network input"""
        return batch['coordinates']
    
    def split_by_y_coordinate(self, batch):
        """
        Split pickup-delivery pairs based on pickup y-coordinate.
        
        Args:
            batch: dictionary containing 'coordinates' [bs, gs+1, 2]
                   Format: [depot, pickup_1, ..., pickup_n/2, delivery_1, ..., delivery_n/2]
        
        Returns:
            vehicle_assignment: [bs, gs+1] tensor with values:
                - 0 for depot
                - 1 for vehicle 1 nodes (upper half pickups and their deliveries)
                - 2 for vehicle 2 nodes (lower half pickups and their deliveries)
            upper_count: [bs] number of pickup pairs assigned to vehicle 1
            lower_count: [bs] number of pickup pairs assigned to vehicle 2
        """
        bs, gs, _ = batch['coordinates'].size()
        half_size = (gs - 1) // 2  # Number of pickup-delivery pairs
        
        # Extract pickup coordinates (indices 1 to half_size+1)
        pickup_coords = batch['coordinates'][:, 1:half_size+1, :]  # [bs, half_size, 2]
        pickup_y = pickup_coords[:, :, 1]  # [bs, half_size]
        
        # Assign to vehicles based on y-coordinate threshold (0.5)
        # 1 = upper half (vehicle 1), 2 = lower half (vehicle 2)
        pickup_assignment = torch.where(pickup_y >= 0.5, 
                                       torch.ones_like(pickup_y).long(), 
                                       torch.ones_like(pickup_y).long() * 2)  # [bs, half_size]
        
        # Build full assignment tensor [bs, gs+1]
        vehicle_assignment = torch.zeros(bs, gs, dtype=torch.long, device=batch['coordinates'].device)
        vehicle_assignment[:, 0] = 0  # depot
        
        # Assign pickups
        vehicle_assignment[:, 1:half_size+1] = pickup_assignment
        
        # Assign corresponding deliveries (same vehicle as pickup)
        vehicle_assignment[:, half_size+1:] = pickup_assignment
        
        # Count nodes per vehicle
        upper_count = (pickup_assignment == 1).sum(dim=1)  # [bs]
        lower_count = (pickup_assignment == 2).sum(dim=1)  # [bs]
        
        return vehicle_assignment, upper_count, lower_count
    
    def get_visited_order_map(self, visited_time):
        """
        Create a visitation order map for feasibility checking.
        
        Args:
            visited_time: [bs, gs] tensor indicating when each node was visited
        
        Returns:
            visited_order_map: [bs, gs, gs] boolean tensor where [i,j,k]=True 
                              means node j was visited before node k in instance i
        """
        bs, gs = visited_time.size()
        visited_time = visited_time % gs
        
        return visited_time.view(bs, gs, 1) > visited_time.view(bs, 1, gs)
    
    def get_real_mask(self, selected_node, visited_order_map, vehicle_assignment=None):
        """
        Get mask for valid swap/insert positions considering precedence constraints.
        
        For PDTSP: delivery must come after its pickup.
        For PDTSP_2V: also ensure nodes belong to correct vehicle.
        
        Args:
            selected_node: [bs, 1] the node being relocated
            visited_order_map: [bs, gs, gs] visitation order
            vehicle_assignment: [bs, gs] optional vehicle assignment (1 or 2)
        
        Returns:
            mask: [bs, gs, gs] valid reinsertion positions
        """
        bs, gs, _ = visited_order_map.size()
        
        mask = visited_order_map.clone()
        
        # Standard PDTSP constraints
        mask[torch.arange(bs), selected_node.view(-1)] = True
        mask[torch.arange(bs), selected_node.view(-1) + gs // 2] = True
        mask[torch.arange(bs), :, selected_node.view(-1)] = True
        mask[torch.arange(bs), :, selected_node.view(-1) + gs // 2] = True
        
        # Additional vehicle constraint for PDTSP_2V
        if vehicle_assignment is not None:
            # vehicle_assignment has size [bs, gs] where gs includes depot + all customer nodes
            # visited_order_map has size [bs, gs, gs] matching visited_time which has same gs
            # selected_node is 0-indexed from [0, gs-1]
            
            # Get vehicle ID of selected node
            # selected_node corresponds directly to indices in vehicle_assignment
            selected_vehicle = vehicle_assignment[torch.arange(bs), selected_node.view(-1)]  # [bs]
            
            # Create vehicle match mask for all nodes: True where vehicle ID matches selected vehicle
            vehicle_match = vehicle_assignment == selected_vehicle.view(bs, 1)  # [bs, gs]
            
            # Broadcast to [bs, gs, gs] and apply
            # A position [i,j,k] is invalid if either node j or k belongs to wrong vehicle
            vehicle_mask_row = vehicle_match.unsqueeze(2).expand(bs, gs, gs)  # [bs, gs, gs]
            vehicle_mask_col = vehicle_match.unsqueeze(1).expand(bs, gs, gs)  # [bs, gs, gs]
            
            # Set to True (invalid) where vehicle doesn't match
            mask = mask | (~vehicle_mask_row) | (~vehicle_mask_col)
        
        return mask
    
    def get_initial_solutions(self, batch, val_m=1):
        """
        Generate initial solutions for both vehicles.
        
        The solution is represented as a unified tour that includes both vehicles' routes.
        Format: [depot, v1_nodes..., depot, v2_nodes..., depot]
        But represented in the standard rec format where rec[i] = next node after node i.
        
        Args:
            batch: data batch with coordinates
            val_m: number of augmentations (not used in initial solution generation)
        
        Returns:
            rec: [bs, gs+1] initial solution in successor representation
        """
        batch_size = batch['coordinates'].size(0)
        
        # Get vehicle assignments
        vehicle_assignment, upper_count, lower_count = self.split_by_y_coordinate(batch)
        
        # Store in batch for later use
        batch['vehicle_assignment'] = vehicle_assignment
        batch['upper_count'] = upper_count
        batch['lower_count'] = lower_count
        
        def get_solution(method):
            half_size = self.size // 2
            
            if method == 'random':
                return self._get_random_solution(batch, batch_size, half_size, 
                                                 vehicle_assignment, upper_count, lower_count)
            
            elif method == 'greedy':
                return self._get_greedy_solution(batch, batch_size, half_size,
                                                vehicle_assignment, upper_count, lower_count)
            
            elif method == 'p2d':
                return self._get_p2d_solution(batch, batch_size, half_size,
                                             vehicle_assignment, upper_count, lower_count)
            
            else:
                raise NotImplementedError(f"Initial solution method '{method}' not implemented")
        
        return get_solution(self.init_val_met).clone()
    
    def _get_random_solution(self, batch, batch_size, half_size, vehicle_assignment, 
                            upper_count, lower_count):
        """Generate random initial solution respecting vehicle assignments"""
        rec = torch.zeros(batch_size, self.size + 1, dtype=torch.long)
        
        for b in range(batch_size):
            # Separate nodes by vehicle
            v1_pickups = []
            v2_pickups = []
            
            for i in range(1, half_size + 1):
                if vehicle_assignment[b, i] == 1:
                    v1_pickups.append(i)
                else:
                    v2_pickups.append(i)
            
            # Build tour for each vehicle
            v1_tour = self._build_random_vehicle_tour(v1_pickups, half_size)
            v2_tour = self._build_random_vehicle_tour(v2_pickups, half_size)
            
            # Merge into unified solution
            combined_tour = [0] + v1_tour + v2_tour
            
            # Convert to successor representation
            for i in range(len(combined_tour) - 1):
                rec[b, combined_tour[i]] = combined_tour[i + 1]
            rec[b, combined_tour[-1]] = 0  # Return to depot
        
        return rec
    
    def _build_random_vehicle_tour(self, pickups, half_size):
        """Build a random feasible tour for one vehicle"""
        if len(pickups) == 0:
            return []
        
        tour = []
        available_pickups = pickups.copy()
        picked_not_delivered = set()  # Items picked but not yet delivered
        
        while available_pickups or picked_not_delivered:
            # Decide whether to pick up or deliver
            can_pickup = len(available_pickups) > 0
            can_deliver = len(picked_not_delivered) > 0
            
            if can_pickup and (not can_deliver or np.random.rand() < 0.5):
                # Pick up
                idx = np.random.randint(len(available_pickups))
                pickup = available_pickups.pop(idx)
                tour.append(pickup)
                picked_not_delivered.add(pickup)
            elif can_deliver:
                # Deliver a picked item
                deliverable = list(picked_not_delivered)
                pickup = deliverable[np.random.randint(len(deliverable))]
                tour.append(pickup + half_size)
                picked_not_delivered.remove(pickup)
        
        return tour
    
    def _get_greedy_solution(self, batch, batch_size, half_size, vehicle_assignment,
                            upper_count, lower_count):
        """Generate greedy nearest-neighbor solution for both vehicles"""
        coordinates = batch['coordinates']  # [bs, gs, 2]
        rec = torch.zeros(batch_size, self.size + 1, dtype=torch.long)
        
        for b in range(batch_size):
            # Separate nodes by vehicle
            v1_nodes = []
            v2_nodes = []
            
            for i in range(1, half_size + 1):
                if vehicle_assignment[b, i] == 1:
                    v1_nodes.append(i)
                else:
                    v2_nodes.append(i)
            
            # Build greedy tour for each vehicle
            v1_tour = self._build_greedy_vehicle_tour(coordinates[b], v1_nodes, half_size)
            v2_tour = self._build_greedy_vehicle_tour(coordinates[b], v2_nodes, half_size)
            
            # Merge into unified solution
            combined_tour = [0] + v1_tour + v2_tour
            
            # Convert to successor representation
            for i in range(len(combined_tour) - 1):
                rec[b, combined_tour[i]] = combined_tour[i + 1]
            rec[b, combined_tour[-1]] = 0
        
        return rec
    
    def _build_greedy_vehicle_tour(self, coordinates, pickups, half_size):
        """Build greedy nearest-neighbor tour for one vehicle"""
        if len(pickups) == 0:
            return []
        
        tour = []
        current_pos = 0  # Start at depot
        available_pickups = set(pickups)
        picked_items = set()
        
        while available_pickups or picked_items:
            candidates = []
            
            # Can pick up any available pickup
            for p in available_pickups:
                dist = torch.norm(coordinates[current_pos] - coordinates[p], p=2)
                candidates.append((dist.item(), p, 'pickup'))
            
            # Can deliver any picked item
            for p in picked_items:
                d = p + half_size
                dist = torch.norm(coordinates[current_pos] - coordinates[d], p=2)
                candidates.append((dist.item(), d, 'delivery'))
            
            if not candidates:
                break
            
            # Choose nearest
            candidates.sort()
            dist, next_node, action_type = candidates[0]
            
            tour.append(next_node)
            current_pos = next_node
            
            if action_type == 'pickup':
                available_pickups.remove(next_node)
                picked_items.add(next_node)
            else:
                pickup_node = next_node - half_size
                picked_items.remove(pickup_node)
        
        return tour
    
    def _get_p2d_solution(self, batch, batch_size, half_size, vehicle_assignment,
                         upper_count, lower_count):
        """Generate pickup-immediately-deliver solution"""
        rec = torch.zeros(batch_size, self.size + 1, dtype=torch.long)
        
        for b in range(batch_size):
            tour = [0]  # Start at depot
            
            # Vehicle 1 nodes (upper half)
            for i in range(1, half_size + 1):
                if vehicle_assignment[b, i] == 1:
                    tour.append(i)  # pickup
                    tour.append(i + half_size)  # immediate delivery
            
            # Vehicle 2 nodes (lower half)
            for i in range(1, half_size + 1):
                if vehicle_assignment[b, i] == 2:
                    tour.append(i)  # pickup
                    tour.append(i + half_size)  # immediate delivery
            
            # Convert to successor representation
            for i in range(len(tour) - 1):
                rec[b, tour[i]] = tour[i + 1]
            rec[b, tour[-1]] = 0  # Return to depot
        
        return rec
    
    def step(self, batch, rec, exchange, pre_bsf, action_record):
        """
        Execute one improvement step (node removal and reinsertion).
        
        Args:
            batch: data batch
            rec: current solution [bs, gs+1]
            exchange: [bs, 3] action (selected_node, first_pos, second_pos)
            pre_bsf: [bs, 2] previous best-so-far costs
            action_record: list of action history tensors
        
        Returns:
            next_state: improved solution
            reward: improvement achieved
            new_bsf: updated best-so-far costs
            action_record: updated action history
        """
        bs, gs = rec.size()
        pre_bsf = pre_bsf.view(bs, -1)
        
        # Update action record
        cur_vec = action_record.pop(0) * 0.
        cur_vec[torch.arange(bs), exchange[:, 0]] = 1.
        action_record.append(cur_vec)
        
        selected = exchange[:, 0].view(bs, 1)
        first = exchange[:, 1].view(bs, 1)
        second = exchange[:, 2].view(bs, 1)
        
        # Execute insertion with vehicle constraint check
        next_state = self.insert_star(rec, selected + 1, first, second, 
                                      batch.get('vehicle_assignment', None))
        
        # Calculate new objective
        new_obj = self.get_costs(batch, next_state)
        
        # Update best-so-far
        now_bsf = torch.min(torch.cat((new_obj[:, None], pre_bsf[:, -1, None]), -1), -1)[0]
        
        # Calculate reward
        reward = pre_bsf[:, -1] - now_bsf
        
        return next_state, reward, torch.cat((new_obj[:, None], now_bsf[:, None]), -1), action_record
    
    def insert_star(self, solution, pair_index, first, second, vehicle_assignment=None):
        """
        Remove a pickup-delivery pair and reinsert at new positions.
        
        Args:
            solution: [bs, gs] current solution in successor format
            pair_index: [bs, 1] pickup node to relocate (1-indexed)
            first: [bs, 1] new position for pickup
            second: [bs, 1] new position for delivery
            vehicle_assignment: [bs, gs] vehicle assignments (optional)
        
        Returns:
            rec: [bs, gs] modified solution
        """
        rec = solution.clone()
        bs, gs = rec.size()
        half_size = (gs - 1) // 2
        
        # Validate vehicle assignment if provided
        if vehicle_assignment is not None and self.do_assert:
            # Check that relocated nodes stay within their vehicle's domain
            pair_vehicle = vehicle_assignment[torch.arange(bs), pair_index.view(-1)]
            first_vehicle = vehicle_assignment[torch.arange(bs), first.view(-1)]
            second_vehicle = vehicle_assignment[torch.arange(bs), second.view(-1)]
            
            # Allow depot (vehicle_assignment == 0) as valid position
            valid_first = (first_vehicle == pair_vehicle) | (first_vehicle == 0)
            valid_second = (second_vehicle == pair_vehicle) | (second_vehicle == 0)
            
            assert valid_first.all(), "First position violates vehicle assignment"
            assert valid_second.all(), "Second position violates vehicle assignment"
        
        # Fix connection for pairing pickup node (remove from tour)
        argsort = rec.argsort()
        pre_pairfirst = argsort.gather(1, pair_index)
        post_pairfirst = rec.gather(1, pair_index)
        rec.scatter_(1, pre_pairfirst, post_pairfirst)
        rec.scatter_(1, pair_index, pair_index)
        
        # Fix connection for pairing delivery node (remove from tour)
        argsort = rec.argsort()
        pre_pairsecond = argsort.gather(1, pair_index + half_size)
        post_pairsecond = rec.gather(1, pair_index + half_size)
        rec.scatter_(1, pre_pairsecond, post_pairsecond)
        
        # Insert delivery at new position
        post_second = rec.gather(1, second)
        rec.scatter_(1, second, pair_index + half_size)
        rec.scatter_(1, pair_index + half_size, post_second)
        
        # Insert pickup at new position
        post_first = rec.gather(1, first)
        rec.scatter_(1, first, pair_index)
        rec.scatter_(1, pair_index, post_first)
        
        return rec
    
    def check_feasibility(self, rec, vehicle_assignment=None):
        """
        Check if solution is feasible.
        
        Constraints:
        1. All nodes visited exactly once
        2. Delivery must come after pickup
        3. Each vehicle only serves its assigned nodes
        
        Args:
            rec: [bs, gs] solution
            vehicle_assignment: [bs, gs] vehicle assignments
        """
        p_size = self.size
        bs = rec.size(0)
        
        # Check 1: All nodes visited
        assert (
            (torch.arange(p_size + 1, out=rec.new())).view(1, -1).expand_as(rec) == 
            rec.sort(1)[0]
        ).all(), "Not visiting all nodes"
        
        # Calculate visited time
        visited_time = torch.zeros((bs, p_size), device=rec.device)
        pre = torch.zeros((bs,), device=rec.device).long()
        
        for i in range(p_size):
            visited_time[torch.arange(bs), rec[torch.arange(bs), pre] - 1] = i + 1
            pre = rec[torch.arange(bs), pre]
        
        # Check 2: Pickup before delivery
        assert (
            visited_time[:, 0:p_size // 2] < 
            visited_time[:, p_size // 2:]
        ).all(), "Delivering without pickup"
        
        # Check 3: Vehicle assignment (if provided)
        if vehicle_assignment is not None:
            # Build actual tour from rec
            tours = []
            for b in range(bs):
                tour = [0]
                current = 0
                for _ in range(p_size):
                    current = rec[b, current].item()
                    if current == 0:
                        break
                    tour.append(current)
                tours.append(tour)
            
            # Check vehicle constraint: 
            # - Pickup and delivery of same pair must be by same vehicle
            # - Within a vehicle's route segment, nodes should belong to that vehicle
            # But transitions between vehicles (v1 -> depot -> v2 or v1 -> v2) are allowed in unified representation
            for b in range(bs):
                # Check pickup-delivery pairing
                for i in range(1, p_size // 2 + 1):
                    pickup_vehicle = vehicle_assignment[b, i].item()
                    delivery_vehicle = vehicle_assignment[b, i + p_size // 2].item()
                    
                    assert pickup_vehicle == delivery_vehicle, \
                        f"Pickup-delivery pair {i} has mismatched vehicles: p{pickup_vehicle} vs d{delivery_vehicle}"
    
    def get_swap_mask(self, selected_node, visited_order_map, vehicle_assignment=None):
        """Get mask for valid swap positions"""
        return self.get_real_mask(selected_node, visited_order_map, vehicle_assignment)
    
    def get_costs(self, batch, rec):
        """
        Calculate total tour length for both vehicles.
        
        CRITICAL: For PDTSP_2V, the unified successor representation chains v1 and v2
        sequentially (depot -> v1_path -> v2_path -> depot). However, the edge between
        v1's last node and v2's first node should NOT be counted directly. Instead,
        both vehicles should return to depot independently:
        - Vehicle 1: depot -> v1_path -> depot
        - Vehicle 2: depot -> v2_path -> depot
        
        This function detects vehicle transitions and correctly handles them.
        
        Args:
            batch: data batch with coordinates and vehicle_assignment
            rec: [bs, gs+1] solution in successor representation
        
        Returns:
            length: [bs] total tour length (vehicle 1 + vehicle 2 both from/to depot)
        """
        batch_size, size = rec.size()
        
        # Check feasibility
        if self.do_assert:
            self.check_feasibility(rec, batch.get('vehicle_assignment', None))
        
        vehicle_assignment = batch.get('vehicle_assignment', None)
        
        if vehicle_assignment is None:
            # Standard PDTSP: simple path length calculation
            d1 = batch['coordinates'].gather(1, rec.long().unsqueeze(-1).expand(batch_size, size, 2))
            d2 = batch['coordinates']
            length = (d1 - d2).norm(p=2, dim=2).sum(1)
        else:
            # PDTSP_2V: handle vehicle transitions correctly
            lengths = []
            
            for b in range(batch_size):
                coords = batch['coordinates'][b]
                total_dist = 0.0
                current = 0  # Start at depot
                
                for step in range(size):
                    next_node = rec[b, current].item()
                    
                    if next_node == 0:
                        # Explicit return to depot
                        total_dist += torch.norm(coords[current] - coords[0], p=2).item()
                        break
                    
                    curr_vehicle = vehicle_assignment[b, current].item()
                    next_vehicle = vehicle_assignment[b, next_node].item()
                    
                    # Detect vehicle transition: v1_node -> v2_node (or vice versa)
                    is_vehicle_transition = (curr_vehicle != 0 and next_vehicle != 0 and 
                                            curr_vehicle != next_vehicle)
                    
                    if is_vehicle_transition:
                        # Vehicle transition detected!
                        # Instead of direct edge current -> next_node,
                        # we compute: current -> depot + depot -> next_node
                        dist_to_depot = torch.norm(coords[current] - coords[0], p=2).item()
                        dist_from_depot = torch.norm(coords[0] - coords[next_node], p=2).item()
                        total_dist += (dist_to_depot + dist_from_depot)
                    else:
                        # Normal edge (within same vehicle or involving depot)
                        total_dist += torch.norm(coords[current] - coords[next_node], p=2).item()
                    
                    current = next_node
                
                lengths.append(total_dist)
            
            length = torch.tensor(lengths, device=batch['coordinates'].device)
        
        return length
    
    def get_costs_separate(self, batch, rec):
        """
        Calculate tour length for each vehicle separately.
        
        Uses the SAME logic as get_costs() to handle vehicle transitions,
        then attributes each cost to the appropriate vehicle.
        
        Returns:
            v1_length: [bs] tour length of vehicle 1 (depot -> v1_path -> depot)
            v2_length: [bs] tour length of vehicle 2 (depot -> v2_path -> depot)
            total_length: [bs] total tour length (should EXACTLY match get_costs)
        """
        batch_size = rec.size(0)
        vehicle_assignment = batch.get('vehicle_assignment', None)
        
        if vehicle_assignment is None:
            # If no assignment provided, calculate it
            vehicle_assignment, _, _ = self.split_by_y_coordinate(batch)
        
        v1_lengths = []
        v2_lengths = []
        
        for b in range(batch_size):
            coords = batch['coordinates'][b]
            v1_dist = 0.0
            v2_dist = 0.0
            current = 0  # Start at depot
            
            for step in range(self.size + 1):
                next_node = rec[b, current].item()
                
                if next_node == 0:
                    # Return to depot - assign to vehicle of current node
                    if current != 0:
                        dist = torch.norm(coords[current] - coords[0], p=2).item()
                        curr_vehicle = vehicle_assignment[b, current].item()
                        if curr_vehicle == 1:
                            v1_dist += dist
                        elif curr_vehicle == 2:
                            v2_dist += dist
                    break
                
                curr_vehicle = vehicle_assignment[b, current].item()
                next_vehicle = vehicle_assignment[b, next_node].item()
                
                # Detect vehicle transition
                is_vehicle_transition = (curr_vehicle != 0 and next_vehicle != 0 and 
                                        curr_vehicle != next_vehicle)
                
                if is_vehicle_transition:
                    # Vehicle transition: current -> depot -> next_node
                    # Assign depot->current edge to current's vehicle
                    dist_to_depot = torch.norm(coords[current] - coords[0], p=2).item()
                    if curr_vehicle == 1:
                        v1_dist += dist_to_depot
                    elif curr_vehicle == 2:
                        v2_dist += dist_to_depot
                    
                    # Assign depot->next edge to next's vehicle
                    dist_from_depot = torch.norm(coords[0] - coords[next_node], p=2).item()
                    if next_vehicle == 1:
                        v1_dist += dist_from_depot
                    elif next_vehicle == 2:
                        v2_dist += dist_from_depot
                else:
                    # Normal edge - assign to appropriate vehicle
                    dist = torch.norm(coords[current] - coords[next_node], p=2).item()
                    
                    if curr_vehicle == 0:
                        # From depot: assign to destination vehicle
                        if next_vehicle == 1:
                            v1_dist += dist
                        elif next_vehicle == 2:
                            v2_dist += dist
                    elif next_vehicle == 0:
                        # To depot: assign to source vehicle
                        if curr_vehicle == 1:
                            v1_dist += dist
                        elif curr_vehicle == 2:
                            v2_dist += dist
                    else:
                        # Within vehicle: assign to that vehicle
                        if curr_vehicle == 1 and next_vehicle == 1:
                            v1_dist += dist
                        elif curr_vehicle == 2 and next_vehicle == 2:
                            v2_dist += dist
                
                current = next_node
            
            v1_lengths.append(v1_dist)
            v2_lengths.append(v2_dist)
        
        v1_lengths = torch.tensor(v1_lengths, device=batch['coordinates'].device)
        v2_lengths = torch.tensor(v2_lengths, device=batch['coordinates'].device)
        total_lengths = v1_lengths + v2_lengths
        
        return v1_lengths, v2_lengths, total_lengths
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        return PDTSP2VDataset(*args, **kwargs)


class PDTSP2VDataset(Dataset):
    """
    Dataset for PDTSP_2V problem.
    
    Uses the same data format as standard PDTSP, but nodes will be automatically
    assigned to vehicles based on pickup y-coordinates during problem solving.
    """
    
    def __init__(self, filename=None, size=20, num_samples=10000, offset=0, distribution=None):
        super(PDTSP2VDataset, self).__init__()
        
        self.data = []
        self.size = size
        
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl', 'File must be .pkl format'
            
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.data = [self.make_instance(args) for args in data[offset:offset+num_samples]]
        else:
            # Generate random instances
            self.data = [{
                'loc': torch.FloatTensor(self.size, 2).uniform_(0, 1),
                'depot': torch.FloatTensor(2).uniform_(0, 1)
            } for i in range(num_samples)]
        
        self.N = len(self.data)
        
        # Build coordinate format
        for i, instance in enumerate(self.data):
            self.data[i]['coordinates'] = torch.cat(
                (instance['depot'].reshape(1, 2), instance['loc']), dim=0
            )
            del self.data[i]['depot']
            del self.data[i]['loc']
        
        print(f'{self.N} PDTSP_2V instances initialized.')
    
    def make_instance(self, args):
        """Convert loaded data to instance format"""
        depot, loc, *args = args
        grid_size = 1
        if len(args) > 0:
            depot_types, customer_types, grid_size = args
        return {
            'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
            'depot': torch.tensor(depot, dtype=torch.float) / grid_size
        }
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        return self.data[idx]
