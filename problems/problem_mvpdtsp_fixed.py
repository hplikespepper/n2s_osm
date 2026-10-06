from problems.problem_mvpdtsp import MVPDTSP


class MVPDTSPFixedAssignment(MVPDTSP):
    """MVPDTSP variant that preserves each pair's initial vehicle assignment."""

    # Keep the model/result integration identical to the regular MVPDTSP.
    NAME = 'mvpdtsp'

    def get_swap_mask(self, selected_node, visited_order_map, top2=None, rec=None):
        base_mask = super().get_swap_mask(
            selected_node, visited_order_map, top2=top2, rec=rec
        )
        if rec is None:
            return base_mask

        selected = selected_node.view(-1, 1)
        original_vehicle, _ = self.get_vehicle_id_and_order(rec)
        source_vehicle = original_vehicle.gather(1, selected).view(-1, 1, 1)

        rec_removed = self.remove_pair_from_rec(rec, selected)
        target_vehicle, _ = self.get_vehicle_id_and_order(rec_removed)
        wrong_vehicle = target_vehicle.unsqueeze(2) != source_vehicle

        # The base mask already requires the two insertion anchors to belong to
        # the same route, so checking the first anchor is sufficient here.
        return base_mask | wrong_vehicle.expand_as(base_mask)
