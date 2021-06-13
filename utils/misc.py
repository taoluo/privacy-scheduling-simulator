#! /usr/bin/python
# Calculate weighted max min fair allocations
import copy

def defuse(event):
    def set_defused(evt):
        evt.defused = True

    if not event.processed:
        event.callbacks.append(set_defused)


def max_min_fair_allocation(desired, capacity):
    # assume equal weight
    # for u,w in weights.items():
    #     break
    # for u1, w1 in weights.items():
    #     assert w==w1
    if sum(desired.values()) <= capacity:
        return desired
    else:
        equal_share = capacity / len(desired)
        if min(desired.values()) > equal_share:
            return {u:equal_share for u in desired}
        else:
            subdesired = {}
            partial_min_max_share = {}

            for u, d in desired.items():
                if d <= equal_share:
                    partial_min_max_share[u] = d
                else:
                    subdesired[u] = d
            # partial_min_max_share = {u:equal_share for u,d in desired.items if d >= equal_share}
            partial_share_sum = sum(partial_min_max_share.values())
            partial_min_max_share.update(max_min_fair_allocation(subdesired, capacity - partial_share_sum))
            return partial_min_max_share

# # slower, borrow from https://github.com/anirudhSK/cell-codel/blob/master/schism/utils/max-min-fairness.py
# def max_min_fair_allocation2(desired, weights, capacity):
#     # global final_share, user, capacity, use_id
#     # OUTPUT: Allocations.
#     final_share = dict()
#     # book-keeping
#     current_share = dict()
#     for user in desired:
#         current_share[user] = 0
#     # capacity = total_capacity
#     # core algm
#     while (len(weights) > 0) and (capacity > 0):
#         total_shares = sum(weights.values())
#         unit_share = capacity / total_shares;
#         user_list = list(weights.keys())
#         for user in user_list:
#             fair_share = unit_share * weights[user];
#             current_share[user] += fair_share
#             if current_share[user] >= desired[user]:
#                 spare_capacity = (current_share[user] - desired[user]);
#                 final_share[user] = desired[user]
#                 current_share.pop(user)
#                 weights.pop(user)
#                 capacity = capacity + spare_capacity - fair_share;
#             else:
#                 capacity = capacity - fair_share;
#     # finalize allocations
#     for user in current_share:
#         final_share[user] = current_share[user]
#     return final_share
#

if __name__ == '__main__':
    # INPUT : Desired Rates, Weights, capacity
    desired = {0: 0.736000, 1: 0.72, 2: 0.108, 3: 0.5144, 4: 0.618, 5: 0.0216, 6: 0.252, 7: 0.02088, 8: 0.0324,
               9: 0.036}
    total_capacity = 1000715

    for user in desired:
        desired[user] *= total_capacity
    weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
    for weight in weights:
        weights[weight] **= (+0.5);
    norm_factor = sum(weights.values())
    for weight in weights:
        weights[weight] /= norm_factor
    weight_copy = copy.deepcopy(weights)

    final_share2 = max_min_fair_allocation(desired, total_capacity)
    # # final_share = max_min_fair_allocation2(desired, weights, total_capacity)
    #
    # # print
    # for user_id in final_share:
    #     assert -1e-5 < (final_share2[user_id] - final_share[user_id] ) < 1e-5
    #     bottleneck = True
    #     if final_share[user_id] == desired[user_id]:
    #         bottleneck = False
    #     print("user ", user_id, "weight %.5f" % weight_copy[user_id], "desired %7.0f" % desired[user_id],
    #           "allocated %7.0f" % final_share[user_id], " bottleneck ", bottleneck)
    #
    # # aggregate stats
    # print("=============TOTAL STATS====================")
    # print("capacity", total_capacity)
    # print("demand", sum(desired.values()))
    # print("amount allocated", sum(final_share.values()))

    pass
