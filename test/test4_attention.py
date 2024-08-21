# Definition for singly-linked list.
import numpy as np



# def speed_direction_batch(dets, tracks):
#     tracks = tracks[..., np.newaxis]
#     CX1, CY1 = (dets[:,0] + dets[:,2])/2.0, (dets[:,1]+dets[:,3])/2.0
#     CX2, CY2 = (tracks[:,0] + tracks[:,2]) /2.0, (tracks[:,1]+tracks[:,3])/2.0
#     dx = CX1 - CX2
#     dy = CY1 - CY2
#     norm = np.sqrt(dx**2 + dy**2) + 1e-6
#     dx = dx / norm
#     dy = dy / norm
#     return dy, dx # size: num_track x num_det
#
# detections = np.array([[500, 500, 700, 700, 0.92],
#                       [550, 450, 750, 760, 0.95],
#                       [400, 350, 600, 800, 0.91],
#                       [300, 200, 400, 500, 0.94]])
#
# previous_obs = np.array([[505, 490, 710, 720, 0.92],
#                         [550, 440, 740, 780, 0.95],
#                         [405, 320, 600, 870, 0.91]])
#
#
# velocities = np.array([[0.8, 0.6],
#                       [0.89, 0.447],
#                       [0.832, 0.5547]])
#
# Y, X = speed_direction_batch(detections, previous_obs)
# inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
# inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
# inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
# diff_angle_cos = inertia_X * X + inertia_Y * Y
# diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
# diff_angle = np.arccos(diff_angle_cos)
# diff_angle = (np.pi /2.0 - np.abs(diff_angle)) / np.pi

def linear_assignment(cost_matrix):
    try:
        import lap
        print('*******************lap*****************************')
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        print('******************linear_sum_assignment*****************************')
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


Matrix1 = np.array([[4, 5, 9],
                    [6, 2, 4],
                    [8, 3, 6],
                    [1, 2, 3]])  # 一个4*3的非方阵

Matrix2 = np.array([[3, 5],
                    [6, 1],
                    [6, 8],
                    [1, 2]])  # 一个4*3的非方阵



matched_indices_r = linear_assignment(Matrix1)
matched_indices_r = list(matched_indices_r)
matched_indices_r.append([8, 10])
matched_indices_r = np.array(matched_indices_r)

matched_indices_c = linear_assignment(Matrix2)
matched_indices_c = list(matched_indices_c)
matched_indices_c.append([0, 4])
matched_indices_c.append([2, 2])
matched_indices_c.append([6, 8])
matched_indices_c = np.array(matched_indices_c)
matched_indices_r = np.empty(shape=(0,2))
print('matched_indices_r:', matched_indices_r)
print('matched_indices_c:', matched_indices_c)

len_r = matched_indices_r.shape[0]
len_c = matched_indices_c.shape[0]

matched_indices = np.ones((len_r + len_c, 3)) * -1


if len_r > 0 and len_c > 0:
    for i in range(len_r):
        for j in range(len_c):
            if matched_indices_r[i, 0] == matched_indices_c[j, 0]:
                matched_indices[i] = [matched_indices_r[i, 0], matched_indices_r[i, 1], matched_indices_c[j, 1]]
                break
            elif j == len_c-1:
                matched_indices[i] = [matched_indices_r[i, 0], matched_indices_r[i, 1], -1]

    idx = matched_indices.shape[0] - 1
    for i in range(len_c):
        for j in range(len_r):
            if matched_indices_c[i, 0] == matched_indices_r[j, 0]:
                break
            elif j == len_r - 1:
                matched_indices[idx] = [matched_indices_c[i, 0], -1, matched_indices_c[i, 1]]
                idx -= 1
elif len_r == 0:
    for i in range(len_c):
        matched_indices[i] = [matched_indices_c[i, 0], -1, matched_indices_c[i, 1]]
elif len_c == 0:
    for i in range(len_r):
        matched_indices[i] = [matched_indices_r[i, 0], matched_indices_r[i, 1], -1]


# for i in range(matched_indices1.shape[0]):
print('matched_indices:', matched_indices)

print('done')
