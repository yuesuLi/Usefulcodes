import motmetrics as mm
import numpy as np

metrics = list(mm.metrics.motchallenge_metrics)  # 即支持的所有metrics的名字列表
"""
['idf1', 'idp', 'idr', 'recall', 'precision', 'num_unique_objects', 'mostly_tracked', 
 'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches', 
 'num_fragmentations', 'mota', 'motp', 'num_transfer', 'num_ascend', 'num_migrate']
"""

acc = mm.MOTAccumulator(auto_id=True)  #创建accumulator
# print('acc:', acc._indices)

# 用第一帧填充该accumulator
acc.update(
    [1, 2],                     # Ground truth objects in this frame
    [1, 2, 3],                  # Detector hypotheses in this frame
    [
        [0.1, np.nan, 0.3],     # Distances from object 1 to hypotheses 1, 2, 3
        [0.5,  0.2,   0.3]      # Distances from object 2 to hypotheses 1, 2, 3
    ]
)

# print('acc:', acc._indices)

# 查看该帧的事件
# print(acc.events) # a pandas DataFrame containing all events
"""
                Type  OId HId    D
FrameId Event
0       0        RAW    1   1  0.1
        1        RAW    1   2  NaN
        2        RAW    1   3  0.3
        3        RAW    2   1  0.5
        4        RAW    2   2  0.2
        5        RAW    2   3  0.3
        6      MATCH    1   1  0.1
        7      MATCH    2   2  0.2
        8         FP  NaN   3  NaN
"""

# 只查看MOT事件，不查看RAW
# print(acc.mot_events) # a pandas DataFrame containing MOT only events
"""
                Type  OId HId    D
FrameId Event
0       6      MATCH    1   1  0.1
        7      MATCH    2   2  0.2
        8         FP  NaN   3  NaN
"""



# 继续填充下一帧
frameid = acc.update(
    [5, 6, 7, 4],  # GT
    [1],     # hypotheses
    [
        [0.2],
        [0.4],
        [0.6],
        [0.1]
    ]
)

# print('acc:', acc._indices)
# print('frameid:', frameid)
# print(acc.mot_events.loc[frameid])
"""
        Type OId  HId    D
Event
2      MATCH   1    1  0.2
3       MISS   2  NaN  NaN
"""

# 继续填充下一帧
frameid = acc.update(
    [1, 2, 3], # GT
    [1, 3], # hypotheses
    [
        [0.6, 0.2],
        [0.1, 0.6],
        [0.05, 0.8]
    ]
)
# print('frameid:', frameid)
# print(acc.mot_events.loc[frameid])
"""
         Type OId HId    D
Event
4       MATCH   1   1  0.6
5      SWITCH   2   3  0.6
"""

mh = mm.metrics.create()

# 打印单个accumulator   metrics=['num_frames', 'mota', 'motp']
summary = mh.compute(acc,
                     metrics=mm.metrics.motchallenge_metrics.append('num_frames'), # 一个list，里面装的是想打印的一些度量
                     name='acc') # 起个名
# print('summary:\n', summary)
strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)

print('strsummary:\n', strsummary)

"""
     num_frames  mota  motp
acc           3   0.5  0.34
"""

# 自定义显示格式
# strsummary = mm.io.render_summary(
#     summary,
#     formatters={'mota' : '{:.2%}'.format},  # 将MOTA的格式改为百分数显示
#     namemap={'mota': 'MOTA', 'motp' : 'MOTP'}  # 将列名改为大写
# )
# print(strsummary)
"""
      num_frames   MOTA      MOTP
full           3 50.00%  0.340000
part           2 50.00%  0.166667
"""

# mh模块中有内置的显示格式
summary = mh.compute_many([acc, acc.events.loc[0:1]],
                          metrics=mm.metrics.motchallenge_metrics.append('num_frames'),
                          names=['full', 'part'])

# print('num:', summary['idf1']['full'])

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)

print(strsummary)
