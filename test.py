import matplotlib.pyplot as plt
import networkx as nx

# 定义有向图的边关系
edge_index = [
    [0, 1, 2, 2, 1, 5, 5, 7, 8, 9, 9, 11, 11, 13, 7, 15, 16, 15, 18, 18, 20, 16, 22, 24, 25, 26, 27, 27, 26, 0],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 24]
]

# 输出edge_index尺寸
print(len(edge_index[0]), len(edge_index[1]))

# 创建有向图对象
G = nx.DiGraph()

# 添加边
edge_list = list(zip(edge_index[0], edge_index[1]))
G.add_edges_from(edge_list)

plt.figure(figsize=(12, 10))

# 尝试自定义分层布局，减少边交叉（Y轴分组，X轴均匀分布）
pos = {}
num_nodes = max(max(edge_index[0]), max(edge_index[1])) + 1
# 按照第一层(0-6)、第二层(7-15)、第三层(16-23)、第四层(24-30)进行分层
layers = [
    list(range(0, 7)),
    list(range(7, 16)),
    list(range(16, 24)),
    list(range(24, 31))
]
y_step = 2.0
for i, layer in enumerate(layers):
    x_coord = range(len(layer))
    y_coord = -i * y_step
    for j, node in enumerate(layer):
        pos[node] = (j*2, y_coord)  # x间隔拉开保证不重叠

# 绘制有向图
nx.draw(G, pos, with_labels=True, node_color='lightblue', arrowsize=20, node_size=700, font_size=12)

plt.title("有向图结构分层可视化（减少边交叉）", fontsize=16)
plt.tight_layout()
plt.show()