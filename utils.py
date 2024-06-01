import numpy as np
from shapely.geometry import Polygon
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
from shapely.ops import unary_union, polygonize
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
import json

def create_polygon(points):
    # 确保点的数量至少为三个以形成一个多边形
    if len(points) < 3:
        raise ValueError("At least three points are required to form a polygon.")

    return Polygon(points)


def process_data(data, image_shape):
    # 子函数1: 解析标签和优先级
    def parse_label(label):
        # 检查标签中是否有下划线，如果有，只在最后一个下划线处分割
        if '_' in label:
            base_label, priority = label.rsplit('_', 1)
            try:
                return base_label, int(priority)
            except ValueError:
                # 如果转换优先级失败，说明可能后缀不是数字
                return label, 0
        else:
            # 如果没有下划线，优先级设为0
            return label, 0

    # 子函数2: 根据多边形和图像尺寸创建掩码
    def create_mask(polygon, shape):
        # 创建网格来代表图像空间
        nx, ny = shape[1], shape[0]  # 注意：numpy的顺序是先行后列，所以这里先用y后用x
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        points = np.vstack((x.flatten(), y.flatten())).T
        
        # 创建多边形路径
        poly_path = Path(polygon.exterior.coords)
        
        # 使用路径来检查哪些点在多边形内部
        grid = poly_path.contains_points(points)
        mask = grid.reshape((ny, nx))
        
        return mask

    # 子函数3: 合并同等级区域
    def merge_same_priority(data, shape):
        labels = {}
        for item in data:
            label, priority = parse_label(item['label'])
            try:
                polygon = create_polygon(item['points'])
                # Polygon(item['points'])
                mask = create_mask(polygon, shape)
            except ValueError as e:
                print(f"Skipping invalid polygon for label '{label}': {str(e)}")
                continue

            if label not in labels:
                labels[label] = {}
            if priority not in labels[label]:
                labels[label][priority] = mask
            else:
                labels[label][priority] = np.logical_or(labels[label][priority], mask)

        # 优先级高的掩码剪掉优先级低的掩码
        max_priority = max(max(priorities.keys()) for priorities in labels.values())
        for category, priorities in labels.items():
            for priority, mask in list(priorities.items()):
                current_mask = np.array(mask)
                for higher_priority in range(priority + 1, max_priority + 1):
                    for other_category, other_priorities in labels.items():
                        if higher_priority in other_priorities:
                            current_mask = np.logical_and(current_mask, np.logical_not(other_priorities[higher_priority]))
                labels[category][priority] = current_mask
        return labels

    # 子函数4: 将所有优先级合并成一个掩码
    def flatten_priorities(labels):
        all_category_mask = {}
        for category, priorities in labels.items():
            mask_shape = next(iter(priorities.values())).shape
            combined_mask = np.zeros(mask_shape, dtype=bool)
            for priority_mask in priorities.values():
                combined_mask = np.logical_or(combined_mask, priority_mask)
            all_category_mask[category] = combined_mask

        # 处理 'water' 类别
        full_mask = np.ones(mask_shape, dtype=bool)
        for label, mask in all_category_mask.items():
            if label != 'water':
                full_mask[mask] = False
        all_category_mask['water'] = np.logical_or(all_category_mask.get('water', np.zeros_like(full_mask)), full_mask)

        return all_category_mask

    # 主函数流程
    labels = merge_same_priority(data, image_shape)
    labels = flatten_priorities(labels) 
    return labels # 掩码类型的数据

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# 超参数设置
FONT = 'SimHei'  # 中文字体设置为黑体
COLOR_MAP = {
    'water': 0, 'land': 1, 'grid_field': 2, 'tidal_flat': 3
}
C_MAP_COLORS = [
    (0.255, 0.298, 0.529),  # 深蓝
    (135/255, 86/255, 72/255),  # 棕色
    (0.612, 0.702, 0.831),  # 浅蓝
    (246/255, 225/255, 198/255)  # 灰色
]
LABELS = ['水域', '陆地', '养殖区域', '潮滩']

def prepare_plot(labels, date):
    plt.rcParams['font.sans-serif'] = [FONT]
    plt.rcParams['axes.unicode_minus'] = False

    cmap = ListedColormap(C_MAP_COLORS)
    visualization_array = np.zeros_like(next(iter(labels.values())), dtype=int)

    for label, mask in labels.items():
        color_index = COLOR_MAP.get(label, 0)  # 未知类别默认为0（水域）
        visualization_array[mask] = color_index

    fig, ax = plt.subplots()
    cax = ax.imshow(visualization_array, cmap=cmap, interpolation='nearest')
    cbar = fig.colorbar(cax, ticks=range(len(COLOR_MAP)))
    cbar.set_ticklabels(LABELS)
    plt.title(date)

    return fig, ax

# 提取形状
def get_shape(data_js):
    return data_js["imageHeight"],data_js["imageWidth"]

# 提取日期
def get_date(filename):
    return filename.split('_')[3]


def prepare_plot_with_coords(labels, date, extent, transformer):
    plt.rcParams['font.sans-serif'] = [FONT]
    plt.rcParams['axes.unicode_minus'] = False

    cmap = ListedColormap(C_MAP_COLORS)
    visualization_array = np.zeros_like(next(iter(labels.values())), dtype=int)

    for label, mask in labels.items():
        color_index = COLOR_MAP.get(label, 0)
        visualization_array[mask] = color_index

    fig, ax = plt.subplots()
    cax = ax.imshow(visualization_array, cmap=cmap, interpolation='nearest', extent=[0, extent[1] - extent[0], 0, extent[2] - extent[3]])
    cbar = fig.colorbar(cax, ticks=range(len(COLOR_MAP)))
    cbar.set_ticklabels(LABELS)
    plt.title(date)

    # 获取经纬度坐标轴
    left_bottom = transformer.transform(extent[0], extent[3])
    right_top = transformer.transform(extent[1], extent[2])

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xticks(np.linspace(0, extent[1] - extent[0], num=5))
    ax.set_yticks(np.linspace(0, extent[2] - extent[3], num=5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{left_bottom[0] + val * (right_top[0] - left_bottom[0]) / (extent[1] - extent[0]):.2f}° E'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{left_bottom[1] + val * (right_top[1] - left_bottom[1]) / (extent[3] - extent[2]):.2f}° N'))

    return fig, ax

def load_transform(filename):
    with open(filename, 'r') as file:
        return json.load(file)['transform']

def load_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)
    