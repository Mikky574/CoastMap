import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
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

# 超参数设置
FONT = 'SimHei'  # 中文字体设置为黑体

COLOR_MAP = {
    'water': (0.255, 0.298, 0.529),
    'land': (135/255, 86/255, 72/255),
    'grid_field': (0.612, 0.702, 0.831),
    'tidal_flat': (246/255, 225/255, 198/255)
}

# 中文到英文的映射
CHINESE_TO_ENGLISH = {
    '水域': 'water',
    '陆地': 'land',
    '养殖区域': 'grid_field',
    '潮滩': 'tidal_flat'
}

def prepare_plot(labels, date, categories_l):
    
    plt.rcParams['font.sans-serif'] = [FONT]
    plt.rcParams['axes.unicode_minus'] = False
    # Create an empty RGB image filled with white color
    img_shape = next(iter(labels.values())).shape
    img = np.ones((*img_shape, 3))

    # Convert categories_l from Chinese to English
    categories_en = [CHINESE_TO_ENGLISH[cat] for cat in categories_l]

    # Apply colors to the image based on the categories
    for category, category_en in zip(categories_l, categories_en):
        mask = labels[category_en]
        color = COLOR_MAP[category_en]
        img[mask] = color

    # Plot the image
    fig, ax = plt.subplots()
    ax.imshow(img, extent=(0, img_shape[1], 0, img_shape[0]), origin='upper')
    
    # Create legend with squares
    handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MAP[CHINESE_TO_ENGLISH[cat]], markersize=10, linestyle='None', label=cat) 
               for cat in categories_l]
    
    ax.legend(handles=handles)
    ax.axis('on')
    
    # Set title with date
    ax.set_title(f"Date: {date}")
    
    return fig, ax

# 提取形状
def get_shape(data_js):
    return data_js["imageHeight"],data_js["imageWidth"]

# 提取日期
def get_date(filename):
    return filename.split('_')[3]

def prepare_plot_with_coords(labels, date, extent, transformer, categories_l):
    fig, ax = prepare_plot(labels, date, categories_l)

    left_bottom = transformer.transform(extent[0], extent[2])
    right_top = transformer.transform(extent[1], extent[3])

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # 从ax中获取图像的宽度和高度
    x_extent = ax.get_xlim()
    y_extent = ax.get_ylim()
    
    width = x_extent[1] - x_extent[0]
    height = y_extent[1] - y_extent[0]
    
    ax.set_xticks(np.linspace(x_extent[0], x_extent[1], num=5))
    ax.set_yticks(np.linspace(y_extent[0], y_extent[1], num=5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{left_bottom[0] + val * (right_top[0] - left_bottom[0]) / width:.2f}° E'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{left_bottom[1] + val * (right_top[1] - left_bottom[1]) / height:.2f}° N'))

    return fig, ax

def load_transform(filename):
    with open(filename, 'r') as file:
        return json.load(file)['transform']

def load_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)
    