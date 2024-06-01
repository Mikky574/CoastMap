from utils import (
    process_data,
    get_shape,
    get_date,
    prepare_plot_with_coords,
    prepare_plot,
    load_data,
    load_transform,
)
import matplotlib.pyplot as plt
from pyproj import Transformer
import os
from tqdm import tqdm

class Json_loader:
    def __init__(self, filename, transform_filename, plot_directory=None, verbose=True):
        self.filename = filename
        self.data = load_data(filename)
        self.image_shape = get_shape(self.data)
        self.labels = process_data(self.data["shapes"], self.image_shape)
        self.date = get_date(filename)
        self.transform = load_transform(transform_filename)
        self.transformer = Transformer.from_crs(
            "epsg:32649", "epsg:4326", always_xy=True
        )
        self.plot_directory = plot_directory if plot_directory else os.getcwd()
        self.verbose = verbose  # 控制是否打印详细信息
        
    def plot(self,categories_l=['水域', '陆地', '养殖区域', '潮滩']):
        fig, ax = prepare_plot(self.labels, self.date,categories_l=categories_l)
        
        plt.show()

    def plot_save(self,categories_l=['水域', '陆地', '养殖区域', '潮滩']):
        save_path = os.path.join(self.plot_directory, f"{self.date}.png")
        fig, ax = prepare_plot(self.labels, self.date,categories_l=categories_l)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        if self.verbose:
            print(f"Saved image to {save_path}")

    def plot_with_coords(self,categories_l=['水域', '陆地', '养殖区域', '潮滩']):
        extent = self.get_extent()
        fig, ax = prepare_plot_with_coords(
            self.labels, self.date, extent, self.transformer,categories_l=categories_l
        )
        plt.show()

    def plot_save_with_coords(self,categories_l=['水域', '陆地', '养殖区域', '潮滩'],dpi=300):
        filename = f"{self.date}.jpg"
        save_path = os.path.join(self.plot_directory, filename)
        extent = self.get_extent()
        fig, ax = prepare_plot_with_coords(
            self.labels, self.date, extent, self.transformer,categories_l=categories_l
        )
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        if self.verbose:
            print(f"Saved image to {save_path}")

    def get_extent(self):
        trans = self.transform
        width, height = self.image_shape[1], self.image_shape[0]
        return [
            trans[0],
            trans[0] + trans[1] * width,
            trans[3] + trans[5] * height,
            trans[3],
        ]

class FolderProcessor:
    def __init__(self, input_dir, transform_filename, plot_directory):
        self.input_dir = input_dir
        self.transform_filename = transform_filename
        self.plot_directory = plot_directory

    def process_folder(self):
        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.json')]
        for filename in tqdm(files, desc="Processing files"):
            full_path = os.path.join(self.input_dir, filename)
            if os.path.isfile(full_path):
                self.plot_and_save(full_path)

    def plot_and_save(self, full_path):
        # 注意这里将 verbose 设置为 False
        data = Json_loader(full_path, self.transform_filename, self.plot_directory, verbose=False)
        data.plot_save_with_coords()

    def run_plotting(self):
        print("Starting to process files...")
        self.process_folder()
        print("Completed processing all files.")
