import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.plot import show

import matplotlib.pyplot as plt
from affine import Affine

def create_subset_dem(x_lims, y_lims, target_resolution, original_dem= "IslandsDEMv1.0_2x2m_zmasl_isn93_57.tif"): 
    #Original file downloaded from https://dem.gis.is/mapview/?application=DEM

    scale = target_resolution / 10 #= original_resolution

    with rasterio.open(original_dem) as src:

        # Create window from bounds
        window = from_bounds(
            x_lims[0],
            x_lims[1],
            y_lims[0],
            y_lims[1],
            src.transform
        )

        # Original subset size
        out_height = int(window.height / scale)
        out_width = int(window.width / scale)

        # Read + resample
        dem = src.read(
            1,
            window=window,
            out_shape=(out_height, out_width),
            resampling=Resampling.bilinear
        )

        # Original transform for subset
        transform = src.window_transform(window)

        # Update transform for new resolution
        new_transform = transform * Affine.scale(scale)

        # Copy metadata
        profile = src.profile.copy()

        # Update metadata
        profile.update({
            "height": out_height,
            "width": out_width,
            "transform": new_transform
        })

        # Save output
        with rasterio.open(
            "subset_dem.tif",
            "w",
            **profile
        ) as dst:

            dst.write(dem, 1)
            
def plot_dem(dem='subset_dem.tif'):

    with rasterio.open(dem) as src:

        fig, ax = plt.subplots(figsize=(8, 8))

        img = show(
            src,
            ax=ax,
            cmap="terrain"
        )

        cbar = plt.colorbar(
            img.get_images()[0],
            ax=ax
        )

        cbar.set_label("Elevation (m)")
        ax.set_title("Subset DEM")

        plt.show()

def get_info_dem(dem='subset_dem.tif'):
    with rasterio.open(dem) as src:

    print(src.crs)
    print(src.bounds)
    print(src.res)
    print(src.width, src.height)