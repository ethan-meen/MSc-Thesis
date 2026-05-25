import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.plot import show

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from affine import Affine

def create_subset_dem(x_lims, y_lims, target_resolution, original_dem= "IslandsDEMv1.0_2x2m_zmasl_isn93_57.tif"): 
    #Original file downloaded from https://dem.gis.is/mapview/?application=DEM

    scale = target_resolution / 2 #= original_resolution

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
            
def plot_dem(dem):

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

def get_info_dem(dem):
    with rasterio.open(dem) as src:

        print('projection:', src.crs)
        print(src.bounds)
        print('res:',src.res)
        print('nx, ny:', src.width, src.height)

def create_dfs_4_gempy(azimuth, dip):

    #horizon data: taken from report
    df_horizons = pd.DataFrame({
        'horizon': ['B1', 'M1', 'B2', 'M2', 'B3', 'M3', 'B4'],
        'MG-37': [0, 77, 266, 365, 745, 945, 1085], 
        'MG-19': [np.nan, 0, 74, 264, 752, 890, 1085], 
        'MG-28': [np.nan, 0, 134, 322, 748, 966, 1245], 
        'MG-05': [0, 41, 184, 392, 748, 975, 1210],
        'MG-21': [0, 120, 358, 563, 930, 1008, 1396], 
        'MG-32': [0, 263, 456, 974, np.nan, np.nan, np.nan], 
        'MG-36': [0, 326, 616, 1000, 1258, 1490, 1760],
        'MG-34': [0, 250, 415, 906, 1270, 1360, 1732],
        'MG-29': [0, 319, 425, 804, 1232, np.nan, np.nan],
        'MG-38': [0, 265, 409, 780, 1102, 1178, 1738],
        'MG-33': [0, 326, 558, 817, 1126, 1300, 1782], 
        'MG-35': [0, 486, 680, 970, 1340, 1453, np.nan],
        'MG-39': [0, 428, 672, 1042, 1377, 1509, np.nan],
        'MG-01': [np.nan, 0, 266, 334, 844, 906, 1012],
        'MG-02': [205, 462, 490, 622, 830, 1023, 1124],
        'MG-03': [np.nan, 0, 246, 383, 610, 873, 1022], 
        'MG-04': [np.nan, 72, 265, 485, 756, 922, 1195], 
        'MG-06': [np.nan, 0, 191, 460, 716, 829, 1115], 
        'MG-07': [np.nan, 0, 270, 311, 578, 951, 1024], 
        'MG-08': [0, 45, 210, 408, 770, 934, 1127], 
        'MG-09': [0, 142, 270, 465, 608, 865, 930], 
        'MG-10': [0, 252, 360, 605, 840, np.nan, np.nan],
        'MG-11': [0, 90, 316, 360, 917, np.nan, np.nan], 
        'MG-12': [0, 95, 167, 340, 638, np.nan, np.nan], 
        'MG-13': [0, 0, 116, 280, 530, 664, 1200], 
        'MG-14': [0, 0, 160, 303, 621, 944, 1060], 
        'MG-15': [0, 0, 174, 277, 660, 832, 902], 
        'MG-16': [0, 0, 88, 176, 710, 830, 1210],
        'MG-17': [0, 230, 306, 586, 816, 1152, 1368], 
        'MG-18': [0, 0, 186, 310, 714, 1042, 1268], 
        'MG-20': [0, 0, 136, 318, 634, 722, 1480], 
        'MG-22': [118, 518, 754, 840, 1100, 1229, np.nan], 
        'MG-23': [108, 518, 742, 1070, np.nan, np.nan, np.nan], 
        'MG-29': [0, 120, 416, 502, 939, 1224, 1320], 
        'MG-25': [0, 216, 512, 743, 938, 1158, 1404], 
        'MG-26': [123, 353, 458, 540, np.nan, np.nan, np.nan], 
        'MG-27': [0, 608, 885, 970, 1107, 1265, 1678], 
        'MG-30': [0, 139, 296, 530, 825, 1486, np.nan], 
        'MG-31': [67, 486, 600, 644, np.nan, np.nan, np.nan]
    })

    #borehole data (coordinates, etc.) taken from gisborhola.csv (find original link)
    df_borehole = pd.read_csv('borehole_data.csv')

    #Melt dataframe
    df = df_horizons.melt(
        id_vars="horizon",
        var_name="name",
        value_name="Z"
    )

    #Merge both dataframes
    df = df.merge(
        df_borehole[["name", "x", "y"]],
        on="name",
        how="left"
    )

    #Remove nans 
    df = df.dropna() 


    df = df.rename(columns={'horizon':'formation', 'x':'X', 'y':'Y'})
    df = df[['name', 'X','Y','Z','formation']]
    df['formation'] = df['formation'].replace({'B1':'unknown', 'M1':'B1', 'B2':'M1', 'M2':'B2', 'B3':'M2', 'M3':'B3', 'B4':'M3'}) #change top to bottom (for gempy)

    #rempve surface points (bc points at surface are not 'bottom' of stratum!)
    df = df[df["Z"] != 0].reset_index(drop=True)

    # open DEM
    dem_path = "../DEM/subset_dem.tif"
    with rasterio.open(dem_path) as src:
        # sample DEM elevations at XY points
        coords = list(zip(df["X"], df["Y"]))
        dem_values = [val[0] for val in src.sample(coords)]
    df['Z'] = dem_values - df["Z"]

    #save surface points
    df[['X', 'Y', 'Z', 'formation']].to_csv('../gempy/df_surfaces_4gempy.csv', index=False)
    display(df[['X', 'Y', 'Z', 'formation']])

    #take an appropriate well for orientation values
    #df = df.loc[df['name'].isin(['MG-35', 'MG-19', 'MG-23', 'MG-25'])]

    df['azimuth'] = [azimuth]*len(df) #rough calculation from cross section (reports)
    df['dip'] = [dip]*len(df)
    df['polarity'] = [1]*len(df) #Polarity is typically defined as 1 (normal,younging direction matches orientation vector) or -1 (reversed)
    df = df.iloc[:, [0, 1, 2, 3, 5, 6, 7, 4]] #reorder

    df[['X', 'Y', 'Z', 'azimuth', 'dip', 'polarity', 'formation']].to_csv('../gempy/df_orientations_4gempy.csv', index=False)
    display(df)