import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.plot import show

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from affine import Affine
import pyvista as pv

import gempy as gp
import gempy_viewer as gpv
import os 

import plotly.graph_objects as go

import xtgeo

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
        print('z_min, z_max:', src.read(1, masked=True).min(), src.read(1, masked=True).max())

def create_dfs_4_gempy(azimuth, dip, silent=True):

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
    df_borehole = pd.read_csv('gempy_inputs/borehole_data.csv')

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
    dem_path = "subset_dem.tif"
    with rasterio.open(dem_path) as src:
        # sample DEM elevations at XY points
        coords = list(zip(df["X"], df["Y"]))
        dem_values = [val[0] for val in src.sample(coords)]
    df['Z'] = dem_values - df["Z"]

    #save surface points
    df[['X', 'Y', 'Z', 'formation']].to_csv('gempy_inputs/df_surfaces_4gempy.csv', index=False)
    if silent==False: 
        display(df[['X', 'Y', 'Z', 'formation']])

    #take an appropriate well for orientation values
    #df = df.loc[df['name'].isin(['MG-35', 'MG-19', 'MG-23', 'MG-25'])]

    df['azimuth'] = [azimuth]*len(df) #rough calculation from cross section (reports)
    df['dip'] = [dip]*len(df)
    df['polarity'] = [1]*len(df) #Polarity is typically defined as 1 (normal,younging direction matches orientation vector) or -1 (reversed)
    df = df.iloc[:, [0, 1, 2, 3, 5, 6, 7, 4]] #reorder

    df[['X', 'Y', 'Z', 'azimuth', 'dip', 'polarity', 'formation']].to_csv('gempy_inputs/df_orientations_4gempy.csv', index=False)
    if silent==False:
        display(df)


def plot_3d_model(
    geomodel, x, y, z,
    downsampling=1,
    show_empty=False,
    scale="linear",   # "linear" or "log"
    log_eps=1e-6      # safety offset for log
):

    data = geomodel[::downsampling, ::downsampling, ::downsampling].astype(np.float32)

    # --- scaling choice ---
    if scale == "log":
        # ensure positivity
        data = np.maximum(data, log_eps)
        data = np.log10(data)
    elif scale == "linear":
        pass
    else:
        raise ValueError("scale must be 'linear' or 'log'")

    grid = pv.ImageData()
    grid.dimensions = np.array(data.shape) + 1

    grid.origin = (x.min(), y.min(), z.min())

    grid.spacing = (
        x[1] - x[0],
        y[1] - y[0],
        z[1] - z[0]
    )

    grid.cell_data["values"] = data.flatten(order="F")

    if scale == "log":
        thresholded = grid.threshold(
            value=(-5.99, data.max()),
            scalars="values"
        )
    else:
        thresholded = grid.threshold(
            value=(1e-3, 1000),
            scalars="values"
        )

    plotter = pv.Plotter()

    if scale == 'log':
        plotter.add_mesh(
            thresholded,
            scalars="log10(values)",
            cmap="viridis",
            opacity=1.0
        )
    else: 
        plotter.add_mesh(
            thresholded,
            scalars="values",
            cmap="viridis",
            opacity=1.0
        )

    plotter.show_grid()
    plotter.show_axes()
    plotter.show()

def save_gempy_results(geomodel):
    lith = geomodel.solutions.raw_arrays.lith_block #* geo_model.solutions.raw_arrays.mask_matrix
    res = geomodel.grid.regular_grid.resolution #10, 10, 60
    lith_3d = lith.reshape(res)

    lith_topo = lith_3d * (~geomodel.grid.topography.topography_mask) #why is it inverted?! :/

    #geo_model.solutions.raw_arrays.mask_matrix #wtf is this?

    x = np.unique(geomodel.grid.regular_grid.values[:, 0])
    y = np.unique(geomodel.grid.regular_grid.values[:, 1])
    z = np.unique(geomodel.grid.regular_grid.values[:, 2])

    #sanity check
    #print(x.shape, y.shape, z.shape)

    #save the model
    np.save("gempy_outputs/lith_topo.npy", lith_topo)
    np.save('gempy_outputs/x_array.npy', x)
    np.save('gempy_outputs/y_array.npy', y)
    np.save('gempy_outputs/z_array.npy', z)

def create_strat_model(x_lims, y_lims, z_lims, resolution, strat_azimuth, strat_dip, target_resolution=10): 
    
    # 1. Create DEM file: 
    if os.path.isfile("subset_dem.tif"): 
        pass
    else: 
        #Original file downloaded from https://dem.gis.is/mapview/?application=DEM (too big for github)
        create_subset_dem(x_lims, y_lims, target_resolution=target_resolution) #NB: need to download original DEM 

    # Plot and check DEM if required
    #geomodel_toolbox.plot_dem('subset_dem.tif')
    get_info_dem('subset_dem.tif')

    # 2. Create/update input files for gempy: 
    create_dfs_4_gempy(azimuth=strat_azimuth, dip=strat_dip)
    

    # 3. Use gempy to make model
    extent = x_lims+y_lims+z_lims[::-1]

    # Create instance of geomodel
    geo_model = gp.create_geomodel(
        project_name = 'tutorial_model',
        extent=extent,
        resolution=resolution,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations='gempy_inputs/df_orientations_4gempy.csv',
            path_to_surface_points='gempy_inputs/df_surfaces_4gempy.csv'
        )
    )

    # Add topography
    gp.set_topography_from_file(
        grid=geo_model.grid,
        filepath='subset_dem.tif')

    #Plot map
    gpv.plot_2d(geo_model, show_topography=True, section_names=['topography'])

    # Display a basic cross section of input data
    gpv.plot_2d(geo_model, show_topography=True)

    # Define structural groups and age/stratigraphic relationship
    gp.map_stack_to_surfaces(
        gempy_model=geo_model,
        mapping_object={
            "Strat_Series1": ('unknown', 'B1', 'M1', 'B2', 'M2', 'B3', 'M3')
            }
    )

    # Compute a solution for the model (takes 17 mins. )
    gp.compute_model(geo_model)

    # Display the result in 2d section
    gpv.plot_2d(geo_model, show_topography=True)

    # Save the results
    save_gempy_results(geomodel=geo_model)

    # Import to plot
    lith_topo = np.load('gempy_outputs/lith_topo.npy')
    x_array = np.load('gempy_outputs/x_array.npy')
    y_array = np.load('gempy_outputs/y_array.npy')
    z_array = np.load('gempy_outputs/z_array.npy')

    # Check sizes
    #print(lith_topo.shape)
    #print(x_array.shape, y_array.shape, z_array.shape)
    
    # Plot 3D model
    plot_3d_model(geomodel=lith_topo, x=x_array, y=y_array, z=z_array, downsampling=1)

def import_strat_model(downsampling_for_plotting=1, show_plot=True, show_info=False):
    # Import saved files
    lith_topo = np.load('gempy_outputs/lith_topo.npy')
    x_array = np.load('gempy_outputs/x_array.npy')
    y_array = np.load('gempy_outputs/y_array.npy')
    z_array = np.load('gempy_outputs/z_array.npy')

    if show_info: 
        # Check sizes
        print(lith_topo.shape)
        print(x_array.shape, y_array.shape, z_array.shape)
    
    if show_plot: 
        # Plot 3D model
        plot_3d_model(geomodel=lith_topo, x=x_array, y=y_array, z=z_array, downsampling=downsampling_for_plotting)

    return lith_topo, x_array, y_array, z_array

def generate_fracture_model(x, y, z, aperture, radius, azimuth, density, show_plot=False):

    # A. Total number of fractures
    vol = ((x.max() - x.min()) * (y.max() - y.min()) * (z.max() - z.min()))

    N = np.random.poisson(lam=vol * density)

    # B. Create grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # C. Generate centers
    x_0 = np.random.choice(x, size=N)
    y_0 = np.random.choice(y, size=N)
    z_0 = np.random.choice(z, size=N)

    # D. Randomize apertures, radii and orientations
    aperture_n = np.random.normal(aperture[0], aperture[1], size=N)
    radius_n   = np.random.normal(radius[0], radius[1], size=N)

    azimuth_n = np.random.normal(
        azimuth[0],
        azimuth[1],
        size=N
    )
    azimuth_n = np.deg2rad(azimuth_n)

    # E. Define directions
    east  = np.array([1.0, 0.0, 0.0])
    north = np.array([0.0, 1.0, 0.0])

    mask = np.zeros(X.shape, dtype=bool)

    # F. Create fractures
    for i in range(N):

        u = np.cos(azimuth_n[i]) * north + np.sin(azimuth_n[i]) * east
        v = -np.sin(azimuth_n[i]) * north + np.cos(azimuth_n[i]) * east

        C = np.array([x_0[i], y_0[i], z_0[i]])

        RX = X - C[0]
        RY = Y - C[1]
        RZ = Z - C[2]

        s = RX*u[0] + RY*u[1] + RZ*u[2]
        t = RZ
        w = RX*v[0] + RY*v[1] + RZ*v[2]

        inside_disk = (s**2 + t**2) <= radius_n[i]**2
        along_azimuth = np.abs(w) <= aperture_n[i]/2

        mask |= (inside_disk & along_azimuth)

    # Save parameters
    df = pd.DataFrame({
        "aperture": [aperture],
        "radius": [radius],
        "azimuth": [azimuth]
    }, index=["mean"])

    # Plot
    if show_plot:
        ix, iy, iz = np.where(mask)

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x[ix],
                    y=y[iy],
                    z=z[iz],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=z[iz],
                        colorscale="Viridis",
                        opacity=0.6
                    )
                )
            ]
        )

        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data"
            )
        )

        fig.show()

    return mask, df

def export_grdecl_from_geomodel(
    geomodel,
    x_array,
    y_array,
    z_array,
    rock_types,
    out_file="grid.grdecl"
):
    import numpy as np
    import xtgeo

    # =========================================================
    # ENSURE Z IS ORDERED TOP -> BOTTOM
    # =========================================================
    geomodel = np.asarray(geomodel)
    z_array = np.asarray(z_array)

    # Keep only depths at or below datum
    keep = z_array <= 0

    if not np.any(keep):
        raise ValueError("No layers found with z <= 0")

    geomodel = geomodel[:, :, keep]
    z_array = z_array[keep]

    # ---------------------------------------------------------
    # Sort so first layer is closest to zero (top)
    #
    # Example:
    # [-100, -80, -60, -40, -20, 0]
    # becomes
    # [0, -20, -40, -60, -80, -100]
    # ---------------------------------------------------------
    sort_idx = np.argsort(z_array)[::-1]

    z_array = z_array[sort_idx]
    geomodel = geomodel[:, :, sort_idx]

    nx, ny, nz = len(x_array), len(y_array), len(z_array)

    dx = x_array[1] - x_array[0]
    dy = y_array[1] - y_array[0]

    if nz > 1:
        dz = abs(np.mean(np.diff(z_array)))
    else:
        raise ValueError("Need at least two z layers")

    if dz == 0:
        raise ValueError("z_array spacing is zero")

    # =========================================================
    # GRID ORIGIN (TOP CORNER)
    # =========================================================
    grid = xtgeo.create_box_grid(
        dimension=(nx, ny, nz),
        origin=(
            x_array[0] - dx / 2,
            y_array[0] - dy / 2,
            z_array[0] - dz / 2   # TOP is first Z value
        ),
        increment=(dx, dy, dz)
    )

    # =========================================================
    # ACTNUM
    # =========================================================
    actnum_vals = (geomodel != 0).astype(int)

    actnum = xtgeo.GridProperty(
        grid,
        name="ACTNUM",
        values=actnum_vals,
        discrete=True,
    )
    grid.set_actnum(actnum)

    # =========================================================
    # FACIES
    # =========================================================
    facies_3 = np.zeros_like(geomodel)

    facies_3[np.isin(geomodel, [1, 3, 5, 7])] = 1
    facies_3[np.isin(geomodel, [2, 4, 6, 8])] = 2
    facies_3[geomodel == 9] = 3

    facies = xtgeo.GridProperty(
        grid,
        name="FACIES",
        values=facies_3,
        discrete=True,
    )

    # =========================================================
    # ROCK PROPERTIES
    # =========================================================
    poro = np.zeros_like(geomodel, dtype=float)
    permx = np.zeros_like(geomodel, dtype=float)

    rock_map = {
        1: rock_types["M"],
        2: rock_types["B"],
        3: rock_types["F"],
    }

    for fid, props in rock_map.items():
        mask = facies_3 == fid
        poro[mask] = props["poro"]
        permx[mask] = props["permx"]

    poro[actnum_vals == 0] = 0
    permx[actnum_vals == 0] = 0

    poro = np.asfortranarray(poro)
    permx = np.asfortranarray(permx)

    poro_prop = xtgeo.GridProperty(
        grid,
        name="PORO",
        values=poro
    )

    perm_prop = xtgeo.GridProperty(
        grid,
        name="PERMX",
        values=permx
)
    # =========================================================
    # EXPORT
    # =========================================================
    grid.to_file(out_file, fformat="grdecl")
    facies.to_file(out_file, fformat="grdecl", append=True)
    poro_prop.to_file(out_file, fformat="grdecl", append=True)
    perm_prop.to_file(out_file, fformat="grdecl", append=True)

    return grid


def export_wells_as_dev():

    # Read wells
    wells = pd.read_csv("gempy_inputs/borehole_data.csv")

    # Sample DEM elevations
    with rasterio.open("subset_dem.tif") as dem:
        coords = list(zip(wells["x"], wells["y"]))
        elevations = [v[0] for v in dem.sample(coords)]

    wells["surface_z"] = elevations

    # Bottom elevation = surface elevation - depth
    wells["bottom_z"] = wells["depth"] - wells["surface_z"]

    # Write ResInsight-style well path file
    with open("vertical_wells.dev", "w") as f:

        for _, row in wells.iterrows():

            f.write(f"WELLNAME: {row['name']}\n")

            # Wellhead
            f.write(
                f"{row['x']:.2f} "
                f"{row['y']:.2f} "
                f"-100.00 "
                #f"{row['surface_z']:.2f} "
                f"0.00\n"
            )

            # Vertical bottom
            f.write(
                f"{row['x']:.2f} "
                f"{row['y']:.2f} "
                f"{row['bottom_z']:.2f} "
                f"{row['depth']:.2f}\n\n"
            )

    print("Created vertical_wells.dev")