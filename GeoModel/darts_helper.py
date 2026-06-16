import numpy as np

import numpy as np

def centers_to_edges(c):
    c = np.asarray(c)

    edges = np.empty(len(c) + 1)

    edges[1:-1] = (c[:-1] + c[1:]) / 2
    edges[0] = c[0] - (c[1] - c[0]) / 2
    edges[-1] = c[-1] + (c[-1] - c[-2]) / 2

    return edges


def write_keyword_array(f, keyword, arr):
    f.write(f"{keyword}\n")

    values = np.asarray(arr).flatten(order="F")

    for i, v in enumerate(values):

        if np.issubdtype(type(v), np.integer):
            f.write(f"{int(v)} ")
        else:
            f.write(f"{float(v):.6g} ")

        if (i + 1) % 8 == 0:
            f.write("\n")

    f.write("\n/\n\n")


def model_to_grdecl(
    geomodel,
    x_array,
    y_array,
    z_array,
    rock_props,
    output_file=["reservoir.grdecl", 'properties.in']
):
    """
    Convert a lithology cube into an Eclipse GRDECL file.

    Parameters
    ----------
    geomodel : ndarray
        Lithology model with shape (NX, NY, NZ)

    x_array, y_array, z_array : 1D ndarray
        Cell-center coordinates

    output_file : str
        Output GRDECL filename

    rock_props : dict
        Dictionary mapping facies IDs to porosity/permeability
    """
    AIR_CODE = 0
    



    lith = np.asarray(geomodel).astype(int)

    NX, NY, NZ = lith.shape

    x_edges = centers_to_edges(x_array)
    y_edges = centers_to_edges(y_array)
    z_edges = centers_to_edges(z_array)

    # --------------------------------------------------------
    # ACTNUM
    # --------------------------------------------------------

    actnum = (lith != AIR_CODE).astype(int)

    # --------------------------------------------------------
    # ROCK PROPERTIES
    # --------------------------------------------------------

    poro = np.zeros_like(lith, dtype=float)
    permx = np.zeros_like(lith, dtype=float)

    for facies, props in rock_props.items():

        mask = lith == facies

        poro[mask] = props["poro"]
        permx[mask] = props["permx"]

    # --------------------------------------------------------
    # WRITE FILE
    # --------------------------------------------------------

    with open(output_file[0], "w") as f:

        # SPECGRID
        f.write("SPECGRID\n")
        f.write(f"{NX} {NY} {NZ} 1 F /\n\n")

        # COORD
        f.write("COORD\n")

        z_top = z_edges[0]
        z_bottom = z_edges[-1]

        for y in y_edges:
            for x in x_edges:
                f.write(
                    f"{x:.6f} {y:.6f} {z_top:.6f} "
                    f"{x:.6f} {y:.6f} {z_bottom:.6f}\n"
                )

        f.write("/\n\n")

        # ZCORN
        f.write("ZCORN\n")

        counter = 0

        for k in range(NZ):

            top = z_edges[k]
            bottom = z_edges[k + 1]

            for _ in range(NX * NY):

                vals = [
                    top, top, top, top,
                    bottom, bottom, bottom, bottom
                ]

                for v in vals:

                    f.write(f"{v:.6f} ")

                    counter += 1

                    if counter % 8 == 0:
                        f.write("\n")

        f.write("/\n\n")

        # ACTNUM
        write_keyword_array(f, "ACTNUM", actnum)

        # FACIES
        #write_keyword_array(f, "FACIES", lith)

        # PORO
        #write_keyword_array(f, "PORO", poro)

        # PERMX
        #write_keyword_array(f, "PERMX", permx)

    print(f"Written: {output_file[0]}")
    print(f"Grid dimensions: {NX} x {NY} x {NZ}")
    print(f"Total cells: {NX * NY * NZ:,}")
    print(f"Active cells: {actnum.sum():,}")
    print(f"Inactive air cells: {(actnum == 0).sum():,}")

    with open(output_file[1], "w") as f:

        # FACIES
        #write_keyword_array(f, "FACIES", lith)

        # PORO
        write_keyword_array(f, "PORO", poro)

        # PERMX
        write_keyword_array(f, "PERMX", permx)
    
    print(f"\nWritten: {output_file[1]}")


def nearest_eclipse_index(value, axis):
    return np.argmin(np.abs(axis - value)) + 1


def wells_to_compdat(wells, x_array, y_array, z_array, outfile="well_data.inc"):

    lines = ["--perforations in Eclipse format I, J, K1-K2, 1-based indices",
             "COMPDAT"]

    for _, row in wells.iterrows():

        I = nearest_eclipse_index(row.x, x_array)
        J = nearest_eclipse_index(row.y, y_array)

        z_casing = -row.casing
        z_depth = -row.depth

        K1 = nearest_eclipse_index(z_casing, z_array)
        K2 = nearest_eclipse_index(z_depth, z_array)

        K1, K2 = sorted((K1, K2))

        lines.append(f"PRD {I} {J} {K1} {K2} /")

    text = "\n".join(lines)

    with open(outfile, "w") as f:
        f.write(text)

    print(f"\nWritten: {outfile}")