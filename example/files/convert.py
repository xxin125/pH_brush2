import sys
import json
import argparse
import numpy as np
import MDAnalysis as mda
import MDAnalysis.core.topologyattrs as TA


def mk_universe(top, coord, frame=None):
    """Build an MDAnalysis Universe from GROMACS TPR (topology) + GRO (coords)."""
    u = mda.Universe(top, coord)
    if frame is not None:
        u.trajectory[frame]
    return u


def set_topology_attr(u, name, values):
    """
    Set or add a topology attribute (e.g. types, type_indices).
    If the attribute exists, overwrite .values. Otherwise add a new TopologyAttr.
    """
    top = u._topology
    for attr in getattr(top, "attrs", []):
        if getattr(attr, "attrname", None) == name:
            attr.values = values
            return
    try:
        u.add_TopologyAttr(name, values)
    except Exception:
        # some attrs expect python lists instead of numpy arrays
        u.add_TopologyAttr(name, list(values))


def find_attr(top, cls):
    """Return the first topology attribute that is an instance of cls (Bonds, Angles, ...)."""
    for attr in getattr(top, "attrs", []):
        if isinstance(attr, cls):
            return attr
    return None


def load_type_map(user_map_file):
    """
    Load and validate the user-supplied atom-type map JSON.

    JSON must look like:
    {
        "AtomTypeString": 1,
        "AnotherType": 2,
        ...
    }

    Rules:
    - user_map_file must not be None
    - each value must be an int >= 1
    - all values must be unique
    """
    if user_map_file is None:
        raise ValueError(
            "You must provide --typemap <file.json>; no default mapping is allowed."
        )

    with open(user_map_file, "r") as f:
        final_map = json.load(f)

    # basic validation
    for k, v in final_map.items():
        if not isinstance(v, int) or v < 1:
            raise ValueError(
                f"type_map invalid: {k} -> {v} (must be int >=1)"
            )

    inv = {}
    for k, v in final_map.items():
        if v in inv:
            raise ValueError(
                f"type_map invalid: {k} and {inv[v]} share same ID {v}"
            )
        inv[v] = k

    return final_map


def remap_atom_types_to_int(u, type_map):
    """
    Use the provided type_map (str -> int) to assign integer atom types.

    This updates:
        u.atoms.type_indices (numpy int array)
        u.atoms.types        (object array containing the same ints)

    If any atom type string in the Universe is missing from type_map, raise.
    """
    orig_types = np.array(u.atoms.types, dtype=object)

    # detect unmapped types
    missing = sorted(set(t for t in orig_types if t not in type_map))
    if missing:
        suggestion = "\n".join([f'    "{t}": <INT_ID>,' for t in missing])
        raise ValueError(
            "Found atom types with no mapping:\n"
            + ", ".join(missing)
            + "\nPlease extend your typemap JSON with e.g.:\n{\n"
            + suggestion
            + "\n}"
        )

    int_ids = np.array([type_map[t] for t in orig_types], dtype=int)

    # push them back into the topology
    set_topology_attr(u, "type_indices", int_ids)
    set_topology_attr(u, "types", int_ids.astype(object))

    return type_map


def ensure_integer_types_for_connectivity(u):
    """
    Generate integer bond/angle/dihedral/improper *types* based on
    the integer atom type IDs currently stored in u.atoms.type_indices.

    For each unique pattern of atom-type IDs, assign a new consecutive integer
    and write that into the .types field of the corresponding TopologyAttr
    (Bonds, Angles, Dihedrals, Impropers).
    """
    atype_ids = u.atoms.type_indices.astype(int)
    top = u._topology

    # Bonds
    bonds_attr = find_attr(top, TA.Bonds)
    if bonds_attr is not None and len(bonds_attr.values) > 0:
        bond_patterns = [
            tuple(sorted((atype_ids[i], atype_ids[j])))
            for i, j in bonds_attr.values
        ]
        uniq_patterns = list(dict.fromkeys(bond_patterns))
        bond_type_map = {pat: idx + 1 for idx, pat in enumerate(uniq_patterns)}
        bonds_attr.types = np.array(
            [bond_type_map[p] for p in bond_patterns],
            dtype=int
        )

    # Angles
    ang_attr = find_attr(top, TA.Angles)
    if ang_attr is not None and len(ang_attr.values) > 0:
        angle_patterns = [
            (atype_ids[i], atype_ids[j], atype_ids[k])
            for i, j, k in ang_attr.values
        ]
        uniq_patterns = list(dict.fromkeys(angle_patterns))
        angle_type_map = {
            pat: idx + 1 for idx, pat in enumerate(uniq_patterns)
        }
        ang_attr.types = np.array(
            [angle_type_map[p] for p in angle_patterns],
            dtype=int
        )

    # Dihedrals
    dih_attr = find_attr(top, TA.Dihedrals)
    if dih_attr is not None and len(dih_attr.values) > 0:
        dihed_patterns = [
            (atype_ids[i], atype_ids[j], atype_ids[k], atype_ids[l])
            for i, j, k, l in dih_attr.values
        ]
        uniq_patterns = list(dict.fromkeys(dihed_patterns))
        dihed_type_map = {
            pat: idx + 1 for idx, pat in enumerate(uniq_patterns)
        }
        dih_attr.types = np.array(
            [dihed_type_map[p] for p in dihed_patterns],
            dtype=int
        )

    # Impropers
    imp_attr = find_attr(top, TA.Impropers)
    if imp_attr is not None and len(imp_attr.values) > 0:
        impr_patterns = [
            (atype_ids[i], atype_ids[j], atype_ids[k], atype_ids[l])
            for i, j, k, l in imp_attr.values
        ]
        uniq_patterns = list(dict.fromkeys(impr_patterns))
        impr_type_map = {
            pat: idx + 1 for idx, pat in enumerate(uniq_patterns)
        }
        imp_attr.types = np.array(
            [impr_type_map[p] for p in impr_patterns],
            dtype=int
        )


def assign_molecule_tags_from_fragments(u):
    """
    Derive per-atom molecule IDs from covalent fragments.

    Each covalently bonded fragment becomes one unique molecule ID.
    The IDs are stored in the current Timestep so that the LAMMPS writer
    will include them as "molecule-ID".
    """
    frags = list(u.atoms.fragments)
    moltag = np.zeros(len(u.atoms), dtype=int)
    for mol_id, frag in enumerate(frags, start=1):
        moltag[frag.indices] = mol_id
    # write into current timestep's aux data
    u.trajectory.ts.data["molecule_tag"] = moltag


def build_massmap(u):
    """
    Build a dict {atom_type_id: mass}.

    Assumes all atoms sharing the same integer atom_type_id also share a mass.
    """
    t = u.atoms.type_indices.astype(int)
    m = u.atoms.masses
    mm = {}
    for tid, mass in zip(t, m):
        if tid not in mm:
            mm[tid] = float(mass)
    return mm


def write_lammps_data(u, outfile):
    """
    Use MDAnalysis to write a first-pass LAMMPS data file.
    This writes coordinates, topology, and also its own Masses section.
    """
    u.atoms.write(outfile, lengthunit="nm", timeunit="ps")


def patch_masses_inplace(outfile, massmap):
    """
    Rewrite the 'Masses' section of the LAMMPS data file in-place.

    We discard whatever MDAnalysis wrote under 'Masses' and instead
    emit masses according to massmap, which is keyed by our integer
    atom type IDs (the type_indices).
    """
    with open(outfile, "r") as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        if line.strip() == "Masses":
            # write corrected Masses block
            new_lines.append("Masses\n\n")
            for tid in sorted(massmap.keys()):
                new_lines.append(f"{tid:5d} {massmap[tid]:.6f}\n")
            new_lines.append("\n")

            # skip original Masses block
            i += 1
            # skip blank lines right after "Masses"
            while i < n and lines[i].strip() == "":
                i += 1
            # skip until we hit the next known section header
            while (
                i < n
                and lines[i].strip()
                not in (
                    "Atoms", "Bonds", "Angles", "Dihedrals", "Impropers",
                    "Velocities", "Pair Coeffs", "Bond Coeffs",
                    "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs"
                )
            ):
                i += 1
            # do not consume the header line we just found; loop continues
            continue

        new_lines.append(line)
        i += 1

    with open(outfile, "w") as f:
        f.writelines(new_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GROMACS (TPR+GRO) -> LAMMPS data with explicit atom type mapping."
    )
    parser.add_argument("tpr_file", help="GROMACS .tpr topology")
    parser.add_argument("gro_file", help="GROMACS .gro coordinates")
    parser.add_argument("out_file", help="LAMMPS data output filename")
    parser.add_argument(
        "--typemap",
        required=True,
        help=(
            "JSON file with {gmx_type(str): lammps_type(int)}. "
            "No defaults; every atom type in this system must appear."
        ),
    )

    args = parser.parse_args()

    # 0) load explicit atom-type mapping
    type_map = load_type_map(args.typemap)

    # 1) build MDAnalysis Universe
    u = mk_universe(args.tpr_file, args.gro_file)

    # 2) assign per-atom integer types (strict mapping)
    remap_atom_types_to_int(u, type_map)

    # 3) build integer connectivity types for bonds/angles/dihedrals/impropers
    ensure_integer_types_for_connectivity(u)

    # 4) assign per-atom molecule IDs from fragments
    assign_molecule_tags_from_fragments(u)

    # 5) build map {int_type : mass}
    massmap = build_massmap(u)

    # 6) write first-pass LAMMPS data
    write_lammps_data(u, args.out_file)

    # 7) replace Masses section with our controlled masses
    patch_masses_inplace(args.out_file, massmap)

    print(f"Done. Wrote LAMMPS data -> {args.out_file}")


if __name__ == "__main__":
    main()
