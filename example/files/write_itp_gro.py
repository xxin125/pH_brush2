#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

SECTION_NAMES = [
    "Masses", "Pair Coeffs", "Bond Coeffs", "Angle Coeffs",
    "Dihedral Coeffs", "Improper Coeffs",
    "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers",
]

bond_params = {
    ("CM", "OM"): (1, 0.282, 40000),
    ("CM", "CM"): (1, 0.281, 15000),
    ("OM", "NC"): (1, 0.393, 16400),
    ("OM", "NH"): (1, 0.383, 16400),
}
angle_params = {
    ("OM", "CM", "CM"): (1, 105.0,  90.0),
    ("CM", "OM", "NC"): (2, 125.0, 112.0),
    ("CM", "OM", "NH"): (2, 120.0, 150.0),
    ("CM", "CM", "OM"): (10,  90.0, 100.0),
    ("CM", "CM", "CM"): (10, 150.0,  40.0),
}
_dihedral_list = [
    ("OM", "CM", "CM", "OM", 1,   0, 0.66, 4),
    ("OM", "CM", "CM", "OM", 1, 180, 1.85, 1),
    ("OM", "CM", "CM", "OM", 1,  30, 0.12, 1),
    ("OM", "CM", "CM", "OM", 1,  60, 0.96, 3),
    ("CM", "CM", "CM", "CM", 1,  45, 0.17, 4),
    ("CM", "CM", "CM", "CM", 1, 115, 0.27, 3),
    ("CM", "CM", "CM", "CM", 1,  20, 1.19, 2),
    ("CM", "CM", "CM", "CM", 1, 160, 1.17, 1),
]
dihedral_params = defaultdict(list)
for at1, at2, at3, at4, func, phi0, kphi, n in _dihedral_list:
    dihedral_params[(at1, at2, at3, at4)].append((func, phi0, kphi, n))

atom_order_names = ["CM", "OM", "NC", "NH"]
charges_by_type = {"SCM": 0.0, "N4M": 0.0, "SQ2": 1.0, "SQ2p": 1.0, "N3a": 0.0}

def _is_section_header(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return any(s.startswith(name) for name in SECTION_NAMES)

def _find_section(lines: List[str], name_prefix: str) -> Tuple[int, int]:
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith(name_prefix):
            j = i + 1
            while j < len(lines) and not _is_section_header(lines[j]):
                j += 1
            return i, j
        i += 1
    return -1, -1

def _sort_section_block(lines: List[str], header_idx: int) -> List[str]:
    j = header_idx + 1
    k = j
    while k < len(lines) and not _is_section_header(lines[k]):
        k += 1
    block = lines[j:k]
    data = []
    nondata = []
    for ln in block:
        st = ln.strip()
        if not st:
            nondata.append(ln)
            continue
        tok0 = st.split()[0]
        try:
            aid = int(tok0)
            data.append((aid, ln))
        except ValueError:
            nondata.append(ln)
    data.sort(key=lambda x: x[0])
    new_block = nondata + [ln for _, ln in data]
    return lines[:j] + new_block + lines[k:]

def sort_lammps_data(in_path: Path, out_path: Path, sort_atoms=True, sort_velocities=True) -> None:
    text = Path(in_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    if sort_atoms:
        i_atoms, _ = _find_section(lines, "Atoms")
        if i_atoms >= 0:
            lines = _sort_section_block(lines, i_atoms)
    if sort_velocities:
        i_vel, _ = _find_section(lines, "Velocities")
        if i_vel >= 0:
            lines = _sort_section_block(lines, i_vel)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

def _parse_atoms_records(atom_lines: List[str]) -> List[Tuple[int,int,int,float,float,float,float,str]]:
    recs = []
    for ln in atom_lines:
        raw = ln.rstrip("\n")
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        tok = s.split()
        if len(tok) < 7:
            continue
        try:
            aid = int(tok[0])
            mol = int(tok[1])
            atype = int(tok[2])
            q = float(tok[3])
            x = float(tok[4]); y = float(tok[5]); z = float(tok[6])
        except Exception:
            continue
        recs.append((aid, mol, atype, q, x, y, z, raw))
    return recs

def read_atoms_sorted(data_sorted_path: Path) -> List[Tuple[int,int,int,float,float,float,float,str]]:
    lines = data_sorted_path.read_text(encoding="utf-8").splitlines()
    i0, i1 = _find_section(lines, "Atoms")
    if i0 < 0:
        raise SystemExit("No 'Atoms' section found")
    atom_lines = lines[i0+1:i1]
    recs = _parse_atoms_records(atom_lines)
    recs.sort(key=lambda r: r[0])
    return recs

def build_chain_monomers(
    recs: List[Tuple[int,int,int,float,float,float,float,str]],
    exclude_types: List[int]
) -> Dict[int, List[Tuple[tuple,tuple,tuple]]]:
    by_mol: dict[int, List[Tuple[int,int,int,float,float,float,float,str]]] = defaultdict(list)
    for rec in recs:
        aid, mol, atype, q, *_ = rec
        if atype in exclude_types:
            continue
        by_mol[mol].append(rec)

    chains: dict[int, List[Tuple[tuple,tuple,tuple]]] = {}
    for mol, atoms in by_mol.items():
        atoms.sort(key=lambda r: r[0])
        if len(atoms) % 3 != 0:
            raise SystemExit(f"mol {mol}: polymer atom count {len(atoms)} not divisible by 3")
        mons = []
        for i in range(0, len(atoms), 3):
            cm, om, xh = atoms[i], atoms[i+1], atoms[i+2]
            mons.append((cm, om, xh))
        chains[mol] = mons
    return chains

def add_bond(lines: List[str], i: int, j: int, at_i: str, at_j: str):
    key = tuple(sorted((at_i, at_j), key=lambda x: atom_order_names.index(x)))
    if key not in bond_params:
        raise KeyError(f"Bond param missing for key={key}")
    func, r0, kb = bond_params[key]
    lines.append(f"{i:5d} {j:5d} {func:5d} {r0:12.3f} {kb:12.0f}")

def add_angle(lines: List[str], i: int, j: int, k: int, at_i: str, at_j: str, at_k: str):
    key = (at_i, at_j, at_k)
    if key not in angle_params:
        raise KeyError(f"Angle param missing for key={key}")
    func, theta0, kth = angle_params[key]
    lines.append(f"{i:5d} {j:5d} {k:5d} {func:5d} {theta0:12.1f} {kth:12.1f}")

def add_dihedral(lines: List[str], a: int, b: int, c: int, d: int, at_a: str, at_b: str, at_c: str, at_d: str):
    key = (at_a, at_b, at_c, at_d)
    if key not in dihedral_params:
        raise KeyError(f"Dihedral param missing for key={key}")
    for func, phi0, kphi, n in dihedral_params[key]:
        lines.append(f"{a:5d} {b:5d} {c:5d} {d:5d} {func:5d} {phi0:12.1f} {kphi:12.2f} {n:12d}")

def write_chain_itp_from_records(
    out_dir: Path, chain_index: int, n_polm: int,
    monomers: List[Tuple[int,int,int,float,float,float,float]],
    pro_threshold: float = 0.5
):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(monomers)

    lines = []
    lines.append("[ moleculetype ]")
    lines.append("; name  nrexcl")
    lines.append(f"POL_{chain_index}    1\n")

    lines.append("[ atoms ]")
    lines.append(";   nr    type   resnr  residu    atom    cgnr  charge    ; note")

    specs = []
    for m in range(min(n_polm, n_total)):
        resnr = m + 1
        specs.append(("SCM", resnr, "PM", "CM", charges_by_type["SCM"], f"PM CM m={m}"))
        specs.append(("N4M", resnr, "PM", "OM", charges_by_type["N4M"], f"PM OM m={m}"))
        specs.append(("SQ2", resnr, "PM", "NC", charges_by_type["SQ2"], f"PM NC m={m}"))
    for rel in range(n_polm, n_total):
        resnr = rel + 1
        x_q = monomers[rel][2][3]
        is_pro = (x_q > pro_threshold)
        atype_third = "SQ2p" if is_pro else "N3a"
        specs.append(("SCM", resnr, "PD", "CM", charges_by_type["SCM"], f"PD CM m={rel-n_polm}"))
        specs.append(("N4M", resnr, "PD", "OM", charges_by_type["N4M"], f"PD OM m={rel-n_polm}"))
        specs.append((atype_third, resnr, "PD", "NH", charges_by_type[atype_third],
                      f"PD NH m={rel-n_polm} PRO={1 if is_pro else 0}"))

    for i, (atype, resnr, residu, atomn, q, note) in enumerate(specs, start=1):
        lines.append(f"{i:6d} {atype:>7s} {resnr:7d} {residu:>7s} {atomn:>7s} {i:7d} {q:7.3f}    ; {note}")

    bond_lines = []
    for m in range(n_total):
        cm = 3*m + 1
        om = 3*m + 2
        xh = 3*m + 3
        add_bond(bond_lines, cm, om, "CM", "OM")
        atom_x = "NC" if m < n_polm else "NH"
        add_bond(bond_lines, om, xh, "OM", atom_x)
        if m < n_total - 1:
            add_bond(bond_lines, cm, 3*(m+1) + 1, "CM", "CM")
    lines.append("\n[ bonds ]")
    lines.append(";  ai    aj funct           c0           c1")
    lines.extend(bond_lines)

    angle_lines = []
    for m in range(n_total):
        cm = 3*m + 1
        om = 3*m + 2
        xh = 3*m + 3
        atom_x = "NC" if m < n_polm else "NH"
        if m < n_total - 1:
            add_angle(angle_lines, om, cm, 3*(m+1) + 1, "OM", "CM", "CM")
            add_angle(angle_lines, cm, 3*(m+1) + 1, 3*(m+1) + 2, "CM", "CM", "OM")
        add_angle(angle_lines, cm, om, xh, "CM", "OM", atom_x)
        if m < n_total - 2:
            add_angle(angle_lines, cm, 3*(m+1) + 1, 3*(m+2) + 1, "CM", "CM", "CM")
    lines.append("\n[ angles ]")
    lines.append(";  ai    aj    ak funct           c0           c1")
    lines.extend(angle_lines)

    dih_lines = []
    for m in range(n_total - 1):
        add_dihedral(dih_lines, 3*m + 2, 3*m + 1, 3*(m+1) + 1, 3*(m+1) + 2, "OM", "CM", "CM", "OM")
    for m in range(n_total - 3):
        add_dihedral(dih_lines, 3*m + 1, 3*(m+1) + 1, 3*(m+2) + 1, 3*(m+3) + 1, "CM", "CM", "CM", "CM")
    lines.append("\n[ dihedrals ]")
    lines.append(";  ai    aj    ak    al funct           c0           c1           c2")
    lines.extend(dih_lines)

    lines.append("\n[ position_restraints ]")
    lines.append(";     i funct       fcx        fcy        fcz")
    lines.append("      1 1 1000 1000 1000")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"POL_{chain_index}.itp"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _gro_name5(s: str) -> str:
    s = (s or "").strip()
    return s[:5]

def gro_atom_line(
    resid: int, resname: str, atomname: str, atomid: int,
    x: float, y: float, z: float,
    vx: Optional[float] = None, vy: Optional[float] = None, vz: Optional[float] = None
) -> str:
    resid_wrapped = resid % 100000
    atomid_wrapped = atomid % 100000
    rn = _gro_name5(resname)
    an = _gro_name5(atomname)
    core = f"{resid_wrapped:5d}{rn:<5}{an:>5}{atomid_wrapped:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
    if vx is not None and vy is not None and vz is not None:
        core += f"{vx:8.4f}{vy:8.4f}{vz:8.4f}"
    return core + "\n"

def write_gro(
    filename: Path,
    title: str,
    records: List[Dict[str, Any]],
    box: List[float]
):
    if any(int(r["resid"]) > 99999 or int(r["atomid"]) > 99999 for r in records):
        print("[warn] GRO: resid/atomid exceed 99999; will wrap mod 100000 in output.")
    with open(filename, "w", encoding="utf-8") as f:
        f.write((title or "Generated by script") + "\n")
        f.write(f"{len(records)}\n")
        for rec in records:
            f.write(
                gro_atom_line(
                    resid=int(rec["resid"]),
                    resname=str(rec["resname"]),
                    atomname=str(rec["atomname"]),
                    atomid=int(rec["atomid"]),
                    x=float(rec["x"]), y=float(rec["y"]), z=float(rec["z"]),
                    vx=rec.get("vx"), vy=rec.get("vy"), vz=rec.get("vz"),
                )
            )
        if len(box) == 3:
            f.write(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}\n")
        elif len(box) == 9:
            f.write("".join(f"{v:10.5f}" for v in box) + "\n")
        else:
            raise ValueError("Box must have 3 or 9 values")

def parse_gro(path: Path) -> Tuple[str, int, List[Dict[str, Any]], List[float]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise SystemExit("Invalid GRO: too few lines")
    title = lines[0].rstrip("\n")
    try:
        natoms = int(lines[1].strip())
    except Exception:
        raise SystemExit("Invalid GRO: atom count line")
    atom_lines = lines[2:2+natoms]
    if len(atom_lines) != natoms:
        raise SystemExit("Invalid GRO: atom lines count mismatch")
    box_line = lines[2+natoms].strip()
    box = [float(v) for v in box_line.split()]
    recs: List[Dict[str, Any]] = []
    for i, ln in enumerate(atom_lines, start=1):
        if len(ln) < 44:
            raise SystemExit(f"Invalid GRO atom line {i}: too short")
        resid = int(ln[0:5])
        resn = ln[5:10].strip()
        atn = ln[10:15].strip()
        atomid = int(ln[15:20])
        rest = ln[20:].strip().split()
        if len(rest) < 3:
            raise SystemExit(f"Invalid GRO atom line {i}: missing coords")
        x, y, z = map(float, rest[:3])
        vx = vy = vz = None
        if len(rest) >= 6:
            vx, vy, vz = map(float, rest[3:6])
        recs.append({
            "resid": resid, "resname": resn, "atomname": atn, "atomid": atomid,
            "x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz
        })
    return title, natoms, recs, box

def update_gro_by_types(
    gro_in: Path, gro_out: Path,
    types_by_id_1based: List[int],
    w_type: int = 7,
    cl_type: int = 8,
    move_cl_to_end: bool = True,
    renumber_ions_after_nonions: bool = True,
    title_suffix: str = " (W/CL normalized; CL at end)"
):
    title, natoms, recs, box = parse_gro(gro_in)
    if natoms != len(types_by_id_1based):
        raise SystemExit(f"GRO atoms ({natoms}) != types length ({len(types_by_id_1based)})")

    mapped = []
    for i, rec in enumerate(recs, start=1):
        t = types_by_id_1based[i-1]
        resn = rec["resname"]
        atn = rec["atomname"]
        if t == w_type:
            resn, atn = "W", "W"
        elif t == cl_type:
            resn, atn = "CL", "CL"
        mapped.append({**rec, "resname": resn, "atomname": atn, "type": t})

    if move_cl_to_end:
        non_cl = [r for r in mapped if r["type"] != cl_type]
        cls = [r for r in mapped if r["type"] == cl_type]
        ordered = non_cl + cls
    else:
        ordered = list(mapped)

    if renumber_ions_after_nonions:
        is_ion = lambda r: (r["type"] == w_type) or (r["type"] == cl_type)
        max_nonion_resid = 0
        for r in ordered:
            if not is_ion(r):
                if r["resid"] > max_nonion_resid:
                    max_nonion_resid = r["resid"]
        next_resid = max_nonion_resid + 1 if max_nonion_resid >= 0 else 1
        for r in ordered:
            if is_ion(r):
                r["resid"] = next_resid
                next_resid += 1

    for k, r in enumerate(ordered, start=1):
        r["atomid"] = k

    out_title = (title or "Generated") + (title_suffix or "")
    write_gro(gro_out, out_title, ordered, box)

def is_section_header(s: str) -> bool:
    s = s.strip()
    return s.startswith("[") and s.endswith("]")

def update_top_molecules_counts(
    top_path: Path,
    species_counts: Dict[str, int],
    species_width: int = 15,
    ensure_present: bool = True,
) -> None:
    lines = top_path.read_text(encoding="utf-8").splitlines(True)
    n = len(lines)

    mol_header_idx = -1
    for i, raw in enumerate(lines):
        if raw.strip().lower().startswith("[ molecules"):
            mol_header_idx = i
            break
    if mol_header_idx < 0:
        raise SystemExit(f"No [ molecules ] section found in {top_path}")

    j = mol_header_idx + 1
    while j < n and not is_section_header(lines[j].strip()):
        j += 1
    block_start = mol_header_idx + 1
    block_end = j

    changed = set()
    for i in range(block_start, block_end):
        raw = lines[i]
        s = raw.strip()
        if not s or s.startswith(";"):
            continue
        parts = s.split()
        name = parts[0]
        if name in species_counts:
            lines[i] = f"{name:<{species_width}}{species_counts[name]}\n"
            changed.add(name)

    if ensure_present:
        to_add = [k for k in species_counts.keys() if k not in changed]
        if to_add:
            insertion: List[str] = []
            for name in to_add:
                insertion.append(f"{name:<{species_width}}{species_counts[name]}\n")
            lines[block_end:block_end] = insertion

    top_path.write_text("".join(lines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(
        description="Sort LAMMPS data -> per-chain ITPs -> (optional) GRO normalize -> (optional) update gmx_full.top W/CL counts"
    )
    ap.add_argument("--in-data", required=True, help="Input LAMMPS data (unsorted)")
    ap.add_argument("--out-data-sorted", default=None, help="Output sorted data (default: <in>.sorted)")
    ap.add_argument("--out-itp-dir", default="itp_from_data", help="Output directory for per-chain ITPs")
    ap.add_argument("--n-polm", type=int, required=True, help="Number of PM monomers per chain")
    ap.add_argument("--exclude-types", default="7,8",
                    help="Comma-separated atom type IDs to exclude when building chains (e.g., '7,8')")
    ap.add_argument("--pro-threshold", type=float, default=0.5,
                    help="Charge threshold for PD third bead protonated (SQ2p) vs unprotonated (N3a)")

    ap.add_argument("--gro-in", default=None, help="Input GRO to normalize (optional)")
    ap.add_argument("--gro-out", default=None, help="Output GRO path (required if --gro-in is given)")
    ap.add_argument("--w-type", type=int, default=7, help="LAMMPS type id for W")
    ap.add_argument("--cl-type", type=int, default=8, help="LAMMPS type id for CL")
    ap.add_argument("--no-move-cl", action="store_true", help="Do not move CL atoms to the end of GRO")
    ap.add_argument("--no-renumber-ions", action="store_true", help="Do not renumber ion residues (W/CL)")

    ap.add_argument("--update-top", default=None, help="Path to gmx_full.top to update [ molecules ] W/CL counts")
    ap.add_argument("--species-W-name", default="W", help="Species name for water in [ molecules ] (default: W)")
    ap.add_argument("--species-CL-name", default="CL", help="Species name for counterion in [ molecules ] (default: CL)")
    ap.add_argument("--species-column-width", type=int, default=15, help="Column width for species name")
    ap.add_argument("--no-ensure", action="store_true", help="Do not append missing species if not present")

    ns = ap.parse_args()

    in_data = Path(ns.in_data).resolve()
    out_sorted = Path(ns.out_data_sorted).resolve() if ns.out_data_sorted else in_data.with_suffix(in_data.suffix + ".sorted")
    out_itp_dir = Path(ns.out_itp_dir).resolve()
    exclude_types = [int(s) for s in ns.exclude_types.split(",") if s.strip()]
    move_cl = not ns.no_move_cl
    renum_ions = not ns.no_renumber_ions

    gro_in = Path(ns.gro_in).resolve() if ns.gro_in else None
    gro_out = Path(ns.gro_out).resolve() if ns.gro_in and ns.gro_out else None
    if gro_in and not gro_out:
        raise SystemExit("--gro-out is required when --gro-in is provided")

    sort_lammps_data(in_data, out_sorted, sort_atoms=True, sort_velocities=True)
    print(f"[done] sorted data -> {out_sorted}")

    atoms = read_atoms_sorted(out_sorted)
    chains_by_mol = build_chain_monomers(atoms, exclude_types=exclude_types)
    out_itp_dir.mkdir(parents=True, exist_ok=True)
    for chain_idx, (mol_id, monomers) in enumerate(sorted(chains_by_mol.items(), key=lambda kv: kv[0])):
        write_chain_itp_from_records(out_itp_dir, chain_idx, int(ns.n_polm), monomers, pro_threshold=float(ns.pro_threshold))
    print(f"[done] wrote {len(chains_by_mol)} ITPs -> {out_itp_dir}")

    types_by_id = [rec[2] for rec in atoms]  # 1-based atom id order from sorted data

    if gro_in:
        update_gro_by_types(
            gro_in=gro_in,
            gro_out=gro_out,
            types_by_id_1based=types_by_id,
            w_type=int(ns.w_type),
            cl_type=int(ns.cl_type),
            move_cl_to_end=move_cl,
            renumber_ions_after_nonions=renum_ions,
            title_suffix=" (W/CL normalized; CL at end)" if move_cl else " (W/CL normalized)"
        )
        print(f"[done] wrote GRO -> {gro_out}")

    W_count = sum(1 for t in types_by_id if t == int(ns.w_type))
    CL_count = sum(1 for t in types_by_id if t == int(ns.cl_type))
    print(f"[counts] W={W_count}, CL={CL_count}")

    if ns.update_top:
        top_path = Path(ns.update_top).resolve()
        if not top_path.exists():
            raise SystemExit(f"File not found: {top_path}")
        update_top_molecules_counts(
            top_path=top_path,
            species_counts={str(ns.species_W_name): W_count, str(ns.species_CL_name): CL_count},
            species_width=int(ns.species_column_width),
            ensure_present=(not ns.no_ensure),
        )
        print(f"[done] updated {top_path.name}: {ns.species_W_name}={W_count}, {ns.species_CL_name}={CL_count}")

    print("[all done]")

if __name__ == "__main__":
    main()
