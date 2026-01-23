
export GMX_MAXCONSTRWARN=-1
export OMP_NUM_THREADS=6
export GMX_MAXBACKUP=-1
set -euo pipefail
IFS=$'\n\t'

# ============================================================
# Configuration
# ============================================================
GMX_CMD="gmx"
N_THREADS_OMP=6
N_MPI=8
MC_BIN="../MC_PH/bin/MC"

ITER_START=1
ITER_END=1000

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
FILES_DIR="${ROOT_DIR}/files"

WORK_DIR="${ROOT_DIR}/workspace"
ARCHIVE_DIR="${ROOT_DIR}/results"
REPORT_DIR="${ARCHIVE_DIR}/reports"
GRO_DIR="${ARCHIVE_DIR}/gros"
STATE_JSON="${ARCHIVE_DIR}/states.json"

MAKE_REF_PY="${FILES_DIR}/make_ref_gro.py"
FIXED_NDX="${FILES_DIR}/S_P.ndx"
FIXED_GROUP="SUB_PM"

# ============================================================
# Safety checks
# ============================================================
[[ -d "${FILES_DIR}" ]] || { echo "ERROR: files/ not found: ${FILES_DIR}"; exit 1; }
[[ -f "${FIXED_NDX}" ]] || { echo "ERROR: fixed index file not found: ${FIXED_NDX}"; exit 1; }
[[ -f "${MAKE_REF_PY}" ]] || { echo "ERROR: make_ref_gro.py not found: ${MAKE_REF_PY}"; exit 1; }

# ============================================================
# Prepare directories
# ============================================================
rm -rf "${WORK_DIR}" "${ARCHIVE_DIR}"
mkdir -p "${WORK_DIR}" "${REPORT_DIR}" "${GRO_DIR}"

echo "=== Init: copy input files to workspace ==="
cp -r "${FILES_DIR}/"* "${WORK_DIR}/"
pushd "${WORK_DIR}" >/dev/null

# ============================================================
# Initialize top and build system
# ============================================================
W_run1=141223
CL_run1=3200
sed -i "19s|.*|W               ${W_run1}|"  "${WORK_DIR}/gmx_ini.top"
sed -i "20s|.*|CL              ${CL_run1}|" "${WORK_DIR}/gmx_ini.top"

echo "=== Step 0: grompp init ==="
${GMX_CMD} grompp -f run0.mdp -o run.tpr -c md.gro -r md.gro -p gmx_ini.top -maxwarn 10

${GMX_CMD} genion -s run.tpr -o 0.gro -nname CL -neutral <<EOF
6
EOF

W_run1=141222
CL_run1=3201
sed -i "19s|.*|W               ${W_run1}|"  "${WORK_DIR}/gmx_ini.top"
sed -i "20s|.*|CL              ${CL_run1}|" "${WORK_DIR}/gmx_ini.top"
${GMX_CMD} grompp -f run0.mdp -o 0.tpr -c 0.gro -r 0.gro -p gmx_ini.top -maxwarn 10

cp 0.gro ref0.gro

# ============================================================
# Initialize states.json
# ============================================================
python - "${STATE_JSON}" <<'PY'
import json, sys, os
path = sys.argv[1]
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump({}, f, indent=2)
PY

# ============================================================
# Helper functions
# ============================================================
get_state_field() {
  local it="$1"; local key="$2"
  python - "${STATE_JSON}" "${it}" "${key}" <<'PY'
import json, sys
path, it, key = sys.argv[1:]
try:
    with open(path) as f:
        d = json.load(f)
    v = d.get(str(it), {}).get(key, "")
    print("" if v is None else v)
except Exception:
    print("")
PY
}

write_state_record() {
  # Args: it Final_State Nprot Nh MC_Skipped
  local it="$1"
  local fs="$2"
  local nprot="$3"
  local nh="$4"
  local skipped="$5"

  python - "${STATE_JSON}" "${it}" "${fs}" "${nprot}" "${nh}" "${skipped}" <<'PY'
import json, sys
path, it, fs, nprot, nh, skipped = sys.argv[1:]

# Load file
try:
    with open(path) as f:
        d = json.load(f)
except FileNotFoundError:
    d = {}

rec = d.get(str(it), {})

# ---- Safe float parsing for Final_State ----
try:
    rec["Final_State"] = float(str(fs).split()[0])
except Exception:
    rec["Final_State"] = None

# Optional fields
if nprot.strip():
    try: rec["Nprot"] = int(nprot)
    except: rec["Nprot"] = nprot
if nh.strip():
    try: rec["Nh"] = int(nh)
    except: rec["Nh"] = nh

rec["MC_Skipped"] = (str(skipped).lower() == "true")
d[str(it)] = rec

# Numeric sort by iteration
d_sorted = dict(sorted(d.items(), key=lambda kv: int(kv[0])))

with open(path, "w") as f:
    json.dump(d_sorted, f, indent=2)
PY
}

parse_report_snapshot() {
  local report_path="$1"
  awk '
    $0 ~ /^\[SNAPSHOT\]/ {
      fs=""; np=""; nh="";
      for(i=1;i<=NF;i++){
        if($i ~ /^Final_State=/){ split($i,a,"="); fs=a[2]; }
        if($i ~ /^Nprot=/){ split($i,a,"="); np=a[2]; }
        if($i ~ /^Nh=/){ split($i,a,"="); nh=a[2]; }
      }
      print fs, np, nh;
      exit
    }' "${report_path}"
}

# ============================================================
# Loop
# ============================================================
PREV_TPR="0.tpr"
PREV_GRO="0.gro"
PREV_TOP="prev_step.top"
cp gmx.top "${PREV_TOP}" 2>/dev/null || cp "${FILES_DIR}/gmx.top" "${PREV_TOP}"

echo "=== Loop start ==="

for ((it=${ITER_START}; it<=${ITER_END}; it++)); do
  echo "=========================================="
  echo "=== Iteration ${it} ==="
  echo "=========================================="

  SKIP_MC=0
  if [[ "${it}" -gt 1 ]]; then
    prev_it=$((it-1))
    prev_fs="$(get_state_field "${prev_it}" "Final_State")"
    if [[ -n "${prev_fs}" ]]; then
      SKIP_MC="$(awk -v x="${prev_fs}" 'BEGIN{eps=1e-8; if (x<=eps || 1.0-x<=eps) print 1; else print 0;}')"
    fi
  fi

  [[ "${SKIP_MC}" -eq 1 ]] && echo "[Decision] SKIP MC" || echo "[Decision] Run MC + MD"

  # ---------------------------
  # Prepare topology
  # ---------------------------
  if [[ "${SKIP_MC}" -eq 0 ]]; then
    cp "${FILES_DIR}/gmx.top" gmx.top
  else
    cp "${PREV_TOP}" gmx.top
  fi

  # ---------------------------
  # Monte Carlo (if needed)
  # ---------------------------
  if [[ "${SKIP_MC}" -eq 0 ]]; then
    [[ -f "${PREV_TPR}" && -f "${PREV_GRO}" ]] || { echo "ERROR: missing PREV_TPR/PREV_GRO"; exit 1; }

    echo "[MC] convert..."
    python convert.py "${PREV_TPR}" "${PREV_GRO}" MC_input.data --typemap types.json

    echo "[MC] run..."
    ${MC_BIN} MC_input.data params.in > mc_stdout.log 2>&1
    [[ -f report.txt ]] || { echo "ERROR: report.txt not found"; exit 1; }

    report_dst="${REPORT_DIR}/report_${it}.txt"
    mv report.txt "${report_dst}"

    read -r FINAL_STATE NPROT NH < <(parse_report_snapshot "${report_dst}")
    [[ -n "${FINAL_STATE}" ]] || { echo "ERROR: cannot parse Final_State"; exit 1; }

    write_state_record "${it}" "${FINAL_STATE}" "${NPROT}" "${NH}" "false"

    echo "[MC] postprocess..."
    python write_itp_gro.py \
        --in-data phase2.data \
        --out-data-sorted phase2_sorted.data \
        --out-itp-dir itp \
        --n-polm 50 \
        --exclude-types 1,7,8,9 \
        --pro-threshold 0.5 \
        --gro-in "${PREV_GRO}" \
        --gro-out best.gro \
        --w-type 8 --cl-type 9 \
        --update-top gmx.top \
        --species-W-name W \
        --species-CL-name CL

  else
    cp "${PREV_GRO}" best.gro
    prev_it=$((it-1))
    prev_fs="$(get_state_field "${prev_it}" "Final_State")"
    prev_np="$(get_state_field "${prev_it}" "Nprot")"
    prev_nh="$(get_state_field "${prev_it}" "Nh")"
    write_state_record "${it}" "${prev_fs}" "${prev_np}" "${prev_nh}" "true"
  fi

  [[ -f best.gro ]] || { echo "ERROR: best.gro not found"; exit 1; }

  # ---------------------------
  # Build reference GRO
  # ---------------------------
  echo "[REF] make ref.gro..."
  python "${MAKE_REF_PY}" \
    --coord best.gro \
    --ref ref0.gro \
    --ndx "${FIXED_NDX}" \
    --group "${FIXED_GROUP}" \
    --out ref.gro

  # ---------------------------
  # GROMACS execution
  # ---------------------------
  echo "[GMX] make_ndx..."
  ${GMX_CMD} make_ndx -f best.gro -o index.ndx <<'EOF'
!2
q
EOF

  echo "[GMX] grompp..."
  ${GMX_CMD} grompp -f run.mdp -o run.tpr -c best.gro -r ref.gro -p gmx.top -n index.ndx -maxwarn 10

  echo "[GMX] mdrun..."
  ${GMX_CMD} mdrun -deffnm md -s run.tpr -v \
    -ntmpi ${N_MPI} -ntomp ${N_THREADS_OMP} \
    -nb gpu -update gpu -pme gpu -npme 1 \
    -dlb auto -dd 7 1 1 -notunepme

  [[ -f md.gro ]] || { echo "ERROR: md.gro missing"; exit 1; }
  cp md.gro "${GRO_DIR}/${it}.gro"

  mv md.gro prev_step.gro
  mv run.tpr prev_step.tpr
  cp gmx.top prev_step.top

  PREV_GRO="prev_step.gro"
  PREV_TPR="prev_step.tpr"
  PREV_TOP="prev_step.top"
done

popd >/dev/null
echo "=== All done. Results in ${ARCHIVE_DIR} ==="
