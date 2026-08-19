#!/usr/bin/bash

# Take in arguments and construct a DAG

export b0=$(basename $0)
export DOREWRITE=0
usage() {
    cat >&2 <<EOF
Submit a series of flux extraction calls to the grid.
Each call handles one range of origin position and half-side, in a square detector.
It also handles a certain range of the dk2nu flux files.

There are two primary steps. 
The first initialises a working area and writes preliminary config files.
These are then edited before jobs are submitted to be run.
The second step consists of running the produced DAG file.

This script handles the first step. Arguments:

 ${b0} 
       -T | --top         Top level directory to write output files to.
	    --f0          Index of first dk2nu file to run.
	    --f1          Index of last  dk2nu file to run.
       -b | --banalyser   Path to beamHist Python analyser script.
            --x1          Lower end of the x positions. (cm)
            --x2          Upper end of the x positions.
	    --y1
	    --y2          As in x.
	    --size        Half side of the square detector. (cm)
	    --z1
	    --z2          As in x.
	    --dz          As in x.
	    --rewrite     Whether to rewrite top level directory.

Use "${b0} -h" to show this help message.
EOF

    exit 0
}

process_args() {

    PRINTUSAGE=0
    
    TEMP=$(getopt -n $0 -s bash -a \
     --longoptions="help top: nfiles: list: \
     banalyser: f0: f1: x1: x2: dx: y1: y2: dy: size: z1: z2: dz: \
     rewrite" \
     -o hT:n:b: -- "$@") || exit 1

    eval set -- "${TEMP}"
    unset TEMP

    let iarg=0
    set -u
    while [ $# -gt 0 ]; do
	let iarg=${iarg}+1
	case "$1" in
	    "--"              ) shift                       ; break ;;
	    -h | --help       ) PRINTUSAGE=1                        ;;
	    -T | --top        ) export OUTPUTTOP="$2"       ; shift ;;
	    -n | --nfiles     ) export NFILES="$2"          ; shift ;;
	    -l | --list       ) export LIST_PATH="$2"       ; shift ;;
	    -b | --banalyser  ) export BANALYSER="$2"       ; shift ;;
	    --x1              ) export X1="$2"              ; shift ;;
	    --x2              ) export X2="$2"              ; shift ;;
	    --dx              ) export DX="$2"              ; shift ;;
	    --y1              ) export Y1="$2"              ; shift ;;
	    --y2              ) export Y2="$2"              ; shift ;;
	    --dy              ) export DY="$2"              ; shift ;;
	    --size            ) export SIZE="$2"            ; shift ;;
	    --z1              ) export Z1="$2"              ; shift ;;
	    --z2              ) export Z2="$2"              ; shift ;;
	    --dz              ) export DZ="$2"              ; shift ;;
	    --f0              ) export FIRST_FILE_IDX="$2"  ; shift ;;
	    --f1              ) export LAST_FILE_IDX="$2"   ; shift ;;
	    --rewrite         ) export DOREWRITE=1                  ;;
	    -*                ) echo "unknown flag $opt ($1)" ; PRINTUSAGE=1 ;;
	esac
	shift # eat up the arg we just used
    done
    set +u

    if [[ ${PRINTUSAGE} -eq 1 ]] ; then
	usage
    fi
}

process_args "$@"

if [ ! -d ${OUTPUTTOP} ] || [ ${DOREWRITE} -eq 1 ] ; then
    echo -e "${OUTGREEN}Making directory ${OUTPUTTOP} -- this is where your job and config files will live${OUTNOCOL}"
    if [[ -d ${OUTPUTTOP} ]] ; then
	rm -rf ${OUTPUTTOP}
    fi
    mkdir -p ${OUTPUTTOP}
    mkdir -p ${OUTPUTTOP}/cfg
    mkdir -p ${OUTPUTTOP}/bin
    mkdir -p ${OUTPUTTOP}/tarball
    mkdir -p ${OUTPUTTOP}/work-products/
elif [ -d ${OUTPUTTOP} ] && [ ${DOREWRITE} -eq 0 ] ; then
    echo -e "${OUTRED}Directory ${OUTPUTTOP} exists, not overwriting. Pass --rewrite if you want this to be rewritten.${OUTNOCOL}"
    exit 1
fi

if [[ -z ${BANALYSER} ]] ; then
    echo -e "${OUTRED}No python analyser for beamHist found. Did you pass -b | --banalyser?${OUTNOCOL}"
    exit 1
fi

if [ -z ${X1} ] || [ -z ${X2} ] || [ -z ${DX} ] ; then
    echo -e "${OUTRED}Check your volume ranges... Not submitting. (reason: at least one required arg is empty)${OUTNOCOL}"
    exit 1
fi
if [ -z ${Y1} ] || [ -z ${Y2} ] || [ -z ${DY} ] ; then
    echo -e "${OUTRED}Check your volume ranges... Not submitting. (reason: at least one required arg is empty)${OUTNOCOL}"
    exit 1
fi
if [ -z ${SIZE} ] ; then
    echo -e "${OUTRED}Check your volume ranges... Not submitting. (reason: at least one required arg is empty)${OUTNOCOL}"
    exit 1
fi
if [ -z ${Z1} ] || [ -z ${Z2} ] || [ -z ${DZ} ] ; then
    echo -e "${OUTRED}Check your volume ranges... Not submitting. (reason: at least one required arg is empty)${OUTNOCOL}"
    exit 1
fi
if [ $(echo "${X2} - ${X1} < 0" | bc -l) == 1 ] || [ $(echo "${Y2} - ${Y1} < 0" | bc -l) == 1 ] || [ $(echo "${Z2} - ${Z1} < 0" | bc -l) == 1 ] ; then
    echo -e "${OUTRED}Check your volume ranges... Not submitting. (reason: at least one lower bound is greater than its upper bound)${OUTNOCOL}"
    exit 1
fi
if [ $(echo "${DX} <= 0" | bc -l) == 1 ] || [ $(echo "${DY} <= 0" | bc -l) == 1 ] || [ $(echo "${DZ} <= 0" | bc -l) == 1 ] ; then
    echo -e "${OUTRED}Check your volume ranges... Not submitting. (reason: at least one step size is non-positive)${OUTNOCOL}"
    exit 1
fi
if [ $(echo "${SIZE} <= 0" | bc -l) == 1 ] ; then
    echo -e "${OUTRED}Check your volume ranges... Not submitting. (reason: half side non-positive)${OUTNOCOL}"
    exit 1
fi

if [ -z ${FIRST_FILE_IDX} ] || [ -z ${LAST_FILE_IDX} ] ; then
    echo -e "${OUTRED}Check your file ranges... Not submitting. (reason: f0 or f1 unset)${OUTNOCOL}"
    exit 1
fi
if [ ${FIRST_FILE_IDX} -gt ${LAST_FILE_IDX} ] || [ ${FIRST_FILE_IDX} -le 0 ] ; then
    echo -e "${OUTRED}Check your file ranges... Not submitting. (f0 = ${FIRST_FILE_IDX}, f1 = ${LAST_FILE_IDX})${OUTNOCOL}"
    exit 1
fi

if [ -z ${LIST_PATH} ] || [ ! -f ${LIST_PATH} ] ; then
    echo -e "${OUTRED}Cannot read input file list at ${LIST_PATH}${OUTNOCOL}"
    exit 1
fi

# Tar up stuff
echo -e "${OUTLTPURPLE}Making tarball.."
FLIST=""
cp ${BANALYSER} ${OUTPUTTOP}/tarball && FLIST=$(basename ${BANALYSER})" "${FLIST}
cp ${LIST_PATH} ${OUTPUTTOP}/tarball && FLIST=$(basename ${LIST_PATH})" "${FLIST}
tar -C ${OUTPUTTOP}/tarball -cvjSf ${OUTPUTTOP}/tarball/analysis.tar ${FLIST}
echo -e "Done making tarball.${OUTNOCOL}"

# write the binary file
export EXECFILE=${OUTPUTTOP}/bin/analyse_beammc.sh

cat > ${EXECFILE} <<EOF
# Define_cfg takes the coordinates (X, Y) to be run over, the z0, z1, and dz, the size, and the first and last files.
# like 73.78 0.0 0.0 500.0 10.0 20.0 1 10 does x, y = 73.78, 0.0, z = (0, 10, ..., 500) cm, of half side 20 cm, files 1 --> 10
define_cfg()
{
  source /cvmfs/fermilab.opensciencegrid.org/packages/common/spack/current/NULL/share/spack/setup-env.sh
  eval \`spack load --sh fife-utils@3.7.8\`

  export X0=\$1
  export Y0=\$2
  export Z0=\$3
  export Z1=\$4
  export DZ=\$5
  export SIZE=\$6
  export F0=\$7
  export F1=\$8

  echo \$X0 \$Y0 \$Z0 \$Z1 \$DZ \$SIZE \$F0 \$F1

  export JOBBANALYSER=\$(pwd)/$(basename ${BANALYSER})
}

setup_python()
{
  # Setup a virtual environment
  python3 -m venv .
  source bin/activate
  # Pull in pip, numpy, pandas, pyarrow, uproot
  python3 -m pip install --upgrade pip > /dev/null && echo "Upgraded pip"
  python3 -m pip install numpy > /dev/null && echo "Installed numpy"
  python3 -m pip install pandas uproot > /dev/null && echo "Installed pandas and uproot"
  python3 -m pip install pyarrow > /dev/null && echo "Installed pyarrow"
  python3 -m pip install tqdm colorama > /dev/null && echo "Installed tqdm and colorama"

  echo "Python setup OK"
}

export BASEDIR=\$(pwd)
export WORKDIR=\$(mktemp -d -p \${BASEDIR})
mkdir -p \$WORKDIR && cd \$WORKDIR

define_cfg \$1 \$2 \$3 \$4 \$5 \$6 \$7 \$8

# copy in the tarball
tSleep=\$(echo "\${RANDOM} % 30" | bc)
echo "Sleeping for \${tSleep} s..."
sleep \${tSleep}
ifdh cp -D ${OUTPUTTOP}/tarball/analysis.tar ./
tar -xvjSf analysis.tar
echo
ls
echo
setup_python

# Symlink all the files to read
mkdir links && cd links
while IFS= read -r line ; do
      ln -s \${line}
done < <(cat \${WORKDIR}/$(basename ${LIST_PATH}) | head -n \${F1} | tail -n \$(echo "\${F1} - \${F0} + 1" | bc))
cd -

# Analyse the histograms
ZLIST=""
nz=\$(echo "(\${Z1} - \${Z0}) / \${DZ}" | bc)
for i in \$(seq 0 \${nz}) ; do
    ZLIST=\${ZLIST}" "\$(printf "%4.3f" \$(echo "\${DZ} * \${i}" | bc -l))
done
cmd="python3 \${JOBBANALYSER} -i links/ -o \$(pwd)/hists.root -t 1 --detector \${X0} \${Y0} 11000.0 -s \${SIZE} -z \${ZLIST}"
echo \${cmd}
eval \${cmd}

# make an md5sum
MSUM=\$(md5sum hists.root | awk -F " " '{print \$1}') 
OUTFILE=\$(pwd)/hists-\${MSUM:0:8}-\${MSUM:8:8}-\${MSUM:16:8}-\${MSUM:24:8}.root
mv hists.root \${OUTFILE}

# Set some attributes for querying metadata
MDFILE=\$(pwd)/md-\${MSUM:0:8}-\${MSUM:8:8}-\${MSUM:16:8}-\${MSUM:24:8}.json

cat > \${MDFILE} <<EAF
{
    "x": "\${X0}",
    "y": "\${Y0}",
    "half_side": "\${SIZE}",
    "z0": "\${Z0}",
    "dz": "\${DZ}",
    "dk2nu_source": "$(basename ${LIST_PATH})",
    "first_file_index": "\${F0}",
    "last_file_index": "\${F1}"
}
EAF

# copy back to job directory
# It's silly that I can't declare the md there, but apparently dCache doesn't support metadata...
ifdh cp -D \${OUTFILE} ${OUTPUTTOP}/work-products/
ifdh cp -D \${MDFILE} ${OUTPUTTOP}/work-products/

# cleanup
echo "Done, cleaning up"
cd \${BASEDIR} && rm -rf \${WORKDIR}/
EOF
chmod u+x ${EXECFILE}

# Now write a DAG file.
JOBSUBMAIN="jobsub_submit -n -G $(id -ng)  --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC  --append_condor_requirements='(TARGET.HAS_CVMFS_sbn_opensciencegrid_org==true)'"
RESOURCES="--expected-lifetime 24h --disk 8GB --memory 2GB"
JOBSUBFULL=${JOBSUBMAIN}" "${RESOURCES}
echo -e "<parallel>" > g4bnb-extract.dag

# How many files will I want?
# One per (x, y) voxel, and I want NFILES dk2nu files per job

NCOMP=$(echo "${LAST_FILE_IDX} - ${FIRST_FILE_IDX}" | bc)
if [[ ${NFILES} -gt ${NCOMP} ]] ; then
    NFILES=${NCOMP}
fi
nf=$(echo "(${LAST_FILE_IDX} - ${FIRST_FILE_IDX} + ${NFILES}-1)/${NFILES}" | bc)

declare -a XVALS=() ; declare -a YVALS=()
nx=$(echo "(${X2} - ${X1}) / ${DX}" | bc)
for i in $(seq 0 ${nx}) ; do
    XVALS+=($(echo "${X1} + ${i} * ${DX}" | bc -l))
done
ny=$(echo "(${Y2} - ${Y1}) / ${DY}" | bc)
for i in $(seq 0 ${ny}) ; do
    YVALS+=($(echo "${Y1} + ${i} * ${DY}" | bc -l))
done

for yy in "${YVALS[@]}" ; do
    for xx in "${XVALS[@]}" ; do
	FIRST_FILE=${FIRST_FILE_IDX}
	for i in $(seq 1 ${nf}) ; do
	    LAST_FILE=$(echo "${FIRST_FILE}+${NFILES}-1" | bc)
	    if [[ ${LAST_FILE_IDX} -lt ${LAST_FILE} ]] ; then
		LAST_FILE=${LAST_FILE_IDX}
	    fi
	    echo -e "${JOBSUBFULL} file://${EXECFILE} ${xx} ${yy} ${Z1} ${Z2} ${DZ} ${SIZE} ${FIRST_FILE} ${LAST_FILE}" >> g4bnb-extract.dag
	    FIRST_FILE=$(echo "${FIRST_FILE} + ${NFILES}" | bc)
	done
    done
done
echo -e "</parallel>" >> g4bnb-extract.dag

mv g4bnb-extract.dag ${OUTPUTTOP}/cfg/

echo -e "${OUTCYAN}Exit status 0${OUTNOCOL}"
