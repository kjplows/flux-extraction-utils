#!/usr/bin/bash

# Take in arguments and construct a DAG

export b0=$(basename $0)
export DOREWRITE=0
usage() {
    cat >&2 <<EOF
Submit a series of beamHist calls to the grid, using the nominal simulation with pion and muon decays switched on.

There are two primary steps. 
The first initialises a working area and writes preliminary config files.
These are then edited before jobs are submitted to be run.
The second step consists of running the produced DAG file.

This script handles the first step. Arguments:

 ${b0} 
       -T | --top         Top level directory to write output files to.
       -v | --version     G4 version on UPS.
       -q | --qualifier   G4 qualifier on UPS.
            --setup       Which UPS to setup with.
       -n | --npot        How many POT to run overall.
       -r | --repeat      How many jobs will run (NPOT per job = NPOT / Njobs)
       -x | --executable  Path to the executable to use.
       -g | --geometry    Path to the geometry file to use.
       -s | --script      Path to the production script template to use.
       -b | --banalyser   Path to beamHist Python analyser script.
	    --rewrite     Whether to rewrite top level directory.

Use "${b0} -h" to show this help message.
EOF

    exit 0
}

process_args() {

    PRINTUSAGE=0
    
    TEMP=$(getopt -n $0 -s bash -a \
     --longoptions="help top: version: qualifier: setup: \
     npot: repeat: executable: geometry: script: banalyser: \
     rewrite" \
     -o hT:v:q:n:r:x:g:s:b: -- "$@") || exit 1

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
	    -v | --version    ) export G4VERSION="$2"       ; shift ;;
	    -q | --qualifier  ) export G4QUALIFIER="$2"     ; shift ;;
	    --setup           ) export INITSETUPSTR="$2"    ; shift ;;
	    -n | --npot       ) export NPOT="$2"            ; shift ;;
	    -r | --repeat     ) export NREPEAT="$2"         ; shift ;;
	    -x | --executable ) export EXECUTABLE="$2"      ; shift ;;
	    -g | --geometry   ) export GEOMETRY="$2"        ; shift ;;
	    -s | --script     ) export PRODSCRIPT="$2"      ; shift ;;
	    -b | --banalyser  ) export BANALYSER="$2"       ; shift ;;
	    --rewrite         ) export DOREWRITE=1                  ;;
	    -*                ) echo "unknown flag $opt ($1)" ; PRINTUSAGE=1 ;;
	esac
	shift # eat up the arg we just used
    done
    set +u

    if [[ ${PRINTUSAGE} -eq 1 ]] ; then
	usage
    fi

    echo -e "${OUTYELLOW}Using tune: version, qualifier = ${G4VERSION}, ${G4QUALIFIER}${OUTNOCOL}"
    echo -e "${OUTYELLOW}Using geometry: $(basename ${GEOMETRY}) with template $(basename ${PRODSCRIPT})${OUTNOCOL}"
    echo -e "${OUTCYAN}I will simulate ${NPOT} POT${OUTNOCOL}"
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
    mkdir -p ${OUTPUTTOP}/work-products/dk2nu-outputs
elif [ -d ${OUTPUTTOP} ] && [ ${DOREWRITE} -eq 0 ] ; then
    echo -e "${OUTRED}Directory ${OUTPUTTOP} exists, not overwriting. Pass --rewrite if you want this to be rewritten.${OUTNOCOL}"
    exit 1
fi

if [[ -z ${EXECUTABLE} ]] ; then
    echo -e "${OUTRED}No executable found. Did you pass -x | --executable?${OUTNOCOL}"
    exit 1
fi
if [[ -z ${GEOMETRY} ]] ; then
    echo -e "${OUTRED}No geometry found. Did you pass -g | --geometry?${OUTNOCOL}"
    exit 1
fi
if [[ -z ${PRODSCRIPT} ]] ; then
    echo -e "${OUTRED}No input template script found. Did you pass -s | --script?${OUTNOCOL}"
    exit 1
fi
if [[ -z ${BANALYSER} ]] ; then
    echo -e "${OUTRED}No python analyser for beamHist found. Did you pass -b | --banalyser?${OUTNOCOL}"
    exit 1
fi

# Tar up stuff
echo -e "${OUTLTPURPLE}Making tarball.."
FLIST=""
cp ${EXECUTABLE} ${OUTPUTTOP}/tarball && FLIST=$(basename ${EXECUTABLE})" "${FLIST}
cp ${GEOMETRY} ${OUTPUTTOP}/tarball && FLIST=$(basename ${GEOMETRY})" "${FLIST}
cp ${PRODSCRIPT} ${OUTPUTTOP}/tarball && FLIST=$(basename ${PRODSCRIPT})" "${FLIST}
cp ${BANALYSER} ${OUTPUTTOP}/tarball && FLIST=$(basename ${BANALYSER})" "${FLIST}
tar -C ${OUTPUTTOP}/tarball -cvjSf ${OUTPUTTOP}/tarball/nominal.tar ${FLIST}
echo -e "Done making tarball.${OUTNOCOL}"

# which CVMFS?
CVMFS_SETUP=
case ${INITSETUPSTR} in
    "uboone") CVMFS_SETUP=/cvmfs/uboone.opensciencegrid.org/products/setup_uboone_mcc9.sh   ;;
    "sbnd")   CVMFS_SETUP=/cvmfs/sbnd.opensciencegrid.org/products/sbnd/setup_sbnd.sh       ;;
    "icarus") CVMFS_SETUP=/cvmfs/icarus.opensciencegrid.org/products/icarus/setup_icarus.sh ;;
    "dune")   CVMFS_SETUP=/cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh       ;;
    *) echo "Unknown setup script, exiting" ; exit 1 ;;
esac

# write the binary file
export EXECFILE=${OUTPUTTOP}/bin/generate_beammc.sh

cat > ${EXECFILE} <<EOF
define_cfg()
{
  # This configuration is for:
  # G4VERSION   = "${G4VERSION}"
  # G4QUALIFIER = "${G4QUALIFIER}"
  # setup from ${CVMFS_SETUP}

  source ${CVMFS_SETUP}
  setup geant4 ${G4VERSION} -q ${G4QUALIFIER}
  setup ifdhc v2_8_1 -q e26:p3915:prof
  setup ifdhc_config v2_8_1
  setup dk2nudata v01_10_01f -q e26:prof
  ups active

  export SUBPROCESS=\$1

  export JOBEXECUTABLE=\$(pwd)/$(basename ${EXECUTABLE})
  export JOBGEOMETRY=\$(pwd)/$(basename ${GEOMETRY})
  export JOBPRODTEMPLATE=\$(pwd)/$(basename ${PRODSCRIPT})
  export JOBPRODSCRIPT=\$(pwd)/production-\${SUBPROCESS}.in
  export JOBBANALYSER=\$(pwd)/$(basename ${BANALYSER})
  export JOBNPOT=\$(echo "${NPOT} / ${NREPEAT}" | bc)
}

setup_python()
{
  # Setup a virtual environment
  python3 -m venv .
  source bin/activate
  # Pull in pip, numpy, pandas, fastparquet, uproot
  # python3 -m pip install --upgrade pip
  python3 -m pip install numpy > /dev/null
  python3 -m pip install pandas==2.2.3 > /dev/null # newer than that and meson seems to complain
  python3 -m pip install uproot "pandas<2.3" > /dev/null
  python3 -m pip install fastparquet "pandas<2.3" > /dev/null
  python3 -m pip install tqdm colorama > /dev/null

  echo "Python setup OK"
}

export BASEDIR=\$(pwd)
export WORKDIR=\$(mktemp -d -p \${BASEDIR})
mkdir -p \$WORKDIR && cd \$WORKDIR

THIS_SUBPROC="0"
if [[ \$# > 0 ]] ; then THIS_SUBPROC=\$1 ; fi
define_cfg \${THIS_SUBPROC}

# copy in the tarball
tSleep=\$(echo "\${RANDOM} % 20" | bc)
echo "Sleeping for \${tSleep} s..."
sleep \${tSleep}
ifdh cp -D ${OUTPUTTOP}/tarball/nominal.tar ./
tar -xvjSf nominal.tar

ls

setup_python
# run a sed command on the template
mkdir dk2nu-outputs
sed -e "s|REPLACE_OUTPUT_FILENAME|dk2nu-outputs/\$(printf "%04d" \${SUBPROCESS}).dk2nu.root|g" -e "s|REPLACE_SEED|\$(echo "10 * \${RANDOM} + \${RANDOM}" | bc)|g" -e "s|REPLACE_GEOMETRY|\${JOBGEOMETRY}|g" -e "s|REPLACE_NPOT|\${JOBNPOT}|g" \${JOBPRODTEMPLATE} > \${JOBPRODSCRIPT}

#cat \${JOBPRODSCRIPT}

# try to run beamHist
cmd="\${JOBEXECUTABLE} \${JOBPRODSCRIPT} > /dev/null"
echo \$cmd
eval \$cmd

# Analyse the histograms
python3 \${JOBBANALYSER} -i dk2nu-outputs/ -o hists.\$(printf "%04d" \${SUBPROCESS}).root --detector 73.78 0.0 11000.0 -s 200.0 -t 1
#python3 run_beamHist.py -h

# copy back to job directory
ifdh cp -D hists.\$(printf "%04d" \${SUBPROCESS}).root ${OUTPUTTOP}/work-products/
ifdh cp -D dk2nu-outputs/\$(printf "%04d" \${SUBPROCESS}).dk2nu.root ${OUTPUTTOP}/work-products/dk2nu-outputs/

# cleanup
cd \${BASEDIR} && rm -rf \${WORKDIR}/
EOF
chmod u+x ${EXECFILE}

# Now write a DAG file.
JOBSUBMAIN="jobsub_submit -n -G $(id -ng)  --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC  --append_condor_requirements='(TARGET.HAS_CVMFS_sbn_opensciencegrid_org==true)' --singularity-image=/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-sl7:latest"
RESOURCES="--expected-lifetime 8h --disk 8GB --memory 2GB"
JOBSUBFULL=${JOBSUBMAIN}" "${RESOURCES}
echo -e "<parallel>" > g4bnb-nominal.dag

# I will submit this many processes.
export LASTPROCESS=$(echo "${NREPEAT} - 1" | bc)
for i in $(seq 0 ${LASTPROCESS}) ; do
    echo -e "${JOBSUBFULL} file://${EXECFILE} ${i}" >> g4bnb-nominal.dag
done
echo -e "</parallel>" >> g4bnb-nominal.dag

mv g4bnb-nominal.dag ${OUTPUTTOP}/cfg/

echo -e "${OUTCYAN}Exit status 0${OUTNOCOL}"
