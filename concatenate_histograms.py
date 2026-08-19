import uproot
from uproot.writing.identify import to_TH1x, to_TAxis
import numpy as np
import argparse
from argparse import RawTextHelpFormatter # for newlines in help, see SO 3853722
from pathlib import Path

if __name__ == "__main__":
    print(Fore.YELLOW + "Hello world!" + Fore.RESET)

    parser = argparse.ArgumentParser(description=Fore.CYAN+"A script to add together calculated histograms."+Fore.RESET, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-i', '--inpath', type=str, required=True, help=Fore.RED+"I will read all the files in this directory."+Fore.RESET)
    parser.add_argument('-o', "--output", type=str, default="./out.root", help=Fore.YELLOW+"Output file"+Fore.RESET)
    parser.add_argument('-t', '--threads', type=int, default=8, help=Fore.YELLOW+f"Number of concurrent threads. Maximum {NPROCESS_MAX}"+Fore.RESET)
    parser.add_argument("--detector", type=float, nargs=3, required=True, help=Fore.RED+"Detector origin in beamline coordinates (cm)"+Fore.RESET)
    parser.add_argument('-s', "--size", type=float, required=True, help=Fore.RED+"Half side of the square detector face (cm)"+Fore.RESET)
    parser.add_argument('--name', type=str, default="UNKNOWN", help=Fore.YELLOW+"Name of the varied parameter."+Fore.RESET)
    parser.add_argument('--sigma', type=str, default="UNKNOWN", help=Fore.YELLOW+"Variation of the parameter in sigma" + Fore.RESET)
    
    args = parser.parse_args()

    main(args)
    
    print(Fore.YELLOW + "Goodbye world!" + Fore.RESET)
