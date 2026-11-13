import argparse
from datetime import datetime, timezone
from daneel.detection import Transit
from daneel.detection import TransitModel
from daneel.parameters import Parameters
from pathlib import Path
from daneel.detection import RandomForestClassifier


def main():
    parser = argparse.ArgumentParser(description="Model and analyse exoplanet transits.")

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        nargs = "+",
        type=str,
        required=True,
        help="Input par files to pass",
    )

    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("-t", "--transit", action="store_true", help="Plot a transit light curve")
    actions.add_argument("-d", "--detection", type = str, choices = ["rf","cnn"] , help="Run exoplanet detection")
    actions.add_argument("-a", "--atmosphere", action="store_true", help="Run atmospheric characterisation")
    parser.add_argument("-o", "--output",nargs = "+", default="lc.png", help="Output image for --transit")

    args = parser.parse_args()

    """Launch Daneel"""
    start = datetime.now(timezone.utc)

    #loading our parameters and the name of the file
    input_pars = []
    name_files = []
    s = len(args.input_file) #length of our initial parameters
    for i in range(s):
        # Load parameters using your specified syntax for each file
        input_pars.append(Parameters(args.input_file[i]).params)
        name_files.append(Path(args.input_file[i]).name)

    if args.transit:

        #if the user gives us n input we have two options:
        #n input = n output ---> we produce n images corresponding to all the inputs the user gave us 
        #n input, 1 output --> we create an image with all the light curve in the same plot 
        #n input != n output ---> ERROR!! otherwise we would have a loss of information for the light curves
        if((len(args.output)!= len(args.input_file)) and len(args.output) !=1):
            print("The number of input files is different compared to the number of output files")
        elif(len(args.output)==1):
           #before we compute all the fluxes and then we plot them together 
           flux = []
           transit1 = Transit(input_pars[0]['transit'])
           flux.append(transit1.compute_light_curve())
           for j in range (1,s):
                transit = Transit(input_pars[j]['transit'])
                flux.append(transit.compute_light_curve())
           transit1.plot_one_figure(flux,name_files,args.output[0])
        else:
            for j in range (s):
                transit = Transit(input_pars[j]['transit'])
                transit.plot_light_curve(name_files,args.output[j])

    elif args.detection:
        for i in range(s):
            if args.detection == "rf":
                rf = RandomForestClassifier(input_pars[i])
                rf.run()
            else: 
                parser.error("detection CNN is not implemented yet; use --transit")
    elif args.atmosphere:
        parser.error("atmospheric characterisation is not implemented yet; use --transit")

    finish = datetime.now(timezone.utc)
    print(f"Daneel completed in {(finish - start).total_seconds():.2f}s")


if __name__ == "__main__":
    main()
