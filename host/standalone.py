import argparse
import sys
import serial.tools.list_ports as stl
import alicpt


def listcom():
    ports = stl.comports()
    if not ports:
        print("No COM ports available")
    for p in ports:
        print(p)


def setvtes(args):
    bias = alicpt.Bias(args.port)
    bias.vTES(args.channel, args.voltage)
    bias.end()


def setishunt(args):
    bias = alicpt.Bias(args.port)
    bias.iSHUNT(args.channel, args.current)
    bias.end()


def setvlnad(args):
    bias = alicpt.Bias(args.port)
    bias.vLNA_D(args.channel, args.current)
    bias.end()


def setilna(args):
    bias = alicpt.Bias(args.port)
    bias.iLNA(args.lnabias, args.current)
    bias.end()


def setvlnag(args):
    bias = alicpt.Bias(args.port)
    bias.vLNA_G(args.lnabias, args.voltage)
    bias.end()


def getalliv(args):
    bias = alicpt.Bias(args.port)
    bias.getAllIV()
    bias.end()


def standalone():
    parser = argparse.ArgumentParser(
        prog="FotonX Bias Control:",
        description="Useful for controlling bias currents and voltages from the command line",
    )
    parser.add_argument(
        "-l",
        "--list-ports",
        action="store_true",
        help="Use this command to list available com ports",
    )

    subparsers = parser.add_subparsers(
        metavar="", title="Control some aspect of the Bias"
    )

    vtes = subparsers.add_parser(
        "vtes",
        help="//Adjusts POT wiper and then reads INA values until I x 400uOhm = V(uV) on Ch (Vreg 1-4)",
    )
    vtes.add_argument("port", help="Provide the COM port for controlling the bias.")
    vtes.add_argument("channel", type=int, help="Channel 1-6")
    vtes.add_argument("voltage", type=float, help="Voltage (mV) to set")
    vtes.set_defaults(func=setvtes)

    ishunt = subparsers.add_parser(
        "ishunt",
        help="//Adjusts POT wiper and then reads INA values until I = I(mA) on Ch (Vreg 1-4)",
    )
    ishunt.add_argument("port", help="Provide the COM port for controlling the bias.")
    ishunt.add_argument("channel", type=int, help="Channel 1-6")
    ishunt.add_argument("current", type=float, help="Current (mA) to set")
    ishunt.set_defaults(func=setishunt)
    vlnad = subparsers.add_parser(
        "vlnad",
        help="//Adjusts POT wiper and then reads INA values until V(bus) - V(shunt) = V(V) on Ch (Vreg 5 and 7)",
    )
    vlnad.add_argument("port", help="Provide the COM port for controlling the bias.")
    vlnad.add_argument("channel", type=int, help="Channel 1-6")
    vlnad.add_argument("current", type=float, help="Current (mA) to set")
    vlnad.set_defaults(func=setvlnad)

    ilna = subparsers.add_parser(
        "ilna",
        help="//Adjusts POT wiper and then reads INA values until I = I(mA) for Ch",
    )
    ilna.add_argument("port", help="Provide the COM port for controlling the bias.")
    ilna.add_argument("lnabias", help="LNA Bias (1 or 2)", type=int)
    ilna.add_argument("current", type=float, help="Current (mA) to set")
    ilna.set_defaults(func=setilna)

    vlnag = subparsers.add_parser(
        "vlnag",
        help="//Adjusts POT wiper and then reads INA values until V(bus) = V(V) on Ch (Vreg 6 and 8)",
    )
    vlnag.add_argument("port", help="Provide the COM port for controlling the bias.")
    vlnag.add_argument("lnabias", help="LNA Bias (1 or 2)", type=int)
    vlnag.add_argument("voltage", type=float, help="voltage (mV) to set")
    vlnag.set_defaults(func=setvlnag)

    getalliv = subparsers.add_parser(
        "getalliv",
        help="//Reads INA values for all Ch. Labels should be displayed as TES1, TES2,... LNA1 D, LNA2 D...",
    )
    getalliv.add_argument("port", help="Provide the COM port for controlling the bias.")
    getalliv.set_defaults(func=getalliv)

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.list_ports:
        listcom()
    else:
        args.func(args)


if __name__ == "__main__":
    standalone()
