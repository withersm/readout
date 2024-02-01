from time import sleep, strftime, localtime
import SerialInterface
import numpy as np
import serial.tools.list_ports as stl


VTES_TOLERANCE = 0.05
#ITES_TOLERANCE = 0.01
ITES_TOLERANCE = 0.04
VLNA_D_TOLERANCE = 0.01
VLNA_G_TOLERANCE = 0.01
ILNA_TOLERANCE = 0.01

INA219CHANMAP = [0, 3, 4, 5, 6, 1, 2]
NAMES = [
    "INA TES1",
    "INA TES2",
    "INA TES3",
    "INA TES4",
    "LNA D Bias 1",
    "LNA D Bias 2",
    "VG LNA 1",
    "VG LMA 2",
]


def listcom():
    """Lists the serial devices connected to the host computer."""
    ports = stl.comports()
    if not ports:
        print("No COM ports available")
    for p in ports:
        print(p)


class Bias:
    """Provides a comprehensive interface for controlling
    the AliCPT TES Bias System.

    :param SerialPort: The name of the attached device
    :type SerialPort: str

    .. code-block::
        :caption: Example Code

            alicpt.listcom()
            alicpt.Bias("COM8")

    .. Warning::

        Connecting to the COM port forces the microcontroller to revert all settings back to default.
        i.e. All channels revert to 0 V

    """

    def __init__(self, SerialPort) -> None:
        self.si = SerialInterface.SerialController(SerialPort)
        self.lna_wiper_g = [0, 0]
        self.port = SerialPort
        self.si.open()

    def get_vTES(self, ch):
        """Reads and calculates the TES Voltage in microVolts

        :param ch: Channel
        :type ch: int
        :return: TES Voltage (in microvolts)
        :rtype: float
        """
        vals = np.zeros(10)
        for i, v in enumerate(vals):
            b, s, curr = self.si.get_bsi(INA219CHANMAP[ch] - 1)
            vals[i] = curr
        VuV = (np.average(vals) * 1e-3) * 0.4e-3 * 1e6 * 1e-2
        return VuV

    def get_iTES(self, ch):
        """Reads and calculates the TES current

        :param ch: Channel to read.
        :type ch: int
        :return: Current in mA
        :rtype: float
        """
        vals = np.zeros(10)
        for i, v in enumerate(vals):
            b, s, curr = self.si.get_bsi(INA219CHANMAP[ch] - 1)
            vals[i] = curr
        return np.average(vals) * 1e-2

    def vTES(self, ch, uv):
        """Adjusts V(out) for specified TES channel (1 - 4) until I(out) provides necessary current to reach TES voltage
        specified by user.  I x 400uOhm = V(uV). Range = 0 – 5V

        :param ch: TES Bias Channel (1-4)
        :type ch: int
        :param uv: Desired voltage in microvolts (0.0 to 5.0)
        :type uv: float
        """
        potvalue = self.si.get_wiper(1, ch - 1)

        MAXITER = 275
        i = 0
        while 1:
            if i > 512:
                print(f"|ERROR| loop exceeded {MAXITER}")
                break

            v_measured = self.get_vTES(ch)
            diff = abs(v_measured - uv) / uv
            print(f"vmeasured= {v_measured}; diff= {diff}")
            if diff < VTES_TOLERANCE:
                break

            if v_measured < uv:
                if potvalue == 255:
                    print("|WARN| Pot is at max setting!")
                    break
                potvalue += 1
                self.si.set_wiper(1, ch - 1, potvalue)

            elif v_measured > uv:
                if potvalue == 0:
                    print("|WARN| Pot is at min setting")
                    break
                potvalue -= 1
                self.si.set_wiper(1, ch - 1, potvalue)
            i += 1

    def set_wiper(self, chan, value):
        """
        Adjust the voltage by changing a potentiometer.
        """
        self.si.set_wiper(1, chan-1, value)

    def get_wiper(self, chan):
        self.si.get_wiper(1, chan-1)


    def iSHUNT(self, ch, imA):
        """Adjusts V(out) for specified TES channel (1-4) until I(out) = current specified by user.
        Range = Depends on total Thévenin resistance of TES bias chain.
        Resistance of cryo wire + TES shunt resistance will limit the maximum current from the bias system.
        Maximum short circuit current of the TES bias system is 25mA. It is assumed that the Thévenin
        resistance of the cryo wire + TES shunts ≈ 200 Ohms which would mean the max current is 12.5 mA.

        Note to folks who use NIST terminology: This is how you set IBias. We should clean up the names.


        :param ch: TES Bias Channel
        :type ch: int
        :param imA: Desired current in mA
        :type imA: float
        """
        potvalue = self.si.get_wiper(1, ch - 1)

        MAXITER = 275
        i = 0
        while 1:
            if i > 512:
                print(f"|ERROR| loop exceeded {MAXITER}")
                break

            i_measured = self.get_iTES(ch)
            #diff = abs(i_measured - imA) / imA
            diff = abs(i_measured - imA)
            print(f"i-measured= {i_measured}; diff= {diff}")
            if diff < ITES_TOLERANCE:
                break

            if i_measured < imA:
                if potvalue == 255:
                    print("|WARN| Pot is at max setting!")
                    break
                potvalue += 1
                self.si.set_wiper(1, ch - 1, potvalue)

            elif i_measured > imA:
                if potvalue == 0:
                    print("|WARN| Pot is at min setting")
                    break
                potvalue -= 1
                self.si.set_wiper(1, ch - 1, potvalue)
            i += 1
        pass

    def vLNA_D(self, LNACh, V):
        """Adjusts V(out) for specified LNA channel (1 or 2) until bias system output voltage reaches user
        specified voltage. TES bias system output voltage ≠ voltage at LNA input.
        Voltage drop between TES bias system output and LNA Vin pin occurs due to cryo wire resistance.
        LNAs should be voltage biased to the operating current specified in their data sheets
        according to their temperature. Using the iLNA_D command is recommended. “D” indicates LNA drain.


        :param LNACh: LNA Bias Channel (1 or 2)
        :type LNACh: int
        :param V: Desired output Voltage
        :type V: float
        """
        ch = wiper = 0
        if LNACh == 1:
            wiper = 1 - 1
        else:
            wiper = 3 - 1

        potvalue = self.si.get_wiper(2, wiper)

        MAXITER = 275
        i = 0
        while 1:
            if i > 512:
                print(f"|ERROR| loop exceeded {MAXITER}")
                break
            vals = np.zeros(10)
            for i, v in enumerate(vals):
                bus_V, shunt_mV, curr_mA = self.si.get_bsi(LNACh - 1)
                vals[i] = bus_V
            bus_V = np.average(vals)

            Vmeasured = bus_V + 0.07

            diff = abs(Vmeasured - V) / V
            print(f"vLNA_D V measured= {Vmeasured}; diff= {diff}")
            if diff < VLNA_D_TOLERANCE:
                break

            if Vmeasured < V:
                if potvalue == 255:
                    print("|WARN| Pot is at max setting!")
                    break
                potvalue += 1
                self.si.set_wiper(2, wiper, potvalue)

            elif Vmeasured > V:
                if potvalue == 0:
                    print("|WARN| Pot is at min setting")
                    break
                potvalue -= 1
                self.si.set_wiper(2, wiper, potvalue)
            i += 1

    def iLNA_D(self, LNACh, setCurrent):
        """Adjusts I(out) for specified channel (1 or 2) until I(out) = current specified by user on LNA Ch.
            “D” indicates LNA drain.
            The ASU 4k LNA typically draws 4.5mA @ Temp = 10K.


        :param LNACh: LNA Bias Channel (1 or 2)
        :type LNACh: int
        :param setCurrent: Desired Current in mA
        :type setCurrent: float
        """
        ch = wiper = 0
        if LNACh == 1:
            wiper = 1 - 1
        else:
            wiper = 3 - 1

        potvalue = self.si.get_wiper(2, wiper)

        MAXITER = 275
        i = 0
        while 1:
            if i > 512:
                print(f"|ERROR| loop exceeded {MAXITER}")
                break
            vals = np.zeros(10)
            for i, v in enumerate(vals):
                bus_V, shunt_mV, curr_mA = self.si.get_bsi(LNACh - 1)
                vals[i] = curr_mA
            curr_mA = np.average(vals) / 100

            diff = abs(curr_mA - setCurrent) / setCurrent
            print(f"iLNA_D current measured= {curr_mA}; diff= {diff}")
            if diff < ILNA_TOLERANCE:
                break

            if curr_mA < setCurrent:
                if potvalue == 255:
                    print("|WARN| Pot is at max setting!")
                    break
                potvalue += 1
                self.si.set_wiper(2, wiper, potvalue)

            elif curr_mA > setCurrent:
                if potvalue == 0:
                    print("|WARN| Pot is at min setting")
                    break
                potvalue -= 1
                self.si.set_wiper(2, wiper, potvalue)
            i += 1

    def vLNA_G(self, lna, voltage):
        """Adjusts V(out) for LNA Gate.
        The ASU LNA Gate is assumed to draw 0 A of current.
        The ASU 4k LNA typically uses a Gate voltage of 0V.

        :param lna: LNA Gate (1 or 2)
        :type lna: int
        :param voltage: Desired Voltage out
        :type voltage: float
        """
        wiper = int(round(voltage / 0.000784))
        if wiper > 255:
            wiper = 255
        if wiper < 0:
            wiper = 0
        self.si.set_wiper(2, (lna * 2) - 1, wiper)

        self.lna_wiper_g[lna - 1] = wiper

    def getAllIV(self):
        """Reads voltage and current values for all TES and LNA channels in the TES bias system."""
        busVs = np.zeros(6)
        shuntMvs = np.zeros(6)
        currents = np.zeros(6)
        print(
            "**************************************** INA READINGS ****************************************"
        )
        print(
            "Channel \t\t INA219 \t Bus Voltage (V) \t Shunt Voltage (mV) \t\t Current (mA)"
        )
        for i in range(6):
            b, s, curr = self.si.get_bsi(INA219CHANMAP[i + 1] - 1)
            busVs[i] = b
            shuntMvs[i] = s
            currents[i] = curr / 100
            print(
                f"{NAMES[i]}    \t\t       {INA219CHANMAP[i+1]}       \t       {b}        \t      {s}      \t        {round(curr/100,3)}              "
            )
        print(f"\nVG LNA1 = {self.lna_wiper_g[0]*.000784}")
        print(f"\nVG LNA2 = {self.lna_wiper_g[1]*.000784}")
        print("\n\n")

    def end(self):
        """Close the comport after use has concluded."""
        self.si.close()
