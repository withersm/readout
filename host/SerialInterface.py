import serial
import struct
import crcxmodem as crc
from time import sleep

SERIAL_READ_TIMEOUT = 0.250  # Seconds
SIZET_MCU_PACKET_STRUCT = 20


class BadPacketException(Exception):
    def __init__(self):
        super().__init__("Received Malformed or No Packet from MCU")


class InvalidChecksumException(Exception):
    def __init__(self):
        super().__init__("Checksum mismatch when communicating with MC")


class InvalidCommandException(Exception):
    def __init__(self):
        super().__init__("Issued invalid command to MC")


class SerialController:
    def __init__(self, sercom) -> None:
        ser = self.ser = serial.Serial(dsrdtr=False)
        ser.port = sercom
        ser.baudrate = 115200
        ser.timeout = SERIAL_READ_TIMEOUT
        self.pinstate = 0

    def open(self):
        if self.ser.is_open:
            return True
        try:
            self.ser.open()
            sleep(0.1)
        except serial.serialutil.SerialException:
            print(
                "\n******************************************\nCOULDN'T FIND COM PORT\n******************************************\n"
            )
            exit()
        except Exception as e:
            raise e

    def close(self):
        if not self.ser.is_open:
            return True
        else:
            try:
                self.ser.close()
            except Exception as e:
                raise e
            return True

    def transact(self, cmd, *args):
        """Processes serial transaction

        :param cmd: Command
        :type cmd: int
        :raises TypeError: args must be an int or float since we're utilizing ctypes
        """
        # Build Payload
        payload = struct.pack("<I", cmd)
        assert (
            len(args) == 3
        ), "Microcontroller expects to fill a struct from the data here"

        for a in args:
            if isinstance(a, int):
                payload = payload + struct.pack("<I", a)
            elif isinstance(a, float):
                payload = payload + struct.pack("<f", a)
            else:
                raise TypeError("Unsupported type, must use int or float")
        payload = payload + crc.calc_crc(payload)

        # Print and Read
        self.ser.write(payload)
        packet = self.ser.read(SIZET_MCU_PACKET_STRUCT)

        if len(packet) != SIZET_MCU_PACKET_STRUCT:
            print(len(packet))
            raise BadPacketException

        # Check checksum
        pktchecksum = packet[-4:]
        newchecksum = crc.calc_crc(packet[0:-4])
        if pktchecksum != newchecksum:
            raise InvalidChecksumException

        # Check for errors
        cmd, _, _, _, _ = struct.unpack("<IIIII", packet)
        if cmd == 0xFFFFFFFF:
            raise InvalidCommandException
        elif cmd == 0xFFFFFFFE:
            raise InvalidChecksumException
        return packet

    def test_connection(self):
        packet = self.transact(1, 0, 0, 0)
        cmd, isConnected, arg2, arg3, _ = struct.unpack("<IIIII", packet)
        if cmd == 1 and isConnected == 1:
            return True
        else:
            return False

    def get_wiper(self, chan, wipernum):
        packet = self.transact(2, chan, wipernum, 0)
        cmd, potvalue, i2cstatus, arg3, _ = struct.unpack("<IIIII", packet)
        return potvalue

    def set_wiper(self, chan, wipernum, value):
        packet = self.transact(3, chan, wipernum, value & 0xFF)
        cmd, i2cstatus, arg2, arg3, _ = struct.unpack("<IIIII", packet)
        return i2cstatus == 0

    def get_gpio(self):
        packet = self.transact(4, 0, 0, 0)
        cmd, gpiostate, arg2, arg3, _ = struct.unpack("<IIIII", packet)
        return gpiostate & 0xFFFF

    def set_allgpio(self, pinstates):
        packet = self.transact(5, pinstates, 0, 0)
        cmd, i2cstatus, arg2, arg3, _ = struct.unpack("<IIIII", packet)
        return i2cstatus == 0

    def set_gpio(self, pin, state):
        p = self.pinstate = self.get_gpio()
        if state:
            p = p | (1 << pin)
            self.pinstate = p
        else:
            p = p & (~(1 << pin))
            self.pinstate = p
        packet = self.transact(5, p, 0, 0)
        cmd, i2cstatus, arg2, arg3, _ = struct.unpack("<IIIII", packet)
        return i2cstatus == 0

    def get_bsi(self, chan):
        packet = self.transact(7, chan, 0, 0)
        cmd, busV, shuntV, current, _ = struct.unpack("<IfffI", packet)
        return round(busV, 6), round(shuntV, 6), round(current, 6)

    def testfloat(self):
        packet = self.transact(8, 0, 0, 0)
        cmd, busV, shuntV, current, _ = struct.unpack("<IfffI", packet)
        return round(busV, 6), round(shuntV, 6), round(current, 6)
