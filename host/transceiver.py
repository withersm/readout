"""
__author__ = "Cody Roberson, Paul Horton"
__copyright__ = "Copyright 2021, ASU"
__license__ = "GPL 3"
__version__ = "0.3"
__maintainer__ = "Cody"
__email__ = "carobers@asu.edu, pahorton@asu.edu"
__status__ = "Prototype"
description: Implements a serial interface for the ASU Transceiver project

 *
 *  Copyright (C) 2021  Arizona State University
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
"""

from serial import Serial
import time


class Transceiver(object):
    def __init__(self):
        super(Transceiver, self).__init__()
        self.connection = False
        self.loopback = False
        self.synth_ref = 0
        self.synth_out = 0
        self.pfd = 0.0
        self.rf_in = -100
        self.rf_out = -100

    def connect(self, device: str) -> None:
        self.channel = Serial()
        self.channel.baudrate = 115200
        self.channel.port = device
        self.channel.timeout = 0.5
        try:
            self.channel.open()
        except:
            print("There was an error while attempting to open a connection")

        if self.channel.is_open:
            self.channel.write(b'gui\n')
            resp = self.channel.readline().strip()
            if resp == b'OK':
                self.connection = True
            else:
                self.channel.write(b'ping\n')
                resp = self.channel.readline().strip()
                if resp == b'OK':
                    self.connection = True

                else:
                    raise ConnectionError("Unable to connect to Transceiver")
                    self.connection = False
        else:
            raise ConnectionError("Unable to connect to Transceiver")
            self.connection = False

    def check_connection(self):
        if self.channel.is_open:
            self.channel.write(b'ping\n')
            resp = self.channel.readline().strip()
            if resp == b'OK':
                self.connection = True
            else:
                self.connection = False
        else:
            self.connection = False
        return self.connection

    def get_loopback(self):
        if self.check_connection():
            self.channel.write(b'get:loopback\n')
            resp = self.channel.readline().strip()
            if resp == b'Enabled':
                self.loopback = True
            else:
                self.loopback = False
            return self.loopback
        else:
            raise ConnectionError("Unable to connect to Transceiver")
    """
    def toggle_loopback(self):
        if self.check_connection():
            # Set loopback
            self.channel.write(b'tlb\n');
            time.sleep(.05)
            resp = self.channel.readline().strip()
            if resp != b'OK':
                raise ConnectionError("Something went wrong, failed to set loopback.")
            return self.get_loopback()
        else:
            raise ConnectionError("Unable to connect to Transceiver")
    """ 
    def set_loopback(self, val: bool):
        if self.check_connection():
            # Set loopback
            self.loopback = val
            if val:
            	self.channel.write(b'set:loopback\nEnable\n')
            else:
            	self.channel.write(b'set:loopback\nDisable\n')
            time.sleep(.05)
            resp = self.channel.readline().strip()
            if resp != b'OK':
                raise ConnectionError("Something went wrong, failed to set loopback.")
            return self.get_loopback()
        else:
            raise ConnectionError("Unable to connect to Transceiver")

    def get_synth_ref(self) -> float:
        if self.check_connection():
            self.channel.write(b'get:synthref\n')
            resp = self.channel.readline().strip()
            try:
                self.synth_ref = float(resp)
            except:
                print("Conversion error")

            return self.synth_ref
        else:
            raise ConnectionError("Unable to connect to Transceiver")

    def set_synth_ref(self, val):
        if self.check_connection():
            # Set synth_ref
            temp = self.synth_ref
            self.synth_ref = val
            cmd = f"set:synthref\n{val}\n".encode('ASCII')
            self.channel.write(cmd)
            time.sleep(.5)
            resp = self.channel.readline().strip()
            if (resp != b'OK'):
                if len(resp) == 0:
                    self.check_connection()
                else:
                    print(resp.decode('ascii'))
                    self.synth_ref = temp
            time.sleep(0.5)
            return self.get_synth_ref()
        else:
            raise ConnectionError("Unable to connect to Transceiver")

    def get_synth_out(self):
        if self.check_connection():
            # Check and set synth_out
            self.channel.write(b'get:synthout\n')
            resp = self.channel.readline().strip()
            try:
                self.synth_out = float(resp)
            except:
                print("Conversion error")
            return self.synth_out
        else:
            raise ConnectionError("Unable to connect to Transceiver")

    def set_synth_out(self, val):
        if self.check_connection():
            # Set synth_out
            temp = self.synth_out
            self.synth_out = val
            cmd = f"set:synthout\n{val}\n".encode('ASCII')
            self.channel.write(cmd)
            time.sleep(.05)
            resp = self.channel.readline().strip()
            if (resp != b'OK'):
                if len(resp) == 0:
                    self.check_connection()
                else:
                    print(resp.decode('ascii'))
                    self.synth_out = temp

            return self.get_synth_out()
        else:
            raise ConnectionError("Unable to connect to Transceiver")

    def get_rf_in(self):
        if self.check_connection():
            # Check and set rf_in
            self.channel.write(b'get:attenuation\n')
            resp = self.channel.readline().strip().decode('ascii')
            val = resp.split(',')
            try:
                self.rf_in = float(val[0])
            except:
                print("Conversion Error")
            return self.rf_in
        else:
            raise ConnectionError("Unable to connect to Transceiver")

    def set_rf_in(self, val):
        if self.check_connection():
            # Set rf_in
            cmd = f"set:attenuation\n1\n{val}\n".encode("ASCII")
            self.channel.write(cmd)
            resp = self.channel.readline().strip()
            if resp == b"OK":
                self.rf_in = val
            else:
                print(resp)
            return self.get_rf_in()
        else:
            raise ConnectionError("Unable to connect to Transceiver")

    def get_rf_out(self):
        if self.check_connection():
            self.channel.write(b'get:attenuation\n')
            resp = self.channel.readline().strip().decode('ascii')
            val = resp.split(',')
            try:
                self.rf_out = float(val[1])
            except:
                print("Conversion Error")

            return self.rf_out
        else:
            raise ConnectionError("Unable to connect to Transceiver")

    def get_fPFD(self):
        if self.check_connection():
            self.channel.write(b'get:synth pfd\n')
            time.sleep(.05)
            resp = self.channel.readline().strip()
            try:
                self.pfd = float(resp)
            except:
                print("Conversion Error")

            return self.pfd
        else:
            raise ConnectionError("Unable to connect to Transceiver")

    def set_rf_out(self, val):
        if self.check_connection():
            # Set rf_out
            cmd = f"set:attenuation\n2\n{val}\n".encode("ASCII")
            self.channel.write(cmd)
            resp = self.channel.readline().strip()
            if resp == b"OK":
                self.rf_out = val
            else:
                print(resp.decode(('ascii')))
            return self.get_rf_out()
        else:
            raise ConnectionError("Unable to connect to Transceiver")


if __name__ == "__main__":
    print("*** TRANSCEIVER SOFTWARE CONTROL TEST ***\n\r")
    dev = Transceiver()

    if dev.connection == False:
        print("check connection when not connected: PASS")

    dev.connect('/dev/ttyACM1')
    if dev.connection == True:
        print("create connection: PASS")

    dev.check_connection()
    if dev.connection == True:
        print("check connection: PASS")

    if dev.get_loopback() == False:
        print("get loopback: PASS")
    else:
        print("get loopback: FAIL")

    if dev.get_synth_ref() == 10.0:
        print("get synthref: PASS")
    else:
        print("get synthref: FAIL")

    if dev.get_synth_out() == 4000.0:
        print("get synthout: PASS")
    else:
        print("get synthout: FAIL")

    if dev.get_rf_in() == 0.0:
        print("get rfin: PASS")
    else:
        print("get rfin: FAIL")

    if dev.get_rf_out() == 0.0:
        print("get rfout: PASS")
    else:
        print("get rfout: FAIL")

    dev.set_loopback(True)
    if dev.get_loopback() == True:
        print("set Loopback -> enable PASS")
    dev.set_loopback(False)
    if dev.get_loopback() == False:
        print("set Loopback -> disable PASS")

    dev.set_synth_out(4000)
    if dev.get_synth_out() == 4000:
        print("set synth to 4000 PASS")
    else:
        print("set synth to 4000 FAIL")

    # case where the arduino code rejects the value reports it, and the python
    # file reverts to the previous value.
    dev.set_synth_out(1)
    if dev.get_synth_out() == 4000:
        print("set synth to 1 PASS")
    else:
        print("set synth to 1 FAIL")(123.560)
    print(f"pfd={dev.get_fPFD()}")
    print("setting to 200")
    dev.set_synth_ref(200)
    print(f"pfd={dev.get_fPFD()}")
    print("setting to 250")
    dev.set_synth_out(6000.560000)
    if dev.get_synth_out() == 6000.560000:
        print("set synth to 6000.560000 PASS")
    else:
        print("set synth to 6000.560000 FAIL")

    print("setting to 0")
    dev.set_synth_ref(0)
    print(f"pfd={dev.get_fPFD()}")
    print("setting to 50")
    dev.set_synth_ref(50)
    print(f"pfd={dev.get_fPFD()}")
    print("setting to 123.560")
    dev.set_synth_ref(123.560)
    print(f"pfd={dev.get_fPFD()}")
    print("setting to 200")
    dev.set_synth_ref(200)
    print(f"pfd={dev.get_fPFD()}")
    print("setting to 250")
    dev.set_synth_ref(205)
    print(f"pfd={dev.get_fPFD()}")
    print("setting to 601")
    dev.set_synth_ref(601)
    print(f"pfd={dev.get_fPFD()}")
    print("setting to 450")
    dev.set_synth_ref(450)
    print(f"pfd={dev.get_fPFD()}")
    print("setting to 300")
    dev.set_synth_ref(300)
    print(f"pfd={dev.get_fPFD()}")
    print("setting to 601")
    dev.set_synth_ref(601)
    print(f"pfd={dev.get_fPFD()}")
    print("setting to 573.25")
    dev.set_synth_ref(573.25)
    print(f"pfd={dev.get_fPFD()}")
    dev.set_rf_out(30)
    # dev.set_rf_out(0)
    dev.set_rf_out(12.3)
    dev.set_rf_out(-30)
    print(dev.get_rf_in())
