"""
This program provides a user-friendly client to communicate
with the Velmex Stepping Motor Controller through the RS232 interface.

Below are essential commands for communicating with the device.
Knowing them isn't necessary to use the client.

Written by Chase Parker and Danny Park
"""


# **** ESSENTIAL COMMANDS ****
# =====================

# m = motor, [1,2,3,4] (we only have 1 and 2)
# x = variable [speed, position, etc.] -- depends on command being used. |x| >= 1

# R - run program
# K - kill program
# V - check motor status -- B (busy), R (ready), J (jog/slew mode), b (jogging/slewing)
# X/Y - return current position of motor 1/2
# C - clear all commands from selected program

# IAmM-0 - set home
# IAmM0 - move motor to home
# IAmMx - move motor to position x
# ImMx - move motor by x steps
# SmMx - set motor speed
# AmMx - motor acceleration

# use commas to separate commands. Every script must end with R
# Example: S1M500, S2M200, IA1M0, IA2M0, R -- specifies motor speeds and moves to home

# Our model uses the E01 BiSlide lead screw
# 1000 steps = 0.25 inches = 6.35 mm
# 1 mm = 157 steps

import time
import os
from datetime import datetime
import serial

direc = "COM3" # device-specific. type "ls /dev" in console to find yours
x = os.path.exists(direc)
command = ''


# configure the serial connections 
# (the parameters differs on the device you are connecting to)
if x:
	ser = serial.Serial(
		port=direc,
		baudrate=9600,
		parity=serial.PARITY_NONE,
		stopbits=serial.STOPBITS_ONE,
		bytesize=serial.EIGHTBITS
	)
else:
	print("Serial port could not be found.")


def isfloat(N): # allows string -> float conversion for input parameters
	try:
		float(N)
		return True
	except ValueError:
		return False

def convert_units(N, unit, rev = False): # convert inches, mm --> steps
	N = float(N)
	if unit == "in":
		if rev:
			return round(N * 0.25 / 1000) # steps --> in
		return round(N / 0.25 * 1000) # in --> steps
	elif unit == "mm":
		if rev:
			return round(N * 6.35 / 1000) # steps --> mm
		return round(N / 6.35 * 1000) # mm --> steps
	else:
		return int(N) # int since steps must be whole numbers

def info(): # returns the positions of each motor in units of steps
	info1 = bytearray()
	ser.write(b"X")
	time.sleep(.05)
	while ser.inWaiting() > 0:
		info1.extend(ser.read(1))
	info1 = info1.decode("utf-8")
	info1 = info1.replace("^", "")
	if info1[0] != "-" or info1[0] != "+" or not info1[0].isnumeric():
		info1 = info1[1:]
	info1 = info1[:-1]
	info1 = convert_units(info1, "mm", True)

	info2 = bytearray()
	ser.write(b"Y")
	time.sleep(.05)
	while ser.inWaiting() > 0:
		info2.extend(ser.read(1))
	info2 = info2.decode("utf-8")
	info2 = info2.replace("^", "")
	if info2[0] != "-" or info2[0] != "+" or not info2[0].isnumeric():
		info2 = info2[1:]
	info2 = info2[:-1]
	info2 = convert_units(info2, "mm", True)

	s = info1, info2
	return s

def record_position():
	sttime = time.time()
	s = info()
	x = s[0]
	y = s[1]
	log = '/Users/FTS/Desktop/log.txt'
	with open(log, 'a') as logfile:
		logfile.write(str(sttime) + ' ' + str(x) + ' ' + str(y) + "\n")


def speed(speed1, speed2): # sets the speed of each motor in units of steps/sec
	global command
	if isfloat(speed1):
		speed1_conv = convert_units(float(speed1), "mm")
		if 1 < speed1_conv < 6000:
			command += "S1M" + str(speed1_conv) + ","
	if isfloat(speed2):
		speed2_conv = convert_units(float(speed2), "mm")
		if 1 < speed2_conv < 6000:
			command += "S2M" + str(speed2_conv) + ","
	else:
		speed2 = "N/A"

def position(pos1, pos2): # sets the absolute position of each motor in units of steps
	global command
	if isfloat(pos1):
		pos1_conv = convert_units(float(pos1), "mm")
		command += "IA1M" + str(pos1_conv) + ","
	if isfloat(pos2):
		pos2_conv = convert_units(float(pos2), "mm")
		command += "IA2M" + str(pos2_conv) + ","

def step(step1, step2): # shifts the motors by n steps
	global command
	if isfloat(step1):
		step1_conv = convert_units(float(step1), "mm")
		command += "I1M" + str(step1_conv) + ","
	if isfloat(step2):
		step2_conv = convert_units(float(step2), "mm")
		command += "I2M" + str(step2_conv) + ","

# version of the raster scan that moves continuously in x
def continuous_step(dim_x, dy, N):
	global command
	dy = convert_units(float(dy), "mm")
	N = int(N)
	dim_x = convert_units(float(dim_x), "mm")
	command += "I1M" + str(dim_x) + ","
	command += "I2M" + str(dy) + ","
	command += "I1M-" + str(dim_x) + ","
	for i in range(N - 1):
		command += "I2M" + str(dy) + ","
		command += "I1M" + str(dim_x) + ","
		command += "I2M" + str(dy) + ","
		command += "I1M-" + str(dim_x) + ","


def set_home(): # sets the "home", or origin, of the coordinate system
	global command
	command += "IA1M-0, IA2M-0, "

def go_home(): # sends both motors to the origin
	global command
	command += "IA1M0, IA2M0, "


# allows users to input their program of choice.
# 07-12-2022 "raster", which implements continuous_step()
# 07-18-2022 "custom", which lets users read in txt files as programs
def program():
	global command
	print("Choose a program to run (raster, custom):\n")
	program = input(">> ")
	if program == "raster" or program == "RASTER":
		dy = input (">> delta y (mm): ")
		dim_x = input(">> length in x (mm): ")
		N = input(">> # of iterations: ")
		time.sleep(0.5)
		print("\nCurrent position (mm): " + str(info()) + '\n')
		time.sleep(0.5)
		choice = input(">> Set current position as home? (Y/N): ")
		if choice == "Y":
			command = set_home(command)
		else:
			print("\nNo action taken...\n")
		pos1 = input(">> x starting point (mm): ")
		pos2 = input(" >> y starting point (mm): ")
		command = position(pos1, pos2, command)

		print("Creating raster...\n")
		time.sleep(1)
		command = continuous_step(command, dim_x, dy, N)

	# write a program by creating a txt file with 1 VXM command per line.
	# no commas necessary and don't end the program with "R"
	# fyi hit enter after the last command in the txt file so it can read it
	if program == "custom" or program == "CUSTOM":
		filename = input("filename: ")
		with open(filename) as f:
			txt = f.read().split("\n")
			for c in txt:
				command += c + ", "
	return command





if ser.isOpen():

	ser.write(b"F") # enable online mode on startup
	while 1:
		time.sleep(0.5)
		print('\nChoose a command from below.\n')
		print('INFO, SPEED, POSITION, STEP, SET_HOME, GO_HOME, PROGRAM, VXM\n')
		command = 'C '
		i = input(">> ")
		if i == 'exit':
			ser.close()
			exit()
		elif i == 'INFO' or i == 'info':
			s = info()
			print("\nCurrent position (mm): " + str(s) + '\n')

		elif i == 'SPEED' or i == 'speed':
			print('\n')
			print('SET THE SPEED OF EACH MOTOR: \n')
			speed1 = input("x speed (mm/sec): ")
			speed2 = input("y speed (mm/sec): ")
			speed(speed1, speed2)
			print("\nx motor: " + speed1 + " mm/sec\ny motor: " + speed2 + " mm/sec\n")

		elif i == 'POSITION' or i == 'position':
			print('\n')
			print('SET ABSOLUTE POSITION OF EACH MOTOR:\n')
			pos1 = input("x position (mm): ")
			pos2 = input("y position (mm): ")
			position(pos1, pos2)
			print("\nNew position: (" + pos1 + ", " + pos2 + ")\n")


		elif i == "STEP" or i == "step":
			print('\n')
			print('CHOOSE HOW MANY STEPS TO MOVE EACH MOTOR:\n')
			step1 = input("x shift (mm): ")
			step2 = input("y shift (mm): ")
			step(step1, step2)
			print("\nx shift: " + step1 + " mm\ny shift: " + step2 + " mm\n")

		elif i == 'SET_HOME' or i == 'set_home':
			print('\n')
			s = info()
			print("Current position (mm): " + str(s) + '\n')
			choice = input('Set current motor positions as new origin? (Y/N): ')
			if choice == "yes" or choice == "YES" or choice == 'y' or choice == "Y":
				set_home()
				print("\nSetting current position as new origin...\n")
			else:
				print("\nNo actions taken...\n")
				time.sleep(0.5)

		elif i == "GO_HOME" or i == "go_home":
			print('\n')
			choice = input('Send motors to origin? (Y/N): ')
			if choice == "yes" or choice == "YES" or choice == 'y' or choice == "Y":
				go_home()
				print("\nSending motors to the origin...\n")
			else:
				print("\nNo actions taken...\n")
				time.sleep(0.5)

		elif i == "program" or i == "PROGRAM":
			print('\n')
			program()
		elif i == "vxm" or i == "VXM":
			print('\nPerform a VXM command or series of commands, separated by commas:\n')
			command = input(">> ")
			command += ", " # to allow R to be appended to the end
		elif i == "K" or i == "k" or i == "kill" or i == "KILL":
			command = "K"
		if command != "K":
			command += "R"

		# for debugging, printing the output of the final command is helpful
		#print(command)

		command = command.encode("utf-8") # converts into bytes so serial can understand
		ser.write(command)
		
		# need to rewrite this — can't kill the program until motor stops,
		# no way to halt it if it gets stuck in an infinite loop

		while True: # read timestamp and position into log file while motor is moving
			ser.write(b'V')
			time.sleep(0.05)
			status = ""
			while ser.inWaiting() > 0:
				status = ser.read(ser.inWaiting()).decode("utf-8").strip()
			if status == "^" or status == "R":
				break
			
