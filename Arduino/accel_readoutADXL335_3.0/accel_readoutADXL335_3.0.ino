// accel readout ADXL335
// 3-axis analog output accelerometer, +/-3g range, 3.3V

#define VERSIONTHING 3.0

//
// v2.0  read out 4th analog chan. for reference (but just once)
//       keep 1 LF, so have 11 bytes instead of 9 bytes per datum
// v3.0  move to the nano, 16 MHz(?) Mega328
//       move to flexitimer2 -- supposedly should work w/ nano
//           aim for readout as fast as can, maybe ~700us cadence?
//       keep 115200 baud, but may need faster
//
// REM: make sure arduino soft is set to the nano
//
// old notes:
//   use timer3 to read out at 1KHz - doesn't work... use timer1
//   code assumes Pro Mini 3.3V, Atmel MEGA328
//
// ADXL335 is +/-3g, so 1g => ~1024/6=170 counts, flipping 
//   over => ~340 counts -- but actually get from ~410 - 610 
//   in all 3 channels from down to up, suggesting +/-5g 

// debug notes:
// 2021-08-07 v1.1 on "orig" pro mini: 
//    often get very few mis-steps in index (since put m++ into 
//      interrupt function rather than loop()), sometimes horrible 
//      hash of index steps (delta = 2, 3, more, randomly and often); 
//      but when get few, they tend to be spaced w/ 256 periodicity, 
//      with delta = 2 at 10 and 256 (starting just after a delta=2)
//    commenting out the init of the char string to send (strictly 
//      not neccessary) seems to reduce the number of bad scans 
//      (w/ many index step errors), but still get the [10,256] 
//      delta=2 step errors

//#include<TimerOne.h>
//#include<TimerThree.h>
//#include<FlexiTimer2.h>

// 1000000      (115200 for debug)
#define BAUD 115200
// 2021-11-11 discovered 1M baud wasn't working, on either no. 1 or no. 2 accel 
// accel reader box, or at least something wasn't working, though it had a couple 
// months earlier, found that wasn't getting *any* LFs; 
// 115200 works, and it ought to be fast enough to 
// run at 1KHz sample rate, so sticking with this
// above are old notes relevant to 3.3v 8MHz mini

// samp_period in us for TimerOne, in seconds for TimerThree
long samp_period = 700;    // time between samples, us
double s_period = 0.000625;   // s 
// new note for nano: timer1: 800us,115200 baud a few COMPLAINs
//   1M baud:
//     700us, 1M baud no apparent COMPLAINs, seems OK
//     625us sometimes has problems
//     575 bad
// flexitimer2 (1M baud): 
//     700us OK, 600us too?   625 looks OK, needs further checking
//     500us OK too? no, sometimes has trouble;   400us, 450us definitely bad
// note at 115200 baud, w/ 11 byte data (1*2 + 4*2 + 1), and ~12 clock 
//     cycles per byte => ~132 clock cycles per data point, means 
//     min (1/115200)*132 = 0.001145s per data point -- but had been doing 
//     fine on 3.3v pro mini with that baud and 0.001s per data point, so 
//     must be more like 10 or 10.5 clock cycles per byte transmission
//   but this may not be the relevant limit, since even with 1M baud 
//     having trouble with those 11 bytes per datum at less than 0.001s

// note: minimum already-formatted output string min ~1400us 
// samp period before run into over-run on sampling; 
// sending binary bytes (8 total, +CR/LF) is fine at 1000us; 
// (all at 1e6 baud, max for the 8MHz pro mini)

int apins[] = {A0, A1, A2, A3};   // add A3
int selftest = 12;

unsigned int avals[4];       // 3->4
unsigned int m=0;          // running index of data values, rolls over
byte get_meas = 0;
byte one_byte = 1;

#define LEN_ALLBYTE 14
   // 12 -> 14


void setup() {

}

void loop() {
    char allstr[30];
    byte allbyte[LEN_ALLBYTE];
    int ii;
    // 

    if(get_meas == 1)  {
      
        // get measurement and print
        avals[0] = analogRead(apins[0]);
        avals[1] = analogRead(apins[1]);
        avals[2] = analogRead(apins[2]);
        avals[3] = analogRead(apins[3]);   // just once
        //avals[0] += analogRead(apins[0]);
        //avals[1] += analogRead(apins[1]);
        //avals[2] += analogRead(apins[2]);
        // two each of x,y,z is OK, 3 each takes too long
        // having two samples averaged helps mitigate 
        // aliasing 
        // when add A3 for ref, 2ea x,y,z and 1ea ref is too long
        // so just do single for x,y,z

        
        // following is little-endian ordered byte pairs
        // useful for pol3, at least
        allbyte[0] = lowByte(m);
        allbyte[1] = highByte(m);
        allbyte[2] = lowByte(avals[0]);
        allbyte[3] = highByte(avals[0]);
        allbyte[4] = lowByte(avals[1]);
        allbyte[5] = highByte(avals[1]);
        allbyte[6] = lowByte(avals[2]);
        allbyte[7] = highByte(avals[2]);
        allbyte[8] = lowByte(avals[3]);
        allbyte[9] = highByte(avals[3]);
        //allbyte[8] = 13;     // CR carriage return
        //allbyte[9] = 10;
        //Serial.write(allbyte, 10);
        allbyte[10] = 10;     // LF line feed    // allbyte index 8->10


        Serial.write(allbyte, 11);  // 9->11
        //m++;
        get_meas = 0;
    }
    
    //for(ii=0;ii<LEN_ALLBYTE;ii++)  allbyte[ii] = 0;

}
