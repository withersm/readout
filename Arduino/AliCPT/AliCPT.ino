/* ping_pin_stat */
/* v01: */
/* look for single-char request for data, respond immediately with 
/*     state of pin(s), format TBD 
/* state can eventually include multiple digital and analog values, 
/*     but initial design is to give state of 2 pins: 
/*   a) digital pin to be connected to strobe ref of chopper wheel 
/*   b) digital pin to be connected to optointerrupter ref of polarizer 
/*         wheel  
/*   this would be queried by control computer at 200Hz to match the 
/*   TES detector readout cadence of AliCPT, thus there is plenty of 
/*   time to mess about with output, but the read of the pins happens 
/*   every time through the loop, so a check for query character happens 
/*   right after digital read (and/or analog read, in principal)
  */

#define VERSIONTHING "1.0"
#define PROG_NAME "AliCPT"

char date_prog[] = {"KLT, 2022-10-24"};


#define USEBAUD 115200
#define LEN_ALLBYTE 9

/* S == ASCII 83 decimal, 123 octal, x053 hex */
byte querycode = 83;

int ndpins = 8;
int dpins[] = {27, 26, 25, 24, 23, 22, 21, 20};
int apins[] = {A0, A1, A2, A3};
unsigned int avals[4];
unsigned int m=0; 

// put these characters in front of initial print statements following 
// reset or boot
char prepend_chars[] = {"# "};     // or ""


void setup() {
    int ii;
    
    Serial.begin(USEBAUD);

    Serial.print(prepend_chars);
    Serial.print(PROG_NAME);
    Serial.print(" v");
    Serial.println(VERSIONTHING);
    
    Serial.print(prepend_chars);
    Serial.println(date_prog);
    
    Serial.print(prepend_chars);
    Serial.println("reading pins: ");
    Serial.println("");

    
    /* set digital inputs, and set pullup Rs */
    for(ii=0;ii<ndpins;ii++)  {
        pinMode(dpins[ii], INPUT_PULLUP);
    }
    // set register A to input mode
    DDRA = B00000000;
}

void loop() {
    int ii, jj, navail;
    byte cval;
    byte allbyte[LEN_ALLBYTE];
    byte a_val;

    if( (navail = Serial.available()) > 0 )  {
        for(jj=0; jj<navail; jj++)  {
            Serial.readBytes(&cval, 1);
            if( cval == querycode )  {

                // Digital signal through register A
                allbyte[0] = PINA;
                //Serial.println(a_val, BIN);
                
                // Analog readout
                avals[0] = analogRead(apins[0]);
                avals[1] = analogRead(apins[1]);
                avals[2] = analogRead(apins[2]);
                avals[3] = analogRead(apins[3]); 
                
                // following is little-endian ordered byte pairs
                allbyte[1] = lowByte(avals[0]);
                allbyte[2] = highByte(avals[0]);
                allbyte[3] = lowByte(avals[1]);
                allbyte[4] = highByte(avals[1]);
                allbyte[5] = lowByte(avals[2]);
                allbyte[6] = highByte(avals[2]);
                allbyte[7] = lowByte(avals[3]);
                allbyte[8] = highByte(avals[3]);

                //Serial.println(allbyte[0], BIN);
                //Serial.println(allbyte[1], BIN);
                //Serial.println(allbyte[2], BIN);
                //Serial.println(allbyte[3], BIN);
                //Serial.println(allbyte[4], BIN);
                //Serial.println(allbyte[5], BIN);
                //Serial.println(allbyte[6], BIN);
                //Serial.println(allbyte[7], BIN);
                //Serial.println(allbyte[8], BIN);
                Serial.write(allbyte, 9);
                delay(1);
                Serial.println("");
                
                
                //Serial.println(""); 
                }
        }
    }
}
