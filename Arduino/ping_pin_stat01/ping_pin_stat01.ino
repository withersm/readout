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
#define PROG_NAME "ping_pin_stat"

char date_prog[] = {"KLT, 2022-10-24"};


#define USEBAUD 115200

/* S == ASCII 83 decimal, 123 octal, x053 hex */
byte querycode = 83;

//int dpins[] = {12, 11, 10, 9, 8, 7, 6, 5, 4};
//int dpins[] = {12, 11};
int dpins[] = {7, 6, 5, 4, 3, 2};
int ndpins = 6;

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

    /* set digital inputs, and set pullup Rs */
    for(ii=0;ii<ndpins;ii++)  {
        pinMode(dpins[ii], INPUT_PULLUP);
    }

    Serial.print(prepend_chars);
    Serial.println("reading pins: ");
    for(ii=0;ii<ndpins;ii++)  {
        Serial.print(prepend_chars);
        Serial.print("  ");
        Serial.println(dpins[ii]);
    }
    Serial.println("");
}

void loop() {
    int dpin_state[ndpins];
    int ii, jj, navail;
    byte cval;

    for(ii=0; ii<ndpins; ii++)  {
        dpin_state[ii] = digitalRead(dpins[ii]);
    }
    if( (navail = Serial.available()) > 0 )  {
        for(jj=0; jj<navail; jj++)  {
            Serial.readBytes(&cval, 1);
            if( cval == querycode )  {
                //Serial.print("cval = ");
                //Serial.println(cval);
                
                for(ii=0;ii<ndpins;ii++)  {
                    // should be single numeral 0 or 1 for digital pin
                    Serial.print(dpin_state[ii]);
                }
                Serial.println(""); }
        }
    }
}
