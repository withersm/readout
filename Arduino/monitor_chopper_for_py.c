/* monitor_chopper */
/* monitor_chopper outputfile [/dev/ttyXXXX] */
/* 
KLT (date, version in defines below)
run timer-interrupt sequence to check Arduino for digital state 
  of chopper wheel; nominal cadence of queries 5ms to match the 200Hz 
  of the TES DAQ; 
will test double that for better interpolation, but 
  in practice if off even 2.5ms with chopper, at 20 Hz (50ms), cos(err) = 0.95, 
  which is awkward but perhaps livable; at 10Hz 2.5ms err cos(err) = 0.988, 
  and if can average over some jitter becomes negligable for the types of 
  optical tests we'll be doing, except perhaps polarization

initial: Arduino is Uno v3, 115200 baud, query request is "S" (LF OK, 
  though not required); returns "0<LF>" or "1<LF>" based on state of 
  single digital input pin, presumed to be attached to chopper
later development: Arduino also monitors 2nd pin, which is attached 
  to optointerrupter on polarizer rotation gizmo -- requires reprogramming 
  Arduino and adding hardware (+5V to optointerrupter and signal return)
*/


#define VERSIONTHING 1.0
#define DESCRIP "2022-10-25 KLT"



#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <termios.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <math.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <sys/sysmacros.h>
#include<dirent.h>
#include<fcntl.h>


void alarm_wake(int sig_num);
void open_serial_ports();
double get_time_double();
void convert_baud();
void rs232_query_response(int fdthing, char aquery[], int ms_delay);


/* cadence: 1/(200 Hz) in usec */ 
useconds_t cadence = 5000;        /* us */



/* note: following serial port params are written as arrays, */
/* following use in FTS software with 3 different ports used at once */
char sprts[1][15] = {"/dev/ttyACM3"};        /* deault */
int nsprts = 1;
long bauds[1] = {115200};           /* for Arduino */
int databits[1] = {8};
int stopbits[1] = {1};
int parity[1] = {0};       /* 0=none, 1=odd, 2=even */
int flowcontrol[1] = {0};   /* 0=none, 1=hard, 2=soft */
int receive_mode[1] = {3};  /* 1=canonical, 2=non-canonical, 3=raw */
int fd[1];               /* file ID for Arduino serial */

char aquery[3];      /* must fill with S<LF><null> later */
int aquery_len = 2;

FILE *ofl;               /* file ID for output text file */



struct timeval tv;
double tpoint;         /* time read */

int warning_flag = 0;   /* set if there is an overrun */




/****************************************************/
/****************************************************/
int main()
{
    return 0;
}

int run(char *argv1, char *argv2)
{
    char buf10[10], buf100[100];
    char outfilename[100];
    unsigned long k, kstart;

    tpoint = -1.0;

    /* establish command line args */
    // printf("argc = %d\n", argc);

    sscanf(argv1, "%s", outfilename);
    sscanf(argv2, "%s", sprts[0]);
    printf("%s %s  %s \n", argv1, outfilename, sprts[0]);



    sprintf(aquery, "S\n");  // adds null as 3rd char

    /* open serial port */
    open_serial_ports();
    usleep(100000);


    /* open output file */
    /* append */
    ofl = fopen(outfilename, "a");



    tcflush(fd[0], TCIFLUSH);
    tcflush(fd[0], TCOFLUSH);

    /* test a few and try to clear the system */
    for(k=0;k<10;k++)  {
        write(fd[0], aquery, 2);
        usleep(5000);
        bzero(buf100, 100);
        read(fd[0], buf100, 99);
        usleep(2000);
    }

    /* set up signal */
    signal(SIGALRM, alarm_wake);
    // ualarm(80000, 80000);
    ualarm(cadence, cadence);


    kstart = 100;   /* don't start writing til after this point; */
                    /* this is even kludgier than query/read sequence */
                    /* above, but could not find problem that systematically */
                    /* lost a char and then added an LF in sequential lines */
                    /* once within the first ~50 samples, every time... */
                    /* so decided to skip */
                    /* writing some comfortable number and start after */
                    /* the system was already "in the groove" */
    k = 0;
    while(1)  {
        // printf("loop\n");
        /* if tpoint > 0, means got a signal and read the time, asked */
        /* Arduino for the pin state */
        if( tpoint > 0 )  {
            //printf("  k=%ld\n", k);
            /* 100us usually works... 200us too */
            /* at 115200 baud, ~9us period, 1 char in ~100us, */
            /* 3 char in ~300us, 400us should be enough */
            /* HOWEVER, might have USB latency of 1ms? */
            usleep(1450);   /* should be enough time for Arduino to respond */
            if( warning_flag > 0 )  {
                warning_flag = 0;
                read(fd[0], buf100, 99);
                sprintf(buf10, "xx");
            }  else  {
                bzero(buf10, 10);
                read(fd[0], buf10, 4);
            }
            // will need to mod buf, readlength, printout for different 
            // quantities monitored
            buf10[2] = '\0';
            if( k >= kstart )  {
                fprintf(ofl, "%.3lf %s\n", tpoint, buf10);
                fflush(ofl);
            }



            tpoint = -1;    /* reset so this conditional isn't run again */
                            /* until alarm_wake() is executed again */
            k++;
        }
        usleep(1000);     /* not too fast through */
    }


}


/****************************************************/
/****************************************************/

void alarm_wake(int sig_num)
{
    suseconds_t usecs;
    struct timespec stttt;
    char buf[1];

    if(sig_num == SIGALRM)  {
        // if( tpoint > 0 )  printf("COMPLAIN: tpoint = %.3lf\n",tpoint);
        if( tpoint > 0 )  warning_flag = 1;
        clock_gettime(CLOCK_REALTIME, &stttt);
        tpoint = stttt.tv_sec + stttt.tv_nsec*1e-9;
        /* now for Arduino query */
        write(fd[0], aquery, 2);
        /* read back in main(), don't dwell here in interrupt handler */
    }

}

//           struct timeval {
//               time_t      tv_sec;     /* seconds */
//               suseconds_t tv_usec;    /* microseconds */
//           };

/***************************************************/

double get_time_double()
{
    struct timespec stttt;
    double ttt;

    clock_gettime(CLOCK_REALTIME, &stttt);

    //printf("time = %.9f\n", tttt.tv_sec + tttt.tv_nsec*1e-9);
    ttt = stttt.tv_sec + stttt.tv_nsec*1e-9;

    return ttt;
}

/***************************************************/

void open_serial_ports()
{
    int pp;
    struct termios newtio;
    char buf[100];
    ssize_t res;
    speed_t baudcode;

    for(pp=0; pp<nsprts; pp++)  {
        fd[pp] = open(sprts[pp], O_RDWR | O_NOCTTY);
        if( fd[pp] < 0 )  {
            perror(sprts[pp]);
            exit(-1);
        }

        bzero(&newtio, sizeof(newtio));

        convert_baud(bauds[pp], &baudcode);
                  /* REM &baudcode is pointer value, pointing to memory */
                  /* location holding the value of baudcode */
        cfsetispeed(&newtio, baudcode);
        cfsetospeed(&newtio, baudcode);
        /* note bauds[] must be been converted to the */
        /* unsigned int code (speed_t) that termios wants */


        /* other flags */
        /* CS8 - 8bit, no parity, 1 stop bit   */
        /* CLOCAL - local connection, no modem control */
        /* CREAD - enable receiving */
        /* CSTOPB stets 2 stop bits */
        /* CS7 7 bits, CS8 8 bits */
        if( databits[pp] == 8 )  {
            newtio.c_cflag |= CS8;
        }
        if( databits[pp] == 7 )  {
            newtio.c_cflag |= CS7;
        }
        if( stopbits[pp] = 2 )  {
            newtio.c_cflag |= CSTOPB;
        }
        //newtio.c_cflag |= CS7 | CLOCAL | CREAD | CSTOPB;
        newtio.c_cflag |= CLOCAL | CREAD;

        //newtio.c_cflag |= PARENB;        /* enable parity adding/checking */
        //newtio.c_cflag &= ~PARENB;
        // PARODD sets odd parity, otherwise even is used
        //newtio.c_cflag &= ~CSTOPB;
        if( parity[pp] == 1 )  {
            newtio.c_cflag |= PARENB | PARODD;
        }
        if( parity[pp] == 2 )  {
            newtio.c_cflag |= PARENB;
        }
        if( parity[pp] == 0 )  {
            newtio.c_cflag &= ~PARENB;
        }

        if( receive_mode[pp] == 1 )  {         /* canonical */
            newtio.c_lflag |= ICANON | ECHO | ECHOE | ISIG;
        }
        if( receive_mode[pp] == 2 )  {         /* non-canonical */
            newtio.c_lflag &= ~ICANON;
            newtio.c_cc[VMIN] = 0;
            newtio.c_cc[VTIME] = 0;
        }
        if( receive_mode[pp] == 3 )  {         /* raw */
            //newtio.c_lflag &= (ICANON | ECHO | ECHOE | ISIG);  /* raw */
            newtio.c_lflag &= ~(ICANON | ECHO | ISIG | ECHONL | IEXTEN);
            newtio.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP
                           | INLCR | IGNCR | ICRNL | IXON);
            newtio.c_oflag &= ~OPOST;  /* no post-processing */
            // newtio.c_cflag &= ~(CSIZE | PARENB);
              /* CSIZE is character size mask, values CS5, CS6, CS7, CS8 */
            newtio.c_cc[VMIN] = 0;
            newtio.c_cc[VTIME] = 1;
        }
        //newtio.c_cflag &= ~CSIZE;
        newtio.c_oflag &= ~OPOST;       /* no post-processing, raw output */

        /* flow control */
        if( flowcontrol[pp] == 0 )  {  /* none */
            newtio.c_iflag &= ~(IXON | IXOFF | IXANY);    /* no soft */
            newtio.c_cflag &= ~CRTSCTS;      /* no hardware flow control */
        }
        if( flowcontrol[pp] == 1 )  {    /* hard */
            newtio.c_iflag &= ~(IXON | IXOFF | IXANY);    /* no soft */
            newtio.c_cflag |= CRTSCTS;
            /* but this might not be consistent with CTS/DTR style? */
        }
        if( flowcontrol[pp] == 2 )  {    /* soft */
            newtio.c_iflag |= IXON | IXOFF | IXANY;
            newtio.c_cflag &= ~CRTSCTS;      /* no hardware flow control */
        }

        if( 0 && (pp==0) )  {
            /* exactly this worked on pol3 int test2.c querying Heidenhain */
            cfsetispeed(&newtio, bauds[pp]);
            cfsetospeed(&newtio, bauds[pp]);
            newtio.c_cflag |= CS7 | CLOCAL | CREAD | CSTOPB;
            newtio.c_cflag |= PARENB;        /* enable parity adding/checking */
            newtio.c_lflag &= (ICANON | ECHO | ECHOE | ISIG);  /* raw */
            newtio.c_iflag &= ~(IXON | IXOFF | IXANY);  /* no soft flow cont */
            newtio.c_cflag &= ~CRTSCTS;      /* no hardware flow control */
            newtio.c_oflag &= ~OPOST;       /* no post-processing, raw output */
            newtio.c_cc[VMIN] = 0;
            newtio.c_cc[VTIME] = 1;

        }

        tcflush(fd[pp], TCIFLUSH);
        tcsetattr(fd[pp], TCSANOW, &newtio);  /* TCSANOW == load new sets now */
        tcflush(fd[pp], TCIFLUSH);
        tcflush(fd[pp], TCOFLUSH);
        if( 0 )  {
            sprintf(buf, "*IDN?\n");
            write(fd[pp], buf, strlen(buf));
            bzero(buf, 100);
            usleep(50000);
            res = read(fd[pp], buf, 99);
        }
        usleep(100000);

        //printf("(%d) %s: *IDN? returns %s (%ld).\n", pp, sprts[pp], buf, res);
    }
}

/***************************************************/

void convert_baud(long baudval, speed_t *destbaud)
{
    /* these conversions reference definitions in termbits.h, which */
    /* is included in termios.h */
    /* the value is assigned to the c_cflag in the termios struct */
    /* destbaud is speed_t, typedef'ed as unsigned int (not long) */
    if( baudval == 50 )  *destbaud = B50;
    if( baudval == 75 )  *destbaud = B75;
    if( baudval == 110 )  *destbaud = B110;
    if( baudval == 134 )  *destbaud = B134;
    if( baudval == 150 )  *destbaud = B150;
    if( baudval == 200 )  *destbaud = B200;
    if( baudval == 300 )  *destbaud = B300;
    if( baudval == 600 )  *destbaud = B600;
    if( baudval == 1200 )  *destbaud = B1200;
    if( baudval == 1800 )  *destbaud = B1800;
    if( baudval == 2400 )  *destbaud = B2400;
    if( baudval == 4800 )  *destbaud = B4800;
    if( baudval == 9600 )  *destbaud = B9600;
    if( baudval == 19200 )  *destbaud = B19200;
    if( baudval == 38400 )  *destbaud = B38400;
    if( baudval == 57600 )  *destbaud = B57600;
    if( baudval == 115200 )  *destbaud = B115200;
    if( baudval == 460800 )  *destbaud = B460800;
    if( baudval == 500000 )  *destbaud = B500000;
    if( baudval == 576000 )  *destbaud = B576000;
    if( baudval == 921600 )  *destbaud = B921600;
    if( baudval == 1000000 )  *destbaud = B1000000;
    if( baudval == 1152000 )  *destbaud = B1152000;
    if( baudval == 1500000 )  *destbaud = B1500000;
    if( baudval == 2000000 )  *destbaud = B2000000;
    if( baudval == 3000000 )  *destbaud = B3000000;
    if( baudval == 4000000 )  *destbaud = B4000000;
}

/****************************************************/

void rs232_query_response(int fdthing, char aquery[], int ms_delay)
{
    char buf[100], buf2[100];

    tcflush(fdthing, TCIOFLUSH);
    write(fdthing, aquery, strlen(aquery));
    usleep(ms_delay*1000);
    bzero(buf, 100);
    read(fdthing, buf, 99);
    strcpy(buf2, aquery);
    buf2[strlen(aquery)-1] = '\0';
    printf(" %s: %s\n", buf2, buf);
    printf(">>>%c<<<\n", aquery[0]);
}

/****************************************************/








