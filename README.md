# mset-component-eq
computing explicit equations for the interior of hyperbolic components for unicritical multibrots of
the form z^N+c

based on

"A parameterization of the period 3 hyperbolic component of the Mandelbrot set
D Giarrusso, Y Fisher
Proc AMS, Volume 123, Number 12. December 1995"

"Groebner Basis, Resultants and the generalized Mandelbrot Set
YH Geum, KG Hare, 2008"

"Sylvester's Identity and Multistep Preserving Gaussian Elimination
EH Bareiss, 1968"


### Disclaimer

Although I tried my best to ensure correctness of the implementation and therefore the results,
the code comes with no warranty.


### Overview

The code computes an explicit equation for the hyperbolic component of given period p of the multibrot of degree D.
A typical output:

    A complex number c is in a period-4 component of
    the multibrot of degree 2 if the following has a
    complex solution m with ||m|| < 1
    ( with largest |integer coefficient| around 10^19 )

    A12*m^12+A11*m^11+A10*m^10+A9*m^9+A8*m^8+A7*m^7+A6*m^6+A5*m^5+A4*m^4+A3*m^3+A2*m^2+A1*m+A0 = 0

    A12 = +1
    A11 = -192+64*c^2
    A10 = +16896+512*c^4-1024*c^3-8192*c^2
    A9 = -901120-49152*c^6+409600*c^2+98304*c^3-98304*c^5
    A8 = +8650752*c^5-2359296*c^7-1114112*c^8-2359296*c^4-2359296*c^3-7864320*c^2+4325376*c^6+32440320
    A7 = -100663296*c^3+8388608*c^10+33554432*c^9+92274688*c^8+163577856*c^7-352321536*c^5-830472192-88080384*c^6-125829120*c^2+50331648*c^4
    other parameters: see file `_z2p4.resultant.txt`

If one puts a complex number c into the above equation and can find a solution m within the unit disc - with mathematical certainty, it
has been proven, that c belongs to the corresponding hyperbolic component.

It is also possible to enter a complex interval and use e.g. interval Newton (see my subdiv-core project) to definitely answer
the root question. If one is definitely contained within the unit circle, the entire complex interval c belongs to the component.

The code is based on computing the resultant of two multivariate polynomials: one describing the strict period P periodic points,
and one describing the derivative of of the cycle minus the parameter m.


### Input

Per command-line:
`mset-component-eq multibrot=2 period=4`

computes the z^2+c classical Mandelbrot set's hyperbolic components of period 4.

Per source code:
`#define _INVOKECLAIMVERIFICATIONS`

If defined, after division of big integers or polynomials some verifications are performed. This is generally used when implementing a faster division routine..


### Output

A textual output as above in the log file. 

All coefficients Ai are also stored in indiviudal text files of the name form `__md2_p3_A5.coeff.txt` 

Intermediate and final matrices of the Bareiss algorithm are stored as `__md2_p4_k23.matrix` The computation be terminated by simply closing the command window. The last saved matrix should then be deleted as it does not contain valid data.


### Implementation

The main parts are big integer arithmetics (`struct BigInt`), the polynomials in at most the three variables z,c,m (`struct ZMCpolynom`) and
the fraction-free Gaussian elimination to compute the determinant (`function fractionFreeGaussDet_TA`).

`const int64_t CHUNKSIZE`
Memory is allocated blockwise of recommended 128 MB (32-bit systems, or 1 GB (64-bit systems) and handed out to the requesting 
object to keep memory fragmentation and memory hole merging to a minimum.

Big integers are stored to base 10^9.


### Limitations
- big integers are fixed at compile time to store at most 128 digits of size 10^9 (`const MAXBIGINTDIGITS`). Overflow is monitored.
- the Bareiss algorithm demands the principal minors to have non-zero determinant. This is not tested beforehand, if present, division by
zero will occur at some point and terminates the program with a message.
- the resultant is determined in its expanded form, but often takes Res=G^q for some small q. This factorization is currently not implemented.
- all polynomial multiplications are performed anew as needed. Upcoming optimization will store intermediate results to reuse those, especially in the a(k)(k,k) multiplications.
- the current computing matrix is stored directly onto hard disc. If the calculation is prematurely ended by closing the command window, computation can be resumed later on after deleting the last matrix file with the highest kNNN numbe as this file dos not yet contain fully valid data.


### Further information

Background on the implementation can be found here

https://fractalforums.org/fractal-mathematics-and-new-theories/28/explicit-equations-for-the-interior-of-hyperbolic-components/3463/msg23928#new


### Contact

Please direct any comments to:

marcm200@freenet.de

Marc Meidlinger, August-September 2020

