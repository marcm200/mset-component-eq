/*

	formula for the hyperbolic component of period P
	of the degree D uniciritcal multibrot z^D+c
	
	based on:
	
	"A parameterization of the period 3 hyperbolic component of the Mandelbrot set
	D Giarrusso, Y Fisher
	Proc AMS, Volume 123, Number 12. December 1995"

	"Groebner Basis, Resultants and the generalized Mandelbrot Set
	YH Geum, KG Hare, 2008"

	"Sylvester's Identity and Multistep Preserving Gaussian Elimination
	EH Bareiss, 1968"
	

	Marc Meidlinger
	August-September 2020
	
	2020 09-14: added openMP support for polynomial multiplication,
		   sorted list to store polynomial coefficients to allow binary search
	2020-09-11: added checks for very small chunk sizes
	
*/

#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include "string.h"
#include "math.h"
#include "time.h"
#include "omp.h"


// defines as compiler switches

// verifies certain claims during computation
// e.g. remainder of division < divisior
// jump shortcut in division does not overshoot etc
// sign and result check for bigint divion
// satisfying the result equations

//#define _INVOKECLAIMVERIFICATIONS

	
// defines as small functions

FILE *flog=NULL;

#define LOGMSG(TT) \
{\
	printf(TT);\
	if (flog) {\
		fprintf(flog,TT);\
		fflush(flog);\
	}\
}

#define LOGMSG2(TT,AA) \
{\
	printf(TT,AA);\
	if (flog) {\
		fprintf(flog,TT,AA);\
		fflush(flog);\
	}\
}

#define LOGMSG3(TT,AA,BB) \
{\
	printf(TT,AA,BB);\
	if (flog) {\
		fprintf(flog,TT,AA,BB);\
		fflush(flog);\
	}\
}

#define LOGMSG4(TT,AA,BB,CC) \
{\
	printf(TT,AA,BB,CC);\
	if (flog) {\
		fprintf(flog,TT,AA,BB,CC);\
		fflush(flog);\
	}\
}

#define TIMEOUT(TT,TIMESPEC,TIMEALL) \
{\
	double lokald=100.0*(double)(TIMESPEC)/(TIMEALL);\
	LOGMSG2("\n%3.0lf%% ",lokald);\
	LOGMSG4("(clocks in %s %i <-> all %i)",\
		TT,TIMESPEC,TIMEALL);\
}


// consts

const int32_t MAXKEYS=2048;
const int32_t INIT_ZMCTERMS=128; // initial terms
const int32_t MAXDIMENSION=(int32_t)1 << 14;
const int32_t MAXDEGREE=(int32_t)1 << 13;

// needs to be a value < 2^30, so doubling fits into
// signed int32_t
const int32_t MAXTERMSPERPOLYNOM=(int32_t)1 << 20;

const int32_t MAXPTR=2048; // allocated chunks
const int32_t UINT30MAX=((int32_t)1 << 30) - 1;
const int32_t INT32MAX=0b01111111111111111111111111111111;
const int64_t INT62MAX=(int64_t)1 << 62;
const int8_t NULLPOLYNOMID=0;
const int8_t NOTNULLPOLYNOMID=1;

// chunk size: might need adjustment according to operating system
// for 32-bit system 128 MB is recommended
const int64_t CHUNKSIZE=(int64_t)1 << 27; 


// structs

struct DynSlowString {
	int32_t memory;
	char* text;
	
	DynSlowString(void);
	DynSlowString(const int32_t);
	virtual ~DynSlowString ();
	
	void add(const char*);
	void add(DynSlowString&);
	void setEmpty(void);
};

#include "bigint.cpp"

// polynom: stores (positive) integer exponents of Z,M,C 
// variables and a BigInt factor

struct ZMCterm {
	BigInt factor;
	int32_t Zexponent,Mexponent,Cexponent;
	
	void getStr(DynSlowString&);
	void save(FILE*);
	void load(FILE*);
	void setTermToZero(void);
	void ausgabe(FILE*);
	void copyFrom(ZMCterm&);
	int8_t isPositiveOne(void);
};

typedef ZMCterm *PZMCterm;
typedef PZMCterm *PPZMCterm;

struct ZMCtermMemoryManager {
	ZMCterm* current;
	int32_t allocatedIdx,freeFromIdx,allocatePerBlockIdx;
	PZMCterm ptr[MAXPTR];
	int32_t anzptr;
	int64_t memory;
	
	ZMCtermMemoryManager();
	ZMCtermMemoryManager(const int64_t);
	virtual ~ZMCtermMemoryManager();
	ZMCterm* getMemory(const int32_t);
	
	void freePhysically(void);
};

struct PZMCtermMemoryManager {
	PZMCterm* current;
	int32_t allocatedIdx,freeFromIdx,allocatePerBlockIdx;
	PPZMCterm ptr[MAXPTR];
	int32_t anzptr;
	int64_t memory;
	
	PZMCtermMemoryManager();
	PZMCtermMemoryManager(const int64_t);
	virtual ~PZMCtermMemoryManager();
	
	PZMCterm* getMemory(const int32_t);
	
	void freePhysically(void);
};

struct ZMCpolynom {
	int32_t anzterms;
	int32_t allocatedterms;
	int32_t ALLOCATEINITIALLY;
	/* highest powers are only increased, but never */
	/* decreased, when eg. a term vanishes */
	int32_t Zdegree; /* highest power of Z */
	int32_t Mdegree,Cdegree;
	int8_t mighthaveremoveablezeroterms; // 0 => definitely none, 1 => may or may not
	
	PZMCterm *terms; // array of pointers
	ZMCtermMemoryManager *tmgr;
	PZMCtermMemoryManager *ptmgr;
	
	ZMCpolynom();
	ZMCpolynom(ZMCtermMemoryManager*,PZMCtermMemoryManager*,const int32_t);
	virtual ~ZMCpolynom();
	
	void clearTerms(void);
	void reConstructor(ZMCtermMemoryManager*,PZMCtermMemoryManager*);
	void initMemory(void);
	void setMemoryManager(ZMCtermMemoryManager*,PZMCtermMemoryManager*);
	void save(FILE*);
	void load(FILE*);
	void jumpOverLoad(FILE*);
	int8_t isZero(void);
	int8_t isPositiveOne(void);
	int8_t searchBy_TZMC(int32_t&,const int32_t,const int32_t,const int32_t);
	void checkDegree(void);
	void setToPositiveOne(void);
	void addTerm_FZMC(BigInt&,const int32_t,const int32_t,const int32_t);
	void addTerm_FZMC(const int64_t,const int32_t,const int32_t,const int32_t);
	void subTerm_FZMC(BigInt&,const int32_t,const int32_t,const int32_t);
	void addTerm_FZMC(ZMCterm&);
	void addTermLast_FZMC(ZMCterm&);
	void subTerm_FZMC(ZMCterm&);
	void subTerm_FZMC(const int64_t,const int32_t,const int32_t,const int32_t);
	void subPoly(ZMCpolynom&);
	void addPoly(ZMCpolynom&);
	void getStr(DynSlowString&);
	void ausgabe(FILE*);
	void removeZeroTerms(void);
	void copyFrom(ZMCpolynom&);
	void setToZero(void);
};

typedef ZMCpolynom *PZMCpolynom;


// square matrix with ZMCpolynomial pointers as entry
// those pointers are seen as having a constant content
// and its object is NOT to be changed
struct MatrixPolynom { 
	int32_t dim;
	
	ZMCpolynom *entryYX; // accessed via Y-index * dim + X-index
	ZMCtermMemoryManager *tmgr;
	PZMCtermMemoryManager *ptmgr;
	
	MatrixPolynom();
	virtual ~MatrixPolynom();
	
	void setDimension(ZMCtermMemoryManager*,PZMCtermMemoryManager*,const int32_t);
	void ausgabe(FILE*);
	void setConstant0polynom(void);
	int8_t entryZero_YX(const int32_t,const int32_t);
	int8_t entryPositiveOne_YX(const int32_t,const int32_t);
	void save(const char*);
	void load(const char*,const int32_t,const int32_t);
	void loadRowCol(const char*,const int32_t,const int32_t);
};

typedef MatrixPolynom *PMatrixPolynom;

// holds on open file for reading a polynomial
// from a matrix on hard disc, mainly used for
// sequential reading of one polynomial after
// the other, jumping over some
struct MatrixLoader {
	FILE *f;
	char fn[1024];
	int32_t dim;
	int32_t currentposx,currentposy; // the current position
	// at which the polynomial in the matrix y,x starts
	
	MatrixLoader();
	virtual ~MatrixLoader();
	
	void prepareForLoading(const char*);
	void close(void);
	int8_t loadAtYX(ZMCpolynom&,const int32_t,const int32_t);
};


// forward declarations

inline void termMul_TAB(ZMCterm&,ZMCterm&,ZMCterm&);
inline void polynomMul_TAB(ZMCpolynom&,ZMCpolynom&,ZMCpolynom&);
inline void polynomMul_TTermB(ZMCpolynom&,ZMCterm&,ZMCpolynom&);
inline void polynomDiv_rf_TRAB(ZMCpolynom&,ZMCpolynom&,ZMCpolynom&,ZMCpolynom&,ZMCpolynom&,ZMCpolynom&,ZMCpolynom&,ZMCpolynom&);

void polynomComposition_at_Z_TNA(ZMCpolynom&,const int32_t,ZMCpolynom&);
void polynomPow_TNA(ZMCpolynom&,const int32_t,ZMCpolynom&);
void polynomDer_at_Z_TA(ZMCpolynom&,ZMCpolynom&);

inline int32_t sum_int32t(const int32_t,const int32_t);
int8_t loadPolynomialFromMatrix_FTYX(const char*,ZMCpolynom&,const int32_t,const int32_t);
inline int8_t exponentVgl_AB(ZMCterm&,const int32_t,const int32_t,const int32_t);


// globals

int8_t MULTIBROT=2; 
ZMCpolynom basefkt; // memory manager will be set later

// memory manager
ZMCtermMemoryManager globalzmctermmgr;
PZMCtermMemoryManager globalpzmctermmgr;

BigInt big1;
char fnbase[1024];
ZMCpolynom fstrictP,fderm; // memory manager will be set later
int32_t currentperiod=1;


// routines

// general
inline int32_t sum_int32t(
	const int32_t a,
	const int32_t b
) {
	int64_t sum=(int64_t)a + (int64_t)b;
	if (
		(sum < (-INT32MAX)) ||
		(sum >   INT32MAX)
	) {
		LOGMSG("\nError. Overflow by int32 additionßn");
		exit(99);
	}
	
	return (int32_t)sum; // lower half
}

char* upper(char* s) {
	if (!s) return NULL;
	
	for(int32_t i=0;i<(int32_t)strlen(s);i++) {
		if ((s[i]>='a')&&(s[i]<='z')) s[i]=s[i]-'a'+'A';
	}

	return s;
}

// polynomial
void ZMCpolynom::save(FILE* f) {
	if (!f) {
		LOGMSG("\nError. Implementation. ZMCpolynom::save no file\n");
		exit(99);
	}
	
	// degrees are not stored as during reading
	// they are set
	
	if (fwrite(&anzterms,1,sizeof(anzterms),f) != sizeof(anzterms)) {
		LOGMSG("\nError. Wrting polynomial. Probably invalid matrix file. Deleting recommended.\n");
		exit(99);
	}
	
	for(int32_t i=0;i<anzterms;i++) {
		terms[i]->save(f);
	}
}

void ZMCpolynom::load(FILE* f) {
	if (!f) {
		LOGMSG("\nError. Implementation. ZMCpolynom::load no file\n");
		exit(99);
	}
	
	if ( (!tmgr) || (!ptmgr) ) {
		LOGMSG("\nError. P/ZMCpolynom::load mgr-pointer nil\n");
		exit(99);
	}
	
	// degrees are not stored as during reading
	// they are set
	
	int32_t anz;
	if (fread(&anz,1,sizeof(anz),f) != sizeof(anz)) {
		LOGMSG("\nError. Reading polynomial. Probably invalid matrix file. Deleting recommended.\n");
		exit(99);
	}
	
	clearTerms();
	
	if (anz == 0) {
		setToZero();
		return;
	}
	
	if (!terms) {
		// allocate exactly that amount
		terms=ptmgr->getMemory(anz);

		if (!terms) {
			LOGMSG("\nError. Memory. ZMCpolynom::load\n");
			exit(99);
		}
		
		// initialize with NULL-pointer !
		for(int32_t i=0;i<anz;i++) terms[i]=NULL;
		
		allocatedterms=anz;
	}
	
	ZMCterm one;
	for(int32_t i=0;i<anz;i++) {
		one.load(f);
		addTerm_FZMC(one);
	}
	
	// now degrees are also set
}

void ZMCpolynom::jumpOverLoad(FILE* f) {
	if (!f) {
		LOGMSG("\nError. Implementation. ZMCpolynom::jumpOverLoad no file\n");
		exit(99);
	}
	
	anzterms=0;
	
	int32_t anz;
	if (fread(&anz,1,sizeof(anz),f) != sizeof(anz)) {
		LOGMSG("\nError. Jump-Reading polynomial. Probably invalid matrix file. Deleting recommended.\n");
		exit(99);
	}

	clearTerms();
	
	ZMCterm one;
	for(int32_t i=0;i<anz;i++) {
		one.load(f);
		// do not add it
	}
	
}

void ZMCpolynom::checkDegree(void) {
	Zdegree=0;
	Mdegree=0;
	Cdegree=0;
	
	for(int32_t i=0;i<anzterms;i++) {
		if (terms[i]->Zexponent > Zdegree) Zdegree=terms[i]->Zexponent;
		if (terms[i]->Mexponent > Mdegree) Mdegree=terms[i]->Mexponent;
		if (terms[i]->Cexponent > Cdegree) Cdegree=terms[i]->Cexponent;
	} // i
	
}

void ZMCpolynom::setToZero(void) {
	clearTerms();
}

void ZMCpolynom::setToPositiveOne(void) {
	clearTerms();
	addTerm_FZMC(1,0,0,0);
}

void ZMCpolynom::copyFrom(ZMCpolynom& A) {
	// stores the terms of the source polynomial A
	// physically as a copy
	
	clearTerms();
	for(int32_t i=0;i<A.anzterms;i++) {
		addTerm_FZMC(
			A.terms[i]->factor,
			A.terms[i]->Zexponent,
			A.terms[i]->Mexponent,
			A.terms[i]->Cexponent
		);
	}

}

void ZMCpolynom::removeZeroTerms(void) {
	if (mighthaveremoveablezeroterms <= 0) return; // free
	
	// if one term with zero present => keep it
	if (anzterms <= 1) {
		mighthaveremoveablezeroterms=0;
		return;
	}
	
	int32_t s0=anzterms-1;
	while (anzterms > 0) {
		int32_t f0=-1;
		for(int32_t i=s0;i>=0;i--) {
			if (terms[i]->factor.vorz == 0) {
				f0=i;
				break;
			}
		} // i
		
		if (f0 < 0) break;
		// [f0] is zero
		// shift everyting from [f0+1..anzterms[ one to the left
		ZMCterm* p=terms[f0];
		for(int32_t k=f0;k<(anzterms-1);k++) {
			terms[k]=terms[k+1];
		}
		terms[anzterms-1]=p;
		anzterms--;
		s0=f0; // can't be zero again
		if (s0 >= anzterms) s0=anzterms-1;
	} // while
	
	mighthaveremoveablezeroterms=0;
	
}

void ZMCpolynom::ausgabe(FILE* f) {
	DynSlowString one;
	one.setEmpty();
	
	getStr(one);
	fprintf(f,"%s",one.text);
}
int8_t ZMCpolynom::isZero(void) {
	if (anzterms == 0) return 1;
	
	// can be terms with a zero factor as polynomials
	// are not condensed after addition/subtraction
	for(int32_t i=0;i<anzterms;i++) {
		if (terms[i]->factor.vorz != 0) return 0;
	}
	
	return 1;
}

int8_t ZMCpolynom::isPositiveOne(void) {
	if (anzterms == 0) return 0;
	
	int32_t found=0;
	
	for(int32_t i=0;i<anzterms;i++) {
		if (
			(terms[i]->Zexponent != 0) ||
			(terms[i]->Mexponent != 0) ||
			(terms[i]->Cexponent != 0)
		) return 0;
		
		if (terms[i]->factor.vorz > 0) {
			if (bigintVgl_AB(terms[i]->factor,big1) == 0) {
				found++; // break at 2, so no overflow
				if (found > 1) return 0;
			}
		}
		
	} // i
	
	if (found == 1) return 1;
	
	return 0;
}

void ZMCpolynom::getStr(DynSlowString& erg) {
	erg.setEmpty();
	
	DynSlowString one(1024);
	for(int32_t i=0;i<anzterms;i++) {
		one.setEmpty();
				
		if (terms[i]->factor.vorz >= 0) {
			erg.add("+");
		}
		
		terms[i]->getStr(one);
		erg.add(one);
	}
	
}

void ZMCpolynom::addTerm_FZMC(
	const int64_t av,
	const int32_t aZexp, /* exponent of Z-variable */
	const int32_t aMexp,
	const int32_t aCexp
) {
	BigInt w;
	w.set_int64(av);
	
	addTerm_FZMC(w,aZexp,aMexp,aCexp);
}

void ZMCpolynom::subTerm_FZMC(ZMCterm& t) {
	subTerm_FZMC(t.factor,t.Zexponent,t.Mexponent,t.Cexponent);
}

void ZMCpolynom::addTerm_FZMC(ZMCterm& t) {
	addTerm_FZMC(t.factor,t.Zexponent,t.Mexponent,t.Cexponent);
}

void ZMCpolynom::subPoly(ZMCpolynom& b) {
	for(int32_t i=0;i<b.anzterms;i++) {
		subTerm_FZMC(*b.terms[i]);
	}

	removeZeroTerms();

}

void ZMCpolynom::addPoly(ZMCpolynom& b) {
	for(int32_t i=0;i<b.anzterms;i++) {
		addTerm_FZMC(*b.terms[i]);
	}
	
	removeZeroTerms();
}

void ZMCpolynom::subTerm_FZMC(
	const int64_t av,
	const int32_t aZexp, /* exponent of Z-variable */
	const int32_t aMexp,
	const int32_t aCexp
) {
	BigInt w;
	w.set_int64(av);
	
	subTerm_FZMC(w,aZexp,aMexp,aCexp);
}

int8_t ZMCpolynom::searchBy_TZMC(
	int32_t &idx,
	const int32_t az,
	const int32_t am,
	const int32_t ac
) {
	idx=-1;
	if (anzterms <= 0) {
		// add at end
		return 0;
	}
	
	// if the rightest term is still not larger
	// add at end
	int8_t vgl=exponentVgl_AB(*terms[anzterms-1],az,am,ac);
	if (vgl == 0) {
		// found it
		idx=anzterms-1;
		return 1;
	}
	if (vgl < 0) {
		// new element azam,ac is even larger
		// (In the exponent ordering sense)
		idx=-1; // add at end
		return 0;
	}
	
	int32_t right=anzterms-1;
	
	// now it holds through the rest of the routine
	// (az,ammac) < [right] 
	
	vgl=exponentVgl_AB(*terms[0],az,am,ac);
	if (vgl == 0) {
		// found it
		idx=0;
		return 1;
	}
	if (vgl > 0) {
		// even smallest element in list is larger than
		// the new one: insert at first position
		idx=0;
		return 0;
	}
	
	int32_t left=0;
	// it holds throughout: [left] < (az,am,ac)
	
	// binary search

	while (1) {
		int32_t len=sum_int32t( (right-left), 1 ); // no overflow possible here
		if (len <= 4) {
			// left, right are strict smaller/larger
			for(int32_t i=left;i<=right;i++) {
				int8_t vgl=exponentVgl_AB(*terms[i],az,am,ac);
				if (vgl == 0) {
					idx=i;
					return 1;
				}
				if (vgl > 0) {
					idx=i;
					return 0;
				}
			} // i
			
			// shall not occur
			LOGMSG("\nImplementation error. binary search\n");
			exit(99);
		} // small region where az,amac could be
		
		int32_t middle=sum_int32t(left,right) >> 1;
		
		vgl=exponentVgl_AB(*terms[middle],az,am,ac);
		if (vgl == 0) {
			// found it
			idx=middle;
			return 1;
		}
		
		if (vgl > 0) {
			right=middle; // loop invariant holds still
		} else if (vgl < 0) {
			left=middle; // loop invariant holds
		}
		
	} // while
	
	// shall not occar
	LOGMSG("\nImplementation error. Outside loop search\n");
	exit(99);
	
}

inline int8_t exponentVgl_AB(
	ZMCterm& A,
	const int32_t bz,
	const int32_t bm,
	const int32_t bc
) {
	if (A.Zexponent > bz) return +1;
	if (A.Zexponent < bz) return -1;
	
	if (A.Mexponent > bm) return +1;
	if (A.Mexponent < bm) return -1;

	if (A.Cexponent > bc) return +1;
	if (A.Cexponent < bc) return -1;
	
	return 0;
}

void ZMCpolynom::addTerm_FZMC(
	BigInt& afactor,
	const int32_t aZexp, /* exponent of Z-variable */
	const int32_t aMexp,
	const int32_t aCexp
) {
	
	if (!terms) {
		initMemory();
		anzterms=0;
	}
	
	int32_t eidx;
	int8_t searched=searchBy_TZMC(eidx,aZexp,aMexp,aCexp);
	
	if (searched > 0) {
		//printf("found at %i\n",eidx);
		if (eidx >= 0) {
			terms[eidx]->factor.addTo(afactor);
			if (terms[eidx]->factor.vorz == 0) mighthaveremoveablezeroterms=1;
			return; // done
		} else {
			LOGMSG("\nError. eids negative.\n");
			exit(99);
		}
	}
	
	// new Z-term
	if (anzterms > (allocatedterms-8)) {
		// allocate new memory and copy all pointers
		PZMCterm *old=terms;
		
		if (allocatedterms >= MAXTERMSPERPOLYNOM) {
			LOGMSG("\nError. Memory. Too many terms requested.\n");
			exit(99);
		}
		int32_t oldalloc=allocatedterms;
		allocatedterms = sum_int32t(allocatedterms,allocatedterms);
		if ( (!tmgr) || (!ptmgr) ) {
			LOGMSG("\nError. Implementation. ZMCpolynom::addTerm.mgr pointer NIL\n");
			exit(99);
		}
		terms=ptmgr->getMemory(allocatedterms);
		
		if (!terms) {
			LOGMSG("\nError. Memory. ZMCpolynom::addTerm_Z_MCpoly\n");
			exit(99);
		}

		// copy used terms
		// oldalloc instead of anzterms, so NIL-pointers
		// are copied from the initial array
		for(int32_t i=0;i<oldalloc;i++) {
			terms[i]=old[i];
		}
		
		// do not start setting pointers to NULL from
		// anzterms as ZMCpolynom could be in reuse, already
		// having still valid pointers to memory used
		// in a former usage of the polynomial
		// so only the new allocated pointer region
		// is set to NULL

		for(int32_t i=oldalloc;i<allocatedterms;i++) {
			terms[i]=NULL;
		}
	
	} // expand array of zmc pointers
	
	if (anzterms > (MAXTERMSPERPOLYNOM-16)) {
		LOGMSG("\nError. Memory. Too many terms in a polynomial.\n");
		exit(99);
	}
	
	// terms[anzterms]: is a valid index
	int32_t insertat=-1;
	if (searched <= 0) {
		if (eidx >= 0) {
			insertat=eidx;
		}
	}
	
	if (insertat < 0) insertat=anzterms;
	else {
		// shift pointers {insertat..anzterms[ one
		// to the right
		ZMCterm* p=terms[anzterms]; // new position
		for(int32_t k=anzterms;k>insertat;k--) {
			terms[k]=terms[k-1];
		} // k
		terms[insertat]=p;
	}
	
	anzterms=sum_int32t(anzterms,1);
	
	if (!terms[insertat]) {
		// get an actual object from the 2nd memory manager
		terms[insertat]=tmgr->getMemory(1);
		if (!terms[insertat]) {
			LOGMSG("\nError. Memory. polynom::add. tmgr cannot hand out more terms\n");
			exit(99);
		} 
	}
	
	// valid pointer
	terms[insertat]->Zexponent=aZexp;
	terms[insertat]->Mexponent=aMexp;
	terms[insertat]->Cexponent=aCexp;
	terms[insertat]->factor.copyFrom(afactor);
	
	if (aZexp > Zdegree) Zdegree=aZexp;
	if (aMexp > Mdegree) Mdegree=aMexp;
	if (aCexp > Cdegree) Cdegree=aCexp;
	
	#ifdef _INVOKECLAIMVERIFICATIONS
	for(int32_t i=1;i<anzterms;i++) {
		if (exponentVgl_AB(*terms[i-1],terms[i]->Zexponent,terms[i]->Mexponent,terms[i]->Cexponent) >= 0) {
			printf("\nError order\n");
			for(int32_t k=0;k<anzterms;k++) {
				printf("#%i: %i,%i,%i\n",
					k,
					terms[k]->Zexponent,
					terms[k]->Mexponent,
					terms[k]->Cexponent);
			}
			exit(99);
		}
	}
	#endif
	
}

void ZMCpolynom::addTermLast_FZMC(
	ZMCterm& A
) {
	// adds the term to the current term list
	// degree is adju
	
	if (!terms) {
		initMemory();
		anzterms=0;
	}
	
	// new Z-term
	if (anzterms > (allocatedterms-8)) {
		// allocate new memory and copy all pointers
		PZMCterm *old=terms;
		
		if (allocatedterms >= MAXTERMSPERPOLYNOM) {
			LOGMSG("\nError. Memory. Too many terms requested.\n");
			exit(99);
		}
		int32_t oldalloc=allocatedterms;
		allocatedterms = sum_int32t(allocatedterms,allocatedterms);
		if ( (!tmgr) || (!ptmgr) ) {
			LOGMSG("\nError. Implementation. ZMCpolynom::addTerm.mgr pointer NIL\n");
			exit(99);
		}
		terms=ptmgr->getMemory(allocatedterms);
		
		if (!terms) {
			LOGMSG("\nError. Memory. ZMCpolynom::addTerm_Z_MCpoly\n");
			exit(99);
		}

		// copy used terms
		// oldalloc instead of anzterms, so NIL-pointers
		// are copied from the initial array
		for(int32_t i=0;i<oldalloc;i++) {
			terms[i]=old[i];
		}
		
		// do not start setting pointers to NULL from
		// anzterms as ZMCpolynom could be in reuse, already
		// having still valid pointers to memory used
		// in a former usage of the polynomial
		// so only the new allocated pointer region
		// is set to NULL

		for(int32_t i=oldalloc;i<allocatedterms;i++) {
			terms[i]=NULL;
		}
	
	} // expand array of zmc pointers
	
	if (anzterms > (MAXTERMSPERPOLYNOM-16)) {
		LOGMSG("\nError. Memory. Too many terms in a polynomial.\n");
		exit(99);
	}
	
	// terms[anzterms]: is a valid index
	int32_t insertat=anzterms;

	anzterms=sum_int32t(anzterms,1);
	
	if (!terms[insertat]) {
		// get an actual object from the 2nd memory manager
		terms[insertat]=tmgr->getMemory(1);
		if (!terms[insertat]) {
			LOGMSG("\nError. Memory. polynom::add. tmgr cannot hand out more terms\n");
			exit(99);
		} 
	}
	
	// valid pointer
	terms[insertat]->copyFrom(A);
	
	if (A.Zexponent > Zdegree) Zdegree=A.Zexponent ;
	if (A.Mexponent > Mdegree) Mdegree=A.Mexponent ;
	if (A.Cexponent > Cdegree) Cdegree=A.Cexponent ;
	
	#ifdef _INVOKECLAIMVERIFICATIONS
	for(int32_t i=1;i<anzterms;i++) {
		if (terms[i-1]->Zexponent > terms[i]->Zexponent) {
			printf("\nError order\n");
			for(int32_t k=0;k<anzterms;k++) {
				printf("#%i: %i,%i,%i\n",
					k,
					terms[k]->Zexponent,
					terms[k]->Mexponent,
					terms[k]->Cexponent);
			}
			exit(99);
		}
	}
	#endif
	
}

void ZMCpolynom::subTerm_FZMC(
	BigInt& afactor,
	const int32_t aZexp, /* exponent of Z-variable */
	const int32_t aMexp,
	const int32_t aCexp
) {
	BigInt minusfactor=afactor;
	minusfactor.vorz *= -1;
	
	addTerm_FZMC(minusfactor,aZexp,aMexp,aCexp);
}

void ZMCpolynom::initMemory(void) {
	if ( (!tmgr) || (!ptmgr) ) {
		LOGMSG("\nError. Implementation. ZMCpolynom::initMemory.mgr pointer NIL\n");
		exit(99);
	}
	
	if (ALLOCATEINITIALLY < 16) ALLOCATEINITIALLY=16;
	
	terms=ptmgr->getMemory(ALLOCATEINITIALLY);

	if (!terms) {
		LOGMSG("\nError. Memory. ZMCpolynom::initMemory\n");
		exit(99);
	}

	for(int32_t i=0;i<ALLOCATEINITIALLY;i++) {
		terms[i]=NULL;
	}

	allocatedterms=ALLOCATEINITIALLY;
}

ZMCpolynom::ZMCpolynom() {
	anzterms=0;
	Zdegree=Mdegree=Cdegree=0;
	allocatedterms=0;
	terms=NULL;
	tmgr=NULL;
	ptmgr=NULL;
	ALLOCATEINITIALLY=INIT_ZMCTERMS;
	mighthaveremoveablezeroterms=0;
}

ZMCpolynom::ZMCpolynom(
	ZMCtermMemoryManager* t,
	PZMCtermMemoryManager* p,
	const int32_t am
) {
	anzterms=0;
	Zdegree=Mdegree=Cdegree=0;
	allocatedterms=0;
	terms=NULL;
	tmgr=t;
	ptmgr=p;
	mighthaveremoveablezeroterms=0;
	
	if (am < 16) ALLOCATEINITIALLY=INIT_ZMCTERMS;
	else ALLOCATEINITIALLY=am;
}

void ZMCpolynom::reConstructor(
	ZMCtermMemoryManager* t,
	PZMCtermMemoryManager* p
) {
	// do not change ALLOCATEINITIALLY, value can be reused
	anzterms=0;
	Zdegree=Mdegree=Cdegree=0;
	allocatedterms=0;
	terms=NULL;
	tmgr=t;
	ptmgr=p;
}

ZMCpolynom::~ZMCpolynom() {
	/* the objects themselves will be freed when destroying */
	/* the memory manager */
}

void ZMCpolynom::clearTerms(void) {
	anzterms=0;
	Zdegree=Mdegree=Cdegree=0;
	mighthaveremoveablezeroterms=0;
}

void ZMCpolynom::setMemoryManager(
	ZMCtermMemoryManager* t,
	PZMCtermMemoryManager* p
) {
	tmgr=t;
	ptmgr=p;
}

// memory manager
ZMCtermMemoryManager::ZMCtermMemoryManager() {
	current=NULL;
	memory=0;
	allocatedIdx=0;
	freeFromIdx=-1;
	double d=CHUNKSIZE; 
	d /= sizeof(ZMCterm);
	allocatePerBlockIdx=(int32_t)floor(d);
	anzptr=0;
}

ZMCtermMemoryManager::ZMCtermMemoryManager(const int64_t membytes) {
	current=NULL;
	memory=0;
	allocatedIdx=0;
	freeFromIdx=-1;
	double d=membytes; 
	d /= sizeof(ZMCterm);
	allocatePerBlockIdx=(int32_t)floor(d);
	anzptr=0;
}


ZMCterm* ZMCtermMemoryManager::getMemory(const int32_t aanz) {
	if (
		(!current) ||
		((sum_int32t(freeFromIdx,aanz)) >= allocatedIdx)
	) {
		if (anzptr >= MAXPTR) {
			LOGMSG("Error. ZMCtermMemoryManager/maxptr.\n");
			exit(99);
		}
		ptr[anzptr]=current=new ZMCterm[allocatePerBlockIdx];
		memory += (sizeof(ZMCterm)*allocatePerBlockIdx);
		anzptr++; // no overflow as checked above
		if (!current) {
			LOGMSG("Error/2. ZMCtermMemoryManager.\n");
			exit(99);
		}
		freeFromIdx=0;
		allocatedIdx=allocatePerBlockIdx;
	}
	
	ZMCterm *p=&current[freeFromIdx];
	freeFromIdx = sum_int32t(freeFromIdx,aanz);
	
	if (freeFromIdx >= (allocatePerBlockIdx-8)) {
		LOGMSG("\nError. ZMCtermMemorymanager. Requested size larger than page.\n");
		exit(99);
	}
	
	return p;
}

void ZMCtermMemoryManager::freePhysically(void) {
	// put into state of constructor just called
	for(int32_t i=0;i<anzptr;i++) {
		if (ptr[i]) delete[] ptr[i];
	}
	memory=0;
	current=NULL;
	allocatedIdx=0;
	freeFromIdx=-1;
	anzptr=0;
}

ZMCtermMemoryManager::~ZMCtermMemoryManager() {
	freePhysically();
}

// memory manager
PZMCtermMemoryManager::PZMCtermMemoryManager() {
	current=NULL;
	memory=0;
	allocatedIdx=0;
	freeFromIdx=-1;
	double d=CHUNKSIZE; 
	d /= sizeof(PZMCterm);
	allocatePerBlockIdx=(int32_t)floor(d);
	anzptr=0;
}

PZMCtermMemoryManager::PZMCtermMemoryManager(const int64_t memsz) {
	current=NULL;
	memory=0;
	allocatedIdx=0;
	freeFromIdx=-1;
	double d=memsz; 
	d /= sizeof(PZMCterm);
	allocatePerBlockIdx=(int32_t)floor(d);
	anzptr=0;
}

PZMCterm* PZMCtermMemoryManager::getMemory(const int32_t aanz) {
	if (
		(!current) ||
		(sum_int32t(freeFromIdx,aanz) >= allocatedIdx)
	) {
		if (anzptr >= MAXPTR) {
			LOGMSG("Error. PZMCtermMemoryManager/maxptr.\n");
			exit(99);
		}
		ptr[anzptr]=current=new PZMCterm[allocatePerBlockIdx];
		memory += (sizeof(PZMCterm)*allocatePerBlockIdx);
		anzptr++; // no overflow as checked above
		if (!current) {
			LOGMSG("Error/2. PZMCtermMemoryManager.\n");
			exit(99);
		}
		freeFromIdx=0;
		allocatedIdx=allocatePerBlockIdx;
	}
	
	PZMCterm *p=&current[freeFromIdx];
	freeFromIdx = sum_int32t(freeFromIdx,aanz);
	
	if (freeFromIdx >= (allocatePerBlockIdx-8)) {
		LOGMSG("\nError. PZMCtermMemorymanager. Requested size larger than page.\n");
		exit(99);
	}

	return p;
}

void PZMCtermMemoryManager::freePhysically(void) {
	// put into state of constructor just called
	for(int32_t i=0;i<anzptr;i++) {
		if (ptr[i]) delete[] ptr[i];
	}
	memory=0;
	
	current=NULL;
	allocatedIdx=0;
	freeFromIdx=-1;
	anzptr=0;
}

PZMCtermMemoryManager::~PZMCtermMemoryManager() {
	freePhysically();
}

void getStrictPeriod_TP(
	ZMCpolynom& astrictp,
	const int32_t period
) {
	/*
	
		computes recursively all sub-strict periods
		even if they are necessarsy multiple times
		
		no bottleneck as almost all computational time 
		is spent in	computing the determinant
	
	*/
	
	// computss basefkt[comp P-fold]-z) 
	// / product strictP' with P' truely dividing P (1 included, but not P)
	
	ZMCpolynom fcompP(astrictp.tmgr,astrictp.ptmgr,-1),
		product(astrictp.tmgr,astrictp.ptmgr,-1),
		tmp(astrictp.tmgr,astrictp.ptmgr,-1),
		substrict(astrictp.tmgr,astrictp.ptmgr,-1);

	polynomComposition_at_Z_TNA(fcompP,period,basefkt);
	
	product.clearTerms();
	product.addTerm_FZMC(1,0,0,0); // positive one
	
	for(int32_t dividing=1;dividing<period;dividing++) {
		if ( (period % dividing) != 0) continue;
		
		//printf("\n  dividing %i\n",dividing);
		getStrictPeriod_TP(substrict,dividing);
		polynomMul_TAB(tmp,substrict,product);
		product.copyFrom(tmp);
		
	} // dividing
	
	// now (fcompP-z) / product will be divisible
	ZMCpolynom rem(astrictp.tmgr,astrictp.ptmgr,-1);
	ZMCpolynom hlp1(astrictp.tmgr,astrictp.ptmgr,-1),
		hlp2(astrictp.tmgr,astrictp.ptmgr,-1),
		hlp3(astrictp.tmgr,astrictp.ptmgr,-1),
		hlp4(astrictp.tmgr,astrictp.ptmgr,-1);
	fcompP.addTerm_FZMC(-1,1,0,0); // -z
	polynomDiv_rf_TRAB(astrictp,rem,fcompP,product,hlp1,hlp2,hlp3,hlp4);
	
	if (rem.isZero() <= 0) {
		LOGMSG("\nError. Construction period formula failed. Division not remainder-free.\n");
		exit(99);
	}
	
	#ifdef _INVOKECLAIMVERIFICATIONS
	// Gegenprobe: fstrictP*product-fcompP = 0
	ZMCpolynom p1(astrictp.tmgr,astrictp.ptmgr,-1);
	polynomMul_TAB(p1,astrictp,product);
	p1.subPoly(fcompP);
	if (p1.isZero() <= 0) {
		LOGMSG("\nError. Implementation. Gegenprobe fstrictP incorrect.\n");
		exit(99);
	}
	#endif
		
}

void setPeriod(const int32_t aper) {
	// compute composition f[z,aper]
	
	ZMCpolynom fcomp(&globalzmctermmgr,&globalpzmctermmgr,-1);
	polynomComposition_at_Z_TNA(fcomp,aper,basefkt);
	
	LOGMSG3("|-> %i-composition f_o%i(z)=",currentperiod,currentperiod); 
	fcomp.ausgabe(stdout);
	fcomp.ausgabe(flog);
	
	// derivative
	fderm.setMemoryManager(&globalzmctermmgr,&globalpzmctermmgr);
	polynomDer_at_Z_TA(fderm,fcomp);
	fderm.addTerm_FZMC(-1,0,1,0); // -m
	
	// computes recursively the necessary strictP subperiodic
	// polynomials and their compositions
	fstrictP.setMemoryManager(&globalzmctermmgr,&globalpzmctermmgr);
	getStrictPeriod_TP(fstrictP,aper);

}

// ZMCterm
void ZMCterm::load(FILE* f) {
	if (!f) {
		LOGMSG("\nError. Implementation. ZMCterm::load no file\n");
		exit(99);
	}
	
	factor.load(f);
	int32_t r=0;
	r += fread(&Zexponent,1,sizeof(Zexponent),f);
	r += fread(&Mexponent,1,sizeof(Mexponent),f);
	r += fread(&Cexponent,1,sizeof(Cexponent),f);
	
	if (r != (sizeof(Zexponent)+sizeof(Mexponent)+sizeof(Cexponent))) {
		LOGMSG("\nError. Reading polynomial term. Probably invalid matrix file. Deleting recommended.\n");
		exit(99);
	}
}

void ZMCterm::save(FILE* f) {
	if (!f) {
		LOGMSG("\nError. Implementation. ZMCterm::save no file\n");
		exit(99);
	}
	
	factor.save(f);
	int32_t r=0;
	r += fwrite(&Zexponent,1,sizeof(Zexponent),f);
	r += fwrite(&Mexponent,1,sizeof(Mexponent),f);
	r += fwrite(&Cexponent,1,sizeof(Cexponent),f);
	if (r != (sizeof(Zexponent)+sizeof(Mexponent)+sizeof(Cexponent)) ) {
		LOGMSG("\nError. Writing polynomial term. Probably invalid matrix file. Deleting recommended.\n");
		exit(99);
	}
}

int8_t ZMCterm::isPositiveOne(void) {
	if (
		(Zexponent != 0) ||
		(Mexponent != 0) ||
		(Cexponent != 0)
	) return 0;
	
	if (bigintVgl_AB(big1,factor) == 0) return 1;
	
	return 0;
}

void ZMCterm::copyFrom(ZMCterm& A) {
	factor.copyFrom(A.factor);
	Zexponent=A.Zexponent;
	Mexponent=A.Mexponent;
	Cexponent=A.Cexponent;
}

void ZMCterm::ausgabe(FILE* f) {
	DynSlowString one;
	one.setEmpty();
	
	getStr(one);
	fprintf(f,"%s",one.text);
}

void ZMCterm::setTermToZero(void) {
	factor.setToZero();
	Zexponent=Mexponent=Cexponent=0;
}

void ZMCterm::getStr(DynSlowString& erg) {
	erg.setEmpty();
	
	if (factor.vorz == 0) {
		erg.add("0");
		return;
	}
	
	DynSlowString one;
	factor.getStr(one); // including vorz
	erg.add(one);
	
	char tmp[1024];
	
	if (Mexponent > 0) {
		erg.add("*m");
		if (Mexponent > 1) {
			erg.add("^");
			sprintf(tmp,"%i",Mexponent);
			erg.add(tmp);
		}
	}

	if (Cexponent > 0) {
		erg.add("*c");
		if (Cexponent > 1) {
			erg.add("^");
			sprintf(tmp,"%i",Cexponent);
			erg.add(tmp);
		}
	}
	
	if (Zexponent > 0) {
		erg.add("*z");
		if (Zexponent > 1) {
			erg.add("^");
			sprintf(tmp,"%i",Zexponent);
			erg.add(tmp);
		}
	}
	
}

// DynSlowString
void DynSlowString::setEmpty(void) {
	if (text) text[0]=0;
}

DynSlowString::DynSlowString(const int32_t amem) {
	memory=amem;
	text=new char[memory];
	if (!text) {
		LOGMSG("\nError. Memory. DynSlowString.\n");
		exit(99);
	}
}

DynSlowString::DynSlowString() {
	memory=0;
	text=NULL;
}

DynSlowString::~DynSlowString() {
	if (text) delete[] text;
}

void DynSlowString::add(const char* atext) {
	if (!atext) return;
	
	int32_t ttlen=strlen(atext);
	if (ttlen <= 0) return;
	
	int32_t currentlen;
	if (text) currentlen=strlen(text); else currentlen=0;
	
	if ( 
		(sum_int32t(currentlen,ttlen) > (memory-8) ) ||
		(!text)
	) {
		// new memory necessary
		int32_t mem0=sum_int32t(currentlen,ttlen);
		memory = sum_int32t(mem0,mem0);
		char* oldtext=text;
		text=new char[memory];
		if (!text) {
			LOGMSG("\nError. DynSlowString::add.\n");
			exit(99);
		}
		if (oldtext) {
			strcpy(text,oldtext);
			delete[] oldtext;
		} else {
			text[0]=0;
		}
	} // allocating
	
	// enough memory allocated
	sprintf(&text[currentlen],"%s",atext);
}

void DynSlowString::add(DynSlowString& as) {
	if (as.text) add(as.text);
}

void resultantZ_tgtAB(
	ZMCpolynom& fres,
	ZMCpolynom& f,
	ZMCpolynom& g
) {
	fres.clearTerms();
}

ZMCpolynom* getZcoeff(ZMCpolynom& fkt) {
	// creates an array of polynoms in M,C that are
	// the coefficients of fkt in Z
	
	ZMCpolynom* erg=new ZMCpolynom[sum_int32t(1,fkt.Zdegree)];
	if (!erg) {
		LOGMSG("\nError. Memory. getZcoeff\n");
		exit(99);
	}
	
	if ( (!fkt.tmgr) || (!fkt.ptmgr) ) {
		LOGMSG("\nError. Implementation. No memory managaer for function to get Z coefficients for, provided.\n");
		exit(99);
	}
	
	for(int32_t degree=0;degree<=fkt.Zdegree;degree++) {
		erg[degree].setMemoryManager(fkt.tmgr,fkt.ptmgr);
		erg[degree].clearTerms();
		
		int8_t sth=0;
		for(int32_t i=0;i<fkt.anzterms;i++) {
			if (fkt.terms[i]->Zexponent != degree) continue;
			
			sth=1;
			erg[degree].addTerm_FZMC(
				fkt.terms[i]->factor,
				0, /* no z-exponent */
				fkt.terms[i]->Mexponent,
				fkt.terms[i]->Cexponent
			);
			
		} // i
		
		if (sth <= 0) {
			erg[degree].addTerm_FZMC(0,0,0,0); // constant-0
		}
		
	} // degree
	
	return erg;
}

ZMCpolynom* getMcoeff(ZMCpolynom& fkt) {
	// creates an array of polynoms in Z,C that are
	// the coefficients of fkt in M
	
	ZMCpolynom* erg=new ZMCpolynom[sum_int32t(1,fkt.Mdegree)];
	if (!erg) {
		LOGMSG("\nError. Memory. getMcoeff\n");
		exit(99);
	}
	
	int32_t ende=sum_int32t(fkt.Mdegree,1);
	for(int32_t i=0;i<ende;i++) {
		erg[i].setMemoryManager(fkt.tmgr,fkt.ptmgr);
	}
	
	for(int32_t degree=0;degree<=fkt.Mdegree;degree++) {
		erg[degree].clearTerms();
		
		int8_t sth=0;
		for(int32_t i=0;i<fkt.anzterms;i++) {
			if (fkt.terms[i]->Mexponent != degree) continue;
			
			sth=1;
			erg[degree].addTerm_FZMC(
				fkt.terms[i]->factor,
				fkt.terms[i]->Zexponent,
				0, /* M */
				fkt.terms[i]->Cexponent
			);
			
		} // i
		
		if (sth <= 0) {
			erg[degree].addTerm_FZMC(0,0,0,0); // constant-0
		}
		
	} // degree
	
	return erg;
}

// MatrixLoader
MatrixLoader::MatrixLoader() {
	f=NULL;
	fn[0]=0;
	dim=0;
	currentposx=currentposy=0;
}

MatrixLoader::~MatrixLoader() {
	if (f) fclose(f);
}

void MatrixLoader::close(void) {
	fclose(f);
}
	
void MatrixLoader::prepareForLoading(const char* afn) {
	strcpy(fn,afn);
	f=fopen(fn,"rb");
	if (!f) {
		LOGMSG2("\nError. Cannot prepare matrix |%s| for loading\n",fn);
		exit(99);
	}
	if (fread(&dim,1,sizeof(dim),f) != sizeof(dim)) {
		LOGMSG2("\nError/2. Cannot prepare matrix |%s| for loading\n",fn);
		exit(99);
	}
	
	// now file pointer is at position 0,0
	currentposx=currentposy=0;
}

int8_t MatrixLoader::loadAtYX(
	ZMCpolynom& res,
	const int32_t ay,
	const int32_t ax
) {
	#define SETTOSTART \
	{\
		fseek(f,0,SEEK_SET);\
		int32_t d;\
		if (fread(&d,1,sizeof(d),f) != sizeof(d)) {\
			LOGMSG2("\nError/3. Cannot prepare matrix |%s| for loading\n",fn);\
			exit(99);\
		}\
		currentposx=currentposy=0;\
	}
	
	// is the position to load EARLIER then current?
	// if so, reset the file
	if ( 
		( (ay == currentposy) && (ax < currentposx) ) ||
		( ay < currentposy )
	) {
		SETTOSTART
	} // to load is earlier
	
	// now current points to either the polynomial to load
	// or an earler one, so the one to load comes "after"
	int8_t whatid;
	
	while (1) {
		if (fread(&whatid,1,sizeof(whatid),f) != sizeof(whatid) ) {
			LOGMSG("\nError/4: MatrixLoader.\n");
			exit(99);
		}
		if ( (currentposx == ax) && (currentposy == ay) ) {
			if (whatid == NULLPOLYNOMID) {
				res.clearTerms();
			} else {
				res.load(f);
			}
			currentposx++;
			if (currentposx >= dim) {
				currentposy++;
				currentposx=0;
				if (currentposy >= dim) {
					// set back, but polynomial found and read
					SETTOSTART
				}
			}

			return 1;
		} // load
		else {
			if (whatid != NULLPOLYNOMID) {
				res.jumpOverLoad(f);
			}
			currentposx++;
			if (currentposx >= dim) {
				currentposx=0;
				currentposy++;
				if (currentposy >= dim) {
					SETTOSTART
					return 0; // not found
				}
			}
		} // jump
	} // while
	
	return 0;
}

// MatrixPolynom
void MatrixPolynom::load(
	const char* afn,
	const int32_t loady,
	const int32_t loadx
) {
	/*
	
		if loady>=0 & loadx >= 0 => only that element is to
		be actually loaded, the rest of the materix is set to
		constant0
	
	*/
	
	FILE *f=fopen(afn,"rb");
	if (!f) {
		LOGMSG2("\nError. Not able to open file for matrix loading |%s|.\n",afn);
		exit(99);
	}
	
	int32_t d;
	if (fread(&d,1,sizeof(dim),f) != sizeof(dim)) {
		LOGMSG("\nError. Reading matrix. Probably invalid file. Deleting recommended.\n");
		exit(99);
	}
	
	if ( (!tmgr) || (!ptmgr) ) {
		LOGMSG("\nError. Implementation. MatrixPolynom::load.memory managaer pointer is nil.ßn");
		exit(99);
	}
	
	// sets mgr to the same value, but this way calling
	// setDimension one is forced to provide a valid manager
	if (!entryYX) {
		setDimension(tmgr,ptmgr,d); // sets dim and allocates memory
	} else {
		if (d != dim) {
			delete[] entryYX;
			setDimension(tmgr,ptmgr,d);
		} else {
			// reConstructor polynomial entries
			for(int32_t y=0;y<dim;y++) {
				for(int32_t x=0;x<dim;x++) {
					// index correct as dim <= 2^14
					entryYX[y*dim+x].reConstructor(tmgr,ptmgr);
				}
			}
		}
	}
	
	for(int32_t y=0;y<dim;y++) {
		for(int32_t x=0;x<dim;x++) {
			int8_t whatid;
			if (fread(&whatid,1,sizeof(whatid),f) != sizeof(whatid)) {
				LOGMSG("\nError. Reading matrix polynomial. Probably invalid matrix file. Deleting recommended.\n");
				exit(99);
			}
			if (whatid == NULLPOLYNOMID) {
				// dim <= 2^14
				entryYX[y*dim+x].clearTerms(); // null-Polynom
			} else {
				if ( (loadx<0) || (loady<0) ) {
					entryYX[y*dim+x].load(f);
				} else {
					if ( (y == loady) && (x == loadx) ) {
						entryYX[y*dim+x].load(f);
					} else {
						entryYX[y*dim+x].jumpOverLoad(f);
					}
				}
			}
		} // x
	} // y
	
	fclose(f);
}

void MatrixPolynom::loadRowCol(
	const char* afn,
	const int32_t row,
	const int32_t col
) {
	/*
		only loads row "row" and column "col" of the
		matrix in file afn. all other entres are set to ZERO
	
	*/
	
	FILE *f=fopen(afn,"rb");
	if (!f) {
		LOGMSG2("\nError. Not able to open file for matrix loading |%s|.\n",afn);
		exit(99);
	}
	
	int32_t d;
	if (fread(&d,1,sizeof(dim),f) != sizeof(dim)) {
		LOGMSG("\nError. Reading matrix. Probably invalid file. Deleting recommended.\n");
		exit(99);
	}
	
	if ( (!tmgr) || (!ptmgr) ) {
		LOGMSG("\nError. Implementation. MatrixPolynom::load.memory managaer pointer is nil.ßn");
		exit(99);
	}
	
	// sets mgr to the same value, but this way calling
	// setDimension one is forced to provide a valid manager
	if (!entryYX) {
		setDimension(tmgr,ptmgr,d); // sets dim and allocates memory
	} else {
		if (d != dim) {
			delete[] entryYX;
			setDimension(tmgr,ptmgr,d);
		} else {
			// reConstructor polynomial entries
			for(int32_t y=0;y<dim;y++) {
				for(int32_t x=0;x<dim;x++) {
					// index correct as dim <= 2^14
					entryYX[y*dim+x].reConstructor(tmgr,ptmgr);
				}
			}
		}
	}
	
	for(int32_t y=0;y<dim;y++) {
		for(int32_t x=0;x<dim;x++) {
			int8_t whatid;
			if (fread(&whatid,1,sizeof(whatid),f) != sizeof(whatid)) {
				LOGMSG("\nError. Reading matrix polynomial. Probably invalid matrix file. Deleting recommended.\n");
				exit(99);
			}
			if (whatid == NULLPOLYNOMID) {
				// dim <= 2^14
				entryYX[y*dim+x].clearTerms(); // null-Polynom
			} else {
				if ( (x == col) || (y == row) ) {
					// load polynomial
					entryYX[y*dim+x].load(f);
				} else {
					entryYX[y*dim+x].jumpOverLoad(f);
				}
			}
		} // x
	} // y
	
	fclose(f);
}

int8_t loadPolynomialFromMatrix_FTYX(
	const char* afn,
	ZMCpolynom& res,
	const int32_t aty,
	const int32_t atx
) {
	/*
	
		loads onlya specific polynomial of a matrix
		the rest is set to zero
	
	*/
	
	FILE *f=fopen(afn,"rb");
	if (!f) {
		LOGMSG2("\nError. Not able to open file for matrix loading |%s|.\n",afn);
		exit(99);
	}
	
	int32_t dim;
	if (fread(&dim,1,sizeof(dim),f) != sizeof(dim)) {
		LOGMSG("\nError. Reading matrix. Probably invalid file. Deleting recommended.\n");
		exit(99);
	}
	
	for(int32_t y=0;y<dim;y++) {
		for(int32_t x=0;x<dim;x++) {
			int8_t whatid;
			if (fread(&whatid,1,sizeof(whatid),f) != sizeof(whatid)) {
				LOGMSG("\nError. Reading matrix polynomial. Probably invalid matrix file. Deleting recommended.\n");
				exit(99);
			}
			if (whatid == NULLPOLYNOMID) {
				if ( (y == aty) && (x == atx) ) {
					res.clearTerms();
					fclose(f);
					return 1;
				}
				// nothing to load or jump over
			} else {
				if ( (y == aty) && (x == atx) ) {
					res.load(f);
					fclose(f);
					return 1;
				}
				res.jumpOverLoad(f);
			}
		} // x
	} // y
	
	fclose(f);
	
	return 0;
}

void MatrixPolynom::save(const char* afn) {
	FILE *f=fopen(afn,"wb");
	if (!f) {
		LOGMSG("\nError. Not able to open file for matrix storage.\n");
		exit(99);
	}
	
	if (fwrite(&dim,1,sizeof(dim),f) != sizeof(dim)) {
		LOGMSG("\nError. Writing matrix. Probably invalid file. Deleting recommended.\n");
		exit(99);
	}
	int8_t nullpolynom=NULLPOLYNOMID;
	int8_t notnullpolynom=NOTNULLPOLYNOMID;
	
	for(int32_t y=0;y<dim;y++) {
		for(int32_t x=0;x<dim;x++) {
			int32_t r=0;
			if (entryZero_YX(y,x) > 0) {
				r=fwrite(&nullpolynom,1,sizeof(nullpolynom),f);
			} else {
				r=fwrite(&notnullpolynom,1,sizeof(nullpolynom),f);
				// dim <= 2^14
				entryYX[y*dim+x].save(f);
			}
			
			if (r != sizeof(nullpolynom)) {
				LOGMSG("\nError. Writing matrix. Probably invalid matrix. Deleting recommended.\n");
				exit(99);
			}
		} // x
	} // y
	
	fclose(f);
}

int8_t MatrixPolynom::entryPositiveOne_YX(
	const int32_t ay,
	const int32_t ax
) {
	if (
		(ax < 0) || (ay < 0) || (ax >= dim) || (ay >= dim)
	) {
		LOGMSG("\nError. Access outside of matrix dimension.\n");
		exit(99);
	}
	
	int8_t found1=0,foundelse=0;
	
	// ax,ay,dim <= 2^14
	for(int32_t i=0;i<entryYX[ay*dim+ax].anzterms;i++) {
		if (entryYX[ay*dim+ax].terms[i]->isPositiveOne() > 0) {
			if (found1 > 0) {
				// error
				return 0;
			}
			found1=1;
		} else {
			foundelse=1;
		}
	} // i
	
	if ( 
		(found1 > 0) && 
		(foundelse <= 0) 
	) return 1;
	
	return 0;
}


int8_t MatrixPolynom::entryZero_YX(
	const int32_t ay,
	const int32_t ax
) {
	if ( (ax < 0) || (ax >= dim) || (ay < 0) || (ay >= dim) ) {
		LOGMSG("\nError. Access to non-existent matrix element\n");
		exit(99);
	}

	// ax,ay,dim <= 2^14
	if (entryYX[ay*dim + ax].isZero() <= 0) return 0;
	
	return 1;
}

void MatrixPolynom::ausgabe(FILE* f) {
	fprintf(f,"dim=%i\n",dim);
	if (!entryYX) {
		fprintf(f,"no poinzers.\n");
		return;
	}
	
	DynSlowString one;
	for(int32_t y=0;y<dim;y++) {
		fprintf(f,"[y%i]: ",y);
		for(int32_t x=0;x<dim;x++) {
			if ( (x % 8) == 0) {
				fprintf(f,"[x%i] ",x);
			}
			// x,y,dim <= 2^14
			entryYX[y*dim+x].getStr(one);
			fprintf(f,"%s / ",one.text);
		}
		
		fprintf(f,"\n");
	} // y
}

MatrixPolynom::MatrixPolynom() {
	dim=0;
	entryYX=NULL;
	tmgr=NULL;
	ptmgr=NULL;
}

MatrixPolynom::~MatrixPolynom() {
	delete[] entryYX; // allocated with new
	// the polynomials themselves are released
	// when the appropirate termManager is destroyed
}

void MatrixPolynom::setDimension(
	ZMCtermMemoryManager* tmem,
	PZMCtermMemoryManager* ptmem,
	const int32_t adim
) {
	if (adim >= MAXDIMENSION) {
		LOGMSG("\nError. Matrix dimension must not exceed 2^14\n");
		exit(99);
	}
	tmgr=tmem;
	ptmgr=ptmem;
	
	if (entryYX) {
		if (dim != adim) {
			delete[] entryYX;
			entryYX=NULL;
		}
	}
	dim=adim;
	if (!entryYX) {
		entryYX=new ZMCpolynom[(int64_t)adim*adim];
	}
	if (!entryYX) {
		LOGMSG("\nError. Memory. MatrixPolynom.\n");
		exit(99);
	}
	
	setConstant0polynom();

}

void MatrixPolynom::setConstant0polynom(void) {
	if (!entryYX) return;
	if ( (!tmgr) || (!ptmgr) ) {
		LOGMSG("\nError. Implementation. Matrix::setConstant0 needs valid memory manager set\n");
		exit(99);
	}
	
	for(int32_t y=0;y<dim;y++) {
		for(int32_t x=0;x<dim;x++) {
			// as if constructor was just called
			entryYX[y*dim+x].reConstructor(tmgr,ptmgr);
		}
	}
}

void setSylvesterMatrix(
	MatrixPolynom& sylvester,
	ZMCpolynom& F,
	ZMCpolynom& G,
	ZMCpolynom* Fcoeff,
	ZMCpolynom* Gcoeff
) {
	int32_t setdim=sum_int32t(F.Zdegree,G.Zdegree);
	LOGMSG3("\nSylvester matrix dimension %i x %i\n",setdim,setdim);
	sylvester.setDimension(
		&globalzmctermmgr,
		&globalpzmctermmgr,
		setdim
	);
	sylvester.setConstant0polynom();
	
	if ( (F.Zdegree >= MAXDEGREE) || (G.Zdegree >= MAXDEGREE) ) {
		LOGMSG("\nError. Polynomial degree cannot exceed 2^14\n");
		exit(99);
	}
	
	// matrix: first M rows: coefficients of F shifted
	int32_t x0=-1;
	for(int32_t y=0;y<G.Zdegree;y++) {
		x0++;
		
		for(int32_t i=0;i<=F.Zdegree;i++) {
			// entry address valid as all y,x0+i,dim< = 2^14
			sylvester.entryYX[y*sylvester.dim+(x0+i)].copyFrom(Fcoeff[F.Zdegree-i]);
		} // i
		
	} // y
	
	// second N rows: coefficients of G shifted
	x0=-1;
	for(int32_t y=G.Zdegree;y<sum_int32t(F.Zdegree,G.Zdegree);y++) {
		x0++;
		
		for(int32_t i=0;i<=G.Zdegree;i++) {
			sylvester.entryYX[y*sylvester.dim+(x0+i)].copyFrom(Gcoeff[G.Zdegree-i]);
		} // i
		
	} // y

}

void polynomMul_TTermB(
	ZMCpolynom& res,
	ZMCterm& A,
	ZMCpolynom& B
) {
	res.clearTerms();
	
	ZMCterm one;
	for(int32_t i=0;i<B.anzterms;i++) {
		termMul_TAB(one,A,*B.terms[i]);
		// as B is sorted, multiplying by the same term A
		// doesn't change the ordering
		res.addTermLast_FZMC(one);
	}

}

void polynomMul_TAB(
	ZMCpolynom& res,
	ZMCpolynom& A,
	ZMCpolynom& B
) {
	// use the memory manager throughout of the resulting
	// variable
	
	res.clearTerms();
	ZMCterm one;
	
	for(int32_t f=0;f<A.anzterms;f++) {
		for(int32_t g=0;g<B.anzterms;g++) {
			// termMul does not invoke a memory managaer
			termMul_TAB(one,*A.terms[f],*B.terms[g]);
			if (one.factor.vorz == 0) res.mighthaveremoveablezeroterms=1;
			
			// here memory manager of result is used
			res.addTerm_FZMC(one);
		} // g
	} // f
	
}

void polynomMul_TAB_split(
	ZMCpolynom& res,
	ZMCpolynom& A,
	ZMCpolynom& B,
	
	/* temporary variables */
	ZMCpolynom& hlp2,
	ZMCpolynom& hlp3,
	ZMCpolynom& hlp4
) {
	// use the memory manager throughout of the resulting
	// variable
	
	hlp2.clearTerms();
	hlp3.clearTerms();
	hlp4.clearTerms();
	
	// splitting polynomials A and B in each two paerts
	// A=A1+A2, B=B1+B2
	// and calculating A1*B1+A1*B2+A2*B1+B2*B2 in parallel
	
	int32_t ah=A.anzterms >> 1;
	int32_t bh=B.anzterms >> 1;
	
	res.clearTerms();
	// res is used for the first parallel section
	
	#pragma omp parallel
	{
		#pragma omp sections
		{
			#pragma omp section
			{
				ZMCterm one1;
				for(int32_t f=0;f<ah;f++) {
					for(int32_t g=0;g<bh;g++) {
						// termMul does not invoke a memory managaer
						termMul_TAB(one1,*A.terms[f],*B.terms[g]);
						// here memory manager of result is used
						res.addTerm_FZMC(one1);
					} // g
				} // f

			} // section A1*B1
			
			#pragma omp section
			{
				ZMCterm one2;
				for(int32_t f=0;f<ah;f++) {
					for(int32_t g=bh;g<B.anzterms;g++) {
						// termMul does not invoke a memory managaer
						termMul_TAB(one2,*A.terms[f],*B.terms[g]);
						// here memory manager of result is used
						hlp2.addTerm_FZMC(one2);
					} // g
				} // f
			} // section A1*B2

			#pragma omp section
			{
				ZMCterm one3;
				for(int32_t f=ah;f<A.anzterms;f++) {
					for(int32_t g=0;g<bh;g++) {
						// termMul does not invoke a memory managaer
						termMul_TAB(one3,*A.terms[f],*B.terms[g]);
						// here memory manager of result is used
						hlp3.addTerm_FZMC(one3);
					} // g
				} // f

			} // section A2*B1

			#pragma omp section
			{
				ZMCterm one4;
				for(int32_t f=ah;f<A.anzterms;f++) {
					for(int32_t g=bh;g<B.anzterms;g++) {
						// termMul does not invoke a memory managaer
						termMul_TAB(one4,*A.terms[f],*B.terms[g]);
						// here memory manager of result is used
						hlp4.addTerm_FZMC(one4);
					} // g
				} // f
			} // section A2*B2

		} // sections
		
	} // parallel
	
	// res already has part A1*B1 stored
	#pragma omp parallel
	{
		#pragma omp sections 
		{
				#pragma omp section
				{
					res.addPoly(hlp2);
				} // section res+hlp2
				
				#pragma omp section
				{
					hlp3.addPoly(hlp4);
				} // section hlp3+hlp4
				
		} // sections
	} // parallel
	
	// res=res+hlp2 + (hlp3=hlp3+hlp4)
	res.addPoly(hlp3);
	// res-mighthaveremoveable is set

}

void termMul_TAB(
	ZMCterm& res,
	ZMCterm& A,
	ZMCterm& B
) {

	bigintMul_TAB(res.factor,A.factor,B.factor);
	
	if (
		(A.Zexponent >= UINT30MAX) ||
		(B.Zexponent >= UINT30MAX) ||
		(A.Mexponent >= UINT30MAX) ||
		(B.Mexponent >= UINT30MAX) ||
		(A.Cexponent >= UINT30MAX) ||
		(B.Cexponent >= UINT30MAX)
	) {
		LOGMSG("\nError. Exponent overflow termMul\n");
		exit(99);
	}
	
	// fits into int32_t, so sum_int32t not necessary
	res.Zexponent=sum_int32t(A.Zexponent,B.Zexponent);
	res.Mexponent=sum_int32t(A.Mexponent,B.Mexponent);
	res.Cexponent=sum_int32t(A.Cexponent,B.Cexponent);

}

void polynomDiv_rf_TRAB(
	ZMCpolynom& dividing,
	ZMCpolynom& remainder,
	ZMCpolynom& A,
	ZMCpolynom& B,
	
	/* locally used variables */
	/* so one can use the same already allocated memory again */
	ZMCpolynom& hlprest,
	ZMCpolynom& hlpt1,
	ZMCpolynom& hlpt2,
	ZMCpolynom& hlpm2
) {
	/* 
	
		general purpose: dividing two polynomials. Mostly called
		when remainder-free division will occur with divisor
		being usually the ones leading to the largest reduction
		
		current implementation: divide A by the leading term(s)
		of B w.r.t. to all occuring variables and take the first
		that is remainder-free
	   
	*/
	
	#define CHECKPOLYID \
	{\
		/* dividing * B + remainder = A */\
		polynomMul_TAB(hlpt1,dividing,B);\
		hlpt1.addPoly(remainder);\
		hlpt1.subPoly(A);\
		hlpt1.removeZeroTerms();\
		\
		if (hlpt1.isZero() <= 0) {\
			LOGMSG("\nError. polynomDiv does not satisfy invariant\n");\
			LOGMSG("\nA: "); A.ausgabe(stdout); A.ausgabe(flog);\
			LOGMSG("\nB: "); B.ausgabe(stdout); B.ausgabe(flog);\
			exit(99);\
		}\
	}
	
	A.removeZeroTerms();
	A.checkDegree();
	B.removeZeroTerms();
	B.checkDegree();
	
	// return holds: dividing * B + remainder = A
	if (A.isZero() > 0) {
		// dividing * B + remainder = A
		// 0 * B + 0 = A
		dividing.setToZero();
		remainder.setToZero();
		return;
	}
	
	if (B.isPositiveOne() > 0) {
		// dividing * B + remainder = A
		// A * 1 + 0 = A
		dividing.copyFrom(A);
		remainder.setToZero();
		return;
	}
	
	if (B.isZero() > 0) {
		LOGMSG("\nError. PolynomDiv. by zero\n");
		exit(99);
	}
	
	// go over B and look for terms with C,M and/or Z-exponent
	// (if positive) of the current degree (has been validated above)
	
	BigInt idiv,irem;
	
	// first try the c variable, the m, then z
	for(int32_t var=2;var>=0;var--) {
		if ( (var==0) && ( (B.Cdegree == 0) || (A.Cdegree == 0) ) ) continue;
		if ( (var==1) && ( (B.Mdegree == 0) || (A.Mdegree == 0) ) ) continue;
		if ( (var==2) && ( (B.Zdegree == 0) || (A.Zdegree == 0) ) ) continue;
		
		for(int32_t bidx=0;bidx<B.anzterms;bidx++) {
			int8_t use=0;
			if ( (var==0) && (B.terms[bidx]->Cexponent == B.Cdegree) ) use=1;
			if ( (var==1) && (B.terms[bidx]->Mexponent == B.Mdegree) ) use=1;
			if ( (var==2) && (B.terms[bidx]->Zexponent == B.Zdegree) ) use=1;
		
			if (use <= 0) continue;
			
			// does A have any term that can be divided here
			
			hlprest.clearTerms();
			hlprest.copyFrom(A);
			dividing.clearTerms();
			
			// degre of hlprest is correct as it is A here
			// and for A checkdegree has been called
			while (1) {
			
				//printf("\n\nrest "); hlprest.ausgabe(stdout);
	
				if (hlprest.isZero() > 0) {
					remainder.clearTerms();
					// dividing already set
					#ifdef _INVOKECLAIMVERIFICATIONS
					CHECKPOLYID
					#endif
					return; 
				}
			
				int32_t resthi=-1;
	
				// degree in hlprest is corrct by call to
				// checkDegree at end of while
				for(int32_t i=0;i<hlprest.anzterms;i++) {
					// to be used,
					// all occuring variables must have a degree
					// at least as high as the chosen B-term
					// otherwise no division can occur
					if (
						(hlprest.terms[i]->Cexponent < B.terms[bidx]->Cexponent) ||
						(hlprest.terms[i]->Mexponent < B.terms[bidx]->Mexponent) ||
						(hlprest.terms[i]->Zexponent < B.terms[bidx]->Zexponent)
					) continue;
					
					int8_t use2=0;
					
					if ( (var==0) && (hlprest.terms[i]->Cexponent == hlprest.Cdegree) ) use2=1;
					if ( (var==1) && (hlprest.terms[i]->Mexponent == hlprest.Mdegree) ) use2=1;
					if ( (var==2) && (hlprest.terms[i]->Zexponent == hlprest.Zdegree) ) use2=1;
					
					if (use2 <= 0) continue;
					
					// variables are dividable
					// now is the factor as well =
				
					bigintDiv_TRAB(
						idiv,
						irem,
						hlprest.terms[i]->factor,
						B.terms[bidx]->factor
					);
				
					if (
						(irem.vorz == 0) &&
						(idiv.vorz != 0)
					) {
						resthi=i;
						break;
					}
				} // i
				//printf("out");
			
				if (resthi < 0) {
					break; // bidx, try next possible term
				}
				
				// idiv usable
				
				ZMCterm tone;
				tone.factor.copyFrom(idiv);
				tone.Zexponent=hlprest.terms[resthi]->Zexponent-B.terms[bidx]->Zexponent;
				tone.Mexponent=hlprest.terms[resthi]->Mexponent-B.terms[bidx]->Mexponent;
				tone.Cexponent=hlprest.terms[resthi]->Cexponent-B.terms[bidx]->Cexponent;
				
				hlpm2.clearTerms();
				hlpm2.addTerm_FZMC(
					idiv,
					hlprest.terms[resthi]->Zexponent-B.terms[bidx]->Zexponent,
					hlprest.terms[resthi]->Mexponent-B.terms[bidx]->Mexponent,
					hlprest.terms[resthi]->Cexponent-B.terms[bidx]->Cexponent
				);
				polynomMul_TTermB(
					hlpt2,*hlpm2.terms[0],B
				);

				hlprest.subPoly(hlpt2);
				hlprest.removeZeroTerms();
				hlprest.checkDegree(); // now degree is correct
				dividing.addPoly(hlpm2);
			
			} // while
					
		} // bidx
	
	} // var
	
	// no remainder free division possible
	// just return the last computed values
	
	remainder.copyFrom(hlprest);
	// dividing already set
	#ifdef _INVOKECLAIMVERIFICATIONS
	CHECKPOLYID
	#endif

}

char* getMatrixFileName(char* erg,const int32_t k) {
	sprintf(erg,"_%s_k%i.matrix",fnbase,k);
	
	return erg;
}

void fractionFreeGaussDet_TA(
	ZMCpolynom& fres
) {

	LOGMSG("\n\nentering fraction-free Gaussian elimination ... ");

	/*
	
		article by E Bareiss
		indexing there goes from 1 to matrix-dimenion
		this was adjusted to C++ memory indexing, so
		every index has been decreased by one
		
		M0=matrix M
		a(k)(y,x)
		a(-1)(0,0)=1
	
	*/
	
	/*
	
		target matrix is stored directly onto disc
		
		Mk matrix: only one element (ayx) is loaded at use time
		(only used once), in memory are additionally only the
		row k and column k
		
		Mk-1: only element a(k-1)(k-1,k-1) is loaded
	
	*/
	
	int32_t lastfullstoredk=0; // NOT -1
	// 0 will not be loaded anyways but copied from
	// Sylvester's matrix
	
	int32_t dimension=sum_int32t(fstrictP.Zdegree,fderm.Zdegree);

	char tmp[2048];
	for(int32_t k=(dimension-1);k>=0;k--) {
		getMatrixFileName(tmp,k);
		FILE *f=fopen(tmp,"rb");
		if (f) {
			lastfullstoredk=k;
			fclose(f);
			break;
		}
	}
	
	ZMCtermMemoryManager tmgr;
	PZMCtermMemoryManager ptmgr;
	
	// previous a(k-1)(k-1,k-1) is loaded
	// directly as a polynom

	MatrixPolynom* Mk=new MatrixPolynom;
	Mk->setDimension(
		&tmgr,
		&ptmgr,
		dimension); // constant0 polynom
	
	int8_t nullpolynom=NULLPOLYNOMID;
	int8_t notnullpolynom=NOTNULLPOLYNOMID;
	const int8_t RE0=-1;
	const int8_t RET1=0;
	const int8_t REERG=1;
	int8_t resultantwouldbe=RE0;
	
	MatrixLoader mloader;
	mloader.f=NULL;
	
	// polynomials which are handled in parallel must each
	// have their own memory manager to avoid colliding
	// requests. The number of objects per memory manager
	// internal page is reduced to not allocate CHUNKSIZE
	// as only one p0olynomial is handled by each manager
	// less than CHUNKSIZE should suffice
	ZMCtermMemoryManager tsplitm[6](CHUNKSIZE >> 2),
		tm1(CHUNKSIZE >> 2),tm2(CHUNKSIZE >> 2);
	PZMCtermMemoryManager ptsplitm[6](CHUNKSIZE >> 2),
		ptm1(CHUNKSIZE >> 2),
		ptm2(CHUNKSIZE >> 2);
	ZMCpolynom t1(&tmgr,&ptmgr,-1),
		t2(&tmgr,&ptmgr,-1),
		splithlp[6];
	for(int32_t spliti=0;spliti<6;spliti++) {
		splithlp[spliti].setMemoryManager(
			&tsplitm[spliti],
			&ptsplitm[spliti]
		);
	}

	// those use one huge memory manager
	ZMCpolynom rem(&tmgr,&ptmgr,-1),
		erg(&tmgr,&ptmgr,-1),
		ak1_k1k1(&tmgr,&ptmgr,-1),
		ak_yx(&tmgr,&ptmgr,-1),
		
		/* helper variables to pass to polynomDiv, Mul to */
		/* be used as temporary variables whose allocated memory */
		/* will be used again */
		hlp1(&tmgr,&ptmgr,-1),
		hlp2(&tmgr,&ptmgr,-1),
		hlp3(&tmgr,&ptmgr,-1),
		hlp4(&tmgr,&ptmgr,-1);
		
	if (lastfullstoredk == (dimension-1) ) {
		// done
		getMatrixFileName(tmp,lastfullstoredk);
		printf("\n  nothing to compute. Reading results.");
		Mk->load(tmp,dimension-1,dimension-1); 
		// resultant is lower right entry
		// dim <= 2^14
		fres.copyFrom(Mk->entryYX[(dimension-1)*dimension+(dimension-1)]);
		return;
	}

	printf("\ncomputing steps k=[%i..%i]",1+lastfullstoredk,dimension-1);
	
	int32_t ystart=0;
	
	for(int32_t k=lastfullstoredk;k<(dimension-1);k++) {
		tmgr.freePhysically();
		ptmgr.freePhysically();
		tm1.freePhysically();
		tm2.freePhysically();
		ptm1.freePhysically();
		ptm2.freePhysically();
		for(int32_t spliti=0;spliti<6;spliti++) {
			tsplitm[spliti].freePhysically();
			ptsplitm[spliti].freePhysically();
		}
		// re-initialize as if just constructed
		
		Mk->setDimension(&tmgr,&ptmgr,dimension);
		ak1_k1k1.reConstructor(&tmgr,&ptmgr);
		ak_yx.reConstructor(&tmgr,&ptmgr);
		
		char filenamenextk[2048];
		getMatrixFileName(filenamenextk,k+1);
		
		FILE *fnextk=fopen(filenamenextk,"wb");
		if (!fnextk) {
			LOGMSG("\nError. Cannot open next matrix to store on disc.\n");
			exit(99);
		}
		
		if (fwrite(&dimension,1,sizeof(dimension),fnextk) != sizeof(dimension)) {
			LOGMSG("\nError. Writing current matrix. Probably invalid matrix. Deleting recommended.\n");
			exit(99);
		}
		
		printf("\n  step %i/%i ... ",k+1,dimension-1);

		char filematrixk[1024];
		filematrixk[0]=0;
		
		if (k == 0) {
			// if only k=0 is stored, it is as if starting anew

			// use Sylvester matrix as parameter to get
			// the current matrix
			printf("initial step with Sylvester matrix ... ");

			printf("\ncreating Sylvester matrix ... ");
			printf("\n  getting z-coefficients ... ");
			ZMCpolynom *fZcoeff=getZcoeff(fstrictP);
			ZMCpolynom *derZcoeff=getZcoeff(fderm);
			printf("done\n");

			setSylvesterMatrix(
				*Mk,
				fstrictP,fderm,
				fZcoeff,derZcoeff
			);
			
			getMatrixFileName(filematrixk,0);
			Mk->save(filematrixk);
			// allocated using global term managaer, not here the local one
		} else {
			// load file[k] at Mk and file[k-1] at Mpreviousk
			printf("loading %i,%i ... ",
				k-1,k);
			getMatrixFileName(filematrixk,k);
			
			Mk->loadRowCol(filematrixk,k,k);

			getMatrixFileName(tmp,k-1);
			if (loadPolynomialFromMatrix_FTYX(
				tmp,ak1_k1k1,k-1,k-1
			) <= 0) {
				LOGMSG2("\nError. Cannot read previous polynomial a(k-1)(k-1,k-1) |%s|\n",tmp);
				exit(99);
			}
		}
		
		if (filematrixk[0] <= 0) {
			LOGMSG("\nError. Implementation. Matrix at k not loadable\n");
			exit(99);
		}
		
		printf("computing ...");
		
		mloader.close();
		// stored in Mk are only row k and column k
		// all others are only needed once in the akk*ayx
		// term, so they will only be read in once
		mloader.prepareForLoading(filematrixk);
		
		t1.reConstructor(&tm1,&ptm1);
		t2.reConstructor(&tm2,&ptm2);
		for(int32_t spliti=0;spliti<6;spliti++) {
			splithlp[spliti].reConstructor(
				&tsplitm[spliti],
				&ptsplitm[spliti]
			);
		}
		
		erg.reConstructor(&tmgr,&ptmgr);
		rem.reConstructor(&tmgr,&ptmgr);
		hlp1.reConstructor(&tmgr,&ptmgr);
		hlp2.reConstructor(&tmgr,&ptmgr);
		hlp3.reConstructor(&tmgr,&ptmgr);
		hlp4.reConstructor(&tmgr,&ptmgr);
		
		// entry valid as indices and dimension <= 2^14
		#define ENTRYPTRKYX(KK,YY,XX) &KK->entryYX[(YY)*dimension+(XX)]

		ZMCpolynom* ak_kk=ENTRYPTRKYX(Mk,k,k);
		int8_t akkkiszero=ak_kk->isZero();
		
		for(int32_t y=ystart;y<dimension;y++) {
			ZMCpolynom* ak_yk=ENTRYPTRKYX(Mk,y,k);
			int8_t akykiszero=ak_yk->isZero();

			if (dimension > 100) printf("%i ",dimension-y);
			else if ( (y & 0b111) == 0) {
				int32_t a=tmgr.anzptr+ptmgr.anzptr;
				if (a > 5) {
					printf(".#%i",tmgr.anzptr);
				} else printf(".");
			}
			
			for(int32_t x=0;x<dimension;x++) {
				if ( 
					(y < (k+1) ) ||
					(x < (k+1) )
				) {
					// storing null polynomial as value is not
					// needed
					if (fwrite(&nullpolynom,1,sizeof(nullpolynom),fnextk) != sizeof(nullpolynom)) {
						LOGMSG("\nError. Writing current matrix. Probably invalid file. Deleting recommended.\n");
						exit(99);
					}
					continue;
				}
				
				//if (dimension > 100) if ( (x & 0b111) == 0) printf("[%i] ",dimension-x);
			
				// works also if it is finally a polynomial
				// with zero terms or the constant-0 1-term polynomial

				/*
								  a(k)kk*a(k)yx - a(k)yk*a(k)kx
					a(k+1)(y,x) = -----------------------------
									a(k-1)(k-1,k-1)
				
				*/
				
				ZMCpolynom* ak_kx=ENTRYPTRKYX(Mk,k,x);
				
				if (mloader.loadAtYX(ak_yx,y,x) <= 0) {
					LOGMSG2("\nError. Cannot load polynomial a(k)(y,x) |%s|\n",
						filematrixk);
					exit(99);
				}

				// if it is a polynomial with 0 terms
				// this can be saved in twow qays: as nullpolynom-flag
				// or as notnullpolynom wth the empty polynomial
				
				t1.clearTerms();
				t2.clearTerms();
				
				// multiplying akkk*akyx and ayk*akx
				// simultaneously
				
				if (
					(akkkiszero <= 0) &&
					(akykiszero <= 0) &&
					(ak_yx.isZero() <= 0) &&
					(ak_kx->isZero() <= 0)
				) {
					// 11,t2 have different memroy managaers, so
					// work collision-free
					#pragma omp parallel
					{
						#pragma omp sections
						{
							#pragma omp section
							{
								polynomMul_TAB_split(t1,*ak_kk,ak_yx,splithlp[0],splithlp[1],splithlp[2]);
							} // section
							
							#pragma omp section
							{
								polynomMul_TAB_split(t2,*ak_yk,*ak_kx,splithlp[3],splithlp[4],splithlp[5]);
							} // section
							
						} // sections
					} // parallel
					
					t1.subPoly(t2);
				} else {
					// this only happens at the first stages
					// of matrix reduction, lator on all
					// entries are pretty large polynomials

					if (
						(akkkiszero <= 0) &&
						(ak_yx.isZero() <= 0)
					) {
						polynomMul_TAB(t1,*ak_kk,ak_yx);
					}
					if (
						(akykiszero <= 0) &&
						(ak_kx->isZero() <= 0)
					) {
						polynomMul_TAB(t2,*ak_yk,*ak_kx);
						t1.subPoly(t2);
					}
				}

				t1.removeZeroTerms();

				// polynomials in splithlp are now free
				// to use again in division
				
				if (k > 0) { 
					// ak1_k1k1 is valid pointer
					if (ak1_k1k1.isPositiveOne() <= 0) {
						polynomDiv_rf_TRAB(
							erg,rem,
							t1,
							ak1_k1k1,
							hlp1,hlp2,hlp3,hlp4
						);
						
						erg.removeZeroTerms();
					
						// remainder must be zero
						if (rem.isZero() <= 0) {
							LOGMSG("\nError. fractionFree. remainder not vanishing.\n");
							exit(99);
						}
						
						// write erg
						if (fwrite(&notnullpolynom,1,sizeof(nullpolynom),fnextk) != sizeof(nullpolynom)) {
							LOGMSG("\nError. Writing current matrix. Probably invalid file. Deleting recommended.\n");
							exit(99);
						}
						erg.save(fnextk);
						resultantwouldbe=REERG;

					} else {
						// denominator is 1, so no division necessary
						// write t1
						if (fwrite(&notnullpolynom,1,sizeof(nullpolynom),fnextk) != sizeof(nullpolynom)) {
							LOGMSG("\nError. Writing current matrix. Probably invalid file. Deleting recommended.\n");
							exit(99);
						}
						t1.save(fnextk);
						resultantwouldbe=RET1;

					}
				} else {
					// write t1
					if (fwrite(&notnullpolynom,1,sizeof(nullpolynom),fnextk) != sizeof(nullpolynom)) {
						LOGMSG("\nError. Save. Matrix not valid/4.\n");
						exit(99);
					};
					t1.save(fnextk);
					resultantwouldbe=RET1;
				}
				
			} // x
			
		} // y
		
		fclose(fnextk);
		
		printf(" saved");
		
	} // k
	
	// resultant now has been stored on file
	
	// physical copy as by returning the lokal memory
	// manager will be destroyed and so its allocated
	// memory
	if (resultantwouldbe == RET1) {
		fres.copyFrom(t1);
	} else if (resultantwouldbe == REERG) {
		fres.copyFrom(erg);
	} else {
		LOGMSG("\nError. No resultant determined.\n");
		exit(99);
	}
	
}

void polynomComposition_at_Z_TNA(
	ZMCpolynom& composition,
	const int32_t nfold,
	ZMCpolynom& basef
) {
	// n-fold composition of basef at variable z
	
	ZMCpolynom powered(composition.tmgr,composition.ptmgr,-1),
		neu(composition.tmgr,composition.ptmgr,-1),
		current(composition.tmgr,composition.ptmgr,-1),
		restterm(composition.tmgr,composition.ptmgr,-1),
		toadd(composition.tmgr,composition.ptmgr,-1);
		
	current.clearTerms();
	current.addTerm_FZMC(1,1,0,0); // variable z
	
	for(int32_t fold=1;fold<=nfold;fold++) {
		// in polynomial current, replace every z with
		// basef and expand
		
		neu.clearTerms();
		
		for(int i=0;i<current.anzterms;i++) {
			if (current.terms[i]->factor.vorz == 0) continue;
			
			if (current.terms[i]->Zexponent <= 0) {
				neu.addTerm_FZMC(*current.terms[i]);
				continue;
			}
			
			// variable Z occuring with positive exponent
			// compute basef^Zexponent
			polynomPow_TNA(
				powered,
				current.terms[i]->Zexponent,
				basef
			);
			// multiply by rest of term
			restterm.clearTerms();
			ZMCterm term;
			term.factor.copyFrom(current.terms[i]->factor);
			term.Zexponent=0;
			term.Mexponent=current.terms[i]->Mexponent;
			term.Cexponent=current.terms[i]->Cexponent;
			restterm.addTerm_FZMC(term);
			
			polynomMul_TAB(toadd,powered,restterm);
			
			neu.addPoly(toadd);
		} //
		
		current.copyFrom(neu);
		
	} // i composition
	
	composition.copyFrom(current);
	
}

void polynomPow_TNA(
	ZMCpolynom& powered,
	const int32_t topower,
	ZMCpolynom& basef
) {
	ZMCpolynom tmp(powered.tmgr,powered.ptmgr,-1);
	powered.clearTerms();
	powered.addTerm_FZMC(1,0,0,0); // positive 1
	
	for(int32_t i=1;i<=topower;i++) {
		// holds: at end of every loop i: powered=(basef)^î
		
		polynomMul_TAB(tmp,powered,basef);
		powered.copyFrom(tmp);
	} // i
	
}

void polynomDer_at_Z_TA(
	ZMCpolynom& fddz,
	ZMCpolynom& f
) {
	// derivative of function f with respect to variable z
	fddz.clearTerms();
	
	for(int32_t i=0;i<f.anzterms;i++) {
		if (f.terms[i]->Zexponent == 0) {
			// constant => go on
			continue;
		}
		
		// derivative of term A*c^e1*m^e2*z^e3 * d/dz
		// = e3*A*c^e1*m^e2*z^(e3-1)
		ZMCterm term;
		term.Cexponent=f.terms[i]->Cexponent;
		term.Mexponent=f.terms[i]->Mexponent;
		term.Zexponent=f.terms[i]->Zexponent - 1;
		int32_t ze=(int32_t)f.terms[i]->Zexponent;
		bigintMul_digit_TDA(
			term.factor,
			ze,
			f.terms[i]->factor
		);
		
		fderm.addTerm_FZMC(term);
	} // i
}

char* getFilenameCoeff(char* erg,const int32_t coeff) {
	sprintf(erg,"_%s_A%i.coeff.txt",
		fnbase,
		coeff);
		
	return erg;
}

			
// main

int32_t main(int32_t argc,char** argv) {
	int32_t t0=clock();
	big1.set_int64(1);

	flog=fopen("resultant.log","at");
	fprintf(flog,"\n----------\nstart resultant\n\n");
	
	for(int32_t i=1;i<argc;i++) {
		upper(argv[i]);
		
		if (strstr(argv[i],"MULTIBROT=") == argv[i]) {
			int32_t a;
			if (sscanf(&argv[i][10],"%i",&a) == 1) {
				if (a >= 2) MULTIBROT=a;
			}
		} else
		if (strstr(argv[i],"PERIOD=") == argv[i]) {
			int32_t a;
			if (sscanf(&argv[i][7],"%i",&a) == 1) {
				if (a >= 1) currentperiod=a;
			}
		}
	} // cmdline parameters
	
	sprintf(fnbase,"_md%i_p%i",
		MULTIBROT,
		currentperiod
	);

	#ifdef _INVOKECLAIMVERIFICATIONS
	LOGMSG("\nINFO: Gegenproben mul/div bigInt, polynomials invoked");
	#endif
	
	LOGMSG3("\ncomponent period %i for multibrot degree %i\n",
		currentperiod,MULTIBROT);
	LOGMSG("\nDeterminant method: fraction-free Gaussian elimination\n");
	
	// fstrictP and ferm are necessary for the
	// dimension of the matrix to allocate 
	// (at start and when resuming computation)
	printf("setting f/period and f' ... ");

	basefkt.setMemoryManager(&globalzmctermmgr,&globalpzmctermmgr);
	basefkt.clearTerms();
	
	// unicritical multibrot z^d+c
	basefkt.addTerm_FZMC(1,MULTIBROT,0,0); // z^MULTIBROT
	basefkt.addTerm_FZMC(1,0,0,1); // c
	
	setPeriod(currentperiod);
	
	if (
		(fstrictP.Zdegree >= MAXDEGREE) ||
		(fstrictP.Mdegree >= MAXDEGREE) ||
		(fstrictP.Cdegree >= MAXDEGREE) ||
		(fderm.Zdegree >= MAXDEGREE) ||
		(fderm.Mdegree >= MAXDEGREE) ||
		(fderm.Cdegree >= MAXDEGREE)
	) {
		LOGMSG("\nError. Degree too high.\n");
		exit(99);
	}
	printf("done\n");
	
	DynSlowString strperiod,strderm;
	fstrictP.getStr(strperiod);
	fderm.getStr(strderm);
	
	LOGMSG4("\n|-> strict period-%i f_p%i(z)=%s\n",currentperiod,currentperiod,strperiod.text);
	LOGMSG3("|-> derivative of compossition [f_o%i(z)]'=%s\n",currentperiod,strderm.text);
	
	ZMCpolynom fres(&globalzmctermmgr,&globalpzmctermmgr,-1);

	// ///////////////////////////////
	//
	// main function call
	//

	fractionFreeGaussDet_TA(fres);
	
	//
	//
	//////////////////////////////////
	
	if (fres.anzterms <= 25) {
		DynSlowString strres;
		fres.getStr(strres);
	
		fprintf(flog,"\nresultant: %s\n",strres.text);
		printf("\n\n|-> resultant = %s\n",strres.text);
	} else {
		printf("\n\n|-> resultant : see log file.\n");
	}
	
	printf("\b\ncreating list of coefficients ... ");
	ZMCpolynom *resMcoeff=getMcoeff(fres);
	printf("done\n");

	printf("storing coefficients ... ");

	char fn[4096];
	for(int32_t d=0;d<=fres.Mdegree;d++) {
		getFilenameCoeff(fn,d);
		FILE *f=fopen(fn,"wt");
		fprintf(f,"/A%i for %i-component of multibrot of degree %i\n",
			d,currentperiod,MULTIBROT);
		fprintf(f,"/per line one term: A*c^exponent\n");
		fprintf(f,"/nbr of terms: %i\n",resMcoeff[d].anzterms);
		for(int32_t t=0;t<resMcoeff[d].anzterms;t++) {
			resMcoeff[d].terms[t]->factor.ausgabe(f);
			fprintf(f,"*c^%i\n",resMcoeff[d].terms[t]->Cexponent);
		}
		
		fprintf(f,".\n");
		fclose(f);
	}
	
	printf("done\n");
	
	LOGMSG("\n----------------------------------------------------\n\n");
	LOGMSG2("A complex number c is in a period-%i component of\n",currentperiod);
	LOGMSG2("the multibrot of degree %i if the following has a\n",MULTIBROT);
	LOGMSG("complex solution m with ||m|| < 1\n");
	// get largest cabsolute coefficient number as an
	// estimate for the number type needed in subsequent 
	// analysis
	BigInt largest;
	largest.set_int64(0);
	
	for(int32_t i=0;i<fres.anzterms;i++) {
		if (bigintVgl_abs_AB(fres.terms[i]->factor,largest) > 0) {
			largest.copyFrom(fres.terms[i]->factor);
		}
	} // i
	
	int32_t l10d=largest.tendigitcount();
	LOGMSG2("  ( with largest |integer coefficient| around 10^%i )\n\n",
		l10d);

	for(int32_t i=fres.Mdegree;i>=0;i--) {
		if (i != fres.Mdegree) {
			LOGMSG("+");
		}
		
		if (i>1) {
			LOGMSG3("A%i*m^%i",i,i);
		} else if (i==1) {
			LOGMSG2("A%i*m",i);
		} else {
			LOGMSG("A0");
		}
	} // i
	
	LOGMSG(" = 0\n\n");
	
	DynSlowString one;
	for(int32_t i=fres.Mdegree;i>=0;i--) {
		LOGMSG2("  A%i = ",i);
		if (resMcoeff[i].anzterms < 25) {
			one.setEmpty();
			resMcoeff[i].getStr(one);
			LOGMSG2("%s\n",one.text);
		} else {
			char tmp[4096];
			LOGMSG2("[see file %s]\n",getFilenameCoeff(tmp,i));
		}
	} // i
	
	LOGMSG("\n|- all coefficients are stored on file");
	LOGMSG("\n----------------------------------------------------\n\n");

	int32_t t1=clock();
	int32_t dt=t1-t0;
	double duration=(double)(dt) / CLOCKS_PER_SEC;
	LOGMSG2("\nduration %.0lf sec\n",duration);

	if (flog) {
		fclose(flog);
		flog=NULL;
	}
	
	return 0;
}
