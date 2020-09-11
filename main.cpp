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
	
*/

#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include "string.h"
#include "math.h"
#include "time.h"


// defines as compiler switches

// verifies certain claims during computation
#define _INVOKECLAIMVERIFICATIONS

// globals	
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

#define INC64(VV) \
{\
	if (VV > INT62MAX) {\
		LOGMSG("\nError. Inc var64 too large.\n");\
		exit(99);\
	}\
	VV++;\
}


// consts

const int32_t INIT_ZMCTERMS=128; // initial terms
const int32_t MAXDIMENSION=(int32_t)1 << 14;
const int32_t MAXDEGREE=(int32_t)1 << 13;
const int32_t MAXTERMSPERPOLYNOM=(int32_t)1 << 28;

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

// dynamical string
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

// big integer
#include "bigint.cpp"

// polynomial term: stores (positive) integer exponents of Z,M,C 
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

// memory manager for the polynomial terms themselves
struct ZMCtermMemoryManager {
	ZMCterm* current;
	int32_t allocatedIdx,freeFromIdx,allocatePerBlockIdx;
	PZMCterm ptr[MAXPTR];
	int32_t anzptr;
	
	ZMCtermMemoryManager();
	virtual ~ZMCtermMemoryManager();
	ZMCterm* getMemory(const int32_t);
	
	void freePhysically(void);
};

// memory manager for arrays of pointers to the terms
struct PZMCtermMemoryManager {
	PZMCterm* current;
	int32_t allocatedIdx,freeFromIdx,allocatePerBlockIdx;
	PPZMCterm ptr[MAXPTR];
	int32_t anzptr;
	
	PZMCtermMemoryManager();
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
	void checkDegree(void);
	int8_t isPositiveOne(void);
	void setToPositiveOne(void);
	void addTerm_FZMC(BigInt&,const int32_t,const int32_t,const int32_t);
	void addTerm_FZMC(const int64_t,const int32_t,const int32_t,const int32_t);
	void subTerm_FZMC(BigInt&,const int32_t,const int32_t,const int32_t);
	void addTerm_FZMC(ZMCterm&);
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
};

typedef MatrixPolynom *PMatrixPolynom;

// forward declarations
inline void termMul_TAB(ZMCterm&,ZMCterm&,ZMCterm&);
inline void polynomMul_TAB(ZMCpolynom&,ZMCpolynom&,ZMCpolynom&);
inline void polynomDiv_rf_TRAB(ZMCpolynom&,ZMCpolynom&,ZMCpolynom&,ZMCpolynom&,ZMCpolynom&,ZMCpolynom&,ZMCpolynom&,ZMCpolynom&);
inline void polynomComposition_at_Z_TNA(ZMCpolynom&,const int32_t,ZMCpolynom&);
inline void polynomPow_TNA(ZMCpolynom&,const int32_t,ZMCpolynom&);
inline void polynomDer_at_Z_TA(ZMCpolynom&,ZMCpolynom&);
/*
			
	during and after computation (polynoimMul, Div etc) is),
	it shall not be assumed that the order of terms of the
	argument polynomials be the same as before
	
*/

inline int32_t sum_int32t(const int32_t,const int32_t);
		
// globals
int32_t currentperiod=1;
int8_t MULTIBROT=2; 
ZMCpolynom basefkt; 
ZMCpolynom fstrictP,fderm; // memory manager will be set later

// memory managers
ZMCtermMemoryManager globalzmctermmgr;
PZMCtermMemoryManager globalpzmctermmgr;

// general
BigInt big1;
char fnbase[256];


// routines

// general
char* chomp(char* s) {
	if (!s) return 0;
	for(int32_t i=strlen(s);i>=0;i--) if (s[i]<32) s[i]=0; else break;
	return s;
}

inline void swapTermPtr(PZMCterm& a,PZMCterm& b) {
	PZMCterm c=a;
	a=b;
	b=c;
}

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

inline void swapInt(int32_t& a,int32_t& b) {
	int32_t c=a;
	a=b;
	b=c;
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
	// if one term with zero present => keep it
	if (anzterms <= 1) return;
	
	// it might happen that by adding/subtracting
	//  terms with factor 0 emerge
	// those will be removed from the term list
	int32_t idx=0;
	
	while (idx<anzterms) {
		// is [idx] a zero term
		if (terms[idx]->factor.vorz == 0) {
			// copy term from end of list
			if ((anzterms-1) != idx) {
				swapTermPtr(terms[idx],terms[anzterms-1]);
			} 
			
			anzterms--;
			// the pointer is still a validly allocated
			// memory part and can be reused if adding terms
			// requires more => not setting to NULL
			
			if (anzterms == 1) break; // not zero, the last 0 should be kept if present
			
			continue; // no idx increase
		}
		
		idx++;
		
		if (anzterms == 1) break;
		
	} // while
	
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
}

void ZMCpolynom::addPoly(ZMCpolynom& b) {
	for(int32_t i=0;i<b.anzterms;i++) {
		addTerm_FZMC(*b.terms[i]);
	}
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

	for(int32_t i=0;i<anzterms;i++) {
		if (
			(terms[i]->Zexponent == aZexp) &&
			(terms[i]->Mexponent == aMexp) && 
			(terms[i]->Cexponent == aCexp) 
		) {
			// already one present, add BigInt numbers
			terms[i]->factor.addTo(afactor);
			return; // done
		}
	} // i
	
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

		// copy used terms, oldalloc instead of anzterms to copy NIL pointers
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
		
		// old term-array will be destroyed when the corresponding
		// memory manager is freed
		
	}
	
	if (anzterms > (MAXTERMSPERPOLYNOM-16)) {
		LOGMSG("\nError. Memory. Too many terms in a polynomial.\n");
		exit(99);
	}
	
	if (!terms[anzterms]) {
		// get an actual object from the 2nd memory manager
		terms[anzterms]=tmgr->getMemory(1);
		if (!terms[anzterms]) {
			LOGMSG("\nError. Memory. polynom::add. tmgr cannot hand out more terms\n");
			exit(99);
		} 
	}
	
	// valid pointer
	terms[anzterms]->factor.copyFrom(afactor);
	terms[anzterms]->Zexponent=aZexp;
	terms[anzterms]->Mexponent=aMexp;
	terms[anzterms]->Cexponent=aCexp;
	anzterms++; // no overflow as checked above
	
	if (aZexp > Zdegree) Zdegree=aZexp;
	if (aMexp > Mdegree) Mdegree=aMexp;
	if (aCexp > Cdegree) Cdegree=aCexp;

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
	allocatedIdx=0;
	freeFromIdx=-1;
	double d=CHUNKSIZE; 
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
	
	return p;
}

void ZMCtermMemoryManager::freePhysically(void) {
	// put into state of constructor just called
	for(int32_t i=0;i<anzptr;i++) {
		if (ptr[i]) delete[] ptr[i];
	}
	
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
	allocatedIdx=0;
	freeFromIdx=-1;
	double d=CHUNKSIZE; 
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
	
	return p;
}

void PZMCtermMemoryManager::freePhysically(void) {
	// put into state of constructor just called
	for(int32_t i=0;i<anzptr;i++) {
		if (ptr[i]) delete[] ptr[i];
	}
	
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
	
		INFO:
	
		inefficient first implementation
		computes recursively all sub-strict periods
		even if they are necessary multiple times
		
		almost all computational time is spent in
		computing the determinant
	
	*/
	
	// computes basefkt[comp P-fold]-z) 
	// product strictP' with P' truely dividing P (1 included, but not P)
	
	ZMCpolynom fcompP(astrictp.tmgr,astrictp.ptmgr,-1),
		product(astrictp.tmgr,astrictp.ptmgr,-1),
		tmp(astrictp.tmgr,astrictp.ptmgr,-1),
		substrict(astrictp.tmgr,astrictp.ptmgr,-1);

	polynomComposition_at_Z_TNA(fcompP,period,basefkt);
	
	product.clearTerms();
	product.addTerm_FZMC(1,0,0,0); // positive one
	
	for(int32_t dividing=1;dividing<period;dividing++) {
		if ( (period % dividing) != 0) continue;
		
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
	// Check: fstrictP*product-fcompP = 0
	ZMCpolynom p1(astrictp.tmgr,astrictp.ptmgr,-1);
	polynomMul_TAB(p1,astrictp,product);
	p1.subPoly(fcompP);
	if (p1.isZero() <= 0) {
		LOGMSG("\nError. Implementation. Check fstrictP incorrect.\n");
		exit(99);
	}
	#endif
		
}

void setPeriod(const int32_t aper) {
	// compute composition f[z,aper]
	
	ZMCpolynom fcomp(&globalzmctermmgr,&globalpzmctermmgr,-1);
	polynomComposition_at_Z_TNA(fcomp,aper,basefkt);
	
	// derivative
	fderm.setMemoryManager(&globalzmctermmgr,&globalpzmctermmgr);
	polynomDer_at_Z_TA(fderm,fcomp);
	fderm.addTerm_FZMC(-1,0,1,0); // -m
	
	fderm.ausgabe(stdout);
	
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

// MatrixPolynom
void MatrixPolynom::load(
	const char* afn,
	const int32_t loady,
	const int32_t loadx
) {
	/*
		if loady>=0 & loadx >= 0 => only that element is to
		be actually loaded, the rest of the matrix is set to
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

void swapZMCpolynom(
	ZMCpolynom& a,
	ZMCpolynom& b
) {
	PZMCterm* tp=a.terms;
	a.terms=b.terms;
	b.terms=tp;
	
	swapInt(a.anzterms,b.anzterms);
	swapInt(a.Zdegree,b.Zdegree);
	swapInt(a.Mdegree,b.Mdegree);
	swapInt(a.Cdegree,b.Cdegree);
	swapInt(a.allocatedterms,b.allocatedterms);
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
			
			// here memory manager of result is used
			res.addTerm_FZMC(one);
		} // g
	} // f
	
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
	
	// try variables in a specific order
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
		
			while (1) {
			
				if (hlprest.isZero() > 0) {
					remainder.clearTerms();
					// dividing already set
					#ifdef _INVOKECLAIMVERIFICATIONS
					CHECKPOLYID
					#endif
					return; 
				}
			
				int32_t resthi=-1;
	
				// degree in hlprest is corrct
				for(int32_t i=0;i<hlprest.anzterms;i++) {
					// all occurin variables must have a degree
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
			
				if (resthi < 0) {
					break; // bidx, try next possible term
				}
				
				// idiv usable
				hlpm2.clearTerms();
				hlpm2.addTerm_FZMC(
					idiv,
					hlprest.terms[resthi]->Zexponent-B.terms[bidx]->Zexponent,
					hlprest.terms[resthi]->Mexponent-B.terms[bidx]->Mexponent,
					hlprest.terms[resthi]->Cexponent-B.terms[bidx]->Cexponent
				);
				polynomMul_TAB(hlpt2,hlpm2,B);
				hlprest.subPoly(hlpt2);
				hlprest.removeZeroTerms();
				hlprest.checkDegree();
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
		hybrid model
		
		matrix Mk: current matrix as in paper
		loaded from disc
		as polynomials therein are fixed, only the necessary
		memoy has been allocated
		
		matrix Mpreviousk: only diagonal element a(k-1)(k1,k1)
		is loaded from disc
		
		target matrix M(k+1) will be computed element
		by element and stored directly onto disc as its
		polynomial will only be used in the next step
		
		Upcoming: Reuse of subexpressions already multiplied
		(especially polynomial a(k)(k,k) times some bigints
	
	*/
	
	int32_t lastfullstoredk=0; // NOT -1
	// 0 will not be loaded anyways but constructed from
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

	MatrixPolynom* Mpreviousk=new MatrixPolynom;
	Mpreviousk->setDimension(
		&tmgr,
		&ptmgr,
		dimension); // constant0 polynom

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
	
	ZMCpolynom rem(&tmgr,&ptmgr,-1),
		t1(&tmgr,&ptmgr,-1),
		t2(&tmgr,&ptmgr,-1),
		erg(&tmgr,&ptmgr,-1),
		
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
		Mk->load(tmp,dimension-1,dimension-1); // just diagonal
		// resultant is lower right entry
		// dim <= 2^14
		fres.copyFrom(Mk->entryYX[(dimension-1)*dimension+(dimension-1)]);
		return;
	}

	printf("\ncomputing steps k=[%i..%i]",1+lastfullstoredk,dimension-1);
	
	for(int32_t k=lastfullstoredk;k<(dimension-1);k++) {

		// freeing memory from last round
		tmgr.freePhysically();
		ptmgr.freePhysically();
		
		// re-initialize as if just constructed
		
		Mk->setDimension(&tmgr,&ptmgr,dimension);
		Mpreviousk->setDimension(&tmgr,&ptmgr,dimension);
		
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

		if (k == 0) {
			// if only k=0 is stored, it is as if starting anew

			// use Sylvester matrix as parameter to get
			// the current matrix
			printf("initial step with Sylvester matrix ... ");

			printf("\n  creating Sylvester matrix ... ");
			printf("\n  getting z-coefficients ... ");
			ZMCpolynom *fZcoeff=getZcoeff(fstrictP);
			ZMCpolynom *derZcoeff=getZcoeff(fderm);
			printf("done\n");

			setSylvesterMatrix(
				*Mk,
				fstrictP,fderm,
				fZcoeff,derZcoeff
			);
			
			getMatrixFileName(tmp,0);
			Mk->save(tmp);
			// allocated using global term managaer, not here the local one
		} else {
			// load file[k] at Mk and file[k-1] at Mpreviousk
			printf("loading %i,%i ... ",
				k-1,k);
			getMatrixFileName(tmp,k);
			
			Mk->load( tmp , -1,-1 ); // full matrix to load

			getMatrixFileName(tmp,k-1);
			Mpreviousk->load( tmp , k-1,k-1 ); // only diagonal element
			
			// if files not present => already terminated
		}
		
		printf("computing ...");
		
		t1.reConstructor(&tmgr,&ptmgr);
		t2.reConstructor(&tmgr,&ptmgr);
		erg.reConstructor(&tmgr,&ptmgr);
		rem.reConstructor(&tmgr,&ptmgr);
		hlp1.reConstructor(&tmgr,&ptmgr);
		hlp2.reConstructor(&tmgr,&ptmgr);
		hlp3.reConstructor(&tmgr,&ptmgr);
		hlp4.reConstructor(&tmgr,&ptmgr);
		
		for(int32_t y=0;y<dimension;y++) {
			if ( (y & 0b111) == 0) {
				printf(" %i",dimension-y);
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
			
				// works also if it is finally a polynomial
				// with zero terms or the constant-0 1-term polynomial

				/*
								  a(k)kk*a(k)yx - a(k)yk*a(k)kx
					a(k+1)(y,x) = -----------------------------
									a(k-1)(k-1,k-1)
				
				*/
				
				// entry valid as indices and dimension <= 2^14
				#define ENTRYPTRKYX(KK,YY,XX) &KK->entryYX[(YY)*dimension+(XX)]
				
				ZMCpolynom* ak_kk=ENTRYPTRKYX(Mk,k,k);
				ZMCpolynom* ak_yx=ENTRYPTRKYX(Mk,y,x);
				ZMCpolynom* ak_yk=ENTRYPTRKYX(Mk,y,k);
				ZMCpolynom* ak_kx=ENTRYPTRKYX(Mk,k,x);
				ZMCpolynom* ak1_k1k1=NULL;
				
				if (k > 0) {
					ak1_k1k1=ENTRYPTRKYX(Mpreviousk,k-1,k-1);
					if (ak1_k1k1->isZero()) {
						LOGMSG("\nError. fractionFree division by zero\n");
						LOGMSG2("element k-1,k-1=%i\n",k-1);
						LOGMSG("\nProbably a principal minor has determinant 0 .\n");
						LOGMSG("Fraction-free Gauss elimination cannot be applied here.\n");
						LOGMSG("Manually exchanging rows and/or columns is recommended (but not implemented so far).");
						exit(99);
					}
				}
				
				t1.clearTerms();
				
				if (
					(ak_kk->isZero() <= 0) &&
					(ak_yx->isZero() <= 0)
				) {
					polynomMul_TAB(t1,*ak_kk,*ak_yx);
				}
				
				t2.clearTerms();
				if (
					(ak_yk->isZero() <= 0) &&
					(ak_kx->isZero() <= 0)
				) {
					polynomMul_TAB(t2,*ak_yk,*ak_kx);
					t1.subPoly(t2);
				}

				t1.removeZeroTerms();
				
				if (k > 0) { 
					// ak1_k1k1 is valid pointer
					if (ak1_k1k1->isPositiveOne() <= 0) {
						polynomDiv_rf_TRAB(
							erg,rem,
							t1,
							*ak1_k1k1,
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
						LOGMSG("\nError. Writing current matrix/3. Probably invalid file. Deleting recommended.\n");
						exit(99);
					}
					t1.save(fnextk);
					resultantwouldbe=RET1;
				}
				
			} // x
			
		} // y
		
		fclose(fnextk);
		
		printf(" saved");
		
	} // k
	
	// resultant now has been stored on file
	
	// physical copy - as by returning the lokal memory
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
			//printf("term[%i] ",i); current.terms[i].ausgabe(stdout);
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
		int32_t ze=(uint32_t)f.terms[i]->Zexponent;
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
	
	flog=fopen("mset-component-eq.log","at");
	fprintf(flog,"\n----------\nstart mset-component-eq\n\n");
	
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
	LOGMSG("\nINFO: Checks mul/div bigInt, polynomials invoked");
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
	basefkt.addTerm_FZMC(1,MULTIBROT,0,0); // z^MULTIBROT
	basefkt.addTerm_FZMC(1,0,0,1); // c

	// basefkt has been set correctly
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
	

	fractionFreeGaussDet_TA(fres);
	
	
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
	LOGMSG2("duration %.0lf sec\n",duration);

	if (flog) {
		fclose(flog);
		flog=NULL;
	}
	
	return 0;
}

