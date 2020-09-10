#ifndef _BIGINT
#define _BIGINT

/*

	big integer on base 10^9
	
	it shall not be assumed that calling a function
	like bigIntMul etc. keeps the arguments constant
	at all times (it does at return, but signs might
	have been changed during computation)
	
	if using this in parallel applications, care must
	be taken if passing the same physical object in different
	parallel threads
	
	Marc Meidlinger
	August-September 2020

*/

// consts

const int64_t UINT62MAX=((int64_t)1 << 62) - 1;
const int32_t INT31MAX=0b01111111111111111111111111111111;
const int32_t MAXBIGINTDIGITS=128;
// BASE*BASE must fit into 63 bits for multiplication purposes
const int32_t BIGINTBASE=1000000000; // 10^9 per 4-byte block

int32_t tenpower[] = {
	1,
	10,
	100,
	1000,
	10000,
	100000,
	1000000,
	10000000,
	100000000
};


// structs

struct BigInt {
	int8_t vorz; // +1, -1, 0
	int32_t highestusedidx;
	int32_t digits[MAXBIGINTDIGITS]; // the higher the index
	// the higher the order
	
	BigInt();
	BigInt(const BigInt&);
	
	void save(FILE*);
	void load(FILE*);
	void set_int64(const int64_t);
	void setTo10power(const int32_t);
	void addTo(BigInt&);
	void subTo(BigInt&);
	void getStr(DynSlowString&);
	void setToZero(void);
	void copyFrom(BigInt&);
	void shiftLeft(const int32_t);
	int32_t tendigitcount(void);
	void ausgabe(FILE*);
	void inc(void);
	void dec(void);

	BigInt& operator=(const BigInt);
};


// forward

void bigintDiv_TRAB(BigInt&,BigInt&,BigInt&,BigInt&);
void bigintMul_TAB(BigInt&,BigInt&,BigInt&);
//void bigintMul_digit_TDA(BigInt&,int64_t&,BigInt&);
void bigintMul_digit_TDA(BigInt&,int32_t&,BigInt&);
void bigintAdd_TAB(BigInt&,BigInt&,BigInt&);
void bigintAdd_abs_TAB(BigInt&,BigInt&,BigInt&);
void bigintSub_TAB(BigInt&,BigInt&,BigInt&);
void bigintSub_abs_TAB(BigInt&,BigInt&,BigInt&);
void bigintSub_abs_ovgl_TAB(BigInt&,BigInt&,BigInt&);
int8_t bigintVgl_AB(BigInt&,BigInt&);
int8_t bigintVgl_abs_AB(BigInt&,BigInt&);


// globals

inline int32_t sum_int32t(const int32_t,const int32_t);


// routines

// BigInt
void BigInt::shiftLeft(const int32_t toleft) {
	// in 10-base, not BIGINTBASE
	if (toleft == 0) return;
	if (toleft < 0) {
		LOGMSG("\nError. Implementation. ::shiftLeft woth negative argument\n");
		exit(99);
	}
	
	// now if toleft >= 9
	// shift in digits[] array positions to the left
	// and fill with 0
	int32_t shift=toleft;
	int32_t how9=0;
	if (shift == 9) { how9=1; shift=0; }
	else {
		how9=shift / 9; // downward rounding
		shift -= how9*9;
	}
	//printf("\ntoleft %i how9 %i shift %i\n",toleft,how9,shift);
	
	if ( (shift < 0) || (shift > 8) ) {
		// error
		LOGMSG("\nError. Implementation. bigint::ShiftLeft div 9\n");
		exit(99);
	}

	if ( (highestusedidx + how9) >= (MAXBIGINTDIGITS-2) ) {
		LOGMSG("\nError. bigInt::ShiftLeft overflow.\n");
		exit(99);
	}
	
	if (how9 > 0) {
		for(int32_t i=(highestusedidx+how9);i>=how9;i--) {
			digits[i]=digits[i-how9];
		}
		for(int32_t i=(how9-1);i>=0;i--) {
			digits[i]=0;
		}
		
		highestusedidx += how9;
	} // how9
	
	if (shift > 0) {
		// then if sth is left
		// do a multiplication by 10^appropriate
		BigInt tmp;
		bigintMul_digit_TDA(tmp,tenpower[shift],*this);
		copyFrom(tmp);
	}
}

void BigInt::load(FILE* f) {
	if (!f) {
		LOGMSG("\nError. Implementation. BigInt::load file\n");
		exit(99);
	}
	
	int32_t r=0;
	r += fread(&vorz,1,sizeof(vorz),f);
	r += fread(&highestusedidx,1,sizeof(highestusedidx),f);
	if (r != (sizeof(vorz)+sizeof(highestusedidx)) ) {
		LOGMSG("\nError. Reading bigint. Probably invalid matrix file. Deleting recommended.\n");
		exit(99);
	}
	
	for(int32_t i=0;i<=highestusedidx;i++) {
		#ifdef _READMATRIX64
		int64_t w;
		if (fread(&w,1,sizeof(w),f) != sizeof(w)) {
			LOGMSG("\nError. Reading bigint. Probably invalid matrix file. Deleting recommended.\n");
			exit(99);
		}
		if (w > INT31MAX) {
			LOGMSG2("\nError. Converting. BigInt:load: %I64d\n",w);
			exit(99);
		}
		digits[i]=w;
		#endif
		#ifndef _READMATRIX64
		if (fread(&digits[i],1,sizeof(digits[i]),f) != sizeof(digits[i])) {
			LOGMSG("\nError. Reading bigint. Probably invalid matrix file. Deleting recommended.\n");
			exit(99);
		}
		#endif
	} // i
}

void BigInt::save(FILE* f) {
	if (!f) {
		LOGMSG("\nError. Implementation. BigInt::save file\n");
		exit(99);
	}
	
	int32_t r=0;
	r += fwrite(&vorz,1,sizeof(vorz),f);
	r += fwrite(&highestusedidx,1,sizeof(highestusedidx),f);
	if (r != (sizeof(vorz)+sizeof(highestusedidx)) ) {
		LOGMSG("\nError. Writing bigint. Probably invalid matrix file. Deleting recommended.\n");
		exit(99);
	}
	for(int32_t i=0;i<=highestusedidx;i++) {
		if (fwrite(&digits[i],1,sizeof(digits[i]),f) != sizeof(digits[i])) {
			LOGMSG("\nError. Writing bigint. Probably invalid matrix file. Deleting recommended.\n");
			exit(99);
		}
	} // i
}

void BigInt::setTo10power(const int32_t pow) {

	if (BIGINTBASE != 1000000000) {
		LOGMSG("\nError. Implementation. BigInt::setTo10power needs base to be 10^9\n");
		exit(99);
	}

	highestusedidx=-1;
	digits[0]=0;
	vorz=1;
	
	int32_t p=pow;
	while (p >= 9) {
		highestusedidx++; 
		digits[highestusedidx]=0;
		p -= 9;
	} // while
	
	highestusedidx++;
	digits[highestusedidx]=tenpower[p];

}

void BigInt::dec(void) {
	if (vorz <= 0) {
		LOGMSG("\nError. BigInt. Decrement only works on positive values.\n");
		exit(99);
	}
	
	for(int32_t i=0;i<=highestusedidx;i++) {
		if (digits[i] > 0) {
			digits[i]--;
			break;
		}
		digits[i]=BIGINTBASE - 1;
	}
	
	while (highestusedidx > 0) {
		if (digits[highestusedidx] == 0) highestusedidx--;
		else break;
	}
	
	// is value zero
	vorz=0;
	for(int32_t i=0;i<=highestusedidx;i++) {
		if (digits[i] != 0) {
			vorz=1;
			break;
		}
	} // check
	
}

void BigInt::inc(void) {
	if (vorz < 0) {
		LOGMSG("\nError. BigInt. Increment only works on positive values.\n");
		exit(99);
	}
	
	vorz=1; // could have been 0 from the start

	int64_t carryover=1; // increment
	
	for(int32_t i=0;i<=highestusedidx;i++) {
		int64_t sum=carryover + (int64_t)digits[i];
		if (sum >= BIGINTBASE) {
			sum -= BIGINTBASE;
			if (sum >= BIGINTBASE) {
				LOGMSG("\nError. BigInt.inc\n");
				exit(99);
			}
			digits[i]=sum;
			carryover=1;
		} else {
			digits[i]=sum;
			return;
		}
	} // i
	
	if (carryover > 0) {
		if ( (highestusedidx+1) >= MAXBIGINTDIGITS) {
			LOGMSG("\nError. BigInt. Increment. Overflow.\n");
			exit(99);
		}
		highestusedidx++;
		digits[highestusedidx]=1;
	}
	
}

void BigInt::setToZero(void) {
	vorz=0;
	highestusedidx=0;
	digits[0]=0;
}

void BigInt::getStr(DynSlowString& erg) {
	erg.setEmpty();
	if (vorz == 0) {
		erg.add("+0");
		return;
	}
	
	char tmp[256];
	if (vorz < 0) erg.add("-");

	for(int32_t i=highestusedidx;i>=0;i--) {
		if (i != highestusedidx) {
			sprintf(tmp,"%09i",digits[i]);
		}
		else sprintf(tmp,"%i",digits[i]);
		erg.add(tmp);
	}
}

void BigInt::subTo(BigInt& av) {
	BigInt res;
	bigintSub_TAB(res,*this,av);
	copyFrom(res);
}

void BigInt::copyFrom(BigInt& av) {
	vorz=av.vorz;
	highestusedidx=av.highestusedidx;
	for(int32_t i=0;i<=highestusedidx;i++) {
		digits[i]=av.digits[i];
	}

}

void BigInt::addTo(BigInt& av) {
	BigInt res;
	bigintAdd_TAB(res,*this,av);
	copyFrom(res);
}

BigInt::BigInt() {
	// do nothing as fastest construction
}

BigInt::BigInt(const BigInt& b) {
	vorz=b.vorz;
	highestusedidx=b.highestusedidx;
	for(int32_t i=0;i<=highestusedidx;i++) {
		digits[i]=b.digits[i];
	}
}

void BigInt::set_int64(const int64_t aw) {
	if (aw == 0) {
		setToZero();
	} else {
		int64_t setv;
		if (aw > 0) {
			vorz=1;
			setv=aw;
		} else {
			vorz=-1;
			setv=-aw;
		}
		
		digits[0]=setv % BIGINTBASE;
		digits[1]=(setv-digits[0]) / BIGINTBASE;
		
		if (digits[1] >= BIGINTBASE) {
			LOGMSG("\nError. Setting BigInt too large\n");
			exit(99);
		}
		
		if (digits[1] != 0) highestusedidx=1;
		else highestusedidx=0;
	}
}

void bigintMul_digit_TDA(
	BigInt& res,
	int32_t& D,
	BigInt& A
) {
	// D is a BIGINTBASE-digit
	if ( (D < 0) || (D >= BIGINTBASE) ) {
		LOGMSG("\nError. Implementation. bigintmul_digit too large\n");
		exit(99);
	}
	
	if (D == 0) {
		res.setToZero();
		return;
	}
	
	// conservatively judged
	// current highstusedidx is smaller, so adding 1 does not overflow
	if ( (A.highestusedidx+1) >= MAXBIGINTDIGITS) {
		LOGMSG("\nError. Overflow. bigintMul\n");
		exit(99);
	}
	
	for(int32_t i=0;i<=(A.highestusedidx+1);i++) {
		res.digits[i]=0; // has to be set to zero as continues addition occurs later
	}
	
	for(int32_t idxa=0;idxa<=A.highestusedidx;idxa++) {
		int64_t w=(int64_t)A.digits[idxa] * D;
		// add w % BASE to res[i]
		// add w / BASE to res[i+1] and shift carry-overs
		
		// adds value 0 < DIG < BIGINTBASE at res[i] and
		// considers carry-over
		#define ADDAT(IDX,DIG) \
		{\
			int64_t carryover=0;\
			for(int32_t lokali=(IDX);lokali<=(A.highestusedidx+1);lokali++) {\
				int64_t sum=carryover + (int64_t)res.digits[lokali];\
				if (lokali == (IDX)) sum += (DIG);\
				\
				if (sum < BIGINTBASE) {\
					carryover=0;\
					res.digits[lokali]=sum;\
					break;\
				} else {\
					carryover=1;\
					sum -= (int64_t)BIGINTBASE;\
					if (sum >= BIGINTBASE) {\
						LOGMSG("\nError. Implementation. Overflow. AddAT/2\n");\
						exit(99);\
					}\
					res.digits[lokali]=sum;\
				}\
				\
				if (carryover == 0) break;\
			}\
			\
			if (carryover != 0) {\
				LOGMSG("\nError. Implementation. ADDAT. Overflow.\n");\
				exit(99);\
			}\
		}
		
		if (w < BIGINTBASE) {
			if (w != 0) {
				ADDAT(idxa,w)
			}
		} else {
			int64_t w1=w % BIGINTBASE;
			int64_t w2=(w-w1) / BIGINTBASE;
			if (w1 != 0) {
				ADDAT(idxa,w1)
			}
			if (w2 != 0) {
				ADDAT(idxa+1,w2)
			}
		}
	} // idxa
	
	for(int32_t i=(A.highestusedidx+1);i>=0;i--) {
		if (res.digits[i] > 0) {
			res.vorz=1;
			res.highestusedidx=i;
			return;
		}
	}
	
	// all digits 0
	res.vorz=0;
	res.highestusedidx=0;

}

void bigintMul_TAB(
	BigInt& res,
	BigInt& A,
	BigInt& B
) {
	// digit-wise
	if (
		(A.vorz == 0) ||
		(B.vorz == 0)
	) {
		res.setToZero();
		return;
	}
	
	res.vorz=A.vorz*B.vorz;
	int32_t w=A.highestusedidx+B.highestusedidx;
	if (w >= MAXBIGINTDIGITS) {
		// might work for some combinations of numbers at those
		// used indices, but it is easier just to increase the MAXDIGIT
		LOGMSG("\nError. bigintMul. Overflow.\n");
		exit(99);
	}
	
	for(int32_t i=0;i<=w;i++) res.digits[i]=0;
	res.highestusedidx=w;
	res.vorz=1;
	
	for(int32_t idxa=0;idxa<=A.highestusedidx;idxa++) {
		BigInt tmp;
		if (A.digits[idxa] == 0) continue;
		if (A.digits[idxa] == 1) {
			tmp.copyFrom(B);
		} else {
			bigintMul_digit_TDA(tmp,A.digits[idxa],B);
		}
		
		// shift result tmp i digits to the right
		if (idxa>0) {
			for(int32_t k=(tmp.highestusedidx+idxa);k>=idxa;k--) {
				tmp.digits[k]=tmp.digits[k-idxa];
			}
			for(int32_t k=(idxa-1);k>=0;k--) tmp.digits[k]=0;
		}
		tmp.highestusedidx += idxa;
		
		// add to res
		BigInt tmp2;
		bigintAdd_abs_TAB(tmp2,res,tmp);
		res=tmp2;
		
	} // idxa
	
	res.vorz=A.vorz*B.vorz;

}

void bigintAdd_TAB(
	BigInt& res,
	BigInt& A,
	BigInt& B
) {
	if (A.vorz==0) {
		if (B.vorz==0) {
			res.setToZero();
			return;
		} else {
			res.copyFrom(B);
			return;
		}
	} else 
	if (B.vorz==0) {
		res.copyFrom(A);
		// is not zero
		return;
	}
	
	// check sign
	if (A.vorz == B.vorz) {
		bigintAdd_abs_TAB(res,A,B);
		res.vorz=A.vorz;
		return; 
	} else {
		if ( (A.vorz > 0) && (B.vorz < 0) ) {
			// erg=a-|b|
			BigInt B2(B);
			B2.vorz=1;
			bigintSub_TAB(res,A,B2);
			return;
		} else {
			// a < 0, b > 0
			// res = -|a|+b = b-|a|
			BigInt A2(A);
			A2.vorz=1;
			bigintSub_TAB(res,B,A2);
			return;
		}
	}

	LOGMSG("\nError. Implementation. bigint::add.\n");
	exit(99);

}

void bigintSub_TAB(
	BigInt& res,
	BigInt& A,
	BigInt& B
) {
	if (A.vorz == 0) {
		// a=0 => a-b=-b
		if (B.vorz == 0) {
			// -b=0
			res.setToZero();
			return;
		} else {
			// -b
			res.copyFrom(B);
			res.vorz*=-1;
			return;
		}
	} else if (B.vorz == 0) {
		// a-b=a-0
		res.copyFrom(A);
		// A != 0
		return;
	}

	if (A.vorz>0) {
		// a>0: a-b=|a|-b
		if (B.vorz>0) {
			// a-b=|a|-|b|
			bigintSub_abs_TAB(res,A,B);
			return;
		} else {
			// a>0, b<0: a-b=a+|b|
			// a-b=|a|-(-|b|)=|a|+|b|
			bigintAdd_abs_TAB(res,A,B);
			return;
		}
	} else {
		// a<0, a-b=-|a|-b
		if (B.vorz>0) {
			// a-b=-|a|-|b|=-(|a|+|b|)
			bigintAdd_abs_TAB(res,A,B);
			res.vorz*=-1;
			return;
		} else {
			// a<0,b<0: a-b=-|a|-b=-|a|-(-|b|)=|b|-|a|
			bigintSub_abs_TAB(res,B,A);
			return;
		}
	}

	LOGMSG("\nError. Implementation. bigintSub\n");
	exit(99);
}

int8_t bigintVgl_AB(BigInt& A,BigInt& B) {
	if (A.vorz == 0) {
		if (B.vorz == 0) return 0;
		if (B.vorz < 0) return +1;
		if (B.vorz > 0) return -1;
	} else if (A.vorz > 0) {
		if (B.vorz <= 0) return +1;
		
		// both positive
		if (A.highestusedidx > B.highestusedidx) return +1;
		if (A.highestusedidx < B.highestusedidx) return -1;
		
		for(int32_t i=A.highestusedidx;i>=0;i--) {
			if (A.digits[i] > B.digits[i]) return +1;
			if (A.digits[i] < B.digits[i]) return -1;
		}
		
		return 0;
	} else if (A.vorz < 0) {
		if (B.vorz >= 0) return -1;

		// both negative
		if (A.highestusedidx > B.highestusedidx) return -1;
		if (A.highestusedidx < B.highestusedidx) return +1;
		
		for(int32_t i=A.highestusedidx;i>=0;i--) {
			if (A.digits[i] > B.digits[i]) return -1;
			if (A.digits[i] < B.digits[i]) return +1;
		}
		
		return 0;
	}
	
	LOGMSG("\nError. Implementation. bigIntVgl\n");
	exit(99);
}

int8_t bigintVgl_abs_AB(BigInt& A,BigInt& B) {
	// zero is considered, but sign not relevant otherwise
	if (A.vorz == 0) {
		if (B.vorz == 0) return 0;
		return -1; // 0 < |B|
	}
	
	if (B.vorz == 0) return +1; // |A|>0
	
	if (A.highestusedidx > B.highestusedidx) return +1;
	if (A.highestusedidx < B.highestusedidx) return -1;
		
	for(int32_t i=A.highestusedidx;i>=0;i--) {
		if (A.digits[i] > B.digits[i]) return +1;
		if (A.digits[i] < B.digits[i]) return -1;
	}
		
	return 0;
}

void bigintAdd_abs_TAB(
	BigInt& res,
	BigInt& A,
	BigInt&B
) {
	// the sign of A,B is NOT considered
	// only the value itself is used
	
	if (A.vorz==0) {
		if (B.vorz==0) res.setToZero();
		else res.copyFrom(B);
		return;
	} else if (B.vorz==0) {
		res.copyFrom(A);
		return;
	}
	
	// neither A nor B are zero
	// just add digits fromlowest to highest with carry-over
	int64_t carryover=0;
	
	int32_t m=A.highestusedidx;
	if (B.highestusedidx > m) m=B.highestusedidx;
	
	res.vorz=1; // will be positive as 0 can not happen
	res.highestusedidx=0;
	res.digits[0]=0;
	
	for(int32_t i=0 /* lowest order */;i<=m;i++) {
		// as digits are UINT32, and adding three numbers can
		// onlyhave at most a carry over of 2,the sum will fit
		// into int64
		int64_t sum=carryover;
		if (i <= A.highestusedidx) sum += (int64_t)A.digits[i];
		if (i <= B.highestusedidx) sum += (int64_t)B.digits[i];
		
		if (sum >= BIGINTBASE) {
			sum -= BIGINTBASE;
			if (sum >= BIGINTBASE) {
				LOGMSG("\nError. Implementation/1. bigint::add\n");
				exit(99);
			}
			carryover=1;
			res.digits[i]=sum;
		} else {
			carryover=0;
			res.digits[i]=sum;
		}
	} // i
	
	res.highestusedidx=m;
	
	if (carryover > 0) {
		res.highestusedidx=m+1;

		if (res.highestusedidx >= MAXBIGINTDIGITS) {
			LOGMSG("\nError. Overflow bigint::Add\n");
			exit(99);
		}
		res.digits[res.highestusedidx]=carryover;
	}
	
}

void bigintSub_abs_ovgl_TAB(
	BigInt& res,
	BigInt& A,
	BigInt& B
) {
	// computes |A|-|B| and it is assumed (and checked outside)
	// that |A| >  |B|
	// school method

	if (A.vorz==0) {
		if (B.vorz==0) {
			res.setToZero();
			return;
		}
		else {
			// Fehler
			LOGMSG("\nError. Implementation/1. sub_abs_vgl out of range\n");
			exit(99);
		}
	} else if (B.vorz==0) {
		res.copyFrom(A);
		return;
	}
	
	// signed here necessary as variable w below can get negative
	int64_t carryover=0;
	int32_t m=A.highestusedidx;
	if (B.highestusedidx > m) m=B.highestusedidx;
	
	for(int32_t i=0;i<=m;i++) {
		int64_t a,b;
		if (i <= A.highestusedidx) a=(int64_t)A.digits[i]; else a=0;
		if (i <= B.highestusedidx) b=(int64_t)B.digits[i]; else b=0;
		// as only 32 bits in digits and carry-over are used at max
		// there is no overflow or underflow by adding or subtracting
		// up to 3 numbers int int64_t
		int64_t w=a - (b + carryover);
		if (w < 0) {
			w += (int64_t)BIGINTBASE;
			if (w < 0) {
				LOGMSG("\nError. Implementation. subabs_ovgl/2\n");
				exit(99);
			}
			// carry over 
			carryover=1;
			res.digits[i]=w % BIGINTBASE;
		} else {
			if (w >= BIGINTBASE) {
				LOGMSG("\nError. Implementation. bigintsubb_ovgl_abs/2\n");
				exit(99);
			}
			
			res.digits[i]=w;
			carryover=0;
		}
	}
	res.highestusedidx=m;
	// leading zeros => trim
	while (res.highestusedidx > 0) {
		if (res.digits[res.highestusedidx] == 0) res.highestusedidx--;
		else break;
	}
	
	if (carryover != 0) {
		LOGMSG("\nError. Overflow bigint_sub_abs_ovgl\n");
		exit(99);
	}
	
	// as |A| > |B|, |A|-|B| > 0
	res.vorz=1;

}

void bigintSub_abs_TAB(
	BigInt& res,
	BigInt& A,
	BigInt& B
) {
	// only absolute value used, sign is NOT considered
	
	int32_t vgl=bigintVgl_abs_AB(A,B);
	
	if (vgl == 0) {
		res.setToZero();
		return;
	}
	
	if (vgl > 0) {
		// |a| > |b|
		bigintSub_abs_ovgl_TAB(res,A,B);
		res.vorz=1;
		return;
	} else {
		// |a| < |b|
		bigintSub_abs_ovgl_TAB(res,B,A);
		res.vorz=-1;
		return;
	}
	
	LOGMSG("\nError. Implementation. bigint::subAbs\n");
	exit(99);
}

BigInt& BigInt::operator=(const BigInt avalue) {
	if (this != &avalue) {
		vorz=avalue.vorz;
		highestusedidx=avalue.highestusedidx;
		for(int32_t i=0;i<=highestusedidx;i++) {
			digits[i]=avalue.digits[i];
		}
	}

	return *this;
}

void BigInt::ausgabe(FILE *f) {
	if (vorz==0) {
		fprintf(f,"+0");
		return;
	}
	
	if (vorz<0) fprintf(f,"-"); else fprintf(f,"+");
	
	for(int32_t i=highestusedidx;i>=0;i--) {
		if (i != highestusedidx) {
			fprintf(f,"%09i",digits[i]);
		} else fprintf(f,"%i",digits[i]);
	}
}

#define CHECKBIGINTDIV \
{\
	BigInt tmp1,tmp2;\
	bigintMul_TAB(tmp1,dividing,B);\
	bigintAdd_TAB(tmp2,tmp1,remainder);\
	if (bigintVgl_AB(tmp2,A) != 0) {\
		LOGMSG("\nError. BigInt Div incorrect test\n");\
		exit(99);\
	}\
	\
	bigintAdd_abs_TAB(tmp2,tmp1,remainder);\
	if (bigintVgl_abs_AB(tmp2,A) > 0) {\
		LOGMSG("\nError. BigInt Div incorrect test/2\n");\
		exit(99);\
	}\
}

void bigintDiv9_abs_TRAB(
	BigInt& dividing,
	BigInt& remainder,
	BigInt& A,
	BigInt& B
) {
	// compute |A|/|B|
	// it holds: 0 <= |A|/|B| < 10
	
	if (
		(A.vorz < 0) ||
		(B.vorz < 0)
	) {
		LOGMSG("\nError. bigintDiv9_abs_TRAB negative sign\n");
		exit(99);
	}
	
	if (A.vorz == 0) {
		dividing.setToZero();
		remainder.setToZero();
		return;
	}
	
	if (B.vorz == 0) {
		LOGMSG("\nError. bigintDiv9_abs_TRAB division by zero");
		exit(99);
	}
	
	// A,B are positive
	
	// naive implementation
	BigInt tmp;
	for(int32_t i=9;i>=0;i--) {
		bigintMul_digit_TDA(tmp,i,B);
		int32_t vgl=bigintVgl_abs_AB(tmp,A);
		if (vgl == 0) {
			dividing.set_int64(i);
			remainder.setToZero();
			#ifdef _INVOKECLAIMVERIFICATIONS
			CHECKBIGINTDIV
			#endif
			return;
		} else if (vgl < 0) {
			// as descending loop => i times
			dividing.set_int64(i);
			bigintSub_abs_TAB(remainder,A,tmp);
			#ifdef _INVOKECLAIMVERIFICATIONS
			CHECKBIGINTDIV
			#endif
			return;
		}
	} // i
	
	LOGMSG("\nError. Implementation. bigintDiv9_abs_TRAB. End of while loop\n");
	exit(99);
}


void bigintDiv_TRAB(
	BigInt& dividing,
	BigInt& remainder,
	BigInt& A,
	BigInt& B
) {
	
	// division is performed in |A| / |B|
	// and sign is adjusted to fit the following

	// at return, it holds: 
	//		dividing * B + remainder = A
	// and
	//		|dividing| * |B| + |remainder| <= |A|
	
	if (A.vorz == 0) {
		// it holds: 0
		dividing.set_int64(0);
		remainder.setToZero();
		#ifdef _INVOKECLAIMVERIFICATIONS
		CHECKBIGINTDIV
		#endif
		return;
	}
	
	if (B.vorz == 0) {
		LOGMSG("\nError. division by zero. BigInt.\n");
		exit(99);
	}
	
	if (
		(B.highestusedidx == 0) &&
		(B.digits[0] == 1)
	) {
		// it holds: dividing * 1 + remainder = A
		dividing.copyFrom(A);
		dividing.vorz *= B.vorz; // A / -1 = -A
		remainder.setToZero();
		#ifdef _INVOKECLAIMVERIFICATIONS
		CHECKBIGINTDIV
		#endif
		return;
	}
	
	int32_t vgl=bigintVgl_abs_AB(A,B);
	
	if (vgl == 0) {
		if (A.vorz == B.vorz) {
			// it holds: 1 * B + 0 = A
			remainder.setToZero();
			dividing.set_int64(1);
		} else if (A.vorz < 0) {
			// div * |B| + rem = -|A|
			remainder.setToZero();
			dividing.set_int64(-1);
		} else { // A > 0, B < 0
			// div * -|B| + rem = |A|
			remainder.setToZero();
			dividing.set_int64(-1);
		}
		
		#ifdef _INVOKECLAIMVERIFICATIONS
		CHECKBIGINTDIV
		#endif
		return;
		
	} // |A|=|B|
	
	if (vgl < 0) {
		// |A| < |B|
		// it holds: 0 * B + A = A no matter the sign
		remainder.copyFrom(A);
		dividing.set_int64(0);
		#ifdef _INVOKECLAIMVERIFICATIONS
		CHECKBIGINTDIV
		#endif
		return;
	}
	
	// |A| > |B|. Sign is not used during division
	
	BigInt rest;
	BigInt absA(A);
	BigInt absB(B);
	if (absA.vorz != 0) absA.vorz=1;
	if (absB.vorz != 0) absB.vorz=1;
	
	rest.copyFrom(absA);
	dividing.setToZero();
	
	int32_t bten=B.tendigitcount();
	
	while (1) {
		
		if (rest.vorz == 0) {
			// dividing is already correct
			remainder.setToZero();
			if (A.vorz > 0) {
				if (B.vorz > 0) {
					// all correct
				} else {
					// -|B|*dividing + 0 = |A|
					dividing.vorz *= -1;
				}
			} else {
				if (B.vorz > 0) {
					// |B|*dividing + 0 = -|A|
					dividing.vorz *= -1;
				} else {
					// -|B|*dividing + 0 = -|A|
				}
			} // A.vorz <= 0
			
			#ifdef _INVOKECLAIMVERIFICATIONS
			CHECKBIGINTDIV
			#endif
			return;
		}
		
		int32_t vgl=bigintVgl_abs_AB(rest,B);
		if (vgl < 0) {
			remainder.copyFrom(rest);
			
			if (A.vorz > 0) {
				if (B.vorz > 0) {
					// |B|*dividing + rem = |A|
					// all positive
				} else {
					// -|B|*dividing + rem = |A|
					dividing.vorz *= -1;
				}
			} else {
				if (B.vorz > 0) {
					// |B|*dividing + rem = -|A|
					dividing.vorz *= -1;
					remainder.vorz *= -1;
				} else {
					// -|B|*dividing + rem = -|A|
					remainder.vorz *= -1;
				}
			}
			
			#ifdef _INVOKECLAIMVERIFICATIONS
			CHECKBIGINTDIV
			#endif
			return;
			
		}
		
		// use a shifted version of |B|
		// compute |rest| / |shifted B|
		// is 0 <= .. < 10
		
		int32_t shiftdigits=rest.tendigitcount() - bten;
		BigInt Bshifted(absB);
		Bshifted.shiftLeft(shiftdigits);

		// now Bshifted could be larger than rest
		// then shift one to the right
		if (bigintVgl_abs_AB(Bshifted,rest) > 0) {
			shiftdigits--;
			Bshifted.copyFrom(absB);
			Bshifted.shiftLeft(shiftdigits);
		}
		
		if (shiftdigits < 0) {
			LOGMSG("\nError. bigintDiv/5 shifted too large\n");
			exit(99);
		}
		
		#ifdef _INVOKECLAIMVERIFICATIONS
		if (bigintVgl_abs_AB(Bshifted,A) > 0) {
			LOGMSG("\nError. bigintDiv/4 shifted too large\n");
			exit(99);
		}
		#endif

		BigInt div9,rem9;
		bigintDiv9_abs_TRAB(div9,rem9,rest,Bshifted);
		
		if (div9.vorz == 0) {
			// error
			LOGMSG("\nError. bigintDiv. B shifted divides 0 times\n");
			exit(99);
		}
		
		div9.shiftLeft(shiftdigits);
		dividing.addTo(div9);
		rest.copyFrom(rem9);
		
	} // while
	
}


int32_t BigInt::tendigitcount(void) {
	// how many digits in base 10 would the current number
	// need (value relevant for division)
	
	if (BIGINTBASE != 1000000000) {
		LOGMSG("\nError. Implementation. Bigint::tendigitcount needs base to be 10^9\n");
		exit(99);
	}
	
	int32_t ct=highestusedidx * 9;
	
	for(int32_t t=8;t>=0;t--) {
		if (digits[highestusedidx] >= tenpower[t]) {
			ct = sum_int32t(ct,t+1);
			break;
		}
	}
	
	return ct;
	
}

#endif
