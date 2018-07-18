//
//  FFTMath.h
//  FFTMetal
//
//  Created by MACMaster on 12/17/15.
//  Copyright © 2015 Abstract Embedded. All rights reserved.
//

#ifndef FFT_MATH_h
#define FFT_MATH_h

static inline unsigned int getNextPowerOfTwo(unsigned int x) {
    x += (x == 0);
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


#endif /* FFT_MATH_h */
