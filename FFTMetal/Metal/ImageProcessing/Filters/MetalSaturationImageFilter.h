//
//  MetalSaturationImageFilter.h
//  FFTMetal
//
//  Created by MACMaster on 12/13/15.
//  Copyright © 2015 Abstract Embedded. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MetalImageFilter.h"

@interface MetalSaturationImageFilter : MetalImageFilter

@property (nonatomic, assign) float saturationFactor;
+ (instancetype)filterWithSaturationFactor:(float)saturation context:(MetalGPGPUContext *)context;

@end
