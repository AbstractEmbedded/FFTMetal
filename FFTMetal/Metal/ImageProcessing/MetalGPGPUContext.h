//
//  MetalGPGPUContext.h
//  FFTMetal
//
//  Created by MACMaster on 12/11/15.
//  Copyright © 2015 Abstract Embedded. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
//#import <Metal/MetalLayer.h>
#import <QuartzCore/QuartzCore.h>

@protocol MTLDevice, MTLLibrary, MTLCommandQueue;
//@class MetalGPGPUContext;


@interface MetalGPGPUContext : NSObject


@property (strong) id<MTLDevice> device;
@property (strong) id<MTLLibrary> library;
@property (strong) id<MTLCommandQueue> commandQueue;
@property(readonly, nonatomic) dispatch_queue_t contextQueue;

+ (MetalGPGPUContext *)sharedImageProcessingContext;
+ (instancetype)newContext;

+ (void)useImageProcessingContext;

+ (void *)contextKey;
+ (dispatch_queue_t)sharedContextQueue;


@end
