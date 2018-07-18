//
//  MetalGPGPUContext.m
//  FFTMetal
//
//  Created by MACMaster on 12/11/15.
//  Copyright © 2015 Abstract Embedded. All rights reserved.
//

#import "MetalGPGPUContext.h"
#import <Metal/Metal.h>

@interface MetalGPGPUContext()
{

}


@end


@implementation MetalGPGPUContext

@synthesize contextQueue = _contextQueue;
static void *metalContextQueueKey;

// Based on Colin Wheeler's example here: http://cocoasamurai.blogspot.com/2011/04/singletons-your-doing-them-wrong.html
+ (MetalGPGPUContext *)sharedImageProcessingContext
{
    static dispatch_once_t pred;
    static MetalGPGPUContext *sharedImageProcessingContext = nil;
    
    dispatch_once(&pred, ^{
        sharedImageProcessingContext = [[[self class] alloc] init];
    });
    return sharedImageProcessingContext;
}

+ (instancetype)newContext
{
    return [[self alloc] initWithDevice:nil];
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
{
    if ((self = [super init]))
    {
    
        _device = device ?: MTLCreateSystemDefaultDevice();
        _library = [_device newDefaultLibrary];
        _commandQueue = [_device newCommandQueue];
    
        //Note:  command queue probably replaces the opengl context queue
        metalContextQueueKey = &metalContextQueueKey;
        _contextQueue = dispatch_queue_create("com.abstractembedded.metalContextQueue", NULL);
        
#if OS_OBJECT_USE_OBJC
        dispatch_queue_set_specific(_contextQueue, metalContextQueueKey, (__bridge void *)self, NULL);
#endif
    
    }
    return self;
}

+ (void *)contextKey {
    return metalContextQueueKey;
}

+ (void)useImageProcessingContext
{
    [[MetalGPGPUContext sharedImageProcessingContext] useAsCurrentContext];
}

- (void)useAsCurrentContext;
{
    /*
    EAGLContext *imageProcessingContext = [self context];
    if ([EAGLContext currentContext] != imageProcessingContext)
    {
        [EAGLContext setCurrentContext:imageProcessingContext];
    }
    */
}

#pragma mark -- Metal Image Output (Analgous to GPUImageOutput)

+ (dispatch_queue_t)sharedContextQueue;
{
    return [[self sharedImageProcessingContext] contextQueue];
}


@end
