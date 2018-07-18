/*
 Copyright (C) 2015 Apple Inc. All Rights Reserved.
 See LICENSE.txt for this sample’s licensing information
 
 Abstract:
 View for Metal Sample Code. Manages screen drawable framebuffers and expects a delegate to repond to render commands to perform drawing.
 */

#import <QuartzCore/CAMetalLayer.h>
#import <Metal/Metal.h>
#import <UIKit/UIKit.h>

@protocol MetalRenderViewDelegate;

@interface MetalRenderView : UIView
@property (nonatomic, weak) IBOutlet id <MetalRenderViewDelegate> delegate;

// view has a handle to the metal device when created
@property (nonatomic, readonly) id <MTLDevice> device;

// the current drawable created within the view's CAMetalLayer
@property (nonatomic, readonly) id <CAMetalDrawable> currentDrawable;

// The current framebuffer can be read by delegate during -[MetalViewDelegate render:]
// This call may block until the framebuffer is available.
@property (nonatomic, readonly) MTLRenderPassDescriptor *renderPassDescriptor;

// set these pixel formats to have the main drawable framebuffer get created with depth and/or stencil attachments
@property (nonatomic) MTLPixelFormat depthPixelFormat;
@property (nonatomic) MTLPixelFormat stencilPixelFormat;
@property (nonatomic) NSUInteger     sampleCount;

@property (nonatomic, weak, readonly) CAMetalLayer *metalLayer;
@property (nonatomic) BOOL layerSizeDidUpdate;

// view controller will be call off the main thread
- (void)display;
-(void)destroyCurrentDrawable;  //destroy current drawable when frame is finished rendering

// release any color/depth/stencil resources. view controller will call when paused.
- (void)releaseTextures;

@end

// rendering delegate (App must implement a rendering delegate that responds to these messages
@protocol MetalRenderViewDelegate <NSObject>

@required
// called if the view changes orientation or size, renderer can precompute its view and projection matricies here for example
- (void)reshape:(MetalRenderView *)view;
// delegate should perform all rendering here
- (void)render:(MetalRenderView *)view;
@end

