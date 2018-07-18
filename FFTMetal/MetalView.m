//
//  MetalView.m
//  FFTMetal
//
//  Created by MACMaster on 12/11/15.
//  Copyright © 2015 Abstract Embedded. All rights reserved.
//

#import "MetalView.h"


@interface MetalView()
{

}

@property (nonatomic, strong) CADisplayLink *displayLink;

//Metal Stuff
@property (nonatomic, strong) CAMetalLayer *metalLayer;
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLRenderPipelineState> pipeline;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLBuffer> positionBuffer;
@property (nonatomic, strong) id<MTLBuffer> colorBuffer;

@end

@implementation MetalView

+ (id)layerClass
{
    return [CAMetalLayer class];
}

- (void)didMoveToWindow
{
    [self redraw];
}

- (void)didMoveToSuperview
{
    [super didMoveToSuperview];
    if (self.superview)
    {
        self.displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(displayLinkDidFire:)];
        [self.displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];
    }
    else
    {
        [self.displayLink invalidate];
        self.displayLink = nil;
    }
}

- (void)displayLinkDidFire:(CADisplayLink *)displayLink
{
    [self redraw];
}

-(void)redraw
{
    id<CAMetalDrawable> drawable = [self.metalLayer nextDrawable];
    id<MTLTexture> framebufferTexture = drawable.texture;
    
    MTLRenderPassDescriptor *renderPass = [MTLRenderPassDescriptor renderPassDescriptor];
    renderPass.colorAttachments[0].texture = framebufferTexture;
    renderPass.colorAttachments[0].clearColor = MTLClearColorMake(0.5, 0.5, 0.5, 1);
    renderPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    renderPass.colorAttachments[0].loadAction = MTLLoadActionClear;
    
    id<MTLCommandBuffer> commandBuffer = [self.commandQueue commandBuffer];
    
    id<MTLRenderCommandEncoder> commandEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPass];
    [commandEncoder setRenderPipelineState:self.pipeline];
    [commandEncoder setVertexBuffer:self.positionBuffer offset:0 atIndex:0 ];
    [commandEncoder setVertexBuffer:self.colorBuffer offset:0 atIndex:1 ];
    [commandEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:1];
    [commandEncoder endEncoding];
    
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];



}

- (instancetype)initWithFrame:(CGRect)frame
{

    
    if ((self = [super initWithFrame:frame]))
    {
    
        //setup Metal Rendering Pipeline
        [self buildDevice];
        [self buildVertexBuffers];
        [self buildPipeline];
    

    }
    
    return self;
}

-(void)buildDevice
{
    _metalLayer = (CAMetalLayer *)[self layer];
    _device = MTLCreateSystemDefaultDevice();
    _metalLayer.device = _device;
    _metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
}

- (void)buildVertexBuffers
{
    static const float positions[] =
    {
        0.0,  0.5, 0, 1,
        -0.5, -0.5, 0, 1,
        0.5, -0.5, 0, 1,
    };
    
    static const float colors[] =
    {
        1, 0, 0, 1,
        0, 1, 0, 1,
        0, 0, 1, 1,
    };
    
    self.positionBuffer = [self.device newBufferWithBytes:positions
                                                   length:sizeof(positions)
                                                  options:MTLResourceOptionCPUCacheModeDefault];
    self.colorBuffer = [self.device newBufferWithBytes:colors
                                                length:sizeof(colors)
                                               options:MTLResourceOptionCPUCacheModeDefault];
}

- (void)buildPipeline
{
    id<MTLLibrary> library = [self.device newDefaultLibrary];
    
    id<MTLFunction> vertexFunc = [library newFunctionWithName:@"vertex_main"];
    id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragment_main"];
    
    MTLRenderPipelineDescriptor *pipelineDescriptor = [MTLRenderPipelineDescriptor new];
    pipelineDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pipelineDescriptor.vertexFunction = vertexFunc;
    pipelineDescriptor.fragmentFunction = fragmentFunc;
    
    NSError *error = nil;
    self.pipeline = [self.device newRenderPipelineStateWithDescriptor:pipelineDescriptor
                                                                error:&error];
    
    if (!self.pipeline)
    {
        NSLog(@"Error occurred when creating render pipeline state: %@", error);
    }
    
    self.commandQueue = [self.device newCommandQueue];
}


/*
// Only override drawRect: if you perform custom drawing.
// An empty implementation adversely affects performance during animation.
- (void)drawRect:(CGRect)rect {
    // Drawing code
}
*/

@end
