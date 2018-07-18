//
//  LiveCameraViewController.mm
//
//  Created by Joe Moulton on 9/13/15.
//  Copyright Abstract Embedded.  All Rights Reserved.
//

#define kCameraFlashDefaultsKey @"CustomCameraFlashIsOn"
#define kCameraTorchDefaultsKey @"CustomCameraTorchIsOn"


#import "LiveCameraViewController.h"
#import <AVFoundation/AVFoundation.h>

//needed for kCGImagePropertyExifDictionary
#import <ImageIO/ImageIO.h>


//Metal Includes
#import <simd/simd.h>
#import <CoreVideo/CVMetalTextureCache.h>

//GPGPU Image Processing
#import "MetalGPGPUContext.h"
#import "MetalImageFilter.h"
#import "MetalSaturationImageFilter.h"
#import "MetalGaussianBlurFilter.h"
#import "MetalFFTImageFilter.h"
#import "MetalImageOutput.h"
#import "UIImage+MetalTextureUtilities.h"

//3D Pipeline Rendering
#import "MetalSharedTypes.h"
#import "AbstractMetalTexture.h"
#import "MetalPVRTexture.h"
#import "SIMDMatrixMath.h"
#import "MetalRenderView.h"



//A UIImage class extension for converting between iOS native image contexts and cv::Mat
//#import "UIImage+OpenCV.h"

static const long kMaxBufferBytesPerFrame = 1024*1024;
static const long kInFlightCommandBuffers = 3;

static const float kFOVY          = 65.0f;
static const simd::float3 kEye    = {0.0f, 0.0f, 0.0f};
static const simd::float3 kCenter = {0.0f, 0.0f, 1.0f};
static const simd::float3 kUp     = {0.0f, 1.0f, 0.0f};

static const float kQuadWidth  = 1.0f;
static const float kQuadHeight = 1.0f;
static const float kQuadDepth  = 0.0f;  //for video outside of a 3d projection, set depth to 0

/*
static const float quad[] =
{
    // ordered facet vertices (xyz), normal (xyz), texCoord (uv)
    kQuadWidth, -kQuadHeight, -kQuadDepth,  0.0,  0.0, -1.0,  1.0, 1.0,
    -kQuadWidth, -kQuadHeight, -kQuadDepth, 0.0,  0.0, -1.0,  0.0, 1.0,
    -kQuadWidth, kQuadHeight, -kQuadDepth,  0.0,  0.0, -1.0,  0.0, 0.0,
    kQuadWidth, kQuadHeight, -kQuadDepth,   0.0,  0.0, -1.0,  1.0, 0.0,
    kQuadWidth, -kQuadHeight, -kQuadDepth,  0.0,  0.0, -1.0,  1.0, 1.0,
    -kQuadWidth, kQuadHeight, -kQuadDepth,  0.0,  0.0, -1.0,  0.0, 0.0
};
*/

static const float quad[] =
/*
{ -1.0f,  -1.0f, 0.0f,  0.0,  0.0, -1.0, 0.0f, 0.0f,
  1.0f,  -1.0f, 0.0f,   0.0,  0.0, -1.0, 1.0f, 0.0f,
  -1.0f,   1.0f, 0.0f,  0.0,  0.0, -1.0, 0.0f, 1.0f,
   1.0f,  -1.0f, 0.0f,  0.0,  0.0, -1.0, 1.0f, 0.0f,
  -1.0f,   1.0f, 0.0f,  0.0,  0.0, -1.0, 0.0f, 1.0f,
   1.0f,   1.0f, 0.0f,  0.0,  0.0, -1.0, 1.0f, 1.0f
};
*/
{
    // ordered facet vertices (xyz), normal (xyz), texCoord (uv)
    kQuadWidth, -kQuadHeight, -kQuadDepth,  0.0,  0.0, -1.0,  1.0, 0.0,
    -kQuadWidth, -kQuadHeight, -kQuadDepth, 0.0,  0.0, -1.0,  0.0, 0.0,
    -kQuadWidth, kQuadHeight, -kQuadDepth,  0.0,  0.0, -1.0,  0.0, 1.0,
    kQuadWidth, kQuadHeight, -kQuadDepth,   0.0,  0.0, -1.0,  1.0, 1.0,
    kQuadWidth, -kQuadHeight, -kQuadDepth,  0.0,  0.0, -1.0,  1.0, 0.0,
    -kQuadWidth, kQuadHeight, -kQuadDepth,  0.0,  0.0, -1.0,  0.0, 1.0
};

static const simd::float4 cubeVertexData[] =
{
    // posx
    { -1.0f,  1.0f,  1.0f, 1.0f },
    { -1.0f, -1.0f,  1.0f, 1.0f },
    { -1.0f,  1.0f, -1.0f, 1.0f },
    { -1.0f, -1.0f, -1.0f, 1.0f },
    
    // negz
    { -1.0f,  1.0f, -1.0f, 1.0f },
    { -1.0f, -1.0f, -1.0f, 1.0f },
    { 1.0f,  1.0f, -1.0f, 1.0f },
    { 1.0f, -1.0f, -1.0f, 1.0f },
    
    // negx
    { 1.0f,  1.0f, -1.0f, 1.0f },
    { 1.0f, -1.0f, -1.0f, 1.0f },
    { 1.0f,  1.0f,  1.0f, 1.0f },
    { 1.0f, -1.0f,  1.0f, 1.0f },
    
    // posz
    { 1.0f,  1.0f,  1.0f, 1.0f },
    { 1.0f, -1.0f,  1.0f, 1.0f },
    { -1.0f,  1.0f,  1.0f, 1.0f },
    { -1.0f, -1.0f,  1.0f, 1.0f },
    
    // posy
    { 1.0f,  1.0f, -1.0f, 1.0f },
    { 1.0f,  1.0f,  1.0f, 1.0f },
    { -1.0f,  1.0f, -1.0f, 1.0f },
    { -1.0f,  1.0f,  1.0f, 1.0f },
    
    // negy
    { 1.0f, -1.0f,  1.0f, 1.0f },
    { 1.0f, -1.0f, -1.0f, 1.0f },
    { -1.0f, -1.0f,  1.0f, 1.0f },
    { -1.0f, -1.0f, -1.0f, 1.0f },
};


@interface LiveCameraViewController ()
{
    UIDeviceOrientation currentDeviceOrientation;
    
    AVCaptureVideoOrientation defaultAVCaptureVideoOrientation;
    AVCaptureVideoOrientation currentAVCaptureVideoOrientation;

    //AVCaptureSession * mCaptureSession;
    BOOL flashIsOn;
    BOOL torchIsOn;
    dispatch_queue_t videoDataOutputQueue;
    
    //the desired resolution and frame rate for receiving camera video frames
    //must match a device resolution/frame rate profile
    int captureWidth;
    int captureHeight;
    int captureRate;

    
    
    //width and height in pixels of the CALayer we will render video frames too
    CGSize previewLayerSize;
    CGSize customPreviewLayerSize;
    CGSize videoLayerSize;
    //determines how the video render layer maintains aspect of the captured image(s) it is presenting
    NSString * videoGravity;
    
    //for measuring latency
    double prevFrameTime;
    UInt64 prevTimeStamp;
    
    //store the previous frame's image buffer for processing phase correlation on every two consecutive frames
    CMSampleBufferRef prevSampleBuffer;
    
    //Metal Process Video Frames Properties
    id <MTLDevice> _device;
    id <MTLCommandQueue> _commandQueue;
    id <MTLLibrary> _defaultLibrary;
    
    dispatch_semaphore_t _inflight_semaphore;
    id <MTLBuffer> _dynamicUniformBuffer[kInFlightCommandBuffers];
    
    // render stage
    id <MTLRenderPipelineState> _pipelineState;
    id <MTLBuffer> _vertexBuffer;
    id <MTLDepthStencilState> _depthState;
    
    // this value will cycle from 0 to g_max_inflight_buffers whenever a display completes ensuring renderer clients
    // can synchronize between g_max_inflight_buffers count buffers, and thus avoiding a constant buffer from being overwritten between draws
    NSUInteger _constantDataBufferIndex;
    
    // global transform data
    simd::float4x4 _projectionMatrix;
    simd::float4x4 _viewMatrix;
    float _rotation;
    float _skyboxRotation;
    
    // skybox
    AbstractMetalTexture *_skyboxTex;
    id <MTLRenderPipelineState> _skyboxPipelineState;
    id <MTLBuffer> _skyboxVertexBuffer;
    
    // texturedQuad
    AbstractMetalTexture *_quadTex;
    id <MTLRenderPipelineState> _quadPipelineState;
    id <MTLBuffer> _quadVertexBuffer;
    id <MTLBuffer> _quadNormalBuffer;
    id <MTLBuffer> _quadTexCoordBuffer;
    
    // Video texture
    AVCaptureSession *_captureSession;
    CVMetalTextureCacheRef _videoTextureCache;
    id <MTLTexture> _videoTexture[3];
    id <MTLBuffer> _videoBuffer[3];


    //Metal Render View
    CADisplayLink *_displayLink;
    // boolean to determine if the first draw has occured
    BOOL _firstDrawOccurred;
    CFTimeInterval _timeSinceLastDraw;
    CFTimeInterval _timeSinceLastDrawPreviousTime;
    // pause/resume
    BOOL _renderLoopPaused;

    id <MTLTexture> _testPatternTexture;
    //cv::Mat * hannWindow;
    //cv::Mat * fft1;
    //cv::Mat * fft2;
}

//Statistics overlay views
@property (nonatomic, retain) UILabel * frameRateLabel;
@property (nonatomic, retain) UILabel * phaseCorrelationLabel;

//AVCaptureVideo class properties

@property (nonatomic) BOOL isUsingFrontFacingCamera;
@property (nonatomic) BOOL setFrameRateViaCaptureDevice;

@property (strong, nonatomic) AVCaptureSession *videoCaptureSession;

//two delegates for receiving frames and detecting features are available
//1 for capturing the raw pixel data
//and 1 for capturing and drawing metadata output
@property (strong, nonatomic) AVCaptureVideoDataOutput *videoDataOutput;
@property (strong, nonatomic) AVCaptureMetadataOutput *metadataOutput;

@property (strong, nonatomic) AVCaptureDevice * captureDevice; //the device camera/sensor that we are currently receiving from

//choose one of these layers shows the video from the camera
@property (nonatomic, retain) CALayer *customPreviewLayer;
@property (strong, nonatomic) AVCaptureVideoPreviewLayer *previewLayer;

//this layer draws on top of the video from the camera
@property (nonatomic, retain) CALayer * drawLayer;


@property (strong, nonatomic) UIToolbar *cameraToolbar;
@property (strong, nonatomic) UIBarButtonItem *pictureButton;
@property (strong, nonatomic) UIBarButtonItem *flashButton;
@property (strong, nonatomic) UIView *cameraPictureTakenFlash;

@property (nonatomic, retain) CIDetector * featDetector;

@property (nonatomic, retain) UISlider * /*CustomSlider */ torchSlider;


//Metal GPGPU Rendering Pipeline
@property (nonatomic, strong) MetalGPGPUContext * context;
//@property (nonatomic, strong) id<MetalTextureProvider> imageProvider;     //instead let this class be its own image provider, since images are generated from camera
@property (nonatomic, strong) MetalSaturationImageFilter *desaturateFilter;
@property (nonatomic, strong) MetalGaussianBlurFilter *blurFilter;
@property (nonatomic, strong) MetalFFTImageFilter *fftFilter;


@property (nonatomic, strong) dispatch_queue_t metalRenderingQueue;
@property (atomic, assign) uint64_t jobIndex;

@property (nonatomic, retain) MetalRenderView * metalRenderView;


@end

@implementation LiveCameraViewController

@synthesize texture = _texture;

-(uint8_t *)dataForImage:(UIImage *)image
{
    CGImageRef imageRef = [image CGImage];
    
    // Create a suitable bitmap context for extracting the bits of the image
    const NSUInteger width = CGImageGetWidth(imageRef);
    const NSUInteger height = CGImageGetHeight(imageRef);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    uint8_t *rawData = (uint8_t *)calloc(height * width * 4, sizeof(uint8_t));
    const NSUInteger bytesPerPixel = 4;
    const NSUInteger bytesPerRow = bytesPerPixel * width;
    const NSUInteger bitsPerComponent = 8;
    CGContextRef context = CGBitmapContextCreate(rawData, width, height,
                                                 bitsPerComponent, bytesPerRow, colorSpace,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(colorSpace);
    
    CGContextTranslateCTM(context, 0, height);
    CGContextScaleCTM(context, 1, -1);
    
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), imageRef);
    CGContextRelease(context);
    
    return rawData;
}
//helper for loading test pattern to mtl texture, later this will be provided as part of the api
-(id<MTLTexture>)texture2DWithImageNamed:(NSString *)imageName device:(id<MTLDevice>)device
{
    UIImage *image = [UIImage imageNamed:imageName];
    CGSize imageSize = CGSizeMake(image.size.width * image.scale, image.size.height * image.scale);
    const NSUInteger bytesPerPixel = 4;
    const NSUInteger bytesPerRow = bytesPerPixel * imageSize.width;
    uint8_t *imageData = [self dataForImage:image];
    
    MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                                 width:imageSize.width
                                                                                                height:imageSize.height
                                                                                             mipmapped:NO];
    id<MTLTexture> texture = [device newTextureWithDescriptor:textureDescriptor];
    
    MTLRegion region = MTLRegionMake2D(0, 0, imageSize.width, imageSize.height);
    [texture replaceRegion:region mipmapLevel:0 withBytes:imageData bytesPerRow:bytesPerRow];
    
    free(imageData);
    
    return texture;
}

- (BOOL)shouldAutorotate
{
    return NO;
}
/*
- (NSUInteger)supportedInterfaceOrientations
{
    //if (self.isLandscapeOK) {
        // for iPhone, you could also return UIInterfaceOrientationMaskAllButUpsideDown
    //    return UIInterfaceOrientationMaskAll;
    //}
    return UIInterfaceOrientationMaskPortrait;
}

- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation
{
    //}
    return UIInterfaceOrientationPortrait;
}
*/

- (void)dealloc {
    
    NSLog(@"LiveCameraViewController DEALLOC");
    //remove notificaiton listerners
    [[NSNotificationCenter defaultCenter] removeObserver: self
                                                    name: UIApplicationDidEnterBackgroundNotification
                                                  object: nil];
    
    [[NSNotificationCenter defaultCenter] removeObserver: self
                                                    name: UIApplicationWillEnterForegroundNotification
                                                  object: nil];
    
    
    //shut down the render loop
    if(_displayLink)
    {
        [self stopRenderLoop];
    }
    
    //shut down the video capture callback
    [[self videoCaptureSession] stopRunning];
    
    _previewLayer = nil;
    _videoCaptureSession = nil;

}


- (BOOL)prefersStatusBarHidden {
    return YES;
}

- (UIStatusBarStyle)preferredStatusBarStyle
{
    return UIStatusBarStyleLightContent;
}

/*
-(UIInterfaceOrientationMask)supportedInterfaceOrientations{
    return UIInterfaceOrientationMaskPortrait | UIInterfaceOrientationMaskPortraitUpsideDown;
}

- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation{
    return UIInterfaceOrientationPortrait;
}
*/

-(void)viewDidAppear:(BOOL)animated
{

    NSLog(@"view did appear");
    NSLog(@"width:  %d", (int)self.view.frame.size.width);
    NSLog(@"height: %d", (int)self.view.frame.size.height);
    //unpause live video processing
    if( self.videoDataOutput )
        [[self.videoDataOutput connectionWithMediaType:AVMediaTypeVideo] setEnabled:YES];
    
    //start running the video capture session to render and process video
    //when this view controller appears (is at the top of the navigation stack)
     [self.videoCaptureSession startRunning];

    [self updateTorch];
    
    [[NSNotificationCenter defaultCenter] addObserver:self  selector:@selector(updateDeviceOrientation:)    name:UIDeviceOrientationDidChangeNotification  object:nil];
    //update the device orientation now to make sure the view has the correct orientation
    [self updateDeviceOrientation:nil];

}

- (void)updateDeviceOrientation:(NSNotification *)notification{
    
    UIDeviceOrientation orientation = [[UIDevice currentDevice] orientation];
    
    /*
    if( orientation == UIDeviceOrientationFaceUp || orientation == UIDeviceOrientationFaceDown )
    {
        if( currentDeviceOrientation )
            return;
        else
            currentDeviceOrientation = UIDeviceOrientationPortrait;
    }
    */

    currentDeviceOrientation = orientation;
    //defaultAVCaptureVideoOrientation = [self translateDeviceOrientationToCaptureOrientation:currentDeviceOrientation];
    

    if( [self shouldAutorotate] )
    {
    [self updateVideoCaptureOrientation:orientation];
    

    if( self.previewLayer )
        [self layoutPreviewLayer];
    if( self.customPreviewLayer )
        [self layoutCustomPreviewLayer];
    if( self.metalRenderView )
        [self layoutMetalRenderView];

    }
}

-(void)customizeNavBar
{
    
    
    [self.navigationController.navigationBar setBackgroundImage:[UIImage new]
                                                  forBarMetrics:UIBarMetricsDefault]; //UIImageNamed:@"transparent.png"
    self.navigationController.navigationBar.shadowImage = [UIImage new];////UIImageNamed:@"transparent.png"
    self.navigationController.navigationBar.translucent = YES;
    self.navigationController.view.backgroundColor = [UIColor clearColor];
    
    self.navigationController.navigationBar.barTintColor = [UIColor whiteColor];
    self.navigationController.navigationBar.tintColor = [UIColor whiteColor];
    
    [self.navigationController.navigationBar setTitleTextAttributes:
     @{NSForegroundColorAttributeName:[UIColor whiteColor]}];
    
     [[self navigationController] setNavigationBarHidden:NO animated:NO];
}

-(AVCaptureVideoOrientation)updateVideoCaptureOrientation:(UIDeviceOrientation)orientation
{

    AVCaptureVideoOrientation captureOrientation;

    if(   orientation ==  UIDeviceOrientationUnknown )
        captureOrientation = AVCaptureVideoOrientationPortrait;
    else if( orientation == UIDeviceOrientationPortrait )
        captureOrientation = AVCaptureVideoOrientationPortrait;
    else if( orientation == UIDeviceOrientationPortraitUpsideDown )
        captureOrientation = AVCaptureVideoOrientationPortraitUpsideDown;
    else if( orientation == UIDeviceOrientationLandscapeLeft )
        captureOrientation = AVCaptureVideoOrientationLandscapeRight;
    else if ( orientation == UIDeviceOrientationLandscapeRight )
        captureOrientation = AVCaptureVideoOrientationLandscapeLeft;
    //else if ( orientation == UIDeviceOrientationFaceUp || orientation == UIDeviceOrientationFaceDown )
    //    return [self translateDeviceOrientationToCaptureOrientation:currentDeviceOrientation];
    
    if( self.videoDataOutput )
    {
        AVCaptureConnection * connection = [self.videoDataOutput connectionWithMediaType:AVMediaTypeVideo];
        
        //defaultAVCaptureVideoOrientation = captureOrientation;
        currentAVCaptureVideoOrientation = captureOrientation;
        //set the video to give the output buffer in portrait orientation rather than landscape
        [connection setVideoOrientation:currentAVCaptureVideoOrientation];
    }
    //[connection setEnabled:YES];

    
    return captureOrientation;
}

-(void)loadTestPatternImage
{
    _testPatternTexture = [self texture2DWithImageNamed:@"TestPattern1920x1080.png" device:_device];


    UIImage * testPattern = [UIImage imageNamed:@"TestPattern1920x1080.png"];
    UIImageView * testPatternImageView = [[UIImageView alloc] initWithImage:testPattern];
    testPatternImageView.center = CGPointMake(self.view.frame.size.width/2.0, self.view.frame.size.height/2.0);
    
    [self.view addSubview:testPatternImageView];
    
    [self.view bringSubviewToFront:_metalRenderView];
    //load to metal texture
}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    
    //[self.navigationController setNavigationBarHidden:YES];
    //[self.view setBackgroundColor:[UIColor blackColor]];
    
    //initialize your resolution, frame rate and orientation settings
    //by hardcoding the convenience properties here
    
    
    captureWidth = 1920;
    captureHeight = 1080;
    captureRate = 60;
    
    previewLayerSize.width  = captureWidth;//self.view.frame.size.width;
    previewLayerSize.height = captureHeight;//self.view.frame.size.height;
    
    customPreviewLayerSize.width = 352;//512;
    customPreviewLayerSize.height = 288;//512;
    
    videoLayerSize.width = self.view.frame.size.height;
    videoLayerSize.height = self.view.frame.size.width;
    
    //since we are only using portrait orientation for now, set it explictly here
    //but we can receive orientation update events if we want
    currentDeviceOrientation = [[UIDevice currentDevice] orientation];
    defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;//[self updateVideoCaptureOrientation:currentDeviceOrientation];//[self translateDeviceOrientationToCaptureOrientation:currentDeviceOrientation];
    
    if( ! [self shouldAutorotate] )
    {
        defaultAVCaptureVideoOrientation = [self updateVideoCaptureOrientation:currentDeviceOrientation];
    }
    
    [self customizeNavBar];
    
    self.navigationItem.title = @"FFT Metal Demo";
    
    if ([UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypeCamera])
    {
        //[[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(MAImagePickerChosen:) name:@"MAIPCSuccessInternal" object:nil];
        
        NSLog(@"view did load");
        
        /*
        //Initialize an AVAudioSession, not sure why yet...
        [[NSNotificationCenter defaultCenter] addObserverForName:UIApplicationWillEnterForegroundNotification object:nil queue:[NSOperationQueue mainQueue] usingBlock:^(NSNotification *notification){
            AudioSessionInitialize(NULL, NULL, NULL, NULL);
            AudioSessionSetActive(YES);
        }];
        
        [[NSNotificationCenter defaultCenter] addObserverForName:UIApplicationDidEnterBackgroundNotification object:nil queue:[NSOperationQueue mainQueue] usingBlock:^(NSNotification *notification)
         {
             AudioSessionSetActive(NO);
         }];
        
        
        AudioSessionInitialize(NULL, NULL, NULL, NULL);
        AudioSessionSetActive(YES);
        */
        // Volume View to hide System HUD
       // _volumeView = [[MPVolumeView alloc] initWithFrame:CGRectMake(-100, 0, 10, 0)];
        //[_volumeView sizeToFit];
        //[self.view addSubview:_volumeView];
        
        //[self setCaptureManager:[[MACaptureSession alloc] init]];

        
        //[self createToolbar];
        //[self createTorchSlider];
        
        //[self createDetectionFilter];
        [self createVideoCaptureSession];
        
        
        //[self addToolbar];
        

    }

    //setup metal gpgpu context
    [self setupMetalGPGPUContext];
    [self setupMetalRenderingContext];


    //[self createStatsOverlay];

    //[self loadTestPatternImage];

    [self addNotificationListeners];
    
    //add notification listener
    AVCaptureDeviceFormat * vFormat = self.captureDevice.activeFormat;
    NSLog(@"Active Video Format:  %@ \n\n%@ \n%@\n",vFormat.mediaType,vFormat.formatDescription,vFormat.videoSupportedFrameRateRanges);

    //self.metalRenderView.hidden = YES;
}

-(void)addNotificationListeners
{
    //  Register notifications to start/stop drawing as this app moves into the background
    [[NSNotificationCenter defaultCenter] addObserver: self
                                             selector: @selector(didEnterBackground:)
                                                 name: UIApplicationDidEnterBackgroundNotification
                                               object: nil];
    
    [[NSNotificationCenter defaultCenter] addObserver: self
                                             selector: @selector(willEnterForeground:)
                                                 name: UIApplicationWillEnterForegroundNotification
                                               object: nil];
}

- (void)setPaused:(BOOL)pause
{
    if(_renderLoopPaused == pause)
    {
        return;
    }
    
    if(_displayLink)
    {
        // inform the delegate we are about to pause
        //[_delegate viewController:self
        //                willPause:pause];
        
        if(pause == YES)
        {
            _renderLoopPaused = pause;
            _displayLink.paused   = YES;
            
            // ask the view to release textures until its resumed
            [(MetalRenderView *)self.view releaseTextures];
        }
        else
        {
            _renderLoopPaused = pause;
            _displayLink.paused   = NO;
        }
    }
}

- (BOOL)isPaused
{
    return _renderLoopPaused;
}

- (void)didEnterBackground:(NSNotification*)notification
{
    [self setPaused:YES];
}

- (void)willEnterForeground:(NSNotification*)notification
{
    [self setPaused:NO];
}

- (void)viewWillAppear:(BOOL)animated
{
    [super viewWillAppear:animated];
    
    // run the game loop
    //[self startRenderLoop];
}

- (void)viewWillDisappear:(BOOL)animated
{
    [super viewWillDisappear:animated];
    

    // end the gameloop
    [self stopRenderLoop];
}

- (void)viewDidDisappear:(BOOL)animated
{

    [[NSNotificationCenter defaultCenter] removeObserver:self name:UIDeviceOrientationDidChangeNotification object:nil];

}

#pragma mark -- Metal GPGPU Context Pipeline

-(void)setupMetalGPGPUContext
{

    //frameRenderingSemaphore = dispatch_semaphore_create(1);

    self.metalRenderingQueue = dispatch_queue_create("MetalRenderingQueue", DISPATCH_QUEUE_SERIAL);

    self.context = [MetalGPGPUContext newContext];
    
    //self.imageProvider = [MBEMainBundleTextureProvider textureProviderWithImageNamed:@"mandrill"
   //                                                                          context:self.context];
    
    //self.desaturateFilter = [MetalSaturationImageFilter filterWithSaturationFactor:0.5 context:self.context];
    
    //tell the saturation metal image filter to receive images generated from this class via the MetalTextureProvider protocol texture property
    //self.desaturateFilter.provider = self;
    
    //self.blurFilter = [MetalGaussianBlurFilter filterWithRadius:7 context:self.context];
    //self.blurFilter.provider = self;

    self.fftFilter = [MetalFFTImageFilter filterWithContext:self.context];
    self.fftFilter.provider = self;
}

#pragma mark -- Render State Updates
- (void)updateRenderState
{
    //_rotation += controller.timeSinceLastDraw * 20.0f;
    //_skyboxRotation += controller.timeSinceLastDraw * 1.0f;
}

- (void)updateConstantBuffer
{
    simd::float4x4 base_model = SIMDMatrixMath::translate(0.0f, 0.0f, 1.0f) * SIMDMatrixMath::rotate(_rotation, 0.0f, 1.0f, 0.0f);
    simd::float4x4 quad_mv = _viewMatrix * base_model;
    
    MetalRenderState::uniforms_t *uniforms = (MetalRenderState::uniforms_t *)[_dynamicUniformBuffer[_constantDataBufferIndex] contents];
    uniforms->modelview_matrix = quad_mv;
    uniforms->normal_matrix = simd::inverse(simd::transpose(quad_mv));
    uniforms->modelview_projection_matrix = _projectionMatrix * quad_mv;
    uniforms->inverted_view_matrix = simd::inverse(_viewMatrix);
    
    // calculate the model view projection data for the skybox
    //simd::float4x4 skyboxModelMatrix = SIMDMatrixMath::scale(10.0f) * SIMDMatrixMath::rotate(_skyboxRotation, 0.0f, 1.0f, 0.0f);
    //simd::float4x4 skyboxModelViewMatrix = _viewMatrix * skyboxModelMatrix;
    
    // write the skybox transformation data into the current constant buffer
    //uniforms->skybox_modelview_projection_matrix = _projectionMatrix * skyboxModelViewMatrix;
    
    // Set the device orientation
    switch (currentDeviceOrientation)
    {
        case UIDeviceOrientationUnknown:
            uniforms->orientation = MetalRenderState::Unknown;
            break;
        case UIDeviceOrientationPortrait:
            uniforms->orientation = MetalRenderState::Portrait;
            break;
        case UIDeviceOrientationPortraitUpsideDown:
            uniforms->orientation = MetalRenderState::PortraitUpsideDown;
            break;
        case UIDeviceOrientationLandscapeRight:
            uniforms->orientation = MetalRenderState::LandscapeRight;
            break;
        case UIDeviceOrientationLandscapeLeft:
            uniforms->orientation = MetalRenderState::LandscapeLeft;
            break;
        default:
            uniforms->orientation = MetalRenderState::Portrait;
            break;
    }
}




#pragma mark -- Metal Rendering Context Pipeline

-(void)startRenderLoop
{
    // create a game loop timer using a display link
    _displayLink = [[UIScreen mainScreen] displayLinkWithTarget:self selector:@selector(renderLoop)];
    _displayLink.frameInterval = 1;
    [_displayLink addToRunLoop:[NSRunLoop mainRunLoop]
                 forMode:NSDefaultRunLoopMode];
}

-(void)renderLoop
{

    if(!_firstDrawOccurred)
    {
        // set up timing data for display since this is the first time through this loop
        _timeSinceLastDraw             = 0.0;
        _timeSinceLastDrawPreviousTime = CACurrentMediaTime();
        _firstDrawOccurred              = YES;
    }
    else
    {
        // figure out the time since we last we drew
        CFTimeInterval currentTime = CACurrentMediaTime();
        
        _timeSinceLastDraw = currentTime - _timeSinceLastDrawPreviousTime;
        
        // keep track of the time interval between draws
        _timeSinceLastDrawPreviousTime = currentTime;
    }
    
    // display (render)
    
    //assert([self.view isKindOfClass:[MetalRenderView class]]);
    
    MetalRenderView * metalView = self.metalRenderView;
    
    // call the display method directly on the render view (setNeedsDisplay: has been disabled in the renderview by default)
    //[(MetalRenderView *)self.view display];
    
    // Create autorelease pool per frame to avoid possible deadlock situations
    // because there are 3 CAMetalDrawables sitting in an autorelease pool.
    
    @autoreleasepool
    {
        // handle display changes here
        if(metalView.layerSizeDidUpdate)
        {
            // set the metal layer to the drawable size in case orientation or size changes
            CGSize drawableSize = self.view.bounds.size;
            drawableSize.width  *= self.view.contentScaleFactor;
            drawableSize.height *= self.view.contentScaleFactor;
            
            metalView.metalLayer.drawableSize = drawableSize;
            
            // renderer delegate method so renderer can resize anything if needed
            [self reshape:metalView];
            
            metalView.layerSizeDidUpdate = NO;
        }
        
        // rendering delegate method to ask renderer to draw this frame's content
        [self render:metalView];
        
        // do not retain current drawable beyond the frame.
        // There should be no strong references to this object outside of this view class
        [metalView destroyCurrentDrawable];
    }

}

- (void)render:(MetalRenderView *)view
{
    // Allow the renderer to preflight 3 frames on the CPU (using a semapore as a guard) and commit them to the GPU.
    // This semaphore will get signaled once the GPU completes a frame's work via addCompletedHandler callback below,
    // signifying the CPU can go ahead and prepare another frame.
    dispatch_semaphore_wait(_inflight_semaphore, DISPATCH_TIME_FOREVER);
    
    // Prior to sending any data to the GPU, constant buffers should be updated accordingly on the CPU.
    [self updateConstantBuffer];
    
    // create a new command buffer for each renderpass to the current drawable
    id <MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    
    // create a render command encoder so we can render into something
    MTLRenderPassDescriptor *renderPassDescriptor = view.renderPassDescriptor;
    if (renderPassDescriptor)
    {
        id <MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        [renderEncoder setDepthStencilState:_depthState];
        
        // render the skybox and the quad
        //[self renderSkybox:renderEncoder view:view name:@"skybox"];
        [self renderTexturedQuad:renderEncoder view:view name:@"envmapQuadMix"];
        //[self renderBufferToTexturedQuad:renderEncoder view:view name:@"renderBufferToTexturedQuad"];
        
        [renderEncoder endEncoding];
        
        // schedule a present once the framebuffer is complete
        [commandBuffer presentDrawable:view.currentDrawable];
    }
    
    // Add a completion handler / block to be called once the command buffer is completed by the GPU. All completion handlers will be returned in the order they were committed.
    __block dispatch_semaphore_t block_sema = _inflight_semaphore;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        
        // GPU has completed rendering the frame and is done using the contents of any buffers previously encoded on the CPU for that frame.
        // Signal the semaphore and allow the CPU to proceed and construct the next frame.
        dispatch_semaphore_signal(block_sema);
    }];
    
    // finalize rendering here. this will push the command buffer to the GPU
    [commandBuffer commit];
    
    // This index represents the current portion of the ring buffer being used for a given frame's constant buffer updates.
    // Once the CPU has completed updating a shared CPU/GPU memory buffer region for a frame, this index should be updated so the
    // next portion of the ring buffer can be written by the CPU. Note, this should only be done *after* all writes to any
    // buffers requiring synchronization for a given frame is done in order to avoid writing a region of the ring buffer that the GPU may be reading.
    _constantDataBufferIndex = (_constantDataBufferIndex + 1) % kInFlightCommandBuffers;
}

- (void)renderSkybox:(id <MTLRenderCommandEncoder>)renderEncoder view:(MetalRenderView *)view name:(NSString *)name
{
    // setup for GPU debugger
    [renderEncoder pushDebugGroup:name];
    
    // set the pipeline state object for the quad which contains its precompiled shaders
    [renderEncoder setRenderPipelineState:_skyboxPipelineState];
    
    // set the vertex buffers for the skybox at both indicies 0 and 1 since we are using its vertices as texCoords in the shader
    [renderEncoder setVertexBuffer:_skyboxVertexBuffer offset:0 atIndex:SKYBOX_VERTEX_BUFFER];
    [renderEncoder setVertexBuffer:_skyboxVertexBuffer offset:0 atIndex:SKYBOX_TEXCOORD_BUFFER];
    
    // set the model view projection matrix for the skybox
    [renderEncoder setVertexBuffer:_dynamicUniformBuffer[_constantDataBufferIndex] offset:0 atIndex:SKYBOX_CONSTANT_BUFFER];
    
    // set the fragment shader's texture and sampler
    [renderEncoder setFragmentTexture:_skyboxTex.texture atIndex:SKYBOX_IMAGE_TEXTURE];
    
    [renderEncoder drawPrimitives: MTLPrimitiveTypeTriangleStrip vertexStart: 0 vertexCount: 24];
    
    [renderEncoder popDebugGroup];
}


- (void)renderTexturedQuad:(id <MTLRenderCommandEncoder>)renderEncoder view:(MetalRenderView *)view name:(NSString *)name
{
    // setup for GPU debugger
    [renderEncoder pushDebugGroup:name];
    
    // set the pipeline state object for the skybox which contains its precompiled shaders
    [renderEncoder setRenderPipelineState:_quadPipelineState];
    
    // set the static vertex buffers
    [renderEncoder setVertexBuffer:_quadVertexBuffer offset:0 atIndex:QUAD_VERTEX_BUFFER];
    
    // read the model view project matrix data from the constant buffer
    [renderEncoder setVertexBuffer:_dynamicUniformBuffer[_constantDataBufferIndex] offset:0 atIndex:QUAD_VERTEX_CONSTANT_BUFFER];
    
    // fragment texture for environment
    //[renderEncoder setFragmentTexture:_skyboxTex.texture atIndex:QUAD_ENVMAP_TEXTURE];
    
    // fragment texture for image to be mixed with reflection
    if (!_videoTexture[_constantDataBufferIndex])
    {
        //NSLog(@"here 1 ");
        //[renderEncoder setFragmentTexture:_quadTex.texture atIndex:QUAD_IMAGE_TEXTURE];
    }
    else
    {
        //NSLog(@"here 2 ");
        [renderEncoder setFragmentTexture:_videoTexture[_constantDataBufferIndex] atIndex:QUAD_IMAGE_TEXTURE];
    
        // inverted view matrix fragment buffer for environment mapping
        [renderEncoder setFragmentBuffer:_dynamicUniformBuffer[_constantDataBufferIndex] offset:0 atIndex:QUAD_FRAGMENT_CONSTANT_BUFFER];
        
        // tell the render context we want to draw our primitives
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6 instanceCount:1];
        
    }
    

    [renderEncoder popDebugGroup];
}

- (void)renderBufferToTexturedQuad:(id <MTLRenderCommandEncoder>)renderEncoder view:(MetalRenderView *)view name:(NSString *)name
{
    // setup for GPU debugger
    [renderEncoder pushDebugGroup:name];
    
    // set the pipeline state object for the skybox which contains its precompiled shaders
    [renderEncoder setRenderPipelineState:_quadPipelineState];
    
    // set the static vertex buffers
    [renderEncoder setVertexBuffer:_quadVertexBuffer offset:0 atIndex:QUAD_VERTEX_BUFFER];
    
    // read the model view project matrix data from the constant buffer
    [renderEncoder setVertexBuffer:_dynamicUniformBuffer[_constantDataBufferIndex] offset:0 atIndex:QUAD_VERTEX_CONSTANT_BUFFER];
    
    // fragment texture for environment
    //[renderEncoder setFragmentTexture:_skyboxTex.texture atIndex:QUAD_ENVMAP_TEXTURE];
    
    // fragment texture for image to be mixed with reflection
    if (!_videoBuffer[_constantDataBufferIndex])
    {
        NSLog(@"Video Buffer is nil.");
        //[renderEncoder setFragmentTexture:_quadTex.texture atIndex:QUAD_IMAGE_TEXTURE];
    }
    else
    {
        //NSLog(@"here 2 ");
        [renderEncoder setFragmentBuffer:_videoBuffer[_constantDataBufferIndex] offset:0 atIndex:0];
        
        // inverted view matrix fragment buffer for environment mapping
       // [renderEncoder setFragmentBytes:[_videoBuffer[_constantDataBufferIndex] contents] length:512*512*2*sizeof(float) atIndex:0];
        
        // tell the render context we want to draw our primitives
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6 instanceCount:1];
        
    }
    
    
    [renderEncoder popDebugGroup];
}

- (void)reshape:(MetalRenderView *)view
{
    // when reshape is called, update the view and projection matricies since this means the view orientation or size changed
    float aspect = std::abs(view.bounds.size.width / view.bounds.size.height);
    _projectionMatrix = SIMDMatrixMath::perspective_fov(kFOVY, aspect, 0.1f, 100.0f);
    _viewMatrix = SIMDMatrixMath::lookAt(kEye, kCenter, kUp);
}

-(void)stopRenderLoop
{
    if(_displayLink)
        [_displayLink invalidate];
}

-(void)setupMetalRenderingContext
{
    _constantDataBufferIndex = 0;
    _inflight_semaphore = dispatch_semaphore_create(kInFlightCommandBuffers);
    
    //create a metal rendering view

    CGFloat viewWidth = videoLayerSize.width;
    CGFloat viewHeight = videoLayerSize.height;
    
    if( currentDeviceOrientation == UIDeviceOrientationPortrait || currentDeviceOrientation == UIDeviceOrientationPortraitUpsideDown )
    {
        CGFloat tmp = viewWidth;
        viewWidth = viewHeight;
        viewHeight = tmp;
    }
    
    self.metalRenderView = [[MetalRenderView alloc] initWithFrame:CGRectMake(0, 0, viewWidth/2., viewHeight/2.)]; //(MetalRenderView *)self.view;
    //self.metalRenderView.layer.borderWidth = 2.0;
    //self.metalRenderView.layer.borderColor = [UIColor redColor].CGColor;
    _metalRenderView.backgroundColor = [UIColor purpleColor];
    _metalRenderView.center = CGPointMake(self.view.frame.size.width/2., self.view.frame.size.height/2.);

    [self layoutMetalRenderView];
    //renderView.delegate = _renderer;
    
    // load all renderer assets before starting game loop
    [self configureMetalRenderView:_metalRenderView];
    
    [self.view addSubview:_metalRenderView];
    [self startRenderLoop];
}

- (void)configureMetalRenderView:(MetalRenderView *)view
{
    // assign device created by the view
    _device = view.device;
    
    // setup view with drawable formats
    view.depthPixelFormat   = MTLPixelFormatDepth32Float;
    view.stencilPixelFormat = MTLPixelFormatInvalid;
    view.sampleCount        = 1;
    
    // create a new command queue
    _commandQueue = [_device newCommandQueue];
    
    _defaultLibrary = [_device newDefaultLibrary];
    if(!_defaultLibrary) {
        NSLog(@">> ERROR: Couldnt create a default shader library");
        // assert here becuase if the shader libary isn't loading, nothing good will happen
        assert(0);
    }
    
    // allocate one region of memory for the constant buffer
    for (int i = 0; i < kInFlightCommandBuffers; i++)
    {
        _dynamicUniformBuffer[i] = [_device newBufferWithLength:kMaxBufferBytesPerFrame options:0];
        _dynamicUniformBuffer[i].label = [NSString stringWithFormat:@"ConstantBuffer%i", i];
    }
    
    // load the quad's pipeline state and buffer data
    [self loadQuadAssets:view];
    
    // load the skybox pipeline state and buffer data
    //[self loadSkyboxAssets:view];
    
    // read a mipmapped pvrtc encoded texture
    //_quadTex = [[MetalPVRTexture alloc] initWithResourceName:@"copper_mipmap_4" extension:@"pvr"];
    //BOOL loaded = [_quadTex loadIntoTextureWithDevice:_device];
    //if (!loaded)
    //    NSLog(@"failed to load PVRTC Texture for quad");
    
    // load the skybox
    
    //_skyboxTex = [[MetalTextureCubeMap alloc] initWithResourceName:@"skybox" extension:@"png"];
    //loaded = [_skyboxTex loadIntoTextureWithDevice:_device];
    //if (!loaded)
    //    NSLog(@"failed to load skybox texture");
    
    
    // setup the depth state
    MTLDepthStencilDescriptor *depthStateDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthStateDesc.depthCompareFunction = MTLCompareFunctionAlways;
    depthStateDesc.depthWriteEnabled = YES;
    _depthState = [_device newDepthStencilStateWithDescriptor:depthStateDesc];
    
    // initialize and load all necessary data for video quad texture
    [self setupVideoQuadTexture];
}

- (void)loadQuadAssets:(MetalRenderView *)view
{
    // read the vertex and fragment shader functions from the library
    id <MTLFunction> vertexProgram = [_defaultLibrary newFunctionWithName:@"renderQuadVertex"];
    id <MTLFunction> fragmentProgram = [_defaultLibrary newFunctionWithName:@"renderQuadFragment"];
    //id <MTLFunction> fragmentProgram = [_defaultLibrary newFunctionWithName:@"renderBufferToQuadFragment"];

    //  create a pipeline state descriptor for the quad
    MTLRenderPipelineDescriptor *quadPipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    quadPipelineStateDescriptor.label = @"TexturedQuadPipelineState";
    
    // set pixel formats that match the framebuffer we are drawing into
    quadPipelineStateDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    quadPipelineStateDescriptor.depthAttachmentPixelFormat      = view.depthPixelFormat;
    quadPipelineStateDescriptor.sampleCount                     = view.sampleCount;
    
    // set the vertex and fragment programs
    quadPipelineStateDescriptor.vertexFunction   = vertexProgram;
    quadPipelineStateDescriptor.fragmentFunction = fragmentProgram;
    
    // generate the pipeline state
    _quadPipelineState = [_device newRenderPipelineStateWithDescriptor:quadPipelineStateDescriptor error:nil];
    
    // setup the skybox vertex, texCoord and normal buffers
    _quadVertexBuffer = [_device newBufferWithBytes:quad length:sizeof(quad) options:MTLResourceOptionCPUCacheModeDefault];
    _quadVertexBuffer.label = @"QuadVertexBuffer";
}

- (void)loadSkyboxAssets:(MetalRenderView *)view
{
    id <MTLFunction> vertexProgram = [_defaultLibrary newFunctionWithName:@"skyboxVertex"];
    id <MTLFunction> fragmentProgram = [_defaultLibrary newFunctionWithName:@"skyboxFragment"];
    
    //  create a pipeline state for the skybox
    MTLRenderPipelineDescriptor *skyboxPipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    skyboxPipelineStateDescriptor.label = @"SkyboxPipelineState";
    
    // the pipeline state must match the drawable framebuffer we are rendering into
    skyboxPipelineStateDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    skyboxPipelineStateDescriptor.depthAttachmentPixelFormat      = view.depthPixelFormat;
    skyboxPipelineStateDescriptor.sampleCount                     = view.sampleCount;
    
    // attach the skybox shaders to the pipeline state
    skyboxPipelineStateDescriptor.vertexFunction   = vertexProgram;
    skyboxPipelineStateDescriptor.fragmentFunction = fragmentProgram;
    
    // finally, read out the pipeline state
    _skyboxPipelineState = [_device newRenderPipelineStateWithDescriptor:skyboxPipelineStateDescriptor error:nil];
    if(!_defaultLibrary) {
        NSLog(@">> ERROR: Couldnt create a pipeline");
        assert(0);
    }
    
    // create the skybox vertex buffer
    _skyboxVertexBuffer = [_device newBufferWithBytes:cubeVertexData length:sizeof(cubeVertexData) options:MTLResourceOptionCPUCacheModeDefault];
    _skyboxVertexBuffer.label = @"SkyboxVertexBuffer";
}


- (void)setupVideoQuadTexture
{
    CVMetalTextureCacheFlush(_videoTextureCache, 0);
    CVReturn textureCacheError = CVMetalTextureCacheCreate(kCFAllocatorDefault, NULL, _device, NULL, &_videoTextureCache);
    
    if (textureCacheError)
    {
        NSLog(@">> ERROR: Couldnt create a texture cache");
        assert(0);
    }
    
}


-(void)createDetectionFilter
{
    NSDictionary *detectorOptions = [[NSDictionary alloc] initWithObjectsAndKeys:CIDetectorAccuracyLow, CIDetectorAccuracy, nil];
    self.featDetector = [CIDetector detectorOfType:CIDetectorTypeRectangle context:nil options:detectorOptions];
}

-(void)createToolbar
{

    //CGFloat kCameraToolBarHeight = 44;
    
    self.cameraToolbar = [[UIToolbar alloc] initWithFrame:CGRectMake(0, self.view.bounds.size.height - kCameraToolBarHeight, self.view.bounds.size.width, kCameraToolBarHeight)];
    //[_cameraToolbar setBackgroundImage:[UIImage imageNamed:@"camera-bottom-bar"] forToolbarPosition:UIToolbarPositionAny barMetrics:UIBarMetricsDefault];

    if ([_cameraToolbar respondsToSelector:@selector(setBackgroundImage:forToolbarPosition:barMetrics:)]) {
        [_cameraToolbar setBackgroundImage:[[UIImage alloc] init] forToolbarPosition:UIToolbarPositionAny barMetrics:UIBarMetricsDefault];
    }
    if ([_cameraToolbar respondsToSelector:@selector(setShadowImage:forToolbarPosition:)]) {
        [_cameraToolbar setShadowImage:[[UIImage alloc] init] forToolbarPosition:UIToolbarPositionAny];
    }

    UIBarButtonItem *cancelButton = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemCancel target:self action:@selector(popToPreviousViewController)];
    cancelButton.accessibilityLabel = @"Close Camera Viewer";
    
    /*
    UIImage *cameraButtonImage = [UIImage imageNamed:@"camera-button"];
    UIImage *cameraButtonImagePressed = [UIImage imageNamed:@"camera-button-pressed"];
    UIButton *pictureButtonRaw = [UIButton buttonWithType:UIButtonTypeCustom];
    [pictureButtonRaw setImage:cameraButtonImage forState:UIControlStateNormal];
    [pictureButtonRaw setImage:cameraButtonImagePressed forState:UIControlStateHighlighted];
    [pictureButtonRaw addTarget:self action:@selector(pictureMAIMagePickerController) forControlEvents:UIControlEventTouchUpInside];
    pictureButtonRaw.frame = CGRectMake(0.0, 0.0, cameraButtonImage.size.width, cameraButtonImage.size.height);
    */
    //_pictureButton = [[UIBarButtonItem alloc] initWithCustomView:pictureButtonRaw];
    
    self.pictureButton = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemCamera target:self action:@selector(captureImage)];

    _pictureButton.accessibilityLabel = @"Take Picture";
    
    /*
    if ([[NSUserDefaults standardUserDefaults] objectForKey:kCameraFlashDefaultsKey] == nil)
    {
        [self storeFlashSettingWithBool:YES];
    }
    
    if ([[NSUserDefaults standardUserDefaults] boolForKey:kCameraFlashDefaultsKey])
    {
        _flashButton = [[UIBarButtonItem alloc] initWithImage:[UIImage imageNamed:@"flash-on-button"] style:UIBarButtonItemStylePlain target:self action:@selector(toggleTorch)];
        _flashButton.accessibilityLabel = @"Disable Camera Flash";
        
        flashIsOn = NO;
        [self setFlashOn:NO];
        
        torchIsOn = YES;
        [self setTorchOn:YES];
    }
    else
    {
    */
        _flashButton = [[UIBarButtonItem alloc] initWithImage:[UIImage imageNamed:@"flash-off-button"] style:UIBarButtonItemStylePlain target:self action:@selector(toggleTorch)];
        _flashButton.accessibilityLabel = @"Enable Camera Flash";
        
        torchIsOn =  NO;
        [self setTorchOn:NO];
        
        flashIsOn = NO;
        [self setFlashOn:NO];
    //}
    
    UIBarButtonItem *flexibleSpace = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemFlexibleSpace target:nil action:nil];
    UIBarButtonItem *fixedSpace = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemFixedSpace target:nil action:nil];
    [fixedSpace setWidth:10.0f];
    
    [_cameraToolbar setItems:[NSArray arrayWithObjects:fixedSpace,cancelButton,flexibleSpace,_pictureButton,flexibleSpace,_flashButton,fixedSpace, nil]];
    


}

-(void)createTorchSlider
{

    self.torchSlider = [[UISlider alloc] initWithFrame:CGRectMake(0, 0, self.view.frame.size.width * 9.0/10.0, 22)];
    //[_progressSlider addTarget:self action:@selector(setProgressSliderAction:) forControlEvents:UIControlEventValueChanged];
    self.torchSlider.center = CGPointMake(self.view.frame.size.width/2.0, _cameraToolbar.frame.origin.y);
    //_torchSlider.respondsToTrackTouches = false;
}

-(void)addToolbar
{
    [self.view addSubview:_cameraToolbar];
    
    //[[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(transitionToMAImagePickerControllerAdjustViewController) name:kImageCapturedSuccessfully object:nil];
    
    _cameraPictureTakenFlash = [[UIView alloc] initWithFrame:CGRectMake(0, 0, self.view.bounds.size.width, self.view.bounds.size.height -_cameraToolbar.frame.size.height)];
    [_cameraPictureTakenFlash setBackgroundColor:[UIColor colorWithRed:0.99f green:0.99f blue:1.00f alpha:1.00f]];
    [_cameraPictureTakenFlash setUserInteractionEnabled:NO];
    [_cameraPictureTakenFlash setAlpha:0.0f];
    [self.view addSubview:_cameraPictureTakenFlash];

}


-(void)popToPreviousViewController
{
    [self.navigationController setNavigationBarHidden:NO];
    [self.navigationController popViewControllerAnimated:YES];

}

//MANAGE THE VIDEO CAPTURE SESSION

-(void)createVideoCaptureSession
{
    self.videoCaptureSession = [[AVCaptureSession alloc] init];

    [self.videoCaptureSession beginConfiguration];

    [self addVideoInputFromCamera];
    [self addSampleBufferVideoOutput];

    [self setCameraCaptureRate];


    //[self addMetadataVideoOutput];
    [self createVideoPreviewLayer];
    //[self createCustomVideoPreviewLayer];
    //[self createDrawLayer];

    [self.videoCaptureSession commitConfiguration];
    
}

-(void)createStatsOverlay
{
    
    //create some labels at the top of the video preview layer

    //assume our custom video preview layer (i.e. the one showing the fft visualization) is smaller than the total screen size and centered in our view
    //we'll center the labels between the bottom of the custom video preview layer and the bottom of the view
    
    CGFloat bottomOfRenderLayer;
    if( _metalRenderView )
        bottomOfRenderLayer = (self.metalRenderView.frame.size.height + self.metalRenderView.frame.origin.y);
    else if( _customPreviewLayer )
            bottomOfRenderLayer = (self.customPreviewLayer.frame.size.height + self.customPreviewLayer.frame.origin.y);
    else
    {
        NSLog(@"No CustomVideoPreviewLayer or MetalRenderView has been created for rendering video");
        return;
    }
    
    
    CGRect containerViewBounds = CGRectMake( 0, bottomOfRenderLayer, self.view.frame.size.width, (self.view.frame.size.height - bottomOfRenderLayer ) );

    //self.frameRateLabel = [[UILabel alloc] initWithFrame:CGRectMake( 0, bottomOfRenderLayer+containerViewBounds.size.height/6.0, containerViewBounds.size.width, containerViewBounds.size.height/3.0 ) ];
    self.frameRateLabel = [[UILabel alloc] initWithFrame:CGRectMake( 0, self.view.frame.size.height*5.0/6.0, self.view.frame.size.width, self.view.frame.size.height/8.0 ) ];
    //self.phaseCorrelationLabel = [[UILabel alloc] initWithFrame:CGRectMake( 0, self.frameRateLabel.frame.origin.y+self.frameRateLabel.frame.size.height, containerViewBounds.size.width, containerViewBounds.size.height/3.0 ) ];

    _frameRateLabel.textAlignment = NSTextAlignmentCenter;
    //_phaseCorrelationLabel.textAlignment = NSTextAlignmentCenter;
    
    _frameRateLabel.textColor = [UIColor whiteColor];
    //_phaseCorrelationLabel.textColor = [UIColor whiteColor];

    _frameRateLabel.text = [NSString stringWithFormat:@"Capture Rate:  %.2f Hz", 0.0f];
    //_phaseCorrelationLabel.text = [NSString stringWithFormat:@"Phase Correlation\tX:\t%d,\tY:\t%d", 0, 0];

    //_frameRateLabel.backgroundColor = [UIColor orangeColor];
    //_phaseCorrelationLabel.backgroundColor = [UIColor purpleColor];
    
    [self.view addSubview:_frameRateLabel];
    //[self.view addSubview:_phaseCorrelationLabel];
    
}

- (void)createVideoPreviewLayer
{

    CGFloat viewWidth = previewLayerSize.width;
    CGFloat viewHeight = previewLayerSize.height;
    
    if( currentDeviceOrientation == UIDeviceOrientationPortrait || currentDeviceOrientation == UIDeviceOrientationPortraitUpsideDown )
    {
        CGFloat tmp = viewWidth;
        viewWidth = viewHeight;
        viewHeight = tmp;
    }

    //create the video preview layer (i.e. the layer that will render video within our view)
    self.previewLayer = [AVCaptureVideoPreviewLayer layerWithSession:self.videoCaptureSession ];
    // create a custom preview layer
    //self.customPreviewLayer = [CALayer layer];
    
    //the preview layer video gravity setting determins how the video will be formatted into the preview layer of pixelar size when shown on screen
    //AVLayerVideoGravityResizeAspect -- Preserve the aspect ratio when fitting the video output inside the preview layer
    //AVLayerVideoGravityResizeAspectFill -- Preserve aspect ratio but fill one dimension of the preview layer such that some video in the other dimensions falls outside of the preview layer
    //AVLayerVideoGravityResize --  Fit video to preview layer do not preserve aspect ratio
    videoGravity = AVLayerVideoGravityResizeAspectFill;
    [_previewLayer setVideoGravity:videoGravity];
    //CGRect layerRect = CGRectMake(0, 0, self.view.bounds.size.width, self.view.bounds.size.height /*- kCameraToolBarHeight*/);
    
    
    //for max resolution
    self.previewLayer.frame = CGRectMake(0,0, viewWidth, viewHeight);//self.view.bounds;
    //self.customPreviewLayer.bounds = CGRectMake(0, 0, videoLayerSize.width, videoLayerSize.height);
    
    [self layoutPreviewLayer];
    
    [self.previewLayer setPosition:CGPointMake(self.view.frame.size.width/2.0,self.view.frame.size.height/2.0)];
    [self.view.layer addSublayer:self.previewLayer];
    //[self.view.layer addSublayer:self.customPreviewLayer];

    
}

-(void)createCustomVideoPreviewLayer
{

    CGFloat viewWidth = customPreviewLayerSize.width;
    CGFloat viewHeight = customPreviewLayerSize.height;
    
    if( currentDeviceOrientation == UIDeviceOrientationPortrait || currentDeviceOrientation == UIDeviceOrientationPortraitUpsideDown )
    {
        CGFloat tmp = viewWidth;
        viewWidth = viewHeight;
        viewHeight = tmp;
    }

    //create the video preview layer (i.e. the layer that will render video within our view)
    //self.previewLayer = [AVCaptureVideoPreviewLayer layerWithSession:self.videoCaptureSession ];
    // create a custom preview layer
    self.customPreviewLayer = [CALayer layer];
    
    //the preview layer video gravity setting determins how the video will be formatted into the preview layer of pixelar size when shown on screen
    //AVLayerVideoGravityResizeAspect -- Preserve the aspect ratio when fitting the video output inside the preview layer
    //AVLayerVideoGravityResizeAspectFill -- Preserve aspect ratio but fill one dimension of the preview layer such that some video in the other dimensions falls outside of the preview layer
    //AVLayerVideoGravityResize --  Fit video to preview layer do not preserve aspect ratio
    //videoGravity = AVLayerVideoGravityResizeAspectFill;
    //[_previewLayer setVideoGravity:videoGravity];
    //CGRect layerRect = CGRectMake(0, 0, self.view.bounds.size.width, self.view.bounds.size.height /*- kCameraToolBarHeight*/);
    
    
    //for max resolution
    //self.previewLayer.frame=CGRectMake(0,0, self.view.frame.size.width, self.view.frame.size.height);//self.view.bounds;
    self.customPreviewLayer.bounds = CGRectMake(0, 0, viewWidth, viewHeight);
    
    [self layoutCustomPreviewLayer];
    
    //[self.previewLayer setPosition:CGPointMake(self.view.frame.size.width/2.0,self.view.frame.size.height/2.0)];
    //[self.view.layer addSublayer:self.previewLayer];
    [self.view.layer addSublayer:self.customPreviewLayer];
}

- (void)layoutPreviewLayer
{
    NSLog(@"layout preview layer");
    if (self.previewLayer != nil) {
        
        //currentDeviceOrientation = [[UIDevice currentDevice] orientation];
        CALayer* layer = self.previewLayer;
        CGRect bounds = self.previewLayer.bounds;
        int rotation_angle = 0;
        bool flip_bounds = false;
        
        switch (currentDeviceOrientation)
        {
            case UIDeviceOrientationPortrait:
                rotation_angle = 270;
                break;
            case UIDeviceOrientationPortraitUpsideDown:
                rotation_angle = 90;
                break;
            case UIDeviceOrientationLandscapeLeft:
                NSLog(@"left");
                rotation_angle = 180;
                break;
            case UIDeviceOrientationLandscapeRight:
                NSLog(@"right");
                rotation_angle = 0;
                break;
            case UIDeviceOrientationFaceUp:
            case UIDeviceOrientationFaceDown:
            default:
                break; // leave the layer in its last known orientation
        }
        
        switch (defaultAVCaptureVideoOrientation)
        {
            case AVCaptureVideoOrientationLandscapeRight:
                rotation_angle += 180;
                break;
            case AVCaptureVideoOrientationPortraitUpsideDown:
                rotation_angle += 270;
                break;
            case AVCaptureVideoOrientationPortrait:
                rotation_angle += 90;
            case AVCaptureVideoOrientationLandscapeLeft:
                break;
            default:
                break;
        }
        rotation_angle = rotation_angle % 360;
        
        if (rotation_angle == 90 || rotation_angle == 270) {
            flip_bounds = true;
        }
        
        if (flip_bounds) {
            NSLog(@"flip bounds");
            bounds = CGRectMake(0, 0, previewLayerSize.height, previewLayerSize.width);
        }
        
        
        /*
        videoGravity = AVLayerVideoGravityResizeAspectFill;
        
        CGFloat aspectRatio = ((CGFloat)( captureWidth)) / ((CGFloat)(captureHeight));
        if( [videoGravity localizedCompare:AVLayerVideoGravityResizeAspectFill] == NSOrderedSame )
        {
            //Preserve aspect ratio; fill layer bounds (that is, fill the smaller dimension to the layer bounds and extend the larger dimension).
            CGRect adjustedVideoLayerBounds = CGRectMake(0, 0, bounds.size.width, bounds.size.height);
            if( bounds.size.height > bounds.size.width )
                adjustedVideoLayerBounds.size.width *= aspectRatio;
            else
                adjustedVideoLayerBounds.size.height *= aspectRatio;
            
            bounds = adjustedVideoLayerBounds;
            //
        }
        */
        
        //center the layer in its parent view
        layer.position = CGPointMake(self.view.frame.size.width/2., self.view.frame.size.height/2.);
        
        //rotate the layer
        layer.affineTransform = CGAffineTransformMakeRotation( DegreesToRadians(rotation_angle) );
        
        //update the layers bounds rect
        layer.bounds = bounds;
    }
    
}


- (void)layoutCustomPreviewLayer
{
    NSLog(@"layout preview layer");
    if (self.view != nil) {
        
        CALayer* layer = self.customPreviewLayer;
        CGRect bounds = self.customPreviewLayer.bounds;
        int rotation_angle = 0;
        bool flip_bounds = false;
        
        switch (currentDeviceOrientation)
        {
            case UIDeviceOrientationPortrait:
                rotation_angle = 270;
                break;
            case UIDeviceOrientationPortraitUpsideDown:
                rotation_angle = 90;
                break;
            case UIDeviceOrientationLandscapeLeft:
                NSLog(@"left");
                rotation_angle = 180;
                break;
            case UIDeviceOrientationLandscapeRight:
                NSLog(@"right");
                rotation_angle = 0;
                break;
            case UIDeviceOrientationFaceUp:
            case UIDeviceOrientationFaceDown:
            default:
                break; // leave the layer in its last known orientation
        }
        
        switch (defaultAVCaptureVideoOrientation)
        {
            case AVCaptureVideoOrientationLandscapeRight:
                rotation_angle += 180;
                break;
            case AVCaptureVideoOrientationPortraitUpsideDown:
                rotation_angle += 270;
                break;
            case AVCaptureVideoOrientationPortrait:
                rotation_angle += 90;
            case AVCaptureVideoOrientationLandscapeLeft:
                break;
            default:
                break;
        }
        rotation_angle = rotation_angle % 360;
        
        if (rotation_angle == 90 || rotation_angle == 270) {
            flip_bounds = true;
        }
        
        if (flip_bounds) {
            NSLog(@"flip bounds");
            bounds = CGRectMake(0, 0, customPreviewLayerSize.height, customPreviewLayerSize.width);
        }
        
        
        
        //videoGravity = AVLayerVideoGravityResizeAspectFill;
        
        CGFloat aspectRatio = ((CGFloat)( captureWidth)) / ((CGFloat)(captureHeight));
        if( [videoGravity localizedCompare:AVLayerVideoGravityResizeAspectFill] == NSOrderedSame || [videoGravity localizedCompare:AVLayerVideoGravityResizeAspect] )
        {
            //Preserve aspect ratio; fill layer bounds (that is, fill the smaller dimension to the layer bounds and extend the larger dimension).
            CGRect adjustedVideoLayerBounds = CGRectMake(0, 0, bounds.size.width, bounds.size.height);
            if( bounds.size.height > bounds.size.width )
                adjustedVideoLayerBounds.size.width *= aspectRatio;
            else
                adjustedVideoLayerBounds.size.height *= aspectRatio;
            
            bounds = adjustedVideoLayerBounds;
            //
        }
        
        
        //center the layer in its parent view
        layer.position = CGPointMake(self.view.frame.size.width/2., self.view.frame.size.height/2.);
        
        //rotate the layer
        layer.affineTransform = CGAffineTransformMakeRotation( DegreesToRadians(rotation_angle) );
        
        //update the layers bounds rect
        layer.bounds = bounds;
    }
    
}

- (void)layoutMetalRenderView
{
    NSLog(@"layout metal layer");
    if (_metalRenderView != nil) {
        
        //CALayer* layer = self.customPreviewLayer;
        CGRect bounds = _metalRenderView.bounds;
        int rotation_angle = 0;
        bool flip_bounds = false;
        
        switch (currentDeviceOrientation)
        {
            case UIDeviceOrientationPortrait:
                rotation_angle = 270;
                break;
            case UIDeviceOrientationPortraitUpsideDown:
                rotation_angle = 90;
                break;
            case UIDeviceOrientationLandscapeLeft:
                NSLog(@"left");
                rotation_angle = 180;
                break;
            case UIDeviceOrientationLandscapeRight:
                NSLog(@"right");
                rotation_angle = 0;
                break;
            case UIDeviceOrientationFaceUp:
            case UIDeviceOrientationFaceDown:
            default:
                break; // leave the layer in its last known orientation
        }
        
        switch (defaultAVCaptureVideoOrientation)
        {
            case AVCaptureVideoOrientationLandscapeRight:
                rotation_angle += 180;
                break;
            case AVCaptureVideoOrientationPortraitUpsideDown:
                rotation_angle += 270;
                break;
            case AVCaptureVideoOrientationPortrait:
                rotation_angle += 90;
            case AVCaptureVideoOrientationLandscapeLeft:
                break;
            default:
                break;
        }
        rotation_angle = rotation_angle % 360;
        
        if (rotation_angle == 90 || rotation_angle == 270) {
            flip_bounds = true;
        }
        
        CGFloat aspectRatio = ((CGFloat)( captureWidth )) / ((CGFloat)( captureHeight));

        if (flip_bounds) {
            NSLog(@"metal render view flip bounds");
            bounds = CGRectMake(0, 0, videoLayerSize.height, videoLayerSize.width);
            
            //aspectRatio = ((CGFloat)( captureHeight)) / ((CGFloat)(captureWidth));
        }
        
        
        //videoGravity = AVLayerVideoGravityResizeAspectFill;
        /*
        //set the metal render view to have the same aspect ratio as the video we're capturing
        if( [videoGravity localizedCompare:AVLayerVideoGravityResizeAspectFill] == NSOrderedSame || [videoGravity localizedCompare:AVLayerVideoGravityResizeAspect] )
        {
            //Preserve aspect ratio; fill layer bounds (that is, fill the smaller dimension to the layer bounds and extend the larger dimension).
            CGRect adjustedVideoLayerBounds = CGRectMake(0, 0, bounds.size.width, bounds.size.height);
            if( bounds.size.height > bounds.size.width )
                adjustedVideoLayerBounds.size.width *= aspectRatio;
            else
                adjustedVideoLayerBounds.size.height *= aspectRatio;
            
            bounds = adjustedVideoLayerBounds;
            //
        }
        */
        
        //center the layer in its parent view
        _metalRenderView.center = CGPointMake(self.view.frame.size.width/2., self.view.frame.size.height/2.);
        
        //rotate the layer
        _metalRenderView.layer.affineTransform = CGAffineTransformMakeRotation( DegreesToRadians(rotation_angle) );
        
        //update the layers bounds rect
        _metalRenderView.bounds = bounds;
    }
    
}


-(void)createDrawLayer
{
    self.drawLayer = [CALayer layer];
    self.drawLayer.frame = self.view.bounds;
    [self.view.layer addSublayer:self.drawLayer];
}

- (void)addVideoInputFromCamera
{
    AVCaptureDevice *captureDevice;
    
    
    NSArray *devices = [AVCaptureDevice devices];
    self.isUsingFrontFacingCamera = NO;
    AVCaptureDevicePosition desiredPosition = AVCaptureDevicePositionBack;

    for (AVCaptureDevice *device in devices)
    {
        if ([device hasMediaType:AVMediaTypeVideo])
        {
            if ([device position] == desiredPosition)
            {
                if ([device position] == AVCaptureDevicePositionFront)
                    self.isUsingFrontFacingCamera = YES;
                
                captureDevice = device;
                //[self setTorchOn:YES];
                
                break;
            }
            


        }
    }
    
    NSError * error;

    
    /*
    if ( [captureDevice respondsToSelector:@selector(setActiveVideoMinFrameDuration:)] )
    {
        if( [captureDevice lockForConfiguration:&error] )
        {
            //[captureDevice setActiveFormat:newFormat];
            self.setFrameRateViaCaptureDevice = YES;

            double maxSeconds = CMTimeGetSeconds(captureDevice.activeVideoMaxFrameDuration);
            double minSeconds = CMTimeGetSeconds(captureDevice.activeVideoMinFrameDuration);
            
            NSLog(@"CMTime max seconds: %g", maxSeconds);
            NSLog(@"CMTime min seconds: %g", minSeconds);

            //limit the frame rate when capturing video from camera here
            
            //[captureDevice setActiveVideoMinFrameDuration:CMTimeMake(1, 30 )];
            //[captureDevice setActiveVideoMaxFrameDuration:CMTimeMake(1, 60 )];
        
            
        
            [captureDevice unlockForConfiguration];

        }
        
        if( error )
            NSLog(@"Error setting frame duration on capture device: \n\n%@", [error localizedDescription]);
    }
    */

    
    //turn on autofocus range restriction
    /*
    if (captureDevice.isAutoFocusRangeRestrictionSupported)
    {
        NSError * error;
        if ([captureDevice lockForConfiguration:&error])
        {
            [captureDevice setAutoFocusRangeRestriction:AVCaptureAutoFocusRangeRestrictionFar];
            [captureDevice unlockForConfiguration];
        }
        
        if( error )
            NSLog(@"%@", error.localizedDescription);
    }
    */
    
    if (/*[captureDevice isFocusPointOfInterestSupported] &&*/ [captureDevice isFocusModeSupported:AVCaptureFocusModeContinuousAutoFocus])
    {
        NSError *error;
        if ([captureDevice lockForConfiguration:&error])
        {
            //[captureDevice setFocusPointOfInterest:pointOfInterest];
            
            [captureDevice setFocusMode:AVCaptureFocusModeContinuousAutoFocus];
            
            if(/*[captureDevice isExposurePointOfInterestSupported] &&*/ [captureDevice isExposureModeSupported:AVCaptureExposureModeContinuousAutoExposure])
            {
                
                
                //[captureDevice setExposurePointOfInterest:pointOfInterest];
                [captureDevice setExposureMode:AVCaptureExposureModeContinuousAutoExposure];
            }
            
            [captureDevice unlockForConfiguration];
            
            NSLog(@"FOCUS OK");
        }
        
        if( error )
            NSLog(@"Error setting auto focus on capture device: \n\n%@", [error localizedDescription]);
    }
    
    //turn off smooth autofocus for non-video recording
    if (captureDevice.isSmoothAutoFocusSupported)
    {
        NSError * error;
        if ([captureDevice lockForConfiguration:&error])
        {
            [captureDevice setSmoothAutoFocusEnabled:NO];
            [captureDevice unlockForConfiguration];
        }
        
        NSLog(@"SMOOTH FOCUS OK");
        
        if( error )
            NSLog(@"Error setting smooth auto focus on capture device: \n\n%@", [error localizedDescription]);
    }
    //else if( captureDevice respondsToSelector:@selector(
    

    
    //turn on low light boost
    if ( [captureDevice respondsToSelector:@selector(isLowLightBoostSupported)] )
    {
        if ([captureDevice lockForConfiguration:nil]) {
            if (captureDevice.isLowLightBoostSupported)
                captureDevice.automaticallyEnablesLowLightBoostWhenAvailable = YES;
            [captureDevice unlockForConfiguration];
        }
        
        NSLog(@"LOW LIGHT BOOST OK");
        
        if( error )
            NSLog(@"Error setting smooth auto focus on capture device: \n\n%@", [error localizedDescription]);
    }
    
    
    //AVCaptureDevice *videoDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    
    
    AVCaptureDeviceInput *backFacingCameraDeviceInput = [AVCaptureDeviceInput deviceInputWithDevice:captureDevice error:&error];
    
    //store a reference to our capture device, so we can reset settings like frame rate if we change the resolution preset
    //after this method has already run
    self.captureDevice = captureDevice;
    
    if (!error)
    {
        if ([_videoCaptureSession canAddInput:backFacingCameraDeviceInput])
        {
            [_videoCaptureSession addInput:backFacingCameraDeviceInput];
        }
        
        
        
    }
    else
    {
        NSLog(@"Error adding video input from camera");
    }
    
    
    
    
}

-(void)setCameraCaptureRate
{
    self.setFrameRateViaCaptureDevice = NO;
    
    NSString * resString = [NSString stringWithFormat:@"%d x %d", captureWidth, captureHeight ];
    
    //must call addInput before setting video capture framerate, otherwise it won't take effect
    for(AVCaptureDeviceFormat *vFormat in [self.captureDevice formats] )
    {
        CMFormatDescriptionRef description= vFormat.formatDescription;
        float maxrate=((AVFrameRateRange*)[vFormat.videoSupportedFrameRateRanges objectAtIndex:0]).maxFrameRate;
        
        NSLog(@"Available Video Format:  %@ \n\n%@ \n%@\n",vFormat.mediaType,vFormat.formatDescription,vFormat.videoSupportedFrameRateRanges);
        
        NSString * formatDescString = [NSString stringWithFormat:@"%@", vFormat.formatDescription ];
        
        //this will choose the highest aviable resolution for capture with the highest frame rate
        //e.g. on iPhone 6 w/ iOS 8 :  1920x1080 @ 60 Hz
        if( maxrate>captureRate-1 && CMFormatDescriptionGetMediaSubType(description)==kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange && [formatDescString rangeOfString:resString].location != NSNotFound)
        {
            if ( YES == [self.captureDevice lockForConfiguration:NULL] )
            {
                NSLog(@"Chose Video Format:  %@ \n\n%@ \n%@\n",vFormat.mediaType,vFormat.formatDescription,vFormat.videoSupportedFrameRateRanges);

                self.captureDevice.activeFormat = vFormat;
                
                [self.captureDevice setActiveVideoMinFrameDuration:CMTimeMake(10,captureRate*10)];
                [self.captureDevice setActiveVideoMaxFrameDuration:CMTimeMake(10,captureRate*10)];
                self.setFrameRateViaCaptureDevice = YES;
                
                [self.captureDevice unlockForConfiguration];
                
                double maxSeconds = CMTimeGetSeconds(self.captureDevice.activeVideoMaxFrameDuration);
                double minSeconds = CMTimeGetSeconds(self.captureDevice.activeVideoMinFrameDuration);
                
                NSLog(@"max frame rate in seconds: %g", maxSeconds);
                NSLog(@"min frame rate in seconds: %g", minSeconds);
                
                break;
            }
        }
    }


}

-(void)setTorchOn:(BOOL)boolWantsFlash
{
    torchIsOn = boolWantsFlash;
    [self updateTorch];
}

-(void)toggleTorch
{
    torchIsOn = !torchIsOn;
    [self updateTorch];

}

-(void)updateTorch
{
    NSArray *devices = [AVCaptureDevice devices];
    //[self.videoCaptureSession stopRunning];
    
    for (AVCaptureDevice *device in devices)
    {
        NSError * error;
        
        if (device.hasTorch) {
            if (torchIsOn)
            {
                [device lockForConfiguration:nil];

                if( device.hasTorch )
                {
                    if( self.torchSlider )
                        [device setTorchModeOnWithLevel:_torchSlider.value error:&error];
                    else
                        device.torchMode = AVCaptureTorchModeOn;
                }
                if( error )
                    NSLog(@"Error setting torch mode level: \n\n%@", error);
                
                [device unlockForConfiguration];
                
                
                [_flashButton setImage:[UIImage imageNamed:@"flash-on-button"]];
                _flashButton.accessibilityLabel = @"Disable Camera Flash";
                //[self storeFlashSettingWithBool:YES];
                
                
            }
            else
            {
                [device lockForConfiguration:nil];
                
                if( device.hasTorch )
                    device.torchMode = AVCaptureTorchModeOff;
                
                
                [device unlockForConfiguration];
                
                [_flashButton setImage:[UIImage imageNamed:@"flash-off-button"]];
                _flashButton.accessibilityLabel = @"Enable Camera Flash";
                //[self storeFlashSettingWithBool:NO];
            }
        }
    }
    
    //[self.videoCaptureSession startRunning];

}


- (void)setFlashOn:(BOOL)boolWantsFlash
{
    flashIsOn = boolWantsFlash;
    [self updateFlash];
}

-(void)toggleFlash
{
    flashIsOn = !flashIsOn;
    [self updateFlash];
}

- (void)updateFlash
{
    NSArray *devices = [AVCaptureDevice devices];
    //[self.videoCaptureSession stopRunning];

    for (AVCaptureDevice *device in devices)
    {
        NSError * error;

        if (device.flashAvailable) {
            if (flashIsOn)
            {
                [device lockForConfiguration:nil];
                device.flashMode = AVCaptureFlashModeOn;

                if( error )
                    NSLog(@"Error setting torch mode level 1.0: \n\n%@", error);

                [device unlockForConfiguration];
                
                [_flashButton setImage:[UIImage imageNamed:@"flash-on-button"]];
                _flashButton.accessibilityLabel = @"Disable Camera Flash";
                //[self storeFlashSettingWithBool:YES];

                
            }
            else
            {
                [device lockForConfiguration:nil];
                device.flashMode = AVCaptureFlashModeOff;
                
                if( error )
                    NSLog(@"Error setting torch mode level 0.5: \n\n%@", error);

                
                [device unlockForConfiguration];
                
                [_flashButton setImage:[UIImage imageNamed:@"flash-off-button"]];
                _flashButton.accessibilityLabel = @"Enable Camera Flash";
                //[self storeFlashSettingWithBool:NO];
            }
        }
    }
    
    [self.videoCaptureSession startRunning];

}

- (void)storeFlashSettingWithBool:(BOOL)flashSetting
{
    [[NSUserDefaults standardUserDefaults] setBool:flashSetting forKey:kCameraFlashDefaultsKey];
    [[NSUserDefaults standardUserDefaults] synchronize];
}

-(void)addSampleBufferVideoOutput
{
    // Make a video data output
    self.videoDataOutput = [[AVCaptureVideoDataOutput alloc] init];
    
    //NSLog(@"%@", self.videoDataOutput.availableVideoCVPixelFormatTypes);

    //set the pixel packing format for the image we want to achieve from the camera data
    // we want BGRA, both CoreGraphics and OpenGL work well with 'BGRA'
    // BGRA: kCVPixelFormatType_32BGRA;
    // grayscale: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    NSDictionary *rgbOutputSettings = [NSDictionary dictionaryWithObject:
                                       [NSNumber numberWithInt:kCVPixelFormatType_32BGRA] forKey:(id)kCVPixelBufferPixelFormatTypeKey];
    [self.videoDataOutput setVideoSettings:rgbOutputSettings];

    //this is a legacy frame rate setting, don't use it anymore
    //self.videoDataOutput.minFrameDuration = CMTimeMake(1, 30);
    
    //this is the default preset; don't explicitly set this or you will have to reset settings such as frame rate to 30 Hz
    //[_videoCaptureSession setSessionPreset:AVCaptureSessionPresetHigh];
    
    /*
    //legacy, don't do this here, use the vFormat method to set resolution/frame rate supported by device
    if( [_videoCaptureSession canSetSessionPreset:AVCaptureSessionPreset640x480] )
    {
        NSLog(@"Set Video Capture Resolution:   352x288");
        [_videoCaptureSession setSessionPreset:AVCaptureSessionPreset640x480];
        
    }
    */

    // discard frames if the data output queue is blocked; we don't want to discard any frames
    [self.videoDataOutput setAlwaysDiscardsLateVideoFrames:YES];

    // create a serial dispatch queue used for the sample buffer delegate
    // a serial dispatch queue must be used to guarantee that video frames will be delivered in order
    // see the header doc for setSampleBufferDelegate:queue: for more information
    dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
    videoDataOutputQueue = dispatch_queue_create("VideoDataOutputQueue", attr);
    //videoDataOutputQueue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH,0);

    //set a delegate to receive camera buffer as a callback for processing on the CPU
    [self.videoDataOutput setSampleBufferDelegate:self queue:videoDataOutputQueue];
    
    
    //add the video data object to the video capture session
    if ( [_videoCaptureSession canAddOutput:self.videoDataOutput] ){
        [_videoCaptureSession addOutput:self.videoDataOutput];
    }
    
    
    //set the video orientation
    //[self.videoDataOutput connectionWithMediaType:AVMediaTypeVideo].videoOrientation = AVCaptureVideoOrientationPortrait;

    //set the AV Media Format to be video
    AVCaptureConnection * connection = [self.videoDataOutput connectionWithMediaType:AVMediaTypeVideo];
    
    /*
    if( !self.setFrameRateViaCaptureDevice )
    {
    
        //legacy frame interval setting
        if( [connection respondsToSelector:@selector(setVideoMinFrameDuration:)] )
            [connection setVideoMinFrameDuration:CMTimeMake(1,15)];
        else if( [self.videoDataOutput respondsToSelector:@selector(setMinFrameDuration:)] )
            [self.videoDataOutput setMinFrameDuration:CMTimeMake(1, 15)];
    
    }
    */
    
    //set the video to give the output buffer in portrait orientation rather than landscape
    [connection setVideoOrientation:defaultAVCaptureVideoOrientation];
    [connection setEnabled:YES];
}

-(void)addMetadataVideoOutput
{
    // Make a video data output
    self.metadataOutput = [[AVCaptureMetadataOutput alloc] init];

    //[_videoCaptureSession setSessionPreset:AVCaptureSessionPresetHigh];
    
    videoDataOutputQueue = dispatch_queue_create("VideoDataOutputQueue", DISPATCH_QUEUE_CONCURRENT);

    if ([_videoCaptureSession canAddOutput:_metadataOutput])
    {
        [_videoCaptureSession addOutput:_metadataOutput];
        [_metadataOutput setMetadataObjectsDelegate:self queue:videoDataOutputQueue];
        [_metadataOutput setMetadataObjectTypes:@[AVMetadataObjectTypeFace]];
    }
    
    // get the output for doing face detection.
    [[self.metadataOutput connectionWithMediaType:AVMediaTypeVideo] setEnabled:YES];

}

- (NSNumber *) exifOrientation: (UIDeviceOrientation) orientation
{
    int exifOrientation;
    /* kCGImagePropertyOrientation values
     The intended display orientation of the image. If present, this key is a CFNumber value with the same value as defined
     by the TIFF and EXIF specifications -- see enumeration of integer constants.
     The value specified where the origin (0,0) of the image is located. If not present, a value of 1 is assumed.
     
     used when calling featuresInImage: options: The value for this key is an integer NSNumber from 1..8 as found in kCGImagePropertyOrientation.
     If present, the detection will be done based on that orientation but the coordinates in the returned features will still be based on those of the image. */
    
    enum {
        PHOTOS_EXIF_0ROW_TOP_0COL_LEFT			= 1, //   1  =  0th row is at the top, and 0th column is on the left (THE DEFAULT).
        PHOTOS_EXIF_0ROW_TOP_0COL_RIGHT			= 2, //   2  =  0th row is at the top, and 0th column is on the right.
        PHOTOS_EXIF_0ROW_BOTTOM_0COL_RIGHT      = 3, //   3  =  0th row is at the bottom, and 0th column is on the right.
        PHOTOS_EXIF_0ROW_BOTTOM_0COL_LEFT       = 4, //   4  =  0th row is at the bottom, and 0th column is on the left.
        PHOTOS_EXIF_0ROW_LEFT_0COL_TOP          = 5, //   5  =  0th row is on the left, and 0th column is the top.
        PHOTOS_EXIF_0ROW_RIGHT_0COL_TOP         = 6, //   6  =  0th row is on the right, and 0th column is the top.
        PHOTOS_EXIF_0ROW_RIGHT_0COL_BOTTOM      = 7, //   7  =  0th row is on the right, and 0th column is the bottom.
        PHOTOS_EXIF_0ROW_LEFT_0COL_BOTTOM       = 8  //   8  =  0th row is on the left, and 0th column is the bottom.
    };
    
    switch (orientation) {
        case UIDeviceOrientationPortraitUpsideDown:  // Device oriented vertically, home button on the top
            exifOrientation = PHOTOS_EXIF_0ROW_LEFT_0COL_BOTTOM;
            break;
        case UIDeviceOrientationLandscapeLeft:       // Device oriented horizontally, home button on the right
            if (self.isUsingFrontFacingCamera)
                exifOrientation = PHOTOS_EXIF_0ROW_BOTTOM_0COL_RIGHT;
            else
                exifOrientation = PHOTOS_EXIF_0ROW_TOP_0COL_LEFT;
            break;
        case UIDeviceOrientationLandscapeRight:      // Device oriented horizontally, home button on the left
            if (self.isUsingFrontFacingCamera)
                exifOrientation = PHOTOS_EXIF_0ROW_TOP_0COL_LEFT;
            else
                exifOrientation = PHOTOS_EXIF_0ROW_BOTTOM_0COL_RIGHT;
            break;
        case UIDeviceOrientationPortrait:            // Device oriented vertically, home button on the bottom
        default:
            exifOrientation = PHOTOS_EXIF_0ROW_RIGHT_0COL_TOP;
            break;
    }
    return [NSNumber numberWithInt:exifOrientation];
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputMetadataObjects:(NSArray *)metadataObjects
       fromConnection:(AVCaptureConnection *)connection
{
    //NSLog(@"capture output");
    NSMutableArray * mObjects = [[NSMutableArray alloc] init];
    for (AVMetadataObject *metadataObject in metadataObjects)
    {
        //NSLog(@"found metadata");

        AVMetadataObject *transformedObject = [self.previewLayer transformedMetadataObjectForMetadataObject:metadataObject];
        [mObjects addObject:transformedObject];
    }
    
    dispatch_async(dispatch_get_main_queue(), ^(void)
    {
        [self clearDrawLayer];
        [self showDetectedObjects:mObjects];
    });
}

- (void)clearDrawLayer
{
    NSArray *sublayers = [self.drawLayer sublayers];
    for (CALayer *sublayer in sublayers)
    {
        [sublayer removeFromSuperlayer];
    }
}

- (void)showDetectedRects:(NSMutableArray*)objects
{
    for (AVMetadataObject *object in objects)
    {
        if ([object isKindOfClass:[AVMetadataFaceObject class]])
        {
            CAShapeLayer *shapeLayer = [CAShapeLayer layer];
            shapeLayer.strokeColor = [UIColor redColor].CGColor;
            shapeLayer.fillColor = [UIColor clearColor].CGColor;
            shapeLayer.lineWidth = 6.0;
            shapeLayer.lineJoin = kCALineJoinRound;
            
            // Create a rectangle path.
            UIBezierPath *path = [UIBezierPath bezierPathWithRect:
                                  object.bounds];
            
            // Set the path on the layer
            //CGPathRef pathRef = [path CGPath];//createPathForPoints([(AVMetadataMachineReadableCodeObject *)object corners]);
            shapeLayer.path = [path CGPath];
            //CFRelease(pathRef);
            [self.drawLayer addSublayer:shapeLayer];
        }
    }
}

- (void)showDetectedObjects:(NSMutableArray*)objects
{
    for (AVMetadataObject *object in objects)
    {
        if ([object isKindOfClass:[AVMetadataFaceObject class]])
        {
            CAShapeLayer *shapeLayer = [CAShapeLayer layer];
            shapeLayer.strokeColor = [UIColor redColor].CGColor;
            shapeLayer.fillColor = [UIColor clearColor].CGColor;
            shapeLayer.lineWidth = 6.0;
            shapeLayer.lineJoin = kCALineJoinRound;
            
            // Create a rectangle path.
            UIBezierPath *path = [UIBezierPath bezierPathWithRect:
                                  object.bounds];
            
            // Set the path on the layer
            //CGPathRef pathRef = [path CGPath];//createPathForPoints([(AVMetadataMachineReadableCodeObject *)object corners]);
            shapeLayer.path = [path CGPath];
            //CFRelease(pathRef);
            [self.drawLayer addSublayer:shapeLayer];
        }
    }
}
/*
static CGMutablePathRef createPathForPoints(NSArray* points)
{
    CGMutablePathRef path = CGPathCreateMutable();
    CGPoint point;
    if ([points count] > 0)
    {
        CGPointMakeWithDictionaryRepresentation((CFDictionaryRef)[points objectAtIndex:0], &point);
        CGPathMoveToPoint(path, nil, point.x, point.y);
        int i = 1;
        while (i < [points count])
        {
            CGPointMakeWithDictionaryRepresentation((CFDictionaryRef)[points objectAtIndex:i], &point);
            CGPathAddLineToPoint(path, nil, point.x, point.y);
            i++;
        }
        CGPathCloseSubpath(path);
    }
    return path;
}
*/
//static const double maxFramesToCountDouble = 1000.0;
//static const int maxFramesToCount = 1000;

//static int frameCount = 1000;

- (id<MTLTexture>)textureForImage:(UIImage *)image context:(MetalGPGPUContext *)context
{
    CGImageRef imageRef = [image CGImage];
    
    // Create a suitable bitmap context for extracting the bits of the image
    NSUInteger width = CGImageGetWidth(imageRef);
    NSUInteger height = CGImageGetHeight(imageRef);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    uint8_t *rawData = (uint8_t *)calloc(height * width * 4, sizeof(uint8_t));
    NSUInteger bytesPerPixel = 4;
    NSUInteger bytesPerRow = bytesPerPixel * width;
    NSUInteger bitsPerComponent = 8;
    CGContextRef bitmapContext = CGBitmapContextCreate(rawData, width, height,
                                                       bitsPerComponent, bytesPerRow, colorSpace,
                                                       kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(colorSpace);
    
    // Flip the context so the positive Y axis points down
    CGContextTranslateCTM(bitmapContext, 0, height);
    CGContextScaleCTM(bitmapContext, 1, -1);
    
    CGContextDrawImage(bitmapContext, CGRectMake(0, 0, width, height), imageRef);
    CGContextRelease(bitmapContext);
    
    MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                                 width:width
                                                                                                height:height
                                                                                             mipmapped:NO];
    id<MTLTexture> texture = [context.device newTextureWithDescriptor:textureDescriptor];
    
    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [texture replaceRegion:region mipmapLevel:0 withBytes:rawData bytesPerRow:bytesPerRow];
    
    free(rawData);
    
    return texture;
}

-(void)writeImageToDisk:(UIImage*)image
{

    NSData *pngData = UIImagePNGRepresentation(image);

    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsPath = [paths objectAtIndex:0]; //Get the docs directory
    NSString *filePath = [documentsPath stringByAppendingPathComponent:@"image.png"]; //Add the file name
    [pngData writeToFile:filePath atomically:YES]; //Write the file
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection
{
    //return;
    //NSLog(@"Did Output Sample Buffer");
    /*
     *  If you are drawing on top of the video render layer
     *  You may want to clear the draw layer on the main thread here
     *  Before processing and drawing a new frame and draw layer
     */
/*
    dispatch_async(dispatch_get_main_queue(), ^(void) {
        
        [self clearDrawLayer];
        
    });
*/
    //AVCaptureDeviceFormat * vFormat = self.captureDevice.activeFormat;
    //NSLog(@"Active Video Format:  %@ \n\n%@ \n%@\n",vFormat.mediaType,vFormat.formatDescription,vFormat.videoSupportedFrameRateRanges);

    /*
     * Check the device orientation
     * Later this can be moved to its own update event so it doesn't need to be checked every frame
     * Though overhead involved with this is already probably very low
     */
     //currentDeviceOrientation = [[UIDevice currentDevice] orientation];

    CVReturn error;
    
    CVImageBufferRef sourceImageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    size_t width = CVPixelBufferGetWidth(sourceImageBuffer);
    size_t height = CVPixelBufferGetHeight(sourceImageBuffer);
    
    
    //NSLog(@"capture width:  %d, capture height: %d", width, height);
    CVMetalTextureRef textureRef;
    error = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, _videoTextureCache, sourceImageBuffer, NULL, MTLPixelFormatBGRA8Unorm, 1024, 1024, 0, &textureRef);
    
    if (error)
    {
        NSLog(@">> ERROR: Couldnt create texture from image");
        assert(0);
    }
    
    _texture = CVMetalTextureGetTexture(textureRef);//_testPattern;
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{

        /*
         * Measure the time between fft calculations
         */
        double newFrameTime = machGetClockS();
        //double frameInterval = newFrameTime - prevFrameTime;
        //NSLog(@"frame interval: %g", newFrameTime - prevFrameTime );
        //prevFrameTime = newFrameTime;
        


    //self.desaturateFilter.saturationFactor = 0.5;
    
    
    /*
    id<MTLTexture> filteredTexture = self.fftFilter.texture;
    
        if( filteredTexture )
        {
            
            int imageWidth = (int)[_videoTexture[_constantDataBufferIndex] width];
            int imageHeight = (int)[_videoTexture[_constantDataBufferIndex] width];
            size_t imageByteCount = imageWidth * imageHeight * 4;
            void *imageBytes = malloc(imageByteCount);
            NSUInteger bytesPerRow = imageWidth * 4;
            MTLRegion region = MTLRegionMake2D(0, 0, imageWidth, imageHeight);
            [_videoTexture[_constantDataBufferIndex] getBytes:imageBytes bytesPerRow:bytesPerRow fromRegion:region mipmapLevel:0];
            
            cv::Mat realOutput((int)[_videoTexture[_constantDataBufferIndex] height] , (int)[_videoTexture[_constantDataBufferIndex] width], CV_32FC1, imageBytes, sizeof(float)*[_videoTexture[_constantDataBufferIndex] width]);
            
            cv::normalize(realOutput, realOutput, 0, 1, CV_MINMAX);
            
            _videoTexture[_constantDataBufferIndex] = filteredTexture;

        }
    */
    
        _videoTexture[_constantDataBufferIndex] = self.fftFilter.texture;

    //_videoBuffer[_constantDataBufferIndex] = self.fftFilter.buffer;
    //_videoTexture[_constantDataBufferIndex] = CVMetalTextureGetTexture(textureRef);
    if (!_videoTexture[_constantDataBufferIndex]) {
        NSLog(@">> ERROR: Couldn't get texture from texture ref");
        assert(0);
    }

    
        double newerFrameTime = machGetClockS();
        double frameInterval = newerFrameTime - newFrameTime;
        //update the frame rate label asynchronously on the main thread so it doesn't hold up any code that comes after this
        dispatch_async(dispatch_get_main_queue(), ^{
            _frameRateLabel.text = [NSString stringWithFormat:@"Frame Rate:\t%.2f ms", (float)(frameInterval)];
        });
    //}
    //else
    //{
     //   UIImage *uiimage = [UIImage imageWithMTLTexture:_videoTexture[_constantDataBufferIndex]];
     //   if( uiimage )
      //      [self writeImageToDisk:uiimage];
    //}
    //else
    //    NSLog(@"out texture width:  %lu, height: %lu", (unsigned long)[_videoTexture[_constantDataBufferIndex] width], (unsigned long)[_videoTexture[_constantDataBufferIndex] height]);
    });

    CVBufferRelease(textureRef);

    /*
     * Here we average the frame latency over time to get the avg frame rate
     * Comment out when not in use
     */
/*
    if( frameCount == maxFramesToCount )
    {
        
        double currFrameTime = machGetClockS();
        NSLog(@"avg frame time:  %g ms", (currFrameTime - prevFrameTime)/maxFramesToCountDouble) ;

        //the current frame time will become the previous frame time
        prevFrameTime = currFrameTime;
        //reset the frame count
        frameCount = 0;


    }
    else
        frameCount++;  //increment the frame count, then let processing occur
*/


    //if (dispatch_semaphore_wait(frameRenderingSemaphore, DISPATCH_TIME_NOW) != 0)
    //{
    //    return;
    //}
    
    //CFRetain(sampleBuffer);
    //runAsynchronouslyOnVideoProcessingQueue(
    //^{
        //Feature Detection Hook.
        //if (self.delegate)
        //{
        //    [self.delegate willOutputSampleBuffer:sampleBuffer];
        //}
        
        //[self processVideoSampleBuffer:sampleBuffer];
        

        //process the buffer
        /*
        // convert from Core Media buffer to Core Video buffer, mostly meaningless
        // CVImageBufferRef is a typedef that is a pointer to a c-style buffer
        CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        //presumably, LockBaseAddress provides a mutex lock around the imageBuffer memory location
        //while this async background cpu processing thread accesses it
        CVPixelBufferLockBaseAddress(imageBuffer, 0);
        
        void* bufferAddress;
        //uint8_t * rawData;
        size_t width;
        size_t height;
        size_t bytesPerRow;
        
        //int format_opencv;
        
        bool grayscale = false;
        OSType format = CVPixelBufferGetPixelFormatType(imageBuffer);
        if (format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange || format == '420f')
        {
            
            //format_opencv = CV_8UC1;
            //NSLog(@"hello i am 8 bit");
            grayscale = true;
            bufferAddress = CVPixelBufferGetBaseAddressOfPlane(imageBuffer, 0);
            //rawData = (uint8_t*)CVPixelBufferGetBaseAddressOfPlane(imageBuffer, 0);

            width = CVPixelBufferGetWidthOfPlane(imageBuffer, 0);
            height = CVPixelBufferGetHeightOfPlane(imageBuffer, 0);
            bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(imageBuffer, 0);
            
        }
        else // expect kCVPixelFormatType_32BGRA
        {
            //NSLog(@"hello i am 32 bit");
            //format_opencv = CV_8UC4;
            
            bufferAddress = CVPixelBufferGetBaseAddress(imageBuffer);
            //rawData = (uint8_t*)CVPixelBufferGetBaseAddress(imageBuffer);

            width = CVPixelBufferGetWidth(imageBuffer);
            height = CVPixelBufferGetHeight(imageBuffer);
            bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
            
        }
        

        //set the current metal context, similar to opengl context if necessary
        //[MetalGPGPUContext useImageProcessingContext];

        
        //bool usingPortraitOrientation = false;
        //if( currentDeviceOrientation == UIDeviceOrientationPortrait )
        //{
            //swap the width and height variables that will be used to create the cv::Matrix image object
            //size_t tmp = height;
            //height = width;
            //width = tmp;
            //usingPortraitOrientation = true;

       // }
        
        int widthInt = (int)width;
        int heightInt = (int)height;
        
        
        //first, get the optimal discrete fourier transform buffer sizes as a power of 2
        // actually, opencv isn't using powers of 2, which would be optimal
        // e.g. 352x288 gets an optimal dft size of 360x288
        //int fftWidth = cv::getOptimalDFTSize(widthInt);
        //int fftHeight = cv::getOptimalDFTSize(heightInt);
        
        
        //if you use a custom video preview layer, you can draw the cvmat to it after you've processed
        //otherwise, you can just draw on top of the preview layer
        //in our case we just wanted to get the results of phase correlation across every two consecutive frames
        //so only the camera output needs to be drawn to the screen, which AVVideoDataOutput is taking care of for us automatically
        if( self.customPreviewLayer  )
        {
            //CGImage* dstImage;
            
            /*
            // (create color space, create graphics context, render buffer)
            CGBitmapInfo bitmapInfo;
            CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
            CGContextRef context;
            
            BOOL iOSimage = YES;
            
            // basically we decide if it's a grayscale, rgb or rgba image
            if (grayscale)
            {
                //NSLog(@"Hello.  I am gray");
                colorSpace = CGColorSpaceCreateDeviceGray();
                bitmapInfo = kCGImageAlphaNone;
            }
            else if (!grayscale)
            {
                //NSLog(@"Hello.  I am Color");
                
                colorSpace = CGColorSpaceCreateDeviceRGB();
                bitmapInfo = kCGImageAlphaNone;
                if (iOSimage) {
                    bitmapInfo |= kCGBitmapByteOrder32Little;
                } else {
                    bitmapInfo |= kCGBitmapByteOrder32Big;
                }
            }
            else
            {
                //NSLog(@"Hello.  I am undefined");

                colorSpace = CGColorSpaceCreateDeviceRGB();
                bitmapInfo = kCGImageAlphaPremultipliedFirst;
                if (iOSimage) {
                    bitmapInfo |= kCGBitmapByteOrder32Little;
                } else {
                    bitmapInfo |= kCGBitmapByteOrder32Big;
                }
            }
            
            
            //create a bitmpa context with the existing buffer data
            context = CGBitmapContextCreate(bufferAddress, width, height, 8, bytesPerRow, colorSpace, bitmapInfo);

            // Flip the context so the positive Y axis points down
            CGContextTranslateCTM(context, 0, height);
            CGContextScaleCTM(context, 1, -1);
            
            //createa a cgimage from the bitmap context raw data
            //CGImage * dstImage = CGBitmapContextCreateImage(context);
            
            CGContextRelease(context);
         
            
            //following brad larson gpu image
            
            //if (dispatch_semaphore_wait(frameRenderingSemaphore, DISPATCH_TIME_NOW) != 0)
            //{
            //    return;
            //}
            
            
            
            MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                                         width:width
                                                                                                        height:height
                                                                                                     mipmapped:NO];
            id<MTLTexture> texture = [self.context.device newTextureWithDescriptor:textureDescriptor];
            
            MTLRegion region = MTLRegionMake2D(0, 0, width, height);
            [texture replaceRegion:region mipmapLevel:0 withBytes:bufferAddress bytesPerRow:bytesPerRow];
            
            _texture = texture;
            
            
            //self.desaturateFilter.saturationFactor = 0.5;
            id<MTLTexture> filteredTexture = self.blurFilter.texture;
            */
            
            //NSLog(@"imageWithMTLTexture");
            //UIImage *uiimage = [UIImage imageWithMTLTexture:filteredTexture];
             
            //free(rawData);
            
            //draw a CGImage to the custom video preview layer, if one exists
            // All drawing to the screen must occur on the main thread
            
            /*
            if( uiimage )
            {
            dispatch_sync(dispatch_get_main_queue(), ^{
                self.customPreviewLayer.contents = (__bridge id)uiimage.CGImage;
            });
            }
            */
            
            // cleanup
            //CGImageRelease(dstImage);
            //CGColorSpaceRelease(colorSpace);
            
            
            
        //}
        

        

        //if we are all done processing
        //store the previous imageBuffer for processing every two consecutive frames
        //retain the c memory object before we exit the function because it is in a memory pool
        //and we want to use it on the next frame
        //CFRetain(sampleBuffer);
        //prevSampleBuffer = sampleBuffer;
        
        //CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
    
        //CFRelease(sampleBuffer);
        //dispatch_semaphore_signal(frameRenderingSemaphore);
    //});

    
    
    //Brian, this following Core Image stuff is extraneous to OpenCV processing procedures, but you may want to know how it works
    /* CIImage processing */

    //If we want to use CIDectector, get a Core Image object representation of the camera video image data
    //CIImage *ciImage = [[CIImage alloc] initWithCVPixelBuffer:pixelBuffer options:(__bridge NSDictionary *)attachments];
    
    //However, we want to use OpenCV Matrix object and CGImage is the most appropriate for converting to cv::Mat
    
    
    //must release all c-style memory manually
    //if (attachments) {
    //    CFRelease(attachments);
    //}
    
    // make sure your device orientation is not locked.
    //UIDeviceOrientation currentDeviceOrientation = [[UIDevice currentDevice] orientation];
    
    //NSDictionary *imageOptions = nil;
    
    //get the correct image orientation for cidetector ciimage format based on the device orientation that
    //the raw camera data was captured with
    //imageOptions = [NSDictionary dictionaryWithObject:[self exifOrientation:currentDeviceOrientation]
    //                                           forKey:CIDetectorImageOrientation];
    
    
    //get the features the cidetector finds in the ciimage
    //NSArray *features = [self.featDetector featuresInImage:ciImage
    //                                              options:imageOptions];
    
    // get the clean aperture
    // the clean aperture is a rectangle that defines the portion of the encoded pixel dimensions
    // that represents image data valid for display.
    //CMFormatDescriptionRef fdesc = CMSampleBufferGetFormatDescription(sampleBuffer);
    //CGRect cleanAperture = CMVideoFormatDescriptionGetCleanAperture(fdesc, false /*originIsTopLeft == false*/);
    
    /* End CIImage Processing */
    
    

}

// Create a UIImage from sample buffer data
- (UIImage *) imageFromSampleBuffer:(CMSampleBufferRef) sampleBuffer
{
    // Get a CMSampleBuffer's Core Video image buffer for the media data
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    // Lock the base address of the pixel buffer
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    
    // Get the number of bytes per row for the pixel buffer
    void *baseAddress = CVPixelBufferGetBaseAddress(imageBuffer);
    
    // Get the number of bytes per row for the pixel buffer
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    // Get the pixel buffer width and height
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    
    // Create a device-dependent RGB color space
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    // Create a bitmap graphics context with the sample buffer data
    CGContextRef context = CGBitmapContextCreate(baseAddress, width, height, 8,
                                                 bytesPerRow, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    // Create a Quartz image from the pixel data in the bitmap graphics context
    CGImageRef quartzImage = CGBitmapContextCreateImage(context);
    // Unlock the pixel buffer
    CVPixelBufferUnlockBaseAddress(imageBuffer,0);
    
    // Free up the context and color space
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    
    // Create an image object from the Quartz image
    UIImage *image = [UIImage imageWithCGImage:quartzImage];
    
    // Release the Quartz image
    CGImageRelease(quartzImage);
    
    return (image);
}


// find where the video box is positioned within the preview layer based on the video size and gravity
- (CGRect)videoPreviewBoxForGravity:(NSString *)gravity
                          frameSize:(CGSize)frameSize
                       apertureSize:(CGSize)apertureSize
{
    CGFloat apertureRatio = apertureSize.height / apertureSize.width;
    CGFloat viewRatio = frameSize.width / frameSize.height;
    
    CGSize size = CGSizeZero;
    if ([gravity isEqualToString:AVLayerVideoGravityResizeAspectFill]) {
        if (viewRatio > apertureRatio) {
            size.width = frameSize.width;
            size.height = apertureSize.width * (frameSize.width / apertureSize.height);
        } else {
            size.width = apertureSize.height * (frameSize.height / apertureSize.width);
            size.height = frameSize.height;
        }
    } else if ([gravity isEqualToString:AVLayerVideoGravityResizeAspect]) {
        if (viewRatio > apertureRatio) {
            size.width = apertureSize.height * (frameSize.height / apertureSize.width);
            size.height = frameSize.height;
        } else {
            size.width = frameSize.width;
            size.height = apertureSize.width * (frameSize.width / apertureSize.height);
        }
    } else if ([gravity isEqualToString:AVLayerVideoGravityResize]) {
        size.width = frameSize.width;
        size.height = frameSize.height;
    }
    
    CGRect videoBox;
    videoBox.size = size;
    if (size.width < frameSize.width)
        videoBox.origin.x = (frameSize.width - size.width) / 2;
    else
        videoBox.origin.x = (size.width - frameSize.width) / 2;
    
    if ( size.height < frameSize.height )
        videoBox.origin.y = (frameSize.height - size.height) / 2;
    else
        videoBox.origin.y = (size.height - frameSize.height) / 2;
    
    return videoBox;
}

// called asynchronously as the capture output is capturing sample buffers, this method asks the face detector
// to detect features and for each draw the green border in a layer and set appropriate orientation
- (void)drawFaces:(NSArray *)features
      forVideoBox:(CGRect)clearAperture
      orientation:(UIDeviceOrientation)orientation
       connection:(AVCaptureConnection *)connection
{
    NSArray *sublayers = [NSArray arrayWithArray:[self.previewLayer sublayers]];
    NSInteger sublayersCount = [sublayers count], currentSublayer = 0;
    NSInteger featuresCount = [features count], currentFeature = 0;
    
    [CATransaction begin];
    [CATransaction setValue:(id)kCFBooleanTrue forKey:kCATransactionDisableActions];
    
    /*// hide all the face layers
    for ( CALayer *layer in sublayers ) {
        if ( [[layer name] isEqualToString:@"FaceLayer"] )
            [layer setHidden:YES];
    }
    */
    if ( featuresCount == 0 ) {
        [CATransaction commit];
        return; // early bail.
    }
    
    CGSize parentFrameSize = [self.view frame].size;
    NSString *gravity = [self.previewLayer videoGravity];
    
    BOOL isMirrored;
    if( [connection respondsToSelector:@selector(isVideoMirrored)] )
    {
        //NSLog(@"AVCapture connection responds to selector(isVideoMirrored)");
        isMirrored = ![connection isVideoMirrored];
    }
    else
        isMirrored = [self.previewLayer isMirrored];
    
    CGRect previewBox = [self videoPreviewBoxForGravity:gravity
                                              frameSize:parentFrameSize
                                           apertureSize:clearAperture.size];
    
    for ( CIRectangleFeature *ff in features ) {
        // find the correct position for the square layer within the previewLayer
        // the feature box originates in the bottom left of the video frame.
        // (Bottom right if mirroring is turned on)
        CGRect faceRect = [ff bounds];
        
        // flip preview width and height
        CGFloat temp = faceRect.size.width;
        faceRect.size.width = faceRect.size.height;
        faceRect.size.height = temp;
        temp = faceRect.origin.x;
        faceRect.origin.x = faceRect.origin.y;
        faceRect.origin.y = temp;
        // scale coordinates so they fit in the preview box, which may be scaled
        CGFloat widthScaleBy = previewBox.size.width / clearAperture.size.height;
        CGFloat heightScaleBy = previewBox.size.height / clearAperture.size.width;
        faceRect.size.width *= widthScaleBy;
        faceRect.size.height *= heightScaleBy;
        faceRect.origin.x *= widthScaleBy;
        faceRect.origin.y *= heightScaleBy;
        
        if ( isMirrored )
            faceRect = CGRectOffset(faceRect, previewBox.origin.x + previewBox.size.width - faceRect.size.width - (faceRect.origin.x * 2), previewBox.origin.y);
        else
            faceRect = CGRectOffset(faceRect, previewBox.origin.x, previewBox.origin.y);
        
        CALayer *featureLayer = nil;
        
        // re-use an existing layer if possible
        while ( !featureLayer && (currentSublayer < sublayersCount) ) {
            CALayer *currentLayer = [sublayers objectAtIndex:currentSublayer++];
            if ( [[currentLayer name] isEqualToString:@"FaceLayer"] ) {
                featureLayer = currentLayer;
                [currentLayer setHidden:NO];
            }
        }
        
        // create a new one if necessary
        if ( !featureLayer ) {
            featureLayer = [[CALayer alloc]init];
            featureLayer.borderWidth = 5.0;
            featureLayer.borderColor = [UIColor redColor].CGColor;
            //featureLayer.contents = (id)self.borderImage.CGImage;
            [featureLayer setName:@"FaceLayer"];
            [self.previewLayer addSublayer:featureLayer];
            featureLayer = nil;
        }
        [featureLayer setFrame:faceRect];
        
        switch (orientation) {
            case UIDeviceOrientationPortrait:
                [featureLayer setAffineTransform:CGAffineTransformMakeRotation(DegreesToRadians(0.))];
                break;
            case UIDeviceOrientationPortraitUpsideDown:
                [featureLayer setAffineTransform:CGAffineTransformMakeRotation(DegreesToRadians(180.))];
                break;
            case UIDeviceOrientationLandscapeLeft:
                [featureLayer setAffineTransform:CGAffineTransformMakeRotation(DegreesToRadians(90.))];
                break;
            case UIDeviceOrientationLandscapeRight:
                [featureLayer setAffineTransform:CGAffineTransformMakeRotation(DegreesToRadians(-90.))];
                break;
            case UIDeviceOrientationFaceUp:
            case UIDeviceOrientationFaceDown:
            default:
                break; // leave the layer in its last known orientation
        }
        currentFeature++;
    }
    
    
    [CATransaction commit];
}


// called asynchronously as the capture output is capturing sample buffers, this method asks the face detector
// to detect features and for each draw the green border in a layer and set appropriate orientation
- (CGRect)drawRects:(NSArray *)features
      forVideoBox:(CGRect)clearAperture
      orientation:(UIDeviceOrientation)orientation
      connection:(AVCaptureConnection *)connection
{
    

    __block NSArray *sublayers = [NSArray arrayWithArray:[self.previewLayer sublayers]];
    __block NSInteger sublayersCount = [sublayers count];
    //NSInteger currentSublayer = 0;
    NSInteger featuresCount = [features count], currentFeature = 0;

    //[CATransaction begin];
    //[CATransaction setValue:(id)kCFBooleanTrue forKey:kCATransactionDisableActions];
    
    // hide all the face layers
    
    //dispatch_async(dispatch_get_main_queue(), ^(void) {

    for ( CALayer *layer in sublayers ) {
        if ( [[layer name] isEqualToString:@"FaceLayer"] )
            [layer setHidden:YES];
    }
    
    //});
    
    if ( featuresCount == 0 ) {
        //[CATransaction commit];
        return CGRectZero; // early bail.
    }

    CGSize parentFrameSize = [self.view frame].size;
    NSString *gravity = [self.previewLayer videoGravity];

    BOOL isMirrored;
    if( [connection respondsToSelector:@selector(isVideoMirrored)] )
    {
        //NSLog(@"AVCapture connection responds to selector(isVideoMirrored)");
        isMirrored = ![connection isVideoMirrored];
    }
    else
        isMirrored = [self.previewLayer isMirrored];
    
    CGRect previewBox = [self videoPreviewBoxForGravity:gravity
                                                        frameSize:parentFrameSize
                                                     apertureSize:clearAperture.size];
    
    
    CGFloat max_width = 0;
    CGFloat max_height = 0;
    
    CGRect largestRectAdjusted;
    CGRect largestRect;
    for ( CIRectangleFeature *ff in features ) {
        // find the correct position for the square layer within the previewLayer
        // the feature box originates in the bottom left of the video frame.
        // (Bottom right if mirroring is turned on)
        CGRect faceRect = [ff bounds];
        largestRect = faceRect;
        
        // flip preview width and height
        CGFloat temp = faceRect.size.width;
        faceRect.size.width = faceRect.size.height;
        faceRect.size.height = temp;
        temp = faceRect.origin.x;
        faceRect.origin.x = faceRect.origin.y;
        faceRect.origin.y = temp;
        // scale coordinates so they fit in the preview box, which may be scaled
        CGFloat widthScaleBy = previewBox.size.width / clearAperture.size.height;
        CGFloat heightScaleBy = previewBox.size.height / clearAperture.size.width;
        faceRect.size.width *= widthScaleBy;
        faceRect.size.height *= heightScaleBy;
        faceRect.origin.x *= widthScaleBy;
        faceRect.origin.y *= heightScaleBy;
        
        //if ( isMirrored )
        //    faceRect = CGRectOffset(faceRect, previewBox.origin.x + previewBox.size.width - faceRect.size.width - (faceRect.origin.x * 2), previewBox.origin.y);
        //else
        
        if( gravity == AVLayerVideoGravityResizeAspectFill )
            faceRect = CGRectOffset(faceRect, -1.0*previewBox.origin.x, previewBox.origin.y);
        else
            faceRect = CGRectOffset(faceRect, previewBox.origin.x, previewBox.origin.y);

        /*
        CALayer *featureLayer = nil;
        
        // re-use an existing layer if possible
        while ( !featureLayer && (currentSublayer < sublayersCount) ) {
            CALayer *currentLayer = [sublayers objectAtIndex:currentSublayer++];
            if ( [[currentLayer name] isEqualToString:@"FaceLayer"] ) {
                featureLayer = currentLayer;
                [currentLayer setHidden:NO];
            }
        }
        
        // create a new one if necessary
        if ( !featureLayer ) {
            featureLayer = [[CALayer alloc]init];
            featureLayer.borderWidth = 5.0;
            featureLayer.borderColor = [UIColor redColor].CGColor;
            //featureLayer.contents = (id)self.borderImage.CGImage;
            [featureLayer setName:@"FaceLayer"];
            [self.previewLayer addSublayer:featureLayer];
            featureLayer = nil;
        }
        [featureLayer setFrame:faceRect];
        */
        if ((faceRect.size.width >= max_width) && (faceRect.size.height >= max_height))
        {
            largestRectAdjusted = faceRect;
            max_width = faceRect.size.width;
            max_height = faceRect.size.height;
        }
        

        currentFeature++;
    }
    
    //draw the largest rect
    
    //if( largestRectAdjusted.size.width > _adjustRect.frame.size.width*0.75 && largestRectAdjusted.size.height > _adjustRect.frame.size.height*0.75)
    //{
    
        //dispatch_async(dispatch_get_main_queue(), ^(void) {
            
            /*
            [self clearDrawLayer];
            
            CAShapeLayer *featureLayer = [CAShapeLayer layer];
            featureLayer.strokeColor = [UIColor colorWithRed:0.0 green:122.0/255.0 blue:1.0 alpha:1.0].CGColor;
            featureLayer.fillColor = [UIColor clearColor].CGColor;
            featureLayer.lineWidth = 3.0;
            //featureLayer.lineJoin = kCALineJoinRound;
            
            // Create a rectangle path.
            UIBezierPath *path = [UIBezierPath bezierPathWithRect:largestRectAdjusted];
            
            // Set the path on the layer
            //CGPathRef pathRef = [path CGPath];//createPathForPoints([(AVMetadataMachineReadableCodeObject *)object corners]);
            featureLayer.path = [path CGPath];
            //CFRelease(pathRef);
            [self.drawLayer addSublayer:featureLayer];
            */
            
            CALayer *featureLayer = nil;
            int sublayerIndex = 0;
            // re-use an existing layer if possible
            while ( !featureLayer && (sublayerIndex < sublayersCount) ) {
                
                CALayer *currentLayer = [sublayers objectAtIndex:sublayerIndex++];
                if ( [[currentLayer name] isEqualToString:@"FaceLayer"] ) {
                    featureLayer = currentLayer;
                    [currentLayer setHidden:NO];
                }
            }
            
            // create a new one if necessary
            if ( !featureLayer ) {
                featureLayer = [[CALayer alloc]init];
                featureLayer.borderWidth = 4.0;
                featureLayer.borderColor = [UIColor colorWithRed:0.0 green:122.0/255.0 blue:1.0 alpha:1.0].CGColor;
                //featureLayer.contents = (id)self.borderImage.CGImage;
                [featureLayer setName:@"FaceLayer"];
                [self.previewLayer addSublayer:featureLayer];
                featureLayer = nil;
            }
            [featureLayer setFrame:largestRectAdjusted];
            
            switch (orientation) {
                case UIDeviceOrientationPortrait:
                    [featureLayer setAffineTransform:CGAffineTransformMakeRotation(DegreesToRadians(0.))];
                    break;
                case UIDeviceOrientationPortraitUpsideDown:
                    [featureLayer setAffineTransform:CGAffineTransformMakeRotation(DegreesToRadians(180.))];
                    break;
                case UIDeviceOrientationLandscapeLeft:
                    [featureLayer setAffineTransform:CGAffineTransformMakeRotation(DegreesToRadians(90.))];
                    break;
                case UIDeviceOrientationLandscapeRight:
                    [featureLayer setAffineTransform:CGAffineTransformMakeRotation(DegreesToRadians(-90.))];
                    break;
                case UIDeviceOrientationFaceUp:
                case UIDeviceOrientationFaceDown:
                default:
                    break; // leave the layer in its last known orientation
            }
        //});

            return largestRectAdjusted;

    //}

    
    return CGRectZero;
    
    //if(
   /// else
    //   return NO;
    //[CATransaction commit];
}

#pragma mark -- Perspective Adjust View Controller

/*
- (void)transitionToPerspectiveAdjustViewController
{
    [self.videoCaptureSession stopRunning];
    
    PerspectiveAdjustViewController *adjustViewController = [[PerspectiveAdjustViewController alloc] init];
    adjustViewController.delegate = self;
    NSLog(@"Still Image Size:   %dx%d", (int)self.stillImage.size.width, (int)self.stillImage.size.height);

    adjustViewController.sourceImage = self.stillImage;
    
    [UIView animateWithDuration:0.05 delay:0.0 options:UIViewAnimationOptionCurveEaseInOut animations:^
     {
         _cameraPictureTakenFlash.alpha = 0.5f;
     }
                     completion:^(BOOL finished)
     {
         [UIView animateWithDuration:0.1 delay:0.0 options:UIViewAnimationOptionCurveEaseInOut animations:^
          {
              _cameraPictureTakenFlash.alpha = 0.0f;
          }
                          completion:^(BOOL finished)
          {
              CATransition* transition = [CATransition animation];
              transition.duration = 0.4;
              transition.type = kCATransitionFade;
              transition.subtype = kCATransitionFromBottom;
              [self.navigationController.view.layer addAnimation:transition forKey:kCATransition];
              [self.navigationController pushViewController:adjustViewController animated:NO];
          }];
     }];
}

- (void) dismissedWithAdjustedImage:(UIImage*)adjustedImage
{
    [self popToPreviousViewController];
    NSLog(@"dismissedWithAdjustedImage");
    if(_delegate && [_delegate respondsToSelector:@selector(dismissedWithScannedImage:)] )
        [_delegate dismissedWithScannedImage:adjustedImage];
}
*/

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

/*
#pragma mark - Navigation

// In a storyboard-based application, you will often want to do a little preparation before navigation
- (void)prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender {
    // Get the new view controller using [segue destinationViewController].
    // Pass the selected object to the new view controller.
}
*/

@end
