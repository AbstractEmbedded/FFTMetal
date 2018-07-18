//
//  ViewController.m
//  FFTMetal
//
//  Created by Joe Moulton on 1/25/18.
//  Copyright © 2018 Abstract Embedded. All rights reserved.
//

#import "ViewController.h"
#import "LiveCameraViewController.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    
    [self customizeNavBar];
}

-(void)customizeNavBar
{
    
    //customize view controller nav bar appearance
    [self.navigationController.navigationBar setBackgroundImage:[UIImage new]
                                                  forBarMetrics:UIBarMetricsDefault]; //UIImageNamed:@"transparent.png"
    self.navigationController.navigationBar.shadowImage = [UIImage new];////UIImageNamed:@"transparent.png"
    self.navigationController.navigationBar.translucent = YES;
    self.navigationController.view.backgroundColor = [UIColor clearColor];
    
    //self.navigationController.navigationBar.barTintColor = [UIColor whiteColor];
    //self.navigationController.navigationBar.tintColor = [UIColor whiteColor];
    
    
    //customize view controller title
    self.navigationItem.title = @"FFT Metal Demo";
    [self.navigationController.navigationBar setTitleTextAttributes:
     @{NSForegroundColorAttributeName:self.view.tintColor}];
    
    //add custom bar button items
    //create a nav bar button, when pressed we will transition to our opengl view controller
    //self.navigationItem.rightBarButtonItem = [[UIBarButtonItem alloc] initWithTitle:@"Start" style:UIBarButtonItemStylePlain target:self action:@selector(startSOM)];
    UIBarButtonItem * oglButton = [[UIBarButtonItem alloc] initWithTitle:@"Start" style:UIBarButtonItemStylePlain target:self action:@selector(startButtonPressed:)];
    [oglButton setTitleTextAttributes:@{
                                        NSFontAttributeName: [UIFont systemFontOfSize:18.0],
                                        //NSForegroundColorAttributeName: [UIColor greenColor]
                                        } forState:UIControlStateNormal];
    
    //UIBarButtonItem * dummyButton = [[UIBarButtonItem alloc] initWithTitle:@"Start" style:UIBarButtonItemStylePlain target:self action:nil];
    
    self.navigationItem.rightBarButtonItem = oglButton;
    
    
    
}

-(void)startButtonPressed:(id)sender
{
    [self transitionToOpenGLVC];
    
    
}

-(void)transitionToOpenGLVC
{
    //set the navigation title to show back button when we push the new view controller
    self.navigationItem.title = @"Back";
    LiveCameraViewController * liveCameraVC = [[LiveCameraViewController alloc] init];
    [self.navigationController pushViewController:liveCameraVC animated:YES];
    
}

- (void)didReceiveMemoryWarning
{
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
