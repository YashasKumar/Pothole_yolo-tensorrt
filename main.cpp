#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolo.h"
#include "sort.h"
#include "logging.h"

using namespace cv;
using namespace std;
using namespace sort;

static Logger logger;

// Generate random colors for visualization
vector<Scalar> generateColors(int numColors) {
    RNG rng(numColors);
    vector<Scalar> colors;
    for (int i = 0; i < numColors; ++i) {
        colors.push_back(Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
    }
    return colors;
}

// Convert YOLO DetectRes format to SORT format
// Convert YOLO DetectRes format to SORT format
Mat convertDetectionsToSort(const vector<DetectRes>& detections) {
    if (detections.empty()) {
        return Mat(0, 5, CV_32F);  // Changed from 6 to 5 columns
    }
    
    Mat sortInput(detections.size(), 5, CV_32F);  // [x1, y1, x2, y2, score]
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        
        // Convert center format (xc, yc, w, h) to corner format (x1, y1, x2, y2)
        float x1 = det.x - det.w / 2.0f;
        float y1 = det.y - det.h / 2.0f;
        float x2 = det.x + det.w / 2.0f;
        float y2 = det.y + det.h / 2.0f;
        
        sortInput.at<float>(i, 0) = x1;
        sortInput.at<float>(i, 1) = y1;
        sortInput.at<float>(i, 2) = x2;
        sortInput.at<float>(i, 3) = y2;
        sortInput.at<float>(i, 4) = det.prob;
    }
    
    return sortInput;
}


// Draw tracked detections
void drawTrackedDetections(Mat& img, const Mat& trackedBboxes, const vector<Scalar>& colors) {
    for (int i = 0; i < trackedBboxes.rows; ++i) {
        // SORT returns: [x1, y1, x2, y2, vx, vy, s, tracker_id]
        float x1 = trackedBboxes.at<float>(i, 0);
        float y1 = trackedBboxes.at<float>(i, 1);
        float x2 = trackedBboxes.at<float>(i, 2);
        float y2 = trackedBboxes.at<float>(i, 3);
        int trackerId = static_cast<int>(trackedBboxes.at<float>(i, 7));
        
        // Draw bounding box
        Rect box(x1, y1, x2 - x1, y2 - y1);
        Scalar color = colors[trackerId % colors.size()];
        rectangle(img, box, color, 2);
        
        // Draw label
        string label = "ID: " + to_string(trackerId);
        int baseline = 0;
        Size textSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        
        // Draw background for text
        rectangle(img, 
                 Point(box.x, box.y - textSize.height - 10),
                 Point(box.x + textSize.width, box.y),
                 color, FILLED);
        
        // Draw text
        putText(img, label, Point(box.x, box.y - 5), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
    }
}


void printUsage(const char* programName) {
    cout << "\n==================================================" << endl;
    cout << "YOLO + SORT Object Tracking with TensorRT" << endl;
    cout << "==================================================" << endl;
    cout << "\nUsage:" << endl;
    cout << "\n1. Build TensorRT Engine:" << endl;
    cout << "  " << programName << " --build-engine -o <onnx_path> -e <engine_output_path>" << endl;
    cout << "\n2. Run Inference:" << endl;
    cout << "  " << programName << " --run -v <video_path> -e <engine_path>" << endl;
    cout << "\nOptions:" << endl;
    cout << "  --build-engine         Build TensorRT engine from ONNX" << endl;
    cout << "  --run                  Run inference on video" << endl;
    cout << "  -v, --video <path>     Video file path (required for --run)" << endl;
    cout << "  -o, --onnx <path>      ONNX model path (required for --build-engine)" << endl;
    cout << "  -e, --engine <path>    TensorRT engine path (required)" << endl;
    cout << "  -h, --help             Show this help message" << endl;
    cout << "\nExamples:" << endl;
    cout << "\n  # Build engine from ONNX:" << endl;
    cout << "  " << programName << " --build-engine -o yolo11n.onnx -e yolo11n.engine" << endl;
    cout << "\n  # Run inference on video:" << endl;
    cout << "  " << programName << " --run -v video.mp4 -e yolo11n.engine" << endl;
    cout << "==================================================" << endl;
}

// Function to build TensorRT engine from ONNX
bool buildEngine(const string& onnxPath, const string& enginePath) {
    cout << "\n==================================================" << endl;
    cout << "Building TensorRT Engine" << endl;
    cout << "==================================================" << endl;
    cout << "ONNX file: " << onnxPath << endl;
    cout << "Output engine: " << enginePath << endl;
    cout << "==================================================" << endl;
    
    // Construct trtexec command
    string cmd = "/usr/src/tensorrt/bin/trtexec "
                 "--onnx=" + onnxPath + " "
                 "--saveEngine=" + enginePath + " "
                 "--fp16 "
                 "--useCudaGraph "
                 "--useSpinWait "
                 "--avgRuns=100 "
                 "--verbose";
    
    cout << "\nExecuting: " << cmd << endl;
    cout << "\nBuilding engine (this may take a few minutes)...\n" << endl;
    
    int result = system(cmd.c_str());
    
    if (result == 0) {
        cout << "\n==================================================" << endl;
        cout << "✅ Engine built successfully!" << endl;
        cout << "Engine saved to: " << enginePath << endl;
        cout << "==================================================" << endl;
        return true;
    } else {
        cerr << "\n==================================================" << endl;
        cerr << "❌ Failed to build engine!" << endl;
        cerr << "==================================================" << endl;
        return false;
    }
}

int main(int argc, char** argv) {
    string mode;
    string videoPath;
    string onnxPath;
    string enginePath;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--build-engine") {
            mode = "build";
        }
        else if (arg == "--run") {
            mode = "run";
        }
        else if (arg == "-v" || arg == "--video") {
            if (i + 1 < argc) {
                videoPath = argv[++i];
            } else {
                cerr << "Error: --video requires a path" << endl;
                return -1;
            }
        }
        else if (arg == "-o" || arg == "--onnx") {
            if (i + 1 < argc) {
                onnxPath = argv[++i];
            } else {
                cerr << "Error: --onnx requires a path" << endl;
                return -1;
            }
        }
        else if (arg == "-e" || arg == "--engine") {
            if (i + 1 < argc) {
                enginePath = argv[++i];
            } else {
                cerr << "Error: --engine requires a path" << endl;
                return -1;
            }
        }
        else {
            cerr << "Error: Unknown argument: " << arg << endl;
            printUsage(argv[0]);
            return -1;
        }
    }
    
    // Validate mode
    if (mode.empty()) {
        cerr << "Error: Must specify either --build-engine or --run" << endl;
        printUsage(argv[0]);
        return -1;
    }
    
    // BUILD ENGINE MODE
    if (mode == "build") {
        // Validate required arguments
        if (onnxPath.empty()) {
            cerr << "Error: --onnx is required for --build-engine" << endl;
            printUsage(argv[0]);
            return -1;
        }
        if (enginePath.empty()) {
            cerr << "Error: --engine is required for --build-engine" << endl;
            printUsage(argv[0]);
            return -1;
        }
        
        // Build the engine
        bool success = buildEngine(onnxPath, enginePath);
        return success ? 0 : -1;
    }
    
    // RUN INFERENCE MODE
    if (mode == "run") {
        // Validate required arguments
        if (videoPath.empty()) {
            cerr << "Error: --video is required for --run" << endl;
            printUsage(argv[0]);
            return -1;
        }
        if (enginePath.empty()) {
            cerr << "Error: --engine is required for --run" << endl;
            printUsage(argv[0]);
            return -1;
        }
        
        // Print configuration
        cout << "\n==================================================" << endl;
        cout << "Running Inference" << endl;
        cout << "==================================================" << endl;
        cout << "Video: " << videoPath << endl;
        cout << "Engine: " << enginePath << endl;
        cout << "==================================================" << endl;
        
        // Create YAML config programmatically
        YAML::Node config;

        // Model parameters
        config["BATCH_SIZE"] = 1;
        config["INPUT_CHANNEL"] = 3;
        config["IMAGE_WIDTH"] = 640;
        config["IMAGE_HEIGHT"] = 640;
        config["INPUT_WIDTH"] = 640;
        config["INPUT_HEIGHT"] = 640;

        // Detection parameters
        config["obj_threshold"] = 0.5;
        config["nms_threshold"] = 0.4;
        config["agnostic"] = false;
        config["CATEGORY_NUM"] = 1;  // Change to 1 for pothole detection

        // File paths
        config["onnx_file"] = "";
        config["engine_file"] = enginePath;
        config["labels_file"] = "";

        // Strides for YOLOv11
        config["strides"] = std::vector<int>{8, 16, 32};

        // Anchors (empty for YOLOv11 - anchor-free)
        std::vector<std::vector<int>> empty_anchors = {{}, {}, {}};
        config["anchors"] = empty_anchors;

        // Num anchors (1 for each stride in YOLOv11)
        config["num_anchors"] = std::vector<int>{1, 1, 1};

        cout << "\nInitializing YOLO model..." << endl;
        YOLO detector(config);
        cout << "Model loaded successfully!" << endl;
        
        // Initialize video capture
        cout << "Opening video file..." << endl;
        VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            cerr << "Error: Cannot open video file: " << videoPath << endl;
            return -1;
        }
        
        // Get video properties
        int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(CAP_PROP_FPS);
        int totalFrames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
        
        cout << "\nVideo Properties:" << endl;
        cout << "  Resolution: " << frameWidth << "x" << frameHeight << endl;
        cout << "  FPS: " << fps << endl;
        cout << "  Total Frames: " << totalFrames << endl;
        
        // Initialize SORT tracker
        Sort::Ptr tracker = make_shared<Sort>(30, 3, 0.3f);
        cout << "\nSORT tracker initialized" << endl;
        
        // Generate colors
        vector<Scalar> colors = generateColors(100);
        
        // Create window
        namedWindow("YOLO + SORT Tracking", WINDOW_NORMAL);
        resizeWindow("YOLO + SORT Tracking", 1280, 720);
        
        Mat frame;
        int frameCount = 0;
        
        cout << "\n==================================================" << endl;
        cout << "Starting tracking... (Press ESC to exit)" << endl;
        cout << "==================================================" << endl;
        
        auto startTime = chrono::high_resolution_clock::now();
        
        while (true) {
            if (!cap.read(frame)) {
                cout << "\nEnd of video reached." << endl;
                break;
            }
            
            frameCount++;
            
            // Run YOLO detection
            vector<Mat> frames = {frame};
            vector<vector<DetectRes>> batch_res = detector.InferenceImages(frames);
            vector<DetectRes> detections = batch_res[0];
            
            // Convert detections to SORT format
            Mat sortDetections = convertDetectionsToSort(detections);
            
            // Update tracker
            Mat trackedBboxes = tracker->update(sortDetections);
            
            // Draw results
            drawTrackedDetections(frame, trackedBboxes, colors);
            
            // Display frame info
            string frameInfo = "Frame: " + to_string(frameCount) + "/" + to_string(totalFrames) +
                              " | Detections: " + to_string(detections.size()) +
                              " | Tracked: " + to_string(trackedBboxes.rows);
            putText(frame, frameInfo, Point(10, 30), 
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            
            // Show frame
            imshow("YOLO + SORT Tracking", frame);
            
            // Press ESC to exit
            int key = waitKey(1);
            if (key == 27) {
                cout << "\nESC pressed. Exiting..." << endl;
                break;
            }
            
            // Print progress
            if (frameCount % 30 == 0) {
                auto currentTime = chrono::high_resolution_clock::now();
                auto duration = chrono::duration_cast<chrono::seconds>(currentTime - startTime);
                double processingFps = (duration.count() > 0) ? frameCount / (double)duration.count() : 0;
                cout << "Progress: " << frameCount << "/" << totalFrames 
                     << " (" << (frameCount * 100 / totalFrames) << "%) "
                     << "| FPS: " << fixed << setprecision(2) << processingFps << endl;
            }
        }
        
        auto endTime = chrono::high_resolution_clock::now();
        auto totalDuration = chrono::duration_cast<chrono::seconds>(endTime - startTime);
        double avgFps = (totalDuration.count() > 0) ? frameCount / (double)totalDuration.count() : 0;
        
        cout << "\n==================================================" << endl;
        cout << "Tracking Complete!" << endl;
        cout << "  Frames Processed: " << frameCount << endl;
        cout << "  Total Time: " << totalDuration.count() << " seconds" << endl;
        cout << "  Average FPS: " << fixed << setprecision(2) << avgFps << endl;
        cout << "==================================================" << endl;
        
        cap.release();
        destroyAllWindows();
    }
    
    return 0;
}
