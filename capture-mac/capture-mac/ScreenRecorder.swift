//
//  ScreenRecorder.swift
//  capture-mac
//

import ScreenCaptureKit
import AVFoundation
import Foundation
import ScreenCaptureKit
import CoreImage
import CoreGraphics
import ImageIO

private var status: String = "Ready"
private var outputRef: ScreenshotStreamOutput?
private var window: SCWindow?
private var minWidth: Int = 200
private var minHeight: Int = 150
private var timer: Timer?

class ScreenshotStreamOutput: NSObject, SCStreamOutput {
    let handler: (CGImage) -> Void
    var hasCaptured = false

    init(handler: @escaping (CGImage) -> Void) {
        self.handler = handler
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard !hasCaptured,
              type == .screen,
              let pixelBuffer = sampleBuffer.imageBuffer else { return }

        hasCaptured = true

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            handler(cgImage)
        }
    }
}

func startCap() {
    stopCap()
    timer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { _ in
        Task {
            await findWindow()
        }
    }
}

func stopCap() {
    if timer != nil {
        timer?.invalidate()
        timer = nil
    }
}

func findWindow() async {
    do {
        if window == nil || !window!.isOnScreen {
            let windows = try await SCShareableContent.current.windows
            for candidate in windows {
                if candidate.owningApplication?.bundleIdentifier != nil && candidate.isOnScreen
                    && Int(candidate.frame.width) > minWidth && Int(candidate.frame.height) > minHeight {
                    window = candidate
                    break
                }
            }
        }
        if window == nil {
            print("No valid window found")
        } else {
            await takeScreenshot()
        }
    } catch {
        status = "Error: \(error.localizedDescription)"
    }
}

func takeScreenshot() async {
    do {
        let filter = SCContentFilter(desktopIndependentWindow: window!)

        // Configure the stream
        let config = SCStreamConfiguration()
        config.width = Int(window!.frame.width)
        config.height = Int(window!.frame.height)
        config.pixelFormat = kCVPixelFormatType_32BGRA

        // Prepare output
        let streamOutput = ScreenshotStreamOutput { cgImage in
            saveToDesktop(image: cgImage)
            status = "Screenshot saved!"
        }
        outputRef = streamOutput

        // Create and start stream
        let stream = SCStream(filter: filter, configuration: config, delegate: nil)
        try stream.addStreamOutput(streamOutput, type: .screen, sampleHandlerQueue: .main)
        try await stream.startCapture()

        // Stop after first frame
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            stream.stopCapture()
        }

    } catch {
        status = "Error: \(error.localizedDescription)"
    }
}

func saveToDesktop(image: CGImage) {
    var buffer: CGImage? = downSample(image: image)
    if buffer == nil {
        buffer = image
    }
    
    let rep = NSBitmapImageRep(cgImage: buffer!)
    if let data = rep.representation(using: .png, properties: [:]) {
        let tempURL = FileManager.default.temporaryDirectory
        let fileURL = tempURL.appendingPathComponent("cap.png")
        do {
            try data.write(to: fileURL)
            print("Saved to: \(fileURL.path)")
        } catch {
            print("Error saving image: \(error)")
        }
    }
}

func downSample(image: CGImage, maxSize: CGSize = CGSize(width: 800, height: 600)) -> CGImage? {
    let originalWidth = CGFloat(image.width)
    let originalHeight = CGFloat(image.height)
    
    if originalWidth < maxSize.width && originalHeight < maxSize.height {
        return nil
    }

    let widthRatio = maxSize.width / originalWidth
    let heightRatio = maxSize.height / originalHeight
    let scaleFactor = min(widthRatio, heightRatio, 1.0)

    let newWidth = Int(originalWidth * scaleFactor)
    let newHeight = Int(originalHeight * scaleFactor)

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(
        data: nil,
        width: newWidth,
        height: newHeight,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        return nil
    }

    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: newWidth, height: newHeight))
    print("Down sampled from \(originalWidth)x\(originalHeight) to \(newWidth)x\(newHeight)")

    return context.makeImage()
}
