//
//  WindowPickerView.swift
//  capture-mac
//

import SwiftUI
import ScreenCaptureKit
import AVFoundation
import CoreImage
import CoreGraphics

private var selectedCode: WindowInfo?
private var selectedPose: WindowInfo?
private var codeTimer: Timer?
private var poseTimer: Timer?

enum CapType {
    case code
    case pose
}

// MARK: - Window Model
struct WindowInfo: Identifiable, Hashable {
    let id: CGWindowID
    let window: SCWindow
    let title: String
    let appName: String
    let bundleIdentifier: String
    var previewImage: NSImage?
    
    init(window: SCWindow) {
        self.id = window.windowID
        self.window = window
        self.title = window.title ?? "Untitled Window"
        self.appName = window.owningApplication?.applicationName ?? "Unknown App"
        self.bundleIdentifier = window.owningApplication?.bundleIdentifier ?? ""
    }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(window.windowID)
    }
    
    static func == (lhs: WindowInfo, rhs: WindowInfo) -> Bool {
        lhs.window.windowID == rhs.window.windowID
    }
}

// MARK: - Window Picker View
struct WindowPickerView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var windows: [WindowInfo] = []
    @State private var isLoading = true
    @State private var errorMessage: String?
    @State private var triggerUpdate = 0
    
    private let minWidth: Int = 200
    private let minHeight: Int = 150
    
    var body: some View {
        VStack(spacing: 20) {
            // Header
            /*
            VStack(spacing: 8) {
                Text("Select a window to capture")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            */
            
            // Content
            if isLoading {
                ProgressView("Loading windows...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                // Window Grid
                ScrollView {
                    LazyVGrid(columns: [
                        GridItem(.adaptive(minimum: 280, maximum: 320), spacing: 16)
                    ], spacing: 16) {
                        ForEach(windows) { windowInfo in
                            WindowCard(
                                windowInfo: windowInfo,
                                isSelectedCode: selectedCode?.id == windowInfo.id,
                                isSelectedPose: selectedPose?.id == windowInfo.id,
                                onSelectCode: {
                                    selectedCode = windowInfo
                                    triggerUpdate += 1
                                },
                                onSelectPose: {
                                    selectedPose = windowInfo
                                    triggerUpdate += 1
                                }
                            )
                        }
                    }
                    .padding()
                }
                .id(triggerUpdate)
            }
            
            // Bottom Controls
            HStack(spacing: 16) {
                Button("Refresh Windows") {
                    Task {
                        await loadWindows()
                    }
                }
                .buttonStyle(.bordered)
                .disabled(isLoading)
                
                Button("Done Picking") {
                    dismiss()
                }
                .buttonStyle(.bordered)
            }
            .padding(.horizontal)
            
            // Status/Error
            if let error = errorMessage {
                Text(error)
                    .foregroundColor(.red)
                    .font(.caption)
                    .padding(.horizontal)
            }
        }
        .padding()
        .frame(minWidth: 800, minHeight: 600)
        .task {
            await loadWindows()
        }
    }
    
    private func loadWindows() async {
        isLoading = true
        errorMessage = nil
        let nameFilter: Set<String> = ["capture-mac", "Desktop", "Dock", "Finder", "Music", "System Information", "Terminal", "Unknown", "Wallpaper"]
        
        do {
            let content = try await SCShareableContent.current
            let validWindows = content.windows.filter { window in
                let appTitle = window.title ?? "Unknown"
                let appName = window.owningApplication?.applicationName ?? "Unknown"
                let skip = nameFilter.contains{ appName.hasPrefix($0) } || nameFilter.contains{ appTitle.hasPrefix($0) }
                
                guard window.owningApplication?.bundleIdentifier != nil,
                      !skip,
                      window.isOnScreen,
                      Int(window.frame.width) >= minWidth,
                      Int(window.frame.height) >= minHeight else {
                    return false
                }
                return true
            }
            
            var windowInfos = validWindows.map { WindowInfo(window: $0) }
            
            for i in 0..<windowInfos.count {
                if let preview = await generatePreview(for: windowInfos[i].window) {
                    windowInfos[i].previewImage = preview
                }
            }
            
            await MainActor.run {
                self.windows = windowInfos
                self.isLoading = false
            }
            
        } catch {
            await MainActor.run {
                self.errorMessage = "Failed to load windows: \(error.localizedDescription)"
                self.isLoading = false
            }
        }
    }
    
    private func generatePreview(for window: SCWindow) async -> NSImage? {
        let filter = SCContentFilter(desktopIndependentWindow: window)
        let config = SCStreamConfiguration()
        
        // Use smaller size for preview
        let previewScale: CGFloat = 0.3
        config.width = Int(window.frame.width * previewScale)
        config.height = Int(window.frame.height * previewScale)
        config.pixelFormat = kCVPixelFormatType_32BGRA
        
        return await withCheckedContinuation { continuation in
            let output = PreviewCaptureOutput { image in
                continuation.resume(returning: image)
            }
            
            Task {
                do {
                    let stream = SCStream(filter: filter, configuration: config, delegate: nil)
                    try stream.addStreamOutput(output, type: .screen, sampleHandlerQueue: .main)
                    try await stream.startCapture()
                    
                    // Stop after capturing one frame
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        stream.stopCapture()
                    }
                } catch {
                    continuation.resume(returning: nil)
                }
            }
        }
    }
}

// MARK: - Window Card Component
struct WindowCard: View {
    let windowInfo: WindowInfo
    let isSelectedCode: Bool
    let isSelectedPose: Bool
    let onSelectCode: () -> Void
    let onSelectPose: () -> Void
    @State private var isHovering = false
    
    var body: some View {
        VStack(spacing: 12) {
            // Preview Image
            Group {
                if let preview = windowInfo.previewImage {
                    Image(nsImage: preview)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } else {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.gray.opacity(0.3))
                        .overlay(
                            VStack {
                                Image(systemName: "macwindow")
                                    .font(.system(size: 24))
                                    .foregroundColor(.secondary)
                                Text("No Preview")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        )
                }
            }
            .frame(height: 120)
            .background(Color.black.opacity(0.05))
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.clear, lineWidth: 2)
                    .stroke((isSelectedCode || isSelectedPose) ? Color.accentColor : Color.clear, lineWidth: 2)
            )
            
            // Window Info
            VStack(alignment: .leading, spacing: 4) {
                Text(windowInfo.title)
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.tail)
                
                Text(windowInfo.appName)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                
                Text("\(Int(windowInfo.window.frame.width))×\(Int(windowInfo.window.frame.height))")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            
            // Action Buttons
            HStack(spacing: 8) {
                Button(isSelectedCode ? "Selected code" : "Select for code") {
                    onSelectCode()
                }
                .buttonStyle(.bordered)
                .disabled(isSelectedCode)
                
                Button(isSelectedPose ? "Selected pose" : "Select for pose") {
                    onSelectPose()
                }
                .buttonStyle(.bordered)
                .disabled(isSelectedPose)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.primary.opacity(isHovering ? 0.05 : 0.02))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.primary.opacity(0.1), lineWidth: 1)
        )
        .scaleEffect(isHovering ? 1.02 : 1.0)
        .animation(.easeInOut(duration: 0.2), value: isHovering)
        .onHover { hovering in
            isHovering = hovering
        }
    }
}

// MARK: - Capture Output Classes
class PreviewCaptureOutput: NSObject, SCStreamOutput {
    let handler: (NSImage?) -> Void
    var hasCaptured = false
    
    init(handler: @escaping (NSImage?) -> Void) {
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
            let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
            handler(nsImage)
        } else {
            handler(nil)
        }
    }
}

class ScreenshotCaptureOutput: NSObject, SCStreamOutput {
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

// MARK: - Utility Functions
func startCap() {
    stopCap()
    
    codeTimer = Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { _ in
        if selectedCode != nil {
            Task {
                await captureWindow(selectedCode!, type: CapType.code)
            }
        }
    }
    
    poseTimer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { _ in
        if selectedPose != nil {
            Task {
                await captureWindow(selectedPose!, type: CapType.pose)
            }
        }
    }
}

func stopCap() {
    codeTimer?.invalidate()
    poseTimer?.invalidate()
    codeTimer = nil
    poseTimer = nil
}

func captureWindow(_ windowInfo: WindowInfo, type: CapType) async {
    do {
        let filter = SCContentFilter(desktopIndependentWindow: windowInfo.window)
        let config = SCStreamConfiguration()
        config.width = Int(windowInfo.window.frame.width)
        config.height = Int(windowInfo.window.frame.height)
        config.pixelFormat = kCVPixelFormatType_32BGRA
        
        let output = ScreenshotCaptureOutput { image in
            saveToDesktop(image: image, type: type)
        }
        
        let stream = SCStream(filter: filter, configuration: config, delegate: nil)
        try stream.addStreamOutput(output, type: .screen, sampleHandlerQueue: .main)
        try await stream.startCapture()
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            stream.stopCapture()
        }
    } catch {
        print("Failed to capture: \(error.localizedDescription)")
    }
}

var prevCodeSus: Bool = false
var prevCodeCount: Int = 0
var totalCode: Int = 1
var susCode: Int = 0
var prevPoseSus: Bool = false
var totalPose: Int = 1
var susPose: Int = 0

func saveToDesktop(image: CGImage, type: CapType) {
    var output = image
    var filename = "cap.png"
    
    switch type {
    case .code:
        filename = "code.png"
        output = grayScale(image: image) ?? image
    case .pose:
        filename = "pose.png"
        output = downSample(image: image) ?? image
    }
    let rep = NSBitmapImageRep(cgImage: output)
    
    if let data = rep.representation(using: .png, properties: [:]) {
        let desktop = FileManager.default.temporaryDirectory
        let fileURL = desktop.appendingPathComponent(filename)
        
        do {
            try data.write(to: fileURL)
            print("Screenshot saved to: \(fileURL.path)")
            
            switch type {
            case .code:
                httpPost(url: codeURL, fileName: fileURL.path) { result in
                    if result == nil {
                        MessageManager.shared.code("Failed to process word per minute")
                        return
                    }
                    
                    var jp = JsonParser()
                    let rmap = jp.getDoubles(from: result!)
                    if rmap.isEmpty {
                        MessageManager.shared.code("Failed to process word per minute")
                        return
                    }
                    
                    let curr = Int(rmap["value"] ?? 0)
                    let diff = curr - prevCodeCount
                    prevCodeCount = curr
                    
                    var wpmtext = "N/A"
                    if diff > 0 && diff < 30 {
                        totalCode += 1
                        if diff >= 7 {
                            if prevCodeSus {
                                susCode += 1
                            }
                            prevCodeSus = true
                            wpmtext = "suspicious"
                        } else {
                            prevCodeSus = false
                            wpmtext = "normal"
                        }
                    }
                    print("sus:\(100 * susCode / totalCode)  wpm:\(diff * 6)  wc:\(curr)")
                    MessageManager.shared.code("\(100 * susCode / totalCode)% suspicious. Currently \(wpmtext).")
                }
            case .pose:
                httpPost(url: poseURL, fileName: fileURL.path) { result in
                    if result == nil {
                        MessageManager.shared.pose("Failed to process pose")
                    }
                    
                    var jp = JsonParser()
                    let rmap = jp.getDoubles(from: result!)
                    if rmap.isEmpty {
                        MessageManager.shared.pose("Failed to process pose")
                        return
                    }
                    
                    let pitch = Int(rmap["head_pose_angles.pitch"] ?? 100)
                    let yaw = Int(rmap["head_pose_angles.yaw"] ?? 100)
                    
                    var posetext = "N/A"
                    if abs(yaw) < 60 && abs(pitch) < 60 {
                        totalPose += 1
                        if abs(yaw) > 11 || pitch < -10 || pitch > 20 {
                            if prevPoseSus {
                                susPose += 1
                            }
                            prevPoseSus = true
                            posetext = "suspicious"
                        } else {
                            prevPoseSus = false
                            posetext = "normal"
                        }
                    }
                    print("sus:\(100 * susPose / totalPose)  pitch:\(pitch)  yaw:\(yaw)")
                    MessageManager.shared.pose("\(100 * susPose / totalPose)% suspicious. Currently \(posetext).")
                }
            }
        } catch {
            print("Error saving screenshot: \(error)")
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

func grayScale(image: CGImage) -> CGImage? {
    let width = image.width
    let height = image.height
    let colorSpace = CGColorSpaceCreateDeviceGray()
    
    guard let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.none.rawValue
    ) else {
        return nil
    }
    
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    print("Gray scaled")
    
    return context.makeImage()
}

// MARK: - Window Picker Manager
class FloatingWindowManager: ObservableObject {
    private var floatingWindow: NSWindow?
    private var codeTarget: SCWindow? = nil
    private var poseTarget: SCWindow? = nil
    
    func getCodeWindow() -> SCWindow? {
        return selectedCode?.window
    }
    
    func getPoseWindow() -> SCWindow? {
        return selectedPose?.window
    }
    
    func showFloatingPicker() {
        let picker = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 700, height: 550),
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        
        picker.isReleasedWhenClosed = false
        picker.title = "Select Windows For Capture"
        picker.contentView = NSHostingView(rootView:
            WindowPickerView()
                .background(Color(NSColor.windowBackgroundColor))
        )
        
        // Make it floating above other windows
        picker.level = .floating
        picker.isMovableByWindowBackground = true
        picker.center()
        picker.makeKeyAndOrderFront(nil)
        floatingWindow = picker
    }
}
