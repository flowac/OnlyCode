import SwiftUI
import ScreenCaptureKit
import AVFoundation
import Foundation
import ScreenCaptureKit
import CoreImage
import CoreGraphics
import ImageIO

struct CaptureApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .defaultPosition(.topTrailing)
    }
}

struct ContentView: View {
    @State private var status: String = "Ready"
    @State private var audioPrefix: String = "Verbal Communication Analysis: "
    @State private var codePrefix: String = "Typing Pattern Analysis: "
    @State private var posePrefix: String = "Head Pose Analysis: "
    @State private var audioStatus: String = "Idle"
    @State private var codeStatus: String = "Idle"
    @State private var poseStatus: String = "Idle"
    @State var audio = AudioRecorder()
    @State private var floatingManager = FloatingWindowManager()
    
    var body: some View {
        VStack(spacing: 20) {
            // Main status display
            Text(status)
                .font(.headline)
                .foregroundColor(.primary)
            
            // Status text boxes
            VStack(spacing: 12) {
                StatusBox(title: "Verbal Communication Analysis", status: audioStatus, color: .blue)
                StatusBox(title: "Typing Pattern Analysis", status: codeStatus, color: .orange)
                StatusBox(title: "Head Pose Analysis", status: poseStatus, color: .green)
            }
            
            // Control buttons
            HStack(spacing: 15) {
                Button("Window Picker") {
                    floatingManager.showFloatingPicker()
                }
                
                Button("Start Capture") {
                    print("Starting capture")
                    status = "Starting..."
                    audioStatus = "Starting..."
                    codeStatus = "Starting..."
                    poseStatus = "Starting..."
                    
                    startAud() { ret in
                        audioStatus = ret
                    }
                    
                    let codeWindow = floatingManager.getCodeWindow()
                    let poseWindow = floatingManager.getPoseWindow()
                    codeStatus = codeWindow != nil ? "Active" : "Idle"
                    poseStatus = poseWindow != nil ? "Active" : "Idle"
                    startCap()
                    
                    // Update status after operations
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        status = "Recording Active"
                    }
                }
                .buttonStyle(.borderedProminent)
                
                Button("Stop Capture") {
                    status = "Stopping..."
                    stopAud()
                    stopCap()
                    print("Stopping capture")
                    
                    // Update status after operations
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        status = "Ready"
                        audioStatus = "Idle"
                        codeStatus = "Idle"
                        poseStatus = "Idle"
                    }
                }
                .buttonStyle(.bordered)
            }
        }
        .frame(width: 400, height: 300)
        .padding()
        .onAppear {
            // Make window always on top
            if let window = NSApplication.shared.windows.first {
                window.level = .floating
                window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            }
        }
    }
}

struct StatusBox: View {
    let title: String
    let status: String
    let color: Color
    
    var body: some View {
        HStack {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(status)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.primary)
            }
            
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(.controlBackgroundColor))
        .cornerRadius(6)
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color(.separatorColor), lineWidth: 0.5)
        )
    }
}
