import SwiftUI
import ScreenCaptureKit
import AVFoundation
import Foundation
import ScreenCaptureKit
import CoreImage
import CoreGraphics
import ImageIO

extension MessageManager {
    static let shared = MessageManager()
}

var voiceURL = "http://104.52.63.66:7001"
var poseURL  = "http://104.52.63.66:7002"
var codeURL  = "http://104.52.63.66:7003"

func initURL() -> String {
    let filename = "api.conf"
    let desktop = FileManager.default.temporaryDirectory
    let fileURL = desktop.appendingPathComponent(filename)
    
    if !FileManager.default.fileExists(atPath: fileURL.path) {
        print("Using default server")
        return "Ready"
    }
    
    do {
        let content = try String(contentsOf: fileURL)
        let lines = content.components(separatedBy: .newlines)
        voiceURL = lines[0]
        poseURL  = lines[1]
        codeURL  = lines[2]
        httpGet(url: voiceURL, msg: "Voice")
        httpGet(url: poseURL,  msg: "Pose")
        httpGet(url: codeURL,  msg: "Code")
    } catch {
        print("Custom server: Error reading config \(fileURL.path): \(error)")
        return "ERROR: Failed custom server init"
    }
    
    print("Custom server URL from \(fileURL.path) applied")
    return "Custom Server Ready"
}

class MessageManager: ObservableObject {
    @Published var audioStatus: String = "Idle"
    @Published var codeStatus: String = "Idle"
    @Published var mainStatus: String = "Ready"
    @Published var poseStatus: String = "Idle"
    @Published var buttonStatus: String = "Start Capture"
    
    func audio(_ status: String) {
        DispatchQueue.main.async {
            self.audioStatus = status
        }
    }
    
    func code(_ status: String) {
        DispatchQueue.main.async {
            self.codeStatus = status
        }
    }
    
    func main(_ status: String) {
        DispatchQueue.main.async {
            self.mainStatus = status
        }
    }
    
    func pose(_ status: String) {
        DispatchQueue.main.async {
            self.poseStatus = status
        }
    }
    
    func button(_ status: String) {
        DispatchQueue.main.async {
            self.buttonStatus = status
        }
    }
    
    func start() {
        self.audio("Starting...")
        self.code("Starting...")
        self.pose("Starting...")
        self.main("Recording Active")
        self.button("Stop Capture")
    }
    
    func stop() {
        self.main("Stopping...")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.audio("Idle")
            self.code("Idle")
            self.pose("Idle")
            self.main("Ready")
            self.button("Start Capture")
        }
    }
}

struct ContentView: View {
    @ObservedObject private var mm = MessageManager.shared
    @State var audio = AudioRecorder()
    @State private var floatingManager = FloatingWindowManager()
    @State private var isCap: Bool = false
    
    init() {
        mm.main(initURL())
    }
    
    var body: some View {
        VStack(spacing: 20) {
            // Main status display
            Text(mm.mainStatus)
                .font(.headline)
                .foregroundColor(.primary)
            
            // Status text boxes
            VStack(spacing: 12) {
                StatusBox(title: "Verbal Communication Analysis", status: mm.audioStatus, color: .blue)
                StatusBox(title: "Typing Pattern Analysis", status: mm.codeStatus, color: .orange)
                StatusBox(title: "Head Pose Analysis", status: mm.poseStatus, color: .green)
            }
            
            // Control buttons
            HStack(spacing: 15) {
                Button("Window Picker") {
                    floatingManager.showFloatingPicker()
                }
                
                Button(mm.buttonStatus) {
                    if isCap {
                        stopAud()
                        stopCap()
                        mm.stop()
                        isCap = false
                        print("Stopping capture")
                    } else {
                        print("Starting capture")
                        mm.start()
                        
                        startAud() { ret in
                            mm.audio(ret)
                        }
                        
                        let codeWindow = floatingManager.getCodeWindow()
                        let poseWindow = floatingManager.getPoseWindow()
                        mm.code(codeWindow != nil ? "Active" : "Idle")
                        mm.pose(poseWindow != nil ? "Active" : "Idle")
                        isCap = true
                        startCap()
                    }
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .frame(width: 300, height: 240)
        .padding()
        .onAppear {
            // Make window always on top
            if let window = NSApplication.shared.windows.first {
                window.level = .floating
                window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            }
        }
        .onReceive(mm.$mainStatus) { _ in }
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
