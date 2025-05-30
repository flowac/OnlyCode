//
//  AudioRecorder.swift
//  capture-mac
//

import AVFoundation

//let process = Process()
//process.executableURL = URL(fileURLWithPath: "/usr/local/bin/ffmpeg")
//process.arguments = ["-i", "input.wav", "output.mp3"]
//try? process.run()

var stopTimer: Timer?
var startTimer: Timer?
var timeLimit = 20.0
var audioRuntime: TimeInterval = timeLimit
var audioRestart: TimeInterval = timeLimit + 1.0

class AudioRecorder: NSObject {
    private let session = AVCaptureSession()
    private var audioFileOutput: AVAudioFile?
    private var engine = AVAudioEngine()
    let outputURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("output.wav")

    func startRecording() {
        let devices = AVCaptureDevice.devices(for: .audio)
        for device in devices {
            print("Device: \(device.localizedName)")
        }
        
        guard let inputDevice = AVCaptureDevice.devices(for: .audio).first(where: {
            $0.localizedName.contains("BlackHole")
        }) else {
            print("❌ BlackHole input device not found.")
            return
        }

        do {
            let audioInput = try AVCaptureDeviceInput(device: inputDevice)
            session.beginConfiguration()
            if session.canAddInput(audioInput) {
                session.addInput(audioInput)
            } else {
                print("❌ Can't add audio input")
                return
            }

            session.commitConfiguration()
        } catch {
            print("❌ Failed to create input: \(error)")
            return
        }

        let inputNode = engine.inputNode
        let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
        print(inputNode.inputFormat(forBus: 0))
        print(format)

        do {
            audioFileOutput = try AVAudioFile(forWriting: outputURL, settings: format.settings)
        } catch {
            print("❌ Could not create output file: \(error)")
            return
        }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            do {
                try self.audioFileOutput?.write(from: buffer)
            } catch {
                print("❌ Error writing buffer: \(error)")
            }
        }

        do {
            try engine.start()
            session.startRunning()
            print("🎙️ Started recording...")
            stopTimer = Timer.scheduledTimer(withTimeInterval: audioRuntime, repeats: false) { _ in
                self.stopRecording()
            }
        } catch {
            print("❌ Error starting engine: \(error)")
        }
    }

    func stopRecording() {
        session.stopRunning()
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        audioFileOutput = nil
        print("Recording stopped. Saved to: \(outputURL.path)")
    }
}

func startAud() {
    startTimer = Timer.scheduledTimer(withTimeInterval: audioRestart, repeats: true) { _ in
        let rec = AudioRecorder()
        rec.startRecording()
    }
    startTimer?.fire()
}

func stopAud() {
    startTimer?.invalidate()
    stopTimer?.fire()
    usleep(200000) // sleep 200ms
    stopTimer?.invalidate()
}
