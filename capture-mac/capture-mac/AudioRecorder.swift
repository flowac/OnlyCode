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
var timeLimit = 60.0
var audioRuntime: TimeInterval = timeLimit
var audioRestart: TimeInterval = timeLimit + 1.0

var totalDP = 0
let tierCount = 4
var density = [Int](repeating: 0, count: 4)
var impact  = [Int](repeating: 0, count: 4)
let tierThreshold = 10

func commsScore() -> Int {
    if totalDP == 0 { return 0 }
    
    var maxDensity = 0
    var maxImpact = 0
    for i in 1..<tierCount {
        if 100 * density[i] / totalDP > tierThreshold {
            maxDensity = i
        }
        if 100 * impact[i] / totalDP > tierThreshold {
            maxImpact = i
        }
    }
    return max(0, maxDensity + maxImpact - 1)
}

class AudioRecorder: NSObject {
    private let session = AVCaptureSession()
    private var audioFileOutput: AVAudioFile?
    private var engine = AVAudioEngine()
    let wavoutURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("output.wav")
    let mp3outURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("output.mp3")
    
    func startRecording() -> Bool {
        let devices = AVCaptureDevice.devices(for: .audio)
        for device in devices {
            print("Device: \(device.localizedName)")
        }
        
        guard let inputDevice = AVCaptureDevice.devices(for: .audio).first(where: {
            $0.localizedName.contains("BlackHole")
        }) else {
            print("❌ BlackHole input device not found.")
            return false
        }

        do {
            let audioInput = try AVCaptureDeviceInput(device: inputDevice)
            session.beginConfiguration()
            if session.canAddInput(audioInput) {
                session.addInput(audioInput)
            } else {
                print("❌ Can't add audio input")
                return false
            }

            session.commitConfiguration()
        } catch {
            print("❌ Failed to create input: \(error)")
            return false
        }

        let inputNode = engine.inputNode
        let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
        print(inputNode.inputFormat(forBus: 0))
        print(format)

        do {
            audioFileOutput = try AVAudioFile(forWriting: wavoutURL, settings: format.settings)
        } catch {
            print("❌ Could not create output file: \(error)")
            return false
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
            return false
        }
        
        return true
    }

    func stopRecording() {
        session.stopRunning()
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        audioFileOutput = nil
        print("Recording stopped. Saved to: \(wavoutURL.path)")
        
        do {
            try convertWavToMp3(inputWavURL: wavoutURL, outputMp3URL: mp3outURL)
            print("Converted to \(mp3outURL.path)")
        } catch {
            print("❌ Error: \(error)")
        }
        
        httpPost(url: voiceURL, fileName: mp3outURL.path) { result in
            if result != nil {
                var jp = JsonParser()
                let rmap = jp.getDoubles(from: result!)
                let cd = Int(rmap["density"] ?? 0)
                let ci = Int(rmap["impact"] ?? 0)
                var score = 0
                
                if cd > 0 && cd < tierCount && ci > 0 && ci < tierCount {
                    totalDP += 1
                    density[cd] += 1
                    impact[ci] += 1
                    score = commsScore()
                }
                print("Score(1-5): \(score), Density: \(cd), Impact: \(ci), Samples: \(totalDP)")
                print("Density \(density[0]), \(density[1]), \(density[2]), \(density[3])")
                print("Impact  \(impact[0]), \(impact[1]), \(impact[2]), \(impact[3])")
                MessageManager.shared.audio("Score: \(score)/5, Density: \(cd), Impact: \(ci)")
            } else {
                MessageManager.shared.audio("Failed to process audio")
            }
        }
    }
}

func startAud(completion: @escaping (String) -> Void) {
    stopAud()
    
    startTimer = Timer.scheduledTimer(withTimeInterval: audioRestart, repeats: true) { _ in
        let rec = AudioRecorder()
        var status = "Failed to record audio"
        if rec.startRecording() {
            status = "Recording"
            completion(status)
        } else {
            completion(status)
        }
    }
    startTimer?.fire()
}

func stopAud() {
    startTimer?.invalidate()
    stopTimer?.fire()
    usleep(200000) // sleep 200ms
    stopTimer?.invalidate()
}

func encodeToMP3(pcmBuffer: UnsafePointer<Int16>, sampleRate: Int, numSamples: Int) -> Data? {
    let lame = lame_init()
    lame_set_in_samplerate(lame, Int32(sampleRate))
    lame_set_num_channels(lame, 1)
    lame_set_VBR(lame, vbr_default)
    lame_init_params(lame)

    let mp3BufferSize = Int(1.25 * Float(numSamples) + 7200)
    let mp3Buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: mp3BufferSize)

    let bytesWritten = lame_encode_buffer(
        lame,
        pcmBuffer,
        nil,
        Int32(numSamples),
        mp3Buffer,
        Int32(mp3BufferSize)
    )

    lame_close(lame)

    if bytesWritten > 0 {
        return Data(bytes: mp3Buffer, count: Int(bytesWritten))
    } else {
        return nil
    }
}

func convertWavToMp3(inputWavURL: URL, outputMp3URL: URL) throws {
    let inputFile = try AVAudioFile(forReading: inputWavURL)
    let inputFormat = inputFile.processingFormat

    // Define 16-bit mono PCM format
    guard let targetFormat = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                           sampleRate: inputFormat.sampleRate,
                                           channels: 1,
                                           interleaved: true) else {
        throw NSError(domain: "WAVError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to create target format"])
    }

    let converter = AVAudioConverter(from: inputFormat, to: targetFormat)!
    let frameCapacity = AVAudioFrameCount(inputFile.length)
    let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: frameCapacity)!
    try inputFile.read(into: inputBuffer)

    let convertedBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: frameCapacity)!

    var error: NSError?
    converter.convert(to: convertedBuffer, error: &error) { inNumPackets, outStatus in
        outStatus.pointee = .haveData
        return inputBuffer
    }

    if let error = error {
        throw error
    }

    guard let channelData = convertedBuffer.int16ChannelData else {
        throw NSError(domain: "WAVError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to access converted PCM data"])
    }

    let numSamples = Int(convertedBuffer.frameLength)
    let sampleRate = Int(targetFormat.sampleRate)
    let pcmPointer = channelData[0]

    guard let mp3Data = encodeToMP3(pcmBuffer: pcmPointer, sampleRate: sampleRate, numSamples: numSamples) else {
        throw NSError(domain: "MP3Error", code: 3, userInfo: [NSLocalizedDescriptionKey: "MP3 encoding failed"])
    }

    try mp3Data.write(to: outputMp3URL)
    print("✅ MP3 saved to: \(outputMp3URL.path)")
}
