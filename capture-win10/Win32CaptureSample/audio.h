#pragma once

#include <windows.h>
#include <mmdeviceapi.h>
#include <mmeapi.h>
#include <audioclient.h>
#include <functiondiscoverykeys.h>
#include <comdef.h>
#include <fstream>
#include <thread>
#include <chrono>
#include <mmreg.h> // Include this header to define WAVEFORMATEX

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "winmm.lib")

#define REFTIMES_PER_SEC   10000000
#define REFTIMES_PER_MILLISEC 10000

class AudioCapture
{
public:
    AudioCapture() : m_terminate(false) {}
    ~AudioCapture() { Stop(); }

    bool Initialize();
    void Start();
    void Stop();

private:
    void CaptureThread();

    IMMDevice* m_pDevice = nullptr;
    IAudioClient* m_pAudioClient = nullptr;
    IAudioCaptureClient* m_pCaptureClient = nullptr;
    std::thread m_captureThread;
    bool m_terminate;
};
