#pragma once

#include <audioclient.h>
#include <mmdeviceapi.h>

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
