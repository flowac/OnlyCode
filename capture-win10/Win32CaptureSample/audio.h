#pragma once

#include "pch.h"
#include <audioclient.h>
#include <mmdeviceapi.h>

#define REFTIMES_PER_SEC   10000000
#define REFTIMES_PER_MILLISEC 10000

class AudioCapture
{
public:
    AudioCapture() : m_terminate(false) {}
    ~AudioCapture() { Stop(); }

    bool Initialize(HWND voice);
    void Start();
    void Stop();

private:
    void CaptureThread();
    winrt::Windows::Foundation::IAsyncOperation<bool> SaveWav(uint8_t* pData, size_t dataSize, WAVEFORMATEX& srcFormat);

    IMMDevice* m_pDevice{ nullptr };
    IAudioClient* m_pAudioClient{ nullptr };
    IAudioCaptureClient* m_pCaptureClient{ nullptr };
    std::thread m_captureThread;
    bool m_terminate;
    HWND m_voice{ nullptr };
};
