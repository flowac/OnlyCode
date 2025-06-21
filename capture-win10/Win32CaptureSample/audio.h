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
    int commsScore();
    void CaptureThread();
    winrt::Windows::Foundation::IAsyncOperation<bool> SaveWav(uint8_t* pData, size_t dataSize, WAVEFORMATEX& srcFormat);

    IMMDevice* m_pDevice{ nullptr };
    IAudioClient* m_pAudioClient{ nullptr };
    IAudioCaptureClient* m_pCaptureClient{ nullptr };
    std::thread m_captureThread;
    bool m_terminate;
    HWND m_voice{ nullptr };

	int m_totalDP{ 0 };
	// 0: N/A, 1: Low, 2: Medium, 3: High
    std::array<int, 4> m_density{};
    std::array<int, 4> m_impact{};
    const int tierThreshold{ 20 }; // Minimum percentage to be in tier
};
