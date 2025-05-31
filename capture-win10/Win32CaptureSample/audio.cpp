#include "audio.h"
#include <lame.h>

static std::ofstream LOGA("audio.log", std::ios_base::out | std::ios_base::trunc);

void WriteWAVHeader(std::ofstream& file, const WAVEFORMATEX* format, DWORD dataSize)
{
    file.write("RIFF", 4);
    DWORD chunkSize = 36 + dataSize;
    file.write(reinterpret_cast<const char*>(&chunkSize), 4);
    file.write("WAVE", 4);

    file.write("fmt ", 4);
    DWORD subchunk1Size = 16;
    file.write(reinterpret_cast<const char*>(&subchunk1Size), 4);
    file.write(reinterpret_cast<const char*>(&format->wFormatTag), 2);
    file.write(reinterpret_cast<const char*>(&format->nChannels), 2);
    file.write(reinterpret_cast<const char*>(&format->nSamplesPerSec), 4);
    file.write(reinterpret_cast<const char*>(&format->nAvgBytesPerSec), 4);
    file.write(reinterpret_cast<const char*>(&format->nBlockAlign), 2);
    file.write(reinterpret_cast<const char*>(&format->wBitsPerSample), 2);

    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&dataSize), 4);
}

void SaveWav(BYTE* pData, DWORD dataSize, WAVEFORMATEX &srcFormat)
{
    LOGA << "format   " << srcFormat.wFormatTag << std::endl;
    LOGA << "channel  " << srcFormat.nChannels << std::endl;
    LOGA << "sample/s " << srcFormat.nSamplesPerSec << std::endl;
    LOGA << "b/sample " << srcFormat.wBitsPerSample << std::endl;
    LOGA << "balign   " << srcFormat.nBlockAlign << std::endl;
    LOGA << "bps      " << srcFormat.nAvgBytesPerSec << std::endl;
    LOGA << "bsize    " << srcFormat.cbSize << std::endl;

    if (false) {
        WAVEFORMATEX dstFormat;
        dstFormat.wFormatTag = WAVE_FORMAT_PCM;
        dstFormat.nChannels = srcFormat.nChannels;
        dstFormat.nSamplesPerSec = srcFormat.nSamplesPerSec;
        dstFormat.wBitsPerSample = 16;
        dstFormat.nBlockAlign = dstFormat.nChannels * (dstFormat.wBitsPerSample / 8);
        dstFormat.nAvgBytesPerSec = dstFormat.nSamplesPerSec * dstFormat.nBlockAlign;
        dstFormat.cbSize = 0;
        int frames = dataSize / (dstFormat.nChannels * dstFormat.wBitsPerSample / 8);

        std::vector<int16_t> output(frames);
        for (UINT32 i = 0; i < frames; ++i) {
            float sample = reinterpret_cast<float*>(pData)[i];
            output[i] = static_cast<int16_t>(max(-1.0f, min(1.0f, sample)) * 32767);
        }

        std::ofstream file("output.wav", std::ios::binary | std::ios_base::out | std::ios_base::trunc);
        if (!file.is_open()) throw std::runtime_error("Could not open wav output file");

        WriteWAVHeader(file, &dstFormat, dataSize);
        file.write(reinterpret_cast<const char*>(output.data()), output.size() * 2);
        file.close();
    }

    std::ofstream mp3("output.mp3", std::ios::binary | std::ios_base::out | std::ios_base::trunc);
    if (!mp3.is_open()) throw std::runtime_error("Could not open mp3 output file");
    size_t readv, writev;
    const int PCM_SIZE = 8192 * 4;
    const int MP3_SIZE = 8192;
    uint8_t mp3_buffer[MP3_SIZE];

    lame_t lame = lame_init();
    lame_set_in_samplerate(lame, srcFormat.nSamplesPerSec);
    lame_set_VBR(lame, vbr_default);
    lame_init_params(lame);
    LOGA << "write mp3 " << dataSize << " bytes\n";

    for (size_t pos = 0; pos < dataSize; pos += readv) {
        readv = min(PCM_SIZE, dataSize - pos);
        if (readv > 0) {
            writev = lame_encode_buffer_interleaved_ieee_float(lame,
                reinterpret_cast<float*>(pData + pos), readv >> 3, mp3_buffer, MP3_SIZE);
        } else {
            writev = lame_encode_flush(lame, mp3_buffer, MP3_SIZE);
        }
        mp3.write(reinterpret_cast<const char*>(mp3_buffer), writev);
    }

    lame_close(lame);
    mp3.close();
}

bool AudioCapture::Initialize()
{
    HRESULT hr;

    // Initialize COM
    hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) return false;

    // Get default audio endpoint
    IMMDeviceEnumerator* pEnumerator = nullptr;
    hr = CoCreateInstance(
        __uuidof(MMDeviceEnumerator),
        NULL,
        CLSCTX_ALL,
        __uuidof(IMMDeviceEnumerator),
        (void**)&pEnumerator);
    if (FAILED(hr)) return false;

    hr = pEnumerator->GetDefaultAudioEndpoint(
        eRender, eConsole, &m_pDevice);
    pEnumerator->Release();
    if (FAILED(hr)) return false;

    // Activate audio client
    hr = m_pDevice->Activate(
        __uuidof(IAudioClient),
        CLSCTX_ALL,
        NULL,
        (void**)&m_pAudioClient);
    if (FAILED(hr)) return false;

    // Get mix format and convert to PCM
    WAVEFORMATEX* pFormat = nullptr;
    hr = m_pAudioClient->GetMixFormat(&pFormat);
    if (FAILED(hr)) return false;

    // Initialize audio client
    hr = m_pAudioClient->Initialize(
        AUDCLNT_SHAREMODE_SHARED,
        AUDCLNT_STREAMFLAGS_LOOPBACK,
        REFTIMES_PER_SEC,
        0,
        pFormat,
        NULL);
    CoTaskMemFree(pFormat);
    if (FAILED(hr)) return false;

    // Get capture client
    hr = m_pAudioClient->GetService(
        __uuidof(IAudioCaptureClient),
        (void**)&m_pCaptureClient);
    if (FAILED(hr)) return false;

    return true;
}

void AudioCapture::Start()
{
    m_terminate = false;
    m_captureThread = std::thread(&AudioCapture::CaptureThread, this);
}

void AudioCapture::Stop()
{
    LOGA.close();
    m_terminate = true;
    if (m_captureThread.joinable())
        m_captureThread.join();

    if (m_pCaptureClient) m_pCaptureClient->Release();
    if (m_pAudioClient) m_pAudioClient->Release();
    if (m_pDevice) m_pDevice->Release();
    CoUninitialize();
}

void AudioCapture::CaptureThread()
{
	constexpr int recordInterval = 60; // seconds
    HRESULT hr;
    WAVEFORMATEX* pFormat = nullptr;
    m_pAudioClient->GetMixFormat(&pFormat);
    m_pAudioClient->Start();

    auto startTime = std::chrono::steady_clock::now();
    std::vector<BYTE> accumulatedBuffer;

    while (!m_terminate)
    {
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();

        if (elapsed >= recordInterval && !accumulatedBuffer.empty())
        {
            SaveWav(accumulatedBuffer.data(), accumulatedBuffer.size(), *pFormat);
            accumulatedBuffer.clear();
            startTime = currentTime;
        }

        // Get captured data
        UINT32 packetLength = 0;
        hr = m_pCaptureClient->GetNextPacketSize(&packetLength);

        if (packetLength == 0)
        {
            Sleep(1);
			continue;
        }

        BYTE* pData;
        UINT32 framesAvailable;
        DWORD flags;
        UINT64 devicePosition;
        UINT64 qpcPosition;

        hr = m_pCaptureClient->GetBuffer(
            &pData,
            &framesAvailable,
            &flags,
            &devicePosition,
            &qpcPosition);

        if (framesAvailable > 0)
        {
            accumulatedBuffer.insert(
                accumulatedBuffer.end(),
                pData,
                pData + framesAvailable * pFormat->nBlockAlign);
        }
        m_pCaptureClient->ReleaseBuffer(framesAvailable);
    }

    m_pAudioClient->Stop();
    CoTaskMemFree(pFormat);
}
