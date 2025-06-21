#include "pch.h"
#include <windows.h>
#include <comdef.h>
#include <functiondiscoverykeys.h>
#include <mmeapi.h>
#include <mmreg.h> // Include this header to define WAVEFORMATEX
#include <time.h>

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "winmm.lib")

#include <lame.h>
#include "Audio.h"
#include "Web.h"

static std::ofstream LOGA("audio.log", std::ios_base::out | std::ios_base::trunc);

winrt::Windows::Foundation::IAsyncOperation<bool> AudioCapture::SaveWav(uint8_t* pData, size_t dataSize, WAVEFORMATEX& srcFormat)
{
	static bool logOnce{ true };
	if (logOnce) {
		logOnce = false;
		LOGA << "format   " << srcFormat.wFormatTag << std::endl;
		LOGA << "channel  " << srcFormat.nChannels << std::endl;
		LOGA << "sample/s " << srcFormat.nSamplesPerSec << std::endl;
		LOGA << "b/sample " << srcFormat.wBitsPerSample << std::endl;
		LOGA << "balign   " << srcFormat.nBlockAlign << std::endl;
		LOGA << "bps      " << srcFormat.nAvgBytesPerSec << std::endl;
		LOGA << "bsize    " << srcFormat.cbSize << std::endl;
	}
	char tbuf[64]{};
	time_t now = time(nullptr);
	ctime_s(tbuf, 64, &now);
	LOGA << tbuf;

	// Assume the source format is IEEE_FLOAT
	const int sampleSize = srcFormat.nChannels * (srcFormat.wBitsPerSample / 8);
	const int sampleCount = static_cast<int>(dataSize / sampleSize);
	const int runtime = sampleCount / srcFormat.nSamplesPerSec;

	const double sampleDuration = 0.5; // seconds
	const int samplesPerSegment = static_cast<int>(srcFormat.nSamplesPerSec * sampleDuration);
	std::vector<uint16_t> timeslot; // segment average sound level for silence detection
	uint32_t sampleSum = 0;
	uint16_t topSample = 100; // init to reasonable value to filter out complete silence
	uint16_t samplesLoud = 0;

	for (int i = 1, j = 0; i <= sampleCount; ++i, j += sampleSize)
	{
		float sample = *reinterpret_cast<float*>(pData + j);
		sampleSum += static_cast<uint16_t>(std::min(1.0f, abs(sample)) * UINT16_MAX);
		if (i % samplesPerSegment == 0) {
			timeslot.push_back(static_cast<uint16_t>(sampleSum / samplesPerSegment));
			sampleSum = 0;
			topSample = std::max(topSample, timeslot.back());
			//LOGA << "Sample " << timeslot.size() << " average: " << timeslot.back() << std::endl;
		}
	}

	uint16_t silenceThreshold = static_cast<uint16_t>(topSample * 0.2);
	for (auto x : timeslot) samplesLoud += (x > silenceThreshold);
	LOGA << "Recorded " << runtime << " seconds, " << samplesLoud * sampleDuration
		<< " seconds effective, threshold " << silenceThreshold
		<< " max " << topSample << std::endl;

	WCHAR fname[12];
	swprintf_s(fname, L"output.mp3");
	std::ofstream mp3(fname, std::ios::binary | std::ios_base::out | std::ios_base::trunc);
	if (!mp3.is_open()) {
		LOGA << "Could not open mp3 output file, aborting save";
		co_return false;
	}

	int readv, writev;
	const int I3E_SIZE = 8192 * 4;
	const int MP3_SIZE = 8192;
	uint8_t mp3_buffer[MP3_SIZE];

	lame_t lame = lame_init();
	lame_set_in_samplerate(lame, srcFormat.nSamplesPerSec);
	lame_set_VBR(lame, vbr_default);
	lame_init_params(lame);
	LOGA << "Encode " << dataSize << " bytes to MP3\n";

	for (size_t pos = 0; pos < dataSize; pos += readv) {
		readv = std::min(I3E_SIZE, static_cast<int>(dataSize - pos));
		if (readv > 0) {
			writev = lame_encode_buffer_interleaved_ieee_float(lame,
				reinterpret_cast<float*>(pData + pos), readv >> 3, mp3_buffer, MP3_SIZE);
		}
		else {
			writev = lame_encode_flush(lame, mp3_buffer, MP3_SIZE);
		}
		mp3.write(reinterpret_cast<const char*>(mp3_buffer), writev);
	}

	lame_close(lame);
	mp3.close();

	//LOGA << "Sending speech to " << winrt::to_string(VOICE_API) << std::endl;
	auto voiceStr = co_await SendHttpPost(fname, VOICE_API);
	if (voiceStr.empty()) {
		LOGA << "Failed to get response from voice API" << std::endl;
		SetWindowText(m_voice, L"Failed to connect to server");
		co_return false;
	}
	//LOGA << "Speech summary: " << winrt::to_string(voiceStr) << std::endl;

	std::unordered_map<winrt::hstring, double> query;
	ParseJson(voiceStr, query);

	int density = static_cast<int>(query[L"density"]);
	int impact = static_cast<int>(query[L"impact"]);
	int score{ 0 };
	if (density > 0 && density < m_density.size() && impact > 0 && impact < m_impact.size()) {
		++m_totalDP;
		++m_density[density];
		++m_impact[impact];
		score = commsScore();
	}

	std::wstring result = std::format(L"Verbal communication status\r\ndensity: {}\r\nimpact: {}\r\nscore(1-5): {}", density, impact, score);
	LOGA << std::format("density: {}, impact: {}, score: {}", density, impact, score) << std::endl;
	SetWindowText(m_voice, result.c_str());
	co_return true;
}

bool AudioCapture::Initialize(HWND voice)
{
	m_voice = voice;

	HRESULT hr;
	hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
	if (FAILED(hr)) return false;

	IMMDeviceEnumerator* pEnumerator = nullptr;
	hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_ALL,
		__uuidof(IMMDeviceEnumerator), (void**)&pEnumerator);
	if (FAILED(hr)) return false;

	hr = pEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &m_pDevice);
	pEnumerator->Release();
	if (FAILED(hr)) return false;

	hr = m_pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, (void**)&m_pAudioClient);
	if (FAILED(hr)) return false;

	WAVEFORMATEX* pFormat = nullptr;
	hr = m_pAudioClient->GetMixFormat(&pFormat);
	if (FAILED(hr)) return false;

	hr = m_pAudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK,
		REFTIMES_PER_SEC, 0, pFormat, NULL);
	CoTaskMemFree(pFormat);
	if (FAILED(hr)) return false;

	hr = m_pAudioClient->GetService(__uuidof(IAudioCaptureClient), (void**)&m_pCaptureClient);
	if (FAILED(hr)) return false;

	m_terminate = false;
	m_captureThread = std::thread(&AudioCapture::CaptureThread, this);

	return true;
}

AudioCapture::~AudioCapture()
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

void AudioCapture::Start()
{
	m_active = true;
	SetWindowText(m_voice, std::format(L"Verbal communication status\r\nscore(1-5): {}", commsScore()).c_str());
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
			if (m_active)
			{
				SaveWav(accumulatedBuffer.data(), accumulatedBuffer.size(), *pFormat);
			}
			accumulatedBuffer.clear();
			startTime = currentTime;
		}

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

int AudioCapture::commsScore()
{
	if (m_totalDP == 0) return 0;

	int maxDensity{ 0 };
	int maxImpact{ 0 };
	for (int i = m_density.size() - 1; i > 0; --i) {
		if (100 * m_density[i] / m_totalDP > tierThreshold) {
			maxDensity = i;
			break;
		}
	}
	for (int i = m_impact.size() - 1; i > 0; --i) {
		if (100 * m_impact[i] / m_totalDP > tierThreshold) {
			maxImpact = i;
			break;
		}
	}
	return std::max(0, maxDensity + maxImpact - 1);
}
