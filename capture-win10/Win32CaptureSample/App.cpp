#include "pch.h"
#include <filesystem>
#include <fstream>
#include <iostream>

#include "App.h"
#include "CaptureSnapshot.h"
#include "Image.h"
#include "Web.h"

#pragma comment(lib, "windowscodecs.lib")
std::ofstream LOGA("app.log", std::ios_base::out | std::ios_base::trunc);

namespace winrt
{
    using namespace Windows::Foundation;
    using namespace Windows::Foundation::Metadata;
    using namespace Windows::Graphics::Capture;
    using namespace Windows::Graphics::DirectX;
    using namespace Windows::Graphics::Imaging;
    using namespace Windows::Storage;
    using namespace Windows::Storage::Streams;
    using namespace Windows::System;
    using namespace Windows::UI;
    using namespace Windows::UI::Composition;
    using namespace Windows::UI::Popups;
}

namespace util
{
    using namespace robmikh::common::desktop;
    using namespace robmikh::common::uwp;
}

App::App(winrt::ContainerVisual root)
{
    m_mainThread = winrt::DispatcherQueue::GetForCurrentThread();
    WINRT_VERIFY(m_mainThread != nullptr);

    auto d3dDevice = util::CreateD3D11Device();
    auto dxgiDevice = d3dDevice.as<IDXGIDevice>();
    m_device = CreateDirect3DDevice(dxgiDevice.get());

    m_timer[CaptureIndex::POSE] = winrt::DispatcherQueue::GetForCurrentThread().CreateTimer();
    m_timer[CaptureIndex::POSE].Interval(std::chrono::milliseconds{ 3300 });
    auto token_pose = m_timer[CaptureIndex::POSE].Tick({ this, &App::OnTickPose });
    m_timer[CaptureIndex::CODE] = winrt::DispatcherQueue::GetForCurrentThread().CreateTimer();
    m_timer[CaptureIndex::CODE].Interval(std::chrono::milliseconds{ 10000 });
    auto token_count = m_timer[CaptureIndex::CODE].Tick({this, &App::OnTickCount});

	LOGA << "Voice API: " << winrt::to_string(VOICE_API) << std::endl;
	LOGA << "Pose API:  " << winrt::to_string(POSE_API)  << std::endl;
	LOGA << "Code API:  " << winrt::to_string(CODE_API)  << std::endl;
}

void App::OnTickPose(winrt::IInspectable const& sender, winrt::IInspectable const& event)
{
    TakeSnapshotAsync(CaptureIndex::POSE);
}

void App::OnTickCount(winrt::IInspectable const& sender, winrt::IInspectable const& event)
{
    TakeSnapshotAsync(CaptureIndex::CODE);
}

winrt::IAsyncOperation<winrt::GraphicsCaptureItem> App::StartCaptureWithPickerAsync(CaptureIndex idx)
{
    m_timer[idx].Stop();
	if (!pwd) pwd = co_await winrt::StorageFolder::GetFolderFromPathAsync(std::filesystem::current_path().c_str());

    auto capturePicker = winrt::GraphicsCapturePicker();
    InitializeObjectWithWindowHandle(capturePicker);
    auto item = co_await capturePicker.PickSingleItemAsync();
    if (item)
    {
        // We might resume on a different thread, so let's resume execution on the
        // main thread. This is important because SimpleCapture uses 
        // Direct3D11CaptureFramePool::Create, which requires the existence of
        // a DispatcherQueue. See CaptureSnapshot for an example that uses 
        // Direct3D11CaptureFramePool::CreateFreeThreaded, which doesn't now have this
        // requirement. See the README if you're unsure of which version of 'Create' to use.
        co_await wil::resume_foreground(m_mainThread);
        StartCaptureFromItem(item, idx);

        TakeSnapshotAsync(idx);
        m_timer[idx].Start();
    }

    co_return item;
}

winrt::IAsyncOperation<bool> App::TakeSnapshotAsync(CaptureIndex idx)
{
	static uint16_t frameCount = 0;
    WCHAR fname[12];
	if (true) swprintf_s(fname, L"%s.png", CaptureName[idx]);
    else swprintf_s(fname, L"%s%03d.png", CaptureName[idx], ++frameCount);

    if (m_capture[idx] == nullptr) co_return false;
    auto item = m_capture[idx]->CaptureItem();
    auto texture = co_await CaptureSnapshot::TakeAsync(m_device, item, winrt::DirectXPixelFormat::B8G8R8A8UIntNormalized);
    D3D11_TEXTURE2D_DESC desc = {};
    texture->GetDesc(&desc);
    LOGA << "Input is " << desc.Width << "x" << desc.Height << std::endl;

    switch (idx) {
    case CaptureIndex::CODE:
    {
        // Gray scale for Tesseract OCR
        auto bytes = util::CopyBytesFromTexture(texture);
        for (uint32_t i = 0, j = 0; i < bytes.size(); i += 4, j++) {
            uint8_t b = bytes[i];
            uint8_t g = bytes[i + 1];
            uint8_t r = bytes[i + 2];
            bytes[j] = static_cast<uint8_t>(0.21f * r + 0.72f * g + 0.07f * b);
        }

        HRESULT hr = SaveGray8AsPng(bytes, desc.Width, desc.Height, fname);
        if (FAILED(hr)) {
            LOGA << "Failed to save Gray8\n";
            co_return false;
        }

        LOGA << "Sending words to " << winrt::to_string(CODE_API) << std::endl;
        auto codeStr = co_await SendHttpPost(fname, CODE_API);
        LOGA << "Word count: " << winrt::to_string(codeStr) << std::endl;

        std::unordered_map<winrt::hstring, double> query;
        query[L"value"] = 0;
        ParseJson(codeStr, query);

        std::wstring result = std::format(L"word count: {}", query[L"value"]);
        SetWindowText(m_code, result.c_str());
        break;
    }
    case CaptureIndex::POSE:
    {
        // Scaling down for FaceLandmarker
        constexpr uint32_t maxWidth = 800;
        constexpr uint32_t maxHeight = 600;
        winrt::guid fileFormatGuid = winrt::BitmapEncoder::PngEncoderId();
        winrt::BitmapPixelFormat bitmapPixelFormat = winrt::BitmapPixelFormat::Bgra8;
        auto file = co_await pwd.CreateFileAsync(fname, winrt::CreationCollisionOption::ReplaceExisting);
        if (!file) { LOGA << "Can't open capture output file\n"; co_return false; }
        auto stream = co_await file.OpenAsync(winrt::FileAccessMode::ReadWrite);
        auto encoder = co_await winrt::BitmapEncoder::CreateAsync(fileFormatGuid, stream);

        float scale = 1;
        for (uint32_t width = desc.Width / scale; width > maxWidth; scale += 0.1, width = desc.Width / scale);
        for (uint32_t height = desc.Height / scale; height > maxHeight; scale += 0.1, height = desc.Height / scale);
        int width = desc.Width / scale;
        int height = desc.Height / scale;

        auto bytes = DownscalePixels(util::CopyBytesFromTexture(texture), desc.Width, desc.Height, scale);
        encoder.SetPixelData(bitmapPixelFormat, winrt::BitmapAlphaMode::Ignore, width, height, 1.0, 1.0, bytes);
        co_await encoder.FlushAsync();

        LOGA << "Sending pose to " << winrt::to_string(POSE_API) << std::endl;
        auto poseStr = co_await SendHttpPost(fname, POSE_API);
        LOGA << "Detected pose: " << winrt::to_string(poseStr) << std::endl;

        std::unordered_map<winrt::hstring, double> query;
        query[L"pitch"] = 0;
        query[L"yaw"] = 0;
        ParseJson(poseStr, query);

        std::wstring result = std::format(L"pitch: {}\r\nyaw:   {}", query[L"pitch"], query[L"yaw"]);
        SetWindowText(m_pose, result.c_str());
        break;
    }
    default:
        LOGA << "Unknown capture index: " << idx << std::endl;
		co_return false;
    }

    co_return true;
}

void App::StartCaptureFromItem(winrt::GraphicsCaptureItem item, CaptureIndex idx)
{
    m_capture[idx] = std::make_unique<SimpleCapture>(m_device, item, m_pixelFormat);
    m_capture[idx]->StartCapture();
}

void App::InitializeObjectWithWindowHandle(winrt::Windows::Foundation::IUnknown const& object)
{
    if (m_mainWindow == nullptr)
    {
        throw winrt::hresult_error(E_FAIL, L"App hasn't been properly initialized!");
    }

    // Provide the window handle to the pickers (explicit HWND initialization)
    auto initializer = object.as<util::IInitializeWithWindow>();
    winrt::check_hresult(initializer->Initialize(m_mainWindow));
}

void App::StopCapture(CaptureIndex idx)
{
    m_timer[idx].Stop();
    if (m_capture[idx])
    {
        m_capture[idx]->Close();
        m_capture[idx] = nullptr;
    }
}

void App::InitializeWithWindow(HWND window)
{
    m_mainWindow = window;
}

void App::SetResultPtr(HWND pose, HWND code, HWND voice)
{
    m_pose  = pose;
    m_code  = code;
    m_voice = voice;

    audio_ready = m_record.Initialize(m_voice);
    if (audio_ready) m_record.Start();
    else LOGA << "Audio recording failed to initialize";
}
