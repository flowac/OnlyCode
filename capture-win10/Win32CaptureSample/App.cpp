#include "pch.h"
#include <filesystem>
#include <fstream>
#include <iostream>

#include "App.h"
#include "CaptureSnapshot.h"
#include "Image.h"
#include "Web.h"

#pragma comment(lib, "windowscodecs.lib")
static std::ofstream LOGA("app.log", std::ios_base::out | std::ios_base::trunc);

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

    m_compositor = root.Compositor();
    m_root = m_compositor.CreateContainerVisual();
    m_content = m_compositor.CreateSpriteVisual();
    m_brush = m_compositor.CreateSurfaceBrush();

    m_root.RelativeSizeAdjustment({ 1, 1 });
    root.Children().InsertAtTop(m_root);

    m_content.AnchorPoint({ 0.5f, 0.5f });
    m_content.RelativeOffsetAdjustment({ 0.5f, 0.5f, 0 });
    m_content.RelativeSizeAdjustment({ 1, 1 });
    m_content.Size({ -80, -80 });
    m_content.Brush(m_brush);
    m_brush.HorizontalAlignmentRatio(0.5f);
    m_brush.VerticalAlignmentRatio(0.5f);
    m_brush.Stretch(winrt::CompositionStretch::Uniform);
    auto shadow = m_compositor.CreateDropShadow();
    shadow.Mask(m_brush);
    m_content.Shadow(shadow);
    m_root.Children().InsertAtTop(m_content);

    auto d3dDevice = util::CreateD3D11Device();
    auto dxgiDevice = d3dDevice.as<IDXGIDevice>();
    m_device = CreateDirect3DDevice(dxgiDevice.get());

    m_timer = winrt::DispatcherQueue::GetForCurrentThread().CreateTimer();
    m_timer.Interval(std::chrono::milliseconds{ 5000 });
    auto token = m_timer.Tick({ this, &App::OnTick });

    audio_ready = m_record.Initialize();
	assert(audio_ready, L"Audio recording failed to initialize");
    if (audio_ready) m_record.Start();
}

void App::OnTick(winrt::IInspectable const &sender, winrt::IInspectable const &event)
{
    TakeSnapshotAsync();
}

winrt::IAsyncOperation<winrt::GraphicsCaptureItem> App::StartCaptureWithPickerAsync()
{
    m_timer.Stop();
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
        StartCaptureFromItem(item);

        TakeSnapshotAsync();
        m_timer.Start();
    }

    co_return item;
}

winrt::IAsyncOperation<bool> App::TakeSnapshotAsync()
{
	static uint16_t frameCount = 0;
    WCHAR fname[12] = L"cap.png";
    if (false) swprintf_s(fname, L"cap%03d.png", ++frameCount);

    if (m_capture == nullptr) co_return false;
    auto item = m_capture->CaptureItem();
    auto texture = co_await CaptureSnapshot::TakeAsync(m_device, item, winrt::DirectXPixelFormat::B8G8R8A8UIntNormalized);
    D3D11_TEXTURE2D_DESC desc = {};
    texture->GetDesc(&desc);
    LOGA << "Input is " << desc.Width << "x" << desc.Height << std::endl;

    if (false) {
        // Scaling down for FaceLandmarker
		constexpr uint32_t maxWidth = 800;
		constexpr uint32_t maxHeight = 600;
        winrt::guid fileFormatGuid = winrt::BitmapEncoder::PngEncoderId();
        winrt::BitmapPixelFormat bitmapPixelFormat = winrt::BitmapPixelFormat::Bgra8;
        auto file = co_await pwd.CreateFileAsync(fname, winrt::CreationCollisionOption::ReplaceExisting);
        if (!file) {LOGA << "Can't open capture output file\n"; co_return false;}
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
    } else {
        // Gray scale for Tesseract OCR
        auto bytes = util::CopyBytesFromTexture(texture);
        std::vector<uint8_t> grayData(desc.Width * desc.Height);
        for (uint32_t i = 0, j = 0; i < bytes.size(); i += 4, j++) {
            uint8_t b = bytes[i];
            uint8_t g = bytes[i + 1];
            uint8_t r = bytes[i + 2];
            grayData[j] = static_cast<uint8_t>(0.21f * r + 0.72f * g + 0.07f * b);
        }

        HRESULT hr = SaveGray8AsPng(grayData, desc.Width, desc.Height, fname);
        if (FAILED(hr)) {
            LOGA << "Failed to save Gray8\n";
            co_return false;
        }
    }

    winrt::hstring result = co_await SendHttpPost(fname, L"http://18.116.80.5:7003");
    LOGA << winrt::to_string(result).c_str() << std::endl;
    co_return true;
}

void App::StartCaptureFromItem(winrt::GraphicsCaptureItem item)
{
    m_capture = std::make_unique<SimpleCapture>(m_device, item, m_pixelFormat);

    auto surface = m_capture->CreateSurface(m_compositor);
    m_brush.Surface(surface);

    m_capture->StartCapture();
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

void App::StopCapture()
{
	m_timer.Stop();
    if (m_capture)
    {
        m_capture->Close();
        m_capture = nullptr;
        m_brush.Surface(nullptr);
    }
}

void App::InitializeWithWindow(HWND window)
{
    m_mainWindow = window;
}
