#include "pch.h"
#include "App.h"
#include "CaptureSnapshot.h"

namespace winrt
{
    using namespace Windows::Foundation;
    using namespace Windows::Foundation::Metadata;
    using namespace Windows::Graphics::Capture;
    using namespace Windows::Graphics::DirectX;
    using namespace Windows::Graphics::Imaging;
    using namespace Windows::Storage;
    using namespace Windows::Storage::Pickers;
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
}

winrt::GraphicsCaptureItem App::TryStartCaptureFromWindowHandle(HWND hwnd)
{
    winrt::GraphicsCaptureItem item{ nullptr };
    try
    {
        item = util::CreateCaptureItemForWindow(hwnd);
        StartCaptureFromItem(item);
    }
    catch (winrt::hresult_error const& error)
    {
        MessageBoxW(m_mainWindow,
            error.message().c_str(),
            L"Cap",
            MB_OK | MB_ICONERROR);
    }
    return item;
}

winrt::GraphicsCaptureItem App::TryStartCaptureFromMonitorHandle(HMONITOR hmon)
{
    winrt::GraphicsCaptureItem item{ nullptr };
    try
    {
        item = util::CreateCaptureItemForMonitor(hmon);
        StartCaptureFromItem(item);
    }
    catch (winrt::hresult_error const& error)
    {
        MessageBoxW(m_mainWindow,
            error.message().c_str(),
            L"Cap",
            MB_OK | MB_ICONERROR);
    }
    return item;
}

winrt::IAsyncOperation<winrt::GraphicsCaptureItem> App::StartCaptureWithPickerAsync()
{
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
    }

    co_return item;
}

winrt::IAsyncOperation<winrt::StorageFile> App::TakeSnapshotAsync()
{
    // Use what we're currently capturing
    if (m_capture == nullptr)
    {
        co_return nullptr;
    }
    auto item = m_capture->CaptureItem();

    // Ask the user where they want to save the snapshot.
    auto savePicker = winrt::FileSavePicker();
    InitializeObjectWithWindowHandle(savePicker);
    savePicker.SuggestedStartLocation(winrt::PickerLocationId::PicturesLibrary);
    savePicker.SuggestedFileName(L"snapshot");
    savePicker.DefaultFileExtension(L".png");
    savePicker.FileTypeChoices().Clear();
    savePicker.FileTypeChoices().Insert(L"PNG image", winrt::single_threaded_vector<winrt::hstring>({ L".png" }));
    savePicker.FileTypeChoices().Insert(L"JPG image", winrt::single_threaded_vector<winrt::hstring>({ L".jpg" }));
    auto file = co_await savePicker.PickSaveFileAsync();
    if (file == nullptr)
    {
        co_return nullptr;
    }

    // Decide on the pixel format depending on the image type
    auto fileExtension = file.FileType();
    winrt::guid fileFormatGuid = {};
    winrt::BitmapPixelFormat bitmapPixelFormat;
    winrt::DirectXPixelFormat pixelFormat;
    if (fileExtension == L".png")
    {
        fileFormatGuid = winrt::BitmapEncoder::PngEncoderId();
        bitmapPixelFormat = winrt::BitmapPixelFormat::Bgra8;
        pixelFormat = winrt::DirectXPixelFormat::B8G8R8A8UIntNormalized;
    }
    else if (fileExtension == L".jpg" || fileExtension == L".jpeg")
    {
        fileFormatGuid = winrt::BitmapEncoder::JpegEncoderId();
        bitmapPixelFormat = winrt::BitmapPixelFormat::Bgra8;
        pixelFormat = winrt::DirectXPixelFormat::B8G8R8A8UIntNormalized;
    }
    else
    {
        // Unsupported
        co_await wil::resume_foreground(m_mainThread);
        MessageBoxW(nullptr,
            L"Unsupported file format!",
            L"Cap",
            MB_OK | MB_ICONERROR);
        co_return nullptr;
    }

    {
        // Get the file stream
        auto stream = co_await file.OpenAsync(winrt::FileAccessMode::ReadWrite);

        // Initialize the encoder
        auto encoder = co_await winrt::BitmapEncoder::CreateAsync(fileFormatGuid, stream);

        // Take the snapshot
        auto texture = co_await CaptureSnapshot::TakeAsync(m_device, item, pixelFormat);

        // Encode the image
        D3D11_TEXTURE2D_DESC desc = {};
        texture->GetDesc(&desc);
        auto bytes = util::CopyBytesFromTexture(texture);
        encoder.SetPixelData(
            bitmapPixelFormat,
            winrt::BitmapAlphaMode::Premultiplied,
            desc.Width,
            desc.Height,
            1.0,
            1.0,
            bytes);
        co_await encoder.FlushAsync();
    }

    co_return file;
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
