#include "pch.h"
#include "Image.h"

namespace winrt {
    using namespace winrt::Windows::Foundation;
    using namespace winrt::Windows::Storage::Streams;
}

std::vector<uint8_t> DownscalePixels(std::vector<uint8_t>& srcPixels, uint32_t srcWidth, uint32_t srcHeight, float scale)
{
    if (scale <= 1) return srcPixels;
    uint32_t dstWidth = srcWidth / scale;
    uint32_t dstHeight = srcHeight / scale;
    std::vector<uint8_t> dstPixels(dstWidth * dstHeight * 4);

    for (uint32_t y = 0; y < dstHeight; ++y) {
        uint32_t srcY = y * scale;
        for (uint32_t x = 0; x < dstWidth; ++x) {
            uint32_t srcX = x * scale;
            memcpy(&dstPixels[(y * dstWidth + x) * 4], &srcPixels[(srcY * srcWidth + srcX) * 4], 4);
        }
    }
    return dstPixels;
}

HRESULT SaveGray8AsPng(const std::vector<uint8_t>& grayData, UINT width, UINT height, LPCWSTR outputFilePath)
{
    HRESULT hr = S_OK;
    IWICImagingFactory* pFactory = nullptr;
    IWICBitmapEncoder* pEncoder = nullptr;
    IWICBitmapFrameEncode* pFrame = nullptr;
    IWICStream* pStream = nullptr;
    WICPixelFormatGUID format = GUID_WICPixelFormat8bppGray;
    UINT stride = width;

    if (SUCCEEDED(hr)) hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pFactory));
    if (SUCCEEDED(hr)) hr = pFactory->CreateStream(&pStream);
    if (SUCCEEDED(hr)) hr = pStream->InitializeFromFilename(outputFilePath, GENERIC_WRITE);
    if (SUCCEEDED(hr)) hr = pFactory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &pEncoder);
    if (SUCCEEDED(hr)) hr = pEncoder->Initialize(pStream, WICBitmapEncoderNoCache);
    if (SUCCEEDED(hr)) hr = pEncoder->CreateNewFrame(&pFrame, nullptr);
    if (SUCCEEDED(hr)) hr = pFrame->Initialize(nullptr);  // Default options
    if (SUCCEEDED(hr)) hr = pFrame->SetSize(width, height);
    if (SUCCEEDED(hr)) hr = pFrame->SetPixelFormat(&format);
    if (SUCCEEDED(hr)) hr = pFrame->WritePixels(height, stride, stride * height, const_cast<BYTE*>(grayData.data()));
    if (SUCCEEDED(hr)) hr = pFrame->Commit();
    if (SUCCEEDED(hr)) hr = pEncoder->Commit();

    if (pFrame) pFrame->Release();
    if (pEncoder) pEncoder->Release();
    if (pStream) pStream->Release();
    if (pFactory) pFactory->Release();
    return hr;
}

winrt::IAsyncOperation<winrt::InMemoryRandomAccessStream> ConvertGray8ToImageStream(std::vector<uint8_t>& grayData, uint32_t width, uint32_t height)
{
    winrt::InMemoryRandomAccessStream stream;
    auto wicFactory = winrt::create_instance<IWICImagingFactory>(CLSID_WICImagingFactory);
    winrt::com_ptr<IWICBitmapEncoder> encoder;
    wicFactory->CreateEncoder(GUID_ContainerFormatPng, nullptr, encoder.put());
    encoder->Initialize(stream.as<IStream>().get(), WICBitmapEncoderNoCache);

    winrt::com_ptr<IWICBitmapFrameEncode> frame;
    encoder->CreateNewFrame(frame.put(), nullptr);
    frame->Initialize(nullptr);
    frame->SetSize(width, height);
    WICPixelFormatGUID format = GUID_WICPixelFormat8bppGray;
    frame->SetPixelFormat(&format);
    frame->WritePixels(height, width, static_cast<UINT>(grayData.size()), const_cast<uint8_t*>(grayData.data()));

    frame->Commit();
    encoder->Commit();

    co_return stream;
}
