#pragma once
#include <windows.h>
#include <vector>

std::vector<uint8_t> DownscalePixels(std::vector<uint8_t>& srcPixels, uint32_t srcWidth, uint32_t srcHeight, float scale);
HRESULT SaveGray8AsPng(const std::vector<uint8_t>& grayData, UINT width, UINT height, LPCWSTR outputFilePath);
winrt::Windows::Foundation::IAsyncOperation<winrt::Windows::Storage::Streams::InMemoryRandomAccessStream> ConvertGray8ToImageStream(
	std::vector<uint8_t>& grayData, uint32_t width, uint32_t height);
