# Interview Anti-Cheat System

A real-time monitoring system that uses OCR to detect potential cheating behaviors during remote coding interviews by analyzing screen content and typing patterns.

## Overview

This system continuously monitors screen content during coding interviews to detect suspicious activities that may indicate cheating, such as:
- Sudden appearance of large code blocks (copy-paste from external sources)
- Abnormal typing speed patterns
- Pre-written solutions appearing instantly

## Goals

1. **Real-time Detection** - Monitor screen content continuously during interviews
2. **Typing Pattern Analysis** - Detect anomalous typing speeds and patterns
3. **Code Authenticity** - Identify when code appears too quickly to be typed normally
4. **Non-intrusive Monitoring** - Minimal impact on legitimate interview process

## Features

### Current Implementation
- ✅ OCR-based word counting using EasyOCR
- ✅ Flask web service for image processing
- ✅ Support for screenshot analysis
- ✅ Basic word count tracking

### Planned Features
- 🔄 **Keras OCR Integration** (in progress)
- 📊 **Typing Speed Analysis** - Track WPM and detect sudden spikes
- 🚨 **Anomaly Detection** - Flag suspicious patterns
- 📸 **Continuous Screen Monitoring** - Automated screenshot capture
- 📈 **Real-time Dashboard** - Live monitoring interface
- 🔍 **Code Pattern Recognition** - Detect common code snippets/solutions
- 📱 **Multi-platform Support** - Windows, macOS, Linux compatibility

## Architecture

```
Screen Capture → OCR Processing → Word Count Analysis → Pattern Detection → Alert System
```

1. **Capture Module** - Takes periodic screenshots
2. **OCR Engine** - Extracts text using EasyOCR/Keras OCR
3. **Analysis Engine** - Processes word counts and typing patterns
4. **Detection System** - Identifies suspicious behavior
5. **Reporting** - Generates alerts and reports

## Usage

### Running the Service
```bash
python count-service.py
```

### API Endpoints
- `GET /` - Health check
- `POST /` - Upload image for word count analysis

## Recent Improvements
- Migrated from pytesseract to EasyOCR for better accuracy
- Added GPU acceleration support
- Optimized processing time
- Enhanced image preprocessing (grayscale, scaling)

## Development Status
Currently on `feature/keras-ocr` branch, working on integrating Keras OCR for improved text recognition accuracy.

