# 🚀 AI Multi-Modal Sentiment Analyzer

<div align="center">

![AI Sentiment Analyzer](https://img.shields.io/badge/AI-Powered-blue?style=for-the-badge&logo=artificial-intelligence)
![React](https://img.shields.io/badge/React-18.1.1-61DAFB?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python)
![Vite](https://img.shields.io/badge/Vite-7.1.7-646CFF?style=for-the-badge&logo=vite)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-4.1.15-38B2AC?style=for-the-badge&logo=tailwind-css)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Node.js Version](https://img.shields.io/badge/Node.js-18+-339933?style=for-the-badge&logo=node.js)](https://nodejs.org/)
[![NPM Version](https://img.shields.io/badge/NPM-9+-CB3837?style=for-the-badge&logo=npm)](https://www.npmjs.com/)

*Unlock the power of AI-driven sentiment analysis across text and images!*

[🌐 Live Demo](https://projectaura7.netlify.app/) | [📖 Documentation](#) | [🐛 Report Bug](https://github.com/yourusername/ai-sentiment-analyzer/issues) | [✨ Request Feature](https://github.com/yourusername/ai-sentiment-analyzer/issues)

</div>

---

## 📋 Table of Contents

- [✨ Overview](#-overview)
- [🎯 Features](#-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🎮 Usage](#-usage)
- [🔌 API Endpoints](#-api-endpoints)
- [📊 How It Works](#-how-it-works)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👨‍💻 Author](#-author)
- [🙏 Acknowledgments](#-acknowledgments)

---

## ✨ Overview

**AI Multi-Modal Sentiment Analyzer** is a cutting-edge full-stack web application that leverages artificial intelligence to analyze sentiment in both text and images. Built with modern technologies, this tool provides real-time sentiment analysis with interactive visualizations, making it perfect for businesses, researchers, and developers who need to understand emotional content across multiple modalities.

### 🎯 Key Highlights

- **Multi-Modal Analysis**: Analyze text, images, or both simultaneously
- **Real-Time Processing**: Instant sentiment detection with confidence scores
- **Interactive Visualizations**: Beautiful charts and metrics powered by Recharts
- **Modern UI/UX**: Sleek, responsive design with dark theme
- **AI-Powered**: Advanced algorithms using NLTK, TextBlob, and custom logic
- **Cross-Platform**: Works seamlessly on desktop and mobile devices

---

## 🎯 Features

### 🔍 Analysis Capabilities

- **📝 Text Sentiment Analysis**
  - Real-time text processing
  - Polarity and subjectivity scoring
  - Confidence-based sentiment classification
  - Support for multiple languages (English primary)

- **🖼️ Image Sentiment Analysis**
  - Filename-based emotion detection
  - File size and format analysis
  - Emotion mapping (joyful, somber, etc.)
  - Support for JPG, PNG, GIF, BMP formats

- **📊 Combined Analysis**
  - Simultaneous text and image processing
  - Unified results dashboard
  - Comparative sentiment metrics

### 🎨 User Experience

- **🌙 Dark Theme**: Eye-friendly interface optimized for extended use
- **📱 Responsive Design**: Perfect on all screen sizes
- **⚡ Fast Performance**: Optimized with Vite and modern React
- **🎯 Interactive Charts**: Dynamic sentiment distribution graphs
- **💡 Smart Examples**: Pre-loaded sample texts for quick testing
- **🔄 Real-Time Feedback**: Instant results with loading animations

### 🛡️ Technical Features

- **🔒 CORS Enabled**: Secure cross-origin requests
- **📏 File Size Limits**: 10MB max upload size
- **🛠️ Error Handling**: Comprehensive error management
- **📈 Scalable Architecture**: Modular design for easy extension

---

## 🛠️ Tech Stack

### Frontend
- **React 19** - Modern UI framework
- **Vite** - Lightning-fast build tool
- **TailwindCSS 4** - Utility-first CSS framework
- **Recharts** - Composable charting library
- **Lucide React** - Beautiful icons
- **Axios** - HTTP client for API calls

### Backend
- **FastAPI** - High-performance Python web framework
- **Python 3.8+** - Core programming language
- **NLTK** - Natural Language Toolkit
- **TextBlob** - Simple NLP library
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

### Development Tools
- **ESLint** - Code linting
- **Concurrently** - Run multiple commands
- **Nodemon** - Auto-restart for development

---

## 🚀 Quick Start

Get up and running in less than 5 minutes!

### Prerequisites

- **Node.js** (v18 or higher)
- **Python** (v3.8 or higher)
- **Git** (for cloning the repository)

### One-Command Setup

```bash
# Clone the repository
git clone https://github.com/Nimit746/Sentiment_Analysis_of_text_and_image.git
cd Sentiment_Analysis_of_text_and_image

# Install all dependencies and start development servers
npm run setup:all  # (if available) or follow manual steps below
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Nimit746/Sentiment_Analysis_of_text_and_image.git
cd Sentiment_Analysis_of_text_and_image
```

### 2. Backend Setup

```bash
# Navigate to server directory
cd server

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install fastapi uvicorn python-multipart textblob nltk werkzeug

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"

# Return to root directory
cd ..
```

### 3. Frontend Setup

```bash
# Install Node.js dependencies
npm install

# Install client dependencies
cd client
npm install
cd ..
```

### 4. Start Development Servers

```bash
# Start all services (frontend, backend, AI API)
npm run dev

# Or run individually:
npm run frontend    # React dev server on :5173
npm run backend     # Node.js server (if applicable)
npm run ai          # FastAPI server on :8000
```

### 5. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Swagger UI)

---

## 🎮 Usage

### Text Analysis

1. Navigate to the **Sentiment Analysis** page
2. Select **📝 Text Analysis** mode
3. Enter your text or click an example
4. Click **"Analyze Text"**
5. View results with sentiment, confidence, and charts

### Image Analysis

1. Select **🖼️ Image Analysis** mode
2. Upload an image (JPG, PNG, GIF, BMP)
3. Click **"Analyze Image"**
4. Review sentiment based on filename and properties

### Combined Analysis

1. Choose **📊 Combined Analysis** mode
2. Enter text (optional) and/or upload image
3. Click **"Analyze Both"**
4. Compare results side-by-side

### Example Texts

Try these sample texts:
- *"I love this product! It's amazing! 😍"*
- *"This is terrible and useless. Worst ever! 😡"*
- *"The product is okay, nothing special 😐"*

---

## 🔌 API Endpoints

### Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API status and welcome message |
| `POST` | `/analyze/text` | Analyze text sentiment |
| `POST` | `/analyze/image` | Analyze image sentiment |
| `POST` | `/analyze/both` | Analyze both text and image |
| `GET` | `/health` | Health check endpoint |

### Text Analysis Request

```bash
curl -X POST "http://localhost:8000/analyze/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this amazing product!"}'
```

### Response Format

```json
{
  "success": true,
  "text": "I love this amazing product!",
  "result": {
    "sentiment": "positive",
    "confidence": 0.875,
    "probabilities": {
      "positive": 0.875,
      "negative": 0.025,
      "neutral": 0.075,
      "mixed": 0.025
    },
    "type": "text",
    "polarity": 0.612,
    "subjectivity": 0.867
  }
}
```

---

## 📊 How It Works

### Text Analysis Algorithm

1. **Preprocessing**: Clean and tokenize input text
2. **Sentiment Scoring**: Use TextBlob for polarity and subjectivity
3. **Keyword Analysis**: Check for positive/negative word patterns
4. **Confidence Calculation**: Weighted scoring based on multiple factors
5. **Classification**: Determine sentiment (positive/negative/neutral/mixed)

### Image Analysis Algorithm

1. **Filename Analysis**: Extract sentiment clues from filename
2. **File Properties**: Analyze size, format, and metadata
3. **Pattern Recognition**: Use predefined emotion mappings
4. **Weighted Scoring**: Combine multiple analysis methods
5. **Emotion Detection**: Map to emotional categories

### Combined Analysis

- Processes text and image independently
- Presents unified results dashboard
- Allows comparison of different modalities

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `npm test`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Guidelines

- Follow the existing code style
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass
- Use meaningful commit messages

### Areas for Contribution

- [ ] Add support for more languages
- [ ] Implement advanced NLP models
- [ ] Add batch processing capabilities
- [ ] Create mobile app version
- [ ] Add export functionality

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Nimit Gupta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👨‍💻 Author

**Nimit Gupta**

- 📧 Email: [guptanimit062@gmail.com](mailto:guptanimit062@gmail.com)
- 🔗 LinkedIn: [Nimit Gupta](www.linkedin.com/in/nimitg726)
- 🐙 GitHub: [Nimit Gupta](https://github.com/Nimit746)
- 🌐 Portfolio: [My Portfolio](https://thegrowthengineer.netlify.app/)

---

## 🙏 Acknowledgments

- **FastAPI** - For the amazing Python web framework
- **React** - For the powerful frontend library
- **TextBlob** - For simple and effective NLP
- **NLTK** - For comprehensive natural language processing
- **TailwindCSS** - For beautiful and responsive styling
- **Recharts** - For stunning data visualizations

### Special Thanks

- Open source community for inspiration and tools
- Contributors and beta testers
- Everyone who provided feedback and suggestions

---

<div align="center">

**Made with ❤️ by Nimit Gupta**

⭐ Star this repo if you found it helpful!

[⬆️ Back to Top](#-ai-multi-modal-sentiment-analyzer)

</div>
