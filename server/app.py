from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import os
import random
import json
import httpx
from werkzeug.utils import secure_filename
import sys
import io
from dotenv import load_dotenv
load_dotenv()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

app = FastAPI(title="Multi-Modal Sentiment Analyzer API with LLM")
# Line 29-35: Update CORS Configuration
FRONTEND_URL = os.getenv('FRONTEND_URL', '')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "https://*.netlify.app"],  # Add Netlify domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')  # Set your API key in environment variable
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "deepseek/deepseek-chat"  # DeepSeek V3 model

# Pydantic Models
class TextRequest(BaseModel):
    text: str
    use_llm: Optional[bool] = False  # Flag to use LLM analysis

class SentimentResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    filename: Optional[str] = None
    result: Optional[dict] = None
    results: Optional[dict] = None
    error: Optional[str] = None

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class LLMSentimentAnalyzer:
    def __init__(self):
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except:
            self.lemmatizer = None
            self.stop_words = set()
        
        # Keyword dictionaries for better analysis
        self.positive_words = {'love', 'excellent', 'amazing', 'great', 'wonderful', 'fantastic', 'good', 'best', 'awesome', 'perfect'}
        self.negative_words = {'hate', 'terrible', 'awful', 'bad', 'worst', 'horrible', 'useless', 'disappointing', 'poor'}
        
        # Check if OpenRouter API key is available
        self.llm_available = bool(OPENROUTER_API_KEY)
        if self.llm_available:
            print("✅ OpenRouter API key found - DeepSeek V3 LLM analysis available")
        else:
            print("⚠️ OpenRouter API key not found - Using traditional analysis only")
    
    async def analyze_with_llm(self, text: str) -> dict:
        """Analyze sentiment using OpenRouter API with DeepSeek V3"""
        if not self.llm_available:
            raise Exception("OpenRouter API key not configured")
        
        try:
            prompt = f"""Analyze the sentiment of the following text and provide a detailed analysis in JSON format.

Text: "{text}"

Provide your analysis in the following JSON structure:
{{
    "sentiment": "positive/negative/neutral/mixed",
    "confidence": 0.0-1.0,
    "probabilities": {{
        "positive": 0.0-1.0,
        "negative": 0.0-1.0,
        "neutral": 0.0-1.0,
        "mixed": 0.0-1.0
    }},
    "emotion": "primary emotion detected",
    "key_phrases": ["list", "of", "key phrases"],
    "reasoning": "brief explanation of the sentiment analysis"
}}

Make sure probabilities sum to 1.0. Be accurate and nuanced in your analysis. Respond with ONLY the JSON, no additional text."""

            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",  # Optional, for rankings
                "X-Title": "Sentiment Analyzer"  # Optional, for rankings
            }
            
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert sentiment analysis assistant. Provide accurate, nuanced sentiment analysis in valid JSON format only. Do not include any text outside the JSON structure."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1000,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                llm_response = result['choices'][0]['message']['content']
                
                # Extract JSON from response (handle potential markdown code blocks)
                llm_response = llm_response.strip()
                if llm_response.startswith('```'):
                    # Remove markdown code blocks
                    llm_response = re.sub(r'^```json\s*', '', llm_response)
                    llm_response = re.sub(r'^```\s*', '', llm_response)
                    llm_response = re.sub(r'\s*```$', '', llm_response)
                    llm_response = llm_response.strip()
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    
                    # Ensure all required fields are present
                    if 'sentiment' not in analysis:
                        analysis['sentiment'] = 'neutral'
                    if 'confidence' not in analysis:
                        analysis['confidence'] = 0.5
                    if 'probabilities' not in analysis:
                        analysis['probabilities'] = {
                            'positive': 0.25,
                            'negative': 0.25,
                            'neutral': 0.4,
                            'mixed': 0.1
                        }
                    
                    # Ensure all sentiment types are in probabilities
                    for sentiment_type in ['positive', 'negative', 'neutral', 'mixed']:
                        if sentiment_type not in analysis['probabilities']:
                            analysis['probabilities'][sentiment_type] = 0.0
                    
                    # Normalize probabilities
                    total = sum(analysis['probabilities'].values())
                    if total > 0:
                        analysis['probabilities'] = {
                            k: round(v/total, 3) 
                            for k, v in analysis['probabilities'].items()
                        }
                    
                    # Add metadata
                    analysis['type'] = 'text'
                    analysis['model'] = 'llm'
                    analysis['llm_model'] = LLM_MODEL
                    analysis['llm_provider'] = 'openrouter'
                    analysis['confidence'] = round(float(analysis['confidence']), 3)
                    
                    # Get TextBlob metrics for comparison
                    blob = TextBlob(text)
                    analysis['polarity'] = round(blob.sentiment.polarity, 3)
                    analysis['subjectivity'] = round(blob.sentiment.subjectivity, 3)
                    
                    return analysis
                else:
                    raise Exception("Could not parse LLM response as JSON")
                    
        except httpx.HTTPError as e:
            print(f"HTTP error calling OpenRouter API: {e}")
            raise Exception(f"LLM API error: {str(e)}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"LLM Response was: {llm_response[:500]}")
            raise Exception("Invalid JSON response from LLM")
        except Exception as e:
            print(f"LLM analysis error: {e}")
            raise Exception(f"LLM analysis failed: {str(e)}")
    
    def analyze_text_sentiment(self, text: str) -> dict:
        """Analyze sentiment for text with enhanced accuracy (traditional method)"""
        try:
            # Clean text
            text_lower = text.lower()
            
            # Use TextBlob for polarity
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Check for strong positive/negative keywords
            has_positive = any(word in text_lower for word in self.positive_words)
            has_negative = any(word in text_lower for word in self.negative_words)
            
            # Adjust polarity based on keywords
            if has_positive and not has_negative:
                polarity = max(polarity, 0.3)
            elif has_negative and not has_positive:
                polarity = min(polarity, -0.3)
            elif has_positive and has_negative:
                # Mixed sentiment
                sentiment = "mixed"
                confidence = 0.65
                probabilities = {
                    'positive': 0.35,
                    'negative': 0.35,
                    'neutral': 0.15,
                    'mixed': 0.35
                }
                return {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'type': 'text',
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'model': 'textblob'
                }
            
            # Determine sentiment based on polarity
            if polarity > 0.1:
                sentiment = "positive"
                confidence = min(0.55 + polarity * 0.4, 0.95)
            elif polarity < -0.1:
                sentiment = "negative" 
                confidence = min(0.55 + abs(polarity) * 0.4, 0.95)
            else:
                sentiment = "neutral"
                confidence = 0.7
            
            # Create realistic probabilities
            if sentiment == "positive":
                probabilities = {
                    'positive': confidence,
                    'negative': (1 - confidence) * 0.2,
                    'neutral': (1 - confidence) * 0.5,
                    'mixed': (1 - confidence) * 0.3
                }
            elif sentiment == "negative":
                probabilities = {
                    'positive': (1 - confidence) * 0.2,
                    'negative': confidence,
                    'neutral': (1 - confidence) * 0.5,
                    'mixed': (1 - confidence) * 0.3
                }
            else:
                probabilities = {
                    'positive': 0.25,
                    'negative': 0.25,
                    'neutral': confidence,
                    'mixed': 0.2
                }
            
            # Normalize probabilities
            total = sum(probabilities.values())
            probabilities = {k: v/total for k, v in probabilities.items()}
            
            return {
                'sentiment': sentiment,
                'confidence': round(confidence, 3),
                'probabilities': {k: round(v, 3) for k, v in probabilities.items()},
                'type': 'text',
                'polarity': round(polarity, 3),
                'subjectivity': round(subjectivity, 3),
                'model': 'textblob'
            }
        except Exception as e:
            print(f"Text analysis error: {e}")
            return self.fallback_text_analysis()
    
    def fallback_text_analysis(self) -> dict:
        """Fallback for text analysis"""
        return {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'probabilities': {'positive': 0.25, 'negative': 0.25, 'neutral': 0.4, 'mixed': 0.1},
            'type': 'text',
            'model': 'fallback'
        }
    
    async def analyze_image_sentiment(self, image_file: UploadFile) -> dict:
        """Realistic image analysis using file properties"""
        try:
            filename = secure_filename(image_file.filename.lower())
            
            # Read file for size analysis
            content = await image_file.read()
            file_size = len(content)
            
            # Reset file pointer
            await image_file.seek(0)
            
            # Analyze based on filename and size patterns
            analysis = self.analyze_image_properties(filename, file_size)
            
            return analysis
            
        except Exception as e:
            print(f"Image analysis error: {e}")
            return self.fallback_image_analysis()
    
    def analyze_image_properties(self, filename: str, file_size: int) -> dict:
        """Analyze image based on filename patterns and file size"""
        # Sentiment keywords in filename
        positive_keywords = ['happy', 'smile', 'joy', 'fun', 'party', 'celebration', 'sunset', 'beach', 
                           'nature', 'colorful', 'bright', 'beautiful', 'love', 'peace', 'flower']
        negative_keywords = ['sad', 'angry', 'cry', 'storm', 'dark', 'lonely', 'broken', 'accident',
                           'war', 'fire', 'disaster', 'crash']
        neutral_keywords = ['document', 'scan', 'product', 'object', 'building', 'street', 'text',
                          'screenshot', 'diagram', 'chart']
        
        # Analyze filename
        filename_sentiment = "neutral"
        filename_confidence = 0.5
        matched_keywords = []
        
        for keyword in positive_keywords:
            if keyword in filename:
                filename_sentiment = "positive"
                filename_confidence = 0.75
                matched_keywords.append(keyword)
                break
                
        for keyword in negative_keywords:
            if keyword in filename:
                filename_sentiment = "negative"
                filename_confidence = 0.7
                matched_keywords.append(keyword)
                break
                
        for keyword in neutral_keywords:
            if keyword in filename:
                filename_sentiment = "neutral"
                filename_confidence = 0.65
                matched_keywords.append(keyword)
                break
        
        # File size analysis (more sophisticated)
        if file_size > 2000000:  # >2MB - high quality/detail
            size_sentiment = "positive"
            size_confidence = 0.7
            detail = "high_detail"
        elif file_size > 500000:  # 500KB-2MB - good quality
            size_sentiment = "positive"
            size_confidence = 0.65
            detail = "moderate_detail"
        elif file_size < 100000:  # <100KB - compressed/simple
            size_sentiment = "neutral"
            size_confidence = 0.55
            detail = "low_detail"
        else:
            size_sentiment = "neutral"
            size_confidence = 0.6
            detail = "standard_detail"
        
        # File type analysis
        file_extension = filename.split('.')[-1] if '.' in filename else ''
        type_analysis = {
            'jpg': {'sentiment': 'neutral', 'confidence': 0.6},
            'jpeg': {'sentiment': 'neutral', 'confidence': 0.6},
            'png': {'sentiment': 'positive', 'confidence': 0.65},
            'gif': {'sentiment': 'positive', 'confidence': 0.7},
            'bmp': {'sentiment': 'neutral', 'confidence': 0.5}
        }.get(file_extension, {'sentiment': 'neutral', 'confidence': 0.5})
        
        # Weighted combination
        sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
        sentiments[filename_sentiment] += filename_confidence * 0.5  # Filename most important
        sentiments[size_sentiment] += size_confidence * 0.3
        sentiments[type_analysis['sentiment']] += type_analysis['confidence'] * 0.2
        
        final_sentiment = max(sentiments, key=sentiments.get)
        final_confidence = min(sentiments[final_sentiment], 0.9)
        
        # Generate emotion
        emotion_map = {
            'positive': ['joyful', 'vibrant', 'peaceful', 'exciting', 'cheerful', 'uplifting'],
            'negative': ['somber', 'intense', 'melancholic', 'dramatic', 'moody', 'tense'],
            'neutral': ['balanced', 'calm', 'straightforward', 'clear', 'moderate', 'stable']
        }
        emotion = random.choice(emotion_map[final_sentiment])
        
        # Create realistic probabilities
        if final_sentiment == 'positive':
            base_probs = {
                'positive': 0.5 + random.uniform(0, 0.3),
                'negative': random.uniform(0.05, 0.15),
                'neutral': random.uniform(0.1, 0.3),
                'mixed': random.uniform(0.05, 0.15)
            }
        elif final_sentiment == 'negative':
            base_probs = {
                'positive': random.uniform(0.05, 0.15),
                'negative': 0.5 + random.uniform(0, 0.3),
                'neutral': random.uniform(0.1, 0.3),
                'mixed': random.uniform(0.05, 0.15)
            }
        else:
            base_probs = {
                'positive': random.uniform(0.15, 0.3),
                'negative': random.uniform(0.15, 0.3),
                'neutral': 0.4 + random.uniform(0, 0.2),
                'mixed': random.uniform(0.05, 0.15)
            }
        
        # Normalize
        total = sum(base_probs.values())
        probabilities = {k: round(v/total, 3) for k, v in base_probs.items()}
        
        return {
            'sentiment': final_sentiment,
            'confidence': round(final_confidence, 3),
            'type': 'image',
            'emotion': emotion,
            'probabilities': probabilities,
            'file_size': file_size,
            'detail_level': detail,
            'model': 'image_heuristic'
        }
    
    def fallback_image_analysis(self) -> dict:
        """Fallback method for image analysis"""
        return {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'type': 'image',
            'emotion': 'unknown',
            'probabilities': {'positive': 0.25, 'negative': 0.25, 'neutral': 0.4, 'mixed': 0.1},
            'model': 'fallback'
        }

# Initialize analyzer
analyzer = LLMSentimentAnalyzer()

@app.get('/')
@app.head('/')
async def root():
    return {
        "message": "Multi-Modal Sentiment Analyzer API with DeepSeek V3",
        "status": "running",
        "llm_available": analyzer.llm_available,
        "llm_model": LLM_MODEL if analyzer.llm_available else None,
        "llm_provider": "OpenRouter (DeepSeek V3)" if analyzer.llm_available else None,
        "endpoints": [
            "/analyze/text (supports use_llm flag)",
            "/analyze/text/llm (LLM-only endpoint)",
            "/analyze/image",
            "/analyze/both",
            "/health",
            "/llm/status"
        ]
    }

@app.get('/llm/status')
async def llm_status():
    """Check LLM availability status"""
    return {
        "llm_available": analyzer.llm_available,
        "llm_model": LLM_MODEL if analyzer.llm_available else None,
        "llm_provider": "OpenRouter" if analyzer.llm_available else None,
        "api_key_configured": bool(OPENROUTER_API_KEY),
        "message": "DeepSeek V3 ready via OpenRouter" if analyzer.llm_available else "Configure OPENROUTER_API_KEY environment variable"
    }

@app.post('/analyze/text')
async def analyze_text_sentiment(request: TextRequest):
    try:
        text = request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail='No text provided')
        
        if len(text) > 5000:
            raise HTTPException(status_code=400, detail='Text too long (max 5000 characters)')
        
        # Check if LLM analysis is requested
        if request.use_llm:
            if not analyzer.llm_available:
                raise HTTPException(
                    status_code=503, 
                    detail='LLM not available. Configure OPENROUTER_API_KEY environment variable.'
                )
            result = await analyzer.analyze_with_llm(text)
        else:
            result = analyzer.analyze_text_sentiment(text)
        
        return {
            'success': True,
            'text': text[:100] + ('...' if len(text) > 100 else ''),
            'result': result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in text analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/analyze/text/llm')
async def analyze_text_with_llm(request: TextRequest):
    """Dedicated endpoint for LLM-based text analysis using DeepSeek V3"""
    try:
        text = request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail='No text provided')
        
        if len(text) > 5000:
            raise HTTPException(status_code=400, detail='Text too long (max 5000 characters)')
        
        if not analyzer.llm_available:
            raise HTTPException(
                status_code=503, 
                detail='LLM not available. Configure OPENROUTER_API_KEY environment variable.'
            )
        
        result = await analyzer.analyze_with_llm(text)
        
        return {
            'success': True,
            'text': text[:100] + ('...' if len(text) > 100 else ''),
            'result': result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in LLM text analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/analyze/image')
async def analyze_image_sentiment(image: UploadFile = File(...)):
    try:
        if not image.filename:
            raise HTTPException(status_code=400, detail='No image selected')
        
        if not allowed_file(image.filename):
            raise HTTPException(
                status_code=400, 
                detail='Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'
            )
        
        # Check file size
        content = await image.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail='File too large (max 10MB)')
        
        # Reset file pointer for analysis
        await image.seek(0)
        
        result = await analyzer.analyze_image_sentiment(image)
        
        return {
            'success': True,
            'filename': secure_filename(image.filename),
            'result': result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in image analysis: {e}")
        raise HTTPException(status_code=500, detail='Internal server error')

@app.post('/analyze/both')
async def analyze_both(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    use_llm: Optional[bool] = Form(False)
):
    """Analyze both text and image together"""
    try:
        results = {}
        
        if text:
            text = text.strip()
            if len(text) > 5000:
                raise HTTPException(status_code=400, detail='Text too long (max 5000 characters)')
            
            if use_llm:
                if not analyzer.llm_available:
                    raise HTTPException(
                        status_code=503, 
                        detail='LLM not available. Configure OPENROUTER_API_KEY environment variable.'
                    )
                results['text'] = await analyzer.analyze_with_llm(text)
            else:
                results['text'] = analyzer.analyze_text_sentiment(text)
        
        if image and image.filename:
            if not allowed_file(image.filename):
                raise HTTPException(status_code=400, detail='Invalid image file type')
            
            # Check file size
            content = await image.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail='File too large (max 10MB)')
            
            await image.seek(0)
            results['image'] = await analyzer.analyze_image_sentiment(image)
        
        if not results:
            raise HTTPException(status_code=400, detail='No text or image provided')
        
        return {
            'success': True,
            'results': results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in combined analysis: {e}")
        raise HTTPException(status_code=500, detail='Internal server error')

@app.get('/health')
async def health():
    return {
        'status': 'healthy',
        'llm_available': analyzer.llm_available,
        'llm_provider': 'OpenRouter (DeepSeek V3)' if analyzer.llm_available else None
    }

if __name__ == '__main__':
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    print("=" * 60)
    print("Starting Multi-Modal Sentiment Analyzer with DeepSeek V3...")
    print(f"Server running at: http://{host}:{port}")
    print(f"LLM Status: {'DeepSeek V3 Available' if analyzer.llm_available else 'Not Configured'}")
    if not analyzer.llm_available:
        print("To enable LLM: Set OPENROUTER_API_KEY environment variable")
        print("Get your key at: https://openrouter.ai/keys")
    print("Press CTRL+C to stop")
    print("=" * 60)
    
    uvicorn.run(app, host=host, port=port)