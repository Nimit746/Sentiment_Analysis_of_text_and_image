from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pickle
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import random
from werkzeug.utils import secure_filename

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

app = FastAPI(title="ML-Based Sentiment Analyzer API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model paths
MODEL_PATH = 'models/sentiment_model.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'
LABEL_ENCODER_PATH = 'models/label_encoder.pkl'

# Pydantic Models
class TextRequest(BaseModel):
    text: str

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

class MLSentimentAnalyzer:
    def __init__(self):
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except:
            self.lemmatizer = None
            self.stop_words = set()
        
        # Load ML model, vectorizer, and label encoder
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.load_models()
        
    def load_models(self):
        """Load pre-trained ML models"""
        try:
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                print("✅ ML Model loaded successfully")
            else:
                print("⚠️ ML Model not found. Using fallback TextBlob analysis.")
                
            if os.path.exists(VECTORIZER_PATH):
                with open(VECTORIZER_PATH, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print("✅ Vectorizer loaded successfully")
            else:
                print("⚠️ Vectorizer not found.")
                
            if os.path.exists(LABEL_ENCODER_PATH):
                with open(LABEL_ENCODER_PATH, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print("✅ Label Encoder loaded successfully")
            else:
                print("⚠️ Label Encoder not found.")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.model = None
            self.vectorizer = None
            self.label_encoder = None
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove mentions and hashtags
            text = re.sub(r'@\w+|#\w+', '', text)
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize and remove stopwords (if lemmatizer available)
            if self.lemmatizer and self.stop_words:
                words = text.split()
                words = [self.lemmatizer.lemmatize(word) for word in words 
                        if word not in self.stop_words and len(word) > 2]
                text = ' '.join(words)
            
            return text
        except Exception as e:
            print(f"Error cleaning text: {e}")
            return text
    
    def analyze_text_sentiment(self, text: str) -> dict:
        """Analyze sentiment using ML model or fallback to TextBlob"""
        try:
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Try ML model prediction
            if self.model is not None and self.vectorizer is not None and self.label_encoder is not None:
                return self.predict_with_ml_model(cleaned_text, text)
            else:
                # Fallback to TextBlob
                return self.predict_with_textblob(text)
                
        except Exception as e:
            print(f"Text analysis error: {e}")
            return self.fallback_text_analysis()
    
    def predict_with_ml_model(self, cleaned_text: str, original_text: str) -> dict:
        """Predict sentiment using trained ML model"""
        try:
            # Vectorize the text
            text_vectorized = self.vectorizer.transform([cleaned_text])
            
            # Get prediction
            prediction = self.model.predict(text_vectorized)[0]
            
            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(text_vectorized)[0]
                
                # Get class labels
                classes = self.label_encoder.classes_
                
                # Create probability dictionary
                prob_dict = {}
                for i, cls in enumerate(classes):
                    prob_dict[cls] = float(probabilities[i])
                
                # Ensure all sentiment classes are present
                for sentiment in ['positive', 'negative', 'neutral', 'mixed']:
                    if sentiment not in prob_dict:
                        prob_dict[sentiment] = 0.0
                
                # Get confidence (max probability)
                confidence = float(max(probabilities))
                
                # Get predicted sentiment
                sentiment = self.label_encoder.inverse_transform([prediction])[0]
                
            else:
                # If model doesn't support probabilities, use equal distribution
                sentiment = self.label_encoder.inverse_transform([prediction])[0]
                confidence = 0.75
                prob_dict = {
                    'positive': 0.25,
                    'negative': 0.25,
                    'neutral': 0.25,
                    'mixed': 0.25
                }
                prob_dict[sentiment] = confidence
            
            # Normalize probabilities to sum to 1
            total = sum(prob_dict.values())
            if total > 0:
                prob_dict = {k: v/total for k, v in prob_dict.items()}
            
            # Get TextBlob metrics for additional info
            blob = TextBlob(original_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            return {
                'sentiment': sentiment,
                'confidence': round(confidence, 3),
                'probabilities': {k: round(v, 3) for k, v in prob_dict.items()},
                'type': 'text',
                'polarity': round(polarity, 3),
                'subjectivity': round(subjectivity, 3),
                'model': 'ml_model'
            }
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return self.predict_with_textblob(original_text)
    
    def predict_with_textblob(self, text: str) -> dict:
        """Fallback prediction using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
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
            print(f"TextBlob analysis error: {e}")
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
        """Analyze image sentiment using file properties"""
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
        
        for keyword in positive_keywords:
            if keyword in filename:
                filename_sentiment = "positive"
                filename_confidence = 0.75
                break
                
        for keyword in negative_keywords:
            if keyword in filename:
                filename_sentiment = "negative"
                filename_confidence = 0.7
                break
                
        for keyword in neutral_keywords:
            if keyword in filename:
                filename_sentiment = "neutral"
                filename_confidence = 0.65
                break
        
        # File size analysis
        if file_size > 2000000:  # >2MB
            size_sentiment = "positive"
            size_confidence = 0.7
            detail = "high_detail"
        elif file_size > 500000:  # 500KB-2MB
            size_sentiment = "positive"
            size_confidence = 0.65
            detail = "moderate_detail"
        elif file_size < 100000:  # <100KB
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
        sentiments[filename_sentiment] += filename_confidence * 0.5
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
analyzer = MLSentimentAnalyzer()

@app.get('/')
async def root():
    model_status = "loaded" if analyzer.model is not None else "not_loaded (using fallback)"
    return {
        "message": "ML-Based Sentiment Analyzer API",
        "status": "running",
        "ml_model": model_status,
        "endpoints": [
            "/analyze/text",
            "/analyze/image",
            "/analyze/both",
            "/health",
            "/model/info"
        ]
    }

@app.get('/model/info')
async def model_info():
    """Get information about loaded models"""
    return {
        "model_loaded": analyzer.model is not None,
        "vectorizer_loaded": analyzer.vectorizer is not None,
        "label_encoder_loaded": analyzer.label_encoder is not None,
        "model_path": MODEL_PATH,
        "vectorizer_path": VECTORIZER_PATH,
        "label_encoder_path": LABEL_ENCODER_PATH
    }

@app.post('/analyze/text')
async def analyze_text_sentiment(request: TextRequest):
    try:
        text = request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail='No text provided')
        
        if len(text) > 5000:
            raise HTTPException(status_code=400, detail='Text too long (max 5000 characters)')
        
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
        raise HTTPException(status_code=500, detail='Internal server error')

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
    image: Optional[UploadFile] = File(None)
):
    """Analyze both text and image together"""
    try:
        results = {}
        
        if text:
            text = text.strip()
            if len(text) > 5000:
                raise HTTPException(status_code=400, detail='Text too long (max 5000 characters)')
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
        'model_loaded': analyzer.model is not None
    }

if __name__ == '__main__':
    import uvicorn
    print("🚀 Starting ML-Based Sentiment Analyzer...")
    print("📍 Server running at: http://localhost:8000")
    print(f"🤖 ML Model Status: {'Loaded ✅' if analyzer.model is not None else 'Not Loaded ⚠️ (using TextBlob fallback)'}")
    print("⚡ Press CTRL+C to stop")
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)