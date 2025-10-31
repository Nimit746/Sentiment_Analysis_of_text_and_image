import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const SentimentAnalysis = () => {
    const [currentMode, setCurrentMode] = useState('text'); // 'text', 'image', 'both'
    const [text, setText] = useState('');
    const [combinedText, setCombinedText] = useState('');
    const [imageFile, setImageFile] = useState(null);
    const [imagePreview, setImagePreview] = useState('');
    const [combinedImageFile, setCombinedImageFile] = useState(null);
    const [combinedImagePreview, setCombinedImagePreview] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [combinedResults, setCombinedResults] = useState(null);
    const [error, setError] = useState('');
    const [useLLM, setUseLLM] = useState(false);

    // Updated API base URL - try localhost:8000 first
    const API_BASE_URL = 'http://localhost:8000';

    const examples = [
        "I love this product! It's amazing! 😊",
        "This is terrible and useless. Worst ever! 😡",
        "The product is okay, nothing special 😐",
        "Great product but delivery was slow 😕"
    ];

    const chartColors = {
        positive: '#10b981',
        negative: '#ef4444',
        neutral: '#eab308',
        mixed: '#a855f7'
    };

    const sentimentEmojis = {
        positive: '😊',
        negative: '😞',
        neutral: '😐',
        mixed: '😕'
    };

    const sentimentDescriptions = {
        positive: 'The analysis shows a generally favorable or optimistic tone.',
        negative: 'The analysis shows a generally unfavorable or pessimistic tone.',
        neutral: 'The analysis shows a balanced or impartial tone.',
        mixed: 'The analysis contains both positive and negative elements.'
    };

    // Mode switching
    const handleModeChange = (mode) => {
        setCurrentMode(mode);
        setResult(null);
        setCombinedResults(null);
        setError('');
    };

    // Text analysis
    const analyzeText = async () => {
        if (!text.trim()) {
            setError('Please enter some text to analyze');
            setTimeout(() => setError(''), 3000);
            return;
        }

        setLoading(true);
        setError('');
        setResult(null);

        try {
            const response = await fetch(`${API_BASE_URL}/analyze/text`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text.trim(),
                    use_llm: useLLM
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Network response was not ok');
            }

            const data = await response.json();

            if (data.success && data.result) {
                setResult(data.result);
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
        } catch (err) {
            console.error('Error:', err);
            setError(err.message || 'Error: Make sure backend is running on http://localhost:8000');
        } finally {
            setLoading(false);
        }
    };

    // Image analysis
    const analyzeImage = async () => {
        if (!imageFile) {
            setError('Please select an image to analyze');
            setTimeout(() => setError(''), 3000);
            return;
        }

        setLoading(true);
        setError('');
        setResult(null);

        const formData = new FormData();
        formData.append('image', imageFile);

        try {
            const response = await fetch(`${API_BASE_URL}/analyze/image`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Network response was not ok');
            }

            const data = await response.json();

            if (data.success && data.result) {
                setResult(data.result);
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
        } catch (err) {
            console.error('Error:', err);
            setError(err.message || 'Error: Make sure backend is running on http://localhost:8000');
        } finally {
            setLoading(false);
        }
    };

    // Combined analysis
    const analyzeBoth = async () => {
        if (!combinedText.trim() && !combinedImageFile) {
            setError('Please provide text, image, or both');
            setTimeout(() => setError(''), 3000);
            return;
        }

        setLoading(true);
        setError('');
        setCombinedResults(null);

        const formData = new FormData();
        if (combinedText.trim()) {
            formData.append('text', combinedText.trim());
        }
        if (combinedImageFile) {
            formData.append('image', combinedImageFile);
        }
        formData.append('use_llm', useLLM);

        try {
            const response = await fetch(`${API_BASE_URL}/analyze/both`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Network response was not ok');
            }

            const data = await response.json();

            if (data.success && data.results) {
                setCombinedResults(data.results);
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
        } catch (err) {
            console.error('Error:', err);
            setError(err.message || 'Error: Make sure backend is running on http://localhost:8000');
        } finally {
            setLoading(false);
        }
    };

    // Image preview handlers
    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setImageFile(file);
            const reader = new FileReader();
            reader.onload = (event) => {
                setImagePreview(event.target.result);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleCombinedImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setCombinedImageFile(file);
            const reader = new FileReader();
            reader.onload = (event) => {
                setCombinedImagePreview(event.target.result);
            };
            reader.readAsDataURL(file);
        }
    };

    const loadExample = (example) => {
        if (currentMode === 'text') {
            setText(example);
        } else if (currentMode === 'both') {
            setCombinedText(example);
        }
    };

    const clearText = () => {
        setText('');
        setResult(null);
        setError('');
    };

    const clearImage = () => {
        setImageFile(null);
        setImagePreview('');
        setResult(null);
        setError('');
    };

    const clearBoth = () => {
        setCombinedText('');
        setCombinedImageFile(null);
        setCombinedImagePreview('');
        setCombinedResults(null);
        setError('');
    };

    const getChartData = (probabilities) => {
        return Object.entries(probabilities).map(([key, value]) => ({
            name: key.charAt(0).toUpperCase() + key.slice(1),
            value: value,
            color: chartColors[key]
        }));
    };

    // Single result display component
    const SingleResult = ({ result }) => (
        <div className="animate-fade-in">
            <h2 className="text-white text-xl sm:text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5">
                Results
            </h2>

            <div className="p-4">
                <div
                    className="bg-cover bg-center flex flex-col items-stretch justify-end rounded-xl pt-[132px] overflow-hidden transform transition-all hover:scale-[1.02] shadow-2xl"
                    style={{
                        backgroundImage: 'linear-gradient(0deg, rgba(0, 0, 0, 0.4) 0%, rgba(0, 0, 0, 0) 100%), url("https://lh3.googleusercontent.com/aida-public/AB6AXuAufPpyiJx9oo_ccxgz-Q0Q8uZT7QFE0edE7_r7X8m6y9K55JrzUx2iDq1OrRW5HcaQNuUUYS4FO_Th9amzO-8tlyImbANsW8hCABH8WhGK9bL6uI55ui1pC7coCgzTjKFit9ORgGUDo_8ospjj3h6qI3jeAU0A2p6ce05Wm5-TOrYC8QSqlH1MTHwKU1fgsevKlKTVXSPklT-idP7cN4FUnmpw8dQl34vm-8HwjLeLGcPMrbZ4Kie4cO2F3PMm3NKPVo2BI1VMqYw")'
                    }}
                >
                    <div className="flex w-full items-end justify-between gap-4 p-4 backdrop-blur-sm bg-black/20">
                        <div className="flex max-w-[440px] flex-1 flex-col gap-1">
                            <p className="text-white tracking-light text-2xl font-bold leading-tight max-w-[440px] flex items-center gap-2">
                                Sentiment
                                <span className="text-3xl">{sentimentEmojis[result.sentiment]}</span>
                            </p>
                            <p className="text-white text-base font-medium leading-normal capitalize">
                                {result.sentiment}
                                {result.emotion && <span className="ml-2 text-sm opacity-80">({result.emotion})</span>}
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* LLM Model Badge */}
            {result.model === 'llm' && (
                <div className="px-4 pb-2">
                    <div className="inline-flex items-center gap-2 bg-linear-to-r from-purple-600 to-blue-600 px-3 py-1 rounded-full">
                        <span className="text-white text-xs font-bold">🤖 Powered by DeepSeek V3</span>
                    </div>
                </div>
            )}

            {/* Confidence Score */}
            <div className="flex flex-col gap-3 p-4">
                <div className="flex gap-6 justify-between items-center">
                    <p className="text-white text-base font-medium leading-normal">Confidence Score</p>
                    <p className="text-white text-lg font-bold leading-normal">
                        {(result.confidence * 100).toFixed(1)}%
                    </p>
                </div>
                <div className="rounded-full bg-[#4d3267] overflow-hidden shadow-inner">
                    <div
                        className="h-3 rounded-full bg-linear-to-r from-[#8013ec] to-[#9124ff] transition-all duration-1000 ease-out shadow-lg"
                        style={{
                            width: `${(result.confidence * 100).toFixed(0)}%`,
                            boxShadow: '0 0 20px rgba(128, 19, 236, 0.5)'
                        }}
                    />
                </div>
            </div>

            {/* LLM-specific data: Key Phrases and Reasoning */}
            {result.key_phrases && result.key_phrases.length > 0 && (
                <div className="px-4 pb-4">
                    <div className="bg-[#2a1a3a] rounded-xl p-4">
                        <h4 className="text-white font-bold mb-2">Key Phrases:</h4>
                        <div className="flex flex-wrap gap-2">
                            {result.key_phrases.map((phrase, idx) => (
                                <span key={idx} className="bg-[#362348] text-white px-3 py-1 rounded-full text-sm">
                                    {phrase}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {result.reasoning && (
                <div className="px-4 pb-4">
                    <div className="bg-[#2a1a3a] rounded-xl p-4">
                        <h4 className="text-white font-bold mb-2">AI Reasoning:</h4>
                        <p className="text-white/90 text-sm leading-relaxed">{result.reasoning}</p>
                    </div>
                </div>
            )}

            {/* Additional Metrics */}
            {(result.polarity !== undefined || result.subjectivity !== undefined || result.file_size !== undefined) && (
                <div className="px-4 pb-4">
                    <div className="bg-[#2a1a3a] rounded-xl p-4 space-y-2">
                        {result.polarity !== undefined && (
                            <div className="flex justify-between items-center">
                                <span className="text-white/80 text-sm">Polarity:</span>
                                <span className="text-white font-semibold">{result.polarity.toFixed(3)}</span>
                            </div>
                        )}
                        {result.subjectivity !== undefined && (
                            <div className="flex justify-between items-center">
                                <span className="text-white/80 text-sm">Subjectivity:</span>
                                <span className="text-white font-semibold">{result.subjectivity.toFixed(3)}</span>
                            </div>
                        )}
                        {result.detail_level && (
                            <div className="flex justify-between items-center">
                                <span className="text-white/80 text-sm">Detail Level:</span>
                                <span className="text-white font-semibold capitalize">{result.detail_level.replace('_', ' ')}</span>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Description */}
            <p className="text-white/90 text-base font-normal leading-normal pb-3 pt-1 px-4 text-center">
                This analysis indicates a <span className="font-bold text-[#8013ec]">{result.sentiment}</span> sentiment with a confidence score of <span className="font-bold text-[#8013ec]">{(result.confidence * 100).toFixed(1)}%</span>. {sentimentDescriptions[result.sentiment]}
            </p>

            {/* Probability Chart */}
            {result.probabilities && (
                <div className="p-4 mt-4">
                    <h3 className="text-white text-lg font-bold mb-4 px-2">Sentiment Distribution</h3>
                    <div className="bg-[#2a1a3a] rounded-xl p-4 shadow-xl">
                        <ResponsiveContainer width="100%" height={200}>
                            <BarChart data={getChartData(result.probabilities)}>
                                <XAxis
                                    dataKey="name"
                                    stroke="#ad92c9"
                                    style={{ fontSize: '12px' }}
                                />
                                <YAxis
                                    domain={[0, 1]}
                                    tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                                    stroke="#ad92c9"
                                    style={{ fontSize: '12px' }}
                                />
                                <Tooltip
                                    formatter={(value) => `${(value * 100).toFixed(1)}%`}
                                    contentStyle={{
                                        backgroundColor: '#362348',
                                        border: 'none',
                                        borderRadius: '8px',
                                        color: '#fff'
                                    }}
                                />
                                <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                                    {getChartData(result.probabilities).map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}
        </div>
    );

    // Combined results display component
    const CombinedResultsDisplay = ({ results }) => (
        <div className="animate-fade-in">
            <h2 className="text-white text-xl sm:text-[22px] font-bold leading-tight tracking-[-0.015em] px-4 pb-3 pt-5">
                Combined Results
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-4">
                {results.text && (
                    <div className="bg-[#2a1a3a] rounded-xl p-6 shadow-xl">
                        <h3 className="text-white text-lg font-bold mb-4 flex items-center gap-2">
                            📝 Text Sentiment
                        </h3>
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <span className="text-white/80">Sentiment:</span>
                                <span className="text-2xl">{sentimentEmojis[results.text.sentiment]}</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-white/80">Type:</span>
                                <span className="text-white font-bold capitalize">{results.text.sentiment}</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-white/80">Confidence:</span>
                                <span className="text-white font-bold">{(results.text.confidence * 100).toFixed(1)}%</span>
                            </div>
                            {results.text.polarity !== undefined && (
                                <div className="flex items-center justify-between">
                                    <span className="text-white/80">Polarity:</span>
                                    <span className="text-white font-semibold">{results.text.polarity.toFixed(3)}</span>
                                </div>
                            )}
                            {results.text.model === 'llm' && (
                                <div className="mt-2">
                                    <span className="inline-flex items-center gap-2 bg-linear-to-r from-purple-600 to-blue-600 px-3 py-1 rounded-full text-xs text-white font-bold">
                                        🤖 DeepSeek V3
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {results.image && (
                    <div className="bg-[#2a1a3a] rounded-xl p-6 shadow-xl">
                        <h3 className="text-white text-lg font-bold mb-4 flex items-center gap-2">
                            🖼️ Image Sentiment
                        </h3>
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <span className="text-white/80">Sentiment:</span>
                                <span className="text-2xl">{sentimentEmojis[results.image.sentiment]}</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-white/80">Type:</span>
                                <span className="text-white font-bold capitalize">{results.image.sentiment}</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-white/80">Confidence:</span>
                                <span className="text-white font-bold">{(results.image.confidence * 100).toFixed(1)}%</span>
                            </div>
                            {results.image.emotion && (
                                <div className="flex items-center justify-between">
                                    <span className="text-white/80">Emotion:</span>
                                    <span className="text-white font-semibold capitalize">{results.image.emotion}</span>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );

    return (
        <div className="relative flex h-auto min-h-screen w-full flex-col bg-[#1a1122]" style={{ fontFamily: '"Space Grotesk", "Noto Sans", sans-serif' }}>
            <div className="layout-container flex h-full grow flex-col">
                <div className="px-4 sm:px-10 lg:px-40 flex flex-1 justify-center py-5">
                    <div className="layout-content-container flex flex-col max-w-[960px] flex-1">
                        <h2 className="text-white tracking-light text-2xl sm:text-[28px] font-bold leading-tight px-4 text-center pb-3 pt-5">
                            Multi-Modal Sentiment Analyzer
                        </h2>

                        {/* LLM Toggle */}
                        <div className="flex justify-center items-center gap-3 px-4 py-3">
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={useLLM}
                                    onChange={(e) => setUseLLM(e.target.checked)}
                                    className="w-5 h-5 rounded bg-[#362348] border-[#8013ec] text-[#8013ec] focus:ring-[#8013ec] focus:ring-offset-0"
                                />
                                <span className="text-white text-sm font-medium">
                                    🤖 Use AI (DeepSeek V3) for enhanced analysis
                                </span>
                            </label>
                        </div>

                        {/* Mode Selector */}
                        <div className="flex justify-center gap-3 px-4 py-6">
                            <button
                                onClick={() => handleModeChange('text')}
                                className={`px-6 py-3 rounded-xl font-bold text-sm transition-all ${currentMode === 'text'
                                        ? 'bg-[#8013ec] text-white shadow-lg shadow-[#8013ec]/50'
                                        : 'bg-[#362348] text-white hover:bg-[#4a2f5c]'
                                    }`}
                            >
                                📝 Text Analysis
                            </button>
                            <button
                                onClick={() => handleModeChange('image')}
                                className={`px-6 py-3 rounded-xl font-bold text-sm transition-all ${currentMode === 'image'
                                        ? 'bg-[#8013ec] text-white shadow-lg shadow-[#8013ec]/50'
                                        : 'bg-[#362348] text-white hover:bg-[#4a2f5c]'
                                    }`}
                            >
                                🖼️ Image Analysis
                            </button>
                            <button
                                onClick={() => handleModeChange('both')}
                                className={`px-6 py-3 rounded-xl font-bold text-sm transition-all ${currentMode === 'both'
                                        ? 'bg-[#8013ec] text-white shadow-lg shadow-[#8013ec]/50'
                                        : 'bg-[#362348] text-white hover:bg-[#4a2f5c]'
                                    }`}
                            >
                                📊 Combined Analysis
                            </button>
                        </div>

                        {/* TEXT MODE */}
                        {currentMode === 'text' && (
                            <div className="space-y-4">
                                <div className="flex max-w-full sm:max-w-[480px] flex-wrap items-end gap-4 px-4 py-3 mx-auto w-full">
                                    <label className="flex flex-col min-w-40 flex-1">
                                        <textarea
                                            value={text}
                                            onChange={(e) => setText(e.target.value)}
                                            placeholder="Enter text to analyze..."
                                            className="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-white focus:outline-0 focus:ring-0 border-none bg-[#362348] focus:border-none min-h-36 placeholder:text-[#ad92c9] p-4 text-base font-normal leading-normal transition-all focus:bg-[#4a2f5c] shadow-lg"
                                        />
                                    </label>
                                </div>

                                <div className="flex justify-center">
                                    <div className="flex flex-1 gap-3 flex-wrap px-4 py-3 max-w-full sm:max-w-[600px] justify-center">
                                        {examples.map((example, index) => (
                                            <button
                                                key={index}
                                                onClick={() => loadExample(example)}
                                                disabled={loading}
                                                className="flex min-w-[84px] cursor-pointer items-center justify-center overflow-hidden rounded-xl h-10 px-4 bg-[#362348] text-white text-sm font-bold leading-normal tracking-[0.015em] hover:bg-[#4a2f5c] transition-all hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
                                            >
                                                <span className="truncate text-xs sm:text-sm">Example {index + 1}</span>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <div className="flex px-4 py-3 justify-center gap-3">
                                    <button
                                        onClick={analyzeText}
                                        disabled={loading || !text.trim()}
                                        className="flex min-w-[84px] max-w-[200px] cursor-pointer items-center justify-center overflow-hidden rounded-xl h-10 px-6 bg-[#8013ec] text-white text-sm font-bold leading-normal tracking-[0.015em] disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#9124ff] transition-all hover:scale-105 hover:shadow-lg hover:shadow-[#8013ec]/50"
                                    >
                                        <span className="truncate">{loading ? 'Analyzing...' : 'Analyze Text'}</span>
                                    </button>
                                    {(text || result) && (
                                        <button
                                            onClick={clearText}
                                            disabled={loading}
                                            className="flex min-w-[84px] max-w-[200px] cursor-pointer items-center justify-center overflow-hidden rounded-xl h-10 px-6 bg-[#362348] text-white text-sm font-bold leading-normal tracking-[0.015em] hover:bg-[#4a2f5c] transition-all hover:scale-105 disabled:opacity-50"
                                        >
                                            <span className="truncate">Clear</span>
                                        </button>
                                    )}
                                </div>
                            </div>
                        )}


                        {/* IMAGE MODE */}
                        {currentMode === 'image' && (
                            <div className="space-y-4">
                                <div className="px-4">
                                    <div
                                        onClick={() => document.getElementById('imageFileInput').click()}
                                        className="border-2 border-dashed border-[#4a2f5c] rounded-xl p-8 text-center cursor-pointer hover:border-[#8013ec] transition-all hover:bg-[#362348]/30"
                                    >
                                        {imagePreview ? (
                                            <img src={imagePreview} alt="Preview" className="max-h-64 mx-auto rounded-lg" />
                                        ) : (
                                            <div>
                                                <p className="text-white text-lg font-semibold mb-2">📁 Upload an Image</p>
                                                <p className="text-white/60 text-sm">Click to select or drag and drop</p>
                                                <p className="text-white/40 text-xs mt-2">Supported: JPG, PNG, GIF, BMP (Max 10MB)</p>
                                            </div>
                                        )}
                                    </div>
                                    <input
                                        id="imageFileInput"
                                        type="file"
                                        accept="image/*"
                                        onChange={handleImageChange}
                                        className="hidden"
                                    />
                                </div>

                                <div className="flex px-4 py-3 justify-center gap-3">
                                    <button
                                        onClick={analyzeImage}
                                        disabled={loading || !imageFile}
                                        className="flex min-w-[84px] max-w-[200px] cursor-pointer items-center justify-center overflow-hidden rounded-xl h-10 px-6 bg-[#8013ec] text-white text-sm font-bold leading-normal tracking-[0.015em] disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#9124ff] transition-all hover:scale-105 hover:shadow-lg hover:shadow-[#8013ec]/50"
                                    >
                                        <span className="truncate">{loading ? 'Analyzing...' : 'Analyze Image'}</span>
                                    </button>
                                    {(imageFile || result) && (
                                        <button
                                            onClick={clearImage}
                                            disabled={loading}
                                            className="flex min-w-[84px] max-w-[200px] cursor-pointer items-center justify-center overflow-hidden rounded-xl h-10 px-6 bg-[#362348] text-white text-sm font-bold leading-normal tracking-[0.015em] hover:bg-[#4a2f5c] transition-all hover:scale-105 disabled:opacity-50"
                                        >
                                            <span className="truncate">Clear</span>
                                        </button>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* BOTH MODE */}
                        {currentMode === 'both' && (
                            <div className="space-y-4">
                                <div className="flex max-w-full sm:max-w-[480px] flex-wrap items-end gap-4 px-4 py-3 mx-auto w-full">
                                    <label className="flex flex-col min-w-40 flex-1">
                                        <textarea
                                            value={combinedText}
                                            onChange={(e) => setCombinedText(e.target.value)}
                                            placeholder="Enter text (optional)..."
                                            className="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-white focus:outline-0 focus:ring-0 border-none bg-[#362348] focus:border-none min-h-36 placeholder:text-[#ad92c9] p-4 text-base font-normal leading-normal transition-all focus:bg-[#4a2f5c] shadow-lg"
                                        />
                                    </label>
                                </div>

                                <div className="flex justify-center">
                                    <div className="flex flex-1 gap-3 flex-wrap px-4 py-3 max-w-full sm:max-w-[600px] justify-center">
                                        {examples.map((example, index) => (
                                            <button
                                                key={index}
                                                onClick={() => loadExample(example)}
                                                disabled={loading}
                                                className="flex min-w-[84px] cursor-pointer items-center justify-center overflow-hidden rounded-xl h-10 px-4 bg-[#362348] text-white text-sm font-bold leading-normal tracking-[0.015em] hover:bg-[#4a2f5c] transition-all hover:scale-105 disabled:opacity-50"
                                            >
                                                <span className="truncate text-xs sm:text-sm">Example {index + 1}</span>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <div className="px-4">
                                    <div
                                        onClick={() => document.getElementById('combinedImageFileInput').click()}
                                        className="border-2 border-dashed border-[#4a2f5c] rounded-xl p-8 text-center cursor-pointer hover:border-[#8013ec] transition-all hover:bg-[#362348]/30"
                                    >
                                        {combinedImagePreview ? (
                                            <img src={combinedImagePreview} alt="Preview" className="max-h-64 mx-auto rounded-lg" />
                                        ) : (
                                            <div>
                                                <p className="text-white text-lg font-semibold mb-2">📁 Upload an Image (Optional)</p>
                                                <p className="text-white/60 text-sm">Click to select or drag and drop</p>
                                            </div>
                                        )}
                                    </div>
                                    <input
                                        id="combinedImageFileInput"
                                        type="file"
                                        accept="image/*"
                                        onChange={handleCombinedImageChange}
                                        className="hidden"
                                    />
                                </div>

                                <div className="flex px-4 py-3 justify-center gap-3">
                                    <button
                                        onClick={analyzeBoth}
                                        disabled={loading || (!combinedText.trim() && !combinedImageFile)}
                                        className="flex min-w-[84px] max-w-[200px] cursor-pointer items-center justify-center overflow-hidden rounded-xl h-10 px-6 bg-[#8013ec] text-white text-sm font-bold leading-normal tracking-[0.015em] disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#9124ff] transition-all hover:scale-105 hover:shadow-lg hover:shadow-[#8013ec]/50"
                                    >
                                        <span className="truncate">{loading ? 'Analyzing...' : 'Analyze Both'}</span>
                                    </button>
                                    {(combinedText || combinedImageFile || combinedResults) && (
                                        <button
                                            onClick={clearBoth}
                                            disabled={loading}
                                            className="flex min-w-[84px] max-w-[200px] cursor-pointer items-center justify-center overflow-hidden rounded-xl h-10 px-6 bg-[#362348] text-white text-sm font-bold leading-normal tracking-[0.015em] hover:bg-[#4a2f5c] transition-all hover:scale-105 disabled:opacity-50"
                                        >
                                            <span className="truncate">Clear All</span>
                                        </button>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* Loading Spinner */}
                        {loading && (
                            <div className="flex justify-center items-center py-8">
                                <div className="relative">
                                    <div className="w-16 h-16 border-4 border-[#362348] border-t-[#8013ec] rounded-full animate-spin"></div>
                                    <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-white text-sm font-semibold">
                                        AI
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Error Message */}
                        {error && (
                            <div className="px-4 py-3 animate-fade-in">
                                <div className="bg-red-900/30 border border-red-500 text-red-200 px-4 py-3 rounded-xl text-center backdrop-blur-sm">
                                    ⚠️ {error}
                                </div>
                            </div>
                        )}

                        {/* Single Result Display (for text and image modes) */}
                        {result && !loading && currentMode !== 'both' && <SingleResult result={result} />}

                        {/* Combined Results Display */}
                        {combinedResults && !loading && currentMode === 'both' && <CombinedResultsDisplay results={combinedResults} />}
                    </div>
                </div>
            </div>

            <style>{`
                @keyframes fade-in {
                    from {
                        opacity: 0;
                        transform: translateY(10px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                .animate-fade-in {
                    animation: fade-in 0.5s ease-out;
                }
            `}</style>
        </div>
    );
};

export default SentimentAnalysis;