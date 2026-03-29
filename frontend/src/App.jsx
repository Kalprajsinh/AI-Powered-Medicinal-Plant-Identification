import { useState, useRef } from "react";
import axios from "axios";
import { BarChart } from 'klearcharts';

const sampleModules = import.meta.glob("../Sample_data/**/*.{jpg,jpeg,png,webp}", { eager: true });
const sampleImages = Object.entries(sampleModules).map(([path, mod]) => {
  const parts = path.split('/');
  const folderName = parts[parts.length - 2];
  const fileName = parts[parts.length - 1];
  // Extract English name from folder (before parenthesis)
  const englishName = folderName.split(' (')[0];
  return {
    name: englishName,
    url: mod.default,
    fileName,
    folder: folderName,
  };
});

export default function App() {
  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => setImage(e.target.result);
    reader.readAsDataURL(file);
    setImageFile(file);
    setResult(null);
    setError(null);
  };

  const handleSampleSelect = async (sample) => {
    try {
      const response = await fetch(sample.url);
      const blob = await response.blob();
      const selectedFile = new File([blob], sample.fileName, { type: blob.type || 'image/jpeg' });
      setImage(sample.url);
      setImageFile(selectedFile);
      setResult(null);
      setError(null);
    } catch (err) {
      setError('Unable to load sample image.');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) handleFile(file);
  };

  const handlePredict = async () => {
    if (!imageFile) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", imageFile);
      const res = await axios.post("https://ai-powered-medicinal-plant-identification.onrender.com/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      if (res.data.success) setResult(res.data.data);
      else setError("Prediction failed.");
    } catch (err) {
      setError(err.response?.data?.message || "Server error. Is the API running?");
    } finally {
      setLoading(false);
    }
  };

  const clearImage = () => {
    setImage(null);
    setImageFile(null);
    setResult(null);
    setError(null);
  };

  const topPrediction = result?.predictions?.[0];
  const confidencePct = (v) => Math.round(v * 100);

  return (
    <div className="min-h-screen bg-gray-50 font-sans">
      {/* Header */}
      <div className="max-w-6xl mx-auto px-6 pt-10 pb-6">
        <h1 className="text-4xl font-extrabold text-gray-900 tracking-tight">Plant Identifier</h1>
        <p className="mt-2 text-gray-500 text-sm max-w-2xl">
          Upload an image of a plant and watch as our AI model analyzes it and shows you exactly what's happening inside the neural network.
        </p>
      </div>

      {/* Main grid */}
      <div className="max-w-6xl mx-auto px-6 pb-16 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left panel */}
        <div className="lg:col-span-1 flex flex-col gap-4">
          {/* Upload card */}
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
            <h2 className="font-semibold text-gray-800 mb-3">Upload Plant Image</h2>

            {!image ? (
              <div
                className={`border-2 border-dashed rounded-xl flex flex-col items-center justify-center gap-2 py-10 cursor-pointer transition-colors ${
                  dragging ? "border-teal-400 bg-teal-50" : "border-gray-200 hover:border-teal-300 hover:bg-gray-50"
                }`}
                onClick={() => fileInputRef.current.click()}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
              >
                <div className="w-12 h-12 bg-teal-50 rounded-xl flex items-center justify-center">
                  <svg className="w-6 h-6 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                </div>
                <p className="font-semibold text-gray-700 text-sm">Drop image here or click</p>
                <p className="text-xs text-gray-400">Upload a plant image (PNG, JPG, WebP)</p>
              </div>
            ) : (
              <div className="relative">
                <img src={image} alt="plant" className="w-full rounded-xl object-cover max-h-64" />
                <button
                  onClick={clearImage}
                  className="absolute top-2 right-2 bg-white rounded-full w-7 h-7 flex items-center justify-center shadow text-gray-500 hover:text-red-500 transition"
                >
                  ✕
                </button>
                <p className="text-xs text-gray-400 mt-2">Image ready for prediction</p>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFile(e.target.files[0])}
            />

            {image && ( 
              <>
                {loading && (
                  <div className="mb-3 rounded-xl border border-orange-300 bg-orange-50 p-3 text-sm text-orange-700">
                    Backend hosted on free instance which can delay requests by 50 seconds or more, please wait while processing image.
                  </div>
                )}
                <button
                  onClick={handlePredict}
                  disabled={loading}
                  className="mt-4 w-full py-2.5 rounded-xl bg-teal-700 hover:bg-teal-800 text-white font-semibold text-sm transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                      </svg>
                      Analyzing...
                    </>
                  ) : "Identify Plant"}
                </button>
              </>
            )}
          </div>

          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
            <h2 className="font-semibold text-gray-800 mb-3">Or try with sample plants</h2>
            <p className="text-xs text-gray-700">Click any plant to see predictions and visualizations</p>
            <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 gap-3">
              {sampleImages.slice(0, 12).map((sample, idx) => (
                <button
                  key={`${sample.name}-${idx}`}
                  onClick={() => handleSampleSelect(sample)}
                  className="border border-gray-200 rounded-xl overflow-hidden text-left hover:shadow-md transition hover:border-teal-300"
                  type="button"
                >
                  <img src={sample.url} alt={sample.name} className="w-full h-20 object-cover" />
                  <div className="p-2 text-xs font-medium text-gray-700 truncate">{sample.name}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Top prediction card */}
          {result && (
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
              <p className="text-xs font-bold tracking-widest text-gray-400 uppercase mb-2">Top Prediction</p>
              <h3 className="text-xl font-bold text-gray-900">{topPrediction.species}</h3>
              <p className="text-sm text-gray-500 mt-1">
                Confidence: <span className="text-teal-600 font-semibold">{confidencePct(topPrediction.confidence)}%</span>
              </p>
              <div className="mt-4 pt-4 border-t border-gray-100 grid grid-cols-2 gap-2 text-xs text-gray-500">
                <div>
                  <span className="block text-gray-400">Model</span>
                  <span className="font-medium text-gray-700">{result.processingStats.modelName}</span>
                </div>
                <div>
                  <span className="block text-gray-400">Inference Time</span>
                  <span className="font-medium text-gray-700">{result.processingStats.inferenceTime}</span>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-2xl p-4 text-sm text-red-600">
              {error}
            </div>
          )}
        </div>

        {/* Right panel */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          {!result ? (
            <div className="bg-gray-100 rounded-2xl flex flex-col items-center justify-center py-24 gap-3">
              <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center">
                <svg className="w-8 h-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6.75a5.25 5.25 0 110 10.5 5.25 5.25 0 010-10.5zM21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="font-semibold text-gray-500">No Image Uploaded Yet</p>
              <p className="text-sm text-gray-400 text-center max-w-xs">
                Upload a plant image on the left to see the AI model's analysis and internal visualizations
              </p>
            </div>
          ) : (
            <>
              {/* Model Predictions */}
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
                <h2 className="font-semibold text-gray-800 mb-1">Model Predictions</h2>
                <p className="text-xs text-gray-400 mb-5">Top predictions with confidence scores</p>

                {/* Bar chart */}
                <BarChart
                  data={result.predictions.map(p => Math.trunc(p.confidence * 10000) / 100)}
                  height={300}
                  width={600}
                  barColor="#10b981"
                  animate={true}
                />

                {/* Progress bars */}
                <div className="flex flex-col gap-3">
                  {result.predictions.map((p, i) => {
                    const pct = confidencePct(p.confidence);
                    return (
                      <div key={i} className="flex items-center gap-3">
                        <span className="text-sm text-gray-700 w-60 shrink-0">{p.species}</span>
                        <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-teal-700 rounded-full transition-all duration-700"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="text-sm font-semibold text-teal-700 w-10 text-right">{pct}%</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Feature Maps */}
              {/* <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
                <h2 className="font-semibold text-gray-800 mb-1">Feature Maps Visualization</h2>
                <p className="text-xs text-gray-400 mb-4">CNN layer activations showing what the model learns</p>
                <div className="grid grid-cols-4 gap-3">
                  {result.featureMaps.slice(0, 8).map((map, i) => (
                    <div key={i} className="rounded-xl overflow-hidden border border-gray-100 aspect-square bg-gray-100">
                      <img
                        src={`data:image/png;base64,${map}`}
                        alt={`Feature map ${i + 1}`}
                        className="w-full h-full object-cover"
                      />
                    </div>
                  ))}
                </div>
                <p className="text-xs text-gray-400 mt-3">
                  Showing {Math.min(8, result.featureMaps.length)} different feature maps extracted from the convolutional layers of the neural network
                </p>
              </div> */}

              {/* Processing Details */}
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
                <h2 className="font-semibold text-gray-800 mb-4">Processing Details</h2>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: "Model Architecture", value: result.processingStats.modelName },
                    { label: "Network Depth", value: `${result.processingStats.layers} Layers` },
                    { label: "Parameters", value: result.processingStats.parameters },
                    { label: "Inference Speed", value: result.processingStats.inferenceTime },
                  ].map(({ label, value }) => (
                    <div key={label} className="bg-gray-50 rounded-xl p-4">
                      <p className="text-xs text-gray-400 mb-1">{label}</p>
                      <p className="font-semibold text-gray-800">{value}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Attention Heatmap */}
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
                <h2 className="font-semibold text-gray-800 mb-1">Attention Heatmap</h2>
                <p className="text-md text-gray-800 mb-4">Shows which regions the model focuses on for its prediction</p>
                <div className="rounded-xl overflow-hidden border border-gray-100 flex flex-col items-center">
                  <img
                    src={`data:image/png;base64,${result.attentionHeatmap}`}
                    alt="Attention Heatmap"
                    className="w-9/12 object-contain rounded-2xl"
                  />
                </div>
                <p className="text-md text-gray-700 mt-3">
                  Red areas indicate high attention (important for prediction), while blue areas indicate low attention. The model focuses on plant morphological features.
                </p>
              </div>

              

              {/* How it works */}
              {/* <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
                <h2 className="font-semibold text-gray-800 mb-4 flex items-center gap-2">
                 How the Visualizations Work
                </h2>
                <ol className="flex flex-col gap-3 text-sm text-gray-600">
                  <li className="flex gap-2">
                    <span className="font-bold text-teal-700 shrink-0">1.</span>
                    <span><span className="font-semibold text-gray-800">Feature Maps:</span> Show intermediate layer outputs as the image passes through the neural network</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-bold text-teal-700 shrink-0">2.</span>
                    <span><span className="font-semibold text-gray-800">Attention Heatmap:</span> Indicates which regions of the image the model focuses on (red = high attention)</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-bold text-teal-700 shrink-0">3.</span>
                    <span><span className="font-semibold text-gray-800">Predictions:</span> Top 3 predictions with confidence scores normalized to probabilities</span>
                  </li>
                </ol>
              </div> */}
            </>
          )}
        </div>
      </div>
    </div>
  );
}