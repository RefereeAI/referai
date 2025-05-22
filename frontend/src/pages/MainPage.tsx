import React, { useState, useRef, useEffect } from "react";
import { PredictResponse, SinglePrediction } from "../api/predict";
import Navbar from "../components/Navbar";
import Toast from "../components/Toast";

const API_URL = import.meta.env.VITE_API_URL;

export default function MainPage() {
  const [selectedVideos, setSelectedVideos] = useState<string[]>([]);
  const [videoDuration, setVideoDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const animationRef = useRef<number | null>(null);
  const videoRefs = useRef<HTMLVideoElement[]>([]);
  const [predictions, setPredictions] = useState<SinglePrediction[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [toast, setToast] = useState<{ message: string; type: "error" | "success" } | null>(null);

  const handlePrediction = async () => {
    const actionId = localStorage.getItem("last_action_id");
    const token = localStorage.getItem("token");

    setLoading(true);

    if (!actionId || !token) {
      setToast({ message: "User not authenticated or action ID not found", type: "error" });
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(`${API_URL}/predict/${actionId}`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to run prediction");
      }

      const data: PredictResponse = await response.json();
      setPredictions(data.results);
      setToast({ message: "Prediction successfully run!", type: "success" });
    } catch (err: any) {
      console.error("Prediction error", err);
      setToast({ message: `Error running prediction: ${err.message}`, type: "error" });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    return () => stopAnimation();
  }, []);


  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      if (videoRefs.current[0]) {
        setCurrentTime(videoRefs.current[0].currentTime);
      }
    }, 500);

    return () => clearInterval(interval);
  }, [isPlaying]);

  useEffect(() => {
    const fetchClips = async () => {
      const stored = localStorage.getItem("last_action_id");
      if (!stored) return;

      setIsLoading(true);

      const token = localStorage.getItem("token");
      if (!token) {
        setToast({ message: "User not authenticated", type: "error" });
        setIsLoading(false);
        return;
      }

      try {
        const res = await fetch(`${API_URL}/action/${stored}`, {
          method: "GET",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (!res.ok) {
          const errorData = await res.json();
          throw new Error(errorData.detail || "Failed to fetch clips");
        }

        const data = await res.json();
        const videoURLs = data.clips.map((clip: any) => {
          const binary = Uint8Array.from(atob(clip.content), (c) => c.charCodeAt(0));
          return URL.createObjectURL(new Blob([binary], { type: "video/mp4" }));
        });

        setSelectedVideos(videoURLs);

        // 2. Fetch existing predictions
        const predRes = await fetch(`${API_URL}/predict/${stored}`, {
          method: "GET",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (predRes.ok) {
          const predData: PredictResponse = await predRes.json();
          if (predData.results && predData.results.length > 0) {
            setPredictions(predData.results);
          }
        } else if (predRes.status !== 404) {
          const err = await predRes.json();
          console.warn("Prediction fetch error:", err.detail);
          setToast({ message: `Prediction fetch error: ${err.detail}`, type: "error" });
        }

      } catch (error: any) {
        console.error("Error fetching clips:", error);
        setToast({ message: `Error fetching clips: ${error.message}`, type: "error" });
      } finally {
        setIsLoading(false);
      }
    };

    fetchClips();
  }, []);

  const startAnimation = () => {
    const update = () => {
      if (videoRefs.current[0]) {
        const time = videoRefs.current[0].currentTime;
        setCurrentTime(time);
        animationRef.current = requestAnimationFrame(update);
      }
    };
    animationRef.current = requestAnimationFrame(update);
  };

  const stopAnimation = () => {
    if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  };


  const togglePlayPause = () => {
    const newPlaying = !isPlaying;
    setIsPlaying(newPlaying);

    videoRefs.current.forEach((video) => {
      newPlaying ? video.play() : video.pause();
    });

    if (newPlaying) {
      startAnimation();
    } else {
      stopAnimation();
    }
  };


  const handleTimeUpdate = (time: number) => {
    videoRefs.current.forEach((video) => {
      video.currentTime = time;
    });
    setCurrentTime(time);
  };


  useEffect(() => {
    const handleSync = () => {
      if (videoRefs.current[0]) {
        setCurrentTime(videoRefs.current[0].currentTime);
      }
    };

    videoRefs.current.forEach((video) => {
      video.addEventListener("timeupdate", handleSync);
    });

    return () => {
      videoRefs.current.forEach((video) => {
        video.removeEventListener("timeupdate", handleSync);
      });
    };
  }, []);

  useEffect(() => {
    const handleEnded = () => {
      const allEnded = videoRefs.current.every(
        (video) => video.ended || video.currentTime === video.duration
      );
      if (allEnded) {
        setIsPlaying(false);
        stopAnimation();
      }
    };

    videoRefs.current.forEach((video) => {
      video.addEventListener("ended", handleEnded);
    });

    return () => {
      videoRefs.current.forEach((video) => {
        video.removeEventListener("ended", handleEnded);
      });
    };
  }, [selectedVideos]);


  return (
    <>
      <Navbar />
      <div className="flex flex-col md:flex-row h-screen">
        {/* Video Grid Section */}
        <div
          className={`flex-[3] grid gap-2 p-2 bg-slate-800 ${selectedVideos.length === 2 ? "grid-cols-1 grid-rows-2" : "grid-cols-2 grid-rows-2"
            }`}
        >
          {isLoading ? (
            <div className="col-span-2 row-span-2 flex items-center justify-center">
              <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            </div>
          ) : (
            selectedVideos.map((video, index) => (
              <div
                key={index}
                className={`rounded-md overflow-hidden ${selectedVideos.length === 3 && index === 2 ? "col-span-2" : ""}`}
              >
                <video
                  muted
                  src={video}
                  controls={false}
                  onLoadedMetadata={(e) => {
                    if (!videoDuration && !isNaN(e.currentTarget.duration)) {
                      setVideoDuration(e.currentTarget.duration);
                    }
                  }}
                  ref={(el) => {
                    if (el) videoRefs.current[index] = el;
                  }}
                  className="w-full h-full object-cover"
                />

              </div>
            ))

          )}
        </div>

        {/* Side Panel */}
        <div className="flex-[1] p-4 h-full bg-slate-900 overflow-auto">
          {/* Controls */}
          <div className="grid grid-cols-3 mb-4 gap-2">
            {/* Range Input */}
            <div className="col-span-2 flex items-center">
              <input
                type="range"
                min={0}
                max={videoDuration || 0}
                step={0.01}
                value={currentTime}
                onInput={(e) => handleTimeUpdate(Number(e.currentTarget.value))}
                className="w-full accent-indigo-600"
              />




            </div>

            {/* Play/Pause Button */}
            <div className="col-span-1 flex justify-center items-center">
              <button
                onClick={togglePlayPause}
                aria-pressed={isPlaying}
                className="text-white bg-indigo-500 hover:bg-indigo-600 focus:outline-none focus:ring-4 focus:ring-blue-300 font-medium rounded-full text-sm px-5 py-2.5"
              >
                {isPlaying ? "Pause" : "Play"}
              </button>
            </div>
          </div>

          {/* Prediction Result */}
          <div className="mt-6">
            {loading ? (
              <div className="mx-auto w-full max-w-md rounded-xl border border-blue-300 p-6 bg-slate-800 shadow animate-pulse">
                <div className="space-y-4">
                  <div className="h-4 bg-gray-600 rounded w-3/4"></div>
                  <div className="h-4 bg-gray-600 rounded w-5/6"></div>
                  <div className="h-4 bg-gray-600 rounded w-2/3"></div>
                  <div className="h-4 bg-gray-600 rounded w-4/5"></div>
                  <div className="h-4 bg-gray-600 rounded w-3/4"></div>
                </div>
              </div>
            ) : (
              predictions &&
              predictions.map((prediction) => (
                <div
                  key={prediction.filename}
                  className="p-6 rounded-2xl shadow-md border border-gray-700 bg-slate-800 max-w-3xl mx-auto"
                >
                  <div className="grid gap-6 md:grid-cols-2">
                    <div>
                      <h2 className="text-lg font-semibold text-white mb-1">Foul Prediction</h2>
                      <p className="text-gray-300">
                        <strong>Result:</strong> {prediction.is_foul ? "Yes" : "No"} (
                        {prediction.foul_confidence.toFixed(1)}% vs{" "}
                        {prediction.no_foul_confidence.toFixed(1)}%)
                      </p>
                    </div>
                    <div>
                      <h2 className="text-lg font-semibold text-white mb-1">Severity</h2>
                      <p className="text-gray-300">
                        No card {prediction.severity.no_card.toFixed(1)}%, Red card{" "}
                        {prediction.severity.red_card.toFixed(1)}%, Yellow card{" "}
                        {prediction.severity.yellow_card.toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  <div className="mt-6 border-t border-gray-700 pt-4">
                    <h3 className="text-md font-semibold text-white mb-2">Foul Model Results</h3>
                    <ul className="list-disc ml-6 text-sm text-gray-300 space-y-1">
                      {prediction.foul_model_results.map((modelResult, index) => (
                        <li key={index}>
                          {modelResult.model}: {modelResult.prediction === 1 ? "Foul" : "No Foul"}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="mt-6 border-t border-gray-700 pt-4">
                    <h3 className="text-md font-semibold text-white mb-2">Severity Model Results</h3>
                    <ul className="list-disc ml-6 text-sm text-gray-300 space-y-1">
                      {prediction.severity_model_results.map((modelResult, index) => (
                        <li key={index}>
                          {modelResult.model}:{" "}
                          {modelResult.prediction === 0
                            ? "No Card"
                            : modelResult.prediction === 1
                              ? "Red Card"
                              : "Yellow Card"}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Run Prediction Button */}
          <div className="flex justify-center items-center">
            <button
              onClick={handlePrediction}
              disabled={loading || selectedVideos.length === 0}
              className="mt-4 px-4 py-2 text-white rounded bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <svg
                    className="animate-spin h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                    ></path>
                  </svg>
                  Predicting...
                </>
              ) : (
                "Run Prediction"
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Show errors */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </>
  );
}
